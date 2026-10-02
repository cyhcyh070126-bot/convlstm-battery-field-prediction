"""Shared training loop with case-level splits and reproducible run metadata."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..checkpoints import load_checkpoint
from ..data.dataset import BatteryDataset, FIELD_DIRECTORIES, discover_cases
from ..models import PaperModel
from .losses import HybridLoss


def build_parser(sequence=False):
    parser = argparse.ArgumentParser(description=(
        "Autoregressive MSE+SSIM fine-tuning" if sequence else "Single-frame MSE+SSIM training"))
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Export root containing one directory per simulation case.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--field", choices=list(FIELD_DIRECTORIES), default="concentration")
    parser.add_argument("--checkpoint", type=Path, required=sequence,
                        help="Initialize from matching-field baseline weights; required for sequence fine-tuning.")
    parser.add_argument("--split-file", type=Path,
                        help="Reuse a saved case-level split.json (recommended for legacy checkpoints).")
    parser.add_argument("--train-fraction", type=float, default=0.9 if sequence else 0.8,
                        help="Used only when no previous split is available.")
    parser.add_argument("--epochs", type=int, default=50 if sequence else 100)
    parser.add_argument("--batch-size", type=int, default=16 if sequence else 64)
    parser.add_argument("--val-batch-size", type=int, default=4 if sequence else 64)
    parser.add_argument("--learning-rate", type=float, default=1e-5 if sequence else 1e-3)
    parser.add_argument("--val-interval", type=int, default=1 if sequence else 5)
    parser.add_argument("--input-length", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--ssim-weight", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Start at 0 on Windows; increase after checking RAM use.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--max-batches", type=int,
                        help="Limit each train/validation epoch for smoke checks, not reported research results.")
    if sequence:
        parser.add_argument("--predict-length", type=int, default=10)
        parser.add_argument("--epsilon-start", type=float, default=1.0)
        parser.add_argument("--epsilon-decay", type=float, default=1e-5,
                            help="Teacher-forcing probability decay per optimizer step.")
    else:
        parser.set_defaults(predict_length=1, epsilon_start=0.0, epsilon_decay=0.0)
    parser.set_defaults(sequence=sequence)
    return parser


def choose_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")
    return device


def make_split(cases, train_fraction=0.8, seed=42, existing=None):
    names = [p.name for p in cases]
    if len(names) < 2:
        raise ValueError("Training and validation require at least two independent simulation cases.")
    if existing is None:
        random.Random(seed).shuffle(names)
        count = max(1, min(len(names) - 1, int(len(names) * train_fraction)))
        existing = {"train": names[:count], "validation": names[count:]}
    train_names = existing.get("train", [])
    val_names = existing.get("validation", [])
    combined = train_names + val_names
    if not train_names or not val_names or len(combined) != len(set(combined)):
        raise ValueError("Split must contain nonempty, disjoint train and validation case lists.")
    if set(combined) != set(names):
        raise ValueError("Split cases do not exactly match --data-dir; use the original case collection.")
    lookup = {p.name: p for p in cases}
    return ([lookup[name] for name in train_names], [lookup[name] for name in val_names], existing)


def sequence_loss(model, inputs, targets, criterion, epsilon=0.0):
    """Unroll the window without detaching predictions (original training rule)."""
    current = inputs
    static = inputs[:, -1, 3:9]
    loss = inputs.new_zeros(())
    for t in range(targets.shape[1]):
        prediction = model(current)
        loss = loss + criterion(prediction, targets[:, t])
        if t + 1 < targets.shape[1]:
            next_field = targets[:, t] if random.random() < epsilon else prediction
            next_frame = torch.cat([next_field, static], dim=1).unsqueeze(1)
            current = torch.cat([current[:, 1:], next_frame], dim=1)
    return loss / targets.shape[1]


def require_fresh_training_output(path):
    """Do not overwrite checkpoints even if an earlier run has no history file."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError("Training output must be a new or empty directory; select a new --output-dir.")


def run(args):
    positive = ("epochs", "batch_size", "val_batch_size", "val_interval", "input_length",
                "predict_length", "image_size", "patch_size")
    if any(getattr(args, key) < 1 for key in positive):
        raise ValueError("Epoch counts, batch sizes, intervals, and dimensions must be positive.")
    if args.num_workers < 0 or (args.max_batches is not None and args.max_batches < 1):
        raise ValueError("num-workers must be nonnegative and max-batches must be positive.")
    if not 0 < args.train_fraction < 1 or not 0 <= args.epsilon_start <= 1:
        raise ValueError("train-fraction must be in (0, 1); epsilon-start must be in [0, 1].")
    if not all(math.isfinite(value) for value in
               (args.learning_rate, args.epsilon_decay, args.ssim_weight)):
        raise ValueError("learning-rate, epsilon-decay, and ssim-weight must be finite.")
    if args.learning_rate <= 0 or args.epsilon_decay < 0:
        raise ValueError("learning-rate must be positive and epsilon-decay nonnegative.")
    if args.ssim_weight < 0:
        raise ValueError("ssim-weight must be nonnegative.")
    if min(args.patch_size, args.image_size) < 11:
        raise ValueError("Image and patch sizes must be at least 11 for SSIM.")
    require_fresh_training_output(args.output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    device = choose_device(args.device)
    model = PaperModel().to(device)
    metadata = load_checkpoint(args.checkpoint, model, device, args.field) if args.checkpoint else {}
    existing_split = metadata.get("split")
    if args.split_file:
        requested = json.loads(args.split_file.read_text(encoding="utf-8"))
        if existing_split and requested != existing_split:
            raise ValueError("Requested split differs from the checkpoint split; this could contaminate validation.")
        existing_split = requested
    if args.sequence and existing_split is None:
        print("Legacy weights contain no split metadata. The new split cannot establish held-out validity "
              "unless it matches pretraining; supply --split-file from pretraining when available.")
    cases = discover_cases(args.data_dir, args.field)
    train_cases, val_cases, split = make_split(cases, args.train_fraction, args.seed, existing_split)
    common = dict(field=args.field, input_length=args.input_length, predict_length=args.predict_length,
                  image_size=args.image_size, patch_size=args.patch_size)
    train_data = BatteryDataset(train_cases, use_patches=True, augment=True, **common)
    val_data = BatteryDataset(val_cases, use_patches=not args.sequence, augment=False, **common)
    loader_options = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if args.num_workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=False, **loader_options)
    criterion = HybridLoss(args.ssim_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items()
              if key not in {"data_dir", "output_dir", "checkpoint", "split_file"}}
    config.update(loss="MSE + alpha * (1 - SSIM)", torch_version=str(torch.__version__))
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(f"Device: {device}; train cases: {len(train_cases)}; validation cases: {len(val_cases)}; "
          f"windows/patches: {len(train_data)} / {len(val_data)}")
    if args.max_batches:
        print("SMOKE MODE: batch-limited losses are not full-dataset research results.")
    best_val, global_step = float("inf"), 0
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss", "epsilon"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_total, train_count, epsilon = 0.0, 0, 0.0
            for batch_index, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                epsilon = max(0.0, args.epsilon_start - global_step * args.epsilon_decay)
                optimizer.zero_grad(set_to_none=True)
                loss = sequence_loss(model, inputs, targets, criterion, epsilon)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training loss became nonfinite.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_total += loss.item() * inputs.shape[0]
                train_count += inputs.shape[0]
                global_step += 1
                if args.max_batches and batch_index + 1 >= args.max_batches:
                    break
            train_loss = train_total / train_count
            val_loss = None
            if epoch % args.val_interval == 0 or epoch == args.epochs:
                model.eval()
                val_total, val_count = 0.0, 0
                with torch.no_grad():
                    for batch_index, (inputs, targets) in enumerate(val_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        loss = sequence_loss(model, inputs, targets, criterion, epsilon=0.0)
                        val_total += loss.item() * inputs.shape[0]
                        val_count += inputs.shape[0]
                        if args.max_batches and batch_index + 1 >= args.max_batches:
                            break
                val_loss = val_total / val_count
                if not np.isfinite(val_loss):
                    raise FloatingPointError("Validation loss became nonfinite.")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model_state_dict": model.state_dict(), "config": config,
                                "split": split, "epoch": epoch, "validation_loss": val_loss},
                               args.output_dir / "best_model.pt")
            writer.writerow(dict(epoch=epoch, train_loss=train_loss,
                                 validation_loss=val_loss, epsilon=epsilon))
            handle.flush()
            print(f"Epoch {epoch}/{args.epochs}: train={train_loss:.6f}; "
                  f"validation={val_loss if val_loss is not None else 'skipped'}; epsilon={epsilon:.5f}")
    print(f"Saved best model and run metadata to {args.output_dir}")
