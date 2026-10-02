"""Evaluate one exported case and save PNG, PDF, GIF, and RGB-image metrics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch_msssim import ssim

from ..checkpoints import load_checkpoint
from ..data.dataset import BatteryDataset, FIELD_DIRECTORIES
from ..models import PaperModel
from ..train.engine import choose_device


def require_fresh_output(path):
    """Prevent stale frames or GIFs from being mixed into a later evaluation."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError("Evaluation output must be a new or empty directory; select a new --output-dir.")


def rollout(model, inputs, steps):
    """Feed each predicted field back with the fixed orientation and C-rate maps."""
    if steps < 1:
        raise ValueError("Prediction length must be positive.")
    static, current, output = inputs[:, -1, 3:9], inputs, []
    for _ in range(steps):
        prediction = model(current)
        output.append(prediction)
        next_frame = torch.cat([prediction, static], dim=1).unsqueeze(1)
        current = torch.cat([current[:, 1:], next_frame], dim=1)
    return torch.stack(output, dim=1)


def to_image(tensor):
    return tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def save_comparison(predictions, targets, labels, output_dir, field, make_gif=True):
    count = len(predictions)
    selected = np.unique(np.linspace(0, count - 1, min(4, count), dtype=int))
    figure, axes = plt.subplots(3, len(selected), figsize=(4 * len(selected), 9), squeeze=False)
    max_error = max(float(np.abs(p - t).mean(axis=-1).max()) for p, t in zip(predictions, targets))
    max_error = max(max_error, 1e-6)
    for column, index in enumerate(selected):
        error = np.abs(predictions[index] - targets[index]).mean(axis=-1)
        axes[0, column].imshow(targets[index])
        axes[1, column].imshow(predictions[index])
        error_plot = axes[2, column].imshow(error, cmap="magma", vmin=0, vmax=max_error)
        axes[0, column].set_title(labels[index], fontsize=9)
        for row in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    for row, label in enumerate(["Simulation export", "ConvLSTM prediction", "Mean absolute RGB error"]):
        axes[row, 0].set_ylabel(label)
    figure.colorbar(error_plot, ax=axes[2, :].tolist(), fraction=0.035, pad=0.02, label="Normalized RGB error")
    figure.suptitle(f"{field.capitalize()}: autoregressive image prediction", fontsize=14)
    figure.savefig(output_dir / "comparison.png", dpi=160, bbox_inches="tight")
    figure.savefig(output_dir / "comparison.pdf", bbox_inches="tight")
    plt.close(figure)
    frames_dir = output_dir / "predicted_frames"
    frames_dir.mkdir(exist_ok=True)
    animation = []
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        pred_image = Image.fromarray(np.rint(prediction * 255).astype(np.uint8))
        pred_image.save(frames_dir / f"prediction_{index + 1:03d}.png")
        if make_gif:
            truth_image = Image.fromarray(np.rint(target * 255).astype(np.uint8))
            width, height = pred_image.size
            panel = Image.new("RGB", (width * 2, height + 44), "white")
            panel.paste(truth_image, (0, 44))
            panel.paste(pred_image, (width, 44))
            draw = ImageDraw.Draw(panel)
            draw.text((8, 5), f"Simulation export | {labels[index]}", fill="black")
            draw.text((width + 8, 5), "Autoregressive prediction", fill="black")
            draw.text((8, 23), f"{field} (RGB image)", fill="black")
            animation.append(panel)
    if animation:
        animation[0].save(output_dir / "rollout.gif", save_all=True, append_images=animation[1:],
                          duration=300, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--field", choices=list(FIELD_DIRECTORIES), default="concentration")
    parser.add_argument("--input-length", type=int, help="Default: saved setting, otherwise 5.")
    parser.add_argument("--image-size", type=int, help="Default: saved setting, otherwise 512.")
    parser.add_argument("--predict-length", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based first input frame.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    require_fresh_output(args.output_dir)
    device = choose_device(args.device)
    model = PaperModel().to(device)
    metadata = load_checkpoint(args.checkpoint, model, device, args.field)
    config = metadata.get("config", {})
    input_length = args.input_length if args.input_length is not None else config.get("input_length", 5)
    image_size = args.image_size if args.image_size is not None else config.get("image_size", 512)
    if min(image_size, input_length, args.predict_length) < 1 or image_size < 11:
        raise ValueError("Lengths must be positive and image-size at least 11 for SSIM.")
    dataset = BatteryDataset([args.case_dir], args.field, input_length, args.predict_length,
                             image_size, image_size, use_patches=False, augment=False)
    if not 0 <= args.start_index < len(dataset):
        raise ValueError(f"start-index must be between 0 and {len(dataset) - 1} for this case.")
    inputs, targets = dataset[args.start_index]
    model.eval()
    with torch.no_grad():
        predictions = rollout(model, inputs.unsqueeze(0).to(device), args.predict_length)[0]
        targets = targets.to(device)
        rows = [{"prediction_step": index + 1,
                 "mse_rgb": torch.mean((p - t) ** 2).item(),
                 "ssim_rgb": ssim(p.unsqueeze(0), t.unsqueeze(0), data_range=1.0, size_average=True).item()}
                for index, (p, t) in enumerate(zip(predictions, targets))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    begin = args.start_index + input_length
    labels = [path.stem for path in dataset.cases[0].frames[begin:begin + args.predict_length]]
    save_comparison([to_image(p) for p in predictions], [to_image(t) for t in targets],
                    labels, args.output_dir, args.field, make_gif=not args.no_gif)
    report = dict(field=args.field, case=args.case_dir.name, checkpoint=args.checkpoint.name,
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  architecture="Conv3d + three ConvLSTM layers + RGB decoder",
                  parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                  input_length=input_length, start_index=args.start_index, image_size=image_size,
                  predict_length=args.predict_length, metrics_domain="RGB images normalized to [0,1]",
                  checkpoint_contains_training_metadata=bool(metadata),
                  checkpoint_from_batch_limited_smoke_run=bool(config.get("max_batches")),
                  mean_mse_rgb=float(np.mean([r["mse_rgb"] for r in rows])),
                  mean_ssim_rgb=float(np.mean([r["ssim_rgb"] for r in rows])))
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved comparisons and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
