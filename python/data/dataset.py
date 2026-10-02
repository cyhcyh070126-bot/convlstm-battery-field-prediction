"""Sliding image windows with the original RGB normalization and patching."""

import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

FIELD_DIRECTORIES = {"concentration": "1_Concentration", "stress": "2_Stress"}


def natural_key(path):
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r"(\d+)", Path(path).name))


def load_rgb(path, image_size=512):
    """Read, resize with OpenCV's original bilinear default, scale to [0, 1]."""
    # imdecode also handles Unicode paths on Windows.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.float32) / 255.0


@dataclass
class SimulationCase:
    folder: Path
    frames: list
    orientation: Path
    c_rate: Path


def read_case(folder, field="concentration", minimum_frames=1):
    folder = Path(folder)
    if field not in FIELD_DIRECTORIES:
        raise ValueError(f"Unknown field: {field}")
    frames = sorted((folder / FIELD_DIRECTORIES[field]).glob("*.png"), key=natural_key)
    if len(frames) < minimum_frames:
        raise ValueError(f"{folder.name}: need {minimum_frames} {field} frames, found {len(frames)}.")
    orientation = sorted((folder / "3_Voronoi_Geometry").glob("05_*.png"))
    c_rate = sorted((folder / "C-rate").glob("*.png"))
    if len(orientation) != 1 or len(c_rate) != 1:
        raise ValueError(f"{folder.name}: expected exactly one 05_*.png orientation and one C-rate PNG.")
    return SimulationCase(folder, frames, orientation[0], c_rate[0])


def discover_cases(data_dir, field="concentration"):
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    cases = sorted(p for p in root.iterdir()
                   if p.is_dir() and (p / FIELD_DIRECTORIES[field]).is_dir())
    if not cases:
        raise ValueError(f"No cases containing {FIELD_DIRECTORIES[field]}/ found under {root}.")
    return cases


class BatteryDataset(Dataset):
    """A sample is (input[T,9,H,W], future[K,3,H,W]).

    The single-frame stage uses 128-pixel grid patches for train and validation.
    The sequence stage uses patches for training and full images for validation.
    Lazy frame loading avoids requiring the full export collection in RAM.
    """

    def __init__(self, folders, field="concentration", input_length=5, predict_length=1,
                 image_size=512, patch_size=128, use_patches=True, augment=False):
        if min(input_length, predict_length, image_size, patch_size) < 1:
            raise ValueError("Sequence lengths and image dimensions must be positive.")
        if use_patches and image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size.")
        self.input_length, self.predict_length = input_length, predict_length
        self.image_size, self.patch_size = image_size, patch_size
        self.use_patches, self.augment = use_patches, augment
        self.grid = image_size // patch_size if use_patches else 1
        self.cases = [read_case(p, field, input_length + predict_length) for p in folders]
        self.static_cache = {}
        self.samples = []
        for case_index, case in enumerate(self.cases):
            for start in range(len(case.frames) - input_length - predict_length + 1):
                for patch in range(self.grid ** 2):
                    self.samples.append((case_index, start, patch))
        if not self.samples:
            raise ValueError("The supplied cases produce no sequence windows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        case_index, start, patch = self.samples[index]
        case = self.cases[case_index]
        if case_index not in self.static_cache:
            self.static_cache[case_index] = np.concatenate([
                load_rgb(case.orientation, self.image_size),
                load_rgb(case.c_rate, self.image_size),
            ], axis=-1)
        static = self.static_cache[case_index]
        paths = case.frames[start:start + self.input_length + self.predict_length]
        sequence = np.stack([np.concatenate([load_rgb(p, self.image_size), static], axis=-1)
                             for p in paths])
        if self.use_patches:
            row, col = divmod(patch, self.grid)
            h, w = row * self.patch_size, col * self.patch_size
            sequence = sequence[:, h:h + self.patch_size, w:w + self.patch_size]
        tensor = torch.from_numpy(sequence).permute(0, 3, 1, 2).contiguous()
        if self.augment:
            if random.random() > 0.5:
                tensor = tensor.flip(-1)
            if random.random() > 0.5:
                tensor = tensor.flip(-2)
            if random.random() > 0.5:
                tensor = torch.rot90(tensor, 1, [-2, -1])
        return tensor[:self.input_length], tensor[self.input_length:, :3]
