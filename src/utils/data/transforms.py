"""Utility factory functions for image transforms used in training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from torchvision import transforms
from torchvision.transforms import functional as F


@dataclass(frozen=True)
class TransformConfig:
    """Configuration options for preprocessing."""

    image_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    pad_fill: int = 0


class SquarePad:
    """Pad rectangular images to square so resize does not distort aspect."""

    def __init__(self, fill: int = 0):
        self.fill = fill

    def __call__(self, img):
        width, height = img.size
        if width == height:
            return img

        max_side = max(width, height)
        pad_left = (max_side - width) // 2
        pad_right = max_side - width - pad_left
        pad_top = (max_side - height) // 2
        pad_bottom = max_side - height - pad_top

        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)


def train_transforms(cfg: TransformConfig = TransformConfig()) -> transforms.Compose:
    """Training pipeline with augmentations and square padding."""

    return transforms.Compose([
        SquarePad(fill=cfg.pad_fill),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ])


def eval_transforms(cfg: TransformConfig = TransformConfig()) -> transforms.Compose:
    """Validation/Test preprocessing with deterministic steps."""

    return transforms.Compose([
        SquarePad(fill=cfg.pad_fill),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ])

