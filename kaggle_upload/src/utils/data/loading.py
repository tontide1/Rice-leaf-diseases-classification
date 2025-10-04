"""Helpers to create PaddyDataset instances and PyTorch DataLoaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Optional

import pandas as pd
from torch.utils.data import DataLoader

from .PaddyDataset import PaddyDataset
from .transforms import TransformConfig, eval_transforms, train_transforms

SplitLiteral = Literal["train", "valid", "test"]


def _load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"metadata file not found: {path}")
    df = pd.read_csv(path)
    required_cols = {"path", "label", "split"}
    if not required_cols.issubset(df.columns):
        missing = ", ".join(sorted(required_cols - set(df.columns)))
        raise ValueError(f"metadata.csv missing columns: {missing}")
    return df


def _load_label_map(path: Path) -> Mapping[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"label2id file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {str(k): int(v) for k, v in mapping.items()}


def build_datasets(
    metadata_path: Path,
    label2id_path: Path,
    cfg: Optional[TransformConfig] = None,
    splits: Iterable[SplitLiteral] = ("train", "valid", "test"),
) -> Dict[SplitLiteral, PaddyDataset]:
    """Create ``PaddyDataset`` objects for requested splits."""

    cfg = cfg or TransformConfig()
    df = _load_metadata(metadata_path)
    label2id = _load_label_map(label2id_path)

    datasets: Dict[SplitLiteral, PaddyDataset] = {}
    for split in splits:
        split_df = df[df["split"] == split].copy()
        if split_df.empty:
            continue
        split_df = split_df.reset_index(drop=True)

        if split == "train":
            tfm = train_transforms(cfg)
        else:
            tfm = eval_transforms(cfg)

        missing_labels = set(split_df["label"]) - set(label2id)
        if missing_labels:
            raise ValueError(
                f"Labels {sorted(missing_labels)} not present in label2id mapping"
            )

        datasets[split] = PaddyDataset(split_df, tfm, label2id)

    if not datasets:
        raise ValueError("No datasets were created. Check metadata splits.")
    return datasets


def build_dataloaders(
    metadata_path: Path,
    label2id_path: Path,
    cfg: Optional[TransformConfig] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_drop_last: bool = False,
    persistent_workers: Optional[bool] = None,
    splits: Iterable[SplitLiteral] = ("train", "valid", "test"),
) -> Dict[SplitLiteral, DataLoader]:
    """Build DataLoaders for the requested splits."""

    datasets = build_datasets(metadata_path, label2id_path, cfg, splits)
    loaders: Dict[SplitLiteral, DataLoader] = {}

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    for split, dataset in datasets.items():
        is_train = split == "train"
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=train_drop_last if is_train else False,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
        loaders[split] = loader

    return loaders

