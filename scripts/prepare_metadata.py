#!/usr/bin/env python

"""Generate metadata CSV for Paddy disease dataset.

This script crawls the dataset directory, collects absolute image paths and
labels (directory names), optionally shuffles the entries, and splits them into
train/validation/test subsets with stratification. The result is written to a
CSV file with columns: ``path``, ``label`` and ``split``.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split


def collect_samples(root: pathlib.Path) -> pd.DataFrame:
    """Collect image samples under ``root`` grouped by sub-directory labels."""

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    rows: List[dict[str, str]] = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name
        for img_path in cls_dir.glob("*"):
            if img_path.is_file():
                rows.append({
                    "path": str(img_path.resolve()),
                    "label": label,
                })

    if not rows:
        raise RuntimeError(f"No image files found under {root}")

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def stratified_split(
    df: pd.DataFrame,
    valid_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:
    """Split dataframe into train/valid/test with stratification."""

    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("valid_ratio must be between 0 and 1")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    if valid_ratio + test_ratio >= 1.0:
        raise ValueError("valid_ratio + test_ratio must be < 1.0")

    temp_ratio = valid_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=42,
    )

    if temp_df.empty:
        raise RuntimeError("Temporary dataframe empty after initial split")

    valid_share = valid_ratio / temp_ratio
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - valid_share,
        stratify=temp_df["label"],
        random_state=42,
    )

    train_df = train_df.assign(split="train")
    valid_df = valid_df.assign(split="valid")
    test_df = test_df.assign(split="test")

    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    return full_df[["path", "label", "split"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset metadata CSV")
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=pathlib.Path("data"),
        help="Root directory containing class sub-folders",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data/metadata.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (fraction of the whole dataset)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (fraction of the whole dataset)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = collect_samples(args.data_root.resolve())
    full_df = stratified_split(df, args.valid_ratio, args.test_ratio)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(args.output, index=False)

    counts = full_df["split"].value_counts()
    print(f"Saved metadata for {len(full_df)} images to {args.output.resolve()}")
    print("Split counts:")
    print(counts.sort_index())


if __name__ == "__main__":
    main()

