"""Generate label-to-id mappings from the training split metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    metadata_path = Path("data/metadata.csv")
    if not metadata_path.exists():
        raise SystemExit("metadata.csv không tồn tại. Hãy chạy prepare_metadata trước.")

    df = pd.read_csv(metadata_path)
    train_df = df[df["split"] == "train"]
    if train_df.empty:
        raise SystemExit("Không tìm thấy dữ liệu train trong metadata.")

    labels = sorted(train_df["label"].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    with (data_dir / "label2id.json").open("w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    with (data_dir / "id2label.json").open("w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    print("Đã tạo mapping label2id và id2label:")
    print(label2id)


if __name__ == "__main__":
    main()

