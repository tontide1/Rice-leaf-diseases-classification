from pathlib import Path
from src.utils.data import TransformConfig, build_dataloaders


def main() -> None:
    loaders = build_dataloaders(
        metadata_path=Path("data/metadata.csv"),
        label2id_path=Path("data/label2id.json"),
        cfg=TransformConfig(image_size=224),
        batch_size=16,
        num_workers=2,
        pin_memory=False,
        splits=("train", "valid", "test"),
    )

    for split, loader in loaders.items():
        x, y = next(iter(loader))
        print(split, x.shape, y.shape, y.unique())

    print("Loaded splits:", list(loaders.keys()))


if __name__ == "__main__":
    main()

