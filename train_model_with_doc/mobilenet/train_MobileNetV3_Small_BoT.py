"""Train MobileNetV3 Small BoT backbone on Paddy dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset

# Fix: ROOT_DIR should point to the project root (current directory)
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# Also add ROOT_DIR to sys.path to allow imports from src
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.backbones import MobileNetV3_Small_BoT  # noqa: E402
from src.training import get_param_groups, train_model  # noqa: E402
from src.utils.data import (  # noqa: E402
    TransformConfig,
    build_datasets,
)
from src.utils.metrics.plot import plot_history  # noqa: E402

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving plots
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small BoT on Paddy dataset")
    # parser.add_argument("--metadata", type=Path, default=Path(__file__).parent / "data/metadata.csv",
    #                     help="Path to metadata CSV")
    # parser.add_argument("--label2id", type=Path, default=Path(__file__).parent / "data/label2id.json",
    #                     help="Path to label2id JSON")
    parser.add_argument("--metadata", type=Path, default=Path(__file__).parent / "data/metadata.csv",
                        help="Path to metadata CSV")
    parser.add_argument("--label2id", type=Path, default=Path(__file__).parent / "data/label2id.json",
                        help="Path to label2id JSON")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (<=0 disables)")
    parser.add_argument("--base-lr", type=float, default=1e-4, help="Base learning rate for backbone")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Learning rate for classifier head")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for optimizer")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads in BoT block")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability before classifier")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Use ImageNet pretrained backbone")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--pin-memory", action="store_true", default=False,
                        help="Pin memory in dataloaders")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Optional limit on number of training samples for quick runs")
    parser.add_argument("--valid-limit", type=int, default=None,
                        help="Optional limit on number of validation samples")
    parser.add_argument("--model-name", type=str, default="MobileNetV3_Small_BoT",
                        help="Model name for logging/checkpoint")
    parser.add_argument("--scheduler-tmax", type=int, default=None,
                        help="Override T_max for CosineAnnealingLR")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                        help="Force device selection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Plot training history after training")
    parser.add_argument("--save-history", action="store_true", default=False,
                        help="Save training history to JSON file")
    return parser.parse_args()


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last=False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def maybe_subset(dataset, limit: Optional[int]):
    if limit is None or limit >= len(dataset):
        return dataset
    indices = list(range(limit))
    return Subset(dataset, indices)


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = TransformConfig(image_size=args.image_size)
    datasets = build_datasets(
        metadata_path=args.metadata,
        label2id_path=args.label2id,
        cfg=cfg,
        splits=("train", "valid"),
    )

    num_classes = len(datasets["train"].label2id) if hasattr(datasets["train"], "label2id") else 4

    train_dataset = maybe_subset(datasets["train"], args.train_limit)
    valid_dataset = maybe_subset(datasets["valid"], args.valid_limit)

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MobileNetV3_Small_BoT(
        num_classes=num_classes,
        heads=args.heads,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    param_groups = get_param_groups(
        model,
        base_lr=args.base_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    t_max = args.scheduler_tmax or args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t_max))

    scaler = GradScaler(enabled=device.type == "cuda")

    patience = args.patience if args.patience > 0 else None

    history, metrics = train_model(
        model_name=args.model_name,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        gpu_aug=None,
        MEAN=None,
        STD=None,
        epochs=args.epochs,
        patience=patience,
        fps_image_size=args.image_size,
    )

    print("\n" + "="*60)
    print("Training finished. Final Metrics:")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<30} {value:.4f}")
        else:
            print(f"  {key:.<30} {value}")
    print("="*60 + "\n")

    # Save history to JSON
    if args.save_history:
        # Create timestamped results directory
        timestamp = datetime.now().strftime("%d_%m_%Y_%H%M")
        run_dir = Path("results") / f"{args.model_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Saving results to: {run_dir}")
        
        # Save training history
        history_file = run_dir / "history.json"
        history_serializable = {
            key: [float(v) for v in value] if hasattr(value, '__iter__') else value
            for key, value in history.items()
        }
        with open(history_file, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        print(f"✓ Training history saved to: {history_file}")
        
        # Save final metrics
        metrics_file = run_dir / "metrics.json"
        metrics_serializable = {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in metrics.items()
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"✓ Final metrics saved to: {metrics_file}")
        
        # Save training plot
        if MATPLOTLIB_AVAILABLE:
            try:
                plot_file = run_dir / "training_plot.png"
                plot_history(
                    history, 
                    title=f"{args.model_name} Training History (Acc: {metrics['valid_acc']:.2%})"
                )
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Training plot saved to: {plot_file}")
            except Exception as e:
                print(f"⚠ Could not save plot: {e}")
        else:
            print("⚠ Matplotlib not available, skipping plot save")
        
        print(f"\n✅ All results saved to: {run_dir}\n")

    # Plot training history (display only)
    if args.plot:
        print("\nDisplaying training plot...")
        try:
            plot_history(
                history, 
                title=f"{args.model_name} Training History (Acc: {metrics['valid_acc']:.2%})"
            )
            if MATPLOTLIB_AVAILABLE:
                plt.show()
        except Exception as e:
            print(f"⚠ Could not display plot: {e}")


if __name__ == "__main__":
    main()

