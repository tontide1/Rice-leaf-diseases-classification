"""Train MobileNetV3 Small BoT Linear backbone on Paddy dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional
import json

import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset

# Fix: ROOT_DIR should point to the project root (current directory)
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# Also add ROOT_DIR to sys.path to allow imports from src
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.backbones import MobileNetV3_Small_BoT_Linear  # noqa: E402
from src.training import get_param_groups, train_model  # noqa: E402
from src.utils.data import (  # noqa: E402
    TransformConfig,
    build_datasets,
)
from src.utils.metrics.plot import plot_history  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3-Small BoT Linear (Linear Attention) on Paddy dataset"
    )
    
    # Data paths
    parser.add_argument(
        "--metadata", 
        type=Path, 
        default=Path(__file__).parent / "data/metadata.csv",
        help="Path to metadata CSV"
    )
    parser.add_argument(
        "--label2id", 
        type=Path, 
        default=Path(__file__).parent / "data/label2id.json",
        help="Path to label2id JSON"
    )
    
    # Model configuration
    parser.add_argument(
        "--image-size", 
        type=int, 
        default=224, 
        help="Input image size"
    )
    parser.add_argument(
        "--heads", 
        type=int, 
        default=4, 
        help="Number of attention heads in BoT Linear block"
    )
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.1, 
        help="Dropout probability before classifier (default: 0.1)"
    )
    parser.add_argument(
        "--pretrained", 
        action="store_true", 
        default=False,
        help="Use ImageNet pretrained backbone"
    )
    
    # Training configuration
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=30, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=10, 
        help="Early stopping patience (<=0 disables)"
    )
    parser.add_argument(
        "--base-lr", 
        type=float, 
        default=5e-5, 
        help="Base learning rate for backbone (default: 5e-5)"
    )
    parser.add_argument(
        "--head-lr", 
        type=float, 
        default=5e-4, 
        help="Learning rate for classifier head (default: 5e-4)"
    )
    parser.add_argument(
        "--weight-decay", 
        type=float, 
        default=1e-2, 
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--scheduler-tmax", 
        type=int, 
        default=None,
        help="Override T_max for CosineAnnealingLR"
    )
    
    # DataLoader configuration
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=4, 
        help="Dataloader workers"
    )
    parser.add_argument(
        "--pin-memory", 
        action="store_true", 
        default=False,
        help="Pin memory in dataloaders"
    )
    
    # Debug & testing
    parser.add_argument(
        "--train-limit", 
        type=int, 
        default=None,
        help="Optional limit on number of training samples for quick runs"
    )
    parser.add_argument(
        "--valid-limit", 
        type=int, 
        default=None,
        help="Optional limit on number of validation samples"
    )
    
    # Logging & output
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="MobileNetV3_Small_BoT_Linear",
        help="Model name for logging/checkpoint"
    )
    parser.add_argument(
        "--plot", 
        action="store_true", 
        default=False,
        help="Plot training history after training"
    )
    parser.add_argument(
        "--save-history", 
        action="store_true", 
        default=False,
        help="Save training history to JSON file"
    )
    
    # Device & seed
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        choices=["cpu", "cuda"],
        help="Force device selection"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    
    return parser.parse_args()


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last=False) -> DataLoader:
    """Create DataLoader with specified configuration."""
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
    """Optionally limit dataset to first N samples for quick testing."""
    if limit is None or limit >= len(dataset):
        return dataset
    indices = list(range(limit))
    return Subset(dataset, indices)


def print_model_info(model, args):
    """Print model architecture information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE: MobileNetV3-Small BoT Linear")
    print("="*70)
    print(f"  Backbone:           MobileNetV3-Small (pretrained={args.pretrained})")
    print(f"  Attention:          BoTNet Linear Block (heads={args.heads})")
    print(f"  Attention Type:     Linear Attention (O(N) complexity)")
    print(f"  Dropout:            {args.dropout}")
    print(f"  Total Parameters:   {total_params:,}")
    print(f"  Trainable Params:   {trainable_params:,}")
    print(f"  Model Size:         ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print("="*70 + "\n")


def print_training_config(args):
    """Print training configuration."""
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"  Epochs:             {args.epochs}")
    print(f"  Batch Size:         {args.batch_size}")
    print(f"  Image Size:         {args.image_size}x{args.image_size}")
    print(f"  Base LR:            {args.base_lr}")
    print(f"  Head LR:            {args.head_lr}")
    print(f"  Weight Decay:       {args.weight_decay}")
    print(f"  Early Stopping:     {'Enabled (patience=' + str(args.patience) + ')' if args.patience > 0 else 'Disabled'}")
    print(f"  Device:             {args.device or 'auto'}")
    print(f"  Num Workers:        {args.num_workers}")
    print(f"  Pin Memory:         {args.pin_memory}")
    print("="*70 + "\n")


def main() -> None:
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Print configuration
    print_training_config(args)

    # Build datasets
    print("Loading datasets...")
    cfg = TransformConfig(image_size=args.image_size)
    datasets = build_datasets(
        metadata_path=args.metadata,
        label2id_path=args.label2id,
        cfg=cfg,
        splits=("train", "valid"),
    )

    num_classes = len(datasets["train"].label2id) if hasattr(datasets["train"], "label2id") else 4
    print(f"✓ Loaded datasets: {len(datasets['train'])} train, {len(datasets['valid'])} valid")
    print(f"✓ Number of classes: {num_classes}\n")

    # Apply dataset limits if specified
    train_dataset = maybe_subset(datasets["train"], args.train_limit)
    valid_dataset = maybe_subset(datasets["valid"], args.valid_limit)

    if args.train_limit:
        print(f"⚠ Training limited to {len(train_dataset)} samples")
    if args.valid_limit:
        print(f"⚠ Validation limited to {len(valid_dataset)} samples\n")

    # Create dataloaders
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

    # Setup device
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Build model
    print("Building model...")
    model = MobileNetV3_Small_BoT_Linear(
        num_classes=num_classes,
        heads=args.heads,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    print_model_info(model, args)

    # Setup training components
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

    scaler = GradScaler('cuda', enabled=device.type == "cuda")

    patience = args.patience if args.patience > 0 else None

    # Train model
    print("Starting training...\n")
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

    # Print final results
    print("\n" + "="*70)
    print("TRAINING FINISHED - FINAL METRICS")
    print("="*70)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<35} {value:.4f}")
        else:
            print(f"  {key:.<35} {value}")
    print("="*70 + "\n")

    # Save history to JSON
    if args.save_history:
        history_file = Path("results") / f"{args.model_name}_history.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {
            key: [float(v) for v in value] if hasattr(value, '__iter__') else value
            for key, value in history.items()
        }
        
        with open(history_file, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        print(f"✓ Training history saved to: {history_file}")
        
        # Also save metrics
        metrics_file = Path("results") / f"{args.model_name}_metrics.json"
        metrics_serializable = {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in metrics.items()
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"✓ Metrics saved to: {metrics_file}")

    # Plot training history
    if args.plot:
        print("\nGenerating training plots...")
        try:
            plot_history(
                history, 
                title=f"{args.model_name} Training History (Acc: {metrics['valid_acc']:.2%})"
            )
            print("✓ Plot displayed successfully")
        except Exception as e:
            print(f"⚠ Could not display plot: {e}")

    print("\n🎉 Training completed successfully!\n")


if __name__ == "__main__":
    main()
