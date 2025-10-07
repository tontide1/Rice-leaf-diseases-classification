"""Train Custom CNN models cho Paddy Disease Classification.

Script này hỗ trợ training 3 custom CNN architectures:
- LightweightCNN: ~1.2M params, balanced
- TinyPaddyNet: ~0.5M params, cực nhẹ
- CompactCNN: ~1.8M params, accuracy cao hơn
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset

# Thêm thư mục src vào path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models.backbones import LightweightCNN, TinyPaddyNet, CompactCNN  # noqa: E402
from src.training import get_param_groups, train_model  # noqa: E402
from src.utils.data import (  # noqa: E402
    TransformConfig,
    build_datasets,
)
from src.utils.metrics import plot_history  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Custom CNN models trên Paddy Disease dataset"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="lightweight",
        choices=["lightweight", "tiny", "compact"],
        help="Chọn model architecture: lightweight (1.2M), tiny (0.5M), compact (1.8M)",
    )
    
    # Đường dẫn dữ liệu
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT_DIR / "data/metadata.csv",
        help="Đường dẫn file metadata CSV",
    )
    parser.add_argument(
        "--label2id",
        type=Path,
        default=ROOT_DIR / "data/label2id.json",
        help="Đường dẫn file label2id JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results",
        help="Thư mục lưu kết quả training",
    )
    
    # Cấu hình model
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability trước classifier",
    )
    parser.add_argument(
        "--width-mult",
        type=float,
        default=1.0,
        help="Width multiplier cho LightweightCNN (chỉ áp dụng cho lightweight)",
    )
    
    # Hyperparameters training
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Kích thước ảnh input (224 hoặc 256)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size cho training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Số epoch training (custom CNN cần train lâu hơn)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (<=0 để tắt)",
    )
    
    # Learning rates
    parser.add_argument(
        "--base-lr",
        type=float,
        default=1e-3,
        help="Learning rate chính (train from scratch nên cao hơn)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay cho optimizer",
    )
    
    # Dataloader settings
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Số workers cho DataLoader",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=False,
        help="Pin memory trong DataLoader (tăng tốc GPU)",
    )
    
    # Debug/testing
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Giới hạn số samples training (để test nhanh)",
    )
    parser.add_argument(
        "--valid-limit",
        type=int,
        default=None,
        help="Giới hạn số samples validation",
    )
    
    # Misc
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Tên model cho logging (mặc định: <model>_<timestamp>)",
    )
    parser.add_argument(
        "--scheduler-tmax",
        type=int,
        default=None,
        help="T_max cho CosineAnnealingLR (mặc định = epochs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Force device (mặc định: auto-detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed cho reproducibility",
    )
    
    return parser.parse_args()


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
) -> DataLoader:
    """Tạo PyTorch DataLoader."""
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
    """Giới hạn số samples trong dataset (để debug nhanh)."""
    if limit is None or limit >= len(dataset):
        return dataset
    indices = list(range(limit))
    return Subset(dataset, indices)


def save_results(output_dir: Path, model_name: str, history: dict, metrics: dict) -> None:
    """Lưu kết quả training vào file JSON và plot."""
    run_dir = output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu history
    history_path = run_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã lưu training history: {history_path}")
    
    # Lưu metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã lưu metrics: {metrics_path}")
    
    # Plot training curves
    plot_path = run_dir / "training_plot.png"
    plot_history(history, save_path=str(plot_path))
    print(f"✅ Đã lưu training plot: {plot_path}")
    
    # In ra metrics cuối cùng
    print("\n" + "="*60)
    print("📊 KẾT QUẢ CUỐI CÙNG")
    print("="*60)
    print(f"Model: {metrics['model_name']}")
    print(f"Số params: {metrics['num_params']:,}")
    print(f"Model size: {metrics['size_mb']:.2f} MB")
    print(f"Valid Accuracy: {metrics['valid_acc']*100:.2f}%")
    print(f"Valid F1-Score: {metrics['valid_f1']:.4f}")
    print(f"FPS (inference): {metrics['fps']:.1f}")
    print(f"Checkpoint: {metrics['ckpt_path']}")
    print("="*60)


def create_model(model_type: str, num_classes: int, dropout: float, 
                width_mult: float = 1.0):
    """Khởi tạo model theo loại."""
    if model_type == "lightweight":
        return LightweightCNN(
            num_classes=num_classes,
            dropout=dropout,
            width_mult=width_mult
        )
    elif model_type == "tiny":
        return TinyPaddyNet(
            num_classes=num_classes,
            dropout=dropout
        )
    elif model_type == "compact":
        return CompactCNN(
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main() -> None:
    """Hàm main để train model."""
    args = parse_args()
    
    # Đặt random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Tạo tên model với timestamp nếu không được chỉ định
    if args.model_name is None:
        timestamp = datetime.now().strftime("%d_%m_%Y_%H%M")
        model_display = {
            "lightweight": "LightweightCNN",
            "tiny": "TinyPaddyNet",
            "compact": "CompactCNN"
        }
        args.model_name = f"{model_display[args.model]}_{timestamp}"
    
    print(f"🚀 Bắt đầu training: {args.model_name}")
    print(f"Model architecture: {args.model}")
    print(f"Device: {args.device or 'auto-detect'}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.base_lr}")
    print(f"Dropout: {args.dropout}")
    if args.model == "lightweight":
        print(f"Width multiplier: {args.width_mult}")
    print("")
    
    # Tạo datasets
    print("📂 Đang load datasets...")
    cfg = TransformConfig(image_size=args.image_size)
    datasets = build_datasets(
        metadata_path=args.metadata,
        label2id_path=args.label2id,
        cfg=cfg,
        splits=("train", "valid"),
    )
    
    num_classes = len(datasets["train"].label2id) if hasattr(datasets["train"], "label2id") else 4
    print(f"✅ Đã load datasets: {len(datasets['train'])} train, {len(datasets['valid'])} valid")
    print(f"Số classes: {num_classes}")
    
    # Áp dụng limit nếu có (để debug)
    train_dataset = maybe_subset(datasets["train"], args.train_limit)
    valid_dataset = maybe_subset(datasets["valid"], args.valid_limit)
    
    if args.train_limit or args.valid_limit:
        print(f"⚠️  Debug mode: train={len(train_dataset)}, valid={len(valid_dataset)}")
    
    # Tạo dataloaders
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
    
    # Khởi tạo model
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\n🔧 Khởi tạo model {args.model}...")
    print(f"Device: {device}")
    
    model = create_model(
        model_type=args.model,
        num_classes=num_classes,
        dropout=args.dropout,
        width_mult=args.width_mult
    ).to(device)
    
    # In số params
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model initialized:")
    print(f"   - Total params: {num_params:,}")
    print(f"   - Trainable params: {num_trainable:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer - đơn giản hơn vì train from scratch
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )
    print(f"✅ Optimizer: AdamW (lr={args.base_lr}, wd={args.weight_decay})")
    
    # Scheduler
    t_max = args.scheduler_tmax or args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, t_max)
    )
    print(f"✅ Scheduler: CosineAnnealingLR (T_max={t_max})")
    
    # Gradient scaler cho mixed precision
    scaler = GradScaler(enabled=device.type == "cuda")
    
    # Early stopping
    patience = args.patience if args.patience > 0 else None
    if patience:
        print(f"✅ Early stopping: patience={patience}")
    
    # Training
    print(f"\n{'='*60}")
    print("🎯 BẮT ĐẦU TRAINING")
    print(f"{'='*60}\n")
    
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
    
    # Lưu kết quả
    print(f"\n{'='*60}")
    print("💾 LƯU KẾT QUẢ")
    print(f"{'='*60}\n")
    save_results(args.output_dir, args.model_name, history, metrics)
    
    print(f"\n✅ HOÀN THÀNH! Kết quả đã được lưu tại: {args.output_dir / args.model_name}")


if __name__ == "__main__":
    main()
