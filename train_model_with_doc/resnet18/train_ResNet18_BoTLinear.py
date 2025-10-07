"""Train ResNet18 BoTLinear backbone on Paddy dataset."""

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

# ROOT_DIR trỏ đến thư mục gốc của project (thư mục hiện tại)
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# Thêm ROOT_DIR vào sys.path để import từ src
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.backbones import ResNet18_BoTLinear  # noqa: E402
from src.training import get_param_groups, train_model  # noqa: E402
from src.utils.data import (  # noqa: E402
    TransformConfig,
    build_datasets,
)
from src.utils.metrics.plot import plot_history  # noqa: E402

try:
    import matplotlib
    matplotlib.use('Agg')  # Backend không tương tác để lưu biểu đồ
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    """Parse các tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Train ResNet18 BoTLinear trên Paddy dataset")
    
    # Đường dẫn dữ liệu
    parser.add_argument("--metadata", type=Path, default=Path(__file__).parent / "data/metadata.csv",
                        help="Đường dẫn đến file metadata CSV")
    parser.add_argument("--label2id", type=Path, default=Path(__file__).parent / "data/label2id.json",
                        help="Đường dẫn đến file label2id JSON")
    
    # Tham số hình ảnh và training
    parser.add_argument("--image-size", type=int, default=224, help="Kích thước ảnh đầu vào")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size cho training")
    parser.add_argument("--epochs", type=int, default=10, help="Số epoch training")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (<=0 để tắt)")
    
    # Tham số optimizer
    parser.add_argument("--base-lr", type=float, default=1e-4, help="Learning rate cho backbone")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Learning rate cho classifier head")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay cho optimizer")
    
    # Tham số model
    parser.add_argument("--heads", type=int, default=4, help="Số attention heads trong BoTLinear block")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability trước classifier")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Sử dụng backbone pretrained từ ImageNet")
    
    # Tham số dataloader
    parser.add_argument("--num-workers", type=int, default=4, help="Số workers cho dataloader")
    parser.add_argument("--pin-memory", action="store_true", default=False,
                        help="Pin memory trong dataloaders")
    
    # Giới hạn dữ liệu (để test nhanh)
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Giới hạn số mẫu training (để chạy nhanh)")
    parser.add_argument("--valid-limit", type=int, default=None,
                        help="Giới hạn số mẫu validation")
    
    # Tham số khác
    parser.add_argument("--model-name", type=str, default="ResNet18_BoTLinear",
                        help="Tên model cho logging/checkpoint")
    parser.add_argument("--scheduler-tmax", type=int, default=None,
                        help="Override T_max cho CosineAnnealingLR")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                        help="Chọn device cụ thể")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Vẽ biểu đồ training history sau khi training")
    parser.add_argument("--save-history", action="store_true", default=False,
                        help="Lưu training history vào file JSON")
    
    return parser.parse_args()


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last=False) -> DataLoader:
    """Tạo DataLoader từ dataset."""
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
    """Tạo subset của dataset nếu có limit."""
    if limit is None or limit >= len(dataset):
        return dataset
    indices = list(range(limit))
    return Subset(dataset, indices)


def main() -> None:
    """Hàm main để chạy training."""
    args = parse_args()

    # Set random seed để đảm bảo reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Cấu hình transform cho ảnh
    cfg = TransformConfig(image_size=args.image_size)
    
    # Load datasets
    datasets = build_datasets(
        metadata_path=args.metadata,
        label2id_path=args.label2id,
        cfg=cfg,
        splits=("train", "valid"),
    )

    # Lấy số lớp từ dataset
    num_classes = len(datasets["train"].label2id) if hasattr(datasets["train"], "label2id") else 4

    # Tạo subset nếu có limit
    train_dataset = maybe_subset(datasets["train"], args.train_limit)
    valid_dataset = maybe_subset(datasets["valid"], args.valid_limit)

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

    # Chọn device (GPU nếu có, không thì CPU)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "="*60)
    print(f"Training Configuration:")
    print("="*60)
    print(f"  Model: {args.model_name}")
    print(f"  Device: {device}")
    print(f"  Image size: {args.image_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(valid_dataset)}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Attention heads: {args.heads}")
    print(f"  Dropout: {args.dropout}")
    print("="*60 + "\n")

    # Kiểm tra số heads hợp lệ
    # Với ResNet18, c_out=512, c_mid=128
    # c_mid phải chia hết cho cả heads và 4 (yêu cầu của PositionalEncoding2D)
    # Các giá trị heads tốt nhất: 1, 2, 4, 8, 16, 32, 64
    import math
    c_mid_base = 128
    lcm = (args.heads * 4) // math.gcd(args.heads, 4)
    c_mid_actual = (c_mid_base // lcm) * lcm
    if c_mid_actual < lcm:
        c_mid_actual = lcm
    
    # Cảnh báo nếu c_mid bị thay đổi nhiều
    if abs(c_mid_actual - c_mid_base) > 20:
        print(f"⚠️  Warning: heads={args.heads} sẽ làm thay đổi c_mid từ {c_mid_base} xuống {c_mid_actual}")
        print(f"   Điều này có thể ảnh hưởng đến performance.")
        print(f"   Các giá trị heads khuyến nghị: 1, 2, 4, 8, 16, 32, 64")
        print(f"   Continuing anyway...\n")

    # Khởi tạo model ResNet18_BoTLinear
    model = ResNet18_BoTLinear(
        num_classes=num_classes,
        heads=args.heads,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer với learning rate khác nhau cho backbone và head
    param_groups = get_param_groups(
        model,
        base_lr=args.base_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    # Learning rate scheduler
    t_max = args.scheduler_tmax or args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t_max))

    # Gradient scaler cho mixed precision training
    scaler = GradScaler(enabled=device.type == "cuda")

    # Early stopping patience
    patience = args.patience if args.patience > 0 else None

    # Bắt đầu training
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

    # In kết quả cuối cùng
    print("\n" + "="*60)
    print("Training Finished! Final Metrics:")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<30} {value:.4f}")
        else:
            print(f"  {key:.<30} {value}")
    print("="*60 + "\n")

    # Lưu history và metrics vào file JSON
    if args.save_history:
        # Tạo thư mục results với timestamp
        timestamp = datetime.now().strftime("%d_%m_%Y_%H%M")
        run_dir = Path("results") / f"{args.model_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Đang lưu kết quả vào: {run_dir}")
        
        # Lưu training history
        history_file = run_dir / "history.json"
        history_serializable = {
            key: [float(v) for v in value] if hasattr(value, '__iter__') else value
            for key, value in history.items()
        }
        with open(history_file, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        print(f"✓ Training history đã lưu: {history_file}")
        
        # Lưu final metrics
        metrics_file = run_dir / "metrics.json"
        metrics_serializable = {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in metrics.items()
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"✓ Final metrics đã lưu: {metrics_file}")
        
        # Lưu biểu đồ training
        if MATPLOTLIB_AVAILABLE:
            try:
                import matplotlib.pyplot as plt  # Import lại để tránh lỗi
                plot_file = run_dir / "training_plot.png"
                plot_history(
                    history, 
                    title=f"{args.model_name} Training History (Acc: {metrics['valid_acc']:.2%})"
                )
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Biểu đồ training đã lưu: {plot_file}")
            except Exception as e:
                print(f"⚠ Không thể lưu biểu đồ: {e}")
        else:
            print("⚠ Matplotlib không khả dụng, bỏ qua việc lưu biểu đồ")
        
        print(f"\n✅ Tất cả kết quả đã được lưu vào: {run_dir}\n")

    # Hiển thị biểu đồ training history (nếu được yêu cầu)
    if args.plot:
        print("\nĐang hiển thị biểu đồ training...")
        try:
            plot_history(
                history, 
                title=f"{args.model_name} Training History (Acc: {metrics['valid_acc']:.2%})"
            )
            if MATPLOTLIB_AVAILABLE:
                import matplotlib.pyplot as plt  # Import lại để tránh lỗi
                plt.show()
        except Exception as e:
            print(f"⚠ Không thể hiển thị biểu đồ: {e}")


if __name__ == "__main__":
    main()
