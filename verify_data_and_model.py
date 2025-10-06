"""Script để verify data preprocessing và test model forward pass."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Thêm src vào path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models.backbones import EfficientNet_Lite0_CA
from src.utils.data import TransformConfig, build_datasets


def verify_data():
    """Kiểm tra data có NaN/Inf không."""
    print("\n" + "="*60)
    print("VERIFY DATA PREPROCESSING")
    print("="*60)
    
    # Load dataset
    cfg = TransformConfig(image_size=224)
    datasets = build_datasets(
        metadata_path=ROOT_DIR / "data/metadata.csv",
        label2id_path=ROOT_DIR / "data/label2id.json",
        cfg=cfg,
        splits=("train",),
    )
    
    train_dataset = datasets["train"]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    print(f"✓ Dataset loaded: {len(train_dataset)} samples")
    print(f"✓ Number of classes: {len(train_dataset.label2id)}")
    
    # Kiểm tra một vài batch
    print("\nKiểm tra 3 batch đầu tiên...")
    for i, (images, labels) in enumerate(train_loader):
        if i >= 3:
            break
        
        print(f"\n  Batch {i+1}:")
        print(f"    - Shape: {images.shape}")
        print(f"    - Dtype: {images.dtype}")
        print(f"    - Min: {images.min().item():.4f}, Max: {images.max().item():.4f}")
        print(f"    - Mean: {images.mean().item():.4f}, Std: {images.std().item():.4f}")
        print(f"    - Has NaN: {torch.isnan(images).any().item()}")
        print(f"    - Has Inf: {torch.isinf(images).any().item()}")
        print(f"    - Labels shape: {labels.shape}, Min: {labels.min().item()}, Max: {labels.max().item()}")
    
    print("\n✅ Data verification completed!")
    return True


def verify_model_forward():
    """Kiểm tra model forward pass với random input."""
    print("\n" + "="*60)
    print("VERIFY MODEL FORWARD PASS")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    
    # Tạo model
    model = EfficientNet_Lite0_CA(
        num_classes=10,  # Paddy dataset có 10 classes
        reduction=16,
        pretrained=False,  # Không dùng pretrained để test init
        dropout=0.2,
    ).to(device)
    
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    
    # Test forward pass với random input
    batch_size = 8
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"\n✓ Input shape: {x.shape}")
    print(f"  - Min: {x.min().item():.4f}, Max: {x.max().item():.4f}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"\n✓ Output shape: {output.shape}")
    print(f"  - Min: {output.min().item():.4f}, Max: {output.max().item():.4f}")
    print(f"  - Has NaN: {torch.isnan(output).any().item()}")
    print(f"  - Has Inf: {torch.isinf(output).any().item()}")
    
    # Test backward pass
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    loss = criterion(output, labels)
    
    print(f"\n✓ Loss: {loss.item():.4f}")
    print(f"  - Has NaN: {torch.isnan(loss).any().item()}")
    print(f"  - Has Inf: {torch.isinf(loss).any().item()}")
    
    # Test backward
    loss.backward()
    
    # Kiểm tra gradients
    has_nan_grad = False
    has_inf_grad = False
    max_grad = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                has_nan_grad = True
                print(f"  ⚠ NaN gradient in: {name}")
            if torch.isinf(param.grad).any():
                has_inf_grad = True
                print(f"  ⚠ Inf gradient in: {name}")
            max_grad = max(max_grad, param.grad.abs().max().item())
    
    print(f"\n✓ Gradient check:")
    print(f"  - Has NaN gradients: {has_nan_grad}")
    print(f"  - Has Inf gradients: {has_inf_grad}")
    print(f"  - Max gradient magnitude: {max_grad:.4f}")
    
    if not has_nan_grad and not has_inf_grad:
        print("\n✅ Model forward/backward pass OK!")
        return True
    else:
        print("\n❌ Model has gradient issues!")
        return False


def main():
    """Chạy tất cả verification."""
    print("\n" + "="*60)
    print("DATA & MODEL VERIFICATION")
    print("="*60)
    
    # Verify data
    data_ok = verify_data()
    
    # Verify model
    model_ok = verify_model_forward()
    
    # Tổng kết
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Data: {'✅ OK' if data_ok else '❌ FAILED'}")
    print(f"  Model: {'✅ OK' if model_ok else '❌ FAILED'}")
    print("="*60 + "\n")
    
    if data_ok and model_ok:
        print("🎉 Tất cả đều OK! Có thể bắt đầu training với config mới.")
    else:
        print("⚠ Có vấn đề cần fix trước khi training!")


if __name__ == "__main__":
    main()
