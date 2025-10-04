"""Train MobileNetV3 Small BoT on Kaggle."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, '/kaggle/working')
sys.path.insert(0, '/kaggle/input/paddy-disease-classification-src')

import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset
import json

from src.models.backbones import MobileNetV3_Small_BoT
from src.training import get_param_groups, train_model
from src.utils.data import TransformConfig, build_datasets
from src.utils.metrics.plot import plot_history


# Kaggle paths
KAGGLE_INPUT = Path("/kaggle/input/paddy-disease-classification")
KAGGLE_WORKING = Path("/kaggle/working")

# Training configuration
CONFIG = {
    "image_size": 224,
    "batch_size": 64,
    "epochs": 30,
    "base_lr": 5e-5,
    "head_lr": 5e-4,
    "weight_decay": 1e-2,
    "patience": 10,
    "heads": 4,
    "dropout": 0.1,
    "pretrained": True,
    "num_workers": 2,  # Kaggle has limited workers
    "pin_memory": True,
    "seed": 42,
}


def main():
    print("="*60)
    print("🚀 Starting Training on Kaggle")
    print("="*60)
    
    # Set seed
    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
    
    # Build datasets
    print("\n📊 Loading datasets...")
    cfg = TransformConfig(image_size=CONFIG["image_size"])
    datasets = build_datasets(
        metadata_path=KAGGLE_INPUT / "metadata.csv",
        label2id_path=KAGGLE_INPUT / "label2id.json",
        cfg=cfg,
        splits=("train", "valid"),
    )
    
    num_classes = len(datasets["train"].label2id)
    print(f"✓ Loaded {len(datasets['train'])} train, {len(datasets['valid'])} valid samples")
    print(f"✓ Number of classes: {num_classes}")
    
    # Create dataloaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        drop_last=True,
        persistent_workers=True,
    )
    
    valid_loader = DataLoader(
        datasets["valid"],
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        persistent_workers=True,
    )
    
    # Build model
    print("\n🏗️ Building model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3_Small_BoT(
        num_classes=num_classes,
        heads=CONFIG["heads"],
        pretrained=CONFIG["pretrained"],
        dropout=CONFIG["dropout"],
    ).to(device)
    
    print(f"✓ Model created on {device}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    param_groups = get_param_groups(
        model,
        base_lr=CONFIG["base_lr"],
        head_lr=CONFIG["head_lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"]
    )
    scaler = GradScaler('cuda', enabled=device.type == "cuda")
    
    # Train
    print("\n🏋️ Starting training...")
    print("="*60)
    
    history, metrics = train_model(
        model_name="MobileNetV3_Small_BoT_Kaggle",
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
        epochs=CONFIG["epochs"],
        patience=CONFIG["patience"],
        fps_image_size=CONFIG["image_size"],
    )
    
    # Save results
    print("\n💾 Saving results...")
    
    # Save metrics
    with open(KAGGLE_WORKING / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save history
    history_serializable = {
        key: [float(v) for v in value] if hasattr(value, '__iter__') else value
        for key, value in history.items()
    }
    with open(KAGGLE_WORKING / "history.json", "w") as f:
        json.dump(history_serializable, f, indent=2)
    
    # Save config
    with open(KAGGLE_WORKING / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    
    # Plot and save
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        plot_history(history, title="Training History")
        import matplotlib.pyplot as plt
        plt.savefig(KAGGLE_WORKING / "training_history.png", dpi=150, bbox_inches='tight')
        print("✓ Plot saved")
    except Exception as e:
        print(f"⚠ Could not save plot: {e}")
    
    # Print final results
    print("\n" + "="*60)
    print("🎉 Training Complete!")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<30} {value:.4f}")
        else:
            print(f"  {key:.<30} {value}")
    print("="*60)
    
    print("\n📥 Output files:")
    print("  - MobileNetV3_Small_BoT_Kaggle_best.pt")
    print("  - metrics.json")
    print("  - history.json")
    print("  - config.json")
    print("  - training_history.png")


if __name__ == "__main__":
    main()
