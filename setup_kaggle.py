"""Setup script to prepare project for Kaggle."""

import shutil
from pathlib import Path
import zipfile


def create_kaggle_structure():
    """Create optimized structure for Kaggle."""
    
    # Create kaggle directory
    kaggle_dir = Path("kaggle_upload")
    kaggle_dir.mkdir(exist_ok=True)
    
    # Copy source code
    src_files = [
        "src/models/backbones.py",
        "src/models/pe.py",
        "src/training/train.py",
        "src/training/param_groups.py",
        "src/utils/data/loading.py",
        "src/utils/data/PaddyDataset.py",
        "src/utils/data/transforms.py",
        "src/utils/metrics/plot.py",
    ]
    
    for src_file in src_files:
        src_path = Path(src_file)
        if src_path.exists():
            dest_path = kaggle_dir / src_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            print(f"✓ Copied: {src_file}")
    
    # Copy __init__.py files
    init_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/training/__init__.py",
        "src/utils/__init__.py",
        "src/utils/data/__init__.py",
        "src/utils/metrics/__init__.py",
    ]
    
    for init_file in init_files:
        init_path = Path(init_file)
        dest_path = kaggle_dir / init_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if init_path.exists():
            shutil.copy2(init_path, dest_path)
        else:
            dest_path.touch()
        print(f"✓ Created: {init_file}")
    
    # Create training script for Kaggle
    create_kaggle_train_script(kaggle_dir)
    
    # Create requirements.txt
    create_requirements(kaggle_dir)
    
    # Create README
    create_readme(kaggle_dir)
    
    # Zip everything
    zip_for_kaggle(kaggle_dir)
    
    print("\n✅ Kaggle upload package ready!")
    print(f"📦 Location: {kaggle_dir}")


def create_kaggle_train_script(kaggle_dir):
    """Create optimized training script for Kaggle."""
    
    script_content = '''"""Train MobileNetV3 Small BoT on Kaggle."""

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
    print("\\n📊 Loading datasets...")
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
    print("\\n🏗️ Building model...")
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
    print("\\n🏋️ Starting training...")
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
    print("\\n💾 Saving results...")
    
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
    print("\\n" + "="*60)
    print("🎉 Training Complete!")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<30} {value:.4f}")
        else:
            print(f"  {key:.<30} {value}")
    print("="*60)
    
    print("\\n📥 Output files:")
    print("  - MobileNetV3_Small_BoT_Kaggle_best.pt")
    print("  - metrics.json")
    print("  - history.json")
    print("  - config.json")
    print("  - training_history.png")


if __name__ == "__main__":
    main()
'''
    
    script_path = kaggle_dir / "train_kaggle.py"
    with open(script_path, "w") as f:
        f.write(script_content)
    print(f"✓ Created: train_kaggle.py")


def create_requirements(kaggle_dir):
    """Create requirements.txt."""
    
    requirements = """torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
pandas>=1.3.0
numpy>=1.21.0
tqdm>=4.62.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
"""
    
    with open(kaggle_dir / "requirements.txt", "w") as f:
        f.write(requirements)
    print("✓ Created: requirements.txt")


def create_readme(kaggle_dir):
    """Create README for Kaggle."""
    
    readme = """# Paddy Disease Classification - Training on Kaggle

## 🚀 Quick Start

### 1. Upload Data to Kaggle Dataset
- Create new dataset named `paddy-disease-classification`
- Upload: `data/metadata.csv`, `data/label2id.json`, `data/images/`

### 2. Create Kaggle Notebook
- Enable GPU (Settings → Accelerator → GPU T4 x2)
- Add dataset as input

### 3. Upload Source Code
- Upload this entire folder as Kaggle Dataset
- Name it: `paddy-disease-classification-src`

### 4. Run Training
```python
# In Kaggle Notebook
!python /kaggle/input/paddy-disease-classification-src/train_kaggle.py
```

## 📊 Configuration

Edit `train_kaggle.py` to change:
- `batch_size`: Default 64 (reduce if OOM)
- `epochs`: Default 30
- `learning_rate`: base_lr=5e-5, head_lr=5e-4
- `image_size`: Default 224

## 📥 Output Files

After training, download from `/kaggle/working/`:
- `MobileNetV3_Small_BoT_Kaggle_best.pt` - Best model checkpoint
- `metrics.json` - Final metrics
- `history.json` - Training history
- `training_history.png` - Loss/Accuracy plots

## 💡 Tips

1. **Enable Internet** in notebook settings to download pretrained weights
2. **Use GPU T4 x2** for faster training
3. **Reduce batch_size** to 32 or 16 if you get OOM errors
4. Training takes ~30-45 minutes for 30 epochs

## 📝 Notes

- Kaggle provides 30 hours/week of GPU time
- Session limit: 12 hours continuous
- Save outputs before session ends!
"""
    
    with open(kaggle_dir / "README.md", "w") as f:
        f.write(readme)
    print("✓ Created: README.md")


def zip_for_kaggle(kaggle_dir):
    """Create zip file for easy Kaggle upload."""
    
    zip_path = Path("kaggle_upload.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in kaggle_dir.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(kaggle_dir)
                zipf.write(file, arcname)
    
    print(f"\n✓ Created zip: {zip_path}")
    print(f"  Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    create_kaggle_structure()