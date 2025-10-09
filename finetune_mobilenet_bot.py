"""
Script fine-tune model MobileNetV3_Small_BoT đã train trước trên dataset_0806.
Dataset đã được chia sẵn thành train/valid/test với 4 classes:
- bacterial_leaf_blight
- brown_spot
- healthy
- leaf_blast
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix
)
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Tắt giới hạn cảnh báo

# Thêm src vào path để import modules
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models.backbones.mobilenet import MobileNetV3_Small_BoT
from src.training.train import train_one_epoch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SquarePad:
    """
    Giữ nguyên tỷ lệ ảnh gốc, không làm méo ảnh. -> padding
    """
    def __init__(self, fill=0):
        self.fill = fill
    
    def __call__(self, image):
        from PIL import Image, ImageOps
        # Đảm bảo image là PIL Image
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")
        
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return ImageOps.expand(image, border=padding, fill=self.fill)


def setup_seed(seed: int = 42):
    """Thiết lập random seed cho reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(
    data_dir: Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoader cho train/valid/test từ dataset_0806.
    
    Args:
        data_dir: Đường dẫn tới thư mục dataset_0806
        image_size: Kích thước ảnh sau resize
        batch_size: Batch size
        num_workers: Số workers cho DataLoader
        
    Returns:
        train_loader, valid_loader, test_loader
    """
    # Transform cho training: augmentation mạnh hơn để giảm overfitting
    train_transform = transforms.Compose([
        # Padding ảnh thành hình vuông để không bị méo
        SquarePad(fill=0),
        # Resize ảnh vuông về kích thước mong muốn
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # Tăng rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Tăng augmentation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Thêm translation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Transform cho validation/test: padding + resize, không có augmentation
    eval_transform = transforms.Compose([
        # Padding ảnh thành hình vuông để không bị méo
        SquarePad(fill=0),
        # Resize ảnh vuông về kích thước mong muốn
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=data_dir / "train",
        transform=train_transform
    )
    
    valid_dataset = datasets.ImageFolder(
        root=data_dir / "valid",
        transform=eval_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=data_dir / "test",
        transform=eval_transform
    )
    
    # Tạo DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Valid samples: {len(valid_dataset)}")
    print(f"[INFO] Test samples: {len(test_dataset)}")
    print(f"[INFO] Classes: {train_dataset.classes}")
    
    return train_loader, valid_loader, test_loader


def load_pretrained_model(
    checkpoint_path: Path,
    num_classes: int = 4,
    device: torch.device = torch.device("cpu")
) -> nn.Module:
    """
    Load model pretrained từ checkpoint.
    
    Args:
        checkpoint_path: Đường dẫn tới file .pt
        num_classes: Số lượng classes
        device: Device để load model
        
    Returns:
        Model đã load weights
    """
    print(f"[INFO] Loading pretrained model từ: {checkpoint_path}")
    
    # Tạo model architecture MobileNetV3_Small_BoT
    model = MobileNetV3_Small_BoT(
        num_classes=num_classes,
        heads=4,  # Số heads cho BoTNet attention
        pretrained=False,
        dropout=0.3  # Tăng dropout từ 0.1 lên 0.3 để regularization mạnh hơn
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        print(f"[INFO] Checkpoint metadata: {checkpoint.get('meta', {})}")
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    
    print(f"[INFO] Model đã được load thành công!")
    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str]
) -> Dict:
    """
    Đánh giá model trên một DataLoader.
    
    Returns:
        Dictionary chứa metrics: loss, acc, f1, recall, predictions
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    running_loss = 0.0
    total = 0
    
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
        
        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Tính toán metrics
    avg_loss = running_loss / total
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Classification report cho từng class
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "recall_macro": recall_macro,
        "predictions": (all_labels, all_preds),
        "report": report
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train model trong 1 epoch.
    
    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        
        pbar.set_postfix(loss=running_loss/total, acc=correct/total)
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def save_training_plot(history: Dict, save_path: Path):
    """Vẽ và lưu biểu đồ training history."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot - chỉ dùng đường line, không có marker
    axes[0].plot(epochs, history["train_loss"], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history["valid_loss"], 'r-', label='Valid Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training và Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot - chỉ dùng đường line, không có marker
    axes[1].plot(epochs, history["train_acc"], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history["valid_acc"], 'r-', label='Valid Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training và Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Đã lưu biểu đồ training tại: {save_path}")


def calculate_fps(model: nn.Module, device: torch.device, image_size: int = 224) -> float:
    """Tính FPS của model."""
    model.eval()
    
    batch_size = 16
    num_runs = 100
    warmup = 20
    
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
    
    # Warmup
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Đo thời gian
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_sec = elapsed_ms / 1000.0
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        elapsed_sec = time.time() - start_time
    
    total_images = num_runs * batch_size
    fps = total_images / elapsed_sec
    
    return fps


def main():
    start_time = time.time()  # Bắt đầu đo thời gian
    
    # ==================== CONFIGURATION ====================
    PRETRAINED_MODEL_PATH = Path("models/train 2/MobileNetV3_Small_BoT_best.pt")
    DATASET_DIR = Path("dataset_0806")
    OUTPUT_DIR = Path("results") / f"MobileNetV3_Small_BoT_finetuned_{datetime.now().strftime('%d_%m_%Y_%H%M')}"
    
    # Hyperparameters - Tối ưu để giảm overfitting
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    PATIENCE = 10  # Tăng patience để cho model học lâu hơn
    BASE_LR = 5e-6  # Giảm LR backbone xuống để học chậm hơn
    HEAD_LR = 2e-4  # Giảm LR head xuống để tránh overfit
    WEIGHT_DECAY = 5e-2  # Tăng weight decay để regularization mạnh hơn
    NUM_WORKERS = 4
    SEED = 42
    
    # Tạo output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {OUTPUT_DIR}")
    
    # Setup seed
    setup_seed(SEED)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # ==================== LOAD DATA ====================
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    train_loader, valid_loader, test_loader = get_dataloaders(
        data_dir=DATASET_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    # Lấy class names từ dataset
    class_names = train_loader.dataset.classes
    num_classes = len(class_names)
    print(f"[INFO] Số classes: {num_classes}")
    print(f"[INFO] Class names: {class_names}")
    
    # ==================== LOAD MODEL ====================
    print("\n" + "="*60)
    print("LOADING PRETRAINED MODEL")
    print("="*60)
    
    model = load_pretrained_model(
        checkpoint_path=PRETRAINED_MODEL_PATH,
        num_classes=num_classes,
        device=device
    )
    
    # Đếm số parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    
    print(f"[INFO] Total parameters: {num_params:,}")
    print(f"[INFO] Trainable parameters: {num_trainable:,}")
    print(f"[INFO] Model size: {model_size_mb:.2f} MB")
    
    # ==================== SETUP TRAINING ====================
    print("\n" + "="*60)
    print("SETUP TRAINING")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    # Tạo param groups: backbone với LR thấp, head với LR cao hơn
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name:  # Classifier head
            head_params.append(param)
        else:  # Backbone
            backbone_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BASE_LR, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY}
    ])
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6
    )
    
    print(f"[INFO] Optimizer: AdamW")
    print(f"[INFO] Base LR (backbone): {BASE_LR}")
    print(f"[INFO] Head LR: {HEAD_LR}")
    print(f"[INFO] Weight decay: {WEIGHT_DECAY}")
    print(f"[INFO] Scheduler: CosineAnnealingLR")
    
    # ==================== TRAINING LOOP ====================
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
        "valid_f1": []
    }
    
    best_f1 = -1.0
    best_epoch = -1
    no_improve_count = 0
    best_model_path = OUTPUT_DIR / "MobileNetV3_Small_BoT_finetuned_best.pt"
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{NUM_EPOCHS}]")
        
        # Train
        train_loss, train_acc = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device
        )
        
        # Validate
        valid_metrics = evaluate_model(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            class_names=class_names
        )
        
        valid_loss = valid_metrics["loss"]
        valid_acc = valid_metrics["accuracy"]
        valid_f1 = valid_metrics["f1_macro"]
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)
        history["valid_f1"].append(valid_f1)
        
        # Print metrics
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc*100:.2f}%")
        print(f"  Valid F1:   {valid_f1:.4f}")
        
        # Check for best model
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            best_epoch = epoch
            no_improve_count = 0
            
            # Save best model
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "valid_f1": valid_f1,
                "valid_acc": valid_acc,
                "optimizer": optimizer.state_dict()
            }, best_model_path)
            
            print(f"  ✓ Best model saved! (F1: {best_f1:.4f})")
        else:
            no_improve_count += 1
            print(f"  No improvement ({no_improve_count}/{PATIENCE})")
        
        # Early stopping
        if no_improve_count >= PATIENCE:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}")
            print(f"[INFO] Best F1: {best_f1:.4f} at epoch {best_epoch}")
            break
    
    # ==================== LOAD BEST MODEL ====================
    print("\n" + "="*60)
    print("LOADING BEST MODEL FOR FINAL EVALUATION")
    print("="*60)
    
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"[INFO] Loaded best model from epoch {checkpoint['epoch']}")
    
    # ==================== FINAL EVALUATION ====================
    print("\n" + "="*60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*60)
    
    valid_metrics = evaluate_model(
        model=model,
        loader=valid_loader,
        criterion=criterion,
        device=device,
        class_names=class_names
    )
    
    print("\n[VALIDATION METRICS]")
    print(f"  Accuracy: {valid_metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score: {valid_metrics['f1_macro']:.4f}")
    print(f"  Recall:   {valid_metrics['recall_macro']:.4f}")
    
    print("\n[PER-CLASS METRICS (Validation)]")
    for class_name in class_names:
        class_metrics = valid_metrics["report"][class_name]
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1-score:  {class_metrics['f1-score']:.4f}")
    
    # ==================== TEST SET EVALUATION ====================
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)
    
    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names
    )
    
    print("\n[TEST METRICS]")
    print(f"  Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score: {test_metrics['f1_macro']:.4f}")
    print(f"  Recall:   {test_metrics['recall_macro']:.4f}")
    
    print("\n[PER-CLASS METRICS (Test)]")
    for class_name in class_names:
        class_metrics = test_metrics["report"][class_name]
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1-score:  {class_metrics['f1-score']:.4f}")
    
    # ==================== CALCULATE FPS ====================
    print("\n" + "="*60)
    print("CALCULATING FPS")
    print("="*60)
    
    fps_value = calculate_fps(model, device, IMAGE_SIZE)
    print(f"[INFO] FPS: {fps_value:.2f}")
    
    # ==================== SAVE RESULTS ====================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Tính tổng thời gian chạy
    total_time = time.time() - start_time
    print(f"[INFO] Tổng thời gian chạy: {total_time:.2f} giây")
    
    # Lưu history.json
    history_path = OUTPUT_DIR / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Đã lưu history.json tại: {history_path}")
    
    # Lưu metrics.json
    metrics = {
        "model_name": "MobileNetV3_Small_BoT_finetuned",
        "size_mb": model_size_mb,
        "num_params": num_params,
        "validation": {
            "accuracy": valid_metrics["accuracy"],
            "f1_macro": valid_metrics["f1_macro"],
            "recall_macro": valid_metrics["recall_macro"]
        },
        "test": {
            "accuracy": test_metrics["accuracy"],
            "f1_macro": test_metrics["f1_macro"],
            "recall_macro": test_metrics["recall_macro"]
        },
        "training": {
            "train_acc": history["train_acc"][-1],
        },
        "fps": fps_value,
        "best_epoch": best_epoch,
        "total_epochs": len(history["train_loss"]),
        "total_time_seconds": total_time  # Lưu tổng thời gian chạy
    }
    
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Đã lưu metrics.json tại: {metrics_path}")
    
    # Lưu classification reports với accuracy
    valid_report_with_acc = {
        "accuracy": valid_metrics["accuracy"],
        "f1_macro": valid_metrics["f1_macro"],
        "recall_macro": valid_metrics["recall_macro"],
        "classification_report": valid_metrics["report"]
    }
    valid_report_path = OUTPUT_DIR / "validation_report.json"
    with open(valid_report_path, "w", encoding="utf-8") as f:
        json.dump(valid_report_with_acc, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Đã lưu validation_report.json tại: {valid_report_path}")
    
    test_report_with_acc = {
        "accuracy": test_metrics["accuracy"],
        "f1_macro": test_metrics["f1_macro"],
        "recall_macro": test_metrics["recall_macro"],
        "classification_report": test_metrics["report"]
    }
    test_report_path = OUTPUT_DIR / "test_report.json"
    with open(test_report_path, "w", encoding="utf-8") as f:
        json.dump(test_report_with_acc, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Đã lưu test_report.json tại: {test_report_path}")
    
    plot_path = OUTPUT_DIR / "training_plot.png"
    save_training_plot(history, plot_path)
    
    # Lưu final model 
    final_model_path = OUTPUT_DIR / "MobileNetV3_Small_BoT_finetuned_final.pt"
    torch.save({
        "model": model.state_dict(),
        "metrics": metrics,
        "class_names": class_names
    }, final_model_path)
    print(f"[INFO] Đã lưu final model tại: {final_model_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"[INFO] Tất cả kết quả đã được lưu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
