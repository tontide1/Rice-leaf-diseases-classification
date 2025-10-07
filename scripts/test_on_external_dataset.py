"""
Script test model đã train trên tập dữ liệu external (KG_sudheshkm).

Hướng dẫn sử dụng:
    python test_on_external_dataset.py \
        --model-path "models/train 1/MobileNetV3_Small_BoT_best.pt" \
        --test-dir "KG_sudheshkm" \
        --model-type "BoT" \
        --batch-size 32
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Import models
import sys
sys.path.append('.')
from src.models.backbones import (
    MobileNetV3_Small_BoT,
    MobileNetV3_Small_BoT_Linear,
    MobileNetV3_Small_CA,
    MobileNetV3_Small_Hybrid,
    MobileNetV3_Small_Vanilla
)


class ExternalTestDataset(torch.utils.data.Dataset):
    """Dataset cho test từ thư mục có cấu trúc class folders"""
    
    def __init__(self, root_dir, transform=None, label2id=None):
        """
        Args:
            root_dir: Thư mục chứa các class folders (e.g., KG_sudheshkm/)
            transform: Transforms cho ảnh
            label2id: Dict mapping từ class name sang class ID
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.label2id = label2id or {}
        
        # Scan tất cả ảnh trong các class folders
        self.samples = []
        self.class_names = []
        
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                self.class_names.append(class_name)
                
                # Lấy class ID từ label2id
                class_id = self.label2id.get(class_name, len(self.label2id))
                
                # Scan tất cả ảnh trong class folder
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, class_id, class_name))
                for img_path in class_dir.glob("*.JPG"):
                    self.samples.append((img_path, class_id, class_name))
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((img_path, class_id, class_name))
        
        print(f"✅ Đã load {len(self.samples)} ảnh từ {len(self.class_names)} classes")
        for i, cls in enumerate(self.class_names):
            count = sum(1 for _, cid, _ in self.samples if cid == self.label2id.get(cls, i))
            print(f"   - {cls}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_id, class_name = self.samples[idx]
        
        # Load ảnh
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, class_id


def load_model(model_path, model_type, num_classes, device):
    """
    Load model từ checkpoint.
    
    Args:
        model_path: Đường dẫn đến file .pt
        model_type: Loại model ("BoT", "BoT_Linear", "CA", "Hybrid", "Vanilla")
        num_classes: Số classes
        device: Device để load model
    
    Returns:
        model: Model đã load weights
    """
    # Khởi tạo model architecture
    if model_type == "BoT":
        model = MobileNetV3_Small_BoT(num_classes=num_classes, heads=4)
    elif model_type == "BoT_Linear":
        model = MobileNetV3_Small_BoT_Linear(num_classes=num_classes, heads=4)
    elif model_type == "CA":
        model = MobileNetV3_Small_CA(num_classes=num_classes, reduction=16)
    elif model_type == "Hybrid":
        model = MobileNetV3_Small_Hybrid(num_classes=num_classes, heads=4, reduction=16)
    elif model_type == "Vanilla":
        model = MobileNetV3_Small_Vanilla(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    print(f"\n📥 Loading checkpoint từ: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load state dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"   ✅ Loaded state_dict từ checkpoint['model']")
        if 'epoch' in checkpoint:
            print(f"   📍 Checkpoint từ epoch: {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint)
        print(f"   ✅ Loaded state_dict trực tiếp")
    
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def test_model(model, test_loader, device, id2label):
    """
    Test model trên test set và tính metrics.
    
    Args:
        model: Model đã load
        test_loader: DataLoader cho test set
        device: Device
        id2label: Dict mapping từ class ID sang class name
    
    Returns:
        results: Dict chứa metrics và predictions
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n🔬 Đang test model...")
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        # Collect predictions
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Tính metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    # Lấy chỉ các classes xuất hiện trong test set
    unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
    target_names = [id2label.get(int(i), f"class_{i}") for i in unique_labels]
    
    report = classification_report(
        all_labels, all_preds, 
        labels=unique_labels,  # Chỉ định labels để match với target_names
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return results


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Vẽ confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2%', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Đã lưu confusion matrix: {save_path}")
    
    plt.close()


def print_results(results, id2label):
    """In kết quả test"""
    print("\n" + "="*80)
    print("KẾT QUẢ TEST TRÊN TẬP EXTERNAL")
    print("="*80)
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"   • Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"   • Precision: {results['precision']*100:.2f}%")
    print(f"   • Recall:    {results['recall']*100:.2f}%")
    print(f"   • F1-Score:  {results['f1_score']*100:.2f}%")
    
    print(f"\n📋 PER-CLASS METRICS:")
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)
    
    report = results['classification_report']
    
    # In các classes có trong test set (bỏ qua 'accuracy', 'macro avg', 'weighted avg')
    for key in sorted(report.keys()):
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics = report[key]
            print(f"{key:<25} "
                  f"{metrics['precision']*100:>10.2f}%  "
                  f"{metrics['recall']*100:>10.2f}%  "
                  f"{metrics['f1-score']*100:>10.2f}%  "
                  f"{metrics['support']:>10}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Test model trên external dataset")
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Đường dẫn đến model checkpoint (.pt file)')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['BoT', 'BoT_Linear', 'CA', 'Hybrid', 'Vanilla'],
                        help='Loại model architecture')
    
    # Data arguments
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Thư mục chứa test data (có structure: class folders)')
    parser.add_argument('--label2id-path', type=str, default='data/label2id.json',
                        help='Đường dẫn đến label2id.json')
    parser.add_argument('--id2label-path', type=str, default='data/id2label.json',
                        help='Đường dẫn đến id2label.json')
    
    # Test arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size cho testing')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Số workers cho DataLoader')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Kích thước ảnh input')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='Thư mục lưu kết quả test')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Lưu predictions vào file')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load label mappings
    print(f"\n📂 Loading label mappings...")
    with open(args.label2id_path, 'r') as f:
        label2id = json.load(f)
    with open(args.id2label_path, 'r') as f:
        id2label = json.load(f)
        # Convert keys to int
        id2label = {int(k): v for k, v in id2label.items()}
    
    num_classes = len(label2id)
    print(f"   ✅ Số classes: {num_classes}")
    print(f"   ✅ Classes: {list(label2id.keys())}")
    
    # Load model
    model = load_model(args.model_path, args.model_type, num_classes, device)
    
    # Setup transforms (giống training)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create dataset và dataloader
    print(f"\n📁 Loading test dataset từ: {args.test_dir}")
    test_dataset = ExternalTestDataset(
        root_dir=args.test_dir,
        transform=transform,
        label2id=label2id
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Test model
    results = test_model(model, test_loader, device, id2label)
    
    # Print results
    print_results(results, id2label)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot confusion matrix
    # Lấy class names cho các classes thực sự xuất hiện trong test set
    unique_labels = np.unique(np.concatenate([results['labels'], results['predictions']]))
    class_names = [id2label.get(int(i), f"class_{i}") for i in unique_labels]
    cm_path = output_dir / f"confusion_matrix_{args.model_type}.png"
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)
    
    # Save metrics to JSON
    metrics = {
        'model_path': args.model_path,
        'model_type': args.model_type,
        'test_dir': args.test_dir,
        'num_samples': len(test_dataset),
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score']),
        'classification_report': results['classification_report']
    }
    
    metrics_path = output_dir / f"test_metrics_{args.model_type}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Đã lưu metrics: {metrics_path}")
    
    # Save predictions (optional)
    if args.save_predictions:
        predictions = {
            'predictions': results['predictions'].tolist(),
            'labels': results['labels'].tolist(),
            'probabilities': results['probabilities'].tolist()
        }
        pred_path = output_dir / f"predictions_{args.model_type}.json"
        with open(pred_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"✅ Đã lưu predictions: {pred_path}")
    
    print(f"\n✅ HOÀN TẤT! Kết quả đã được lưu trong: {output_dir}")


if __name__ == "__main__":
    main()
