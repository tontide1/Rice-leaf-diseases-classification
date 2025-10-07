"""
Script test batch tất cả các models trên nhiều external datasets.

Sử dụng:
    python batch_test_external.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import pandas as pd
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
    MobileNetV3_Small_Hybrid,
    MobileNetV3_Small_Vanilla,
    ResNet18_BoT,
    ResNet18_BoTLinear,
    ResNet18_Hybrid,
    EfficientNetV2_S_CA,
    EfficientNet_Lite0_CA,
    CompactCNN,
)




# Các model cần test
MODEL_CONFIGS = [
    {
        'name': 'EfficientNet_Lite0_CA',
        'path': 'models/train 2/EfficientNet_Lite0_CA_07_10_2025_0048_best.pt',
        'class': EfficientNet_Lite0_CA,
        'params': {'num_classes': 4, 'reduction': 16}
    },
    {
        'name': 'EfficientNetV2_S_CA',
        'path': 'models/train 2/EfficientNetV2_S_CA_best.pt',
        'class': EfficientNetV2_S_CA,
        'params': {'num_classes': 4, 'reduction': 32}  # Fixed: reduction=32 (1280/40)
    },
    {
        'name': 'ResNet18_BoT',
        'path': 'models/train 2/ResNet18_BoT_large_batch_best.pt',
        'class': ResNet18_BoT,
        'params': {'num_classes': 4, 'heads': 4}
    },
    {
        'name': 'ResNet18_BoTLinear',
        'path': 'models/train 2/ResNet18_BoTLinear_pretrained_best.pt',
        'class': ResNet18_BoTLinear,
        'params': {'num_classes': 4, 'heads': 4, 'bottleneck_dim': 120}  # Fixed: bottleneck_dim=120
    },
    {
        'name': 'ResNet18_Hybrid',
        'path': 'models/train 2/ResNet18_Hybrid_best.pt',
        'class': ResNet18_Hybrid,
        'params': {'num_classes': 4, 'heads': 4, 'reduction': 32, 'bottleneck_dim': 128}  # Fixed: reduction=32, bottleneck_dim=128
    },
    {
        'name': 'CompactCNN',
        'path': 'models/train 2/CNN_base_07_10_2025_1048_best.pt',
        'class': CompactCNN,
        'params': {'num_classes': 4, 'dropout': 0.2}
    },
    {
        'name': 'MobileNetV3_Small_BoT',
        'path': 'models/train 2/MobileNetV3_Small_BoT_best.pt',
        'class': MobileNetV3_Small_BoT,
        'params': {'num_classes': 4, 'heads': 4}
    },
    {
        'name': 'MobileNetV3_Small_Hybrid',
        'path': 'models/train 2/MobileNetV3_Small_Hybrid_best.pt',
        'class': MobileNetV3_Small_Hybrid,
        'params': {'num_classes': 4, 'heads': 4, 'reduction': 16}
    },
    {
        'name': 'MobileNetV3_Small_Vanilla',
        'path': 'models/train 2/MobileNetV3_Small_Vanilla_best.pt',
        'class': MobileNetV3_Small_Vanilla,
        'params': {'num_classes': 4}
    },
]

# Các dataset cần test
TEST_DATASETS = [
    {
        'name': 'KG_DMH',
        'path': 'test/KG_DMH/test',
        'classes': ['brown_spot', 'healthy', 'leaf_blast']
    },
    {
        'name': 'KG_bahribahri',
        'path': 'test/KG_bahribahri_test/test',
        'classes': ['bacterial_leaf_blight', 'brown_spot', 'healthy']
    },
    {
        'name': 'KG_iashiqul',
        'path': 'test/KG_iashiqul/test',
        'classes': ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast']
    },
    {
        'name': 'Loki',
        'path': 'test/Loki/test',
        'classes': ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast']
    },
    {
        'name': 'UCI',
        'path': 'test/UCI/test',
        'classes': ['bacterial_leaf_blight', 'brown_spot']
    },
    {
        'name': 'mendeley',
        'path': 'test/mendeley/test',
        'classes': ['bacterial_leaf_blight', 'brown_spot', 'leaf_blast']
    },
    {
        'name': 'paddy_disease_v1',
        'path': 'test/paddy-disease-classification_1/test',
        'classes': ['bacterial_leaf_blight', 'blast', 'brown_spot', 'healthy']
    },
    {
        'name': 'vbookshelf',
        'path': 'test/vbookshelf/test',
        'classes': ['bacterial_leaf_blight', 'brown_spot']
    },
]

# Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 4
IMAGE_SIZE = 224

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory
OUTPUT_DIR = Path('test_results/batch_external')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATASET CLASS
# ============================================================================

class ExternalTestDataset(torch.utils.data.Dataset):
    """Dataset cho test từ thư mục có cấu trúc class folders"""
    
    def __init__(self, root_dir, transform=None, label2id=None):
        """
        Args:
            root_dir: Thư mục chứa các class folders
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
                
                # Map "blast" -> "leaf_blast" nếu cần
                if class_name == 'blast':
                    class_name = 'leaf_blast'
                
                self.class_names.append(class_name)
                
                # Lấy class ID từ label2id
                class_id = self.label2id.get(class_name, -1)
                
                # Scan tất cả ảnh trong class folder
                for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
                    for img_path in class_dir.glob(ext):
                        self.samples.append((img_path, class_id, class_name))
    
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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_label_mappings(label2id_path='data/label2id.json', 
                        id2label_path='data/id2label.json'):
    """Load label mappings từ JSON files"""
    with open(label2id_path, 'r') as f:
        label2id = json.load(f)
    
    with open(id2label_path, 'r') as f:
        id2label_raw = json.load(f)
    
    # Convert keys thành integers
    id2label = {int(k): v for k, v in id2label_raw.items()}
    
    return label2id, id2label


def get_test_transforms(image_size=224):
    """Lấy transforms cho test"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def load_model(model_config, device):
    """
    Load model từ checkpoint.
    
    Args:
        model_config: Dict chứa thông tin model
        device: Device để load model
    
    Returns:
        model: Model đã load weights
    """
    # Khởi tạo model architecture
    model = model_config['class'](**model_config['params'])
    
    # Load checkpoint
    checkpoint = torch.load(model_config['path'], map_location=device, weights_only=False)
    
    # Load state dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
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
    
    for images, labels in test_loader:
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
    unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
    target_names = [id2label.get(int(i), f"class_{i}") for i in unique_labels]
    
    report = classification_report(
        all_labels, all_preds,
        labels=unique_labels,
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
        'probabilities': all_probs,
        'unique_labels': unique_labels
    }
    
    return results


def plot_confusion_matrix(cm, class_names, save_path):
    """Vẽ và lưu confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
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
    
    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.ylabel('True Label', fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results_json(results, save_path):
    """Lưu kết quả thành JSON file"""
    # Convert numpy arrays thành lists
    serializable_results = {
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score']),
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': results['classification_report'],
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)


# ============================================================================
# MAIN TESTING PIPELINE
# ============================================================================

def main():
    print("="*80)
    print("BATCH TESTING EXTERNAL DATASETS")
    print("="*80)
    print(f"\n📱 Device: {DEVICE}")
    print(f"📊 Số lượng models: {len(MODEL_CONFIGS)}")
    print(f"📂 Số lượng datasets: {len(TEST_DATASETS)}")
    print(f"🔢 Batch size: {BATCH_SIZE}")
    print(f"📏 Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    # Load label mappings
    print("\n📋 Đang load label mappings...")
    label2id, id2label = load_label_mappings()
    print(f"   ✅ Label mappings: {label2id}")
    
    # Get transforms
    test_transforms = get_test_transforms(IMAGE_SIZE)
    
    # Storage cho tất cả results
    all_results = []
    
    # Timestamp cho session này
    timestamp = datetime.now().strftime("%d_%m_%Y_%H%M")
    session_dir = OUTPUT_DIR / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Kết quả sẽ được lưu vào: {session_dir}")
    
    # Loop qua tất cả models
    for model_idx, model_config in enumerate(MODEL_CONFIGS, 1):
        model_name = model_config['name']
        
        print("\n" + "="*80)
        print(f"[{model_idx}/{len(MODEL_CONFIGS)}] MODEL: {model_name}")
        print("="*80)
        
        # Load model
        print(f"\n📥 Đang load model từ: {model_config['path']}")
        try:
            model = load_model(model_config, DEVICE)
            print(f"   ✅ Đã load model thành công!")
        except Exception as e:
            print(f"   ❌ Lỗi khi load model: {e}")
            continue
        
        # Tạo thư mục cho model này
        model_dir = session_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Loop qua tất cả datasets
        for dataset_idx, dataset_config in enumerate(TEST_DATASETS, 1):
            dataset_name = dataset_config['name']
            dataset_path = dataset_config['path']
            
            print(f"\n{'─'*80}")
            print(f"  [{dataset_idx}/{len(TEST_DATASETS)}] Dataset: {dataset_name}")
            print(f"{'─'*80}")
            
            # Kiểm tra dataset có tồn tại không
            if not Path(dataset_path).exists():
                print(f"     ⚠️  Dataset không tồn tại: {dataset_path}")
                continue
            
            # Load dataset
            print(f"     📂 Đang load dataset từ: {dataset_path}")
            try:
                dataset = ExternalTestDataset(
                    root_dir=dataset_path,
                    transform=test_transforms,
                    label2id=label2id
                )
                
                if len(dataset) == 0:
                    print(f"     ⚠️  Dataset rỗng!")
                    continue
                
                print(f"     ✅ Đã load {len(dataset)} ảnh")
                
                # Tạo DataLoader
                test_loader = DataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=True
                )
                
            except Exception as e:
                print(f"     ❌ Lỗi khi load dataset: {e}")
                continue
            
            # Test model
            print(f"     🔬 Đang test model...")
            try:
                results = test_model(model, test_loader, DEVICE, id2label)
                
                # In kết quả
                print(f"     📊 Accuracy:  {results['accuracy']*100:.2f}%")
                print(f"     📊 Precision: {results['precision']*100:.2f}%")
                print(f"     📊 Recall:    {results['recall']*100:.2f}%")
                print(f"     📊 F1-Score:  {results['f1_score']*100:.2f}%")
                
                # Lưu confusion matrix
                cm_save_path = model_dir / f"cm_{dataset_name}.png"
                class_names = [id2label[int(i)] for i in results['unique_labels']]
                plot_confusion_matrix(
                    results['confusion_matrix'],
                    class_names,
                    cm_save_path
                )
                print(f"     💾 Đã lưu confusion matrix: {cm_save_path.name}")
                
                # Lưu kết quả JSON
                json_save_path = model_dir / f"results_{dataset_name}.json"
                save_results_json(results, json_save_path)
                print(f"     💾 Đã lưu metrics: {json_save_path.name}")
                
                # Thêm vào all_results
                all_results.append({
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score'],
                    'num_samples': len(dataset),
                })
                
            except Exception as e:
                print(f"     ❌ Lỗi khi test: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Giải phóng memory
        del model
        torch.cuda.empty_cache()
    
    # ========================================================================
    # TỔNG HỢP KẾT QUẢ
    # ========================================================================
    
    print("\n" + "="*80)
    print("TỔNG HỢP KẾT QUẢ")
    print("="*80)
    
    if len(all_results) == 0:
        print("❌ Không có kết quả nào được tạo!")
        return
    
    # Tạo DataFrame
    df = pd.DataFrame(all_results)
    
    # Lưu thành CSV
    csv_path = session_dir / "summary_all_results.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n💾 Đã lưu tổng hợp kết quả: {csv_path}")
    
    # Tạo pivot table cho từng metric
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for metric in metrics:
        pivot = df.pivot(index='model_name', columns='dataset_name', values=metric)
        pivot_path = session_dir / f"summary_{metric}.csv"
        pivot.to_csv(pivot_path, encoding='utf-8')
        print(f"💾 Đã lưu bảng {metric}: {pivot_path.name}")
    
    # In bảng tổng hợp accuracy
    print("\n📊 BẢNG TỔNG HỢP ACCURACY (%):")
    print("-"*80)
    pivot_acc = df.pivot(index='model_name', columns='dataset_name', values='accuracy') * 100
    print(pivot_acc.to_string(float_format=lambda x: f"{x:.2f}"))
    
    # In model tốt nhất cho từng dataset
    print("\n🏆 MODEL TỐT NHẤT CHO TỪNG DATASET:")
    print("-"*80)
    for dataset_name in df['dataset_name'].unique():
        dataset_df = df[df['dataset_name'] == dataset_name]
        best_row = dataset_df.loc[dataset_df['accuracy'].idxmax()]
        print(f"  • {dataset_name:<25} → {best_row['model_name']:<30} "
              f"(Acc: {best_row['accuracy']*100:.2f}%)")
    
    # In dataset tốt nhất cho từng model
    print("\n🏆 DATASET TỐT NHẤT CHO TỪNG MODEL:")
    print("-"*80)
    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name]
        best_row = model_df.loc[model_df['accuracy'].idxmax()]
        print(f"  • {model_name:<30} → {best_row['dataset_name']:<25} "
              f"(Acc: {best_row['accuracy']*100:.2f}%)")
    
    print("\n" + "="*80)
    print("✅ HOÀN THÀNH!")
    print("="*80)
    print(f"\n📂 Tất cả kết quả đã được lưu vào: {session_dir}")


if __name__ == '__main__':
    main()
