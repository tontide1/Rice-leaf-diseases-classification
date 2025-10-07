# Hướng Dẫn Training MobileNetV3-Small CA (Coordinate Attention)

## Giới Thiệu

File `train_MobileNetV3_Small_CA.py` được sử dụng để huấn luyện mô hình MobileNetV3-Small với Coordinate Attention (CA) block trên tập dữ liệu Paddy Disease Classification.

**Đặc điểm Coordinate Attention:**

- Kết hợp cả channel attention và spatial attention
- Encode thông tin vị trí (coordinate) vào attention map
- Hiệu quả với các tác vụ cần nhận diện vị trí chính xác (như disease localization)
- Complexity: O(HW) - hiệu quả hơn standard attention O(H²W²)

## Yêu Cầu

### 1. Cấu Trúc Thư Mục

Đảm bảo cấu trúc thư mục như sau:

```
Paddy-Disease-Classification-final/
├── data/
│   ├── metadata.csv
│   ├── label2id.json
│   └── images/
├── train_MobileNetV3_Small_CA.py
├── src/
│   ├── models/
│   │   ├── backbones/
│   │   │   └── mobilenet.py
│   │   └── attention/
│   │       └── ca.py
│   ├── training/
│   │   ├── train.py
│   │   └── param_groups.py
│   └── utils/
│       ├── data/
│       └── metrics/
└── results/
```

### 2. Dữ Liệu Cần Thiết

- `data/metadata.csv`: File chứa thông tin ảnh và nhãn
- `data/label2id.json`: File mapping từ tên nhãn sang ID
- `data/images/`: Thư mục chứa ảnh training/validation

## Cách Sử Dụng

### 1. Training Cơ Bản

```bash
python train_MobileNetV3_Small_CA.py
```

### 2. Training với Tham Số Tùy Chỉnh

```bash
python train_MobileNetV3_Small_CA.py \
    --epochs 30 \
    --batch-size 64 \
    --base-lr 5e-5 \
    --head-lr 5e-4 \
    --reduction 16 \
    --pretrained
```

### 3. Quick Test (Training Nhanh với Dữ Liệu Giới Hạn)

```bash
python train_MobileNetV3_Small_CA.py \
    --epochs 2 \
    --batch-size 16 \
    --train-limit 100 \
    --valid-limit 50 \
    --pretrained
```

## Tham Số Dòng Lệnh

### Dữ Liệu & Đường Dẫn

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--metadata` | `data/metadata.csv` | Đường dẫn đến file metadata |
| `--label2id` | `data/label2id.json` | Đường dẫn đến file label mapping |

### Cấu Hình Model

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--image-size` | 224 | Kích thước ảnh đầu vào |
| `--reduction` | 16 | Reduction ratio cho CA block (thường 8, 16, hoặc 32) |
| `--dropout` | 0.1 | Dropout rate trước classifier |
| `--pretrained` | False | Sử dụng pretrained weights từ ImageNet |

### Training Configuration

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--epochs` | 30 | Số epoch training |
| `--batch-size` | 32 | Batch size |
| `--patience` | 10 | Early stopping patience (<=0 để tắt) |
| `--base-lr` | 5e-5 | Learning rate cho backbone |
| `--head-lr` | 5e-4 | Learning rate cho classifier head |
| `--weight-decay` | 1e-2 | Weight decay cho optimizer |
| `--scheduler-tmax` | None | T_max cho CosineAnnealingLR |

### DataLoader Configuration

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--num-workers` | 4 | Số workers cho DataLoader |
| `--pin-memory` | False | Pin memory trong DataLoader |

### Debug & Testing

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--train-limit` | None | Giới hạn số mẫu training (cho test nhanh) |
| `--valid-limit` | None | Giới hạn số mẫu validation (cho test nhanh) |
| `--model-name` | MobileNetV3_Small_CA | Tên model cho logging/checkpoint |
| `--device` | auto | Device (cpu/cuda) |
| `--seed` | 42 | Random seed |

### Visualization & Logging

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--plot` | False | Hiển thị biểu đồ training history sau khi train |
| `--save-history` | False | Tự động lưu: history JSON + metrics JSON + biểu đồ PNG (DPI 300) |

## Ví Dụ Thực Tế

### 1. Training Full Dataset với Pretrained Weights (Khuyến Nghị)

```bash
python train_MobileNetV3_Small_CA.py \
    --epochs 30 \
    --batch-size 64 \
    --base-lr 5e-5 \
    --head-lr 5e-4 \
    --weight-decay 1e-2 \
    --patience 10 \
    --reduction 16 \
    --pretrained \
    --num-workers 4 \
    --pin-memory \
    --image-size 224 \
    --save-history
```

**Kết quả mong đợi:**

- Validation Accuracy: ~99.0-99.3%
- Training time: ~35-45 phút (GPU GTX 1660 SUPER)
- Model size: ~7.32 MB
- Inference: ~8.81 ms/batch (batch=16)

### 2. Training với Reduction Khác Nhau

#### Reduction = 8 (Nhiều params hơn, chính xác hơn)

```bash
python train_MobileNetV3_Small_CA.py \
    --epochs 30 \
    --reduction 8 \
    --pretrained \
    --save-history
```

#### Reduction = 32 (Ít params hơn, nhanh hơn)

```bash
python train_MobileNetV3_Small_CA.py \
    --epochs 30 \
    --reduction 32 \
    --pretrained \
    --save-history
```

### 3. Training trên GPU với 2 Models Đồng Thời

```bash
# Terminal 1
python train_MobileNetV3_Small_CA.py \
    --epochs 30 \
    --batch-size 64 \
    --num-workers 4 \
    --pretrained \
    --image-size 224 \
    --save-history

# Terminal 2 (có thể train model khác)
python train_MobileNetV3_Small_BoT.py \
    --epochs 30 \
    --batch-size 64 \
    --num-workers 4 \
    --pretrained \
    --image-size 224 \
    --save-history
```

### 4. Training từ Scratch

```bash
python train_MobileNetV3_Small_CA.py \
    --epochs 50 \
    --batch-size 32 \
    --base-lr 1e-3 \
    --head-lr 1e-2 \
    --weight-decay 1e-2 \
    --patience 15 \
    --reduction 16
```

### 5. Fine-tuning với Learning Rate Thấp

```bash
python train_MobileNetV3_Small_CA.py \
    --epochs 20 \
    --batch-size 32 \
    --base-lr 1e-5 \
    --head-lr 1e-4 \
    --pretrained \
    --patience 8
```

### 6. Training với Auto-Save và Display

```bash
python train_MobileNetV3_Small_CA.py \
    --epochs 30 \
    --batch-size 64 \
    --pretrained \
    --save-history \
    --plot
```

## Đầu Ra

### 1. Model Checkpoint

Checkpoint được lưu trong thư mục gốc:

- `MobileNetV3_Small_CA_best.pt` - Best model (lowest validation loss)

### 2. Results Files (với `--save-history`)

Tự động lưu trong thư mục `results/MobileNetV3_Small_CA_{timestamp}/`:

```
results/
└── MobileNetV3_Small_CA_06_10_2025_1430/
    ├── history.json        # Training curves data
    ├── metrics.json        # Final performance metrics
    └── training_plot.png   # Visualization (DPI 300)
```

#### `history.json` format

```json
{
  "train_loss": [0.523, 0.234, 0.156, ...],
  "valid_loss": [0.489, 0.298, 0.201, ...],
  "train_acc": [0.854, 0.921, 0.945, ...],
  "valid_acc": [0.867, 0.912, 0.933, ...]
}
```

#### `metrics.json` format

```json
{
  "model_name": "MobileNetV3_Small_CA",
  "size_mb": 7.32,
  "valid_acc": 0.9920,
  "valid_f1": 0.9920,
  "fps": 3400.50,
  "num_params": 1918834,
  "ckpt_path": "MobileNetV3_Small_CA_best.pt"
}
```

### 3. Expected Results

Với cấu hình tối ưu (pretrained, 30 epochs, batch_size=64, reduction=16):

- **Validation Accuracy**: ~99.0-99.3%
- **F1-Score**: ~99.0-99.3%
- **Inference Speed**: ~3,400 FPS (batch=16)
- **Model Size**: ~7.32 MB
- **Parameters**: ~1.92M
- **Training Time**: ~35-45 phút (GTX 1660 SUPER)

## So Sánh với Các Variants Khác

| Model | Params | Size (MB) | Inference (ms) | Accuracy | Đặc điểm |
|-------|--------|-----------|----------------|----------|----------|
| **MobileNetV3_Small_CA** | **1.92M** | **7.32** | **8.81** | **~99.2%** | **Channel + Spatial, coordinate encoding** |
| MobileNetV3_Small_BoT | 1.75M | 6.69 | 6.13 | ~99.3% | Self-attention, O(N²) |
| MobileNetV3_Small_BoT_Linear | 1.75M | 6.69 | 6.98 | ~99.2% | Linear attention, O(N) |
| MobileNetV3_Small_ECA | 1.86M | 7.08 | 6.37 | ~99.1% | Efficient channel attention |
| MobileNetV3_Small_Hybrid | 2.15M | 8.19 | 26.01 | ~99.4% | CA + BoT combined |

**Khi nào chọn CA:**

- ✅ Khi cần attention mechanism nhẹ và hiệu quả
- ✅ Khi tác vụ yêu cầu spatial awareness (vị trí chính xác)
- ✅ Khi muốn balance giữa accuracy và speed
- ✅ Khi có constraint về memory nhưng cần accuracy cao

## Troubleshooting

### Lỗi: "FileNotFoundError: metadata file not found"

**Giải pháp:**

```bash
# Chỉ định đúng đường dẫn
python train_MobileNetV3_Small_CA.py \
    --metadata path/to/your/metadata.csv \
    --label2id path/to/your/label2id.json
```

### Lỗi: "ModuleNotFoundError: No module named 'src'"

**Giải pháp:**

- Đảm bảo chạy từ thư mục gốc của project
- Kiểm tra cấu trúc thư mục `src/` tồn tại

### Out of Memory (OOM)

**Giải pháp:**

```bash
# Giảm batch size
python train_MobileNetV3_Small_CA.py --batch-size 32

# Hoặc tăng reduction (giảm params)
python train_MobileNetV3_Small_CA.py --reduction 32

# Hoặc giảm image size
python train_MobileNetV3_Small_CA.py --image-size 192
```

### Training Quá Chậm

**Giải pháp:**

```bash
# Tăng số workers
python train_MobileNetV3_Small_CA.py \
    --num-workers 8 \
    --pin-memory

# Hoặc test với subset nhỏ
python train_MobileNetV3_Small_CA.py \
    --train-limit 1000 \
    --valid-limit 200
```

### Accuracy Không Cải Thiện

**Giải pháp:**

```bash
# Thử reduction khác
python train_MobileNetV3_Small_CA.py --reduction 8

# Hoặc tăng learning rate
python train_MobileNetV3_Small_CA.py \
    --base-lr 1e-4 \
    --head-lr 1e-3

# Hoặc tăng dropout
python train_MobileNetV3_Small_CA.py --dropout 0.2
```

## Tips & Best Practices

### 1. Chọn Reduction Ratio

- **reduction=8**: Nhiều parameters, accuracy cao hơn, chậm hơn
- **reduction=16**: Balance tốt (khuyến nghị) ⭐
- **reduction=32**: Ít parameters, nhanh hơn, accuracy thấp hơn chút

### 2. Chọn Learning Rate

- **Pretrained model**: `base_lr=5e-5`, `head_lr=5e-4` (khuyến nghị)
- **From scratch**: `base_lr=1e-3`, `head_lr=1e-2`

### 3. Batch Size & Num Workers

**Với cấu hình 12 cores, 12GB RAM, GTX 1660 SUPER:**

- **Training 1 model**: `batch_size=64`, `num_workers=4-6`
- **Training 2 models**: `batch_size=64`, `num_workers=4`
- **Training 3 models**: `batch_size=48`, `num_workers=3`

### 4. Early Stopping

- Sử dụng `--patience 10` để tránh overfitting
- Tắt early stopping với `--patience 0` nếu muốn train đủ epochs

### 5. Monitoring

Theo dõi các chỉ số trong quá trình training:

```bash
# Monitor GPU
watch -n 2 'nvidia-smi'

# Monitor logs
tail -f logs/ca_train.log

# Check progress
grep "Epoch" logs/ca_train.log | tail -10
```

## Kiến Trúc Model

### MobileNetV3-Small CA

```
Input (3x224x224)
    ↓
[MobileNetV3-Small Backbone] (pretrained on ImageNet)
    ↓
Feature Maps (576 channels)
    ↓
[Coordinate Attention Block]
    ├─ X-Avg Pool → Conv → BN → ReLU
    ├─ Y-Avg Pool → Conv → BN → ReLU
    └─ Concat → Conv → Sigmoid → Split
       ├─ Channel attention weights
       └─ Spatial attention weights
    ↓
[AdaptiveAvgPool2d]
    ↓
[Dropout 0.1]
    ↓
[Linear Classifier] → Output (num_classes)
```

**Đặc điểm:**

- Backbone: MobileNetV3-Small (efficient, lightweight)
- Attention: Coordinate Attention (channel + spatial with position encoding)
- Parameters: ~1.92M
- Size: ~7.32 MB
- Optimal for: Mobile deployment, tasks requiring spatial awareness

## Coordinate Attention vs Other Attentions

### Ưu điểm của CA

- ✅ Encode cả channel và spatial information
- ✅ Nhẹ hơn standard self-attention (O(HW) vs O(H²W²))
- ✅ Hiệu quả với position-sensitive tasks
- ✅ Dễ integrate vào các CNN architectures

### Nhược điểm

- ⚠️ Chậm hơn một chút so với ECA (~8.81ms vs 6.37ms)
- ⚠️ Nhiều params hơn simple channel attention
- ⚠️ Không mạnh bằng full self-attention cho global context

## Tham Khảo

- Model Implementation: `src/models/backbones/mobilenet.py` (class `MobileNetV3_Small_CA`)
- CA Block: `src/models/attention/ca.py`
- Training Logic: `src/training/train.py`
- Data Loading: `src/utils/data/loading.py`
- Paper: "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021)

## License

MIT License
