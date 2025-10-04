# Hướng Dẫn Training MobileNetV3-Small Hybrid

## Giới Thiệu

File `train_MobileNetV3_Small_Hybrid.py` được sử dụng để huấn luyện mô hình MobileNetV3-Small Hybrid với kết hợp **Coordinate Attention (CA)** và **BoT (Bottleneck Transformer)** block trên tập dữ liệu Paddy Disease Classification.

### Điểm Khác Biệt với BoT Model

| Đặc điểm | MobileNetV3_Small_BoT | MobileNetV3_Small_Hybrid |
|----------|----------------------|--------------------------|
| **Attention Blocks** | BoT only | CA + BoT (stacked) |
| **Default Dropout** | 0.1 | 0.2 |
| **Complexity** | Lower | Higher |
| **Parameters** | ~1.75M | ~1.76M |
| **Accuracy** | ~99.3% | Potentially higher |
| **Training Time** | Standard | Slightly longer |

## Yêu Cầu

### 1. Cấu Trúc Thư Mục

Đảm bảo cấu trúc thư mục như sau:

```
Paddy-Disease-Classification-final/
├── data/
│   ├── metadata.csv
│   ├── label2id.json
│   └── images/
├── train_MobileNetV3_Small_Hybrid.py
├── src/
│   ├── models/
│   │   ├── attention/
│   │   │   ├── botblock.py
│   │   │   └── cablock.py
│   │   └── backbones/
│   │       └── mobilenet.py
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
python train_MobileNetV3_Small_Hybrid.py
```

### 2. Training với Tham Số Tùy Chỉnh

```bash
python train_MobileNetV3_Small_Hybrid.py \
    --epochs 30 \
    --batch-size 32 \
    --heads 4 \
    --reduction 16 \
    --dropout 0.2 \
    --pretrained
```

### 3. Quick Test (Training Nhanh với Dữ Liệu Giới Hạn)

```bash
python train_MobileNetV3_Small_Hybrid.py \
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
| `--heads` | 4 | Số attention heads trong BoT block |
| `--reduction` | 16 | Reduction ratio cho Coordinate Attention (khác với BoT: không có tham số này) |
| `--dropout` | 0.2 | Dropout rate trước classifier (cao hơn BoT: 0.1) |
| `--pretrained` | False | Sử dụng pretrained weights từ ImageNet |

### Training Configuration

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--epochs` | 30 | Số epoch training (nhiều hơn default của BoT: 10) |
| `--batch-size` | 32 | Batch size |
| `--patience` | 10 | Early stopping patience (cao hơn BoT: 5) |
| `--base-lr` | 5e-5 | Learning rate cho backbone (thấp hơn BoT: 1e-4) |
| `--head-lr` | 5e-4 | Learning rate cho classifier head (thấp hơn BoT: 1e-3) |
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
| `--model-name` | MobileNetV3_Small_Hybrid | Tên model cho logging/checkpoint |
| `--device` | auto | Device (cpu/cuda) |
| `--seed` | 42 | Random seed |

### Visualization & Logging

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--plot` | False | Hiển thị biểu đồ training history sau khi train |
| `--save-history` | False | Lưu training history vào file JSON |

## Ví Dụ Thực Tế

### 1. Training Full Dataset với Pretrained Weights (Khuyến Nghị)

```bash
python train_MobileNetV3_Small_Hybrid.py \
    --epochs 30 \
    --batch-size 64 \
    --base-lr 5e-5 \
    --head-lr 5e-4 \
    --heads 4 \
    --reduction 16 \
    --dropout 0.2 \
    --weight-decay 1e-2 \
    --patience 10 \
    --pretrained \
    --num-workers 8 \
    --pin-memory \
    --image-size 224 \
    --plot \
    --save-history
```

**Kết quả mong đợi:**

- Validation Accuracy: ~99.3-99.5% (có thể cao hơn BoT model)
- Training time: ~35-50 phút (GPU T4, lâu hơn BoT ~10-15%)
- Output: `MobileNetV3_Small_Hybrid_best.pt`, `results/MobileNetV3_Small_Hybrid_history.json`

### 2. Training từ Scratch

```bash
python train_MobileNetV3_Small_Hybrid.py \
    --epochs 50 \
    --batch-size 32 \
    --base-lr 1e-3 \
    --head-lr 1e-2 \
    --dropout 0.3 \
    --weight-decay 1e-2 \
    --patience 15 \
    --num-workers 4
```

**Lưu ý:** Training từ scratch với Hybrid model cần:

- Nhiều epochs hơn (50-70)
- Dropout cao hơn (0.3) để tránh overfitting
- Patience cao hơn (15-20)

### 3. Thử Nghiệm Các Reduction Ratios

```bash
# Reduction = 8 (nhiều parameters hơn, chậm hơn)
python train_MobileNetV3_Small_Hybrid.py \
    --epochs 30 \
    --batch-size 48 \
    --reduction 8 \
    --pretrained

# Reduction = 32 (ít parameters hơn, nhanh hơn)
python train_MobileNetV3_Small_Hybrid.py \
    --epochs 30 \
    --batch-size 64 \
    --reduction 32 \
    --pretrained
```

### 4. Training với GPU Cụ Thể

```bash
CUDA_VISIBLE_DEVICES=0 python train_MobileNetV3_Small_Hybrid.py \
    --epochs 30 \
    --batch-size 32 \
    --pretrained \
    --device cuda
```

### 5. Training trên CPU (Không khuyến nghị)

```bash
python train_MobileNetV3_Small_Hybrid.py \
    --epochs 5 \
    --batch-size 8 \
    --device cpu \
    --num-workers 2 \
    --train-limit 1000
```

### 6. Training với Visualization

```bash
python train_MobileNetV3_Small_Hybrid.py \
    --epochs 30 \
    --batch-size 64 \
    --pretrained \
    --plot \
    --save-history
```

Sẽ tạo:

- Biểu đồ training loss/accuracy
- File `results/MobileNetV3_Small_Hybrid_history.json`
- File `results/MobileNetV3_Small_Hybrid_metrics.json`

### 7. Fine-tuning với Dropout Cao

```bash
python train_MobileNetV3_Small_Hybrid.py \
    --epochs 25 \
    --batch-size 32 \
    --base-lr 1e-5 \
    --head-lr 1e-4 \
    --dropout 0.3 \
    --pretrained \
    --patience 12
```

## Đầu Ra

### 1. Checkpoint Files

Model checkpoint sẽ được lưu trong thư mục gốc project:

- `MobileNetV3_Small_Hybrid_best.pt`: Best model checkpoint

### 2. Training Metrics

Sau khi training, metrics sẽ được in ra console:

```plaintext
Training finished. Metrics:
  model_name: MobileNetV3_Small_Hybrid
  size_mb: 6.70
  valid_acc: 0.9940
  valid_f1: 0.9941
  fps: 3850.45
  num_params: 1760000
  ckpt_path: MobileNetV3_Small_Hybrid_best.pt
```

### 3. Training History

Training history được trả về dưới dạng dictionary chứa:

- `train_loss`: List các giá trị loss theo epoch
- `valid_loss`: List các giá trị validation loss
- `train_acc`: List các giá trị accuracy
- `valid_acc`: List các giá trị validation accuracy
- `learning_rates`: List các learning rates theo epoch

### 4. Expected Results

Với cấu hình tối ưu (pretrained, 30 epochs, batch_size=64), bạn có thể đạt:

- **Validation Accuracy**: ~99.3-99.5%
- **F1-Score**: ~99.4%
- **Inference Speed**: ~3,850 FPS (chậm hơn BoT ~3%)
- **Model Size**: ~6.70 MB (lớn hơn BoT ~0.03 MB)
- **Training Time**: ~35-50 phút (GPU T4)

## Kiến Trúc Model

### MobileNetV3-Small Hybrid (CA + BoT)

```plaintext
Input (3x224x224)
    ↓
[MobileNetV3-Small Backbone] (pretrained on ImageNet)
    ↓
Feature Maps (576 channels)
    ↓
[Coordinate Attention Block]
    ├─ X-Direction Attention (reduction=16)
    └─ Y-Direction Attention (reduction=16)
    ↓
Feature Maps (576 channels)
    ↓
[BoTNet Block] (Self-Attention with 4 heads)
    ├─ Multi-Head Self Attention
    └─ Relative Position Encoding
    ↓
Feature Maps (576 channels)
    ↓
[AdaptiveAvgPool2d]
    ↓
[Dropout 0.2]
    ↓
[Linear Classifier] → Output (num_classes)
```

**Đặc điểm:**

- **Backbone**: MobileNetV3-Small (efficient, lightweight)
- **Attention Stack**:
  - CABlock: Spatial attention theo cả 2 chiều X và Y
  - BoTBlock: Multi-head self-attention với relative position encoding
- **Parameters**: ~1.76M (chỉ tăng ~10K so với BoT model)
- **Size**: ~6.70 MB
- **Optimal for**:
  - Cần accuracy cao nhất có thể
  - Có thể chấp nhận tăng nhẹ complexity
  - Cần cả spatial và self-attention

## Troubleshooting

### Lỗi: "FileNotFoundError: metadata file not found"

**Giải pháp:**

```bash
# Chỉ định đúng đường dẫn
python train_MobileNetV3_Small_Hybrid.py \
    --metadata path/to/your/metadata.csv \
    --label2id path/to/your/label2id.json
```

### Lỗi: "ModuleNotFoundError: No module named 'src'"

**Giải pháp:**

- Đảm bảo chạy từ thư mục gốc của project
- Kiểm tra cấu trúc thư mục `src/` tồn tại
- Kiểm tra file `src/models/attention/cablock.py` và `botblock.py` tồn tại

### Out of Memory (OOM)

**Giải pháp:**

```bash
# Giảm batch size (Hybrid model tốn memory hơn BoT ~5-10%)
python train_MobileNetV3_Small_Hybrid.py --batch-size 16

# Hoặc giảm image size
python train_MobileNetV3_Small_Hybrid.py --image-size 192

# Hoặc giảm reduction ratio
python train_MobileNetV3_Small_Hybrid.py --reduction 32
```

### Training Quá Chậm

**Giải pháp:**

```bash
# Tăng số workers
python train_MobileNetV3_Small_Hybrid.py \
    --num-workers 8 \
    --pin-memory

# Hoặc test với subset nhỏ
python train_MobileNetV3_Small_Hybrid.py \
    --train-limit 1000 \
    --valid-limit 200

# Hoặc tăng reduction ratio (giảm complexity)
python train_MobileNetV3_Small_Hybrid.py --reduction 32
```

### Overfitting

**Giải pháp:**

```bash
# Tăng dropout
python train_MobileNetV3_Small_Hybrid.py --dropout 0.3

# Tăng weight decay
python train_MobileNetV3_Small_Hybrid.py --weight-decay 2e-2

# Giảm learning rate
python train_MobileNetV3_Small_Hybrid.py --base-lr 1e-5 --head-lr 1e-4
```

## Tips & Best Practices

### 1. Chọn Learning Rate

- **Pretrained model**: `base_lr=5e-5` đến `1e-5`, `head_lr=5e-4` đến `1e-4`
  - Thấp hơn BoT vì model phức tạp hơn, dễ diverge
- **From scratch**: `base_lr=5e-4` đến `1e-3`, `head_lr=5e-3` đến `1e-2`
  - Cũng thấp hơn BoT từ scratch

### 2. Chọn Reduction Ratio

- **reduction=8**: Nhiều parameters, accuracy cao hơn, chậm hơn, dễ overfit
- **reduction=16**: Cân bằng tốt (khuyến nghị)
- **reduction=32**: Ít parameters, nhanh hơn, accuracy có thể thấp hơn

### 3. Batch Size

- **GPU 8GB**: batch_size=48-64 (giảm 10-20% so với BoT)
- **GPU 4GB**: batch_size=16-24
- **CPU**: batch_size=4-8 (không khuyến nghị)

### 4. Early Stopping

- Sử dụng `--patience 10-15` (cao hơn BoT vì cần nhiều thời gian converge)
- Model phức tạp hơn nên cần patience cao hơn

### 5. Dropout

- **Default**: 0.2 (đã tăng so với BoT: 0.1)
- **If overfitting**: 0.3-0.4
- **If underfitting**: 0.1-0.15

### 6. Image Size

- **224x224**: Standard, khuyến nghị
- **192x192**: Nhanh hơn ~30%, accuracy giảm ~0.5%
- **256x256**: Chậm hơn ~30%, có thể tăng accuracy ~0.3%

### 7. Monitoring

Theo dõi các chỉ số trong quá trình training:

- **Loss giảm đều đặn**: Good, model đang học
- **Loss tăng sau vài epochs**: Overfitting hoặc learning rate cao
- **Loss dao động mạnh**: Learning rate quá cao, giảm xuống
- **Accuracy không cải thiện**:
  - Learning rate quá thấp
  - Cần tăng patience
  - Model có thể đã converge

### 8. So Sánh với BoT

Khi nào dùng Hybrid thay vì BoT:

- ✅ **Dùng Hybrid khi**: Cần accuracy cao nhất, có GPU đủ mạnh, không quá quan tâm tốc độ
- ✅ **Dùng BoT khi**: Cần tốc độ nhanh, deploy trên mobile/edge devices, accuracy ~99% là đủ

## Performance Comparison

| Metric | MobileNetV3_Small_BoT | MobileNetV3_Small_Hybrid |
|--------|----------------------|--------------------------|
| **Accuracy** | ~99.33% | ~99.40-99.50% |
| **F1-Score** | ~99.33% | ~99.41% |
| **FPS** | ~3,969 | ~3,850 (-3%) |
| **Size** | ~6.67 MB | ~6.70 MB (+0.03 MB) |
| **Parameters** | ~1.75M | ~1.76M (+10K) |
| **Training Time** | ~30-45 min | ~35-50 min (+15%) |
| **Memory Usage** | Standard | +5-10% |

## Tham Khảo

- Model Implementation: `src/models/backbones/mobilenet.py` (class `MobileNetV3_Small_Hybrid`)
- Training Logic: `src/training/train.py`
- Data Loading: `src/utils/data/loading.py`
- Coordinate Attention: `src/models/attention/cablock.py`
- BoTNet Block: `src/models/attention/botblock.py`

## Related Files

- `train_MobileNetV3_Small_BoT.py` - Training script cho BoT-only model
- `train_MobileNetV3_Small_BoT.md` - Documentation cho BoT model

## License

MIT License
