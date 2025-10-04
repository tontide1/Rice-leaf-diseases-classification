# Hướng Dẫn Training MobileNetV3-Small BoT

## Giới Thiệu

File `train_MobileNetV3_Small_BoT.py` được sử dụng để huấn luyện mô hình MobileNetV3-Small với BoT (Bottleneck Transformer) block trên tập dữ liệu Paddy Disease Classification.

## Yêu Cầu

### 1. Cấu Trúc Thư Mục

Đảm bảo cấu trúc thư mục như sau:

```
Paddy-Disease-Classification-final/
├── data/
│   ├── metadata.csv
│   ├── label2id.json
│   └── images/
├── train_MobileNetV3_Small_BoT.py
├── src/
│   ├── models/
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
python train_MobileNetV3_Small_BoT.py
```

### 2. Training với Tham Số Tùy Chỉnh

```bash
python train_MobileNetV3_Small_BoT.py \
    --epochs 20 \
    --batch-size 32 \
    --base-lr 1e-4 \
    --head-lr 1e-3 \
    --pretrained
```

### 3. Quick Test (Training Nhanh với Dữ Liệu Giới Hạn)

```bash
python train_MobileNetV3_Small_BoT.py \
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
| `--dropout` | 0.1 | Dropout rate trước classifier |
| `--pretrained` | False | Sử dụng pretrained weights từ ImageNet |

### Training Configuration

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--epochs` | 10 | Số epoch training |
| `--batch-size` | 32 | Batch size |
| `--patience` | 5 | Early stopping patience (<=0 để tắt) |
| `--base-lr` | 1e-4 | Learning rate cho backbone |
| `--head-lr` | 1e-3 | Learning rate cho classifier head |
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
| `--model-name` | MobileNetV3_Small_BoT | Tên model cho logging/checkpoint |
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
python train_MobileNetV3_Small_BoT.py \
    --epochs 30 \
    --batch-size 64 \
    --base-lr 5e-5 \
    --head-lr 5e-4 \
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

- Validation Accuracy: ~99.3%
- Training time: ~30-45 phút (GPU T4)
- Output: `MobileNetV3_Small_BoT_best.pt`, `results/MobileNetV3_Small_BoT_history.json`

### 2. Training từ Scratch

```bash
python train_MobileNetV3_Small_BoT.py \
    --epochs 50 \
    --batch-size 32 \
    --base-lr 1e-3 \
    --head-lr 1e-2 \
    --weight-decay 1e-2 \
    --patience 15 \
    --num-workers 4
```

### 3. Training với GPU Cụ Thể

```bash
CUDA_VISIBLE_DEVICES=0 python train_MobileNetV3_Small_BoT.py \
    --epochs 20 \
    --batch-size 32 \
    --pretrained \
    --device cuda
```

### 4. Training trên CPU

```bash
python train_MobileNetV3_Small_BoT.py \
    --epochs 5 \
    --batch-size 8 \
    --device cpu \
    --num-workers 2
```

### 5. Fine-tuning với Learning Rate Thấp

```bash
python train_MobileNetV3_Small_BoT.py \
    --epochs 20 \
    --batch-size 32 \
    --base-lr 1e-5 \
    --head-lr 1e-4 \
    --pretrained \
    --patience 8
```

### 6. Training với Visualization

```bash
python train_MobileNetV3_Small_BoT.py \
    --epochs 30 \
    --batch-size 64 \
    --pretrained \
    --plot \
    --save-history
```

Sẽ tạo:

- Biểu đồ training loss/accuracy
- File `results/MobileNetV3_Small_BoT_history.json`
- File `results/MobileNetV3_Small_BoT_metrics.json`

## Đầu Ra

### 1. Checkpoint Files

Model checkpoint sẽ được lưu trong thư mục gốc project:

- `MobileNetV3_Small_BoT_best.pt`: Best model checkpoint

### 2. Training Metrics

Sau khi training, metrics sẽ được in ra console:

```plaintext
Training finished. Metrics:
  model_name: MobileNetV3_Small_BoT
  size_mb: 6.67
  valid_acc: 0.9933
  valid_f1: 0.9933
  fps: 3969.14
  num_params: 1748980
  ckpt_path: MobileNetV3_Small_BoT_best.pt
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

- **Validation Accuracy**: ~99.3%
- **F1-Score**: ~99.3%
- **Inference Speed**: ~3,969 FPS
- **Model Size**: ~6.67 MB
- **Training Time**: ~30-45 phút (GPU T4)

## Troubleshooting

### Lỗi: "FileNotFoundError: metadata file not found"

**Giải pháp:**

```bash
# Chỉ định đúng đường dẫn
python train_MobileNetV3_Small_BoT.py \
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
python train_MobileNetV3_Small_BoT.py --batch-size 16

# Hoặc giảm image size
python train_MobileNetV3_Small_BoT.py --image-size 192
```

### Training Quá Chậm

**Giải pháp:**

```bash
# Tăng số workers
python train_MobileNetV3_Small_BoT.py \
    --num-workers 8 \
    --pin-memory

# Hoặc test với subset nhỏ
python train_MobileNetV3_Small_BoT.py \
    --train-limit 1000 \
    --valid-limit 200
```

## Tips & Best Practices

### 1. Chọn Learning Rate

- **Pretrained model**: `base_lr=1e-5` đến `5e-5`, `head_lr=1e-4` đến `5e-4`
- **From scratch**: `base_lr=1e-3` đến `5e-3`, `head_lr=1e-2` đến `5e-2`

### 2. Batch Size

- **GPU 8GB**: batch_size=32-64
- **GPU 4GB**: batch_size=16-32
- **CPU**: batch_size=8-16

### 3. Early Stopping

- Sử dụng `--patience 5-10` để tránh overfitting
- Tắt early stopping với `--patience 0` hoặc `--patience -1`

### 4. Image Size

- 224x224: Standard, cân bằng accuracy và speed
- 192x192: Nhanh hơn, accuracy thấp hơn một chút
- 256x256: Chậm hơn, có thể tăng accuracy

### 5. Monitoring

Theo dõi các chỉ số trong quá trình training:

- Loss giảm đều: Good
- Loss tăng: Learning rate quá cao hoặc overfitting
- Accuracy không cải thiện: Learning rate quá thấp hoặc model capacity không đủ

## Kiến Trúc Model

### MobileNetV3-Small BoT

```
Input (3x224x224)
    ↓
[MobileNetV3-Small Backbone] (pretrained on ImageNet)
    ↓
Feature Maps (576 channels)
    ↓
[BoTNet Block] (Self-Attention with 4 heads)
    ↓
[AdaptiveAvgPool2d]
    ↓
[Dropout 0.1]
    ↓
[Linear Classifier] → Output (num_classes)
```

**Đặc điểm:**

- Backbone: MobileNetV3-Small (efficient, lightweight)
- Attention: BoTNet block với multi-head self-attention
- Parameters: ~1.75M
- Size: ~6.67 MB
- Optimal for: Mobile deployment, real-time inference

## Tham Khảo

- Model Implementation: `src/models/backbones/mobilenet.py` (class `MobileNetV3_Small_BoT`)
- Training Logic: `src/training/train.py`
- Data Loading: `src/utils/data/loading.py`
- BoTNet Block: `src/models/attention/botblock.py`

## License

MIT License
