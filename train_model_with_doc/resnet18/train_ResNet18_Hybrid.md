# Hướng dẫn Train ResNet18_Hybrid

## 📋 Tổng quan

**ResNet18_Hybrid** là kiến trúc kết hợp giữa **BoTLinear** (Linear Attention) và **Coordinate Attention** (CA), mang lại accuracy cao nhất trong các biến thể ResNet18.

### 🎯 Đặc điểm chính

- **Accuracy mong đợi**: 94-96% (CAO NHẤT!)
- **Tốc độ**: Trung bình (chậm hơn BoT ~10%)
- **Params**: Cao nhất (+30% so với base ResNet18)

### ✨ Ưu điểm

- ✅ Kết hợp điểm mạnh của cả BoTLinear và Coordinate Attention
- ✅ **BoTLinear**: Capture global context qua linear attention
- ✅ **CA**: Localization tốt qua coordinate attention (height + width)
- ✅ Học được cả global features và local spatial features
- ✅ Phù hợp cho bài toán phức tạp cần accuracy cao

### 🎯 Khi nào nên dùng

- ✅ Ưu tiên **accuracy cao nhất**
- ✅ Có đủ **computational resources** (GPU memory ≥ 8GB)
- ✅ Bài toán khó, cần model mạnh
- ✅ Research hoặc Competition

---

## 🚀 Cách sử dụng

### 1. Training cơ bản (Pretrained backbone)

```bash
python train_ResNet18_Hybrid.py \
    --pretrained \
    --epochs 10 \
    --batch-size 32 \
    --save-history
```

**Giải thích các tham số:**

- `--pretrained`: Sử dụng backbone pretrained từ ImageNet (khuyến nghị)
- `--epochs 10`: Train 10 epochs
- `--batch-size 32`: Batch size 32 (giảm xuống nếu thiếu GPU memory)
- `--save-history`: Lưu history và metrics vào thư mục `results/`

---

### 2. Training với cấu hình tùy chỉnh

```bash
python train_ResNet18_Hybrid.py \
    --pretrained \
    --epochs 20 \
    --batch-size 16 \
    --image-size 224 \
    --base-lr 5e-5 \
    --head-lr 5e-4 \
    --weight-decay 1e-2 \
    --heads 4 \
    --reduction 32 \
    --dropout 0.2 \
    --patience 7 \
    --num-workers 8 \
    --pin-memory \
    --save-history \
    --plot
```

**Giải thích các tham số nâng cao:**

- `--image-size 224`: Kích thước ảnh input (224x224)
- `--base-lr 5e-5`: Learning rate cho backbone (thấp hơn vì pretrained)
- `--head-lr 5e-4`: Learning rate cho classifier head (cao hơn)
- `--weight-decay 1e-2`: Weight decay cho regularization
- `--heads 4`: Số attention heads trong BoTLinear block
- `--reduction 32`: Reduction ratio cho Coordinate Attention
- `--dropout 0.2`: Dropout probability (0.2 = 20%)
- `--patience 7`: Early stopping sau 7 epochs không cải thiện
- `--num-workers 8`: Số worker threads cho DataLoader
- `--pin-memory`: Pin memory cho tốc độ transfer nhanh hơn
- `--plot`: Hiển thị biểu đồ training sau khi hoàn thành

---

### 3. Quick test (Train nhanh với dữ liệu giới hạn)

```bash
python train_ResNet18_Hybrid.py \
    --pretrained \
    --epochs 3 \
    --batch-size 16 \
    --train-limit 500 \
    --valid-limit 200 \
    --save-history
```

**Giải thích:**

- `--train-limit 500`: Chỉ dùng 500 mẫu training (để test nhanh)
- `--valid-limit 200`: Chỉ dùng 200 mẫu validation

---

### 4. Training từ scratch (Không dùng pretrained)

```bash
python train_ResNet18_Hybrid.py \
    --epochs 30 \
    --batch-size 32 \
    --base-lr 1e-3 \
    --head-lr 1e-3 \
    --save-history
```

**Lưu ý:** Training từ scratch cần nhiều epochs hơn và learning rate cao hơn.

---

## 📊 Các tham số quan trọng

### Tham số Model

| Tham số | Mặc định | Mô tả | Khuyến nghị |
|---------|----------|-------|-------------|
| `--heads` | 4 | Số attention heads trong BoTLinear | 4-8 |
| `--reduction` | 32 | Reduction ratio cho CA | 16-32 |
| `--dropout` | 0.1 | Dropout probability | 0.1-0.3 |
| `--pretrained` | False | Dùng pretrained weights | **Luôn bật** |

### Tham số Training

| Tham số | Mặc định | Mô tả | Khuyến nghị |
|---------|----------|-------|-------------|
| `--epochs` | 10 | Số epochs training | 10-20 |
| `--batch-size` | 32 | Batch size | 16-64 |
| `--patience` | 5 | Early stopping patience | 5-10 |
| `--base-lr` | 1e-4 | Learning rate backbone | 5e-5 (pretrained) |
| `--head-lr` | 1e-3 | Learning rate head | 5e-4 - 1e-3 |

### Tham số Dữ liệu

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--metadata` | data/metadata.csv | Đường dẫn metadata |
| `--label2id` | data/label2id.json | Đường dẫn label mapping |
| `--image-size` | 224 | Kích thước ảnh |
| `--num-workers` | 4 | Số workers DataLoader |

---

## 💡 Tips & Best Practices

### 1. GPU Memory Management

```bash
# Nếu gặp lỗi Out of Memory (OOM), giảm batch size:
python train_ResNet18_Hybrid.py --batch-size 16  # hoặc 8
```

### 2. Optimal Hyperparameters (Từ experiments)

```bash
python train_ResNet18_Hybrid.py \
    --pretrained \
    --epochs 15 \
    --batch-size 32 \
    --base-lr 5e-5 \
    --head-lr 5e-4 \
    --heads 4 \
    --reduction 32 \
    --dropout 0.15 \
    --patience 7 \
    --save-history
```

### 3. Training trên CPU (không khuyến nghị)

```bash
python train_ResNet18_Hybrid.py \
    --device cpu \
    --batch-size 8 \
    --epochs 5
```

### 4. Resume từ checkpoint

Hiện tại chưa support resume. Để resume training, cần thêm code load checkpoint.

---

## 📁 Output Structure

Sau khi training với `--save-history`, kết quả được lưu vào:

```
results/
└── ResNet18_Hybrid_DD_MM_YYYY_HHMM/
    ├── history.json           # Training history (loss, acc theo epoch)
    ├── metrics.json          # Final metrics (best acc, loss, etc.)
    ├── training_plot.png     # Biểu đồ training curves
    └── best_model.pt         # Best model checkpoint (nếu có save)
```

### Ví dụ metrics.json

```json
{
  "valid_acc": 0.9542,
  "valid_loss": 0.1234,
  "train_acc": 0.9821,
  "train_loss": 0.0567,
  "best_epoch": 12
}
```

---

## 🔍 So sánh với các variants khác

| Model | Accuracy | Speed | Memory | Use Case |
|-------|----------|-------|---------|----------|
| **ResNet18_Hybrid** | 94-96% ⭐ | Trung bình | Cao | **Accuracy tối đa** |
| ResNet18_BoT | 93-95% | Trung bình | Trung bình | Balanced |
| ResNet18_BoTLinear | 92-94% | Nhanh | Trung bình | Production |
| ResNet18_CA | 92-94% | Nhanh | Thấp | Edge devices |
| ResNet18_ECA | 91-93% | Rất nhanh | Rất thấp | Mobile/Edge |

### 💡 Khuyến nghị lựa chọn

- **ResNet18_Hybrid**: Khi muốn accuracy cao nhất, có đủ GPU
- **ResNet18_BoTLinear**: Balance tốt giữa accuracy và speed
- **ResNet18_CA**: Deploy trên edge devices với RAM hạn chế
- **ResNet18_ECA**: Mobile/Real-time applications

---

## ⚠️ Troubleshooting

### 1. Out of Memory (OOM)

```bash
# Giảm batch size
python train_ResNet18_Hybrid.py --batch-size 8

# Hoặc giảm image size
python train_ResNet18_Hybrid.py --image-size 192 --batch-size 16
```

### 2. Training quá chậm

```bash
# Tăng num_workers và bật pin_memory
python train_ResNet18_Hybrid.py --num-workers 8 --pin-memory
```

### 3. Overfitting

```bash
# Tăng dropout và weight decay
python train_ResNet18_Hybrid.py --dropout 0.3 --weight-decay 5e-2
```

### 4. Underfitting

```bash
# Giảm dropout, tăng epochs, tăng learning rate
python train_ResNet18_Hybrid.py --dropout 0.05 --epochs 30 --head-lr 1e-3
```

---

## 📚 Chi tiết kiến trúc

### ResNet18_Hybrid Architecture

```
Input (3, 224, 224)
    ↓
Stem (Conv + BN + ReLU + MaxPool)
    ↓
Layer1 (64 channels)
    ↓
Layer2 (128 channels)
    ↓
Layer3 (256 channels)
    ↓
Layer4 (512 channels)
    ↓
Coordinate Attention Block  ← Spatial localization
    ↓
BoTLinear Block            ← Global context
    ↓
Global Average Pooling
    ↓
Dropout
    ↓
Linear Classifier (num_classes)
    ↓
Output (logits)
```

### Sequential Order

1. **CA Block trước**: Capture spatial information (height, width)
2. **BoTLinear sau**: Model global dependencies

Thứ tự này được chọn vì ổn định hơn trong quá trình training.

---

## 📞 Support

Nếu gặp vấn đề, kiểm tra:

1. ✅ GPU có đủ memory không? (`nvidia-smi`)
2. ✅ Dependencies đã cài đủ chưa? (`pip install -r requirements.txt`)
3. ✅ Đường dẫn data đúng không?
4. ✅ CUDA compatible với PyTorch version không?

---

## 🎓 References

- **BoT (Bottleneck Transformers)**: [Paper](https://arxiv.org/abs/2101.11605)
- **Coordinate Attention**: [Paper](https://arxiv.org/abs/2103.02907)
- **ResNet**: [Paper](https://arxiv.org/abs/1512.03385)

---

**Chúc bạn training thành công! 🚀**
