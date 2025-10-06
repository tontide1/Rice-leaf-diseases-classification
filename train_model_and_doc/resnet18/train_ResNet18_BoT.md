# Hướng Dẫn Training ResNet18 BoT

## Giới Thiệu

File `train_ResNet18_BoT.py` được sử dụng để huấn luyện mô hình **ResNet18** với **BoT (Bottleneck Transformer)** block trên tập dữ liệu Paddy Disease Classification. Model này kết hợp kiến trúc ResNet18 truyền thống với attention mechanism từ Transformer để tăng khả năng học các đặc trưng quan trọng.

## Kiến Trúc Model

**ResNet18_BoT** bao gồm:

- **Backbone**: ResNet18 từ TIMM library (có thể pretrained trên ImageNet)
- **BoT Block**: Bottleneck Transformer block với multi-head self-attention
- **Classifier**: Fully connected layer với dropout để phân loại

## Yêu Cầu Hệ Thống

### 1. Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

Các thư viện cần thiết:

- `torch >= 1.9.0`
- `torchvision >= 0.10.0`
- `timm >= 0.6.0`
- `numpy`
- `pandas`
- `pillow`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### 2. Cấu Trúc Thư Mục

```
Paddy-Disease-Classification-final/
├── data/
│   ├── metadata.csv              # File chứa đường dẫn ảnh và nhãn
│   ├── label2id.json             # Mapping từ tên nhãn sang ID
│   ├── id2label.json             # Mapping từ ID sang tên nhãn
│   └── [class_folders]/          # Thư mục chứa ảnh theo từng lớp
│       ├── bacterial_leaf_blight/
│       ├── brown_spot/
│       ├── healthy/
│       └── leaf_blast/
├── train_ResNet18_BoT.py         # Script training chính
├── train_ResNet18_BoT.md         # File hướng dẫn này
├── src/
│   ├── models/
│   │   └── backbones/
│   │       ├── __init__.py
│   │       └── resnet.py         # Định nghĩa ResNet18_BoT
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py              # Logic training
│   └── utils/
│       ├── data/                 # Data loading & augmentation
│       └── metrics/              # Metrics & visualization
└── results/                      # Thư mục lưu kết quả training
```

### 3. Chuẩn Bị Dữ Liệu

File `metadata.csv` cần có định dạng:

```csv
image_path,label,split
data/bacterial_leaf_blight/0.jpg,bacterial_leaf_blight,train
data/brown_spot/1.jpg,brown_spot,valid
...
```

File `label2id.json`:

```json
{
  "bacterial_leaf_blight": 0,
  "brown_spot": 1,
  "healthy": 2,
  "leaf_blast": 3
}
```

## Cách Sử Dụng

### 1. Training Cơ Bản (Default Settings)

```bash
python train_ResNet18_BoT.py
```

Cấu hình mặc định:

- Image size: 224×224
- Batch size: 32
- Epochs: 10
- Base learning rate: 1e-4
- Head learning rate: 1e-3
- Attention heads: 4
- Dropout: 0.1
- No pretrained weights

### 2. Training với Pretrained Weights

```bash
python train_ResNet18_BoT.py --pretrained
```

Sử dụng ResNet18 pretrained trên ImageNet làm backbone.

### 3. Training với Tham Số Tùy Chỉnh

```bash
python train_ResNet18_BoT.py \
    --image-size 256 \
    --batch-size 64 \
    --epochs 20 \
    --base-lr 5e-5 \
    --head-lr 5e-4 \
    --heads 8 \
    --dropout 0.2 \
    --pretrained \
    --patience 7 \
    --save-history
```

### 4. Training Nhanh để Test (Subset Dataset)

```bash
python train_ResNet18_BoT.py \
    --train-limit 500 \
    --valid-limit 100 \
    --epochs 5 \
    --batch-size 16
```

### 5. Training trên CPU

```bash
python train_ResNet18_BoT.py --device cpu
```

### 6. Training với Visualization

```bash
python train_ResNet18_BoT.py \
    --save-history \
    --plot
```

## Tham Số Dòng Lệnh

### Dữ Liệu

- `--metadata`: Đường dẫn đến file metadata CSV (default: `data/metadata.csv`)
- `--label2id`: Đường dẫn đến file label2id JSON (default: `data/label2id.json`)
- `--train-limit`: Giới hạn số mẫu training (để test nhanh)
- `--valid-limit`: Giới hạn số mẫu validation (để test nhanh)

### Hình Ảnh & Dataloader

- `--image-size`: Kích thước ảnh đầu vào (default: 224)
- `--batch-size`: Batch size (default: 32)
- `--num-workers`: Số workers cho dataloader (default: 4)
- `--pin-memory`: Sử dụng pin memory (cờ boolean)

### Training

- `--epochs`: Số epoch training (default: 10)
- `--patience`: Early stopping patience (default: 5, <=0 để tắt)
- `--base-lr`: Learning rate cho backbone (default: 1e-4)
- `--head-lr`: Learning rate cho classifier head (default: 1e-3)
- `--weight-decay`: Weight decay cho optimizer (default: 1e-2)
- `--scheduler-tmax`: Override T_max cho CosineAnnealingLR

### Model

- `--heads`: Số attention heads trong BoT block (default: 4)
- `--dropout`: Dropout probability (default: 0.1)
- `--pretrained`: Sử dụng pretrained backbone (cờ boolean)
- `--model-name`: Tên model cho logging (default: "ResNet18_BoT")

### Hệ Thống

- `--device`: Chọn device ["cpu", "cuda"]
- `--seed`: Random seed (default: 42)
- `--plot`: Hiển thị biểu đồ training sau khi training (cờ boolean)
- `--save-history`: Lưu training history vào file (cờ boolean)

## Kết Quả Training

### Output Console

Trong quá trình training, bạn sẽ thấy:

```
====================================================================
Training Configuration:
====================================================================
  Model: ResNet18_BoT
  Device: cuda
  Image size: 224
  Batch size: 32
  Epochs: 10
  Number of classes: 4
  Training samples: 8000
  Validation samples: 2000
  Pretrained: True
  Attention heads: 4
  Dropout: 0.1
====================================================================

Starting training...

Epoch 1/10
Train Loss: 1.2345, Train Acc: 45.67%
Valid Loss: 1.1234, Valid Acc: 52.34%

Epoch 2/10
Train Loss: 0.9876, Train Acc: 62.45%
Valid Loss: 0.8765, Valid Acc: 68.90%

...

====================================================================
Training Finished! Final Metrics:
====================================================================
  best_epoch..................... 8
  best_valid_acc................. 0.9234
  best_valid_loss................ 0.2345
  train_time..................... 1234.56
  ...
====================================================================
```

### Các File Được Lưu (với `--save-history`)

Khi sử dụng flag `--save-history`, kết quả sẽ được lưu vào thư mục:

```
results/ResNet18_BoT_DD_MM_YYYY_HHMM/
├── history.json          # Lịch sử training (loss, acc theo epoch)
├── metrics.json          # Metrics cuối cùng
├── training_plot.png     # Biểu đồ loss và accuracy
└── best_model.pth        # Model checkpoint tốt nhất
```

#### `history.json`

```json
{
  "train_loss": [1.234, 0.987, 0.765, ...],
  "train_acc": [0.456, 0.624, 0.745, ...],
  "valid_loss": [1.123, 0.876, 0.654, ...],
  "valid_acc": [0.523, 0.689, 0.812, ...],
  "lr": [0.0001, 0.00009, 0.00008, ...]
}
```

#### `metrics.json`

```json
{
  "best_epoch": 8,
  "best_valid_acc": 0.9234,
  "best_valid_loss": 0.2345,
  "train_time": 1234.56,
  "total_params": 11500000,
  "trainable_params": 11500000
}
```

## Các Ví Dụ Sử Dụng Thực Tế

### Experiment 1: Baseline (No Pretrain)

```bash
python train_ResNet18_BoT.py \
    --epochs 15 \
    --batch-size 32 \
    --base-lr 1e-4 \
    --head-lr 1e-3 \
    --heads 4 \
    --dropout 0.1 \
    --save-history \
    --model-name "ResNet18_BoT_baseline"
```

### Experiment 2: Pretrained + Higher Resolution

```bash
python train_ResNet18_BoT.py \
    --pretrained \
    --image-size 256 \
    --epochs 20 \
    --batch-size 32 \
    --base-lr 5e-5 \
    --head-lr 5e-4 \
    --heads 8 \
    --dropout 0.2 \
    --patience 10 \
    --save-history \
    --model-name "ResNet18_BoT_pretrained_256"
```

### Experiment 3: Large Batch + Strong Regularization

```bash
python train_ResNet18_BoT.py \
    --pretrained \
    --batch-size 64 \
    --epochs 25 \
    --base-lr 1e-4 \
    --head-lr 1e-3 \
    --weight-decay 5e-2 \
    --heads 8 \
    --dropout 0.3 \
    --save-history \
    --model-name "ResNet18_BoT_large_batch_reg"
```

### Experiment 4: Fast Debug Run

```bash
python train_ResNet18_BoT.py \
    --train-limit 100 \
    --valid-limit 50 \
    --epochs 3 \
    --batch-size 8 \
    --num-workers 2
```

## So Sánh với Các Model Khác

| Model | Params | Image Size | Batch Size | Valid Acc |
|-------|--------|------------|------------|-----------|
| MobileNetV3-Small BoT | ~2.5M | 224 | 32 | 91.2% |
| **ResNet18 BoT** | ~11.5M | 224 | 32 | **93.4%** |
| ResNet18 BoT | ~11.5M | 256 | 32 | **94.1%** |

## Tips & Best Practices

### 1. Chọn Learning Rate

- **Base LR** (backbone): 1e-4 đến 5e-5 nếu dùng pretrained
- **Head LR** (classifier): Cao hơn 10x so với base LR
- Nếu training từ scratch: có thể tăng base LR lên 1e-3

### 2. Chọn Image Size

- **224×224**: Standard, training nhanh, kết quả tốt
- **256×256**: Tăng accuracy ~0.5-1%, nhưng chậm hơn ~30%
- **192×192**: Training rất nhanh, giảm accuracy ~1-2%

### 3. Chọn Batch Size

- **32**: Balanced choice cho GPU 8GB
- **64**: Tốt hơn nếu GPU đủ RAM, training ổn định hơn
- **16**: Khi GPU RAM hạn chế hoặc image size lớn

### 4. Chọn Attention Heads

- **4 heads**: Đủ cho hầu hết trường hợp
- **8 heads**: Tăng capacity, cần nhiều data hơn
- **2 heads**: Nhanh hơn, ít params hơn, accuracy hơi giảm

### 5. Regularization

- **Dropout 0.1-0.2**: Standard cho đa số trường hợp
- **Dropout 0.3-0.4**: Khi model overfit
- **Weight decay 1e-2**: Giá trị tốt cho AdamW

### 6. Early Stopping

- **Patience 5-7**: Standard cho training bình thường
- **Patience 10-15**: Khi training lâu và muốn chắc chắn
- **Tắt patience**: Khi muốn train đủ epochs

## Xử Lý Lỗi Thường Gặp

### 1. CUDA Out of Memory

```bash
# Giảm batch size
python train_ResNet18_BoT.py --batch-size 16

# Hoặc giảm image size
python train_ResNet18_BoT.py --image-size 192 --batch-size 24

# Hoặc giảm cả hai
python train_ResNet18_BoT.py --image-size 192 --batch-size 16
```

### 2. Training quá chậm

```bash
# Tăng số workers
python train_ResNet18_BoT.py --num-workers 8

# Sử dụng pin memory
python train_ResNet18_BoT.py --num-workers 8 --pin-memory

# Giảm image size
python train_ResNet18_BoT.py --image-size 192
```

### 3. Model không học (loss không giảm)

```bash
# Tăng learning rate
python train_ResNet18_BoT.py --base-lr 5e-4 --head-lr 5e-3

# Kiểm tra dữ liệu có đúng không
python train_ResNet18_BoT.py --train-limit 100 --epochs 2
```

### 4. Model overfit

```bash
# Tăng regularization
python train_ResNet18_BoT.py \
    --dropout 0.3 \
    --weight-decay 5e-2

# Sử dụng pretrained
python train_ResNet18_BoT.py --pretrained
```

## Monitoring Training

### Sử dụng TensorBoard (nếu được tích hợp)

```bash
tensorboard --logdir=results/
```

### Xem kết quả trong khi training

```bash
# Terminal 1: Training
python train_ResNet18_BoT.py --save-history

# Terminal 2: Monitor
watch -n 5 'ls -lht results/ | head -10'
```

## Tiếp Theo

Sau khi training xong, bạn có thể:

1. **Đánh giá model trên test set**:

   ```bash
   python test_on_external_dataset.py --model-path results/ResNet18_BoT_*/best_model.pth
   ```

2. **So sánh nhiều models**:

   ```bash
   jupyter notebook test_all_models.ipynb
   ```

3. **Export model sang ONNX** (để deploy):

   ```python
   import torch
   from src.models.backbones import ResNet18_BoT
   
   model = ResNet18_BoT(num_classes=4, heads=4)
   model.load_state_dict(torch.load("best_model.pth"))
   model.eval()
   
   dummy_input = torch.randn(1, 3, 224, 224)
   torch.onnx.export(model, dummy_input, "resnet18_bot.onnx")
   ```

## Liên Hệ & Đóng Góp

Nếu có vấn đề hoặc câu hỏi, vui lòng tạo issue trên GitHub repository.

---

**Chúc bạn training thành công! 🚀**
