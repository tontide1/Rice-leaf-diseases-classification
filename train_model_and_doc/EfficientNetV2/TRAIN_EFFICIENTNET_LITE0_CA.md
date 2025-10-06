# 🚀 Hướng Dẫn Training EfficientNet-Lite0 + CA

## 📋 Mục Lục

1. [Giới Thiệu Model](#giới-thiệu-model)
2. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
3. [Cài Đặt](#cài-đặt)
4. [Chuẩn Bị Dữ Liệu](#chuẩn-bị-dữ-liệu)
5. [Training](#training)
6. [Các Tham Số Quan Trọng](#các-tham-số-quan-trọng)
7. [Kết Quả Mong Đợi](#kết-quả-mong-đợi)
8. [Troubleshooting](#troubleshooting)
9. [Tips & Tricks](#tips--tricks)

---

## 🎯 Giới Thiệu Model

**EfficientNet-Lite0 + Coordinate Attention** là model cực kỳ nhẹ được tối ưu cho **mobile và edge devices**.

### Đặc Điểm Nổi Bật

| Đặc điểm | Giá trị |
|----------|---------|
| **Số parameters** | ~4.7M (nhẹ nhất!) |
| **FLOPs** | ~0.4G |
| **Accuracy mong đợi** | 90-92% |
| **FPS (GPU T4)** | 600+ |
| **FPS (CPU)** | ~80 |
| **Model size** | ~19 MB |

### So Sánh Với Các Model Khác

| Model | Params | FLOPs | Accuracy | FPS (GPU) | FPS (CPU) |
|-------|--------|-------|----------|-----------|-----------|
| EfficientNet-Lite0 + CA | 4.7M | 0.4G | 90-92% | 600+ | 80 |
| MobileNetV3-Small + BoT | 3.5M | 0.3G | 91-93% | 550 | 70 |
| ResNet18 + BoT | 11M | 1.8G | 93-95% | 400 | 45 |
| EfficientNetV2-S + CA | 21.5M | 3.0G | 93-95% | 320 | 25 |

### Khi Nào Nên Dùng Model Này?

✅ **Nên dùng khi:**

- Deploy trên **mobile devices** (Android/iOS)
- Chạy trên **edge devices** (Raspberry Pi, Jetson Nano)
- Cần **real-time inference** với độ trễ thấp
- Giới hạn **RAM/Storage** (< 50MB model size)
- Thiết bị chạy **pin** (battery-powered)
- CPU inference là chính

❌ **Không nên dùng khi:**

- Cần accuracy tuyệt đối cao nhất (>95%)
- Có GPU server mạnh và không quan tâm tốc độ
- Dataset có nhiễu phức tạp hoặc cần multi-scale features

---

## 💻 Yêu Cầu Hệ Thống

### Tối Thiểu (CPU)

- **CPU:** 4 cores
- **RAM:** 8GB
- **Storage:** 2GB (dữ liệu + model)
- **OS:** Linux/Windows/macOS

### Khuyến Nghị (GPU)

- **GPU:** NVIDIA GPU với >= 4GB VRAM (GTX 1650 trở lên)
- **RAM:** 16GB
- **Storage:** SSD với >= 10GB
- **CUDA:** >= 11.0

### Software

- **Python:** >= 3.8
- **PyTorch:** >= 1.10
- **timm:** >= 0.6.0
- **CUDA Toolkit:** 11.0+ (nếu dùng GPU)

---

## 🔧 Cài Đặt

### 1. Clone Repository

```bash
git clone https://github.com/tontide1/Rice-leaf-diseases-classification.git
cd Rice-leaf-diseases-classification
```

### 2. Tạo Virtual Environment

```bash
# Dùng venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Hoặc dùng conda
conda create -n paddy python=3.10
conda activate paddy
```

### 3. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

**File `requirements.txt` cần có:**

```txt
torch>=1.10.0
torchvision>=0.11.0
timm>=0.6.0
pillow>=9.0.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

### 4. Kiểm Tra Cài Đặt

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
```

---

## 📂 Chuẩn Bị Dữ Liệu

### Cấu Trúc Thư Mục

Dữ liệu phải có cấu trúc sau:

```
Paddy-Disease-Classification-final/
├── data/
│   ├── metadata.csv          # File metadata chứa (path, label, split)
│   ├── label2id.json         # Mapping từ tên class -> ID
│   ├── id2label.json         # Mapping từ ID -> tên class
│   ├── bacterial_leaf_blight/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── brown_spot/
│   ├── healthy/
│   └── leaf_blast/
└── scripts/
    └── train_efficientnet_lite0_ca.py
```

### File `metadata.csv`

```csv
path,label,split
data/bacterial_leaf_blight/0.jpg,bacterial_leaf_blight,train
data/bacterial_leaf_blight/1.jpg,bacterial_leaf_blight,train
data/brown_spot/100.jpg,brown_spot,valid
data/healthy/200.jpg,healthy,test
...
```

**Các cột bắt buộc:**

- `path`: Đường dẫn tương đối đến ảnh
- `label`: Tên class (string)
- `split`: train/valid/test

### File `label2id.json`

```json
{
  "bacterial_leaf_blight": 0,
  "brown_spot": 1,
  "healthy": 2,
  "leaf_blast": 3
}
```

### Tạo Metadata (Nếu Chưa Có)

```bash
python scripts/prepare_metadata.py \
    --data-dir data \
    --output-dir data \
    --train-ratio 0.7 \
    --valid-ratio 0.15 \
    --test-ratio 0.15
```

---

## 🏃 Training

### Quick Start (Mặc Định)

```bash
python scripts/train_efficientnet_lite0_ca.py
```

Lệnh này sẽ:

- Dùng pretrained weights từ ImageNet
- Image size = 224
- Batch size = 64
- Epochs = 30
- Early stopping patience = 7
- Base LR = 1e-4, Head LR = 1e-3

### Training Đầy Đủ (Khuyến Nghị)

```bash
python scripts/train_efficientnet_lite0_ca.py \
    --metadata data/metadata.csv \
    --label2id data/label2id.json \
    --output-dir results \
    --image-size 224 \
    --batch-size 64 \
    --epochs 30 \
    --patience 7 \
    --base-lr 1e-4 \
    --head-lr 1e-3 \
    --weight-decay 1e-2 \
    --reduction 16 \
    --dropout 0.2 \
    --pretrained \
    --num-workers 4 \
    --pin-memory \
    --seed 42
```

### Training Nhanh (Debug/Test)

```bash
python scripts/train_efficientnet_lite0_ca.py \
    --train-limit 500 \
    --valid-limit 100 \
    --epochs 5 \
    --batch-size 32
```

### Training Từ Scratch (Không Pretrained)

```bash
python scripts/train_efficientnet_lite0_ca.py \
    --no-pretrained \
    --epochs 50 \
    --base-lr 1e-3 \
    --patience 10
```

### Training Trên CPU

```bash
python scripts/train_efficientnet_lite0_ca.py \
    --device cpu \
    --batch-size 32 \
    --num-workers 2
```

### Training Với Image Size Lớn Hơn

```bash
python scripts/train_efficientnet_lite0_ca.py \
    --image-size 256 \
    --batch-size 32  # Giảm batch size vì ảnh lớn hơn
```

---

## ⚙️ Các Tham Số Quan Trọng

### 1. Dữ Liệu

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--metadata` | `data/metadata.csv` | Đường dẫn file metadata |
| `--label2id` | `data/label2id.json` | Đường dẫn label mapping |
| `--output-dir` | `results` | Thư mục lưu kết quả |

### 2. Model Architecture

| Tham số | Mặc định | Ý nghĩa | Khuyến nghị |
|---------|----------|---------|-------------|
| `--reduction` | 16 | Reduction ratio của CA block | 16 (balanced), 8 (more params), 32 (lighter) |
| `--dropout` | 0.2 | Dropout rate trước classifier | 0.2-0.3 |
| `--pretrained` | True | Dùng ImageNet weights | Luôn bật (trừ khi train from scratch) |

### 3. Training Hyperparameters

| Tham số | Mặc định | Ý nghĩa | Khuyến nghị |
|---------|----------|---------|-------------|
| `--image-size` | 224 | Kích thước ảnh đầu vào | 224 (nhanh), 256 (chính xác hơn) |
| `--batch-size` | 64 | Batch size | GPU: 64-128, CPU: 16-32 |
| `--epochs` | 30 | Số epoch | 30-50 (pretrained), 80-100 (from scratch) |
| `--patience` | 7 | Early stopping patience | 7-10 |

### 4. Learning Rates

| Tham số | Mặc định | Ý nghĩa | Khuyến nghị |
|---------|----------|---------|-------------|
| `--base-lr` | 1e-4 | LR cho backbone | Pretrained: 1e-4 đến 5e-4, Scratch: 1e-3 |
| `--head-lr` | 1e-3 | LR cho classifier | 5x-10x base-lr |
| `--weight-decay` | 1e-2 | L2 regularization | 1e-2 đến 5e-2 |

### 5. DataLoader

| Tham số | Mặc định | Ý nghĩa | Khuyến nghị |
|---------|----------|---------|-------------|
| `--num-workers` | 4 | Số workers load dữ liệu | CPU cores - 2 |
| `--pin-memory` | False | Pin memory cho GPU | Bật khi dùng GPU |

### 6. Debug/Test

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--train-limit` | None | Giới hạn samples train (debug) |
| `--valid-limit` | None | Giới hạn samples valid |
| `--device` | Auto | Force CPU/CUDA |
| `--seed` | 42 | Random seed |

---

## 📊 Kết Quả Mong Đợi

### Sau Khi Training

Script sẽ tạo thư mục kết quả:

```
results/EfficientNet_Lite0_CA_07_10_2025_1430/
├── EfficientNet_Lite0_CA_07_10_2025_1430_best.pt  # Checkpoint tốt nhất
├── history.json                                    # Training history
├── metrics.json                                    # Final metrics
└── training_plot.png                               # Loss/accuracy curves
```

### File `metrics.json`

```json
{
  "model_name": "EfficientNet_Lite0_CA_07_10_2025_1430",
  "size_mb": 19.2,
  "valid_acc": 0.9145,
  "valid_f1": 0.9138,
  "fps": 612.3,
  "num_params": 4723456,
  "ckpt_path": "EfficientNet_Lite0_CA_07_10_2025_1430_best.pt"
}
```

### Metrics Chi Tiết

| Metric | Giá trị Mong Đợi |
|--------|------------------|
| **Validation Accuracy** | 90-92% |
| **Validation F1-Score** | 0.90-0.92 |
| **Training Time** | ~30-45 phút (GPU), ~4-6 giờ (CPU) |
| **FPS (Inference)** | 600+ (GPU), 80+ (CPU) |
| **Model Size** | ~19 MB |

### Training Curves

Một training thành công nên có:

- **Train loss:** Giảm đều, không fluctuate quá nhiều
- **Valid loss:** Giảm theo train loss, không tăng sớm (overfitting)
- **Train accuracy:** Tăng dần lên 95%+
- **Valid accuracy:** Tăng dần lên 90-92%

---

## 🐛 Troubleshooting

### 1. Out of Memory (OOM)

**Triệu chứng:**

```
RuntimeError: CUDA out of memory
```

**Giải pháp:**

```bash
# Giảm batch size
python scripts/train_efficientnet_lite0_ca.py --batch-size 32

# Hoặc giảm image size
python scripts/train_efficientnet_lite0_ca.py --image-size 192 --batch-size 48

# Hoặc tắt pin-memory
python scripts/train_efficientnet_lite0_ca.py --batch-size 32  # Không dùng --pin-memory
```

### 2. DataLoader Chậm

**Triệu chứng:** Training rất chậm, GPU utilization thấp

**Giải pháp:**

```bash
# Tăng num-workers
python scripts/train_efficientnet_lite0_ca.py --num-workers 8 --pin-memory

# Kiểm tra disk I/O (nên dùng SSD)
```

### 3. Model Không Converge

**Triệu chứng:** Loss không giảm hoặc accuracy quá thấp

**Giải pháp:**

```bash
# Tăng learning rate
python scripts/train_efficientnet_lite0_ca.py --base-lr 5e-4 --head-lr 2e-3

# Kiểm tra dữ liệu có đúng không
python -c "from src.utils.data import build_datasets; datasets = build_datasets(...); print(len(datasets['train']))"

# Dùng pretrained weights
python scripts/train_efficientnet_lite0_ca.py --pretrained
```

### 4. Overfitting

**Triệu chứng:** Train acc >> Valid acc, valid loss tăng sớm

**Giải pháp:**

```bash
# Tăng dropout
python scripts/train_efficientnet_lite0_ca.py --dropout 0.3

# Tăng weight decay
python scripts/train_efficientnet_lite0_ca.py --weight-decay 5e-2

# Tăng data augmentation (cần sửa transforms)
```

### 5. Training Quá Chậm Trên CPU

**Giải pháp:**

```bash
# Giảm batch size và workers
python scripts/train_efficientnet_lite0_ca.py \
    --device cpu \
    --batch-size 16 \
    --num-workers 2

# Hoặc dùng Google Colab/Kaggle với GPU miễn phí
```

### 6. Import Error

**Triệu chứng:**

```
ModuleNotFoundError: No module named 'src'
```

**Giải pháp:**

```bash
# Chạy từ thư mục root
cd Paddy-Disease-Classification-final
python scripts/train_efficientnet_lite0_ca.py

# Hoặc set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

---

## 💡 Tips & Tricks

### 1. Tối Ưu Tốc Độ Training

```bash
# GPU với batch size lớn
python scripts/train_efficientnet_lite0_ca.py \
    --batch-size 128 \
    --num-workers 8 \
    --pin-memory

# Dùng mixed precision (đã có sẵn trong code)
# EfficientNet-Lite0 nhẹ nên có thể dùng batch rất lớn
```

### 2. Transfer Learning Tốt Hơn

```bash
# Freeze backbone vài epoch đầu, rồi unfreeze
# (Cần custom code, chưa có sẵn trong script)

# Hoặc dùng differential LR lớn hơn
python scripts/train_efficientnet_lite0_ca.py \
    --base-lr 1e-5 \
    --head-lr 1e-3  # 100x head LR
```

### 3. Tăng Accuracy

```bash
# Tăng image size (trade-off: chậm hơn)
python scripts/train_efficientnet_lite0_ca.py --image-size 256 --batch-size 32

# Giảm reduction (nhiều params hơn)
python scripts/train_efficientnet_lite0_ca.py --reduction 8

# Train lâu hơn
python scripts/train_efficientnet_lite0_ca.py --epochs 50 --patience 15
```

### 4. Ensemble Models

```bash
# Train nhiều models với seeds khác nhau
for seed in 42 123 456 789; do
    python scripts/train_efficientnet_lite0_ca.py \
        --seed $seed \
        --model-name EfficientNet_Lite0_CA_seed${seed}
done

# Rồi ensemble predictions (cần custom code)
```

### 5. Export Cho Production

Sau khi train xong, export model:

```python
import torch
from src.models.backbones import EfficientNet_Lite0_CA

# Load checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("results/.../model_best.pt", map_location=device)

# Load vào model
model = EfficientNet_Lite0_CA(num_classes=4).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# Export sang TorchScript (để deploy)
example_input = torch.randn(1, 3, 224, 224).to(device)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("efficientnet_lite0_ca.pt")

# Hoặc export sang ONNX (cho mobile/edge)
torch.onnx.export(
    model,
    example_input,
    "efficientnet_lite0_ca.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
)
```

### 6. Test Trên External Dataset

```bash
# Dùng script test (nếu có)
python test_on_external_dataset.py \
    --model-path results/.../model_best.pt \
    --test-dir path/to/external/data \
    --output test_results/
```

### 7. Benchmark Inference Speed

```python
from src.utils.metrics import fps

# Đo FPS
fps_value = fps(
    model=model,
    batch_size=16,
    image_size=224,
    steps=100,
    warmup=20
)
print(f"FPS: {fps_value:.1f}")
```

---

## 📚 Tài Liệu Tham Khảo

### Papers

- **EfficientNet:** [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- **EfficientNetV2:** [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
- **Coordinate Attention:** [Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907)

### Code References

- **timm (PyTorch Image Models):** <https://github.com/huggingface/pytorch-image-models>
- **EfficientNet-Lite:** <https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite>

### Dataset

- **Paddy Doctor:** <https://www.kaggle.com/competitions/paddy-disease-classification>

---

## 📞 Liên Hệ & Support

Nếu gặp vấn đề hoặc có câu hỏi:

1. **GitHub Issues:** [Create an issue](https://github.com/tontide1/Rice-leaf-diseases-classification/issues)
2. **Email:** <tontide1@example.com>
3. **Kaggle:** <https://www.kaggle.com/tontide1>

---

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

---

**Chúc bạn training thành công! 🎉**
