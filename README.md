# 🌾 Paddy Disease Classification — Dual Language Overview

## 🇬🇧 English Version

### 🔥 Marketing Highlights

- **Top accuracy 98.56%** with `MobileNetV3_Small_ECA`, reaching ~**1.0K fps** for real-time applications.
- **Rich model zoo**: ResNet18 (BoT, Hybrid, Linear, CA, ECA, MultiScale), EfficientNetV2-S, MobileNetV3, and ultra-light custom CNNs for edge deployments.
- **Optimized training stack**: mixed precision, differential learning rates, early stopping, automated FPS benchmarking.
- **Full data pipeline**: scripts for metadata normalization, label mapping, and rice-leaf-specific augmentations.
- **Insightful analytics**: training curves, confusion matrices, and per-class reports for both internal and external datasets.
- **Deployment ready**: straightforward ONNX/TensorRT export paths, plus CLI for quick external evaluation.

### 📦 Project Structure

```text
Paddy-Disease-Classification-final/
├── src/                  # Core code (models, training loop, utilities)
├── scripts/              # CLI utilities: data prep, testing, benchmarking
├── data/                 # metadata.csv, label2id.json, id2label.json
├── results/              # training history, metrics, plots, checkpoints
├── models/               # pre-trained checkpoints
├── imgs/                 # marketing visuals for documentation
└── requirements.txt      # aligned Python dependencies
```

### 🧠 Architecture & Key Features

**Production focus:**

- `src/training/train.py` leverages `torch.amp` with GradScaler, logs accuracy/F1 automatically, stores the best checkpoint, and measures FPS.
- `src/utils/data/transforms.py` fine-tunes augmentations (square padding, resize, color jitter, light rotations) for rice leaf imagery.
- `src/utils/metrics/benchmark.py` centralizes `evaluate`, `fps`, and `predict` for both training and inference scenarios.

**Innovative model zoo:**

- **ResNet18 family**: blends BoT Attention, Linear Attention, Coordinate Attention, and Efficient Channel Attention into Hybrid variants.
- **MobileNetV3 family**: BoT, CA, Hybrid, and ECA adaptations deliver high accuracy with outstanding throughput.
- **EfficientNetV2-S variants**: vanilla, CA, ECA, BoTLinear, Hybrid to suit varying deployment goals.
- **Custom CNNs** (`LightweightCNN`, `TinyPaddyNet`, `CompactCNN`): under 1.8M parameters, ideal for edge/IoT devices.

**Robust data handling:**

- `scripts/prepare_metadata.py` scans datasets into stratified metadata (`path`, `label`, `split`).
- `scripts/generate_label_map.py` builds UTF-8 `label2id.json` and `id2label.json` for seamless training/inference.
- Reference internal dataset: ~12.2K training / 1.5K validation images (based on ResNet18 training logs), easily expandable by re-running the pipeline.

### 📊 Performance Highlights

| Model | Valid ACC | Valid F1 | FPS (T4) | Params | Best Fit |
|-------|-----------|----------|----------|--------|----------|
| **MobileNetV3_Small_ECA** | **98.56%** | **98.54%** | 1,011 | 1.85M | Production realtime |
| ResNet18_BoT_large_batch | 97.32% | 97.32% | 1,070 | 11.36M | High-accuracy, fine-tuning |
| ResNet18_BoTLinear_pretrained | 96.47% | 96.46% | 1,119 | 11.35M | Speed/accuracy balance |
| ResNet18_Hybrid | 95.68% | 95.69% | 792 | 11.65M | Attention-rich F1 |
| EfficientNet_Lite0_CA | 95.27% | 95.24% | 865 | 4.4M | Mid-tier edge devices |

> 🔍 Explore `results/*/metrics.json` and `history.json` for per-epoch loss, accuracy, F1, and the best checkpoint of every run.

### ⚙️ Quickstart

1. **Environment setup**

   ```bash
   git clone https://github.com/<your-account>/Paddy-Disease-Classification-final.git
   cd Paddy-Disease-Classification-final
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Data preparation**

   ```bash
   python scripts/prepare_metadata.py \
     --data-root /absolute/path/to/paddy-dataset \
     --output data/metadata.csv

   python scripts/generate_label_map.py
   ```

   Recommended directory structure:

   ```text
   dataset/
   ├── BacterialLeafBlight/
   ├── BrownSpot/
   ├── LeafSmut/
   └── Healthy/
   ```

3. **Train a model** (example: ResNet18 BoTLinear)

   ```bash
   python train_model_with_doc/resnet18/train_ResNet18_BoTLinear.py \
     --pretrained \
     --epochs 25 \
     --batch-size 32 \
     --base-lr 5e-5 \
     --head-lr 5e-4 \
     --heads 4 \
     --dropout 0.15 \
     --save-history
   ```

   Prefer an edge-ready backbone? Import `train_model` from `src/training/train.py` and spin up a short script around `TinyPaddyNet`.
4. **Evaluate on an external dataset**

   ```bash
   python scripts/test_on_external_dataset.py \
     --model-path "models/train 1/MobileNetV3_Small_BoT_best.pt" \
     --test-dir /absolute/path/to/external_dataset \
     --model-type BoT_Linear \
     --batch-size 32 \
     --save-predictions
   ```

   The CLI exports confusion matrices, classification reports, and JSON metrics under `test_results/`.

### 📬 Contact & Credits

- Author: **Tontide1**
- Data sources: Paddy Leaf Diseases datasets (Kaggle) plus internal collections.

**🌟 Don't forget to star (⭐) this Repo and include it in your portfolio to show off your Vision AI capabilities!**

## 🇬🇧 Vietnamese Version

### 🔥 Điểm nhấn

- **Accuracy đỉnh 98.56%** với `MobileNetV3_Small_ECA`, đạt ~**1.0K fps** trên GPU T4 — phù hợp sản phẩm real-time.
- **Model Zoo phong phú**: ResNet18 (BoT, Hybrid, Linear, CA, ECA, MultiScale), EfficientNetV2-S, MobileNetV3, Custom CNN siêu nhẹ cho edge.
- **Pipeline huấn luyện tối ưu**: Mixed precision, differential learning rate, early stopping, benchmark FPS tự động.
- **Chuỗi dữ liệu khép kín**: scripts chuẩn hóa metadata, tạo label map, augmentation chuyên biệt cho lá lúa.
- **Bộ kết quả trực quan**: lịch sử loss/accuracy, confusion matrix, classification report cho cả tập nội bộ lẫn external.
- **Sẵn sàng triển khai**: dễ dàng xuất sang ONNX/TensorRT, hỗ trợ test nhanh trên dataset ngoài.

### 📦 Cấu trúc dự án

```text
Paddy-Disease-Classification-final/
├── src/                  # Code lõi (models, training, utils)
├── scripts/              # CLI scripts: chuẩn hóa dữ liệu, test, benchmark
├── data/                 # metadata.csv, label2id.json, id2label.json
├── results/              # lịch sử training, metrics, plots, checkpoints
├── models/               # checkpoints đã huấn luyện
├── imgs/                 # hình minh họa dùng cho README
└── requirements.txt      # dependencies
```

### 🧠 Kiến trúc & tính năng nổi bật

**Tập trung production:**

- `src/training/train.py` sử dụng `torch.amp` + GradScaler, tự log accuracy/F1, lưu checkpoint tốt nhất, đo FPS tự động.
- `src/utils/data/transforms.py` tối ưu augmentation: pad vuông, resize, jitter màu, xoay nhẹ — phù hợp hình ảnh lá lúa.
- `src/utils/metrics/benchmark.py` cung cấp `evaluate`, `fps`, `predict` tái sử dụng cho training và inference.

**Model Zoo sáng tạo:**

- **ResNet18 series**: kết hợp BoT Attention, Linear Attention, Coordinate Attention, Efficient Channel Attention, kiến trúc Hybrid.
- **MobileNetV3 series**: các biến thể BoT, CA, Hybrid, ECA duy trì hiệu năng cao với tốc độ vượt trội.
- **EfficientNetV2-S**: các phiên bản vanilla, CA, ECA, BoTLinear, Hybrid tùy mục tiêu.
- **Custom CNN** (`LightweightCNN`, `TinyPaddyNet`, `CompactCNN`): <1.8M tham số, tối ưu cho thiết bị edge/IoT.

**Dữ liệu & tiền xử lý chuẩn chỉnh:**

- `scripts/prepare_metadata.py` quét dataset → tạo metadata có `path`, `label`, `split` (train/valid/test) với stratify.
- `scripts/generate_label_map.py` sinh `label2id.json` và `id2label.json` UTF-8 phục vụ training/inference.
- Dataset nội bộ tham chiếu: ~12.2K ảnh train / 1.5K ảnh valid (trích từ log huấn luyện ResNet18), dễ mở rộng khi bổ sung dữ liệu.

### 📊 Kết quả nổi bật

| Model | ACC Valid | F1 Valid | FPS (T4) | Params | Use case |
|-------|-----------|----------|----------|--------|---------|
| **MobileNetV3_Small_ECA** | **98.56%** | **98.54%** | 1,011 | 1.85M | Production realtime |
| ResNet18_BoT_large_batch | 97.32% | 97.32% | 1,070 | 11.36M | Accuracy cao, dễ fine-tune |
| ResNet18_BoTLinear_pretrained | 96.47% | 96.46% | 1,119 | 11.35M | Cân bằng tốc độ & độ chuẩn |
| ResNet18_Hybrid | 95.68% | 95.69% | 792 | 11.65M | Kết hợp attention, F1 nổi trội |
| EfficientNet_Lite0_CA | 95.27% | 95.24% | 865 | 4.4M | Edge device tầm trung |

> 🔍 `results/*/metrics.json` và `history.json` lưu đầy đủ loss, accuracy, F1 theo epoch cùng checkpoint tốt nhất cho mỗi lần huấn luyện.

### ⚙️ Bắt đầu nhanh

1. **Cài môi trường**

   ```bash
   git clone https://github.com/<your-account>/Paddy-Disease-Classification-final.git
   cd Paddy-Disease-Classification-final
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Chuẩn bị dữ liệu**

   ```bash
   python scripts/prepare_metadata.py \
     --data-root /absolute/path/to/paddy-dataset \
     --output data/metadata.csv

   python scripts/generate_label_map.py
   ```

   Cấu trúc thư mục dữ liệu mẫu:

   ```text
   dataset/
   ├── BacterialLeafBlight/
   ├── BrownSpot/
   ├── LeafSmut/
   └── Healthy/
   ```

3. **Huấn luyện model** (ví dụ với ResNet18 BoTLinear)

   ```bash
   python train_model_with_doc/resnet18/train_ResNet18_BoTLinear.py \
     --pretrained \
     --epochs 25 \
     --batch-size 32 \
     --base-lr 5e-5 \
     --head-lr 5e-4 \
     --heads 4 \
     --dropout 0.15 \
     --save-history
   ```

   Muốn thử model gọn nhẹ (TinyPaddyNet), import `train_model` từ `src/training/train.py` và viết script ngắn để huấn luyện.
4. **Đánh giá trên tập external**

   ```bash
   python scripts/test_on_external_dataset.py \
     --model-path "models/train 1/MobileNetV3_Small_BoT_best.pt" \
     --test-dir /absolute/path/to/external_dataset \
     --model-type BoT_Linear \
     --batch-size 32 \
     --save-predictions
   ```

   Script tạo confusion matrix, classification report và JSON metrics trong `test_results/`.

### 📬 Liên hệ & Credits

- Tác giả: **Tontide1**
- Dữ liệu: Paddy Leaf Diseases (Kaggle) kết hợp nguồn nội bộ.

**🌟 Đừng quên gắn sao (⭐) Repo này và đưa vào portfolio để thể hiện năng lực Vision AI của bạn!**
---
