# Hướng Dẫn Training MobileNetV3-Small Vanilla (Baseline CNN)

## Giới Thiệu

File `train_MobileNetV3_Small_Vanilla.py` được sử dụng để huấn luyện **baseline CNN model** - MobileNetV3-Small **KHÔNG CÓ** attention mechanism trên tập dữ liệu Paddy Disease Classification.

**Mục đích của model này:**

- 🎯 **Baseline để so sánh**: Đánh giá contribution của các attention mechanisms
- 📊 **Pure CNN performance**: Hiệu suất của CNN thuần túy không có attention
- ⚡ **Speed benchmark**: Đo tốc độ inference của backbone gốc
- 🔬 **Scientific comparison**: So sánh công bằng với 8 attention-enhanced models

**Đặc điểm MobileNetV3-Small Vanilla:**

- **Không có attention mechanism**: Pure depthwise separable convolutions
- **Lightest**: Ít parameters nhất (chỉ có backbone + classifier)
- **Fastest training**: Không có overhead từ attention computation
- **Baseline accuracy**: Đo accuracy ceiling mà attention có thể improve

## Tại Sao Cần Baseline Model?

### **1. Đánh Giá Contribution của Attention**

```python
# Nếu không có baseline:
"BoT model đạt 99.3% accuracy"  
→ Không biết bao nhiêu % là do backbone, bao nhiêu % là do attention?

# Với baseline:
Vanilla: 98.5% accuracy  ← Backbone contribution
BoT:     99.3% accuracy  
         ─────
         +0.8% ← Attention contribution! 

→ Rõ ràng attention giúp tăng 0.8%!
```

---

### **2. So Sánh Cost vs Benefit**

| Model | Params | Time (ms) | Accuracy | Cost | Benefit |
|-------|--------|-----------|----------|------|---------|
| **Vanilla** | **1.53M** | **~5.0** | **98.5%** | **Baseline** | **Baseline** |
| BoT | 1.75M | 6.13 | 99.3% | +220K params, +23% time | +0.8% acc |
| ECA | 1.86M | 6.37 | 99.1% | +330K params, +27% time | +0.6% acc |
| Hybrid | 2.15M | 26.01 | 99.4% | +620K params, +420% time | +0.9% acc |

**→ Vanilla cho thấy: Hybrid tăng 0.9% accuracy nhưng chậm hơn 5.2x!**

---

### **3. Scientific Rigor**

Trong research, luôn cần **ablation study**:

```
Hypothesis: "Attention mechanisms improve accuracy"

Experiment:
├─ Control group:  Vanilla CNN (no attention)
└─ Test groups:    BoT, CA, ECA, Hybrid, ... (with attention)

Result: 
If attention models >> vanilla → Hypothesis confirmed! ✅
If attention models ≈ vanilla → Attention không có ích ❌
```

---

## Yêu Cầu

### 1. Cấu Trúc Thư Mục

```text
Paddy-Disease-Classification-final/
├── data/
│   ├── metadata.csv
│   ├── label2id.json
│   └── images/
├── train_MobileNetV3_Small_Vanilla.py
├── src/
│   ├── models/
│   │   └── backbones/
│   │       └── mobilenet.py
│   ├── training/
│   └── utils/
└── results/
```

### 2. Dữ Liệu Cần Thiết

- `data/metadata.csv`: File chứa thông tin ảnh và nhãn
- `data/label2id.json`: File mapping từ tên nhãn sang ID
- `data/images/`: Thư mục chứa ảnh training/validation

## Cách Sử Dụng

### 1. Training Cơ Bản

```bash
python train_MobileNetV3_Small_Vanilla.py
```

### 2. Training với Tham Số Tùy Chỉnh

```bash
python train_MobileNetV3_Small_Vanilla.py \
    --epochs 30 \
    --batch-size 64 \
    --base-lr 5e-5 \
    --head-lr 5e-4 \
    --pretrained
```

### 3. Quick Test

```bash
python train_MobileNetV3_Small_Vanilla.py \
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
| `--dropout` | 0.1 | Dropout rate trước classifier |
| `--pretrained` | False | Sử dụng pretrained weights từ ImageNet |

### Training Configuration

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--epochs` | 30 | Số epoch training |
| `--batch-size` | 32 | Batch size |
| `--patience` | 10 | Early stopping patience |
| `--base-lr` | 5e-5 | Learning rate cho backbone |
| `--head-lr` | 5e-4 | Learning rate cho classifier head |
| `--weight-decay` | 1e-2 | Weight decay cho optimizer |

### DataLoader Configuration

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--num-workers` | 4 | Số workers cho DataLoader |
| `--pin-memory` | False | Pin memory trong DataLoader |

### Visualization & Logging

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--plot` | False | Hiển thị biểu đồ training history |
| `--save-history` | False | Lưu history JSON + metrics JSON + plot PNG |

## Ví Dụ Thực Tế

### 1. Training Baseline (Khuyến Nghị)

```bash
python train_MobileNetV3_Small_Vanilla.py \
    --epochs 30 \
    --batch-size 64 \
    --base-lr 5e-5 \
    --head-lr 5e-4 \
    --weight-decay 1e-2 \
    --patience 10 \
    --pretrained \
    --num-workers 4 \
    --pin-memory \
    --image-size 224 \
    --save-history
```

**Kết quả mong đợi:**

- Validation Accuracy: ~98.0-98.5%
- Training time: ~25-35 phút (GPU GTX 1660 SUPER)
- Model size: ~5.84 MB (smallest!)
- **Inference: ~5.0 ms/batch** ⚡ **FASTEST POSSIBLE**

---

### 2. Training Tất Cả Models Để So Sánh

```bash
# Step 1: Train Baseline (QUAN TRỌNG!)
python train_MobileNetV3_Small_Vanilla.py \
    --epochs 30 --batch-size 64 --pretrained \
    --num-workers 4 --image-size 224 --save-history

# Step 2: Train Attention Models
python train_MobileNetV3_Small_BoT.py \
    --epochs 30 --batch-size 64 --pretrained \
    --num-workers 4 --image-size 224 --save-history

python train_MobileNetV3_Small_CA.py \
    --epochs 30 --batch-size 64 --pretrained \
    --num-workers 4 --image-size 224 --save-history

python train_MobileNetV3_Small_ECA.py \
    --epochs 30 --batch-size 64 --pretrained \
    --num-workers 4 --image-size 224 --save-history

# Step 3: So sánh results/
```

---

### 3. Parallel Training (3 Models Đồng Thời)

```bash
# Terminal 1: Vanilla Baseline
python train_MobileNetV3_Small_Vanilla.py \
    --epochs 30 --batch-size 64 --pretrained \
    --num-workers 3 --image-size 224 --save-history &

# Terminal 2: ECA (Fastest attention)
python train_MobileNetV3_Small_ECA.py \
    --epochs 30 --batch-size 64 --pretrained \
    --num-workers 3 --image-size 224 --save-history &

# Terminal 3: BoT (Best accuracy)
python train_MobileNetV3_Small_BoT.py \
    --epochs 30 --batch-size 48 --pretrained \
    --num-workers 3 --image-size 224 --save-history &

wait
```

---

## Đầu Ra

### 1. Model Checkpoint

- `MobileNetV3_Small_Vanilla_best.pt` - Best model

### 2. Results Files

```text
results/
└── MobileNetV3_Small_Vanilla_06_10_2025_1600/
    ├── history.json
    ├── metrics.json
    └── training_plot.png
```

### 3. Expected Results

Với cấu hình tối ưu (pretrained, 30 epochs, batch_size=64):

- **Validation Accuracy**: ~98.0-98.5%
- **F1-Score**: ~98.0-98.5%
- **Inference Speed**: ~4,900 FPS (batch=16) ⚡ **FASTEST**
- **Model Size**: ~5.84 MB **SMALLEST**
- **Parameters**: ~1.53M **LIGHTEST**
- **Training Time**: ~25-35 phút (GTX 1660 SUPER)

---

## So Sánh với Tất Cả Models

### **Bảng So Sánh Đầy Đủ (9 Models)**

| # | Model | Params | Size (MB) | Time (ms) | Accuracy | Gain vs Vanilla | Cost vs Vanilla |
|---|-------|--------|-----------|-----------|----------|-----------------|-----------------|
| **0** | **Vanilla** | **1.53M** | **5.84** | **~5.0** ⚡ | **~98.5%** | **Baseline** | **Baseline** |
| 1 | BoT | 1.75M | 6.69 | 6.13 | ~99.3% | **+0.8%** | +220K params, +23% time |
| 2 | BoT_Linear | 1.75M | 6.69 | 6.98 | ~99.2% | **+0.7%** | +220K params, +40% time |
| 3 | CA | 1.92M | 7.32 | 8.81 | ~99.2% | **+0.7%** | +390K params, +76% time |
| 4 | ECA | 1.86M | 7.08 | 6.37 | ~99.1% | **+0.6%** | +330K params, +27% time |
| 5 | Hybrid | 2.15M | 8.19 | 26.01 | ~99.4% | **+0.9%** | +620K params, +420% time |
| 6 | MobileViT_XXS | 0.95M | 3.64 | 19.48 | ~99.0% | +0.5% | -580K params, +290% time |
| 7 | ResNet18_BoT | 11.36M | 43.34 | 14.97 | ~99.3% | +0.8% | +9.83M params, +199% time |
| 8 | ResNet18_BoTLinear | 11.36M | 43.34 | 14.93 | ~99.3% | +0.8% | +9.83M params, +199% time |

---

### **Insights từ So Sánh**

#### **1. Accuracy Improvement Analysis**

```text
Attention Mechanisms Contribution:
├─ Best:     Hybrid (+0.9%)    → Worth it nếu không care speed
├─ Good:     BoT (+0.8%)       → Best balance (accuracy vs speed)
├─ OK:       CA, Linear (+0.7%) → Decent improvement
└─ Minimal:  ECA (+0.6%)       → Fastest, acceptable tradeoff
```

#### **2. Speed Analysis**

```text
Vanilla:  5.0 ms   ← BASELINE (fastest possible)
ECA:      6.37 ms  (+27%)  ← Minimal overhead
BoT:      6.13 ms  (+23%)  ← Surprisingly fast!
CA:       8.81 ms  (+76%)  ← Moderate overhead
Hybrid:   26.01 ms (+420%) ← Heavy computation
```

**→ BoT và ECA có best speed-accuracy tradeoff!**

#### **3. Cost-Benefit Analysis**

**Most Efficient (ROI):**

1. **BoT**: +0.8% accuracy, +23% time → **0.035% gain per 1% time**
2. **ECA**: +0.6% accuracy, +27% time → **0.022% gain per 1% time**
3. **Linear**: +0.7% accuracy, +40% time → **0.018% gain per 1% time**

**Least Efficient:**

- **Hybrid**: +0.9% accuracy, +420% time → **0.002% gain per 1% time**

---

## Khi Nào Chọn Vanilla?

### **✅ CHỌN VANILLA KHI:**

1. **Extreme Speed Required**
   - Real-time video processing (>200 FPS)
   - Embedded systems với limited compute
   - Battery-powered devices

2. **Baseline Experiment**
   - Research paper ablation study
   - Evaluating attention contributions
   - Scientific comparison

3. **98.5% Accuracy Đủ**
   - Task không cần perfect accuracy
   - Cost-sensitive deployment
   - Large-scale serving với tight latency budget

4. **Training Resources Limited**
   - Nhanh nhất để train (25-35 phút)
   - Ít VRAM nhất
   - Dễ debug nhất

---

### **❌ KHÔNG CHỌN VANILLA KHI:**

1. **Accuracy Critical** → Dùng Hybrid hoặc BoT
2. **Có Đủ Resources** → Dùng attention models
3. **Cần Interpretability** → Attention maps giúp explain
4. **Research Contribution** → Vanilla quá basic

---

## Kiến Trúc Model

### MobileNetV3-Small Vanilla

```text
Input (3x224x224)
    ↓
[MobileNetV3-Small Backbone] (pretrained on ImageNet)
    │
    ├─ Initial Conv (16 channels)
    ├─ Inverted Residual Blocks (MBConv)
    │   ├─ Depthwise Separable Convolutions
    │   ├─ Squeeze-Excitation (built-in SE modules)
    │   └─ Hard-Swish activations
    └─ Final Conv (576 channels)
    ↓
[AdaptiveAvgPool2d] → (B, 576, 1, 1)
    ↓
[Dropout 0.1]
    ↓
[Linear Classifier] → (B, num_classes)
```

**Đặc điểm:**

- **Pure CNN**: Chỉ có convolutions, không có attention
- **Depthwise Separable**: Efficient convolutions
- **Built-in SE**: MobileNetV3 có SE modules (nhưng khác với external attention)
- **Lightest**: 1.53M params
- **Fastest**: ~5.0 ms inference

---

## Tips & Best Practices

### 1. Training Strategy

```bash
# Luôn train Vanilla TRƯỚC để có baseline!
python train_MobileNetV3_Small_Vanilla.py --pretrained --save-history

# Sau đó train attention models
# So sánh với baseline để evaluate attention contribution
```

### 2. Hyperparameters

**Vanilla dùng CÙNG hyperparameters với attention models:**

- `--base-lr 5e-5`, `--head-lr 5e-4`
- `--batch-size 64`
- `--weight-decay 1e-2`

**→ Đảm bảo fair comparison!**

### 3. Expected Performance

```python
# Typical results:
Vanilla: 98.0-98.5% accuracy
BoT:     99.2-99.4% accuracy
         ───────────
         +0.7-0.9% improvement from attention

# If attention models < 98.5%:
→ Something wrong with attention implementation!

# If attention models >> 99.5%:
→ Possible overfitting, check regularization
```

---

## Troubleshooting

### Vanilla Accuracy Quá Thấp (<97%)

**Nguyên nhân:**

- Không dùng pretrained weights
- Learning rate quá cao/thấp
- Data augmentation quá mạnh

**Giải pháp:**

```bash
# Đảm bảo dùng pretrained
python train_MobileNetV3_Small_Vanilla.py --pretrained

# Kiểm tra learning rate
python train_MobileNetV3_Small_Vanilla.py \
    --base-lr 5e-5 \
    --head-lr 5e-4
```

---

### Attention Models Không Tốt Hơn Vanilla

**Nguyên nhân:**

- Attention modules có bug
- Overfitting trên attention
- Learning rate không phù hợp cho attention

**Giải pháp:**

```bash
# Test gradient flow
python test_all_models.ipynb  # Run gradient flow tests

# Tăng regularization cho attention models
python train_MobileNetV3_Small_BoT.py \
    --dropout 0.2 \
    --weight-decay 2e-2
```

---

## Benchmark Script

Tạo script để so sánh tất cả models:

```bash
#!/bin/bash
# compare_all_models.sh

echo "="*80
echo "TRAINING ALL 9 MODELS FOR COMPARISON"
echo "="*80

# 1. Baseline
echo "[1/9] Training Vanilla Baseline..."
python train_MobileNetV3_Small_Vanilla.py \
    --epochs 30 --batch-size 64 --pretrained \
    --num-workers 4 --image-size 224 --save-history

# 2-9. Attention Models
for model in BoT BoT_Linear CA ECA Hybrid; do
    echo "[Training MobileNetV3_Small_$model..."
    python train_MobileNetV3_Small_$model.py \
        --epochs 30 --batch-size 64 --pretrained \
        --num-workers 4 --image-size 224 --save-history
done

echo "="*80
echo "TRAINING COMPLETED! Check results/ directory"
echo "="*80

# Compare results
python scripts/compare_results.py
```

---

## Tham Khảo

- Model Implementation: `src/models/backbones/mobilenet.py` (class `MobileNetV3_Small_Vanilla`)
- Training Logic: `src/training/train.py`
- MobileNetV3 Paper: "Searching for MobileNetV3" (ICCV 2019)
- Baseline: Pure CNN without external attention mechanisms

---

## License

MIT License
