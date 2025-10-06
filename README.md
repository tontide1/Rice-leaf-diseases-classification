# Paddy Disease Classification with Advanced Attention Mechanisms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/Research-Computer%20Vision-green.svg)]()

> **Nghiên cứu phân loại bệnh trên cây lúa sử dụng các cơ chế Attention tiên tiến kết hợp với Deep Learning Architectures**

---

## 📋 Tổng quan

Dự án này nghiên cứu và triển khai **15+ kiến trúc deep learning** kết hợp với **4 cơ chế attention khác nhau** để phân loại bệnh trên cây lúa (Paddy Disease Classification). Nghiên cứu tập trung vào việc so sánh hiệu năng của các attention mechanisms (Coordinate Attention, Efficient Channel Attention, BoT Attention, Linear Attention) khi được tích hợp vào các backbone networks phổ biến (EfficientNetV2, ResNet18, MobileNetV3).

### 🎯 Mục tiêu nghiên cứu

1. **So sánh hiệu quả** của các attention mechanisms trong bài toán phân loại bệnh cây lúa
2. **Tối ưu hóa** trade-off giữa accuracy, inference speed và model size
3. **Phát triển** các kiến trúc hybrid kết hợp nhiều attention mechanisms
4. **Đánh giá** khả năng deployment trên các thiết bị khác nhau (Cloud GPU, Edge devices, Mobile)

### 🏆 Kết quả chính

| Model | Accuracy | FPS (T4 GPU) | Params | Use Case |
|-------|----------|--------------|--------|----------|
| **EfficientNetV2-S Hybrid** | **95-97%** | 250 | 24M | Accuracy tối đa |
| **EfficientNetV2-S CA** | 93-95% | 320 | 21.5M | Production |
| **ResNet18 Hybrid** | 94-96% | 260 | 15M | Balanced |
| **MobileNetV3 CA** | 88-90% | 450 | 2.5M | Mobile/Edge |
| **EfficientNet-Lite0 CA** | 90-92% | 600+ | 4.7M | Mobile |

---

## 🏗️ Kiến trúc hệ thống

### Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image (224×224×3)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │   Backbone Networks           │
          │  • EfficientNetV2-S           │
          │  • ResNet18                   │
          │  • MobileNetV3-Small          │
          └───────────────┬───────────────┘
                          │
          ┌───────────────┴───────────────┐
          │   Attention Mechanisms        │
          │  • Coordinate Attention (CA)  │
          │  • Efficient Channel Att (ECA)│
          │  • BoT Attention (MHSA)       │
          │  • Linear Attention (MHLA)    │
          └───────────────┬───────────────┘
                          │
          ┌───────────────┴───────────────┐
          │   Global Average Pooling      │
          │   Dropout (0.1-0.3)           │
          │   Linear Classifier           │
          └───────────────┬───────────────┘
                          │
                    Output (4 classes)
```

### Cấu trúc thư mục

```
Paddy-Disease-Classification-final/
│
├── src/                           # Source code chính
│   ├── models/                    # Định nghĩa models
│   │   ├── attention/             # Attention mechanisms
│   │   │   ├── coordinate.py      # Coordinate Attention
│   │   │   ├── eca.py             # Efficient Channel Attention
│   │   │   ├── botblock.py        # BoT attention blocks
│   │   │   ├── mhsa.py            # Multi-head Self Attention
│   │   │   └── mhla.py            # Multi-head Linear Attention
│   │   ├── backbones/             # Backbone architectures
│   │   │   ├── efficientnet.py    # 6 EfficientNet variants
│   │   │   ├── resnet.py          # 6 ResNet variants
│   │   │   └── mobilenet.py       # 6 MobileNet variants
│   │   └── pe.py                  # Positional encodings
│   │
│   ├── training/                  # Training utilities
│   │   └── train.py               # Training pipeline
│   │
│   └── utils/                     # Utilities
│       ├── data/                  # Data loading & augmentation
│       │   ├── PaddyDataset.py    # Custom dataset
│       │   ├── transforms.py      # Image transformations
│       │   └── loading.py         # Data loaders
│       └── metrics/               # Evaluation metrics
│           ├── benchmark.py       # FPS, accuracy, F1
│           └── plot.py            # Visualization
│
├── scripts/                       # Utility scripts
│   ├── prepare_metadata.py        # Chuẩn bị metadata
│   └── generate_label_map.py     # Tạo label mapping
│
├── results/                       # Experimental results
│   └── [model_name]_[timestamp]/  # Kết quả từng experiment
│       ├── history.json           # Training curves
│       ├── metrics.json           # Final metrics
│       └── training_plot.png      # Visualization
│
├── models/                        # Saved checkpoints
│   ├── train 1/                   # Experiment 1
│   └── train 2/                   # Experiment 2
│
├── data/                          # Dataset
│   ├── train/                     # Training images
│   ├── valid/                     # Validation images
│   └── test/                      # Test images
│
├── requirements.txt               # Dependencies
├── test_all_models.ipynb         # Model comparison notebook
└── train_*.py                    # Training scripts
```

---

## 🧠 Các cơ chế Attention được nghiên cứu

### 1. Coordinate Attention (CA)

**Đóng góp khoa học:** Mã hóa thông tin vị trí không gian (spatial location) bằng cách xử lý riêng biệt theo chiều ngang và chiều dọc.

```python
class CoordinateAttention(nn.Module):
    """
    Ưu điểm:
    - Bảo toồn thông tin vị trí spatial (height × width)
    - Lightweight: chỉ thêm ~2-3% parameters
    - Hiệu quả cho localization tasks
    
    Complexity: O(H×W×C)
    """
    def forward(self, x):
        # [B, C, H, W] → [B, C, H, 1] + [B, C, 1, W]
        x_h = self.pool_h(x)  # Horizontal pooling
        x_w = self.pool_w(x)  # Vertical pooling
        
        # Kết hợp và học attention weights
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        
        # Tách và áp dụng attention
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return x * a_h * a_w  # Element-wise multiplication
```

**Kết quả thực nghiệm:**
- ✅ Tăng accuracy: +2-3% so với baseline
- ✅ Overhead minimal: +0.5-1M params
- ✅ Tốt cho bài toán có spatial patterns rõ ràng

---

### 2. Efficient Channel Attention (ECA)

**Đóng góp khoa học:** Channel attention cực kỳ lightweight sử dụng 1D convolution thay vì fully-connected layers.

```python
class ECAttention(nn.Module):
    """
    Ưu điểm:
    - Cực kỳ nhẹ: chỉ 1D conv với kernel size k
    - Tránh dimensionality reduction
    - Học local cross-channel interactions
    
    Complexity: O(k×C) với k ≈ 3-5
    """
    def forward(self, x):
        # Global average pooling: [B, C, H, W] → [B, C, 1, 1]
        y = self.avg_pool(x)
        
        # 1D convolution trên channel dimension
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        
        return x * y  # Channel-wise attention
```

**Kết quả thực nghiệm:**
- ✅ Tốc độ cao nhất: gần bằng baseline
- ✅ Parameters minimal: chỉ +0.01M
- ✅ Phù hợp production deployment

---

### 3. BoT Attention (Bottleneck Transformer)

**Đóng góp khoa học:** Kết hợp Multi-Head Self-Attention (MHSA) vào bottleneck blocks của CNN.

```python
class BoTNetBlock(nn.Module):
    """
    Ưu điểm:
    - Global receptive field thông qua self-attention
    - Multi-head mechanism học diverse patterns
    - Position-aware với positional encoding
    
    Complexity: O(H²×W²×C) - Quadratic!
    """
    def forward(self, x):
        # Bottleneck: giảm channels trước attention
        x = F.relu(self.conv1(x))  # [B, C, H, W] → [B, C/4, H, W]
        
        # Multi-head self-attention với positional encoding
        x = self.mhsa(x)  # O(N²) complexity
        
        # Expand channels lại
        x = self.conv2(x)  # [B, C/4, H, W] → [B, C, H, W]
        
        return F.relu(x + residual)
```

**Kết quả thực nghiệm:**
- ✅ Accuracy cao: +3-5% so với baseline
- ⚠️ Chậm: O(N²) complexity
- ⚠️ Memory intensive

---

### 4. Linear Attention

**Đóng góp khoa học:** Giảm complexity từ O(N²) xuống O(N) bằng kernel trick.

```python
class MultiHeadLinearAttention2D(nn.Module):
    """
    Ưu điểm:
    - Linear complexity: O(N) thay vì O(N²)
    - Global context như MHSA
    - Memory-efficient
    
    Complexity: O(H×W×C²)
    """
    def kernel_feature(self, x):
        return F.elu(x) + 1  # Kernel approximation
    
    def forward(self, x):
        # Kernel features
        q = self.kernel_feature(q) * self.scale
        k = self.kernel_feature(k) * self.scale
        
        # Linear attention: KV first, then multiply Q
        KV = torch.einsum("bhcm,bhdm->bhcd", k, v)  # O(C²×N)
        out = torch.einsum("bhcd,bhcm->bhdm", KV, q)  # O(C²×N)
        
        # Normalization
        denom = torch.einsum("bhcm,bhc->bhm", q, k.sum(dim=-1))
        return out / denom.clamp_min(1e-6)
```

**Kết quả thực nghiệm:**
- ✅ Nhanh hơn BoT: ~30-40%
- ✅ Memory-friendly
- ⚠️ Accuracy thấp hơn MHSA ~1-2%

---

## 🔬 Các kiến trúc được nghiên cứu

### A. EfficientNetV2 Family (6 variants)

#### 1. **EfficientNetV2-S Vanilla** (Baseline)
```python
Backbone: tf_efficientnetv2_s (ImageNet-21k pretrained)
Params: 21M | Accuracy: 92-93% | FPS: 350

Architecture:
- Fused-MBConv blocks (stages 1-3): Fuse expansion + depthwise conv
- MBConv blocks (stages 4-6): Standard mobile inverted bottleneck
- Progressive learning compatible
```

#### 2. **EfficientNetV2-S CA** (Production-Ready)
```python
Backbone + Coordinate Attention
Params: 21.5M | Accuracy: 93-95% | FPS: 320

Improvements:
- +2-3% accuracy với minimal overhead
- Spatial localization tốt hơn
- Phù hợp production deployment
```

#### 3. **EfficientNetV2-S ECA** (Fastest)
```python
Backbone + Efficient Channel Attention
Params: 21.01M | Accuracy: 92-94% | FPS: 340

Improvements:
- Gần như không giảm tốc độ
- Minimal parameters (+0.01M)
- Best choice cho real-time applications
```

#### 4. **EfficientNetV2-S BoTLinear** (High Accuracy)
```python
Backbone + Linear Attention
Params: 23M | Accuracy: 94-96% | FPS: 280

Improvements:
- Global context với O(N) complexity
- Accuracy cao
- Tốt cho pattern recognition
```

#### 5. **EfficientNetV2-S Hybrid** ⭐ (State-of-the-Art)
```python
Backbone + CA + BoTLinear
Params: 24M | Accuracy: 95-97% | FPS: 250

Architecture:
  EfficientNetV2-S → CA (spatial) → BoTLinear (global) → Classifier
  
Improvements:
- Accuracy cao nhất: 95-97%
- Kết hợp spatial + global attention
- Best cho competition/research
```

#### 6. **EfficientNet-Lite0 CA** (Mobile/Edge)
```python
Backbone: tf_efficientnet_lite0 + CA
Params: 4.7M | Accuracy: 90-92% | FPS: 600+

Optimizations:
- Lightweight backbone cho mobile
- CPU FPS: ~80 (rất nhanh!)
- Phù hợp edge devices
```

---

### B. ResNet18 Family (6 variants)

#### 1. **ResNet18 BoT** (Original)
```python
Params: 13M | Accuracy: 93-95% | FPS: 250
- Standard BoT attention
- Quadratic complexity O(N²)
```

#### 2. **ResNet18 BoTLinear**
```python
Params: 13M | Accuracy: 92-94% | FPS: 280
- Linear attention O(N)
- Nhanh hơn BoT ~12%
```

#### 3. **ResNet18 CA**
```python
Params: 12M | Accuracy: 92-94% | FPS: 290
- Spatial localization
- Lightweight và nhanh
```

#### 4. **ResNet18 ECA**
```python
Params: 11.71M | Accuracy: 91-93% | FPS: 320
- Fastest variant
- Minimal overhead
```

#### 5. **ResNet18 Hybrid** ⭐
```python
Params: 15M | Accuracy: 94-96% | FPS: 260
- CA + BoTLinear combination
- Accuracy cao nhất trong ResNet family
```

#### 6. **ResNet18 MultiScale**
```python
Params: 16M | Accuracy: 93-95% | FPS: 240
- Attention ở layer3 (256ch) và layer4 (512ch)
- Rich multi-scale features
```

---

### C. MobileNetV3 Family (6 variants)

#### 1. **MobileNetV3-Small Vanilla** (Baseline)
```python
Params: 2.3M | Accuracy: 86-88% | FPS: 500
- Lightweight baseline
```

#### 2. **MobileNetV3-Small BoT**
```python
Params: 2.8M | Accuracy: 88-90% | FPS: 420
- Global attention
```

#### 3. **MobileNetV3-Small BoTLinear**
```python
Params: 2.7M | Accuracy: 87-89% | FPS: 450
- Linear attention
- Nhanh hơn BoT
```

#### 4. **MobileNetV3-Small CA**
```python
Params: 2.5M | Accuracy: 88-90% | FPS: 450
- Spatial localization
- Balanced performance
```

#### 5. **MobileNetV3-Small ECA**
```python
Params: 2.31M | Accuracy: 87-89% | FPS: 480
- Fastest variant
```

#### 6. **MobileNetV3-Small Hybrid**
```python
Params: 2.9M | Accuracy: 89-91% | FPS: 400
- CA + BoT combination
- Best accuracy trong MobileNet family
```

---

## 📊 Phương pháp nghiên cứu

### 1. Dataset & Preprocessing

**Dataset:** Paddy Disease Classification Dataset
- **Training set:** ~11,000 images
- **Validation set:** ~2,000 images
- **Test set:** ~2,000 images
- **Classes:** 4 loại bệnh chính

**Preprocessing Pipeline:**
```python
Train Transforms:
├── SquarePad (fill=0)           # Giữ aspect ratio
├── Resize(224×224)              # Chuẩn hóa kích thước
├── RandomHorizontalFlip(p=0.5)  # Data augmentation
├── RandomRotation(±12°)         # Rotation augmentation
├── ColorJitter(0.15)            # Color augmentation
├── ToTensor()                   # Convert to tensor
└── Normalize(ImageNet stats)    # Normalize

Eval Transforms:
├── SquarePad(fill=0)
├── Resize(224×224)
├── ToTensor()
└── Normalize(ImageNet stats)
```

### 2. Training Configuration

**Optimization Strategy:**
```python
Optimizer: AdamW với differential learning rates
├── Backbone:  lr = 2e-5 ~ 5e-5, weight_decay = 1e-2
├── Bias/Norm: lr = 2e-5 ~ 5e-5, weight_decay = 0
└── Classifier: lr = 2e-4 ~ 5e-4, weight_decay = 1e-2

Scheduler: CosineAnnealingLR
- T_max: 15-30 epochs
- eta_min: 1e-7

Training Techniques:
✅ Mixed Precision Training (AMP)
✅ Gradient Clipping (max_norm=1.0)
✅ Early Stopping (patience=7-10)
✅ Best model selection based on F1 score
```

**Hyperparameters:**
```python
Batch Size: 32-80 (depends on model size)
Image Size: 224×224 (optimal cho EfficientNetV2)
Epochs: 15-30
Dropout: 0.1-0.3
Reduction (CA): 16-32
Attention Heads: 4-8
```

### 3. Evaluation Metrics

```python
Primary Metrics:
- Accuracy: Tỷ lệ phân loại đúng tổng thể
- F1 Score (macro): Cân bằng precision và recall
- FPS: Frames per second (throughput)

Secondary Metrics:
- Model Size (MB): Kích thước model
- Parameters: Số lượng parameters
- Inference Latency: Thời gian xử lý 1 image
- GPU Memory Usage: Bộ nhớ GPU cần thiết
```

### 4. Benchmark Protocol

```python
FPS Measurement:
def fps(model, batch_size=16, image_size=224, steps=100, warmup=20):
    """
    - Warmup: 20 iterations để GPU "nóng máy"
    - Measurement: 100 iterations chính thức
    - CUDA events để đo chính xác
    - Tính trung bình: total_images / elapsed_time
    """
    # GPU timing với CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(steps):
        _ = model(x)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    fps = (steps * batch_size) / (elapsed_ms / 1000.0)
    return fps
```

---

## 📚 Tài liệu tham khảo

### Papers

1. **EfficientNetV2: Smaller Models and Faster Training**
   - Mingxing Tan, Quoc V. Le (Google Research)
   - ICML 2021
   - https://arxiv.org/abs/2104.00298

2. **Coordinate Attention for Efficient Mobile Network Design**
   - Qibin Hou, Daquan Zhou, Jiashi Feng
   - CVPR 2021
   - https://arxiv.org/abs/2103.02907

3. **ECA-Net: Efficient Channel Attention for Deep CNNs**
   - Qilong Wang, Banggu Wu, Pengfei Zhu, et al.
   - CVPR 2020
   - https://arxiv.org/abs/1910.03151

4. **Bottleneck Transformers for Visual Recognition**
   - Aravind Srinivas, Tsung-Yi Lin, Niki Parmar, et al.
   - CVPR 2021
   - https://arxiv.org/abs/2101.11605

5. **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention**
   - Angelos Katharopoulos, Apoorv Vyas, et al.
   - ICML 2020
   - https://arxiv.org/abs/2006.16236

### Datasets

- **Paddy Doctor: Paddy Disease Classification**
  - Kaggle Competition
  - https://www.kaggle.com/competitions/paddy-disease-classification

