"""EfficientNetV2 backbones cho Paddy Disease Classification.

EfficientNetV2 là thế hệ backbone tiên tiến với:
- Fused-MBConv blocks: nhanh hơn MBConv truyền thống
- Progressive Learning: train nhanh và ổn định hơn
- Optimized cho resolution 224x224 (perfect cho dataset này)
"""

import torch
import torch.nn as nn
import timm

from ..attention import (
    BoTNetBlock,
    BoTNetBlockLinear,
    CABlock,
    ECABlock,
    CoordinateAttention,
    ECAttention
)


class EfficientNetV2_S_Vanilla(nn.Module):
    """EfficientNetV2-S baseline (không có attention).
    
    📊 Thông số:
    - Params: ~21M
    - FLOPs: ~2.9G
    - Accuracy mong đợi: 92-93%
    - FPS (GPU T4): ~350
    - FPS (CPU): ~35
    
    ✨ Ưu điểm:
    - Accuracy cao từ backbone mạnh
    - Training nhanh nhờ Progressive Learning
    - Fused-MBConv blocks tối ưu tốc độ
    - Balanced giữa accuracy và speed
    
    🎯 Khi nào dùng:
    - Cần baseline mạnh để compare
    - Deploy production không cần attention
    - Training nhanh với pretrained weights
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        
        # Load EfficientNetV2-S backbone
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            num_classes=0,  # Không dùng classifier mặc định
            drop_rate=0.0   # Dropout riêng ở cuối
        )
        
        # Feature dimension từ backbone
        self.out_channels = self.backbone.num_features  # 1280 cho EfficientNetV2-S
        
        # Global pooling và classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)
    
    def forward(self, x):
        # Extract features từ backbone
        x = self.backbone.forward_features(x)  # [B, 1280, H, W]
        
        # Global pooling và classify
        x = self.pool(x)  # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)  # [B, 1280]
        x = self.dropout(x)
        x = self.fc(x)  # [B, num_classes]
        
        return x
    
    def get_classifier(self):
        """Trả về classifier layer để tính differential learning rates."""
        return self.fc


class EfficientNetV2_S_CA(nn.Module):
    """EfficientNetV2-S với Coordinate Attention.
    
    📊 Thông số:
    - Params: ~21.5M (+0.5M so với vanilla)
    - FLOPs: ~3.0G
    - Accuracy mong đợi: 93-95%
    - FPS (GPU T4): ~320
    
    ✨ Ưu điểm:
    - CA giúp localize vị trí bệnh trên lá
    - Lightweight attention (chỉ +0.5M params)
    - Tốt cho spatial information
    - Balance accuracy vs speed
    
    🎯 Khi nào dùng:
    - Bệnh có vị trí đặc trưng trên lá
    - Cần spatial localization
    - Ưu tiên balance accuracy-speed
    """
    def __init__(self, num_classes: int = 4, reduction: int = 32, 
                 pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            num_classes=0
        )
        
        self.out_channels = self.backbone.num_features  # 1280
        
        # Coordinate Attention block
        self.ca_block = CABlock(
            c_in=self.out_channels,
            c_out=self.out_channels,
            reduction=reduction
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)  # [B, 1280, H, W]
        x = self.ca_block(x)  # Coordinate Attention
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
    
    def get_classifier(self):
        return self.fc


class EfficientNetV2_S_ECA(nn.Module):
    """EfficientNetV2-S với Efficient Channel Attention.
    
    📊 Thông số:
    - Params: ~21.01M (+0.01M - minimal!)
    - FLOPs: ~2.92G
    - Accuracy mong đợi: 92-94%
    - FPS (GPU T4): ~340 (gần như không giảm!)
    
    ✨ Ưu điểm:
    - Cực kỳ lightweight (chỉ 1D conv)
    - Tốc độ gần bằng vanilla
    - Channel-wise attention hiệu quả
    - Best choice cho production
    
    🎯 Khi nào dùng:
    - Production deployment
    - Ưu tiên tốc độ inference
    - Limited resources
    - Real-time applications
    """
    def __init__(self, num_classes: int = 4, k_size: int = 3,
                 pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            num_classes=0
        )
        
        self.out_channels = self.backbone.num_features
        
        # ECA: cực kỳ lightweight
        self.eca_block = ECABlock(
            c_in=self.out_channels,
            c_out=self.out_channels,
            k_size=k_size
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.eca_block(x)  # ECA attention
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
    
    def get_classifier(self):
        return self.fc


class EfficientNetV2_S_BoTLinear(nn.Module):
    """EfficientNetV2-S với BoTLinear (Linear Attention).
    
    📊 Thông số:
    - Params: ~23M (+2M)
    - FLOPs: ~3.5G
    - Accuracy mong đợi: 94-96%
    - FPS (GPU T4): ~280
    
    ✨ Ưu điểm:
    - Global context via linear attention
    - Tốt cho pattern recognition
    - O(n) complexity thay vì O(n²)
    - Accuracy cao
    
    🎯 Khi nào dùng:
    - Ưu tiên accuracy
    - Có đủ GPU memory
    - Bệnh cần global context
    """
    def __init__(self, num_classes: int = 4, heads: int = 4,
                 pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            num_classes=0
        )
        
        self.out_channels = self.backbone.num_features
        
        # BoTLinear block
        self.bot_block = BoTNetBlockLinear(
            c_in=self.out_channels,
            c_out=self.out_channels,
            heads=heads
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.bot_block(x)  # Linear Attention
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
    
    def get_classifier(self):
        return self.fc


class EfficientNetV2_S_Hybrid(nn.Module):
    """EfficientNetV2-S Hybrid: CA + BoTLinear.
    
    📊 Thông số:
    - Params: ~24M (+3M)
    - FLOPs: ~3.8G
    - Accuracy mong đợi: 95-97% (CAO NHẤT!)
    - FPS (GPU T4): ~250
    
    ✨ Ưu điểm:
    - Kết hợp spatial (CA) + global (BoTLinear)
    - Accuracy cao nhất trong các variants
    - Học được multi-level features
    - Best cho competition
    
    🎯 Khi nào dùng:
    - Ưu tiên accuracy tối đa
    - Có GPU mạnh (≥8GB)
    - Competition/Research
    - Bài toán khó
    """
    def __init__(self, num_classes: int = 4, heads: int = 4, reduction: int = 32,
                 pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            num_classes=0
        )
        
        self.out_channels = self.backbone.num_features
        
        # Hybrid: CA first (spatial), then BoTLinear (global)
        self.ca_block = CABlock(
            c_in=self.out_channels,
            c_out=self.out_channels,
            reduction=reduction
        )
        
        self.bot_block = BoTNetBlockLinear(
            c_in=self.out_channels,
            c_out=self.out_channels,
            heads=heads
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)  # [B, 1280, H, W]
        
        # Sequential: CA first for spatial, then BoTLinear for global
        x = self.ca_block(x)      # Coordinate Attention
        x = self.bot_block(x)     # Linear Attention
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
    
    def get_classifier(self):
        return self.fc


# ============================================================================
# LIGHTWEIGHT VARIANT: EfficientNet-Lite0
# ============================================================================

class EfficientNet_Lite0_CA(nn.Module):
    """EfficientNet-Lite0 với Coordinate Attention.
    
    📊 Thông số:
    - Params: ~4.7M (NHẸ NHẤT!)
    - FLOPs: ~0.4G
    - Accuracy mong đợi: 90-92%
    - FPS (GPU T4): ~600+ 
    - FPS (CPU): ~80 (RẤT NHANH!)
    
    ✨ Ưu điểm:
    - Cực kỳ nhẹ và nhanh
    - Optimized cho mobile/edge devices
    - Không có SE blocks (giảm complexity)
    - Swish thay vì hard-swish
    
    🎯 Khi nào dùng:
    - Mobile deployment
    - Edge devices (Raspberry Pi, Jetson Nano)
    - Real-time inference
    - Limited RAM/Storage
    - Battery-powered devices
    """
    def __init__(self, num_classes: int = 4, reduction: int = 16,
                 pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        
        # EfficientNet-Lite0: optimized cho mobile
        self.backbone = timm.create_model(
            "tf_efficientnet_lite0",
            pretrained=pretrained,
            num_classes=0
        )
        
        self.out_channels = self.backbone.num_features  # 1280
        
        # Lightweight CA với reduction nhỏ hơn
        self.ca_block = CABlock(
            c_in=self.out_channels,
            c_out=self.out_channels,
            reduction=reduction
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.ca_block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
    
    def get_classifier(self):
        return self.fc
