import sys
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Thêm src vào path để import modules
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models.backbones.mobilenet import MobileNetV3_Small_CA


class SquarePad:
    """
    Giữ nguyên tỷ lệ ảnh gốc, không làm méo ảnh bằng cách padding thành hình vuông.
    """
    def __init__(self, fill=0):
        self.fill = fill
    
    def __call__(self, image):
        from PIL import ImageOps
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return ImageOps.expand(image, border=padding, fill=self.fill)


def load_model(model_path: Path, device: str = "cuda") -> nn.Module:
    model = MobileNetV3_Small_CA(
        num_classes=4, 
        reduction=16, 
        pretrained=False,
        dropout=0.0
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Xử lý nhiều trường hợp lưu checkpoint
    if isinstance(checkpoint, dict):
        # In ra các keys để debug
        print(f"Keys trong checkpoint: {list(checkpoint.keys())}")
        
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load state dict vào model
    model.load_state_dict(state_dict, strict=True)
    
    model = model.to(device)
    model.eval()
    
    print(f"Đã load model thành công từ: {model_path}")
    return model


def load_labels(label_file: Path) -> Dict[int, str]:
    with open(label_file, "r") as f:
        id2label = json.load(f)
    return {int(k): v for k, v in id2label.items()}


def preprocess_image(image_path: Path, image_size: int = 224) -> torch.Tensor:
    """
    Tiền xử lý ảnh đầu vào.
    
    Args:
        image_path: Đường dẫn đến ảnh cần test
        image_size: Kích thước ảnh sau resize
        
    Returns:
        Tensor ảnh đã xử lý (1, C, H, W)
    """
    # Transform giống như khi train/validation
    transform = transforms.Compose([
        SquarePad(fill=0),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load và transform ảnh
    image = Image.open(image_path).convert("RGB")    
    image_tensor = transform(image)
    
    # Thêm batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def predict(
    model: nn.Module,
    image_tensor: torch.Tensor,
    id2label: Dict[int, str],
    device: str = "cuda"
) -> tuple:
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        all_probs = probabilities.cpu().numpy()[0]
    
    predicted_class = id2label[predicted_idx]
    
    return predicted_class, confidence, all_probs


def main():
    # Cấu hình
    ROOT_DIR = Path(__file__).resolve().parent
    LABEL_FILE = ROOT_DIR / "data" / "id2label.json"
    IMAGE_PATH = ROOT_DIR / "test_DL.jpg"
    MODEL_PATH = ROOT_DIR / "results" / "MobileNetV3_Small_CA_finetuned_10_10_2025_1303" / "MobileNetV3_Small_CA_finetuned_final.pt"                                                                                                                                                                                        ; IMAGE_PATH = ROOT_DIR / "1_1.jpg" 
    IMAGE_SIZE = 224
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Kiểm tra file tồn tại
    if not MODEL_PATH.exists():
        print(f"LỖI: Không tìm thấy file model tại {MODEL_PATH}")
        return
    
    if not IMAGE_PATH.exists():
        print(f"LỖI: Không tìm thấy ảnh test tại {IMAGE_PATH}")
        return
    
    if not LABEL_FILE.exists():
        print(f"LỖI: Không tìm thấy file label tại {LABEL_FILE}")
        return
    
    # Load model và labels
    print("\nĐang load model...")
    model = load_model(MODEL_PATH, device=DEVICE)
    id2label = load_labels(LABEL_FILE)
    
    print(f"\nCác class có thể dự đoán:")
    for idx, label in id2label.items():
        print(f"  [{idx}] {label}")
    print()
    
    # Tiền xử lý ảnh
    print("Đang xử lý ảnh...")
    image_tensor = preprocess_image(IMAGE_PATH, image_size=IMAGE_SIZE)
    print(f"Tensor shape: {image_tensor.shape}")
    
    # Dự đoán
    print("\nĐang dự đoán...")
    predicted_class, confidence, all_probs = predict(
        model, image_tensor, id2label, device=DEVICE
    )
    
    # Hiển thị kết quả
    print("KẾT QUẢ DỰ ĐOÁN:")
    print(f"{predicted_class}")



if __name__ == "__main__":
    main()
