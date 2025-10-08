"""
Script vẽ biểu đồ training history từ file history.json.
Sử dụng: python plot_cnn_base_history.py
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_history(history_file, title, output_file):
    """
    Vẽ biểu đồ training history từ file JSON.
    
    Args:
        history_file: Đường dẫn đến file history.json
        title: Tiêu đề của biểu đồ
        output_file: Đường dẫn lưu file ảnh output
    """
    # Load history từ file JSON
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Tạo figure với 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot 1: Loss curves
    axes[0].plot(history["train_loss"], label="Train loss", linewidth=2)
    axes[0].plot(history["valid_loss"], label="Valid loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_title("Loss", fontsize=13, fontweight='bold')
    
    # Plot 2: Accuracy curves
    axes[1].plot(history["train_acc"], label="Train acc", linewidth=2)
    axes[1].plot(history["valid_acc"], label="Valid acc", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Acc", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].set_title("Accuracy", fontsize=13, fontweight='bold')
    
    # Set main title
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ: {output_path}")
    
    # Show figure
    plt.show()


if __name__ == "__main__":
    # Đường dẫn file input và output
    history_file = "history.json"
    output_file = "imgs/cnn_base_training_history.png"
    
    # Tính toán best validation accuracy từ history
    with open(history_file, 'r') as f:
        history = json.load(f)
    best_valid_acc = max(history["valid_acc"]) * 100  # Convert sang phần trăm
    
    # Tạo tiêu đề với accuracy
    title = f"CNN BASE HISTORY (Acc: {best_valid_acc:.2f}%)"
    
    # Vẽ biểu đồ
    plot_training_history(history_file, title, output_file)
