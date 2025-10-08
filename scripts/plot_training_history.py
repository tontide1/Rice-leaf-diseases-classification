"""
Script vẽ biểu đồ training history từ file JSON.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Đọc file history.json
history_file = 'history.json'
with open(history_file, 'r') as f:
    history = json.load(f)

# Lấy dữ liệu
train_loss = history['train_loss']
valid_loss = history['valid_loss']
train_acc = history['train_acc']
valid_acc = history['valid_acc']

# Số epochs
epochs = range(1, len(train_loss) + 1)

# Tính accuracy cuối cùng (%)
final_valid_acc = valid_acc[-1] * 100

# Tạo figure với 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Loss
ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
ax1.plot(epochs, valid_loss, 'r-', linewidth=2, label='Valid Loss', marker='s', markersize=4)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(1, len(epochs))

# Tìm best loss
best_valid_loss = min(valid_loss)
best_loss_epoch = valid_loss.index(best_valid_loss) + 1
ax1.axvline(x=best_loss_epoch, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.text(best_loss_epoch, max(train_loss) * 0.95, 
         f'Best: {best_valid_loss:.3f}', 
         fontsize=9, color='green', ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.8))

# Plot Accuracy
ax2.plot(epochs, np.array(train_acc) * 100, 'b-', linewidth=2, label='Train Acc', marker='o', markersize=4)
ax2.plot(epochs, np.array(valid_acc) * 100, 'r-', linewidth=2, label='Valid Acc', marker='s', markersize=4)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(1, len(epochs))
ax2.set_ylim(50, 100)

# Tìm best accuracy
best_valid_acc = max(valid_acc) * 100
best_acc_epoch = valid_acc.index(max(valid_acc)) + 1
ax2.axvline(x=best_acc_epoch, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.text(best_acc_epoch, 52, 
         f'Best: {best_valid_acc:.2f}%', 
         fontsize=9, color='green', ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.8))

# Thêm title chính cho toàn bộ figure
fig.suptitle(f'CNN BASE HISTORY (Acc: {final_valid_acc:.2f}%)', 
             fontsize=16, fontweight='bold', y=1.02)

# Thêm thông tin tổng hợp
info_text = f'Epochs: {len(epochs)} | Best Valid Acc: {best_valid_acc:.2f}% (epoch {best_acc_epoch}) | Final Valid Acc: {final_valid_acc:.2f}%'
fig.text(0.5, -0.02, info_text, ha='center', fontsize=10, 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))

plt.tight_layout()

# Lưu figure
output_path = 'imgs/cnn_base_training_history.png'
Path('imgs').mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'✅ Đã lưu biểu đồ training history tại: {output_path}')

# Hiển thị
plt.show()

# In thống kê
print(f'\n📊 THỐNG KÊ TRAINING:')
print(f'{"="*50}')
print(f'Tổng số epochs: {len(epochs)}')
print(f'Best Valid Loss: {best_valid_loss:.4f} (epoch {best_loss_epoch})')
print(f'Best Valid Accuracy: {best_valid_acc:.2f}% (epoch {best_acc_epoch})')
print(f'Final Valid Accuracy: {final_valid_acc:.2f}%')
print(f'Train Acc (final): {train_acc[-1]*100:.2f}%')
print(f'Overfitting: {(train_acc[-1] - valid_acc[-1])*100:.2f}%')
print(f'{"="*50}')
