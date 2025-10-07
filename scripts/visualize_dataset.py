#!/usr/bin/env python3
"""
Dataset Visualization Script for Paddy Disease Classification

Tạo các biểu đồ phân tích dữ liệu hiện tại:
- Phân bố classes
- Tỷ lệ train/valid/test
- Sample images từ mỗi class
- Thông tin thống kê chi tiết
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from PIL import Image
import random
import warnings

warnings.filterwarnings('ignore')

# Thiết lập style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Đọc dữ liệu
def load_data():
    """Đọc và xử lý dữ liệu metadata"""
    df = pd.read_csv('data/metadata.csv')

    # Map tên class cho đẹp hơn
    class_names = {
        'bacterial_leaf_blight': 'Bacterial Leaf Blight',
        'brown_spot': 'Brown Spot',
        'healthy': 'Healthy',
        'leaf_blast': 'Leaf Blast'
    }
    df['class_name'] = df['label'].map(class_names)

    print(" THÔNG TIN DATASET")
    print(f"┌{'─' * 50}┐")
    print(f"│ Tổng số samples: {len(df):,} │")
    print(f"│ Số classes: {df.label.nunique()} │")
    print(f"│ Classes: {', '.join(sorted(df.label.unique()))} │")
    print(f"└{'─' * 50}┘")

    return df

def plot_class_distribution(df):
    """Vẽ biểu đồ phân bố các classes"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Bar chart tổng quan
    class_counts = df['class_name'].value_counts()
    colors = sns.color_palette("husl", len(class_counts))

    axes[0, 0].bar(range(len(class_counts)), class_counts.values,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    axes[0, 0].set_xticks(range(len(class_counts)))
    axes[0, 0].set_xticklabels([label.split()[-1] for label in class_counts.index],
                               rotation=45, ha='right')
    axes[0, 0].set_title(' Tổng số samples theo loại bệnh', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Số lượng samples', fontsize=12)

    # Thêm số liệu trên bar
    for i, v in enumerate(class_counts.values):
        axes[0, 0].text(i, v + 50, f'{v:,}', ha='center', va='bottom', fontweight='bold')

    # 2. Pie chart tỷ lệ
    axes[0, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[0, 1].set_title('🥧 Tỷ lệ phân bố các loại bệnh', fontsize=14, fontweight='bold')

    # 3. Bar chart theo split
    split_data = []
    for split in ['train', 'valid', 'test']:
        split_df = df[df['split'] == split]
        split_counts = split_df['class_name'].value_counts()
        split_data.append(split_counts)

    x = np.arange(len(class_counts))
    width = 0.25

    for i, (split, counts) in enumerate(zip(['train', 'valid', 'test'], split_data)):
        axes[1, 0].bar(x + i*width - width, counts.values, width,
                       label=split.upper(), alpha=0.8, edgecolor='black')

    axes[1, 0].set_xlabel('Loại bệnh')
    axes[1, 0].set_ylabel('Số lượng samples')
    axes[1, 0].set_title(' Phân bố theo Train/Valid/Test', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([label.split()[-1] for label in class_counts.index],
                               rotation=45, ha='right')
    axes[1, 0].legend()

    # 4. Heatmap class vs split
    class_split = pd.crosstab(df['class_name'], df['split'])
    sns.heatmap(class_split, annot=True, fmt=',', cmap='YlOrRd',
                cbar_kws={'label': 'Số lượng samples'}, ax=axes[1, 1])
    axes[1, 1].set_title('🔥 Heatmap: Class vs Split', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Split')
    axes[1, 1].set_ylabel('Loại bệnh')

    plt.tight_layout()
    plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
    print(" Đã lưu biểu đồ tổng quan: data_visualization.png")

    return fig

def plot_sample_images(df, num_samples_per_class=3):
    """Hiển thị sample images từ mỗi class"""
    fig, axes = plt.subplots(len(df['class_name'].unique()), num_samples_per_class,
                             figsize=(15, 12))

    classes = sorted(df['class_name'].unique())

    for i, class_name in enumerate(classes):
        # Lấy random samples từ class này
        class_samples = df[df['class_name'] == class_name].sample(
            num_samples_per_class, random_state=42)

        for j, (_, sample) in enumerate(class_samples.iterrows()):
            try:
                # Đọc ảnh
                img_path = sample['path']
                img = Image.open(img_path)

                # Hiển thị ảnh
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"{class_name}\n({sample['split'].upper()})",
                                     fontsize=10, fontweight='bold')
                axes[i, j].axis('off')

            except Exception as e:
                print(f"❌ Không thể đọc ảnh {img_path}: {e}")
                axes[i, j].text(0.5, 0.5, 'Image\nError',
                               ha='center', va='center', transform=axes[i, j].transAxes)

    plt.suptitle('🖼️ Sample Images từ mỗi loại bệnh', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    print("✅ Đã lưu sample images: sample_images.png")

    return fig

def plot_detailed_statistics(df):
    """Thống kê chi tiết hơn"""
    print("\n THỐNG KÊ CHI TIẾT")

    # Tính toán thống kê
    total_samples = len(df)
    num_classes = df['label'].nunique()
    class_balance = df['label'].value_counts()
    split_balance = df['split'].value_counts()

    print("┌" + "─" * 70 + "┐")
    print("│" + " ".ljust(68) + "│")
    print(f"│ Tổng số samples: {total_samples:,}".ljust(69) + "│")
    print(f"│ Số classes: {num_classes}".ljust(69) + "│")
    print(f"│ Train/Valid/Test ratio: {split_balance['train']/total_samples*100:.1f}% / {split_balance['valid']/total_samples*100:.1f}% / {split_balance['test']/total_samples*100:.1f}%".ljust(69) + "│")
    print("│" + " ".ljust(68) + "│")
    print("├" + "─" * 68 + "┤")
    print("│ CLASS DISTRIBUTION:" + " ".ljust(50) + "│")
    print("├" + "─" * 68 + "┤")

    for label, count in class_balance.items():
        percentage = count / total_samples * 100
        bar_length = int(percentage / 2)  # Scale down for display
        bar = "█" * bar_length

        class_name = {
            'bacterial_leaf_blight': 'Bacterial Leaf Blight',
            'brown_spot': 'Brown Spot',
            'healthy': 'Healthy',
            'leaf_blast': 'Leaf Blast'
        }[label]

        print(f"│ {class_name:<25} {count:>6,} samples ({percentage:>5.1f}%) {bar:<20} │")

    print("└" + "─" * 70 + "┘")

    # Kiểm tra balance
    print("\n ĐÁNH GIÁ BALANCE:")
    print(f"Standard deviation of class counts: {class_balance.std():.1f}")
    print(f"Coefficient of variation: {class_balance.std()/class_balance.mean()*100:.1f}%")

    if class_balance.std() / class_balance.mean() < 0.1:
        print(" Dataset khá cân bằng!")
    elif class_balance.std() / class_balance.mean() < 0.2:
        print(" Dataset hơi mất cân bằng")
    else:
        print(" Dataset mất cân bằng nghiêm trọng!")

    return class_balance

def analyze_image_properties(df, sample_size=100):
    """Phân tích kích thước ảnh (nếu cần)"""
    print("\n📷 PHÂN TÍCH ẢNH (Sample)")
    sample_paths = df['path'].sample(sample_size, random_state=42)

    widths, heights = [], []
    for path in sample_paths:
        try:
            with Image.open(path) as img:
                widths.append(img.width)
                heights.append(img.height)
        except:
            continue

    if widths and heights:
        print(f" Kích thước ảnh trung bình: {np.mean(widths):.0f}×{np.mean(heights):.0f}")
        print(f" Kích thước ảnh trung vị: {np.median(widths):.0f}×{np.median(heights):.0f}")
        print(f" Min/Max: {min(widths)}×{min(heights)} / {max(widths)}×{max(heights)}")

        # Tỷ lệ aspect ratio
        aspect_ratios = [w/h for w, h in zip(widths, heights)]
        print(f" Aspect ratio trung bình: {np.mean(aspect_ratios):.2f}")
    else:
        print("❌ Không thể đọc được thông tin ảnh")

def main():
    """Main visualization function"""
    print(" BẮT ĐẦU VISUALIZE DATASET")
    print("=" * 60)

    # Load data
    df = load_data()

    # Vẽ biểu đồ
    fig1 = plot_class_distribution(df)
    fig2 = plot_sample_images(df)
    stats = plot_detailed_statistics(df)
    analyze_image_properties(df)

    print("\n HOÀN THÀNH VISUALIZATION!")
    print(" Files đã tạo:")
    print("  • data_visualization.png - Biểu đồ tổng quan")
    print("  • sample_images.png - Sample images")

    plt.show()

if __name__ == "__main__":
    main()
