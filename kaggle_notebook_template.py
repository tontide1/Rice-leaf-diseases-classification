"""Create Kaggle notebook template."""

import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌾 Paddy Disease Classification Training\n",
                "\n",
                "Training MobileNetV3-Small with BoT on Paddy Disease dataset"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1️⃣ Setup Environment"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "import sys\n",
                "import os\n",
                "\n",
                "# Add source code to path\n",
                "sys.path.insert(0, '/kaggle/input/paddy-disease-classification-src')\n",
                "\n",
                "# Check GPU\n",
                "!nvidia-smi\n",
                "\n",
                "# Check PyTorch\n",
                "import torch\n",
                "print(f'PyTorch version: {torch.__version__}')\n",
                "print(f'CUDA available: {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2️⃣ Verify Data"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "\n",
                "# Check data\n",
                "DATA_PATH = Path('/kaggle/input/paddy-disease-classification')\n",
                "\n",
                "print('📁 Data structure:')\n",
                "for item in DATA_PATH.rglob('*'):\n",
                "    if item.is_file():\n",
                "        print(f'  {item.relative_to(DATA_PATH)}')\n",
                "\n",
                "# Load metadata\n",
                "df = pd.read_csv(DATA_PATH / 'metadata.csv')\n",
                "print(f'\\n📊 Dataset info:')\n",
                "print(f'  Total samples: {len(df)}')\n",
                "print(f'  Classes: {df[\"label\"].nunique()}')\n",
                "print(f'\\n  Class distribution:')\n",
                "print(df['label'].value_counts())"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3️⃣ Train Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Run training script\n",
                "!python /kaggle/input/paddy-disease-classification-src/train_kaggle.py"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4️⃣ View Results"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "import json\n",
                "from IPython.display import Image, display\n",
                "\n",
                "# Load metrics\n",
                "with open('/kaggle/working/metrics.json', 'r') as f:\n",
                "    metrics = json.load(f)\n",
                "\n",
                "print('🎯 Final Metrics:')\n",
                "print('='*50)\n",
                "for key, value in metrics.items():\n",
                "    if isinstance(value, float):\n",
                "        print(f'  {key:.<30} {value:.4f}')\n",
                "    else:\n",
                "        print(f'  {key:.<30} {value}')\n",
                "\n",
                "# Display plot\n",
                "print('\\n📊 Training History:')\n",
                "display(Image('/kaggle/working/training_history.png'))"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5️⃣ Save Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Model already saved to /kaggle/working/\n",
                "# Download it from the Output section on the right →\n",
                "\n",
                "!ls -lh /kaggle/working/*.pt\n",
                "!ls -lh /kaggle/working/*.json\n",
                "!ls -lh /kaggle/working/*.png"
            ],
            "outputs": []
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save notebook
with open('kaggle_notebook.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✓ Created: kaggle_notebook.ipynb")
print("📤 Upload this to Kaggle Notebook")