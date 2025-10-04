# Paddy Disease Classification - Training on Kaggle

## 🚀 Quick Start

### 1. Upload Data to Kaggle Dataset
- Create new dataset named `paddy-disease-classification`
- Upload: `data/metadata.csv`, `data/label2id.json`, `data/images/`

### 2. Create Kaggle Notebook
- Enable GPU (Settings → Accelerator → GPU T4 x2)
- Add dataset as input

### 3. Upload Source Code
- Upload this entire folder as Kaggle Dataset
- Name it: `paddy-disease-classification-src`

### 4. Run Training
```python
# In Kaggle Notebook
!python /kaggle/input/paddy-disease-classification-src/train_kaggle.py
```

## 📊 Configuration

Edit `train_kaggle.py` to change:
- `batch_size`: Default 64 (reduce if OOM)
- `epochs`: Default 30
- `learning_rate`: base_lr=5e-5, head_lr=5e-4
- `image_size`: Default 224

## 📥 Output Files

After training, download from `/kaggle/working/`:
- `MobileNetV3_Small_BoT_Kaggle_best.pt` - Best model checkpoint
- `metrics.json` - Final metrics
- `history.json` - Training history
- `training_history.png` - Loss/Accuracy plots

## 💡 Tips

1. **Enable Internet** in notebook settings to download pretrained weights
2. **Use GPU T4 x2** for faster training
3. **Reduce batch_size** to 32 or 16 if you get OOM errors
4. Training takes ~30-45 minutes for 30 epochs

## 📝 Notes

- Kaggle provides 30 hours/week of GPU time
- Session limit: 12 hours continuous
- Save outputs before session ends!
