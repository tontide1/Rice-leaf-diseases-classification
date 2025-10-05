import torch, time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def accuracy_fscore(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, f1

def confusion(y_true, y_pred, normalize=None):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    return cm

@torch.inference_mode()
def fps(model, batch_size, image_size, steps=100, warmup=20):
    """
    Đo FPS (Frames Per Second) của model.
    
    Args:
        model: Model cần đo
        batch_size: Batch size để test
        image_size: Kích thước ảnh (H=W)
        steps: Số lần inference để đo (default=100, tăng từ 20)
        warmup: Số lần warmup GPU (default=20, tăng từ 10)
    
    Returns:
        fps: Frames per second (số ảnh xử lý được trong 1 giây)
    """
    device = next(model.parameters()).device
    model.eval()

    # Tạo dummy input
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    
    # Warmup: cho GPU "nóng máy" trước khi đo
    if device.type == "cuda": 
        torch.cuda.synchronize()
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda": 
        torch.cuda.synchronize()
    
    # Đo thời gian chính xác
    if device.type == "cuda":
        # Sử dụng CUDA events để đo chính xác trên GPU
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(steps):
            _ = model(x)
        
        end_event.record()
        torch.cuda.synchronize()
        
        # Elapsed time in milliseconds
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_sec = elapsed_ms / 1000.0
    else:
        # CPU timing
        t0 = time.time()
        for _ in range(steps):
            _ = model(x)
        t1 = time.time()
        elapsed_sec = t1 - t0
    
    # Tính FPS
    total_images = steps * batch_size
    fps_value = total_images / elapsed_sec
    
    return fps_value

@torch.no_grad()
def predict(model, loader, MEAN=None, STD=None):
    device = next(model.parameters()).device
    model.eval()

    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if (MEAN != None) and (STD != None):
            x = (x - MEAN) / STD

        logits = model(x)
        p = logits.argmax(1).cpu().numpy()
        
        y_pred.append(p)
        y_true.append(y.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return y_true, y_pred

@torch.no_grad()
def evaluate(model, loader, criterion, MEAN=None, STD=None):
    device = next(model.parameters()).device
    model.eval()

    y_true, y_pred = [], []
    valid_running, valid_total = 0.0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if (MEAN != None) and (STD != None):
            x = (x - MEAN) / STD

        logits = model(x)
        valid_running += criterion(logits, y).item() * y.size(0)
        valid_total += y.size(0)

        p = logits.argmax(1).cpu().numpy()
        
        y_pred.append(p)
        y_true.append(y.cpu().numpy())

    valid_loss = valid_running / valid_total
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc, f1 = accuracy_fscore(y_true, y_pred)
    
    return valid_loss, acc, f1

