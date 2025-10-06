from .benchmark import accuracy_fscore, fps, evaluate, predict, confusion
from .plot import (
    plot_history,
    plot_confusion_matrix,
    plot_loss_comparision,
    plot_accuracy_vs_fps,
    plot_param
)

# Alias để tương thích với tên cũ
plot_training_history = plot_history

__all__ = [
    # Benchmark functions
    'accuracy_fscore',
    'fps',
    'predict',
    'evaluate',
    'confusion',
    
    # Plot functions
    'plot_history',
    'plot_training_history',  # Alias
    'plot_confusion_matrix',
    'plot_loss_comparision',
    'plot_accuracy_vs_fps',
    'plot_param'
]