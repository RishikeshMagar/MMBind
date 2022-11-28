import math
import torch
from torch import nn


def adjust_learning_rate(
    optimizer, epoch, lr, min_lr, n_epochs, warmup_epochs, patience_epochs, 
):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    elif epoch < warmup_epochs + patience_epochs:
        lr = lr
    else:
        prev_epochs = warmup_epochs + patience_epochs
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - prev_epochs) / (n_epochs - prev_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
