import torch
import torchvision
import numpy as np


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

def calculate_iou(predictions, targets, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = predictions == cls
        target_inds = targets == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))  # Avoid division by zero
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)
