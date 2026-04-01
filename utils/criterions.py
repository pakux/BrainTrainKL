import BrainTrain.config as cfg
import torch
import torch.nn as nn
import numpy as np

def get_criterion(device, train_labels=None):
    """Get loss function based on task"""
    if cfg.TASK == 'classification':
        if train_labels is not None:
            train_labels = np.asarray(train_labels, dtype=np.int64)
            class_counts = np.bincount(train_labels, minlength=cfg.N_CLASSES).astype(np.float64)
            valid_counts = np.maximum(class_counts, 1.0)
            class_weights = class_counts.sum() / (cfg.N_CLASSES * valid_counts)
            # If a class is missing in the split, zero-out its loss contribution.
            class_weights[class_counts == 0] = 0.0
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            print(f"Class weights: {class_weights}")
            missing_classes = np.where(class_counts == 0)[0]
            if len(missing_classes) > 0:
                print(f"Warning: missing classes in train split: {missing_classes.tolist()}")
            criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

    elif cfg.TASK == 'regression':
        criterion = nn.L1Loss().to(device)
        print("Using L1 Loss for regression")

    else:
        raise ValueError(f"Invalid TASK: {cfg.TASK}")
    
    return criterion
