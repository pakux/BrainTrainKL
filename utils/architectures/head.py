import torch
import torch.nn as nn
import config as cfg


class ClassifierHeadMLP_(nn.Module):
    def __init__(self, backbone, output_dim=cfg.N_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(64, output_dim)  # assumes final feature dim is 64

    def forward(self, x):
        feats = self.backbone(x, return_projection=False)  # Only use features
        return self.classifier(feats)


class RegressorHeadMLP_(nn.Module):
    def __init__(self, backbone, output_dim=cfg.N_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Linear(64, output_dim)  # assumes final feature dim is 64

    def forward(self, x):
        # Only use backbone features (no projection head if SSL)
        feats = self.backbone(x, return_projection=False)
        return self.regressor(feats)
