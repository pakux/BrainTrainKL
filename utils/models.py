import config as cfg
from utils.architectures import sfcn_cls, sfcn_ssl2, head, lora_layers
import torch
import torch.nn as nn
import monai

try:
    from monai.networks.nets import SwinTransformer as MonaiSwinTransformer
except Exception:  # pragma: no cover - fallback for older MONAI
    from monai.networks.nets.swin_unetr import SwinTransformer as MonaiSwinTransformer


def _pick_patch_size(img_size):
    for size in (4, 2, 1):
        if img_size % size == 0:
            return size
    return 1


def _pick_window_size(patch_grid):
    for size in (7, 6, 4, 3, 2):
        if patch_grid % size == 0:
            return size
    return 2


class SwinTransformerClassifier(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        if cfg.SWIN_PATCH_SIZE and cfg.SWIN_WINDOW_SIZE:
            patch_size = tuple(cfg.SWIN_PATCH_SIZE)
            window_size = tuple(cfg.SWIN_WINDOW_SIZE)
        else:
            patch_size = _pick_patch_size(cfg.IMG_SIZE)
            patch_grid = max(cfg.IMG_SIZE // patch_size, 1)
            window_size = _pick_window_size(patch_grid)
            patch_size = (patch_size, patch_size, patch_size)
            window_size = (window_size, window_size, window_size)

        self.backbone = MonaiSwinTransformer(
            in_chans=cfg.N_CHANNELS,
            embed_dim=48,
            window_size=window_size,
            patch_size=patch_size,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            use_checkpoint=False,
            spatial_dims=3,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.LazyLinear(output_dim)

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        feats = self.pool(feats).flatten(1)
        return self.classifier(feats)


def create_model(device):
    """Create model based on training mode and task"""
    output_dim = 1 if cfg.TASK == "regression" else cfg.N_CLASSES

    if cfg.TRAINING_MODE == "sfcn":
        model = sfcn_cls.SFCN(output_dim=output_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), cfg.LEARNING_RATE)
        print(f"Using SFCN for {cfg.TASK}")

    elif cfg.TRAINING_MODE == "dense":
        model = monai.networks.nets.DenseNet121(
            spatial_dims=3, in_channels=cfg.N_CHANNELS, out_channels=output_dim
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), cfg.LEARNING_RATE)
        print(f"Using DenseNet121 for {cfg.TASK}")

    elif cfg.TRAINING_MODE == "swin":
        model = SwinTransformerClassifier(output_dim=output_dim).to(device)
        # Initialize lazy head before parameter counting
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                cfg.N_CHANNELS,
                cfg.IMG_SIZE,
                cfg.IMG_SIZE,
                cfg.IMG_SIZE,
                device=device,
            )
            _ = model(dummy)
        optimizer = torch.optim.AdamW(model.parameters(), cfg.LEARNING_RATE)
        print(f"Using SwinTransformer for {cfg.TASK}")

    elif cfg.TRAINING_MODE in ["linear", "ssl-finetuned"]:
        backbone = sfcn_ssl2.SFCN()
        checkpoint = torch.load(cfg.PRETRAINED_MODEL, map_location=device)
        backbone.load_state_dict(checkpoint["state_dict"], strict=False)
        model = head.ClassifierHeadMLP_(backbone, output_dim=output_dim).to(device)

        if cfg.TRAINING_MODE == "linear":
            for param in model.backbone.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(
                model.classifier.parameters(), cfg.LEARNING_RATE
            )
            print(f"Using Linear Probing for {cfg.TASK}")
        else:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), cfg.LEARNING_RATE)
            print(f"Using SSL Fine-tuning for {cfg.TASK}")

    elif cfg.TRAINING_MODE == "lora":
        backbone = sfcn_ssl2.SFCN()
        checkpoint = torch.load(cfg.PRETRAINED_MODEL, map_location=device)
        backbone.load_state_dict(checkpoint["state_dict"], strict=False)

        backbone = lora_layers.apply_lora_to_model(
            backbone,
            rank=cfg.LORA_RANK,
            alpha=cfg.LORA_ALPHA,
            target_modules=cfg.LORA_TARGET_MODULES,
        )

        for name, param in backbone.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

        model = head.ClassifierHeadMLP_(backbone, output_dim=output_dim).to(device)

        lora_params = [
            p
            for n, p in model.backbone.named_parameters()
            if "lora" in n and p.requires_grad
        ]
        classifier_params = list(model.classifier.parameters())

        optimizer = torch.optim.AdamW(
            lora_params + classifier_params, cfg.LEARNING_RATE
        )

        print(f"LoRA applied for {cfg.TASK}")
        print(f"Trainable LoRA params: {sum(p.numel() for p in lora_params):,}")
        print(
            f"Trainable classifier params: {sum(p.numel() for p in classifier_params):,}"
        )
    else:
        raise ValueError(f"Invalid TRAINING_MODE: {cfg.TRAINING_MODE}")

    return model, optimizer


# ------------------------------
# Model loader
# ------------------------------
def load_model(model_path, device):
    model, _ = create_model(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    return model
