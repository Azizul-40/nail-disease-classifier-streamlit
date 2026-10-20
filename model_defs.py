"""
Model definitions -- exact copies of STEP26 (dataset_1) and STEP23B (dataset_2)
architectures, pulled directly from this pipeline's own STEP41/STEP40/STEP38/39
scripts on Drive (NOT reconstructed/guessed). These are checkpoint-compatible
with the STEP37-locked state_dicts (dataset_1_final_locked.pt / dataset_2_final_locked.pt).
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

CBAM_REDUCTION = 16
CBAM_KERNEL_SIZE = 7

LABEL_MAPS = {
    "dataset_1": {
        "Acral_Lentiginous_Melanoma": 0,
        "Healthy_Nail": 1,
        "Onychogryphosis": 2,
        "blue_finger": 3,
        "clubbing": 4,
        "pitting": 5,
    },
    "dataset_2": {
        "healthy": 0,
        "onychomycosis": 1,
        "psoriasis": 2,
    },
}

MODEL_NAMES = {
    "dataset_1": "STEP26 -- ConvNeXt-Tiny + CBAM (locked, dataset-1: 6-class)",
    "dataset_2": "STEP23B -- ConvNeXt-Tiny standalone (locked, dataset-2: 3-class)",
}

IMAGE_SIZE = 224  # STEP08 config


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * self.sigmoid(att)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True)[0]
        att = self.conv(torch.cat([avg_map, max_map], dim=1))
        return x * self.sigmoid(att)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ConvNeXtCBAM(nn.Module):
    """STEP26: ConvNeXt-Tiny backbone + CBAM attention on the final feature map (dataset_1 locked model)."""

    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        backbone.classifier = nn.Identity()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.feature_dim = 768
        self.cbam = CBAM(channels=self.feature_dim, reduction=CBAM_REDUCTION, kernel_size=CBAM_KERNEL_SIZE)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # x must be float in [0,1] range, shape (N,3,224,224) -- normalization happens inside.
        xn = (x - self.mean) / self.std
        f = self.features(xn)
        f = self.cbam(f)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)
        return self.classifier(f)


class ConvNeXtTFull(nn.Module):
    """STEP23B: ConvNeXt-Tiny, every parameter trainable, single-stage (dataset_2 locked model)."""

    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        self.m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = self.m.classifier[2].in_features
        self.m.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.m((x - self.mean) / self.std)


MODEL_BUILDERS = {
    "dataset_1": ConvNeXtCBAM,
    "dataset_2": ConvNeXtTFull,
}

# Google Drive file IDs of the STEP37-locked checkpoints (for gdown auto-download).
DRIVE_FILE_IDS = {
    "dataset_1": "1vDQA0iFiQOQhkyx7WG71WDX8Zt12eqmx",  # dataset_1_final_locked.pt
    "dataset_2": "1k0cTLeXdqYOyNCePdrY8d8mgpKboijcu",  # dataset_2_final_locked.pt
}

# STEP37 lock record SHA256 (for optional integrity verification at app startup).
LOCKED_SHA256 = {
    "dataset_1": "3a34fb409073e629d3eab83dd3449c74be8c8dc18a3abfefd404c8510a31bfce",
    "dataset_2": "09d815457633a80e8c515c16857af6dca3d8f7309f87730d76802b54a142ab68",
}
