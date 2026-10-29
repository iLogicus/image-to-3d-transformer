import timm
import torch
import torch.nn as nn


class PointHead(nn.Module):
    # Trainable part of RGB2Point (same layers as official model.py)
    def __init__(self, in_dim, num_points, num_heads, dim_feedforward, hidden_dim=4096):
        super().__init__()
        self.num_points = num_points
        self.aggregator = nn.Linear(in_dim, hidden_dim)
        # Contextual Feature Integrator: self-attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        # Geometric Projection Module: MLP -> N x 3 coordinates
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_feedforward, num_points * 3),
        )

    def forward(self, feat):
        x = self.aggregator(feat).unsqueeze(1)  # (B, 1, 4096): only one token
        x, _ = self.attention(x, x, x)
        x = self.mlp(x.flatten(1))
        return x.view(-1, self.num_points, 3)


class RGB2Point(nn.Module):
    def __init__(self, backbone, num_points, num_heads, dim_feedforward, pretrained=True):
        super().__init__()
        # num_classes=0: timm returns the CLS token feature (768 for ViT-B)
        self.vit = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.head = PointHead(self.vit.num_features, num_points, num_heads, dim_feedforward)

    def forward(self, images):
        with torch.no_grad():
            feat = self.vit(images)
        return self.head(feat)


def build_model(model_cfg, num_points, pretrained=True):
    return RGB2Point(model_cfg["backbone"], num_points, model_cfg["num_heads"],
                     model_cfg["dim_feedforward"], pretrained)
