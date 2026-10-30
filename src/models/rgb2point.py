import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

POOL = 7  # V1: the 14 x 14 patch grid is average-pooled to 7 x 7


def image_tokens(vit, images, tokens):
    # "cls": only the CLS feature, as RGB2Point (B, 768)
    # "pooled": CLS + patch tokens pooled to 7 x 7, for V1 (B, 1 + 49, 768)
    x = vit.forward_features(images)  # (B, 1 + 14*14, 768), x[:, 0] is the CLS token
    if tokens == "cls":
        return x[:, 0]
    cls, patches = x[:, :1], x[:, 1:]
    side = int(patches.shape[1] ** 0.5)  # 14
    grid = patches.transpose(1, 2).reshape(len(x), -1, side, side)  # (B, 768, 14, 14)
    grid = F.adaptive_avg_pool2d(grid, POOL)  # (B, 768, 7, 7)
    return torch.cat([cls, grid.flatten(2).transpose(1, 2)], dim=1)


class PointHead(nn.Module):
    # Trainable part of RGB2Point (same layers as official model.py).
    # use_attention=False is V0: the attention on one token is removed.
    def __init__(self, in_dim, num_points, num_heads, dim_feedforward, hidden_dim=4096, use_attention=True):
        super().__init__()
        self.num_points = num_points
        self.aggregator = nn.Linear(in_dim, hidden_dim)
        # Contextual Feature Integrator: self-attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) if use_attention else None
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
        if self.attention is not None:
            x, _ = self.attention(x, x, x)
        x = self.mlp(x.flatten(1))
        return x.view(-1, self.num_points, 3)


class QueryHead(nn.Module):
    # V1: learnable queries read the image tokens with a Transformer decoder.
    # Each query outputs a small group of points (num_points / num_queries).
    def __init__(self, in_dim, num_tokens, num_points, num_queries, dim, num_layers, num_heads):
        super().__init__()
        self.points_per_query = num_points // num_queries
        self.input_proj = nn.Linear(in_dim, dim)
        self.token_pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        # Each layer: self-attention between queries, cross-attention to image tokens, feed-forward
        layer = nn.TransformerDecoderLayer(dim, num_heads, dim_feedforward=4 * dim, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(dim))
        self.out = nn.Linear(dim, self.points_per_query * 3)

    def forward(self, tokens):
        memory = self.input_proj(tokens) + self.token_pos       # (B, 50, dim)
        queries = self.queries.expand(len(tokens), -1, -1)      # (B, 256, dim)
        x = self.decoder(queries, memory)                       # (B, 256, dim)
        return self.out(x).view(len(tokens), -1, 3)             # (B, 256 * 4, 3)

    def attention_maps(self, tokens):
        # Same computation as forward, written layer by layer, so we can also return
        # the cross-attention weights (which image tokens each query looks at).
        memory = self.input_proj(tokens) + self.token_pos
        x = self.queries.expand(len(tokens), -1, -1)
        maps = []
        for layer in self.decoder.layers:
            h = layer.norm1(x)
            x = x + layer.self_attn(h, h, h)[0]
            h = layer.norm2(x)
            out, weights = layer.multihead_attn(h, memory, memory)  # weights: (B, 256, 50), mean over heads
            x = x + out
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
            maps.append(weights)
        points = self.out(self.decoder.norm(x)).view(len(tokens), -1, 3)
        return points, maps


def build_head(model_cfg, in_dim, num_points):
    if model_cfg["head"] == "query":
        return QueryHead(in_dim, 1 + POOL * POOL, num_points, model_cfg["num_queries"], model_cfg["dim"],
                         model_cfg["num_layers"], model_cfg["num_heads"])
    return PointHead(in_dim, num_points, model_cfg["num_heads"], model_cfg["dim_feedforward"],
                     use_attention=model_cfg["head"] == "rgb2point")


class RGB2Point(nn.Module):
    # Full model (frozen ViT + trainable head), used for inference and timing
    def __init__(self, model_cfg, num_points, pretrained=True):
        super().__init__()
        self.tokens = model_cfg["tokens"]
        self.vit = timm.create_model(model_cfg["backbone"], pretrained=pretrained, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.head = build_head(model_cfg, self.vit.num_features, num_points)

    def forward(self, images):
        with torch.no_grad():
            feat = image_tokens(self.vit, images, self.tokens)
        return self.head(feat)


def build_model(model_cfg, num_points, pretrained=True):
    return RGB2Point(model_cfg, num_points, pretrained)
