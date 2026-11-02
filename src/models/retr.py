import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D-RETR-S (Shi et al., BMVC 2021), rewritten in plain PyTorch from github.com/FomalhautB/3D-RETR
# (src/models/decoder.py VoxelDecoderCNN, src/models/transformer/*, based on lucidrains/x-transformers).
# Image -> encoder tokens -> Transformer decoder with n^3 learnable queries -> 3D CNN -> 32^3 logits.


class Attention(nn.Module):
    # Multi-head attention as in the official code: no bias on q, k, v, dropout on the attention weights.
    # context=None is self-attention, otherwise cross-attention (queries read the image tokens).
    def __init__(self, dim, heads, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, x, context=None):
        if context is None:
            context = x
        b = x.shape[0]
        # (B, N, heads * dim_head) -> (B, heads, N, dim_head)
        q = self.to_q(x).view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(context).view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(context).view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        # softmax(q k^T / sqrt(dim_head)) v, dropout on the attention weights only while training
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(b, -1, self.heads * self.dim_head)
        return self.to_out(out)


class DecoderLayer(nn.Module):
    # Pre-norm layer: x + SelfAttn(LN(x)), x + CrossAttn(LN(x), image), x + FF(LN(x))
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = Attention(dim, heads, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, context):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv3d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv3d(ch, ch, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class RETR(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        dim = cfg["decoder_dim"]
        self.patch_num = cfg["patch_num"]  # queries form an n x n x n grid (baseline n = 4, V2 n = 8)
        # Encoder, fine-tuned. Baseline: DeiT-Tiny distilled (dim 192). V1: DINOv2 ViT-S/14 with registers (dim 384).
        # img_size=224 for every encoder (DINOv2 default is 518)
        self.encoder = timm.create_model(cfg["encoder"], pretrained=pretrained, num_classes=0, img_size=224,
                                         attn_drop_rate=cfg["encoder_dropout"])
        # V1: project the image tokens to the decoder size (384 -> 192). Baseline: sizes are equal, nothing to do.
        if self.encoder.num_features != dim:
            self.context_proj = nn.Linear(self.encoder.num_features, dim)
        else:
            self.context_proj = nn.Identity()
        # Transformer decoder: one learnable query per block of the voxel grid (n = 4: 8^3 voxels, n = 8: 4^3 voxels)
        self.queries = nn.Embedding(self.patch_num ** 3, dim)
        self.layers = nn.ModuleList([DecoderLayer(dim, cfg["decoder_heads"], cfg["decoder_dropout"])
                                     for _ in range(cfg["decoder_depth"])])
        self.norm = nn.LayerNorm(dim)
        # 3D CNN: (192, n, n, n) -> (64, n, n, n) -> 2 ResBlocks -> upsample x2 until 32 -> (1, 32, 32, 32).
        # Baseline / V1: n = 4, 3 upsamples (8, 16, 32). V2: n = 8, 2 upsamples (16, 32).
        ch = cfg["cnn_dim"]
        layers = [nn.Conv3d(dim, ch, 1), ResBlock(ch), ResBlock(ch)]
        size = self.patch_num
        while size < 32:
            layers += [nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1), nn.ReLU()]
            size *= 2
        layers.append(nn.Conv3d(ch, 1, 1))
        self.cnn = nn.Sequential(*layers)

    def forward(self, images):
        # All tokens after the last LayerNorm. DeiT: CLS, distillation, 14 x 14 patches (B, 198, 192).
        # DINOv2: CLS, 4 registers, 16 x 16 patches (B, 261, 384), projected to 192.
        context = self.context_proj(self.encoder.forward_features(images))
        x = self.queries.weight.unsqueeze(0).expand(len(images), -1, -1)  # (B, n^3, 192)
        for layer in self.layers:
            x = layer(x, context)
        x = self.norm(x)
        # Query i -> position (h, w, d) of the n x n x n grid, channels first for Conv3d
        n = self.patch_num
        x = x.transpose(1, 2).reshape(len(images), -1, n, n, n)  # (B, 192, n, n, n)
        return self.cnn(x).squeeze(1)  # (B, 32, 32, 32) logits; sigmoid > 0.5 means occupied


def load_trained(cfg, device):
    # Model of one baseline 2 run with its best checkpoint (chosen on val IoU), ready for inference
    model = RETR(cfg["model"], pretrained=False).to(device).eval()
    ckpt = torch.load(os.path.join(cfg["output_dir"], "best.pt"), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model
