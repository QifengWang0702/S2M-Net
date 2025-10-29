# net.py

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# timm is optional; used for ViT/SAM variants
try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


# ========= 1) Radial positional encoding (learnable + sinusoidal) =========
class RadialPositionalEncoding(nn.Module):
    """
    Provides per-slice positional embeddings combining sinusoidal encoding and an optional learnable vector.
    """
    def __init__(self, S: int, d_model: int, learnable: bool = True, use_sin: bool = True):
        super().__init__()
        self.S = int(S)
        self.d_model = int(d_model)
        self.use_sin = bool(use_sin)
        self.learnable = bool(learnable)
        if learnable:
            self.slice_embed = nn.Parameter(torch.randn(1, S, d_model) * 0.02)
        else:
            self.register_parameter("slice_embed", None)

    def forward(self, B: int, device: torch.device):
        pe = None
        if self.use_sin:
            pos = torch.arange(self.S, device=device).float().unsqueeze(1)           # [S,1]
            dim = torch.arange(self.d_model, device=device).float().unsqueeze(0)     # [1,D]
            div = torch.exp(-(math.log(10000.0) * (2 * (dim // 2)) / self.d_model))  # [1,D]
            sin = torch.sin(pos * div)
            cos = torch.cos(pos * div)
            sin[:, 1::2] = 0
            cos[:, ::2] = 0
            pe = (sin + cos).unsqueeze(0).repeat(B, 1, 1)  # [B,S,D]
        if self.learnable:
            le = self.slice_embed.repeat(B, 1, 1)
            return (pe + le) if pe is not None else le
        else:
            return pe if pe is not None else torch.zeros(B, self.S, self.d_model, device=device)


# ========= 2) Per-slice encoders: input [BS,C,H,W] → output [BS,D] =========
class CNNPerSliceStem(nn.Module):
    """
    Extracts per-slice features with depthwise separable convolutions and global average pooling, followed by a linear projection.
    """
    def __init__(self, in_ch=5, out_dim=512):
        super().__init__()
        ch1, ch2, ch3 = 64, 128, 256
        self.net = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(in_ch, ch1, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ch1), nn.ReLU(inplace=True),

            # DW(3x3) + PW(1x1)
            nn.Conv2d(ch1, ch1, 3, padding=1, groups=ch1, bias=False),
            nn.Conv2d(ch1, ch2, 1, bias=False),
            nn.BatchNorm2d(ch2), nn.ReLU(inplace=True),

            # 128 -> 64
            nn.Conv2d(ch2, ch2, 3, stride=2, padding=1, groups=ch2, bias=False),
            nn.Conv2d(ch2, ch2, 1, bias=False),
            nn.BatchNorm2d(ch2), nn.ReLU(inplace=True),

            # DW + PW (no downsampling)
            nn.Conv2d(ch2, ch2, 3, padding=1, groups=ch2, bias=False),
            nn.Conv2d(ch2, ch3, 1, bias=False),
            nn.BatchNorm2d(ch3), nn.ReLU(inplace=True),

            # 64 -> 32
            nn.Conv2d(ch3, ch3, 3, stride=2, padding=1, groups=ch3, bias=False),
            nn.Conv2d(ch3, ch3, 1, bias=False),
            nn.BatchNorm2d(ch3), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(ch3, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [BS,C,H,W] → [BS,D]
        f = self.net(x).flatten(1)
        return self.fc(f)


class DWConvPW(nn.Module):
    """Applies depthwise 3x3 and pointwise 1x1 convolutions with batch normalization and SiLU activation."""
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(ci, ci, 3, stride=stride, padding=1, groups=ci, bias=False)
        self.pw = nn.Conv2d(ci, co, 1, bias=False)
        self.bn = nn.BatchNorm2d(co)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class EfficientBlock(nn.Module):
    """
    Builds a residual block from two DWConvPW layers; uses a projection when spatial resolution changes.
    """
    def __init__(self, c, stride=1):
        super().__init__()
        self.conv1 = DWConvPW(c, c, stride=stride)
        self.conv2 = DWConvPW(c, c, stride=1)
        self.proj = None
        if stride != 1:
            self.proj = nn.Sequential(
                nn.Conv2d(c, c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c)
            )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.proj is not None:
            identity = self.proj(identity)
        return F.silu(out + identity, inplace=True)


class UNetEncoder2D(nn.Module):
    """
    Encodes slices with a lightweight UNet-like hierarchy using DWConvPW blocks and progressive downsampling; outputs pooled features projected to out_dim.
    """
    def __init__(self, in_ch=5, base=24, out_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 7, stride=2, padding=3, bias=False),  # 256→128
            nn.BatchNorm2d(base),
            nn.SiLU(inplace=True)
        )
        # 128→64
        self.enc1 = nn.Sequential(
            EfficientBlock(base, stride=2),
            EfficientBlock(base, stride=1),
        )
        # 64→32
        self.enc2_in = nn.Conv2d(base, base*2, 3, stride=2, padding=1, bias=False)
        self.enc2_bn = nn.BatchNorm2d(base*2)
        self.enc2 = nn.Sequential(
            EfficientBlock(base*2, 1),
            EfficientBlock(base*2, 1),
        )
        # 32→16
        self.enc3_in = nn.Conv2d(base*2, base*4, 3, stride=2, padding=1, bias=False)
        self.enc3_bn = nn.BatchNorm2d(base*4)
        self.enc3 = nn.Sequential(
            EfficientBlock(base*4, 1),
            EfficientBlock(base*4, 1),
        )

        self.out_fc = nn.Linear(base*4, out_dim)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)                               # [BS,b, 128,128]
        x = self.enc1(x)                               # [BS,b,  64, 64]
        x = F.silu(self.enc2_bn(self.enc2_in(x)), True)# [BS,2b, 32, 32]
        x = self.enc2(x)
        x = F.silu(self.enc3_bn(self.enc3_in(x)), True)# [BS,4b, 16, 16]
        x = self.enc3(x)
        v = F.adaptive_avg_pool2d(x, 1).flatten(1)     # [BS, 4b]
        return self.out_fc(v)                          # [BS, D]


class ResUNetEncoder2D(nn.Module):
    """
    Encodes slices with bottleneck residual blocks (1x1 → DW3x3 → 1x1) and staged downsampling; outputs pooled features projected to out_dim.
    """
    def __init__(self, in_ch=5, base=24, out_dim=256, bottleneck_ratio=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 7, stride=2, padding=3, bias=False),  # 256→128
            nn.BatchNorm2d(base),
            nn.SiLU(inplace=True)
        )

        def bottleneck(ci, co, stride=1, br=0.5):
            mid = max(8, int(co * br))
            layers = [
                nn.Conv2d(ci, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True),
                nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True),
                nn.Conv2d(mid, co, 1, bias=False), nn.BatchNorm2d(co),
            ]
            proj = None
            if stride != 1 or ci != co:
                proj = nn.Sequential(nn.Conv2d(ci, co, 1, stride=stride, bias=False), nn.BatchNorm2d(co))
            return nn.Sequential(*layers), proj

        self.b1, self.p1 = bottleneck(base,     base,     stride=1, br=bottleneck_ratio)  # 128→128
        self.b2, self.p2 = bottleneck(base,     base*2,   stride=2, br=bottleneck_ratio)  # 128→ 64
        self.b3, self.p3 = bottleneck(base*2,   base*4,   stride=2, br=bottleneck_ratio)  #  64→ 32
        self.b4, self.p4 = bottleneck(base*4,   base*4,   stride=1, br=bottleneck_ratio)  #  32→ 32

        self.out_fc = nn.Linear(base*4, out_dim)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)                    # [BS, b, 128,128]
        y = self.b1(x); x = F.silu(y + (self.p1(x) if self.p1 else x), True)
        y = self.b2(x); x = F.silu(y + (self.p2(x) if self.p2 else x), True)  # [BS,2b, 64,64]
        y = self.b3(x); x = F.silu(y + (self.p3(x) if self.p3 else x), True)  # [BS,4b, 32,32]
        y = self.b4(x); x = F.silu(y + (self.p4(x) if self.p4 else x), True)  # [BS,4b, 32,32]
        v = F.adaptive_avg_pool2d(x, 1).flatten(1)                            # [BS,4b]
        return self.out_fc(v)                                                 # [BS,D]


class TransUNetEncoderVec(nn.Module):
    """
    Extracts features via shallow CNN downsampling followed by a ViT; aggregates by mean pooling over tokens and projects to out_dim.
    """
    def __init__(self, in_ch=5, out_dim=256,
                 vit_name: str = "vit_base_patch16_224",
                 pretrained: bool = False,
                 vit_img_size: int = 64):
        super().__init__()
        if not _HAS_TIMM:
            raise RuntimeError("timm is required for TransUNetEncoderVec")
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, stride=2, padding=3), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),   nn.ReLU(True)
        )
        self.vit = timm.create_model(
            vit_name, pretrained=pretrained, in_chans=128, num_classes=0, img_size=vit_img_size
        )
        self.proj = nn.Linear(self.vit.num_features, out_dim)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)                      # [BS,128,64,64]
        feats = self.vit.forward_features(x)  # [BS,N,C] or [BS,C,h,w]
        if feats.ndim == 3:
            B, N, C = feats.shape
            if N != 0 and (N - 1) == int(math.sqrt(N - 1))**2:
                feats = feats[:, 1:, :]
            v = feats.mean(dim=1)             # [BS,C]
        elif feats.ndim == 4:
            v = feats.mean(dim=[2, 3])        # [BS,C]
        else:
            raise RuntimeError(f"Unexpected ViT features shape: {list(feats.shape)}")
        return self.proj(v)                   # [BS,D]


class SAMImageEncoderVec(nn.Module):
    """
    Extracts features with a ViT on full-resolution inputs; removes the CLS token if present, aggregates by mean pooling, and projects to out_dim.
    """
    def __init__(self, in_ch=5, out_dim=256,
                 vit_name="vit_base_patch16_224",
                 pretrained=False,
                 vit_img_size=256):
        super().__init__()
        if not _HAS_TIMM:
            raise RuntimeError("timm is required for SAMImageEncoderVec")
        self.vit = timm.create_model(
            vit_name, pretrained=pretrained, in_chans=in_ch, num_classes=0, img_size=vit_img_size
        )
        self.proj = nn.Linear(self.vit.num_features, out_dim)

    def forward(self, x: torch.Tensor):
        feats = self.vit.forward_features(x)  # [BS,N,C] or [BS,C,h,w]
        if feats.ndim == 4:
            v = feats.mean(dim=[2, 3])            # [BS,C]
        elif feats.ndim == 3:
            B, N, C = feats.shape
            if N != 0 and (N - 1) == int(math.sqrt(N - 1))**2:
                feats = feats[:, 1:, :]           # drop CLS
            v = feats.mean(dim=1)                  # [BS,C]
        else:
            raise RuntimeError(f"Unexpected ViT features shape: {list(feats.shape)}")
        return self.proj(v)                         # [BS,D]


def build_slice_encoder(name: str, in_ch: int, out_dim: int) -> nn.Module:
    """
    Constructs the per-slice encoder module while keeping the downstream pipeline unchanged.
    """
    n = name.lower()
    if n == "cnn":
        return CNNPerSliceStem(in_ch, out_dim)
    if n == "unet":
        return UNetEncoder2D(in_ch, base=32, out_dim=out_dim)
    if n == "resunet":
        return ResUNetEncoder2D(in_ch, base=32, out_dim=out_dim)
    if n == "transunet":
        return TransUNetEncoderVec(in_ch=in_ch, out_dim=out_dim,
                                   vit_name="vit_base_patch16_224", pretrained=False, vit_img_size=64)
    if n == "sam":
        return SAMImageEncoderVec(in_ch=in_ch, out_dim=out_dim,
                                  vit_name="vit_base_patch16_224", pretrained=False, vit_img_size=256)
    raise ValueError(f"Unknown encoder_name={name}")


# ========= 3) Transformer and pooling (legacy core) =========
class TransformerBlock(nn.Module):
    """Applies a Transformer encoder over slice tokens with configurable depth and heads."""
    def __init__(self, d_model=512, nhead=8, dim_ff=1024, dropout=0.1, num_layers=4):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class AttnPool(nn.Module):
    """Aggregates sequence features with a learnable query via attention weighting."""
    def __init__(self, d_model=512):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.query.expand(B, -1, -1)
        attn = torch.softmax((q @ x.transpose(1, 2)) / math.sqrt(D), dim=-1)
        g = attn @ x
        return g.squeeze(1)  # [B,D]


class Slice2MeshOldCore(nn.Module):
    """
    Converts radial slice stacks to GHD parameters via per-slice encoding, positional encoding, optional Transformer refinement, and pooled representation; predicts GH, R, s, T with MLP heads.
    """
    def __init__(self,
                 S:int=37, in_per_slice:int=5, num_basis:int=49,
                 encoder_name:str="cnn", d_model:int=512,
                 use_radial_trans:bool=True, nhead:int=8, dim_ff:int=1024, num_layers:int=4,
                 dropout:float=0.1):
        super().__init__()
        self.S = int(S)
        self.in_per_slice = int(in_per_slice)
        self.num_basis = int(num_basis)
        self.d_model = int(d_model)
        self.use_rt = bool(use_radial_trans)

        # Per-slice encoder
        self.slice_encoder = build_slice_encoder(encoder_name, in_ch=in_per_slice, out_dim=d_model)

        # Positional encoding
        self.posenc = RadialPositionalEncoding(S=self.S, d_model=d_model, learnable=True, use_sin=True)

        # Radial Transformer
        if self.use_rt:
            self.trans = TransformerBlock(d_model=d_model, nhead=nhead, dim_ff=dim_ff, dropout=dropout, num_layers=num_layers)
            self.pool = AttnPool(d_model=d_model)
        else:
            self.trans = None
            self.pool = None

        # Regression heads
        self.head = nn.Sequential(nn.Linear(d_model, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, 512), nn.GELU())
        self.head_GH = nn.Linear(512, num_basis * 3)
        self.head_R  = nn.Linear(512, 3)
        self.head_s  = nn.Linear(512, 1)
        self.head_T  = nn.Linear(512, 3)

    def forward(self, x: torch.Tensor, stats: Optional[dict] = None):
        """
        x: [B,S,C,H,W]  →  outputs: GH [B,K,3], R [B,3], s [B,1], T [B,3]
        """
        B, S, C, H, W = x.shape
        assert S == self.S, f"S mismatch: got {S}, expect {self.S}"
        assert C == self.in_per_slice, f"in_per_slice mismatch: got {C}, expect {self.in_per_slice}"

        if stats is not None:
            mean = float(stats.get('img_mean', 0.0)); std = float(stats.get('img_std', 1.0))
            x = (x - mean) / (std + 1e-6)

        xs = x.reshape(B * S, C, H, W)
        tok = self.slice_encoder(xs).view(B, S, -1)     # [B,S,D]

        pe = self.posenc(B, device=x.device)            # [B,S,D]
        y = tok + pe
        if self.use_rt:
            y = self.trans(y)                           # [B,S,D]
            g = self.pool(y)                            # [B,D]
        else:
            g = y.mean(dim=1)

        h  = self.head(g)                               # [B,512]
        GH = self.head_GH(h).reshape(B, self.num_basis, 3)
        R  = self.head_R(h)
        s  = F.softplus(self.head_s(h)) + 1e-3
        T  = self.head_T(h)
        return GH, R, s, T


# ========= 4) Builder and configuration =========
@dataclass
class ModelCfg:
    encoder_name: str = "cnn"     # cnn | unet | resunet | transunet | sam
    S: int = 37
    in_per_slice: int = 5
    num_basis: int = 49
    d_model: int = 512
    use_radial_trans: bool = True
    nhead: int = 8
    dim_ff: int = 1024
    num_layers: int = 4
    dropout: float = 0.1

def build_model(cfg: ModelCfg) -> nn.Module:
    return Slice2MeshOldCore(
        S=cfg.S, in_per_slice=cfg.in_per_slice, num_basis=cfg.num_basis,
        encoder_name=cfg.encoder_name, d_model=cfg.d_model,
        use_radial_trans=cfg.use_radial_trans, nhead=cfg.nhead, dim_ff=cfg.dim_ff,
        num_layers=cfg.num_layers, dropout=cfg.dropout
    )