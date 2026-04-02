import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath

'''
@Author: WDTT
@Time: Apr 2026
@Project: EAAC-S2S v2
'''

# ═══════════════════════════════════════════════════════════════════
#  Encoder / Decoder  (SimVP-style ConvSC backbone)
# ═══════════════════════════════════════════════════════════════════

def sampling_generator(N, reverse=False):
    """Returns a list of N booleans marking which ConvSC layers downsample/upsample.
    Pattern [F, F, T, F] → only the 3rd layer samples."""
    samplings = [False, False, True, False] * (N // 2)
    result = samplings[:N]
    return list(reversed(result)) if reverse else result


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, upsampling=False, act_norm=False):
        super().__init__()
        self.act_norm = act_norm
        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size, 1, padding),
                nn.PixelShuffle(2)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3,
                 downsampling=False, upsampling=False, act_norm=True):
        super().__init__()
        stride  = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(C_in, C_out, kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding, act_norm=act_norm)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """4-layer ConvSC: layers 0-1 full-res, layer 2 downsamples 2×, layer 3 half-res.
    Returns (latent @ H/2×W/2,  enc1 skip @ H×W)."""
    def __init__(self, C_in, C_hid, N_S=4, spatio_kernel=3):
        super().__init__()
        samplings = sampling_generator(N_S)        # [F, F, T, F]
        self.enc = nn.Sequential(
            ConvSC(C_in,  C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)          # full-resolution skip feature
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1            # latent: (B, C_S, H/2, W/2)


class Decoder(nn.Module):
    """4-layer ConvSC: upsamples 2× at layer 2; last layer fuses enc1 skip connection."""
    def __init__(self, C_hid, C_out, N_S=4, spatio_kernel=3):
        super().__init__()
        samplings = sampling_generator(N_S, reverse=True)   # [F, T, F, F]
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s) for s in samplings[:-1]],
             ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1):
        for i in range(len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)   # skip fusion at last layer
        return self.readout(Y)


# ═══════════════════════════════════════════════════════════════════
#  Timecode Embedding  (cosine day-of-year encoding)
# ═══════════════════════════════════════════════════════════════════

class TimecodeEmbedding(nn.Module):
    """[cos(2π·doy/365), sin(2π·doy/365)] → MLP → (B, embed_dim)"""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, day_of_year):
        """day_of_year: (B,) float, values 1..365"""
        angle = 2.0 * math.pi * day_of_year / 365.0
        tc = torch.stack([angle.cos(), angle.sin()], dim=-1)   # (B, 2)
        return self.mlp(tc)                                     # (B, embed_dim)


# ═══════════════════════════════════════════════════════════════════
#  Inception Block  (parallel 3×3, 5×5, 7×7 grouped convolutions)
# ═══════════════════════════════════════════════════════════════════

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups=4):
        super().__init__()
        groups = groups if in_channels % groups == 0 else 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class InceptionBlock(nn.Module):
    """1×1 projection → three parallel grouped-conv branches (3×3, 5×5, 7×7) → element-wise sum."""
    def __init__(self, C_in, C_hid, C_out, incep_ker=(3, 5, 7), groups=4):
        super().__init__()
        self.proj     = nn.Conv2d(C_in, C_hid, 1)
        self.branches = nn.ModuleList([
            GroupConv2d(C_hid, C_out, k, k // 2, groups) for k in incep_ker
        ])

    def forward(self, x):
        x = self.proj(x)
        return sum(branch(x) for branch in self.branches)


# ═══════════════════════════════════════════════════════════════════
#  Shift & Norm  (FiLM affine modulation + LayerNorm)
# ═══════════════════════════════════════════════════════════════════

class ShiftAndNorm(nn.Module):
    """FiLM: x ← x·(1+γ) + β,  then LayerNorm over channel dim.
    γ and β are predicted from the timecode embedding."""
    def __init__(self, dim, tc_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(tc_dim, dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, tc):
        """x: (B,C,H,W),  tc: (B, tc_dim)"""
        gamma, beta = self.film(tc).chunk(2, dim=-1)            # (B,C) each
        x = x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


# ═══════════════════════════════════════════════════════════════════
#  Self-Attention  (MHSA on H×W spatial tokens)
# Patchify (p=1) is implicit: each spatial position is treated as
# one token, flattened to a sequence of length H×W inside this module.
# ═══════════════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=12, drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} must be divisible by num_heads {num_heads}'
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = drop   # passed to scaled_dot_product_attention

    def forward(self, x):
        """x: (B, C, H, W) → (B, C, H, W)"""
        B, C, H, W = x.shape
        N = H * W
        # ── Patchify: flatten spatial → token sequence ──
        tokens = x.flatten(2).transpose(1, 2)                        # (B, N, C)
        q, k, v = (
            self.qkv(tokens)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)                                  # (3, B, heads, N, head_dim)
            .unbind(0)
        )
        # Flash-attention-compatible call (PyTorch ≥ 2.0 uses memory-efficient kernel)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.
        )                                                             # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        # ── Unpatchify: restore spatial layout ──
        return out.transpose(1, 2).reshape(B, C, H, W)


# ═══════════════════════════════════════════════════════════════════
#  MixMLP  (ConvNeXt-style FFN: 1×1 → DW3×3 → GELU → 1×1)
# ═══════════════════════════════════════════════════════════════════

class MixMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop=0.):
        super().__init__()
        hidden   = int(dim * mlp_ratio)
        self.fc1  = nn.Conv2d(dim, hidden, 1)
        self.dw   = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Conv2d(hidden, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ═══════════════════════════════════════════════════════════════════
#  ST Block  (red box in diagram)
#  ShiftNorm → SelfAttn → residual
#  ShiftNorm → MixMLP  → residual
# ═══════════════════════════════════════════════════════════════════

class STBlock(nn.Module):
    def __init__(self, dim, tc_dim, num_heads=12, mlp_ratio=4,
                 drop=0., drop_path=0., init_value=1e-2):
        super().__init__()
        self.norm1 = ShiftAndNorm(dim, tc_dim)
        self.attn  = SelfAttention(dim, num_heads, drop)
        self.norm2 = ShiftAndNorm(dim, tc_dim)
        self.mlp   = MixMLP(dim, mlp_ratio, drop)
        # layer-scale parameters for training stability
        self.ls1   = nn.Parameter(init_value * torch.ones(dim))
        self.ls2   = nn.Parameter(init_value * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, tc):
        x = x + self.drop_path(self.ls1[:, None, None] * self.attn(self.norm1(x, tc)))
        x = x + self.drop_path(self.ls2[:, None, None] * self.mlp (self.norm2(x, tc)))
        return x


# ═══════════════════════════════════════════════════════════════════
#  Spatio-Temporal Processor
# ═══════════════════════════════════════════════════════════════════

class SpatioTemporalProcessor(nn.Module):
    """
    Full processor flow:
      (B, C_in, H', W')
        ─► InceptionBlock + FiLM(timecode)       [spatial domain, H'×W']
        ─► [implicit Patchify p=1]
        ─► N × STBlock(ShiftNorm+Attn, ShiftNorm+MixMLP)
        ─► [implicit Unpatchify]
        ─► out_proj
      (B, C_out, H', W')
    """
    def __init__(self, C_in, C_hid, C_out, tc_dim,
                 N_blks=16, num_heads=12, mlp_ratio=4, drop=0., drop_path=0.1):
        super().__init__()
        self.inception      = InceptionBlock(C_in, C_hid, C_hid)
        self.inception_film = nn.Linear(tc_dim, C_hid * 2)
        nn.init.zeros_(self.inception_film.weight)
        nn.init.zeros_(self.inception_film.bias)

        dp_rates = torch.linspace(0, drop_path, N_blks).tolist()
        self.blocks = nn.ModuleList([
            STBlock(C_hid, tc_dim, num_heads, mlp_ratio, drop, dp_rates[i])
            for i in range(N_blks)
        ])
        self.out_proj = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, x, tc):
        """x: (B, C_in, H', W'),  tc: (B, tc_dim)"""
        # ── Inception Block + FiLM conditioning (diagram: bottom ⊕ from MLP) ──
        x = self.inception(x)
        gamma, beta = self.inception_film(tc).chunk(2, dim=-1)   # (B, C_hid) each
        x = x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        # ── N ST Blocks (patchify/unpatchify implicit inside SelfAttention) ──
        for block in self.blocks:
            x = block(x, tc)

        return self.out_proj(x)                                   # (B, C_out, H', W')


# ═══════════════════════════════════════════════════════════════════
#  EAAC  (top-level model)
# ═══════════════════════════════════════════════════════════════════

class EAAC(nn.Module):
    """
    EAAC-S2S v2  –  autoregressive pentad forecasting model.

    Args:
        in_shape:     (T_in, C_in, H, W)   e.g. (2, 34, 66, 70)
        out_shape:    (T_out, C_out, H, W)  e.g. (1, 34, 66, 70)
        C_S:          encoder/decoder hidden channels  (default 256)
        C_T:          transformer hidden dim            (default 768)
        N_S:          number of ConvSC layers           (default 4)
        N_blks:       number of ST Blocks               (default 16)
        tc_embed_dim: timecode embedding dimension      (default 128)
        num_heads:    attention heads  (C_T must be divisible, default 12)
        mlp_ratio:    MixMLP expansion ratio            (default 4)
        drop:         dropout rate
        drop_path:    stochastic depth max rate
    """
    def __init__(self, in_shape, out_shape,
                 C_S=256, C_T=768, N_S=4, N_blks=16,
                 tc_embed_dim=128, num_heads=12, mlp_ratio=4,
                 drop=0., drop_path=0.1, **kwargs):
        super().__init__()
        T_in,  C_in,  H, W = in_shape
        T_out, C_out, _, _ = out_shape

        self.T_in  = T_in
        self.T_out = T_out
        self.C_out = C_out
        self.C_S   = C_S

        # ── Timecode ──
        self.tc_embed = TimecodeEmbedding(tc_embed_dim)

        # ── Encoder / Decoder ──
        self.encoder   = Encoder(C_in, C_S, N_S)
        self.decoder   = Decoder(C_S, C_out, N_S)
        self.skip_conv = nn.Conv2d(T_in * C_S, T_out * C_S, 1)

        # ── Spatio-Temporal Processor ──
        self.processor = SpatioTemporalProcessor(
            C_in=T_in * C_S, C_hid=C_T, C_out=T_out * C_S,
            tc_dim=tc_embed_dim, N_blks=N_blks,
            num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop=drop, drop_path=drop_path
        )

    def forward(self, x, day_of_year=None):
        B, T, C, H, W = x.shape

        if day_of_year is None:
            day_of_year = x.new_zeros(B)
        tc = self.tc_embed(day_of_year.float())         # (B, tc_embed_dim)

        x_flat       = x.reshape(B * T, C, H, W)
        latent, enc1 = self.encoder(x_flat)             # latent: (B·T, C_S, H', W')
        _, C_S, H_, W_ = latent.shape                   #         enc1:  (B·T, C_S, H,  W)

        lat_bt = latent.reshape(B, T * C_S, H_, W_)
        proc   = self.processor(lat_bt, tc)             # (B, T_out·C_S, H', W')

        _, _, H_s, W_s = enc1.shape
        enc1_bt  = enc1.reshape(B, T * C_S, H_s, W_s)
        skip     = self.skip_conv(enc1_bt)              # (B, T_out·C_S, H_s, W_s)
        skip_f   = skip.reshape(B * self.T_out, C_S, H_s, W_s)

        proc_f = proc.reshape(B * self.T_out, C_S, H_, W_)
        Y = self.decoder(proc_f, skip_f)                # (B·T_out, C_out, H, W)
        return Y.reshape(B, self.T_out, self.C_out, H, W)






if __name__ == '__main__':
    cfg_EAAC_V2 = dict(
        in_shape      = (2, 34, 66, 70),   # T_in=2, 34 variables, 66×70 grid
        out_shape     = (1, 34, 66, 70),   # autoregressive: predict next pentad
        C_S           = 256,               # encoder/decoder channels
        C_T           = 768,               # transformer hidden dim
        N_S           = 4,                 # ConvSC layers in encoder/decoder
        N_blks        = 16,                # ST Blocks
        tc_embed_dim  = 128,               # timecode embedding dim
        num_heads     = 12,                # 768 / 12 = 64 per head
        mlp_ratio     = 4,
        drop          = 0.,
        drop_path     = 0.1,
    )
    device = torch.device('cuda:0')
    torch.manual_seed(42)

    print('=' * 60)
    print('  EAAC-S2S v2  –  shape & gradient test')
    print('=' * 60)

    model = EAAC(**cfg_EAAC_V2).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters : {n_params / 1e6:.1f} M')

    B = 2
    x   = torch.randn(B, 2, 34, 66, 70, device=device)
    doy = torch.randint(1, 366, (B,), device=device).float()

    # ── Forward pass ──
    with torch.no_grad():
        out = model(x, doy)

    print(f'Input  shape : {tuple(x.shape)}')
    print(f'Output shape : {tuple(out.shape)}')
    assert out.shape == (B, 1, 34, 66, 70), f'Shape mismatch: {out.shape}'
    print('Forward pass OK')

    # ── Backward pass (required by CNOP gradient computation) ──
    x_grad = torch.randn(B, 2, 34, 66, 70, device=device).requires_grad_(True)
    out_grad = model(x_grad, doy)
    loss = out_grad.sum()
    loss.backward()
    assert x_grad.grad is not None
    print('Backward pass OK')

    # ── Autoregressive rollout (6 steps, as used in CNOP) ──
    current = x.clone()
    preds = []
    with torch.no_grad():
        for step in range(6):
            pred = model(current, doy)              # (B, 1, C, H, W)
            preds.append(pred)
            current = torch.cat([current[:, 1:], pred], dim=1)
    forecast = torch.cat(preds, dim=1)              # (B, 6, C, H, W)
    assert forecast.shape == (B, 6, 34, 66, 70)
    print(f'6-step autoregressive rollout OK  →  {tuple(forecast.shape)}')
