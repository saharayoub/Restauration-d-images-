"""
server.py — Flask backend for Image Denoising Studio.
SwinIR color denoising, CPU by default, optional CUDA via USE_CUDA=1.
"""

import os
import io
import math
import base64
import time
import traceback
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from skimage.metrics import structural_similarity as ssim_func

from download_model import get_model_path, MAX_IMAGE_SIZE

# ── Device ────────────────────────────────────────────────────────────────────
USE_CUDA = os.environ.get("USE_CUDA", "0").strip() == "1"
device   = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="front-end/dist", static_url_path="")

# ── Tiling ────────────────────────────────────────────────────────────────────
TILE_SIZE    = 512
TILE_OVERLAP = 32


# =============================================================================
# SwinIR — self-contained, matches 005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth
# =============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features    = out_features    or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """B H W C  →  (B*nW) Wh Ww C"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
                   W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """(B*nW) Wh Ww C  →  B H W C"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                        window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size   # (Wh, Ww)
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0]-1) * (2*window_size[1]-1), num_heads))

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten  = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0]*self.window_size[1],
            self.window_size[0]*self.window_size[1], -1).permute(2,0,1).contiguous()
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        return self.proj_drop(self.proj((attn @ v).transpose(1,2).reshape(B_, N, C)))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.window_size      = window_size
        self.shift_size       = shift_size

        self.norm1     = norm_layer(dim)
        self.attn      = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio),
                             act_layer=act_layer, drop=drop)

        # Attention mask is built dynamically per forward call based on actual x_size,
        # so we do NOT pre-build it in __init__ (avoids resolution mismatch at inference).

    def _build_attn_mask(self, H: int, W: int) -> Optional[torch.Tensor]:
        """Build cyclic-shift attention mask for the given spatial size."""
        if self.shift_size == 0:
            return None
        ws = self.window_size
        img_mask = torch.zeros(1, H, W, 1, device=next(self.parameters()).device)
        h_slices = (slice(0, -ws), slice(-ws, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -ws), slice(-ws, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws).view(-1, ws*ws)
        attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor, x_size: Tuple[int,int]) -> torch.Tensor:
        H, W    = x_size
        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x).view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))

        # window attention
        mask      = self._build_attn_mask(H, W)
        x_windows = window_partition(x, self.window_size).view(-1, self.window_size**2, C)
        x_windows = self.attn(x_windows, mask=mask)

        # reverse window + shift
        x = window_reverse(
            x_windows.view(-1, self.window_size, self.window_size, C),
            self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1,2))

        x = shortcut + self.drop_path(x.view(B, H*W, C))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) \
                          if downsample else None

    def forward(self, x: torch.Tensor, x_size: Tuple[int,int]) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, x_size)
        if self.downsample:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """B×C×H×W  →  B×HW×C, with optional LayerNorm."""
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: B×C×H×W
        x = x.flatten(2).transpose(1, 2)                  # B×HW×C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """B×HW×C  →  B×C×H×W."""
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: Tuple[int,int]) -> torch.Tensor:
        B, HW, C = x.shape
        return x.transpose(1, 2).view(B, C, x_size[0], x_size[1])


class RSTB(nn.Module):
    """Residual Swin Transformer Block."""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection="1conv"):
        super().__init__()
        self.residual_group = BasicLayer(
            dim=dim, input_resolution=input_resolution, depth=depth,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim//4, 3, 1, 1), nn.LeakyReLU(0.2, True),
                nn.Conv2d(dim//4, dim//4, 1, 1, 0), nn.LeakyReLU(0.2, True),
                nn.Conv2d(dim//4, dim, 3, 1, 1))

        # No norm on RSTB-level patch_embed — checkpoint has no keys for it
        self.patch_embed   = PatchEmbed(img_size, patch_size, 0, dim, None)
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, 0, dim, None)

    def forward(self, x: torch.Tensor, x_size: Tuple[int,int]) -> torch.Tensor:
        # x: B×HW×C
        feat    = self.residual_group(x, x_size)       # B×HW×C
        spatial = self.patch_unembed(feat, x_size)     # B×C×H×W
        spatial = self.conv(spatial)                   # B×C×H×W
        return self.patch_embed(spatial) + x           # B×HW×C


class SwinIR(nn.Module):
    """
    SwinIR for color image denoising.
    Weights: eugenesiow/SwinIR  →  005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth
    """
    def __init__(self, img_size=128, patch_size=1, in_chans=3,
                 embed_dim=180, depths=(6,6,6,6,6,6), num_heads=(6,6,6,6,6,6),
                 window_size=8, mlp_ratio=2.0, qkv_bias=True, qk_scale=None,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=1, img_range=1.0,
                 upsampler="", resi_connection="1conv", **kwargs):
        super().__init__()

        # ── Saved hyperparameters needed at inference time ─────────────────────
        self.img_range   = img_range
        self.upsampler   = upsampler
        self.upscale     = upscale
        self.window_size = window_size      # FIX: was missing → AttributeError in forward
        self.embed_dim   = embed_dim

        # RGB mean — registered as buffer so it moves to the right device automatically
        if in_chans == 3:
            self.register_buffer("mean",
                torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1))
        else:
            self.register_buffer("mean", torch.zeros(1, 1, 1, 1))

        # ── Shallow feature extraction ────────────────────────────────────────
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Top-level patch_embed WITH norm  (checkpoint has patch_embed.norm.weight/bias)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        # ── Deep feature extraction (RSTB stack) ──────────────────────────────
        self.num_layers = len(depths)
        self.layers = nn.ModuleList([
            RSTB(dim=embed_dim,
                 input_resolution=(img_size, img_size),
                 depth=depths[i], num_heads=num_heads[i],
                 window_size=window_size, mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate,
                 drop_path=drop_path_rate,
                 norm_layer=norm_layer, downsample=None,
                 use_checkpoint=use_checkpoint,
                 img_size=img_size, patch_size=patch_size,
                 resi_connection=resi_connection)
            for i in range(self.num_layers)
        ])
        self.norm = norm_layer(embed_dim)

        # ── Reconstruction head ───────────────────────────────────────────────
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last       = nn.Conv2d(embed_dim, in_chans,  3, 1, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _pad_to_window(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Pad spatial dims to multiples of window_size. Returns (padded, H_orig, W_orig)."""
        _, _, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, H, W

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """conv_first → patch_embed → RSTB layers → norm → spatial."""
        x_size = (x.shape[2], x.shape[3])

        x   = self.conv_first(x)                           # B×embed×H×W
        res = x                                             # skip connection
        seq = self.patch_embed(x)                          # B×HW×C

        for layer in self.layers:
            seq = layer(seq, x_size)

        seq = self.norm(seq)                               # B×HW×C
        feat = seq.transpose(1,2).view(-1, self.embed_dim, x_size[0], x_size[1])
        return self.conv_after_body(feat) + res            # B×embed×H×W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad to window_size multiple, remember original size for cropping
        x, H_orig, W_orig = self._pad_to_window(x)

        x = (x - self.mean) * self.img_range

        # Global residual: input + reconstruction from deep features
        x = x + self.conv_last(self._forward_features(x))

        x = x / self.img_range + self.mean
        x = x.clamp(0.0, 1.0)

        # Crop padding back off
        return x[:, :, :H_orig, :W_orig]


# =============================================================================
# Model loading
# =============================================================================

_model: Optional[SwinIR] = None


def _load_model() -> SwinIR:
    m = SwinIR(
        upscale=1, in_chans=3, img_size=128, window_size=8, img_range=1.0,
        depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6],
        mlp_ratio=2, upsampler="", resi_connection="1conv",
    ).to(device)

    model_path = get_model_path()
    print(f"  Loading weights from {model_path} …")
    sd = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(sd, dict):
        sd = sd.get("params", sd.get("state_dict", sd))
    sd = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}

    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"  ⚠  Missing keys    : {len(missing)} — {missing[:3]}")
    if unexpected:
        print(f"  ⚠  Unexpected keys : {len(unexpected)} — {unexpected[:3]}")
    if not missing and not unexpected:
        print("  ✓ All keys matched exactly")

    m.eval()
    print(f"  ✓ SwinIR ready on {device}")
    return m


def get_model() -> SwinIR:
    global _model
    if _model is None:
        _model = _load_model()
    return _model


# =============================================================================
# Metrics
# =============================================================================

def calculate_psnr(original: np.ndarray, denoised: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float64) - denoised.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(original: np.ndarray, denoised: np.ndarray) -> float:
    o = original.astype(np.float64) / 255.0
    d = denoised.astype(np.float64)  / 255.0
    return float(np.mean([
        ssim_func(o[:,:,c], d[:,:,c], data_range=1.0) for c in range(3)
    ]))


# =============================================================================
# Inference
# =============================================================================

def _run_tile(tile_np: np.ndarray, model: SwinIR) -> np.ndarray:
    """H×W×3 float32 [0,1]  →  H×W×3 float32 [0,1]"""
    t = torch.from_numpy(tile_np).permute(2,0,1).unsqueeze(0).float().to(device)
    if device.type == "cuda":
        with torch.amp.autocast("cuda"), torch.no_grad():
            out = model(t)
    else:
        with torch.no_grad():
            out = model(t)
    return out.squeeze(0).permute(1,2,0).cpu().numpy()


def _tile_positions(dim: int, tile: int, stride: int) -> List[int]:
    """Return list of tile start positions covering [0, dim)."""
    if dim <= tile:
        return [0]
    pts: List[int] = []
    start = 0
    while start + tile < dim:
        pts.append(start)
        start += stride
    pts.append(dim - tile)          # guarantee full coverage at the end
    return sorted(set(pts))


def _denoise_direct(img_f32: np.ndarray, model: SwinIR) -> np.ndarray:
    return np.clip(_run_tile(img_f32, model) * 255.0, 0, 255).astype(np.uint8)


def _denoise_tiled(img_f32: np.ndarray, model: SwinIR) -> np.ndarray:
    h, w    = img_f32.shape[:2]
    out_acc = np.zeros((h, w, 3), dtype=np.float32)
    wgt_acc = np.zeros((h, w, 1), dtype=np.float32)
    stride  = TILE_SIZE - TILE_OVERLAP

    def blend_weights() -> np.ndarray:
        bw = np.ones((TILE_SIZE, TILE_SIZE, 1), dtype=np.float32)
        f  = TILE_OVERLAP // 2
        for i in range(f):
            v = (i + 1) / (f + 1)
            bw[i,  :] *= v;  bw[-(i+1), :] *= v
            bw[:, i]  *= v;  bw[:, -(i+1)] *= v
        return bw

    bw = blend_weights()

    for y in _tile_positions(h, TILE_SIZE, stride):
        for x in _tile_positions(w, TILE_SIZE, stride):
            tile = img_f32[y:y+TILE_SIZE, x:x+TILE_SIZE]
            # Pad tile to TILE_SIZE if image is smaller than TILE_SIZE
            th, tw = tile.shape[:2]
            if th < TILE_SIZE or tw < TILE_SIZE:
                pad = np.pad(tile, ((0, TILE_SIZE-th), (0, TILE_SIZE-tw), (0,0)), mode="reflect")
                out = _run_tile(pad, model)[:th, :tw]
            else:
                out = _run_tile(tile, model)
            out_acc[y:y+th, x:x+tw] += out * bw[:th, :tw]
            wgt_acc[y:y+th, x:x+tw] += bw[:th, :tw]

    return np.clip(out_acc / np.maximum(wgt_acc, 1e-8) * 255.0, 0, 255).astype(np.uint8)


def process_image(img: Image.Image) -> Tuple[np.ndarray, float, float]:
    w, h = img.size
    if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
        raise ValueError(
            f"Image {w}×{h} exceeds max {MAX_IMAGE_SIZE}×{MAX_IMAGE_SIZE} px."
        )
    model    = get_model()
    original = np.array(img, dtype=np.uint8)
    img_f32  = original.astype(np.float32) / 255.0

    denoised = _denoise_direct(img_f32, model) if (w <= TILE_SIZE and h <= TILE_SIZE) \
               else _denoise_tiled(img_f32, model)

    return denoised, calculate_psnr(original, denoised), calculate_ssim(original, denoised)


# =============================================================================
# Routes
# =============================================================================

@app.route("/api/health")
def health_check():
    return jsonify({
        "status":         "ok",
        "device":         str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name":       torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })


@app.route("/api/denoise", methods=["POST"])
def denoise():
    start = time.time()

    if "image" not in request.files:
        return jsonify({"error": "No image provided. Use field name 'image'."}), 400

    try:
        img = Image.open(request.files["image"].stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    w, h = img.size
    if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
        return jsonify({"error": (
            f"Image {w}×{h} exceeds the maximum allowed size "
            f"({MAX_IMAGE_SIZE}×{MAX_IMAGE_SIZE} px). Please resize first."
        )}), 413

    try:
        denoised_arr, psnr, ssim = process_image(img)

        buf = io.BytesIO()
        Image.fromarray(denoised_arr).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "denoised":        f"data:image/png;base64,{b64}",
            "psnr":            round(psnr, 2),
            "ssim":            round(ssim, 4),
            "processing_time": round(time.time() - start, 2),
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Inference failed. Check server logs."}), 500


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


# =============================================================================
# Entry point
# =============================================================================

def start(port: int = 8000) -> None:
    _load_model()
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)


if __name__ == "__main__":
    start()