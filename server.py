import os
import io
import base64
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import convolve

# Import SwinIR implementation and download utilities
from download_model import download_all, get_model_path, MODELS_DIR, MAX_IMAGE_SIZE

app = Flask(__name__, static_folder='front-end/dist', static_url_path='')

# Device configuration
USE_CUDA = os.environ.get('USE_CUDA', '0') == '1'
device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')

# =============================================================================
# SwinIR Model Implementation (Self-contained)
# =============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                      slice(-self.window_size, -self.shift_size),
                      slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                      slice(-self.window_size, -self.shift_size),
                      slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                               num_heads=num_heads, window_size=window_size,
                               shift_size=0 if (i % 2 == 0) else window_size // 2,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop, attn_drop=attn_drop,
                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                               norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class RSTB(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.flatten(2).transpose(1, 2)

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, x_size[0], x_size[1])
        return x.permute(0, 2, 3, 1).contiguous()

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class SwinIR(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.upscale = upscale
        self.upsampler = upsampler

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(img_size, img_size),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=drop_path_rate,
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection)
            self.layers.append(layer)

        self.norm = norm_layer(embed_dim)

        if upsampler == '':
            # for image denoising
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        else:
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, x_size):
        x = self.conv_first(x)
        x_first = x
        res = self.conv_after_body(self.forward_transformer(x_first, x_size)) + x_first
        return res

    def forward_transformer(self, x, x_size):
        x_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.embed_dim, x_size[0], x_size[1])
        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == '':
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_transformer(x_first, (x.shape[2], x.shape[3]))) + x_first
            x = x + self.conv_last(res)
        else:
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_transformer(x, (x.shape[2], x.shape[3]))) + x
            x = self.conv_before_upsample(x)
            x = self.upsample(x)
            x = self.conv_last(x)

        x = x / self.img_range + self.mean
        return x

# =============================================================================
# Model Loading
# =============================================================================

def load_model():
    model = SwinIR(
        upscale=1,
        in_chans=3,
        img_size=128,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='',
        resi_connection='1conv'
    ).to(device)
    
    model_path = get_model_path()
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle different state dict formats
    if 'params' in state_dict:
        model.load_state_dict(state_dict['params'], strict=True)
    elif 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'], strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    return model

# Global model instance
model = None

def get_model():
    global model
    if model is None:
        model = load_model()
    return model

# =============================================================================
# Image Processing with Tiling
# =============================================================================

TILE_SIZE = 512
TILE_OVERLAP = 32

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR between two images (0-255 range)."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM between two images using scipy."""
    try:
        from skimage.metrics import structural_similarity as ssim_func
        
        # Convert to float in [0, 1] range
        img1_float = img1.astype(np.float64) / 255.0
        img2_float = img2.astype(np.float64) / 255.0
        
        # Calculate SSIM for each channel
        ssim_values = []
        for c in range(3):
            ssim_val = ssim_func(img1_float[:, :, c], img2_float[:, :, c], 
                                data_range=1.0)
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    except ImportError:
        # Fallback if scikit-image not available
        return 0.0


def process_image_tiled(img: Image.Image, model: nn.Module, device: torch.device) -> Tuple[Image.Image, float, float]:
    """
    Process image with tiling strategy for large images.
    Returns (denoised_image, psnr, ssim).
    """
    w, h = img.size
    
    # Check maximum size
    if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
        raise ValueError(f"Image dimensions ({w}x{h}) exceed maximum allowed size of {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE}")
    
    # Convert to numpy for metrics calculation
    img_array = np.array(img).astype(np.float32)
    
    # If image is small enough, process directly
    if w <= TILE_SIZE and h <= TILE_SIZE:
        return process_image_direct(img, model, device, img_array)
    
    # Tiled processing for larger images
    return process_image_with_tiles(img, model, device, img_array)


def process_image_direct(img: Image.Image, model: nn.Module, device: torch.device, 
                         original_array: np.ndarray) -> Tuple[Image.Image, float, float]:
    """Process small image directly without tiling."""
    img_tensor = torch.from_numpy(original_array / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            output = model(img_tensor)
    
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    
    # Calculate metrics
    psnr = calculate_psnr(original_array, output)
    ssim = calculate_ssim(original_array, output)
    
    return Image.fromarray(output), psnr, ssim


def process_image_with_tiles(img: Image.Image, model: nn.Module, device: torch.device,
                             original_array: np.ndarray) -> Tuple[Image.Image, float, float]:
    """Process large image using overlapping tiles."""
    w, h = img.size
    channels = 3
    
    # Prepare output array
    output_array = np.zeros((h, w, channels), dtype=np.float32)
    weight_array = np.zeros((h, w, channels), dtype=np.float32)
    
    # Calculate tile positions with overlap
    stride = TILE_SIZE - TILE_OVERLAP
    
    def get_tile_coords(dim_size: int) -> list:
        """Get tile start positions for a dimension."""
        if dim_size <= TILE_SIZE:
            return [(0, dim_size)]
        coords = []
        start = 0
        while start < dim_size:
            end = min(start + TILE_SIZE, dim_size)
            coords.append((start, end))
            if end == dim_size:
                break
            start += stride
            # Adjust last tile to fit exactly
            if start + TILE_SIZE > dim_size:
                start = dim_size - TILE_SIZE
                if start < 0:
                    start = 0
        return coords
    
    x_coords = get_tile_coords(w)
    y_coords = get_tile_coords(h)
    
    # Create blending weights (linear feathering at edges)
    def create_blend_weights(tile_h: int, tile_w: int) -> np.ndarray:
        weights = np.ones((tile_h, tile_w, 1), dtype=np.float32)
        # Feather edges
        feather = TILE_OVERLAP // 2
        for i in range(feather):
            weight = (i + 1) / (feather + 1)
            weights[i, :, :] *= weight
            weights[-(i+1), :, :] *= weight
            weights[:, i, :] *= weight
            weights[:, -(i+1), :] *= weight
        return weights
    
    # Process each tile
    for y_start, y_end in y_coords:
        for x_start, x_end in x_coords:
            tile_h = y_end - y_start
            tile_w = x_end - x_start
            
            # Extract tile
            tile = original_array[y_start:y_end, x_start:x_end]
            
            # Pad if necessary (shouldn't happen with our logic, but safety)
            pad_h = TILE_SIZE - tile_h
            pad_w = TILE_SIZE - tile_w
            
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            
            # Process tile
            tile_tensor = torch.from_numpy(tile / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    tile_output = model(tile_tensor)
            
            tile_output = tile_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            tile_output = np.clip(tile_output * 255.0, 0, 255)
            
            # Remove padding
            if pad_h > 0 or pad_w > 0:
                tile_output = tile_output[:tile_h, :tile_w]
            
            # Blend weights
            weights = create_blend_weights(tile_h, tile_w)
            
            # Add to output with blending
            output_array[y_start:y_end, x_start:x_end] += tile_output * weights
            weight_array[y_start:y_end, x_start:x_end] += weights
    
    # Normalize by weights
    output_array = output_array / np.maximum(weight_array, 1e-8)
    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
    
    # Calculate metrics
    psnr = calculate_psnr(original_array, output_array)
    ssim = calculate_ssim(original_array, output_array)
    
    return Image.fromarray(output_array), psnr, ssim

# =============================================================================
# API Routes
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })

@app.route('/api/denoise', methods=['POST'])
def denoise():
    start_time = time.time()
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400
    
    w, h = img.size
    
    # Check maximum size
    if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
        return jsonify({
            'error': f'Image dimensions ({w}x{h}) exceed maximum allowed size of {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE}. '
                     f'Please resize your image or use a smaller image.'
        }), 413
    
    try:
        model = get_model()
        denoised_img, psnr, ssim = process_image_tiled(img, model, device)
        
        # Convert to base64
        buffer = io.BytesIO()
        denoised_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'denoised': f'data:image/png;base64,{img_base64}',
            'psnr': round(psnr, 2),
            'ssim': round(ssim, 4),
            'processing_time': round(processing_time, 2)
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

# =============================================================================
# Server Start Function
# =============================================================================

def start(port: int = 8000):
    """Start the Flask server."""
    app.run(host='0.0.0.0', port=port, threaded=True)

if __name__ == '__main__':
    start()