import os
import requests
from tqdm import tqdm
from pathlib import Path

MODELS_DIR = Path(os.environ.get("HF_HOME", "./models/hf_cache"))
MAX_IMAGE_SIZE = 2048  # Maximum allowed image dimension

def ensure_dir():
    """Ensure models directory exists."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url: str, dest_path: Path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)

def download_swinir():
    """Download SwinIR denoising model."""
    ensure_dir()
    
    # Use the official SwinIR release from GitHub
    model_url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
    model_path = MODELS_DIR / "swinir_noise50.pth"
    
    if model_path.exists():
        print(f"SwinIR model already exists at {model_path}")
        return str(model_path)
    
    print("Downloading SwinIR model from official release...")
    try:
        download_file(model_url, model_path)
        print(f"Model downloaded to {model_path}")
        return str(model_path)
    except Exception as e:
        # Fallback: try alternative noise level
        print(f"Primary download failed, trying alternative...")
        alt_url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth"
        alt_path = MODELS_DIR / "swinir_noise25.pth"
        download_file(alt_url, alt_path)
        return str(alt_path)

def get_model_path():
    """Get path to downloaded model, downloading if necessary."""
    model_path = MODELS_DIR / "swinir_noise50.pth"
    if not model_path.exists():
        alt_path = MODELS_DIR / "swinir_noise25.pth"
        if alt_path.exists():
            return str(alt_path)
        raise FileNotFoundError("Model not found. Run download_all() first.")
    return str(model_path)

def download_all():
    """Download all required models."""
    print("=" * 60)
    print("  Image Denoising Studio - Model Download")
    print("=" * 60)
    print("SwinIR model not found in cache. Starting download...")
    print("  → Downloading SwinIR weights from official GitHub releases…")
    
    path = download_swinir()
    
    print("=" * 60)
    print("  Model download complete!")
    print(f"  Location: {path}")
    print("=" * 60)
    return path

if __name__ == "__main__":
    download_all()