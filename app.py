"""
Image Denoising Studio — Local entry point.
Handles model download (if needed) then starts the Flask + React server.
"""

import os
import sys
from pathlib import Path

PORT       = int(os.environ.get("PORT", 8000))
MODELS_DIR = Path(os.environ.get("HF_HOME", "./models/hf_cache"))


def ensure_models():
    """Download SwinIR weights if not already cached."""
    from download_model import download_all, get_model_path
    try:
        get_model_path()
        print("SwinIR model already cached.")
    except FileNotFoundError:
        print("SwinIR model not found in cache. Starting download...")
        download_all()


def main():
    print("\n" + "=" * 60)
    print("  Image Denoising Studio")
    print("=" * 60)

    # Step 1: ensure model weights are downloaded FIRST
    # This must complete before we import server (which loads the model)
    ensure_models()

    # Step 2: now it's safe to import server (it will load the model)
    print(f"\nLoading model into memory... (this may take a moment)")
    print(f"\nOnce ready, open:  http://localhost:{PORT}\n")

    # Import server ONLY after model is guaranteed to exist
    import server
    server.start(port=PORT)


if __name__ == "__main__":
    main()