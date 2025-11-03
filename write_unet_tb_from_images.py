#!/usr/bin/env python3
"""Pack real images into the HLS testbench input format for Mini U-Net."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # Pillow >= 9.1
except AttributeError:  # pragma: no cover - fallback for older Pillow
    RESAMPLE_BILINEAR = Image.BILINEAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write sample images into HLS tb_input_features.dat")
    parser.add_argument("--image-dir", type=Path, default=Path("sampleimages"))
    parser.add_argument(
        "--output", type=Path, default=Path("hls4ml_prj_unet/tb_data/tb_input_features.dat"), help="Destination .dat file"
    )
    parser.add_argument("--image-size", type=int, default=64, help="Square resize dimension expected by the model")
    parser.add_argument(
        "--float-format",
        default="{:.8f}",
        help="Formatter for floating-point serialization (default: {:.8f})",
    )
    return parser.parse_args()


def collect_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not image_paths:
        raise FileNotFoundError(f"No images with .jpg/.jpeg/.png/.bmp extensions found in {image_dir}")
    return image_paths


def load_and_normalize(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as raw:
        rgb = raw.convert("RGB")
        resized = rgb.resize((size, size), RESAMPLE_BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    return arr


def save_dat(path: Path, tensor_nhwc: np.ndarray, float_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = float_format.format
    with path.open("w") as handle:
        for sample in tensor_nhwc:
            flat = sample.reshape(-1)
            handle.write(" ".join(formatter(val) for val in flat))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    image_paths = collect_images(args.image_dir)

    nhwc = np.stack([load_and_normalize(path, args.image_size) for path in image_paths], axis=0)

    # HLS testbench expects NHWC layout; Data already shaped as (N,H,W,C)
    save_dat(args.output, nhwc.astype(np.float32), args.float_format)

    print(f"Wrote {nhwc.shape[0]} samples to {args.output}")
    print(f"Shape per sample: {nhwc.shape[1:]} (H, W, C)")


if __name__ == "__main__":
    main()
