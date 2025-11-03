#!/usr/bin/env python3
"""Run Mini U-Net on sample images and stash reference artifacts for HLS comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_mini_unet import MiniUNet


try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # Pillow 9.1+
except AttributeError:  # pragma: no cover - Pillow < 9.1 fallback
    RESAMPLE_BILINEAR = Image.BILINEAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Mini U-Net torch references for HLS validation")
    parser.add_argument("--model", type=Path, default=Path("mini_unet_oxfordpet.pt"))
    parser.add_argument("--image-dir", type=Path, default=Path("sampleimages"))
    parser.add_argument("--output-dir", type=Path, default=Path("unet-results/torch"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device selection")
    return parser.parse_args()


def resolve_device(selector: str) -> torch.device:
    if selector == "cpu":
        return torch.device("cpu")
    if selector == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: Path, base_channels: int, device: torch.device) -> MiniUNet:
    checkpoint = torch.load(model_path, map_location=device)
    model = MiniUNet(base_channels=base_channels).to(device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_images(image_dir: Path) -> list[tuple[Path, Image.Image]]:
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not image_paths:
        raise FileNotFoundError(f"No images with common extensions found in {image_dir}")
    items: list[tuple[Path, Image.Image]] = []
    for path in image_paths:
        with Image.open(path) as raw:
            img = raw.convert("RGB")
        items.append((path, img))
    return items


def make_overlay(base_rgb: Image.Image, prob: np.ndarray, alpha: float, threshold: float) -> Image.Image:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    base = base_rgb.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    if alpha <= 0.0:
        return base.convert("RGB")
    denom = max(1e-6, 1.0 - threshold)
    activated = np.where(prob >= threshold, (prob - threshold) / denom, 0.0)
    overlay_alpha = (activated * alpha * 255.0).clip(0, 255).round().astype(np.uint8)
    overlay.putalpha(Image.fromarray(overlay_alpha, mode="L"))
    return Image.alpha_composite(base, overlay).convert("RGB")


def save_dat(path: Path, tensor_nhwc: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for sample in tensor_nhwc:
            flat = sample.reshape(-1)
            handle.write(" ".join(f"{val:.8f}" for val in flat))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    model = load_model(args.model, args.base_channels, device)
    images = collect_images(args.image_dir)

    output_dir = args.output_dir
    resized_dir = output_dir / "resized"
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    prob_vis_dir = output_dir / "probability_maps"

    for directory in (output_dir, resized_dir, mask_dir, overlay_dir, prob_vis_dir):
        directory.mkdir(parents=True, exist_ok=True)

    inputs_nchw: list[np.ndarray] = []
    logits_nchw: list[np.ndarray] = []
    probs_nchw: list[np.ndarray] = []
    metadata: dict[str, object] = {
        "image_size": args.image_size,
        "num_classes": 1,
        "input_channels": 3,
        "torch_output_dir": str(output_dir.resolve()),
        "mask_threshold": args.mask_threshold,
        "overlay_alpha": args.overlay_alpha,
        "samples": [],
        "artifacts": {},
    }

    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        for idx, (img_path, pil_img) in enumerate(images):
            stem = img_path.stem
            resized = pil_img.resize((args.image_size, args.image_size), RESAMPLE_BILINEAR)
            arr = np.array(resized, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1)).copy()
            tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = sigmoid(logits)

            inputs_nchw.append(tensor.squeeze(0).cpu().numpy())
            logits_nchw.append(logits.squeeze(0).cpu().numpy())
            probs_nchw.append(probs.squeeze(0).cpu().numpy())

            resized_path = resized_dir / f"{stem}.png"
            mask_path = mask_dir / f"{stem}_mask.png"
            overlay_path = overlay_dir / f"{stem}_overlay.png"
            prob_path = prob_vis_dir / f"{stem}_prob.png"

            resized.save(resized_path)

            prob_map = probs.squeeze(0).squeeze(0).cpu().numpy()
            mask_binary = (prob_map >= args.mask_threshold).astype(np.uint8)

            mask_img = Image.fromarray((mask_binary * 255).astype(np.uint8), mode="L")
            mask_img.save(mask_path)

            prob_img = Image.fromarray((prob_map * 255.0).round().astype(np.uint8), mode="L")
            prob_img.save(prob_path)

            overlay_img = make_overlay(resized, prob_map, args.overlay_alpha, args.mask_threshold)
            overlay_img.save(overlay_path)

            metadata["samples"].append(
                {
                    "name": stem,
                    "index": idx,
                    "original_image": img_path.name,
                    "resized_image": f"resized/{stem}.png",
                    "torch_mask": f"masks/{stem}_mask.png",
                    "torch_overlay": f"overlays/{stem}_overlay.png",
                    "probability_map": f"probability_maps/{stem}_prob.png",
                }
            )

    inputs_nchw_array = np.stack(inputs_nchw, axis=0)
    logits_nchw_array = np.stack(logits_nchw, axis=0)
    probs_nchw_array = np.stack(probs_nchw, axis=0)

    inputs_nhwc = np.transpose(inputs_nchw_array, (0, 2, 3, 1))
    logits_nhwc = np.transpose(logits_nchw_array, (0, 2, 3, 1))
    probs_nhwc = np.transpose(probs_nchw_array, (0, 2, 3, 1))

    np.save(output_dir / "inputs_nhwc.npy", inputs_nhwc)
    np.save(output_dir / "torch_logits_nhwc.npy", logits_nhwc)
    np.save(output_dir / "torch_probs_nhwc.npy", probs_nhwc)

    input_dat = output_dir / "tb_input_features.dat"
    torch_ref_dat = output_dir / "torch_reference_logits.dat"

    save_dat(input_dat, inputs_nhwc.astype(np.float32))
    save_dat(torch_ref_dat, logits_nhwc.astype(np.float32))

    metadata["artifacts"] = {
        "inputs_nhwc": "inputs_nhwc.npy",
        "torch_logits_nhwc": "torch_logits_nhwc.npy",
        "torch_probs_nhwc": "torch_probs_nhwc.npy",
        "tb_input_features": input_dat.name,
        "torch_reference_logits": torch_ref_dat.name,
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Processed {len(images)} images")
    print(f"Wrote NHWC inputs to {input_dat}")
    print(f"Wrote torch logits to {torch_ref_dat}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
