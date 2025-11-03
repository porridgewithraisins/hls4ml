#!/usr/bin/env python3
"""Parse HLS Mini U-Net simulation logs and render comparison artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HLS results.log into visual artifacts")
    parser.add_argument(
        "--results-log",
        type=Path,
        default=Path("hls4ml_prj_unet/tb_data/results.log"),
        help="results.log emitted by the oneAPI HLS project",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("unet-results/torch/metadata.json"),
        help="Metadata JSON produced by generate_unet_torch_outputs.py",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("unet-results/hls"))
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--overlay-alpha", type=float, default=None)
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> dict[str, object]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return json.loads(metadata_path.read_text())


def extract_floats(results_path: Path) -> np.ndarray:
    if not results_path.exists():
        raise FileNotFoundError(f"results.log not found: {results_path}")
    text = results_path.read_text()
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    if not matches:
        raise ValueError(f"No numeric tokens found in {results_path}")
    return np.array(matches, dtype=np.float32)


def resolve_resample() -> int:
    try:
        return Image.Resampling.NEAREST
    except AttributeError:  # pragma: no cover - Pillow < 9.1
        return Image.NEAREST


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


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)

    torch_output_dir = Path(metadata["torch_output_dir"])  # type: ignore[index]
    mask_threshold = args.mask_threshold if args.mask_threshold is not None else float(metadata.get("mask_threshold", 0.5))
    overlay_alpha = args.overlay_alpha if args.overlay_alpha is not None else float(metadata.get("overlay_alpha", 0.45))

    samples = metadata.get("samples", [])  # type: ignore[assignment]
    if not samples:
        raise ValueError("Metadata contains no samples; run the torch reference script first")

    image_size = int(metadata.get("image_size", 64))
    num_classes = int(metadata.get("num_classes", 1))
    elements_per_sample = image_size * image_size * num_classes

    raw_vals = extract_floats(args.results_log)
    sample_count = len(samples)
    expected = sample_count * elements_per_sample
    if len(raw_vals) < expected:
        raise ValueError(f"results.log has {len(raw_vals)} values; expected at least {expected}. Did the simulation finish?")
    if len(raw_vals) > expected:
        raw_vals = raw_vals[:expected]

    logits_nhwc = raw_vals.reshape(sample_count, image_size, image_size, num_classes)
    probs_nhwc = 1.0 / (1.0 + np.exp(-logits_nhwc))

    output_dir = args.output_dir
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    prob_dir = output_dir / "probability_maps"

    for directory in (output_dir, mask_dir, overlay_dir, prob_dir):
        directory.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "hls_logits_nhwc.npy", logits_nhwc)
    np.save(output_dir / "hls_probs_nhwc.npy", probs_nhwc)

    torch_artifacts = metadata.get("artifacts", {})  # type: ignore[assignment]
    torch_logits_rel = torch_artifacts.get("torch_logits_nhwc")  # type: ignore[assignment]
    torch_logits_path = (torch_output_dir / torch_logits_rel) if torch_logits_rel else None

    metrics: dict[str, object] = {
        "mask_threshold": mask_threshold,
        "overlay_alpha": overlay_alpha,
        "num_samples": sample_count,
        "image_size": image_size,
    }

    if torch_logits_path and torch_logits_path.exists():
        torch_logits = np.load(torch_logits_path)
        if torch_logits.shape != logits_nhwc.shape:
            raise ValueError(f"Shape mismatch: torch logits {torch_logits.shape} vs HLS logits {logits_nhwc.shape}")
        diff = logits_nhwc - torch_logits
        global_mae = float(np.mean(np.abs(diff)))
        global_max = float(np.max(np.abs(diff)))
        per_sample = []
        for idx, sample in enumerate(samples):
            sample_diff = diff[idx]
            per_sample.append(
                {
                    "name": sample["name"],  # type: ignore[index]
                    "mae": float(np.mean(np.abs(sample_diff))),
                    "max_abs": float(np.max(np.abs(sample_diff))),
                }
            )
        metrics["logit_mae"] = global_mae
        metrics["logit_max_abs"] = global_max
        metrics["per_sample"] = per_sample

    repainted = resolve_resample()
    for idx, sample in enumerate(samples):
        name = sample["name"]  # type: ignore[index]
        base_image_path = torch_output_dir / sample["resized_image"]  # type: ignore[index]
        if not base_image_path.exists():
            raise FileNotFoundError(f"Missing resized image for overlay: {base_image_path}")

        with Image.open(base_image_path) as raw:
            base_img = raw.convert("RGB")
        prob_map = probs_nhwc[idx, ..., 0]
        mask_binary = (prob_map >= mask_threshold).astype(np.uint8)

        mask_img = Image.fromarray((mask_binary * 255).astype(np.uint8), mode="L")
        mask_img = mask_img.resize(base_img.size, resample=repainted)
        mask_img.save(mask_dir / f"{name}_mask.png")

        prob_img = Image.fromarray((prob_map * 255.0).round().astype(np.uint8), mode="L")
        prob_img = prob_img.resize(base_img.size, resample=repainted)
        prob_img.save(prob_dir / f"{name}_prob.png")

        overlay_img = make_overlay(base_img, prob_map, overlay_alpha, mask_threshold)
        overlay_img.save(overlay_dir / f"{name}_overlay.png")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Parsed {sample_count} samples from {args.results_log}")
    if "logit_mae" in metrics:
        print(
            "Logit MAE={mae:.6f}, MaxAbs={mx:.6f}".format(
                mae=metrics["logit_mae"],
                mx=metrics["logit_max_abs"],
            )
        )
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
