#!/usr/bin/env python3
"""Compute MAE, MSE, and max absolute error between two flat text dumps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

# copilot: sanity check MAE MSE max abs error between two files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference",
        nargs="?",
        default="tb_data/tb_output_predictions.dat",
        help="Path to the reference (PyTorch) output file.",
    )
    parser.add_argument(
        "candidate",
        nargs="?",
        default="tb_data/results.log",
        help="Path to the candidate (HLS/oneAPI) output file.",
    )
    return parser.parse_args()


def load_values(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    data: list[float] = []
    with path.open() as handle:
        for line in handle:
            # Guard against stray whitespace
            values = [float(token) for token in line.split() if token]
            data.extend(values)
    if not data:
        raise ValueError(f"No numeric data found in {path}")
    return np.asarray(data, dtype=float)


def main() -> None:
    args = parse_args()
    ref_path = Path(args.reference)
    cand_path = Path(args.candidate)

    reference = load_values(ref_path)
    candidate = load_values(cand_path)

    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {candidate.shape}")

    diff = candidate - reference
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff ** 2)
    max_abs = float(np.max(np.abs(diff)))

    print(f"Count {reference.size}")
    print(f"MAE {mae}")
    print(f"MSE {mse}")
    print(f"MaxAbs {max_abs}")


if __name__ == "__main__":
    main()
