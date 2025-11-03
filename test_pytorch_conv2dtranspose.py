import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model


class SimpleModel(nn.Module):
    """ConvTranspose2d stress test with larger channel counts."""

    def __init__(self):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels=7,
            out_channels=5,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )

    def forward(self, x):
        return self.conv_t(x)


def save_flat(arr: np.ndarray, path: str) -> None:
    """Persist a flattened array as space-separated floats."""
    with open(path, "w") as f:
        for val in arr.ravel():
            f.write(f"{val} ")
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="oneAPI ConvTranspose2d sweep harness")
    parser.add_argument(
        "--strategy",
        default=os.getenv("HLS_STRATEGY", "resource"),
        help="Layer strategy override (e.g. resource, latency)",
    )
    parser.add_argument(
        "--reuse-factor",
        type=int,
        default=int(os.getenv("HLS_REUSE_FACTOR", "1")),
        help="Model and layer reuse factor",
    )
    parser.add_argument(
        "--parallelization-factor",
        type=int,
        default=int(os.getenv("HLS_PARALLELIZATION_FACTOR", "1")),
        help="ConvTranspose filter parallelization lanes",
    )
    parser.add_argument(
        "--io-type",
        default=os.getenv("HLS_IO_TYPE", "io_parallel"),
        choices=["io_parallel", "io_stream"],
        help="Top-level I/O interface style",
    )

    args = parser.parse_args()

    torch.manual_seed(42)

    model = SimpleModel()
    x = torch.randn(1, 7, 6, 6)

    with torch.no_grad():
        torch_output = model(x)

    os.makedirs("hls4ml_prj/tb_data", exist_ok=True)

    x_channels_last = np.transpose(x.detach().numpy(), (0, 2, 3, 1))
    torch_channels_last = np.transpose(torch_output.detach().numpy(), (0, 2, 3, 1))

    input_file = "hls4ml_prj/tb_data/tb_input_features.dat"
    output_file = "hls4ml_prj/tb_data/tb_output_predictions.dat"

    save_flat(x_channels_last, input_file)
    save_flat(torch_channels_last, output_file)

    config = config_from_pytorch_model(model, input_shape=(7, 6, 6), granularity="name")

    strategy = args.strategy.lower()
    reuse_factor = max(1, args.reuse_factor)
    parallelization = max(1, args.parallelization_factor)

    config.setdefault("Model", {})
    config.setdefault("LayerName", {})
    layer_cfg = config["LayerName"].setdefault("conv_t", {})

    config["Model"]["ChannelsLastConversion"] = "internal"
    config["Model"]["Strategy"] = strategy
    config["Model"]["ReuseFactor"] = reuse_factor
    config["Model"]["IOType"] = args.io_type
    config["IOType"] = args.io_type
    config["InputData"] = input_file
    config["OutputPredictions"] = output_file

    layer_cfg["Strategy"] = strategy
    layer_cfg["ReuseFactor"] = reuse_factor
    layer_cfg["ParallelizationFactor"] = parallelization

    print("Applied hls4ml overrides:")
    print(f"  IOType={config['IOType']}")
    print(f"  Strategy={strategy}")
    print(f"  ReuseFactor={reuse_factor}")
    print(f"  ParallelizationFactor={parallelization}")

    hls_model = convert_from_pytorch_model(
        model,
        output_dir="hls4ml_prj",
        backend="oneAPI",
        hls_config=config,
    )

    hls_model.write()

    print("PyTorch output shape:", torch_output.shape)
    print("HLS project written to:", hls_model.config.get_output_dir())
