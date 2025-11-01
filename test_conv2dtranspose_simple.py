import torch
import torch.nn as nn
import numpy as np
from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels=2,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, x):
        return self.conv_t(x)


model = SimpleModel()

with torch.no_grad():
    model.conv_t.weight.zero_()
    model.conv_t.weight[0, 0, :, :] = torch.tensor(
        [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [2.0, 1.0, 1.0]]
    )
    model.conv_t.weight[1, 0, :, :] = torch.tensor(
        [[1.0, 0.0, 1.0], [2.0, 2.0, 3.0], [0.0, 1.0, 1.0]]
    )
    model.conv_t.bias[0] = 1.0

print("Weight tensor shape:", model.conv_t.weight.shape)
print("Kernel for input channel 0 (3x3):")
print(model.conv_t.weight[0, 0].detach().numpy())
print("Kernel for input channel 1 (3x3):")
print(model.conv_t.weight[1, 0].detach().numpy())

x = torch.tensor(
    [
        [
            [[1.0, 2.0], [2.0, 1.0]],
            [[1.0, 3.0], [2.0, 2.0]],
        ]
    ]
)  # shape (1, 2, 2, 2)

print("\nInput shape:", x.shape)
print("Input channel 0 (2x2):")
print(x[0, 0].detach().numpy())
print("Input channel 1 (2x2):")
print(x[0, 1].detach().numpy())

with torch.no_grad():
    torch_output = model(x)

print("\nOutput shape:", torch_output.shape)
print("Output (2x2):")
print(torch_output[0, 0].numpy())
print("Output flattened:", torch_output[0, 0].numpy().flatten())

import os

os.makedirs("hls4ml_prj/tb_data", exist_ok=True)

x_np = x.detach().numpy()  # (1, 2, 2, 2)
x_channels_last = np.transpose(x_np, (0, 2, 3, 1))  # (1, 2, 2, 2)
x_flat = x_channels_last.flatten()
print("\nInput (channels_last, flattened):", x_flat)

with open("hls4ml_prj/tb_data/tb_input_features.dat", "w") as f:
    for val in x_flat:
        f.write(f"{val} ")
    f.write("\n")

output_np = torch_output.detach().numpy()  # (1, 1, 2, 2)
output_channels_last = np.transpose(output_np, (0, 2, 3, 1))  # (1, 2, 2, 1)
output_flat = output_channels_last.flatten()
print("Output (channels_last, flattened):", output_flat)

with open("hls4ml_prj/tb_data/tb_output_predictions.dat", "w") as f:
    for val in output_flat:
        f.write(f"{val} ")
    f.write("\n")

config = config_from_pytorch_model(model, input_shape=(2, 2, 2), granularity="name")
config["Model"]["ChannelsLastConversion"] = "internal"
config["InputData"] = "hls4ml_prj/tb_data/tb_input_features.dat"
config["OutputPredictions"] = "hls4ml_prj/tb_data/tb_output_predictions.dat"

hls_model = convert_from_pytorch_model(
    model, output_dir="hls4ml_prj", backend="Vivado", hls_config=config
)

print("\nModel conversion completed!")
print(f"Output directory: {hls_model.config.get_output_dir()}")

print("\nAttempting to write HLS code...")
try:
    hls_model.write()
    print("HLS code generation successful!")
    print(f"Check the output in: {hls_model.config.get_output_dir()}")
except Exception as e:
    print(f"Error during HLS code generation: {e}")
    import traceback

    traceback.print_exc()
