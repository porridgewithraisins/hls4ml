import torch
import torch.nn as nn
import numpy as np
from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model


# Define a simple model with Conv2D
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        return self.conv(x)


# Create model instance
model = SimpleModel()

# Check the weights
print("Model weights:")
print(f"  Weight shape: {model.conv.weight.shape}")
print(
    f"  Weight values (first 10): {model.conv.weight.detach().flatten()[:10].numpy()}"
)
print(f"  Bias shape: {model.conv.bias.shape}")
print(f"  Bias values: {model.conv.bias.detach().numpy()}")

# Create sample input
x = torch.randn(1, 2, 4, 4)  # (batch, channels, height, width)

# Get PyTorch output for reference
with torch.no_grad():
    torch_output = model(x)

print("PyTorch output shape:", torch_output.shape)

# Convert to hls4ml
config = config_from_pytorch_model(model, input_shape=(2, 4, 4), granularity="name")

hls_model = convert_from_pytorch_model(
    model, output_dir="hls4ml_prj_conv2d", backend="oneAPI", hls_config=config
)

# Print model info
print("\nhls4ml model created!")
print("Model layers:")
for layer in hls_model.get_layers():
    print(f"  - {layer.name}: {layer.class_name}")
    if hasattr(layer, "attributes"):
        print(f"    strategy: {layer.get_attr('strategy', 'N/A')}")
        print(f"    implementation: {layer.get_attr('implementation', 'N/A')}")

    # Check weights for Conv2D
    if layer.class_name == "Conv2D":
        w = layer.get_weights("weight")
        b = layer.get_weights("bias")
        print(f"    Weight data shape: {w.data.shape}")
        print(f"    Weight values (first 10): {w.data.flatten()[:10]}")
        print(f"    Bias data shape: {b.data.shape}")
        print(f"    Bias values: {b.data.flatten()}")

print("\nModel conversion completed!")
print(f"\nOutput directory: {hls_model.config.get_output_dir()}")

# Try to write the HLS code to see if templates are needed
print("\nAttempting to write HLS code...")
try:
    hls_model.write()
    print("HLS code generation successful!")
    print(f"Check the output in: {hls_model.config.get_output_dir()}")
except Exception as e:
    print(f"Error during HLS code generation: {e}")
    import traceback

    traceback.print_exc()
