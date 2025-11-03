#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # skip tensors come from higher-resolution encoder stages; cat preserves alignment for even-sized inputs.
        return self.conv(torch.cat([skip, x], dim=1))


class MiniUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=16, num_classes=1, depth=3):
        super().__init__()
        if depth != 3:
            raise ValueError('Only depth=3 supported in this quick harness.')

        filters = [base_channels, base_channels * 2, base_channels * 4]

        self.inc = DoubleConv(in_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])

        self.bottleneck = DoubleConv(filters[2], filters[2] * 2)

        self.up1 = Up(filters[2] * 2, filters[1], filters[1])
        self.up2 = Up(filters[1], filters[0], filters[0])

        self.outc = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.bottleneck(x3)
        x = self.up1(x4, x2)
        x = self.up2(x, x1)
        return self.outc(x)


def save_flat(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for val in arr.ravel():
            f.write(f'{val} ')
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mini U-Net oneAPI conversion harness')
    parser.add_argument('--base-channels', type=int, default=16, help='Base channel count for the encoder stem')
    parser.add_argument('--image-size', type=int, default=64, choices=[32, 64, 96, 128])
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for synthetic input sample')
    parser.add_argument(
        '--strategy', default=os.getenv('HLS_STRATEGY', 'resource'), help='Layer strategy override (resource/latency)'
    )
    parser.add_argument('--reuse-factor', type=int, default=int(os.getenv('HLS_REUSE_FACTOR', '1')))
    parser.add_argument('--parallelization-factor', type=int, default=int(os.getenv('HLS_PARALLELIZATION_FACTOR', '1')))
    parser.add_argument('--io-type', default=os.getenv('HLS_IO_TYPE', 'io_parallel'), choices=['io_parallel', 'io_stream'])
    parser.add_argument('--output-dir', default='hls4ml_prj_unet', help='Project output directory')
    parser.add_argument('--load', type=Path, help='Optional checkpoint (.pt) to load into the Mini U-Net')
    parser.add_argument(
        '--precision',
        type=str,
        default='ap_fixed<18,8>',
        help='HLS result precision type (e.g. ap_fixed<18,8> or ap_fixed<24,10>)',
    )

    args = parser.parse_args()

    torch.manual_seed(0)

    model = MiniUNet(base_channels=args.base_channels)
    if args.load:
        load_path = Path(args.load)
        if not load_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {load_path}')
        checkpoint = torch.load(load_path, map_location='cpu')
        state_dict = checkpoint.get('model_state', checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print('Warning: state_dict mismatch when loading checkpoint:')
            if missing:
                print('  Missing keys:', ', '.join(missing))
            if unexpected:
                print('  Unexpected keys:', ', '.join(unexpected))
        print(f'Loaded weights from {load_path}')

    model.eval()  # ensure batchnorm layers use inference statistics during reference dump

    dummy_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    with torch.no_grad():
        ref_output = model(dummy_input)

    output_dir = Path(args.output_dir)
    tb_dir = output_dir / 'tb_data'
    input_file = tb_dir / 'tb_input_features.dat'
    output_file = tb_dir / 'tb_output_predictions.dat'

    x_channels_last = dummy_input.permute(0, 2, 3, 1).numpy()
    y_channels_last = ref_output.permute(0, 2, 3, 1).numpy()

    save_flat(x_channels_last, input_file)
    save_flat(y_channels_last, output_file)

    hls_config = config_from_pytorch_model(model, input_shape=(3, args.image_size, args.image_size), granularity='name')
    hls_config.setdefault('Model', {})
    hls_config.setdefault('LayerName', {})

    hls_config['Model']['ChannelsLastConversion'] = 'internal'
    hls_config['Model']['IOType'] = args.io_type
    hls_config['Model']['Strategy'] = args.strategy.lower()
    hls_config['Model']['ReuseFactor'] = max(1, args.reuse_factor)
    hls_config['InputData'] = str(input_file)
    hls_config['OutputPredictions'] = str(output_file)

    precision_overrides = {
        'bottleneck_block_0': args.precision,
        'bottleneck_block_3': args.precision,
        'up1_up': args.precision,
        'up1_conv_block_0': args.precision,
        'up1_conv_block_3': args.precision,
        'up2_up': args.precision,
        'up2_conv_block_0': args.precision,
        'up2_conv_block_3': args.precision,
        'outc': args.precision,
    }

    for name in ['inc', 'down1', 'down2', 'bottleneck', 'up1', 'up2', 'outc']:
        layer_cfg = hls_config['LayerName'].setdefault(name, {})
        layer_cfg['Strategy'] = args.strategy.lower()
        layer_cfg['ReuseFactor'] = max(1, args.reuse_factor)
        layer_cfg['ParallelizationFactor'] = max(1, args.parallelization_factor)

    for name, precision in precision_overrides.items():
        layer_cfg = hls_config['LayerName'].setdefault(name, {})
        precision_cfg = layer_cfg.setdefault('Precision', {})
        precision_cfg['result'] = precision
        # weight entries default to auto; leave untouched to avoid unnecessary resource growth.

    print('Applied hls4ml overrides:')
    print(f'  OutputDir={output_dir}')
    print(f'  IOType={hls_config["Model"]["IOType"]}')
    print(f'  Strategy={args.strategy.lower()}')
    print(f'  ReuseFactor={hls_config["Model"]["ReuseFactor"]}')
    print(f'  ParallelizationFactor={max(1, args.parallelization_factor)}')

    hls_model = convert_from_pytorch_model(
        model,
        output_dir=str(output_dir),
        backend='oneAPI',
        hls_config=hls_config,
        io_type=args.io_type,
    )

    hls_model.write()

    print('PyTorch output shape:', tuple(ref_output.shape))
    print('HLS project written to:', hls_model.config.get_output_dir())
