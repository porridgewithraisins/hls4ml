#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
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
        return self.conv(torch.cat([skip, x], dim=1))


class MiniUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=16, num_classes=1):
        super().__init__()
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


class PetSegmentation(datasets.OxfordIIITPet):
    def __init__(self, root, split, size):
        super().__init__(root, split=split, target_types="segmentation", download=True)
        self.size = size
        self.img_tf = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ]
        )
        self.mask_tf = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST),
            ]
        )

    def __getitem__(self, index):
        img, mask = super().__getitem__(index)
        img = self.img_tf(img)
        mask = self.mask_tf(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8, copy=True))
        mask = (mask > 0).float()
        return img, mask.unsqueeze(0)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = PetSegmentation(args.data_dir, "trainval", args.image_size)
    test_ds = PetSegmentation(args.data_dir, "test", args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = MiniUNet(base_channels=args.base_channels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=False)
        for imgs, masks in progress:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            progress.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]", leave=False)
            for imgs, masks in progress:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                progress.set_postfix(loss=loss.item())

        print(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={train_loss/len(train_loader.dataset):.4f}  "
            f"val_loss={val_loss/len(test_loader.dataset):.4f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "mini_unet_oxfordpet.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    model.eval()
    onnx_path = args.output_dir / "mini_unet_oxfordpet.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=13,
    )
    print(f"Exported ONNX to {onnx_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mini U-Net on Oxford-IIIT Pet")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
