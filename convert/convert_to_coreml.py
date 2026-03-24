"""
Convert a trained BraTS 2020 3D U-Net from PyTorch checkpoint to CoreML.

Produces three variants:
  - FP16 (10.7 MB, default mlprogram precision)
  - Int8 palettized (5.4 MB)
  - Int4 palettized (2.7 MB)

Usage:
    python convert_to_coreml.py \
        --checkpoint /path/to/brats_ckpt_epoch150.pt \
        --output-dir ./models

Runs on CPU. No GPU needed. Tested on Google Colab (Linux) and macOS.
"""

import argparse
import os

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ops = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ops(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, base_filters=32, levels=3):
        super().__init__()
        self.levels = levels
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = in_channels
        for i in range(levels):
            ch_out = base_filters * (2 ** i)
            self.encoders.append(ConvBlock(ch_in, ch_out))
            self.pools.append(nn.MaxPool3d(2))
            ch_in = ch_out
        bottleneck_ch = base_filters * (2 ** levels)
        self.bottleneck = ConvBlock(ch_in, bottleneck_ch)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(levels)):
            ch_enc = base_filters * (2 ** i)
            ch_up_in = bottleneck_ch if i == levels - 1 else base_filters * (2 ** (i + 1))
            self.upconvs.append(
                nn.ConvTranspose3d(ch_up_in, ch_enc, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(ch_enc * 2, ch_enc))
        self.head = nn.Conv3d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.head(x)


def dir_size_mb(path):
    return sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(path)
        for f in fns
    ) / (1024 * 1024)


METADATA = {
    "author": "Guillaume Esclozas",
    "description": (
        "3D U-Net for brain lesion segmentation. "
        "Trained on BraTS 2020, 4 MRI modalities (FLAIR, T1, T1ce, T2), "
        "4 output classes (background, NCR/NET, edema, enhancing tumor). "
        "Input: [1, 4, 128, 128, 128]. Best foreground Dice: 0.7317."
    ),
    "license": "Research use. BraTS 2020 data subject to challenge license.",
    "version": "1.0",
    "input_desc": (
        "4-channel MRI stack: FLAIR, T1, T1ce, T2. "
        "Z-score normalized on nonzero voxels. Shape [1, 4, 128, 128, 128]."
    ),
    "output_desc": (
        "Raw logits per class. Argmax along axis 1 gives label map: "
        "0=background, 1=NCR/NET, 2=edema, 3=enhancing tumor."
    ),
}


def apply_metadata(mlmodel):
    mlmodel.author = METADATA["author"]
    mlmodel.short_description = METADATA["description"]
    mlmodel.license = METADATA["license"]
    mlmodel.version = METADATA["version"]
    return mlmodel


def rename_output(mlmodel, save_path, new_name="segmentation"):
    """coremltools renames outputs to things like var_356 due to internal
    MIL name collisions. This fixes it after conversion."""
    spec = mlmodel.get_spec()
    old_name = spec.description.output[0].name
    if old_name == new_name:
        return mlmodel

    ct.utils.rename_feature(spec, old_name, new_name)
    spec.description.input[0].shortDescription = METADATA["input_desc"]
    spec.description.output[0].shortDescription = METADATA["output_desc"]

    weights_dir = os.path.join(save_path, "Data/com.apple.CoreML/weights")
    return ct.models.MLModel(spec, weights_dir=weights_dir)


def save_variant(mlmodel, path, label):
    mlmodel = apply_metadata(mlmodel)
    mlmodel.save(path)
    mlmodel = rename_output(mlmodel, path)
    mlmodel = apply_metadata(mlmodel)
    mlmodel.save(path)
    print(f"  [{label}] {path} ({dir_size_mb(path):.1f} MB)")
    return mlmodel


def verify(path, label):
    m = ct.models.MLModel(path)
    s = m.get_spec()
    inp = s.description.input[0]
    out = s.description.output[0]
    assert inp.name == "input", f"Expected input name 'input', got '{inp.name}'"
    assert out.name == "segmentation", f"Expected output name 'segmentation', got '{out.name}'"
    assert list(inp.type.multiArrayType.shape) == [1, 4, 128, 128, 128]
    assert list(out.type.multiArrayType.shape) == [1, 4, 128, 128, 128]
    assert m.author == METADATA["author"]
    print(f"  [{label}] verified OK")


def main():
    parser = argparse.ArgumentParser(description="Convert BraTS 3D U-Net to CoreML")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", default="./models", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = UNet3D()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    dummy = torch.randn(1, 4, 128, 128, 128)
    traced = torch.jit.trace(model, dummy)

    print("\nConverting...")
    fp16_path = os.path.join(args.output_dir, "brats_unet3d_fp16.mlpackage")
    mlmodel_fp16 = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 4, 128, 128, 128))],
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel_fp16 = save_variant(mlmodel_fp16, fp16_path, "FP16")

    int8_path = os.path.join(args.output_dir, "brats_unet3d_int8.mlpackage")
    mlmodel_int8 = palettize_weights(
        mlmodel_fp16,
        OptimizationConfig(global_config=OpPalettizerConfig(nbits=8)),
    )
    save_variant(mlmodel_int8, int8_path, "Int8")

    int4_path = os.path.join(args.output_dir, "brats_unet3d_int4.mlpackage")
    mlmodel_int4 = palettize_weights(
        mlmodel_fp16,
        OptimizationConfig(global_config=OpPalettizerConfig(nbits=4)),
    )
    save_variant(mlmodel_int4, int4_path, "Int4")

    print("\nVerifying...")
    verify(fp16_path, "FP16")
    verify(int8_path, "Int8")
    verify(int4_path, "Int4")

    print("\nDone.")


if __name__ == "__main__":
    main()
