"""Convert the exact DINOv3 ViT-L/16 LVD1689M native checkpoint for Transformers 5.5.

The input is the user-selected ciqiangxu/DINOv3 ModelScope mirror. Conversion
renames tensors and splits fused QKV; it does not train or approximate weights.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

EXPECTED_SHA256 = "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"


def convert_state(original):
    """Map all native tensors, preserving effective masked QKV biases exactly."""
    state = dict(original)
    converted = {
        "embeddings.cls_token": state.pop("cls_token"),
        "embeddings.register_tokens": state.pop("storage_tokens"),
        "embeddings.mask_token": state.pop("mask_token").unsqueeze(0),
        "embeddings.patch_embeddings.weight": state.pop("patch_embed.proj.weight"),
        "embeddings.patch_embeddings.bias": state.pop("patch_embed.proj.bias"),
        "norm.weight": state.pop("norm.weight"),
        "norm.bias": state.pop("norm.bias"),
    }
    periods = state.pop("rope_embed.periods")
    for i in range(24):
        source, target = f"blocks.{i}.", f"model.layer.{i}."
        q, k, v = state.pop(source + "attn.qkv.weight").chunk(3, dim=0)
        for name, value in zip(("q", "k", "v"), (q, k, v), strict=True):
            converted[target + f"attention.{name}_proj.weight"] = value
        bias = state.pop(source + "attn.qkv.bias") * state.pop(source + "attn.qkv.bias_mask")
        q, k, v = bias.chunk(3)
        if torch.count_nonzero(k):
            raise ValueError("Expected the official checkpoint's masked zero key bias")
        converted[target + "attention.q_proj.bias"] = q
        converted[target + "attention.v_proj.bias"] = v
        for old, new in (
            ("norm1", "norm1"),
            ("norm2", "norm2"),
            ("attn.proj", "attention.o_proj"),
            ("mlp.fc1", "mlp.up_proj"),
            ("mlp.fc2", "mlp.down_proj"),
        ):
            for suffix in ("weight", "bias"):
                converted[target + f"{new}.{suffix}"] = state.pop(source + f"{old}.{suffix}")
        for index in (1, 2):
            converted[target + f"layer_scale{index}.lambda1"] = state.pop(source + f"ls{index}.gamma")
    if state:
        raise ValueError(f"Unconverted tensors: {list(state)}")
    return converted, periods


def main():
    """Strict-load, compare against Meta's implementation, and save a local HF checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with args.checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Wrong native checkpoint SHA-256: {digest}")
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    converted, periods = convert_state(original)
    config = DINOv3ViTConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_register_tokens=4,
        patch_size=16,
        layerscale_value=1e-5,
        layer_norm_eps=1e-5,
        rope_theta=100.0,
        pos_embed_rescale=2.0,
        key_bias=False,
    )
    config.native_rope_periods = periods.float().tolist()
    config._attn_implementation = "sdpa"
    model = DINOv3ViTModel(config)
    model.load_state_dict(converted, strict=True)
    # Transformers otherwise regenerates ideal FP32 periods; the original
    # checkpoint contains BF16-rounded periods used by Meta's own strict loader.
    model.rope_embeddings.inv_freq.copy_(1 / periods.float())

    sys.path.insert(0, str(args.official_source.resolve()))
    from dinov3.hub.backbones import dinov3_vitl16

    reference = dinov3_vitl16(pretrained=False)
    reference.load_state_dict(original, strict=True)
    reference.to(args.device).eval()
    model.to(args.device).eval()
    generator = torch.Generator().manual_seed(42)
    pixels = torch.randn(2, 3, 224, 224, generator=generator).to(args.device)
    with torch.inference_mode():
        native = reference.forward_features(pixels)
        tokens = model(pixel_values=pixels).last_hidden_state
    comparisons = {}
    for key, actual in (("x_norm_clstoken", tokens[:, 0]), ("x_norm_patchtokens", tokens[:, 5:])):
        expected = native[key]
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-4)
        comparisons[key] = float((actual - expected).abs().max())
    args.output.mkdir(parents=True, exist_ok=True)
    model.cpu().save_pretrained(args.output, safe_serialization=True)
    shutil.copy(args.official_source / "LICENSE.md", args.output / "LICENSE.md")
    receipt = {
        "status": "passed",
        "source_repository": "ciqiangxu/DINOv3",
        "source_file_revision": "7fcf8b63b5971dd702236309b667236bfde72d40",
        "native_checkpoint": args.checkpoint.name,
        "sha256": digest,
        "variant": "dinov3_vitl16_pretrain_lvd1689m",
        "strict_load": True,
        "native_rope_periods_preserved": True,
        "parameters": sum(p.numel() for p in model.parameters()),
        "fp32_forward_max_abs_errors": comparisons,
        "official_code_revision": subprocess.check_output(
            ["git", "-C", str(args.official_source), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (args.output / "conversion.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
