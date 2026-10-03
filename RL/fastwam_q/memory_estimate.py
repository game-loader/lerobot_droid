"""Count real ViT-L/16 + T5-base + decoder parameters; optionally measure synthetic CUDA updates.

This is a Q-only memory probe with random weights, not policy training or a quality evaluation.
No checkpoints are downloaded and FastWAM is not loaded.
"""

import argparse
import json
from typing import TYPE_CHECKING

import torch
from torch import nn

from lerobot.utils.import_utils import _transformers_available, require_package

from .config import FastWAMQConfig
from .model import DINOv3ImageEncoder, FastWAMQFunction, FastWAMQNetwork
from .trainer import FastWAMQTrainer

if TYPE_CHECKING or _transformers_available:
    from transformers import DINOv3ViTConfig, DINOv3ViTModel, T5Config, T5EncoderModel


def backbone_configs():
    """Architecture of official DINOv3 ViT-L/16 and T5-v1.1-base (encoder only)."""
    require_package("transformers", extra="fastwam")
    vision = DINOv3ViTConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_register_tokens=4,
        layerscale_value=1e-5,
    )
    text = T5Config(
        d_model=768,
        d_kv=64,
        d_ff=2048,
        num_layers=12,
        num_heads=12,
        feed_forward_proj="gated-gelu",
        vocab_size=32128,
    )
    return vision, text


def estimate_q_memory(config):
    """Static resident state in GiB for FP32 masters/AdamW, BF16 autocast (the trainer's default)."""
    vision, text = backbone_configs()
    with torch.device("meta"):
        backbone = DINOv3ViTModel(vision)
        q = FastWAMQNetwork(config, DINOv3ImageEncoder(config, backbone), text_dim=text.d_model)
        text_model = T5EncoderModel(text) if config.use_text else None
    online = sum(p.numel() for p in q.parameters())
    dino = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in q.parameters() if p.requires_grad)
    frozen_text = sum(p.numel() for p in text_model.parameters()) if text_model is not None else 0
    weights = 4 * (2 * online + frozen_text)
    gradients = 4 * trainable
    adam = 8 * trainable
    return {
        "online_parameters": online,
        "vision_parameters": dino,
        "text_parameters": frozen_text,
        "decoder_and_adapters_parameters": online - dino,
        "trainable_parameters": trainable,
        "weights_including_target_gib": weights / 2**30,
        "gradients_gib": gradients / 2**30,
        "adam_moments_gib": adam / 2**30,
        "static_training_state_gib": (weights + gradients + adam) / 2**30,
        "note": "Static state only; excludes activations, AMP casts, allocator, and FastWAM.",
    }


class SyntheticTextEncoder(nn.Module):
    """A real frozen T5 encoder fed synthetic token ids; used only by the memory probe."""

    def __init__(self, config):
        """Construct the component from its configuration and supplied dependencies."""
        super().__init__()
        _, text = backbone_configs()
        self.model = T5EncoderModel(text).requires_grad_(False).eval()
        self.hidden_size = text.d_model
        self.length = config.text_max_length

    @torch.no_grad()
    def forward(self, tasks, device):
        """Compute the component output for the supplied batch."""
        tokens = torch.ones(len(tasks), self.length, dtype=torch.long, device=device)
        encoded = self.model(input_ids=tokens, attention_mask=tokens).last_hidden_state
        return encoded, torch.zeros_like(tokens, dtype=torch.bool)


def profile_cuda(config, batches):
    """Measure two optimizer steps per batch size, including lazy Adam state allocation."""
    vision, _ = backbone_configs()
    q = FastWAMQFunction(
        config,
        dino_backbone=DINOv3ViTModel(vision),
        text_encoder=SyntheticTextEncoder(config) if config.use_text else None,
    )
    trainer = FastWAMQTrainer(q)
    results = []
    for size in batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        h = config.chunk_size
        batch = {
            "images": torch.randint(
                256, (size, len(config.camera_keys), 3, *config.image_size), dtype=torch.uint8
            ),
            "next_images": torch.randint(
                256, (size, len(config.camera_keys), 3, *config.image_size), dtype=torch.uint8
            ),
            "action_chunk": torch.randn(size, h, config.action_dim),
            "next_action_chunk": torch.randn(size, h, config.action_dim),
            "rewards": torch.zeros(size, h),
            "valid": torch.ones(size, h, dtype=torch.bool),
            "next_valid": torch.ones(size, h, dtype=torch.bool),
            "done": torch.zeros(size, dtype=torch.bool),
            "tasks": ["synthetic"] * size,
        }
        try:
            for _ in range(2):
                metrics = trainer.update(batch)
            torch.cuda.synchronize()
            results.append(
                {
                    "batch_size": size,
                    "loss": metrics["loss"],
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                }
            )
        except torch.OutOfMemoryError:
            results.append({"batch_size": size, "out_of_memory": True})
            break
    return results


def main():
    """Print exact static-state accounting or a bounded, synthetic real-architecture CUDA profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--freeze-dino", action="store_true")
    parser.add_argument("--no-checkpointing", action="store_true")
    args = parser.parse_args()
    config = FastWAMQConfig(
        camera_keys=tuple(f"view{i}" for i in range(args.views)),
        freeze_dino=args.freeze_dino,
        gradient_checkpointing=not args.no_checkpointing,
    )
    result = {
        **estimate_q_memory(config),
        "views": args.views,
        "gradient_checkpointing": config.gradient_checkpointing,
        "freeze_dino": config.freeze_dino,
        "precision": config.amp_dtype,
        "torch": torch.__version__,
    }
    if args.profile:
        result["gpu"] = torch.cuda.get_device_name()
        result["synthetic_cuda_profile"] = profile_cuda(config, args.batches)
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
