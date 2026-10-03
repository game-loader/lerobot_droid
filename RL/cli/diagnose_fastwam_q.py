"""Trace a saved critic on paired real batches without taking optimizer steps."""

import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import default_collate

from RL.fastwam_q.cached_data import CachedDemoChunkDataset
from RL.fastwam_q.checkpoint import load_checkpoint
from RL.fastwam_q.diagnostics import AttentionTrace, LayerTrace, parameter_report
from RL.fastwam_q.model import chunk_targets, hl_gauss_target

VARIANTS = (
    "bf16_default",
    "fp32_default",
    "bf16_math",
    "bf16_no_dropout",
    "fp32_no_dropout",
    "bf16_math_no_dropout",
)


def run_variant(q, batch, variant, seed, output, capture_layers=()):
    """Preserve weights/targets and vary precision, attention backend or dropout only."""
    q.train()
    q.zero_grad(set_to_none=True)
    no_dropout = "no_dropout" in variant
    dropout_settings = []
    if no_dropout:
        for module in q.online.modules():
            if isinstance(module, (nn.Dropout, nn.MultiheadAttention)):
                attribute = "p" if isinstance(module, nn.Dropout) else "dropout"
                dropout_settings.append((module, attribute, getattr(module, attribute)))
                setattr(module, attribute, 0.0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    backend = sdpa_kernel(SDPBackend.MATH) if "math" in variant else contextlib.nullcontext()
    tracer = LayerTrace(q.online)
    started = time.monotonic()
    try:
        with (
            AttentionTrace(capture_layers) as attention,
            backend,
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=variant.startswith("bf16")),
        ):
            loss, metrics = q.td_loss(batch)
        loss.backward()
        result = {
            "variant": variant,
            "seed": seed,
            "metrics": {name: float(value) for name, value in metrics.items()},
            "parameters": parameter_report(q.online),
            "layers": tracer.report(),
            "attention_kernels": attention.calls,
            "seconds": time.monotonic() - started,
            "checkpointing": False,
            "input_gradient_note": "Input gradients contain total fan-out contributions, not isolated local Jacobians.",
        }
        output.write_text(json.dumps(result, indent=2) + "\n")
        if attention.saved:
            torch.save(attention.saved, output.with_suffix(".attention.pt"))
        print(
            json.dumps(
                {
                    "path": str(output),
                    "variant": variant,
                    "loss": float(loss.detach()),
                    "grad_norm": result["parameters"]["global_grad_l2"],
                }
            ),
            flush=True,
        )
    finally:
        tracer.close()
        for module, attribute, value in dropout_settings:
            setattr(module, attribute, value)
        q.zero_grad(set_to_none=True)
    return result


def run(args):
    """Use existing caches and saved weights; no robot, new video cache, or optimizer."""
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    q, normalizer = load_checkpoint(args.checkpoint, device="cuda")
    q.config.gradient_checkpointing = False
    q.online.image_encoder.backbone.gradient_checkpointing_disable()
    if args.freeze_backbone:
        q.online.image_encoder.backbone.requires_grad_(False)
    dataset = CachedDemoChunkDataset(args.cache, config=q.config)
    index_matrix = np.load(args.plan / "batch_indices.npy")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        q.text_features(dataset.tasks, "cuda")
    variants = list(args.variants)
    random.Random(args.seed).shuffle(variants)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "freeze_backbone": args.freeze_backbone,
        "updates": args.updates,
        "variants_in_execution_order": variants,
        "seed": args.seed,
        "optimizer_updates": 0,
        "training_states_modified": False,
        "limitation": "Paired static-gradient diagnosis, not independent training replicates.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for update in args.updates:
        raw = default_collate([dataset[int(index)] for index in index_matrix[update - 1]])
        batch = {
            key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in raw.items()
        }
        for key in ("action_chunk", "next_action_chunk"):
            batch[key] = normalizer(batch[key])
        for variant in variants:
            path = args.output / f"update_{update:04d}_{variant}.json"
            if not path.exists():
                run_variant(q, batch, variant, args.seed + update, path, args.capture_layers)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        text = q.text_features(batch["tasks"], "cuda")
        next_q = q.q_values(
            q.target(batch["next_images"], batch["next_action_chunk"], text, batch["next_valid"])
        )
        targets = chunk_targets(batch["rewards"], batch["valid"], batch["done"], next_q, q.config.gamma)
        distribution = hl_gauss_target(targets, q.bin_centers, q.config.hl_gauss_sigma)
    (args.output / "target_range.json").write_text(
        json.dumps(
            {
                "min": float(targets.min()),
                "max": float(targets.max()),
                "sum_error_max": float((distribution.sum(-1) - 1).abs().max()),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    """Parse the checkpoint and paired-batch diagnostic settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", nargs="+", type=int, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--capture-layers", nargs="*", type=int, default=[])
    parser.add_argument("--seed", type=int, default=20260930)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
