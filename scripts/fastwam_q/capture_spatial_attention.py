"""Save per-observation critic attention maps and cached RGB for truthful spatial overlays."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import default_collate

from RL.fastwam_q.cached_data import CachedDemoChunkDataset
from RL.fastwam_q.checkpoint import load_checkpoint


class SpatialProbe:
    """Capture actual SDPA inputs and summarize selected cross-attention layers per sample."""

    def __init__(self, valid, indices, layers):
        """Select observations without changing model values, dropout or attention outputs."""
        self.original = functional.scaled_dot_product_attention
        self.valid = functional.pad(valid, (1, 0), value=True)
        self.indices, self.layers = indices, layers
        self.calls, self.maps = 0, {}

    def __enter__(self):
        """Install the hook in this probe process only."""
        functional.scaled_dot_product_attention = self.forward
        return self

    def __exit__(self, *_):
        """Restore the original function."""
        functional.scaled_dot_product_attention = self.original

    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        """Compute real output unchanged, and recompute probability in explicit FP32."""
        output = self.original(query, key, value, attn_mask, dropout_p, is_causal, **kwargs)
        if query.shape[-2] != self.valid.shape[-1]:
            return output
        layer, kind = self.calls // 2 + 1, self.calls % 2
        self.calls += 1
        if kind != 1 or layer not in self.layers:
            return output
        with torch.autocast("cuda", enabled=False):
            scores = query.float() @ key.float().transpose(-1, -2) / math.sqrt(query.shape[-1])
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores.masked_fill_(~attn_mask, -torch.inf)
                else:
                    scores += attn_mask.float()
            probability = scores.softmax(-1)
            mean = (probability * self.valid[:, None, :, None]).sum(-2) / self.valid.sum(-1)[:, None, None]
            self.maps[f"layer{layer}_cls"] = probability[self.indices, :, 0].cpu().numpy()
            self.maps[f"layer{layer}_valid_query_mean"] = mean[self.indices].cpu().numpy()
        return output


def main():
    """Choose the first two distinct episodes per task in one fixed batch, never by heatmap appearance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update", type=int, default=2030)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 6, 12, 17, 18])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    q, normalizer = load_checkpoint(args.checkpoint, device="cuda", with_target=False)
    q.config.gradient_checkpointing = False
    q.online.image_encoder.backbone.gradient_checkpointing_disable()
    dataset = CachedDemoChunkDataset(args.cache, config=q.config)
    indices = np.load(args.plan / "batch_indices.npy", mmap_mode="r")[args.update - 1]
    rows = dataset.rows[indices]
    selected = []
    examples = []
    for task_index, task in enumerate(dataset.tasks):
        seen = set()
        for batch_index, row in enumerate(rows):
            episode = int(dataset.data["episode_index"][row])
            if int(dataset.data["task_index"][row]) != task_index or episode in seen:
                continue
            seen.add(episode)
            selected.append(batch_index)
            examples.append(
                {
                    "task_index": task_index,
                    "task": task,
                    "example": len(seen) - 1,
                    "batch_index": batch_index,
                    "dataset_index": int(indices[batch_index]),
                    "row": int(row),
                    "episode": episode,
                    "frame": int(dataset.data["frame_index"][row]),
                }
            )
            if len(seen) == 2:
                break
    raw = default_collate([dataset[int(index)] for index in indices])
    for sample_index, example in enumerate(examples):
        images = raw["images"][example["batch_index"]].permute(0, 2, 3, 1).numpy()
        example["images"] = []
        for view, (camera, pixels) in enumerate(zip(q.config.camera_keys, images, strict=True)):
            name = f"sample_{sample_index:02d}_view{view}.png"
            Image.fromarray(pixels).save(args.output / name)
            example["images"].append({"camera": camera, "file": name})
    batch = {
        name: value.to("cuda") if isinstance(value, torch.Tensor) else value for name, value in raw.items()
    }
    batch["action_chunk"] = normalizer(batch["action_chunk"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        text = q.text_features(batch["tasks"], "cuda")
        with SpatialProbe(batch["valid"], selected, args.layers) as probe:
            logits = q.online(batch["images"], batch["action_chunk"], text, batch["valid"])
        values = q.q_values(logits)[selected].cpu().tolist()
    for example, value in zip(examples, values, strict=True):
        example["q_value"] = value
    np.savez_compressed(args.output / "probabilities.npz", **probe.maps)
    side = q.config.image_size[0] // q.online.image_encoder.patch_size
    text_slots = probe.maps[f"layer{args.layers[-1]}_cls"].shape[-1] - len(q.config.camera_keys) * side * side
    metadata = {
        "checkpoint": str(args.checkpoint),
        "sampling_plan": str(args.plan),
        "sampling_update": args.update,
        "mode": "eval, BF16 forward; FP32 pre-dropout attention recomputation; no optimizer/backward",
        "selection": "first two distinct episodes per task in fixed plan batch, not chosen by attention pattern",
        "layers": args.layers,
        "attention_calls": probe.calls,
        "text_slots": text_slots,
        "patch_grid": [side, side],
        "camera_keys": q.config.camera_keys,
        "examples": examples,
        "coordinate_mapping": "image resized without crop to 224x224; 14x14 patch cell stretched back to native RGB",
        "limitations": "attention probability, not causal Q attribution; examples from training pool; CLS differs from all-query mean",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
