"""Stream all decoder attention statistics from saved weights without backward or optimizer updates."""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import default_collate

from RL.fastwam_q.cached_data import CachedDemoChunkDataset
from RL.fastwam_q.checkpoint import load_checkpoint


def key_labels(kind, size, cameras, side):
    """Name action keys or batch-padded text slots and camera patch keys."""
    if kind == "self":
        return ["CLS", *[f"action[{i}]" for i in range(size - 1)]]
    text = size - len(cameras) * side * side
    return [
        *[f"text[{i}]" for i in range(text)],
        *[
            f"{camera}:patch({row},{column})"
            for camera in cameras
            for row in range(side)
            for column in range(side)
        ],
    ]


def group_name(label):
    """Identify the modality/camera of a named key."""
    if label.startswith("text["):
        return "text"
    if label.startswith("action["):
        return "action"
    return label.split(":")[0]


def top_tokens(counts, limit=8):
    """Report winning-key frequencies, not attention mass or semantic correctness."""
    total = sum(counts.values())
    return [
        {"token": key, "count": count, "fraction": count / total} for key, count in counts.most_common(limit)
    ]


def top_probability_tokens(mass, rows, limit=8):
    """Report mean token probabilities, distinct from argmax selection frequency."""
    return [{"token": key, "mean_attention_weight": value / rows} for key, value in mass.most_common(limit)]


class AttentionProbe:
    """Intercept only decoder SDPA calls and leave actual model outputs and RNG unchanged."""

    def __init__(self, batch, cameras, side):
        """Set valid-query/task metadata for this forward pass."""
        self.original = functional.scaled_dot_product_attention
        self.valid = functional.pad(batch["valid"], (1, 0), value=True)
        self.tasks = batch["tasks"]
        self.cameras, self.side = cameras, side
        self.records = []

    def __enter__(self):
        """Install the local process hook."""
        functional.scaled_dot_product_attention = self.forward
        return self

    def __exit__(self, *_):
        """Restore SDPA even if this read-only probe fails."""
        functional.scaled_dot_product_attention = self.original

    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        """Execute original SDPA, then inspect its projected Q/K in FP32 before dropout."""
        output = self.original(query, key, value, attn_mask, dropout_p, is_causal, **kwargs)
        if query.shape[-2] != self.valid.shape[-1]:
            return output
        index = len(self.records)
        kind = "self" if index % 2 == 0 else "cross"
        # Explicitly disable autocast here: float() alone does not force FP32 matmul inside autocast.
        with torch.autocast("cuda", enabled=False):
            scores = query.float() @ key.float().transpose(-1, -2) / math.sqrt(query.shape[-1])
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores.masked_fill_(~attn_mask, -torch.inf)
                else:
                    scores += attn_mask.float()
            probability = scores.softmax(-1)
            maxima, winners = probability.max(-1)
            entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
            valid = self.valid[:, None].expand_as(maxima)
            cls = valid.clone()
            cls[:, :, 1:] = False
            actions = valid.clone()
            actions[:, :, 0] = False
            saturated = maxima > 1 - 1e-7
            heads, keys = query.shape[1], key.shape[-2]
            labels = key_labels(kind, keys, self.cameras, self.side)
            offsets = torch.arange(heads, device=query.device)[None, :, None] * keys
            hist = (
                torch.bincount((winners + offsets)[valid], minlength=heads * keys).reshape(heads, keys).cpu()
            )
            key_mass = (probability * valid[..., None]).sum((0, 2)).cpu().tolist()
            groups = list(dict.fromkeys(group_name(label) for label in labels))
            mass = {}
            for group in groups:
                positions = [i for i, label in enumerate(labels) if group_name(label) == group]
                amount = probability[..., positions].sum(-1)
                mass[group] = (amount * valid).sum((0, 2)).cpu().tolist()
            record = {
                "layer": index // 2 + 1,
                "kind": kind,
                "dropout": dropout_p,
                "query_shape": list(query.shape),
                "key_shape": list(key.shape),
                "valid_rows": int(valid.sum()),
                "cls_rows": int(cls.sum()),
                "action_rows": int(actions.sum()),
                "saturated_count": int((saturated & valid).sum()),
                "cls_saturated_count": int((saturated & cls).sum()),
                "action_saturated_count": int((saturated & actions).sum()),
                "pmax_gt_099_count": int(((maxima > 0.99) & valid).sum()),
                "pmax_sum": float(maxima[valid].sum()),
                "entropy_sum": float(entropy[valid].sum()),
                "query_rms": float(query.float().square().mean().sqrt()),
                "key_rms": float(key.float().square().mean().sqrt()),
                "score_abs_max": float(scores.masked_fill(~torch.isfinite(scores), 0).abs().max()),
                "head_rows": valid.sum((0, 2)).cpu().tolist(),
                "head_saturated": (saturated & valid).sum((0, 2)).cpu().tolist(),
                "head_entropy_sum": (entropy * valid).sum((0, 2)).cpu().tolist(),
                "head_winner_counts": [
                    {label: count for label, count in zip(labels, counts.tolist(), strict=True) if count}
                    for counts in hist
                ],
                "head_probability_mass_sum": mass,
                "key_labels": labels,
                "head_key_probability_mass_sum": key_mass,
                "action_winner_matches_cls": int(
                    ((winners[:, :, 1:] == winners[:, :, :1]) & valid[:, :, 1:]).sum()
                ),
                "sample_head_same_key_count": int(((winners == winners[:, :, :1]) | ~valid).all(-1).sum()),
                "sample_head_count": query.shape[0] * heads,
                "per_task": {},
            }
            for task in dict.fromkeys(self.tasks):
                samples = torch.tensor([item == task for item in self.tasks], device=query.device)
                mask = valid & samples[:, None, None]
                counts = torch.bincount(winners[mask], minlength=keys).cpu().tolist()
                record["per_task"][task] = {
                    "rows": int(mask.sum()),
                    "saturated": int((saturated & mask).sum()),
                    "winner_counts": {
                        label: count for label, count in zip(labels, counts, strict=True) if count
                    },
                    "key_probability_mass_sum": (probability * mask[..., None]).sum((0, 1, 2)).cpu().tolist(),
                }
        self.records.append(record)
        return output


def aggregate(records):
    """Pool row counts across batches; preserve heads/tasks and named token histograms."""
    total = sum(item["valid_rows"] for item in records)
    cls = sum(item["cls_rows"] for item in records)
    actions = sum(item["action_rows"] for item in records)
    head_counts = [Counter() for _ in records[0]["head_rows"]]
    head_mass = [Counter() for _ in records[0]["head_rows"]]
    for item in records:
        for counts, added in zip(head_counts, item["head_winner_counts"], strict=True):
            counts.update({key: value for key, value in added.items() if value})
        for counts, added in zip(head_mass, item["head_key_probability_mass_sum"], strict=True):
            counts.update(dict(zip(item["key_labels"], added, strict=True)))
    overall = sum(head_counts, Counter())
    groups = Counter()
    for key, count in overall.items():
        groups[group_name(key)] += count
    mass = {
        group: sum(sum(item["head_probability_mass_sum"][group]) for item in records) / total
        for group in records[0]["head_probability_mass_sum"]
    }
    result = {
        "layer": records[0]["layer"],
        "kind": records[0]["kind"],
        "valid_rows": total,
        "saturated_fraction": sum(item["saturated_count"] for item in records) / total,
        "cls_saturated_fraction": sum(item["cls_saturated_count"] for item in records) / cls,
        "action_saturated_fraction": sum(item["action_saturated_count"] for item in records) / actions,
        "pmax_gt_099_fraction": sum(item["pmax_gt_099_count"] for item in records) / total,
        "pmax_mean": sum(item["pmax_sum"] for item in records) / total,
        "entropy_nats_mean": sum(item["entropy_sum"] for item in records) / total,
        "query_rms_per_batch": [item["query_rms"] for item in records],
        "key_rms_per_batch": [item["key_rms"] for item in records],
        "score_abs_max": max(item["score_abs_max"] for item in records),
        "winner_groups": {key: count / total for key, count in groups.items()},
        "probability_mass_groups": mass,
        "top_tokens": top_tokens(overall),
        "top_probability_tokens": top_probability_tokens(sum(head_mass, Counter()), total),
        "action_winner_matches_cls_fraction": sum(item["action_winner_matches_cls"] for item in records)
        / actions,
        "sample_head_all_valid_queries_share_key_fraction": sum(
            item["sample_head_same_key_count"] for item in records
        )
        / sum(item["sample_head_count"] for item in records),
        "per_head": [],
        "per_task": {},
    }
    for head, counts in enumerate(head_counts):
        rows = sum(item["head_rows"][head] for item in records)
        result["per_head"].append(
            {
                "head": head,
                "saturated_fraction": sum(item["head_saturated"][head] for item in records) / rows,
                "entropy_nats_mean": sum(item["head_entropy_sum"][head] for item in records) / rows,
                "top_tokens": top_tokens(counts, 6),
                "top_probability_tokens": top_probability_tokens(head_mass[head], rows, 6),
                "probability_mass_groups": {
                    group: sum(item["head_probability_mass_sum"][group][head] for item in records) / rows
                    for group in mass
                },
            }
        )
    for task in dict.fromkeys(task for item in records for task in item["per_task"]):
        items = [item["per_task"][task] for item in records if task in item["per_task"]]
        counts = Counter()
        token_mass = Counter()
        for item in items:
            counts.update({key: value for key, value in item["winner_counts"].items() if value})
        for item in records:
            if task in item["per_task"]:
                token_mass.update(
                    dict(
                        zip(
                            item["key_labels"],
                            item["per_task"][task]["key_probability_mass_sum"],
                            strict=True,
                        )
                    )
                )
        rows = sum(item["rows"] for item in items)
        result["per_task"][task] = {
            "rows": rows,
            "saturated_fraction": sum(item["saturated"] for item in items) / rows,
            "top_tokens": top_tokens(counts),
            "top_probability_tokens": top_probability_tokens(token_mass, rows),
        }
    return result


def main():
    """Probe train/dropout and eval forwards on existing cached RGB; no production state changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[2030, 2031, 2032])
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    q, normalizer = load_checkpoint(args.checkpoint, device="cuda", with_target=False)
    q.config.gradient_checkpointing = False
    q.online.image_encoder.backbone.gradient_checkpointing_disable()
    dataset = CachedDemoChunkDataset(args.cache, config=q.config)
    plan = np.load(args.plan / "batch_indices.npy", mmap_mode="r")
    side = q.config.image_size[0] // q.online.image_encoder.patch_size
    modes = {"training_forward": [], "eval": []}
    metadata = []
    started = time.monotonic()
    with torch.no_grad():
        for update in args.updates:
            indices = plan[update - 1]
            batch = default_collate([dataset[int(index)] for index in indices])
            metadata.append(
                {
                    "sampling_update": update,
                    "dataset_indices": indices.tolist(),
                    "rows": dataset.rows[indices].tolist(),
                    "tasks": batch["tasks"],
                    "valid_lengths": batch["valid"].sum(-1).tolist(),
                }
            )
            batch = {
                key: value.to("cuda") if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            batch["action_chunk"] = normalizer(batch["action_chunk"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                text = q.text_features(batch["tasks"], "cuda")
            for mode, traces in modes.items():
                q.train(mode == "training_forward")
                torch.manual_seed(args.seed + update)
                with (
                    AttentionProbe(batch, q.config.camera_keys, side) as probe,
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    logits = q.online(batch["images"], batch["action_chunk"], text, batch["valid"])
                traces.append(probe.records)
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "update": update,
                            "attention_calls": len(probe.records),
                            "q_mean": float(q.q_values(logits).mean()),
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    results = {
        "checkpoint": str(args.checkpoint),
        "sampling_plan": str(args.plan),
        "definition": "FP32 recomputation from actual BF16 projected Q/K; pre-dropout pmax > 1-1e-7",
        "scope": "3 batches of 64 from demonstration training pool, not generalization/semantic validation",
        "optimizer_updates": 0,
        "seed": args.seed,
        "batches": metadata,
        "text_tokens_by_task": {
            task: q.text_encoder.tokenizer.convert_ids_to_tokens(q.text_encoder.tokenizer(task)["input_ids"])
            for task in dataset.tasks
        },
        "modes": {},
    }
    for mode, traces in modes.items():
        results["modes"][mode] = [aggregate(list(items)) for items in zip(*traces, strict=True)]
    (args.output / "attention_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    # Retain compact scalar/histogram inputs for independently checking pooled results.
    (args.output / "per_batch_attention.json").write_text(json.dumps(modes) + "\n")
    print(args.output / "attention_summary.json", flush=True)


if __name__ == "__main__":
    main()
