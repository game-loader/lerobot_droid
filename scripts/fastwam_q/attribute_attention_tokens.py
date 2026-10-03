"""Attribute saved attention saturation to valid queries and context tokens, without training."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def batch_metadata(cache, plan, update, chunk_size, camera_keys):
    """Recover the captured batch from numeric cache metadata only; never open RGB records."""
    manifest = json.loads((cache / "manifest.json").read_text())
    ends, rows = {}, []
    for stream in manifest["streams"]:
        episode = stream["episode"]
        if stream["camera"] in camera_keys and episode not in ends:
            with np.load(cache / stream["index"]) as index:
                ends[episode] = int(index["rows"][-1]) + 1
                rows.extend(index["rows"].tolist())
    selected = np.sort(rows)[np.load(plan / "batch_indices.npy", mmap_mode="r")[update - 1]]
    episodes = np.load(cache / "episode_index.npy", mmap_mode="r")[selected]
    tasks = np.load(cache / "task_index.npy", mmap_mode="r")[selected]
    frames = np.load(cache / "frame_index.npy", mmap_mode="r")[selected]
    lengths = np.minimum(
        chunk_size, [ends[int(ep)] - int(row) for ep, row in zip(episodes, selected, strict=True)]
    )
    valid = torch.cat(
        [
            torch.ones(len(selected), 1, dtype=torch.bool),
            torch.arange(chunk_size)[None] < torch.as_tensor(lengths)[:, None],
        ],
        dim=1,
    )
    return {
        "rows": selected.tolist(),
        "episodes": episodes.tolist(),
        "frames": frames.tolist(),
        "task_indices": tasks.tolist(),
        "task_strings": manifest["audit"]["tasks"],
        "valid_chunk_lengths": lengths.tolist(),
    }, valid


def distribution(values):
    """Summarize a captured scalar distribution using fixed descriptive quantiles."""
    values = values.float()
    quantiles = values.quantile(torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0]))
    return {
        key: float(value)
        for key, value in zip(("min", "median", "p90", "p99", "max"), quantiles, strict=True)
    }


def summarize(scores, valid):
    """Compute pre-dropout softmax statistics over the chosen attention rows."""
    probability = scores.softmax(-1)
    maxima, winners = probability.max(-1)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
    selected = maxima[valid]
    return (
        {
            "rows": selected.numel(),
            "saturated_fraction": float((selected > 1 - 1e-7).float().mean()),
            "pmax_mean": float(selected.mean()),
            "entropy_nats_mean": float(entropy[valid].mean()),
            "effective_keys_geometric_mean": float(entropy[valid].mean().exp()),
        },
        maxima,
        winners,
        entropy,
    )


def top_tokens(winners, mask, text_count, camera_keys, side):
    """Label most frequently selected text/action keys and row-major image patches."""

    def label(index):
        if text_count is None:
            return "CLS" if index == 0 else f"action_step_{index - 1}"
        if index < text_count:
            return f"text_token_{index}"
        view, patch = divmod(index - text_count, side * side)
        return f"{camera_keys[view]}:patch({patch // side},{patch % side})"

    counts = Counter(winners[mask].tolist())
    total = sum(counts.values())
    return [
        {"key": index, "token": label(index), "count": count, "fraction": count / total}
        for index, count in counts.most_common(12)
    ]


def groups(winners, mask, text_count, camera_keys, patch_count):
    """Count which modality/view owns the winning key."""
    selected = winners[mask]
    if text_count is None:
        return {"CLS": float((selected == 0).float().mean()), "action": float((selected > 0).float().mean())}
    return {
        "text": float((selected < text_count).float().mean()),
        **{
            name: float(
                ((selected >= text_count + i * patch_count) & (selected < text_count + (i + 1) * patch_count))
                .float()
                .mean()
            )
            for i, name in enumerate(camera_keys)
        },
    }


def analyze(name, entry, query_valid, metadata, camera_keys, side):
    """Analyze exact saved projected Q/K values, separately from the critic value softmax."""
    query, key = [entry[field].float() for field in ("query", "key")]
    scores = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
    if entry["mask"] is not None:
        scores += entry["mask"].float()
    valid = query_valid[:, None].expand(scores.shape[:-1])
    cls = valid.clone()
    cls[:, :, 1:] = False
    action = valid.clone()
    action[:, :, 0] = False
    padding = ~valid
    all_stats, maxima, winners, entropy = summarize(scores, torch.ones_like(valid))
    text_count = key.shape[-2] - len(camera_keys) * side * side if "cross_attn" in name else None
    top_two = scores.topk(2, dim=-1).values
    gaps = top_two[..., 0] - top_two[..., 1]
    active_saturation = (maxima > 1 - 1e-7) & valid
    same_as_cls = winners[:, :, 1:] == winners[:, :, :1]
    action_valid = valid[:, :, 1:]
    same_all_queries = ((winners == winners[:, :, :1]) | ~valid).all(-1)
    result = {
        "decoder_layer_1based": int(name.split(".")[1]) + 1,
        "query_shape": list(query.shape),
        "key_shape": list(key.shape),
        "text_context_slots": text_count,
        "query_rms": float(query.square().mean().sqrt()),
        "key_rms": float(key.square().mean().sqrt()),
        "score_abs_max_finite": float(scores[torch.isfinite(scores)].abs().max()),
        "all_rows": all_stats,
        "valid_rows": summarize(scores, valid)[0],
        "cls_rows": summarize(scores, cls)[0],
        "valid_action_rows": summarize(scores, action)[0],
        "padded_action_rows": summarize(scores, padding)[0] if padding.any() else None,
        "valid_top1_minus_top2_score_gap": distribution(gaps[valid]),
        "valid_winner_groups": groups(winners, valid, text_count, camera_keys, side * side),
        "saturated_valid_winner_groups": groups(
            winners, active_saturation, text_count, camera_keys, side * side
        ),
        "valid_action_winner_matches_cls_fraction": float(same_as_cls[action_valid].float().mean()),
        "sample_head_all_valid_queries_share_key_fraction": float(same_all_queries.float().mean()),
        "top_valid_tokens": top_tokens(winners, valid, text_count, camera_keys, side),
        "per_head": [],
        "per_task": [],
        "per_query": [],
        "counterfactual_only": {},
    }
    for head in range(query.shape[1]):
        mask = torch.zeros_like(valid)
        mask[:, head] = valid[:, head]
        result["per_head"].append(
            {
                "head": head,
                "saturated_fraction": float((maxima[mask] > 1 - 1e-7).float().mean()),
                "entropy_nats_mean": float(entropy[mask].mean()),
                "winner_groups": groups(winners, mask, text_count, camera_keys, side * side),
                "top_tokens": top_tokens(winners, mask, text_count, camera_keys, side),
            }
        )
    task_ids = torch.tensor(metadata["task_indices"])
    for task in task_ids.unique().tolist():
        mask = valid & (task_ids == task)[:, None, None]
        result["per_task"].append(
            {
                "task_index": task,
                "samples": int((task_ids == task).sum()),
                "saturated_fraction": float((maxima[mask] > 1 - 1e-7).float().mean()),
                "winner_groups": groups(winners, mask, text_count, camera_keys, side * side),
                "top_tokens": top_tokens(winners, mask, text_count, camera_keys, side),
            }
        )
    for index in range(query.shape[-2]):
        mask = valid[:, :, index]
        result["per_query"].append(
            {
                "query": "CLS" if index == 0 else f"action_step_{index - 1}",
                "valid_rows": int(mask.sum()),
                "saturated_fraction": float((maxima[:, :, index][mask] > 1 - 1e-7).float().mean()),
            }
        )
    for scale in (0.1, 0.01, 0.001):
        result["counterfactual_only"][f"score_scale_{scale}"] = summarize(scores * scale, valid)[0]
    q_norm = query / query.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    k_norm = key / key.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    normalized = q_norm @ k_norm.transpose(-1, -2) / math.sqrt(query.shape[-1])
    if entry["mask"] is not None:
        normalized += entry["mask"].float()
    result["counterfactual_only"]["per_token_qk_unit_rms"] = summarize(normalized, valid)[0]
    return result


def embedding_attribution(name, entry, checkpoint, query_valid, camera_keys, side):
    """Decompose the linear K projection into learned camera/position and remaining inputs."""
    from safetensors import safe_open

    layer = int(name.split(".")[1])
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        projection = weights.get_tensor(f"online.layers.{layer}.cross_attn.in_proj_weight").chunk(3)[1]
        position = weights.get_tensor("online.image_encoder.patch_position")
        camera = weights.get_tensor("online.image_encoder.view_embedding.weight")[:, None].expand_as(position)
    query, key = [entry[field].float() for field in ("query", "key")]
    text_count = key.shape[-2] - len(camera_keys) * side * side
    valid = query_valid[:, None].expand(query.shape[:-1])
    scores = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
    if entry["mask"] is not None:
        scores += entry["mask"].float()
    result = {"limitation": "linear FP32 decomposition of saved BF16 K; upstream queries kept fixed"}
    for label, embeddings in (
        ("camera", camera),
        ("position", position),
        ("camera_plus_position", camera + position),
    ):
        contribution = (embeddings.flatten(0, 1) @ projection.T).view(-1, query.shape[1], query.shape[-1])
        contribution = contribution.permute(1, 0, 2)
        component_scores = torch.zeros_like(scores)
        component_scores[..., text_count:] = (
            query @ contribution.transpose(-1, -2) / math.sqrt(query.shape[-1])
        )
        stats, _, winners, _ = summarize(scores - component_scores, valid)
        result[label] = {
            "input_embedding_rms": float(embeddings.square().mean().sqrt()),
            "projected_key_contribution_rms": float(contribution.square().mean().sqrt()),
            "remaining_visual_key_rms": float(
                (key[..., text_count:, :] - contribution).square().mean().sqrt()
            ),
            "without_component": stats,
            "without_component_winner_groups": groups(winners, valid, text_count, camera_keys, side * side),
            "without_component_top_tokens": top_tokens(winners, valid, text_count, camera_keys, side),
        }
    return result


def main():
    """Run a read-only CPU analysis of existing attention captures and numeric cache indices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--update", type=int, default=2030)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, help="Matching saved weights for linear embedding attribution"
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    config = json.loads(args.config.read_text())
    cameras = config["camera_keys"]
    metadata, valid = batch_metadata(args.cache, args.plan, args.update, config["chunk_size"], cameras)
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        metadata["text_tokens_by_task"] = [
            tokenizer.convert_ids_to_tokens(tokenizer(task, truncation=True, max_length=128)["input_ids"])
            for task in metadata["task_strings"]
        ]
    result = {
        "provenance": {"input": str(args.input), "sampling_update": args.update},
        "definition": "pre-dropout attention softmax over keys; saturated means FP32 pmax > 1-1e-7",
        "limitations": "one saved failed-update batch, not a whole-run statistic or evidence of semantic correctness",
        "batch": metadata,
        "layers": {},
    }
    entries = torch.load(args.input, map_location="cpu", weights_only=True)
    for name, entry in entries.items():
        result["layers"][name] = analyze(name, entry, valid, metadata, cameras, 14)
        if args.checkpoint and "cross_attn" in name:
            result["layers"][name]["context_embedding_attribution"] = embedding_attribution(
                name, entry, args.checkpoint, valid, cameras, 14
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
    for name, entry in result["layers"].items():
        print(
            name,
            json.dumps(
                {key: entry[key] for key in ("all_rows", "valid_rows", "cls_rows", "valid_winner_groups")}
            ),
        )


if __name__ == "__main__":
    main()
