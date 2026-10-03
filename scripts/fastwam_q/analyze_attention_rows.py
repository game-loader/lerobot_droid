"""Locate spurious BF16 query derivatives in identical captured attention rows."""

import argparse
import json
import math

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel


def attention_gradient(entry, dtype):
    """Replay the same efficient kernel/dropout RNG with different arithmetic precision."""
    query, key, value = [entry[name].to("cuda", dtype).requires_grad_() for name in ("query", "key", "value")]
    mask = entry["mask"]
    mask = None if mask is None else mask.to("cuda", dtype if mask.is_floating_point() else mask.dtype)
    torch.cuda.set_rng_state(entry["cuda_rng"].cpu())
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        output = functional.scaled_dot_product_attention(query, key, value, mask, entry["dropout"])
    output.backward(entry["gradient"].to("cuda", dtype))
    return query.grad.detach().float(), output.detach().float()


def analyze(entry):
    """Compare row gradients on bit-identical Q/K/V values, including saturated softmax rows."""
    bf_gradient, bf_output = attention_gradient(entry, torch.bfloat16)
    fp_gradient, fp_output = attention_gradient(entry, torch.float32)
    query, key = [entry[name].to("cuda", torch.float32) for name in ("query", "key")]
    scores = (query @ key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    if entry["mask"] is not None:
        scores += entry["mask"].to("cuda", torch.float32)
    probability = scores.softmax(-1)
    maxima = probability.max(-1).values
    saturated = maxima > 1 - 1e-7
    bf_norm, fp_norm = bf_gradient.norm(dim=-1), fp_gradient.norm(dim=-1)
    error = (bf_gradient - fp_gradient).norm(dim=-1)
    worst = torch.topk(error.flatten(), 5).indices.cpu().tolist()
    return {
        "query_rms": float(query.square().mean().sqrt()),
        "key_rms": float(key.square().mean().sqrt()),
        "score_abs_max_finite": float(scores[torch.isfinite(scores)].abs().max()),
        "softmax_entropy_mean": float(-(probability * probability.clamp_min(1e-30).log()).sum(-1).mean()),
        "saturated_row_fraction": float(saturated.float().mean()),
        "query_grad_bf16_l2": float(bf_gradient.norm()),
        "query_grad_fp32_l2": float(fp_gradient.norm()),
        "query_grad_relative_error": float(
            (bf_gradient - fp_gradient).norm() / fp_gradient.norm().clamp_min(1e-30)
        ),
        "query_grad_cosine": float(
            functional.cosine_similarity(bf_gradient.flatten(), fp_gradient.flatten(), dim=0)
        ),
        "saturated_query_grad_bf16_l2": float(bf_gradient[saturated].norm()),
        "saturated_query_grad_fp32_l2": float(fp_gradient[saturated].norm()),
        "forward_relative_error": float((bf_output - fp_output).norm() / fp_output.norm()),
        "worst_rows": [
            {
                "flattened_row": row,
                "p_max": float(maxima.flatten()[row]),
                "bf16_grad_l2": float(bf_norm.flatten()[row]),
                "fp32_grad_l2": float(fp_norm.flatten()[row]),
                "gradient_error_l2": float(error.flatten()[row]),
            }
            for row in worst
        ],
    }


def main():
    """Analyze a saved attention replay without loading or updating the critic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = {name: analyze(entry) for name, entry in torch.load(args.input, weights_only=True).items()}
    with open(args.output, "w") as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
