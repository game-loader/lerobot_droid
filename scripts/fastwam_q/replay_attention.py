"""Replay saved SDPA inputs and upstream gradients independently of Q/TD/DINO."""

import argparse
import contextlib
import json

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel


def replay(entry, backend, precision, dropout):
    """Use identical Q/K/V and upstream gradients for every backend/precision."""
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    tensors = [entry[key].to("cuda", dtype).requires_grad_() for key in ("query", "key", "value")]
    mask = entry["mask"]
    # The original CUDA autocast casts floating masks with Q/K/V at the SDPA boundary.
    mask = None if mask is None else mask.to("cuda", dtype if mask.is_floating_point() else mask.dtype)
    torch.cuda.set_rng_state(entry["cuda_rng"].cpu())
    context = sdpa_kernel(getattr(SDPBackend, backend)) if backend else contextlib.nullcontext()
    with context:
        output = functional.scaled_dot_product_attention(
            *tensors, mask, dropout, entry["is_causal"], **entry["kwargs"]
        )
    kernel = output.grad_fn.name()
    output.backward(entry["gradient"].to("cuda", dtype))
    return {
        "kernel": kernel,
        "output_abs_difference": float(
            (output.detach().float() - entry["output"].to("cuda").float()).abs().max()
        ),
        "gradient_l2": {
            key: float(tensor.grad.float().norm())
            for key, tensor in zip(("query", "key", "value"), tensors, strict=True)
        },
        "gradient_abs_max": {
            key: float(tensor.grad.float().abs().max())
            for key, tensor in zip(("query", "key", "value"), tensors, strict=True)
        },
    }


def main():
    """Publish unsupported kernels as errors instead of silently substituting them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    saved = torch.load(args.input, weights_only=True)
    results = {}
    for name, entry in saved.items():
        results[name] = {}
        for precision in ("bf16", "fp32"):
            for dropout in (entry["dropout"], 0.0):
                for backend in (None, "MATH", "EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION"):
                    label = f"{precision}_{backend or 'DEFAULT'}_dropout{dropout}"
                    try:
                        result = replay(entry, backend, precision, dropout)
                    except RuntimeError as error:
                        result = {"error": str(error)}
                    results[name][label] = result
                    print(json.dumps({"name": name, "variant": label, **result}), flush=True)
    with open(args.output, "w") as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
