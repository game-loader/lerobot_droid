"""Read selected saved attention projection scales without loading a full critic."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def spectral_estimate(weight):
    """Estimate the leading singular value by deterministic power iteration."""
    vector = torch.ones(weight.shape[1])
    vector /= vector.norm()
    for _ in range(30):
        left = weight @ vector
        left /= left.norm().clamp_min(1e-30)
        vector = weight.T @ left
        vector /= vector.norm().clamp_min(1e-30)
    return float((weight @ vector).norm())


def main():
    """Compare absolute and anisotropic projection scales at saved training states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    results = {}
    for checkpoint in args.checkpoints:
        measurements = {}
        with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
            for index in (0, 14, 16, 17):
                for kind in ("self_attn", "cross_attn"):
                    name = f"online.layers.{index}.{kind}.in_proj_weight"
                    for component, weight in zip(
                        ("query", "key", "value"), weights.get_tensor(name).chunk(3), strict=True
                    ):
                        measurements[f"{name}.{component}"] = {
                            "weight_frobenius": float(weight.norm()),
                            "spectral_estimate": spectral_estimate(weight),
                        }
        results[str(checkpoint)] = measurements
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
