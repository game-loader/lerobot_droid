"""Normalization and configuration from the custom FR3 FastWAM deployment bundle."""

import json
from pathlib import Path

import torch
import yaml


class BundleActionNormalizer:
    """Use fixed global action z-scores and clipping from the custom FastWAM processor."""

    def __init__(self, mean, std):
        """Retain the original per-coordinate statistics, shared across chunk steps."""
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    @classmethod
    def from_bundle(cls, bundle):
        """Read statistics without loading BC weights or recomputing dataset statistics."""
        fields = json.loads((Path(bundle) / "dataset_stats.json").read_text())["action"]["default"]
        return cls(fields["global_mean"], fields["global_std"])

    def __call__(self, actions):
        """Match FastWAM's affine transform and [-5, 5] clipping in FP32."""
        mean, std = self.mean.to(actions.device), self.std.to(actions.device)
        scale = 1 / (std + 1e-8)
        return (actions.float() * scale - mean * scale).clamp(-5, 5)

    def state_dict(self):
        """Save the exact normalization convention alongside the Q checkpoint."""
        return {"kind": "fastwam_bundle_global_zscore_v1", "mean": self.mean, "std": self.std}


def bundle_q_options(bundle):
    """Extract action horizon/dimension and camera keys from the supplied training config."""
    data = yaml.safe_load((Path(bundle) / "config.yaml").read_text())["data"]["train"]
    processor = data["processor"]
    if processor["use_stepwise_action_norm"] or processor["norm_default_mode"] != "z-score":
        raise ValueError("This adapter expects the FR3 bundle's global z-score normalization")
    return {
        "action_dim": data["shape_meta"]["action"][0]["shape"],
        "chunk_size": data["num_frames"] - 1,
        "camera_keys": tuple(f"observation.images.{v['key']}" for v in data["shape_meta"]["images"]),
    }
