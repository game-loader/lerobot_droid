"""Read-only activation and gradient tracing for the action-query critic."""

import math

import torch
import torch.nn.functional as functional
from torch import nn


def tensor_statistics(tensor):
    """Keep scalar GPU measurements, not activation graphs or full tensor copies."""
    value = tensor.detach().float()
    return {
        "rms": value.square().mean().sqrt(),
        "l2": value.norm(),
        "abs_max": value.abs().max(),
        "mean": value.mean(),
        "nonfinite_fraction": (~torch.isfinite(value)).float().mean(),
    }


def norm_statistics(tensor, module):
    """Measure the variance and inverse scale seen by an actual LayerNorm."""
    value = tensor.detach().float()
    axes = tuple(range(value.ndim - len(module.normalized_shape), value.ndim))
    variance = value.var(axes, unbiased=False).flatten()
    quantiles = torch.quantile(variance, torch.tensor([0.0, 0.01, 0.5], device=value.device))
    gain = module.weight.detach().float().abs().max() if module.weight is not None else 1.0
    return {
        "variance_min": quantiles[0],
        "variance_p01": quantiles[1],
        "variance_median": quantiles[2],
        "inverse_scale_max": (quantiles[0] + module.eps).rsqrt(),
        "jacobian_bound": gain * (quantiles[0] + module.eps).rsqrt(),
        "epsilon": module.eps,
    }


def first_tensor(value):
    """Extract the feature tensor from attention tuples and HF model outputs."""
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    if isinstance(value, (tuple, list)):
        return next((first_tensor(v) for v in value if first_tensor(v) is not None), None)
    return None


def traced_module(name, module):
    """Trace every decoder block, its branches/norms, and every DINO block/norm."""
    if isinstance(module, (nn.LayerNorm, nn.MultiheadAttention)):
        return True
    if name.isdigit():
        return True
    if name in {"image_encoder", "action_projection", "text_projection", "head"}:
        return True
    parts = name.split(".")
    if parts[0] == "layers":
        return len(parts) == 2 or (len(parts) == 3 and parts[-1] == "ffn")
    return len(parts) >= 2 and parts[-2] in {"layer", "blocks"} and parts[-1].isdigit()


class LayerTrace:
    """Hooks do not change values; input gradients include all fan-out paths."""

    def __init__(self, network):
        """Attach hooks to the selected network modules."""
        self.measurements = {}
        self.handles = [
            module.register_forward_hook(self.hook(name))
            for name, module in network.named_modules()
            if traced_module(name, module)
        ]

    def watch(self, name, tensor):
        """Measure a feature tensor and subscribe to its backward gradient."""
        self.measurements[name] = tensor_statistics(tensor)
        if tensor.requires_grad:
            tensor.register_hook(lambda gradient: self.capture_gradient(name, gradient))

    def capture_gradient(self, name, gradient):
        """Store scalar gradient measurements."""
        self.measurements[name + ".gradient"] = tensor_statistics(gradient)

    def hook(self, name):
        """Attach only scalar-statistic callbacks to the live autograd tensors."""

        def capture(module, args, output):
            for i, argument in enumerate(args):
                if isinstance(argument, torch.Tensor) and argument.is_floating_point():
                    self.watch(f"{name}.input{i}", argument)
            tensor = first_tensor(output)
            if tensor is not None:
                self.watch(name + ".output", tensor)
            if isinstance(module, nn.LayerNorm):
                self.measurements[name + ".normalization"] = norm_statistics(args[0], module)

        return capture

    def close(self):
        """Remove all forward hooks."""
        for handle in self.handles:
            handle.remove()

    def report(self):
        """Materialize the scalar measurements on CPU."""
        return {key: {k: float(v) for k, v in values.items()} for key, values in self.measurements.items()}


def parameter_report(network):
    """Report group norms and the largest individual parameter gradients."""
    groups = {}
    parameters = []
    for name, parameter in network.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        parts = name.split(".")
        if parts[0] == "layers":
            group = ".".join(parts[:2])
        elif "layer" in parts or "blocks" in parts:
            index = parts.index("layer") if "layer" in parts else parts.index("blocks")
            group = ".".join(parts[: index + 2])
        else:
            group = ".".join(parts[:3]) if name.startswith("image_encoder.backbone") else parts[0]
        norm = gradient.norm()
        groups.setdefault(group, []).append(norm)
        parameters.append((name, norm, gradient.abs().max(), parameter.detach().float().norm()))
    group_norms = {name: float(torch.stack(norms).norm()) for name, norms in groups.items()}
    largest = sorted(
        (
            {"name": name, "grad_l2": float(norm), "grad_abs_max": float(maximum), "weight_l2": float(weight)}
            for name, norm, maximum, weight in parameters
        ),
        key=lambda entry: entry["grad_l2"],
        reverse=True,
    )[:30]
    return {
        "global_grad_l2": math.sqrt(sum(norm**2 for norm in group_norms.values())),
        "groups": group_norms,
        "largest_parameters": largest,
    }


class AttentionTrace:
    """Capture actual SDPA kernel names and selected decoder inputs for isolated replay."""

    def __init__(self, capture_layers=()):
        """Select zero-based decoder layers whose attention tensors should be retained."""
        self.original = functional.scaled_dot_product_attention
        self.capture_layers = capture_layers
        self.calls = []
        self.saved = {}

    def __enter__(self):
        """Intercept SDPA only for the lifetime of this diagnostic context."""
        functional.scaled_dot_product_attention = self.forward
        return self

    def __exit__(self, *_):
        """Restore the original PyTorch function even if the probe fails."""
        functional.scaled_dot_product_attention = self.original

    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        """Run the original function unchanged, recording its actual autograd kernel."""
        selected = torch.is_grad_enabled() and query.requires_grad and query.shape[-2] == 33
        index = len(self.calls)
        name = f"layers.{index // 2}.{'self_attn' if index % 2 == 0 else 'cross_attn'}"
        capture = selected and index // 2 in self.capture_layers
        rng = torch.cuda.get_rng_state() if capture else None
        output = self.original(query, key, value, attn_mask, dropout_p, is_causal, **kwargs)
        if selected:
            self.calls.append(
                {
                    "name": name,
                    "query_shape": list(query.shape),
                    "key_shape": list(key.shape),
                    "dtype": str(query.dtype),
                    "dropout": dropout_p,
                    "kernel": output.grad_fn.name(),
                }
            )
        if capture:
            entry = {
                "query": query.detach().cpu(),
                "key": key.detach().cpu(),
                "value": value.detach().cpu(),
                "output": output.detach().cpu(),
                "mask": attn_mask.detach().cpu() if attn_mask is not None else None,
                "dropout": dropout_p,
                "is_causal": is_causal,
                "kwargs": kwargs,
                "cuda_rng": rng.cpu(),
            }
            self.saved[name] = entry
            output.register_hook(lambda gradient: entry.update(gradient=gradient.detach().cpu()))
        return output
