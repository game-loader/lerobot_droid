"""Gaussian stochastic-flow transitions, separate from ordinary SmolVLA inference.

SmolVLA uses x(t)=(1-t)*data+t*noise, v=noise-data, and integrates t=1 -> 0.
Its score is -(x+(1-t)*v)/t. The reverse SDE drift is v-g^2*score/2.
We use constant diffusion g and Euler-Maruyama. Every training step, including
the last, has strictly positive variance; replay uses the SAME variance.
This is not a DDIM scheduler or a negative-MSE pseudo likelihood.
"""

import math
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class FlowConfig:
    steps: int = 10
    noise_level: float = 0.1

    def __post_init__(self):
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if isinstance(self.noise_level, bool) or not math.isfinite(self.noise_level) or self.noise_level <= 0:
            raise ValueError("noise_level must be finite and positive")


def flow_mean_std(x: Tensor, velocity: Tensor, index: int, config: FlowConfig) -> tuple[Tensor, float]:
    if not 0 <= index < config.steps or velocity.shape != x.shape:
        raise ValueError("Invalid flow step index or velocity shape")
    t = 1.0 - index / config.steps  # strictly positive, never evaluate the score at t=0
    dt = -1.0 / config.steps
    g = config.noise_level
    drift = velocity.float() + (g * g / (2 * t)) * (x.float() + (1 - t) * velocity.float())
    return x.float() + dt * drift, g * math.sqrt(-dt)


def gaussian_log_prob(sample: Tensor, mean: Tensor, std: float) -> Tensor:
    if not math.isfinite(std) or std <= 0 or sample.shape != mean.shape:
        raise ValueError("Gaussian replay requires matching tensors and positive finite std")
    return (
        -0.5 * ((sample.detach().float() - mean.float()) / std).square()
        - math.log(std)
        - 0.5 * math.log(2 * math.pi)
    )


@dataclass(frozen=True)
class FlowTrace:
    latents: Tensor
    next_latents: Tensor
    log_probs: Tensor

    @property
    def actions(self):
        return self.next_latents[-1]
