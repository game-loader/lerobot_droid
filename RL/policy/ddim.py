# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Private stochastic DDIM transition used by diffusion reinforcement learning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from diffusers import DDIMScheduler
from torch import Tensor


@dataclass(frozen=True)
class DDIMStepOutput:
    previous_sample: Tensor
    mean: Tensor
    raw_std: Tensor
    std: Tensor
    probability_std: Tensor
    log_prob: Tensor


def _finite_number(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted) or (positive and converted <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}, got {value!r}")
    return converted


def _validate_tensor_pair(
    model_output: Tensor, sample: Tensor, *, check_finite: bool
) -> None:
    for name, value in (("model_output", model_output), ("sample", sample)):
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating-point torch.Tensor")
        if value.ndim < 2 or value.numel() == 0:
            raise ValueError(f"{name} must be nonempty with at least two dimensions")
        if check_finite and not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
    if model_output.shape != sample.shape:
        raise ValueError(
            f"model_output and sample shapes must match, got {tuple(model_output.shape)} "
            f"and {tuple(sample.shape)}"
        )
    if model_output.device != sample.device or model_output.dtype != sample.dtype:
        raise ValueError("model_output and sample must share device and dtype")


def _validate_generator_device(
    generator: torch.Generator | None, device: torch.device
) -> None:
    if generator is None:
        return
    if not isinstance(generator, torch.Generator):
        raise ValueError(f"generator must be a torch.Generator, got {type(generator).__name__}")
    generator_device = torch.device(generator.device)
    if generator_device.type != device.type:
        raise ValueError(
            f"generator device {generator_device} does not match sample device {device}"
        )
    if (
        generator_device.type == "cuda"
        and generator_device.index is not None
        and device.index is not None
        and generator_device.index != device.index
    ):
        raise ValueError(
            f"generator device {generator_device} does not match sample device {device}"
        )


def stochastic_ddim_step(
    *,
    scheduler: DDIMScheduler,
    model_output: Tensor,
    timestep: int,
    previous_timestep: int | None,
    sample: Tensor,
    eta: float,
    sigma_min: float,
    sigma_max: float,
    probability_sigma_min: float | None = None,
    previous_sample: Tensor | None = None,
    generator: torch.Generator | None = None,
    check_finite: bool = True,
) -> DDIMStepOutput:
    """Sample or replay one explicit stochastic DDIM schedule transition.

    ``std`` controls the transition mean and sampled exploration noise.
    ``probability_std`` is independently floored for the PPO likelihood only.
    """

    if not isinstance(scheduler, DDIMScheduler):
        raise ValueError(f"scheduler must be a DDIMScheduler, got {type(scheduler).__name__}")
    if not isinstance(check_finite, bool):
        raise ValueError(f"check_finite must be a bool, got {check_finite!r}")
    _validate_tensor_pair(model_output, sample, check_finite=check_finite)
    if isinstance(timestep, bool) or not isinstance(timestep, int):
        raise ValueError(f"timestep must be an integer, got {timestep!r}")
    num_train_timesteps = int(scheduler.config.num_train_timesteps)
    if not 0 <= timestep < num_train_timesteps:
        raise ValueError(
            f"timestep must be in [0, {num_train_timesteps}), got {timestep!r}"
        )
    if previous_timestep is not None:
        if isinstance(previous_timestep, bool) or not isinstance(previous_timestep, int):
            raise ValueError(
                f"previous_timestep must be an integer or None, got {previous_timestep!r}"
            )
        if not 0 <= previous_timestep < timestep:
            raise ValueError(
                "previous_timestep must be nonnegative and smaller than timestep, "
                f"got timestep={timestep} previous_timestep={previous_timestep}"
            )
    eta = _finite_number("eta", eta)
    if eta < 0:
        raise ValueError(f"eta must be nonnegative, got {eta!r}")
    sigma_min = _finite_number("sigma_min", sigma_min, positive=True)
    sigma_max = _finite_number("sigma_max", sigma_max, positive=True)
    if sigma_max < sigma_min:
        raise ValueError(
            f"sigma_max must be at least sigma_min ({sigma_min}), got {sigma_max}"
        )
    probability_sigma_min = (
        sigma_min
        if probability_sigma_min is None
        else _finite_number(
            "probability_sigma_min", probability_sigma_min, positive=True
        )
    )
    if bool(getattr(scheduler.config, "thresholding", False)):
        raise ValueError("DDIM thresholding is not supported by the RL transition")

    alpha_t = scheduler.alphas_cumprod[timestep].to(sample)
    alpha_previous = (
        scheduler.final_alpha_cumprod.to(sample)
        if previous_timestep is None
        else scheduler.alphas_cumprod[previous_timestep].to(sample)
    )
    beta_t = (1.0 - alpha_t).clamp_min(torch.finfo(sample.dtype).eps)
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        predicted_clean = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
        predicted_noise = model_output
    elif prediction_type == "sample":
        predicted_clean = model_output
        predicted_noise = (sample - alpha_t.sqrt() * predicted_clean) / beta_t.sqrt()
    else:
        raise ValueError(f"unsupported prediction_type={prediction_type!r}")
    if scheduler.config.clip_sample:
        clip_range = float(scheduler.config.clip_sample_range)
        predicted_clean = predicted_clean.clamp(-clip_range, clip_range)

    variance = ((1.0 - alpha_previous) / beta_t) * (1.0 - alpha_t / alpha_previous)
    raw_std = eta * variance.clamp_min(0).sqrt()
    std = raw_std.clamp(min=sigma_min, max=sigma_max)
    probability_std = std.clamp_min(probability_sigma_min)
    direction_scale = (1.0 - alpha_previous - std.square()).clamp_min(0).sqrt()
    mean = alpha_previous.sqrt() * predicted_clean + direction_scale * predicted_noise
    if previous_sample is None:
        _validate_generator_device(generator, sample.device)
        noise = torch.randn(
            sample.shape,
            dtype=sample.dtype,
            device=sample.device,
            generator=generator,
        )
        previous_sample = mean + std * noise
    else:
        if (
            not isinstance(previous_sample, Tensor)
            or previous_sample.shape != sample.shape
            or previous_sample.device != sample.device
            or previous_sample.dtype != sample.dtype
        ):
            raise ValueError("previous_sample must match sample shape, device, and dtype")
        if check_finite and not torch.isfinite(previous_sample).all().item():
            raise ValueError("previous_sample must contain only finite values")
    if previous_sample is None:
        raise AssertionError("previous_sample must be populated before log-probability evaluation")
    # DPPO-style surrogate likelihood: preserve rollout noise while preventing
    # near-deterministic tail transitions from dominating the policy ratio.
    residual = (previous_sample.detach() - mean) / probability_std
    log_prob = (
        -0.5 * residual.square()
        - probability_std.log()
        - 0.5 * math.log(2.0 * math.pi)
    )
    if check_finite:
        for name, value in (
            ("previous_sample", previous_sample),
            ("mean", mean),
            ("raw_std", raw_std),
            ("std", std),
            ("probability_std", probability_std),
            ("log_prob", log_prob),
        ):
            if not torch.isfinite(value).all().item():
                raise ValueError(f"DDIM {name} contains non-finite values")
    return DDIMStepOutput(
        previous_sample=previous_sample,
        mean=mean,
        raw_std=raw_std,
        std=std,
        probability_std=probability_std,
        log_prob=log_prob,
    )
