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

"""LeRobot Diffusion Policy adapter for stochastic denoising traces."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from diffusers import DDIMScheduler
from torch import Tensor

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from RL.adapters.checkpoint import CheckpointAdapter
from RL.config import TraceConfig
from RL.policy.ddim import stochastic_ddim_step
from RL.types import DenoisingTrace, ObservationBatch


class DiffusionRLAdapter:
    """Sample and replay private stochastic DDIM traces without changing standard inference."""

    def __init__(self, checkpoint: CheckpointAdapter, trace_config: TraceConfig) -> None:
        if not isinstance(checkpoint, CheckpointAdapter):
            raise ValueError(
                f"checkpoint must be a CheckpointAdapter, got {type(checkpoint).__name__}"
            )
        if not isinstance(trace_config, TraceConfig):
            raise ValueError(
                f"trace_config must be a TraceConfig, got {type(trace_config).__name__}"
            )
        self.checkpoint = checkpoint
        self.trace_config = trace_config
        config = checkpoint.policy.config
        action_feature = config.action_feature
        state_feature = config.robot_state_feature
        if action_feature is None or len(action_feature.shape) != 1:
            raise ValueError(f"policy must define a vector action feature, got {action_feature!r}")
        if state_feature is None or len(state_feature.shape) != 1:
            raise ValueError(f"policy must define a vector state feature, got {state_feature!r}")
        if config.n_obs_steps <= 0 or config.horizon <= 0 or config.n_action_steps <= 0:
            raise ValueError("policy observation, horizon, and action step counts must be positive")
        self._execution_start = config.n_obs_steps - 1
        self._execution_end = self._execution_start + config.n_action_steps
        if self._execution_end > config.horizon:
            raise ValueError(
                "policy executable action window exceeds its horizon, "
                f"window=({self._execution_start}, {self._execution_end}) horizon={config.horizon}"
            )

        source_scheduler = checkpoint.policy.diffusion.noise_scheduler.config
        self.scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            set_alpha_to_one=bool(getattr(source_scheduler, "set_alpha_to_one", True)),
            steps_offset=int(getattr(source_scheduler, "steps_offset", 0)),
            prediction_type=config.prediction_type,
            thresholding=bool(getattr(source_scheduler, "thresholding", False)),
            clip_sample_range=config.clip_sample_range,
            timestep_spacing=str(getattr(source_scheduler, "timestep_spacing", "leading")),
            rescale_betas_zero_snr=bool(
                getattr(source_scheduler, "rescale_betas_zero_snr", False)
            ),
        )
        self.scheduler.set_timesteps(trace_config.num_inference_steps)
        self._timesteps = tuple(int(timestep) for timestep in self.scheduler.timesteps.tolist())
        if not self._timesteps:
            raise ValueError("DDIM inference schedule must be nonempty")
        if any(
            current <= following
            for current, following in zip(
                self._timesteps, self._timesteps[1:], strict=False
            )
        ):
            raise ValueError(f"DDIM timesteps must be strictly descending, got {self._timesteps}")
        checkpoint.policy.eval()

    @property
    def policy(self) -> DiffusionPolicy:
        return self.checkpoint.policy

    @property
    def timesteps(self) -> tuple[int, ...]:
        return self._timesteps

    @property
    def execution_slice(self) -> slice:
        return slice(self._execution_start, self._execution_end)

    def _prepare_global_conditioning(self, observation: ObservationBatch) -> Tensor:
        if not isinstance(observation, ObservationBatch):
            raise ValueError(
                f"observation must be an ObservationBatch, got {type(observation).__name__}"
            )
        device = get_device_from_parameters(self.policy)
        dtype = get_dtype_from_parameters(self.policy)
        normalized = self.checkpoint.normalize_observation(observation.to(device))
        batch = {
            key: value.to(dtype=dtype) if value.is_floating_point() else value
            for key, value in normalized.features.items()
        }
        if OBS_STATE not in batch:
            raise ValueError(f"observation is missing required policy state key {OBS_STATE!r}")
        expected_state_shape = (
            observation.batch_size(),
            self.policy.config.n_obs_steps,
            self.policy.config.robot_state_feature.shape[0],
        )
        if batch[OBS_STATE].shape != expected_state_shape:
            raise ValueError(
                f"{OBS_STATE} must have shape {expected_state_shape}, "
                f"got {tuple(batch[OBS_STATE].shape)}"
            )

        image_keys: Sequence[str] = tuple(self.policy.config.image_features)
        if image_keys:
            missing = [key for key in image_keys if key not in batch]
            if missing:
                raise ValueError(f"observation is missing policy image features: {missing}")
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in image_keys], dim=-4)
        conditioning = self.policy.diffusion._prepare_global_conditioning(batch)
        if conditioning.shape[0] != observation.batch_size():
            raise ValueError(
                "policy conditioning batch size disagrees with observation, "
                f"got {tuple(conditioning.shape)}"
            )
        if not torch.isfinite(conditioning).all().item():
            raise ValueError("policy conditioning contains non-finite values")
        return conditioning

    def sample_trace(
        self,
        observation: ObservationBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> DenoisingTrace:
        """Sample a detached complete trace from the current policy."""

        device = get_device_from_parameters(self.policy)
        dtype = get_dtype_from_parameters(self.policy)
        batch_size = observation.batch_size()
        horizon = self.policy.config.horizon
        action_dim = self.policy.config.action_feature.shape[0]
        latents: list[Tensor] = []
        next_latents: list[Tensor] = []
        log_probs: list[Tensor] = []
        self.policy.eval()
        with torch.no_grad():
            global_cond = self._prepare_global_conditioning(observation)
            sample = torch.randn(
                (batch_size, horizon, action_dim),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            for index, timestep in enumerate(self._timesteps):
                previous_timestep = (
                    self._timesteps[index + 1] if index + 1 < len(self._timesteps) else None
                )
                latents.append(sample.detach())
                timestep_batch = torch.full(
                    (batch_size,), timestep, dtype=torch.long, device=device
                )
                model_output = self.policy.diffusion.unet(
                    sample, timestep_batch, global_cond=global_cond
                )
                output = stochastic_ddim_step(
                    scheduler=self.scheduler,
                    model_output=model_output,
                    timestep=timestep,
                    previous_timestep=previous_timestep,
                    sample=sample,
                    eta=self.trace_config.eta,
                    sigma_min=self.trace_config.sigma_min,
                    sigma_max=self.trace_config.sigma_max,
                    generator=generator,
                )
                next_latents.append(output.previous_sample.detach())
                log_probs.append(output.log_prob.detach())
                sample = output.previous_sample
        return DenoisingTrace(
            latents=torch.stack(latents),
            next_latents=torch.stack(next_latents),
            timesteps=torch.tensor(self._timesteps, dtype=torch.long, device=device),
            old_log_prob=torch.stack(log_probs),
            final_actions=sample.detach(),
        )

    def recompute_log_prob(
        self, observation: ObservationBatch, trace: DenoisingTrace
    ) -> Tensor:
        """Replay stored transitions under current weights with differentiable means."""

        if not isinstance(trace, DenoisingTrace):
            raise ValueError(f"trace must be a DenoisingTrace, got {type(trace).__name__}")
        expected_shape = (
            len(self._timesteps),
            observation.batch_size(),
            self.policy.config.horizon,
            self.policy.config.action_feature.shape[0],
        )
        if trace.latents.shape != expected_shape:
            raise ValueError(
                f"trace latents must have shape {expected_shape}, got {tuple(trace.latents.shape)}"
            )
        trace_timesteps = tuple(int(value) for value in trace.timesteps.detach().cpu().tolist())
        if trace_timesteps != self._timesteps:
            raise ValueError(
                f"trace timesteps disagree with adapter schedule: {trace_timesteps} != {self._timesteps}"
            )
        device = get_device_from_parameters(self.policy)
        dtype = get_dtype_from_parameters(self.policy)
        self.policy.eval()
        global_cond = self._prepare_global_conditioning(observation)
        replayed: list[Tensor] = []
        for index, timestep in enumerate(self._timesteps):
            previous_timestep = (
                self._timesteps[index + 1] if index + 1 < len(self._timesteps) else None
            )
            sample = trace.latents[index].to(device=device, dtype=dtype)
            stored_previous = trace.next_latents[index].to(device=device, dtype=dtype)
            timestep_batch = torch.full(
                (observation.batch_size(),), timestep, dtype=torch.long, device=device
            )
            model_output = self.policy.diffusion.unet(
                sample, timestep_batch, global_cond=global_cond
            )
            output = stochastic_ddim_step(
                scheduler=self.scheduler,
                model_output=model_output,
                timestep=timestep,
                previous_timestep=previous_timestep,
                sample=sample,
                eta=self.trace_config.eta,
                sigma_min=self.trace_config.sigma_min,
                sigma_max=self.trace_config.sigma_max,
                previous_sample=stored_previous,
            )
            replayed.append(output.log_prob)
        result = torch.stack(replayed)
        if not torch.isfinite(result).all().item():
            raise ValueError("replayed log probability contains non-finite values")
        return result

    def executable_log_prob(self, log_prob: Tensor) -> Tensor:
        if not isinstance(log_prob, Tensor) or log_prob.ndim != 4:
            raise ValueError("log_prob must have shape [denoise,batch,horizon,action_dim]")
        expected_tail = (
            self.policy.config.horizon,
            self.policy.config.action_feature.shape[0],
        )
        if tuple(log_prob.shape[-2:]) != expected_tail:
            raise ValueError(
                f"log_prob must end in shape {expected_tail}, got {tuple(log_prob.shape)}"
            )
        return log_prob[:, :, self.execution_slice, :]

    def executable_actions(self, trace: DenoisingTrace) -> Tensor:
        if not isinstance(trace, DenoisingTrace):
            raise ValueError(f"trace must be a DenoisingTrace, got {type(trace).__name__}")
        expected_shape = (
            trace.final_actions.shape[0],
            self.policy.config.horizon,
            self.policy.config.action_feature.shape[0],
        )
        if trace.final_actions.shape != expected_shape:
            raise ValueError(
                f"trace final_actions must have shape {expected_shape}, "
                f"got {tuple(trace.final_actions.shape)}"
            )
        normalized = trace.final_actions[:, self.execution_slice, :]
        return self.checkpoint.unnormalize_action(normalized)
