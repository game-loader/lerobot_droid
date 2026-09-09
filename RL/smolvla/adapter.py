"""SmolVLA action-expert adapter over frozen full-prefix K/V features."""

import copy
from pathlib import Path

import torch

from RL.smolvla.features import Condition, FrozenSmolVLM
from RL.smolvla.flow import FlowConfig, FlowTrace, flow_mean_std, gaussian_log_prob


def load_policy(checkpoint: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, NormalizationMode
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep

    checkpoint = checkpoint.resolve(strict=True)
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise ValueError("Expected a SmolVLA checkpoint")
    config.device = device
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True)
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    normalizer = next(s for s in pre.steps if isinstance(s, NormalizerProcessorStep))
    unnormalizer = next(s for s in post.steps if isinstance(s, UnnormalizerProcessorStep))
    for feature_type in (FeatureType.STATE, FeatureType.ACTION):
        if normalizer.norm_map.get(feature_type) != NormalizationMode.MEAN_STD:
            raise ValueError("This SmolVLA adapter requires saved MEAN_STD state/action normalization")
    if unnormalizer.norm_map.get(FeatureType.ACTION) != NormalizationMode.MEAN_STD:
        raise ValueError("Action postprocessor must use saved MEAN_STD statistics")
    state_stats = normalizer.state_dict()
    action_stats = unnormalizer.state_dict()
    for key, dimension in (
        ("observation.state", config.robot_state_feature.shape[0]),
        ("action", config.action_feature.shape[0]),
    ):
        for stat in ("mean", "std"):
            value = state_stats.get(f"{key}.{stat}")
            if value is None or value.shape != (dimension,) or not torch.isfinite(value).all():
                raise ValueError(f"Missing or incompatible trained statistics for {key}.{stat}")
        if (state_stats[f"{key}.std"] < 0).any():
            raise ValueError("Negative normalization std")
    for stat in ("mean", "std"):
        if not torch.equal(state_stats[f"action.{stat}"], action_stats[f"action.{stat}"]):
            raise ValueError("Input/output action statistics disagree")
    active = (state_stats["action.max"] - state_stats["action.min"]).abs() > 1e-6
    if not active.any():
        raise ValueError("No varying action dimensions in the checkpoint")
    return policy, pre, post, active.to(device)


class SmolVLAFlowAdapter:
    def __init__(self, policy, flow: FlowConfig):
        self.policy, self.flow = policy, flow
        self.horizon = policy.config.chunk_size
        self.execution_steps = policy.config.n_action_steps
        self.action_dim = policy.config.action_feature.shape[0]
        self.max_action_dim = policy.config.max_action_dim
        self.head_dim = policy.model.vlm_with_expert.config.text_config.head_dim
        self.policy.eval()

    def behavior_copy(self):
        # Only trainable actor components need duplication; share the EXACT frozen backbone.
        shared = [self.policy.model.vlm_with_expert.vlm, self.policy.model.state_proj]
        old = copy.deepcopy(self.policy, memo={id(module): module for module in shared})
        old.requires_grad_(False)
        return SmolVLAFlowAdapter(old, self.flow)

    def velocity(self, condition: Condition, latent, index):
        wrapper = self.policy.model.vlm_with_expert
        dtype = next(wrapper.vlm.parameters()).dtype
        return self.policy.model.denoise_step(
            prefix_pad_masks=condition.mask,
            past_key_values=condition.cache(self.head_dim, dtype),
            x_t=latent,
            timestep=torch.full((latent.shape[0],), 1.0 - index / self.flow.steps, device=latent.device),
        )

    @torch.no_grad()
    def sample(self, condition: Condition, generator=None, *, stochastic=True, noise=None):
        self.policy.eval()
        shape = (condition.kv.shape[0], self.horizon, self.max_action_dim)
        x = (
            torch.randn(shape, device=condition.kv.device, generator=generator)
            if noise is None
            else noise.clone()
        )
        if tuple(x.shape) != shape:
            raise ValueError("Noise must cover the full prediction horizon and padded action width")
        xs, ys, logs = [], [], []
        for index in range(self.flow.steps):
            velocity = self.velocity(condition, x, index)
            if stochastic:
                mean, std = flow_mean_std(x, velocity, index, self.flow)
                y = mean + std * torch.randn(x.shape, device=x.device, generator=generator)
                log = gaussian_log_prob(y, mean, std)
            else:
                y = x - velocity / self.flow.steps
                log = torch.zeros_like(y)  # deterministic evaluation; NEVER use these as training log-probs
            xs.append(x.detach())
            ys.append(y.detach())
            logs.append(log.detach())
            x = y
        return FlowTrace(torch.stack(xs), torch.stack(ys), torch.stack(logs))

    def replay_step(self, condition, trace, index):
        mean, std = flow_mean_std(
            trace.latents[index], self.velocity(condition, trace.latents[index], index), index, self.flow
        )
        return gaussian_log_prob(trace.next_latents[index], mean, std)

    def executed(self, value):
        return value[..., : self.execution_steps, : self.action_dim]

    def actor_parameters(self):
        return [p for p in self.policy.parameters() if p.requires_grad]

    def synchronize_to(self, behavior):
        source = dict(self.policy.named_parameters())
        with torch.no_grad():
            for name, parameter in behavior.policy.named_parameters():
                if source[name].requires_grad:
                    parameter.copy_(source[name])


def make_adapters(policy, preprocessor, flow):
    encoder = FrozenSmolVLM(policy, preprocessor)
    current = SmolVLAFlowAdapter(policy, flow)
    return encoder, current, current.behavior_copy()
