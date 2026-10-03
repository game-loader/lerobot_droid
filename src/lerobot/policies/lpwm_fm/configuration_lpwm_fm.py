"""Language-conditioned LPWM + ordinary Flow Matching A/B experiment configuration."""

import math
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.policies.lpwm_imf.configuration_lpwm_imf import LPWMIMFConfig


@PreTrainedConfig.register_subclass("lpwm-fm")
@dataclass
class LPWMFMConfig(LPWMIMFConfig):
    """Keep real DLP structural defaults; B adds GT-action world supervision to A."""

    variant: str = "A"
    horizon: int = 16
    n_action_steps: int = 8
    num_inference_steps: int = 10
    action_dim: int = 7
    state_dim: int = 8
    language_dim: int = 512
    hidden_dim: int = 256
    n_heads: int = 8
    scene_n_layers: int = 2
    expert_n_layers: int = 4
    world_n_layers: int = 4
    world_hidden_dim: int = 256
    world_n_heads: int = 8
    dropout: float = 0.0
    action_token_repeat: int = 3
    world_weight: float = 1.0
    reconstruction_weight: float = 1.0
    prior_weight: float = 1e-3
    dynamics_weight: float = 1.0
    world_warmup_steps: int = 0
    world_ramp_steps: int = 0
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.variant not in {"A", "B"}:
            raise ValueError("variant must be 'A' (FM only) or 'B' (FM + GT-action world loss).")
        for name in (
            "horizon",
            "n_action_steps",
            "num_inference_steps",
            "action_dim",
            "state_dim",
            "language_dim",
            "hidden_dim",
            "n_heads",
            "scene_n_layers",
            "expert_n_layers",
            "world_n_layers",
            "world_hidden_dim",
            "world_n_heads",
            "action_token_repeat",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps must not exceed horizon.")
        if self.hidden_dim % self.n_heads or self.world_hidden_dim % self.world_n_heads:
            raise ValueError("Transformer widths must be divisible by their head counts.")
        for name in ("world_weight", "reconstruction_weight", "prior_weight", "dynamics_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in ("world_warmup_steps", "world_ramp_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if self.freeze_encoder:
            raise ValueError("A/B require a trainable shared DLP encoder; freeze_encoder must be False.")

    @property
    def world_layers(self) -> int:
        """Native robot dynamics compatibility name."""
        return self.world_n_layers

    @property
    def rec_weight(self) -> float:
        """World module reconstruction coefficient."""
        return self.reconstruction_weight

    @property
    def dyn_weight(self) -> float:
        """World module dynamic KL coefficient."""
        return self.dynamics_weight

    def validate_features(self) -> None:
        super().validate_features()
        if self.robot_state_feature is None or self.robot_state_feature.shape != (self.state_dim,):
            raise ValueError(f"observation.state must be a STATE feature with shape ({self.state_dim},).")
        if self.action_feature is None or self.action_feature.shape != (self.action_dim,):
            raise ValueError(f"action must be an ACTION feature with shape ({self.action_dim},).")

    @property
    def action_delta_indices(self) -> list[int]:
        """Targets start at the current frame, not at the start of observation history."""
        return list(range(self.horizon))
