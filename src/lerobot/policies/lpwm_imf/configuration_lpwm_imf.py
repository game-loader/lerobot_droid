"""Configuration for the encoder-only first stage of LPWM-IMF."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub.constants import CONFIG_NAME

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, LRSchedulerConfig


@PreTrainedConfig.register_subclass("lpwm-imf")
@dataclass
class LPWMIMFConfig(PreTrainedConfig):
    """Configure the real DLPv3 visual encoder, without an action or dynamics head.

    Defaults match the visual portion of upstream LPWM's Sketchy configuration.
    LPWM encodes 64 particles before its decoder selects 30; the decoder itself
    is not constructed here. Gaussian foreground/background features are used.
    """

    n_obs_steps: int = 2
    image_size: int = 128
    normalize_rgb: bool = False
    freeze_encoder: bool = False
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    patch_size: int = 16
    n_kp_per_patch: int = 1
    n_kp_prior: int = 64
    n_kp_enc: int = 64
    n_kp_dec: int = 30  # Also controls upstream foreground masking before background encoding.
    anchor_s: float = 0.25
    mask_bg_in_enc: bool = True
    learned_feature_dim: int = 4
    learned_bg_feature_dim: int = 4

    obj_ch_mult_prior: tuple[int, ...] = (1, 4, 8)
    obj_ch_mult: tuple[int, ...] = (1, 4, 8)
    obj_base_ch: int = 32
    obj_final_cnn_ch: int = 32
    bg_ch_mult: tuple[int, ...] = (1, 1, 1, 2, 4)
    bg_base_ch: int = 32
    bg_final_cnn_ch: int = 32
    pad_mode: str = "zeros"
    use_resblock: bool = False
    num_res_blocks: int = 1
    cnn_mid_blocks: bool = False
    mlp_hidden_dim: int = 256
    pint_enc_layers: int = 1
    pint_enc_heads: int = 1
    attn_norm_type: str = "rms"
    dropout: float = 0.1
    optimizer_lr: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        positive = (
            "n_obs_steps",
            "image_size",
            "patch_size",
            "n_kp_per_patch",
            "n_kp_prior",
            "n_kp_enc",
            "n_kp_dec",
            "learned_feature_dim",
            "learned_bg_feature_dim",
            "obj_base_ch",
            "obj_final_cnn_ch",
            "bg_base_ch",
            "bg_final_cnn_ch",
            "num_res_blocks",
            "mlp_hidden_dim",
            "pint_enc_layers",
            "pint_enc_heads",
        )
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size.")
        proposals = (self.image_size // self.patch_size) ** 2 * self.n_kp_per_patch
        if self.n_kp_prior != proposals or self.n_kp_enc != proposals:
            raise ValueError(
                "This encoder uses upstream LPWM's unfiltered particle grid: n_kp_prior and n_kp_enc "
                f"must equal (image_size // patch_size)**2 * n_kp_per_patch = {proposals}."
            )
        if self.n_kp_dec > self.n_kp_enc:
            raise ValueError("n_kp_dec must not exceed n_kp_enc.")
        if not 0 < self.anchor_s < 1:
            raise ValueError("anchor_s must be strictly between 0 and 1.")
        if self.mlp_hidden_dim % self.pint_enc_heads:
            raise ValueError("mlp_hidden_dim must be divisible by pint_enc_heads.")
        if self.attn_norm_type not in {"rms", "ln"}:
            raise ValueError("attn_norm_type must be 'rms' or 'ln'.")
        if self.pad_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError("Unsupported CNN pad_mode.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        for name in ("obj_ch_mult_prior", "obj_ch_mult", "bg_ch_mult"):
            multipliers = getattr(self, name)
            if not multipliers or any(not isinstance(m, int) or m < 1 for m in multipliers):
                raise ValueError(f"{name} must contain positive integer channel multipliers.")
            base = self.bg_base_ch if name == "bg_ch_mult" else self.obj_base_ch
            if any((base * m) % 4 for m in multipliers):
                raise ValueError(f"{name} channels must be divisible by upstream GroupNorm's 4 groups.")
        glimpse = round(self.anchor_s * (self.image_size - 1))
        sizes = (
            (self.patch_size, self.obj_ch_mult_prior),
            (glimpse, self.obj_ch_mult),
            (self.image_size, self.bg_ch_mult),
        )
        if any(size < 2 ** (len(mult) - 1) for size, mult in sizes):
            raise ValueError("Image, patch or glimpse size is too small for the configured CNN downsampling.")
        if self.normalization_mapping.get("VISUAL", NormalizationMode.IDENTITY) != NormalizationMode.IDENTITY:
            raise ValueError("LPWM-IMF requires raw RGB input; VISUAL normalization must be IDENTITY.")

    @property
    def latent_dim(self) -> int:
        """Return foreground descriptor width: position, scale, depth, presence, appearance."""
        return 6 + self.learned_feature_dim

    def encoder_kwargs(self) -> dict[str, Any]:
        """Map this config to the pinned upstream DLPEncoder, stopping before latent context."""
        names = (
            "image_size",
            "patch_size",
            "n_kp_per_patch",
            "n_kp_prior",
            "n_kp_enc",
            "n_kp_dec",
            "anchor_s",
            "mask_bg_in_enc",
            "learned_feature_dim",
            "learned_bg_feature_dim",
            "obj_ch_mult_prior",
            "obj_ch_mult",
            "obj_base_ch",
            "obj_final_cnn_ch",
            "bg_ch_mult",
            "bg_base_ch",
            "bg_final_cnn_ch",
            "pad_mode",
            "use_resblock",
            "num_res_blocks",
            "cnn_mid_blocks",
            "mlp_hidden_dim",
            "attn_norm_type",
            "dropout",
        )
        return {
            **{name: getattr(self, name) for name in names},
            "cdim": 3,
            "n_views": 1,  # Independent views share weights; not upstream channel-concatenated multiview.
            "warmup_n_kp_ratio": 1.0,
            "kp_range": (-1, 1),
            "kp_activation": "tanh",
            "features_dist": "gauss",
            "obj_on_min": math.log(1e-4),
            "obj_on_max": math.log(100),
            "use_z_orig": True,
            "filtering_heuristic": "none",
            "particle_positional_embed": True,
            "particle_score": False,
            "projection_dim": self.mlp_hidden_dim,
            "pte_layers": self.pint_enc_layers,
            "pte_heads": self.pint_enc_heads,
            "timestep_horizon": self.n_obs_steps,
            "interaction_features": True,
            "interaction_depth": True,
            "interaction_obj_on": False,
            "add_particle_temp_embed": False,
            "context_dim": 0,
            "ctx_enc": None,
            "init_zero_bias": True,
            "init_ssm_last_layer": True,
            "init_conv_layers": True,
            "init_conv_fg_std": 0.02,
            "init_conv_bg_std": 0.005,
        }

    def _save_pretrained(self, save_directory: Path) -> None:
        # Serialize the registry tag explicitly, without requiring draccus's newer
        # two-argument encode API used by the shared base class.
        payload = draccus.encode(self)
        payload["type"] = self.type
        with (save_directory / CONFIG_NAME).open("w") as file:
            json.dump(payload, file, indent=4)

    def validate_features(self) -> None:
        """Require RGB cameras, without requiring robot state or actions for this stage."""
        if not self.image_features:
            raise ValueError("LPWM-IMF requires at least one VISUAL input feature.")
        for name, feature in self.image_features.items():
            if len(feature.shape) != 3 or feature.shape[0] != 3 or min(feature.shape) < 1:
                raise ValueError(f"{name} must have positive RGB shape (3, height, width).")

    @property
    def observation_delta_indices(self) -> list[int]:
        """Return the requested observation history indices."""
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> None:
        """Do not request action targets before an IMF head/objective is implemented."""
        return None

    @property
    def reward_delta_indices(self) -> None:
        """Do not request rewards for visual encoding."""
        return None

    def get_optimizer_preset(self) -> AdamConfig:
        """Provide an optimizer for callers that explicitly supply their own encoder objective."""
        return AdamConfig(lr=self.optimizer_lr)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """Do not prescribe a scheduler for the encoder-only stage."""
        return None
