#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, CosineDecayWithWarmupSchedulerConfig, LRSchedulerConfig


@PreTrainedConfig.register_subclass("imf-attnres")
@dataclass
class IMFAttnResConfig(PreTrainedConfig):
    """Configuration for the IMF-AttnRes diffusion-policy variant."""

    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    drop_n_last_frames: int | None = None

    # Vision encoder settings (same lightweight RGB encoder pattern as DiffusionPolicy).
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True
    output_tokens_per_camera: bool = False

    # IMF-AttnRes transformer head.
    n_layer: int = 12
    n_head: int = 8
    n_emb: int = 768
    p_drop_emb: float = 0.1
    p_drop_attn: float = 0.1
    causal_attn: bool = False
    time_as_cond: bool = True
    obs_as_cond: bool = True
    n_cond_layers: int = 0
    backbone_type: str = "attnres_full"
    n_kv_head: int = 8
    attn_res_ffn_mult: float = 2.667
    attn_res_eps: float = 1e-6
    attn_res_rope_theta: float = 10000.0

    # Inference / optimization.
    num_inference_steps: int = 1
    do_mask_loss_for_padding: bool = True
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    optimizer_grad_clip_norm: float = 10.0
    scheduler_type: str = "none"
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()
        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = self.horizon - self.n_action_steps - self.n_obs_steps + 1
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {self.n_obs_steps}.")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}.")
        if not (1 <= self.n_action_steps <= self.horizon - self.n_obs_steps + 1):
            raise ValueError(
                "n_action_steps must satisfy 1 <= n_action_steps <= horizon - n_obs_steps + 1. "
                f"Got {self.n_action_steps=}, {self.horizon=}, {self.n_obs_steps=}."
            )
        if self.num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {self.num_inference_steps}.")
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"vision_backbone must be a torchvision ResNet name, got {self.vision_backbone}.")
        if self.n_head < 1:
            raise ValueError(f"n_head must be >= 1, got {self.n_head}.")
        if self.n_kv_head < 1:
            raise ValueError(f"n_kv_head must be >= 1, got {self.n_kv_head}.")
        if self.n_emb % self.n_head != 0:
            raise ValueError(f"n_emb={self.n_emb} must be divisible by n_head={self.n_head}.")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head={self.n_head} must be divisible by n_kv_head={self.n_kv_head}."
            )
        if self.backbone_type == "attnres_full" and (self.n_emb // self.n_head) % 2 != 0:
            raise ValueError(
                "attnres_full uses RoPE, which requires an even per-head dimension. "
                f"Got n_emb={self.n_emb}, n_head={self.n_head}, "
                f"head_dim={self.n_emb // self.n_head}."
            )
        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"resize_shape must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"crop_ratio must be in (0, 1]. Got {self.crop_ratio}.")
        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"crop_shape must have positive dimensions. Got {self.crop_shape}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        if self.scheduler_type in ("none", "", None):
            return None
        if self.scheduler_type != "cosine_decay_with_warmup":
            raise ValueError(
                "IMF-AttnRes supports scheduler_type='none' or "
                f"'cosine_decay_with_warmup', got {self.scheduler_type!r}."
            )
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError("IMF-AttnRes requires 'observation.state' as an input feature.")
        if self.action_feature is None:
            raise ValueError("IMF-AttnRes requires 'action' as an output feature.")
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("IMF-AttnRes requires at least one image or an environment state input.")
        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"crop_shape should fit within image shapes. Got {self.crop_shape} for {key}: {image_ft.shape}."
                    )
        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_ft.shape:
                    raise ValueError(f"{key} does not match {first_key}; all image shapes must match.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
