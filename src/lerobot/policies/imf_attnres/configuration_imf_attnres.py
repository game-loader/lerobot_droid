#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig
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

    # Optional SmolVLM visual-language encoder path. Disabled by default so the
    # existing ResNet RGB encoder remains the standard behavior.
    use_smolvlm_vl_encoder: bool = False
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = True
    freeze_vlm_encoder: bool = True
    vlm_encoder_torch_dtype: str | None = "bfloat16"
    vlm_tokenizer_max_length: int = 48
    vlm_pad_language_to: str = "longest"
    vlm_tokenizer_padding_side: str = "right"
    vlm_tokenizer_truncation: bool = True
    vlm_resize_shape: tuple[int, int] = (512, 512)
    # Maximum flattened image batch passed through SmolVLM's ViT at once.
    # 0 disables chunking and preserves the previous single-call behavior.
    vlm_image_forward_batch_size: int = 0
    # Text conditioning for the SmolVLM path:
    #   "embedding"    — current memory-light path: use token embeddings only.
    #   "transformer"  — run language embeddings through the first vlm_text_num_layers
    #                    SmolVLM text transformer layers, like SmolVLA's truncated VLM.
    vlm_text_encoder_mode: str = "embedding"
    vlm_text_num_layers: int = 16
    # In transformer mode, include the projected robot state token in the SmolVLM
    # text-layer prefix, matching SmolVLA. Set False to append state after VLM text layers.
    vlm_state_in_text_layers: bool = True
    # How SmolVLM conditions the IMF action head:
    #   "flat_tokens" — encode the visual-language-state prefix once and prepend it to the action head.
    #   "layerwise"   — expose SmolVLM prefix hidden states per action-head layer, then couple them
    #                   with IMF action tokens layer-by-layer in a SmolVLA-style pattern.
    vlm_conditioning_mode: str = "flat_tokens"
    # Layer-wise coupling schedule. With the default "cross_attn" and interval 2, even action-head
    # layers jointly self-attend over prefix+action tokens, while odd layers cross-attend action tokens
    # to the corresponding SmolVLM prefix state.
    vlm_layerwise_attention_mode: str = "cross_attn"
    vlm_layerwise_self_attn_every_n_layers: int = 2
    vlm_hidden_size: int | None = None
    vlm_tokens_per_step: int | None = None

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
    # Transformer backbone architecture for the action head:
    #   "attnres_full"    — AttnRes with full multi-head attention + RoPE (default, best performing)
    #   "attnres_diff"    — AttnRes with differential attention (subtracts two attention heads for noise cancellation)
    #   "diff_transformer" — Standalone Differential Transformer backbone (separate implementation from AttnRes)
    #   "vanilla"         — Standard nn.TransformerEncoder/Decoder (no RoPE, uses learned positional embeddings)
    backbone_type: str = "attnres_full"
    # Number of key-value heads for GQA (grouped-query attention). Must divide n_head evenly.
    n_kv_head: int = 8
    # FFN hidden-dim multiplier relative to n_emb (hidden = n_emb * ffn_mult, rounded to nearest multiple of 256).
    attn_res_ffn_mult: float = 2.667
    # Epsilon for RMSNorm layers in AttnRes/DiffTransformer backbones.
    attn_res_eps: float = 1e-6
    # Base frequency for Rotary Position Embeddings (RoPE).
    attn_res_rope_theta: float = 10000.0

    # Inference / loss computation / optimization.
    num_inference_steps: int = 1
    do_mask_loss_for_padding: bool = True
    p_mean: float = -0.4
    p_std: float = 1.0
    data_proportion: float = 0.5
    loss_type: str = "pseudo_huber"
    pseudo_huber_delta: float = 1.0
    # Action latent representation before denoising:
    #   "dct"      — Apply Discrete Cosine Transform to action sequences, denoising in frequency domain.
    #                Encourages smooth trajectories by weighting high-frequency components more in the loss.
    #   "identity" — Denoise raw action sequences directly (no transform).
    action_latent_mode: str = "dct"
    # DCT loss: extra weight on high-frequency coefficients. Higher values penalize jittery actions more.
    dct_loss_high_freq_weight: float = 1.0
    # DCT loss: exponent for frequency-dependent weighting curve. weight_k = 1 + high_freq_weight * (k/K)^power.
    dct_loss_freq_power: float = 2.0
    # Semigroup consistency regularization: enforces that composing two flow steps equals a single direct step
    # (i.e., flow(r→t) ≈ flow(r→s) ∘ flow(s→t)), improving temporal coherence of the learned flow field.
    enable_semigroup_consistency: bool = False
    # Target weight for the semigroup consistency loss term (added to the main velocity loss).
    semigroup_loss_weight: float = 0.0
    # Training step at which semigroup loss begins (allows the model to learn basic flow first).
    semigroup_start_step: int = 0
    # Number of steps to linearly ramp semigroup_loss_weight from 0 to target after start_step.
    semigroup_warmup_steps: int = 0
    # Minimum time gap between sampled r, s, t points. Prevents degenerate near-zero intervals.
    semigroup_min_time_delta: float = 1e-3
    # EMA decay used for the no-grad teacher action head in semigroup consistency.
    semigroup_teacher_ema_decay: float = 0.995
    # Log detailed per-step diagnostics (velocity norms, loss buckets, gradient stats) to the logger.
    enable_imf_diagnostics: bool = False
    # Loss threshold above which a training step is flagged as a "spike" in diagnostics.
    imf_diagnostics_spike_loss_threshold: float = 0.2
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
        if self.use_smolvlm_vl_encoder:
            self.normalization_mapping = dict(self.normalization_mapping)
            self.normalization_mapping["VISUAL"] = NormalizationMode.IDENTITY
            self.normalization_mapping[FeatureType.VISUAL] = NormalizationMode.IDENTITY
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
        if self.p_std <= 0:
            raise ValueError(f"p_std must be > 0, got {self.p_std}.")
        if not (0.0 <= self.data_proportion <= 1.0):
            raise ValueError(f"data_proportion must be in [0, 1], got {self.data_proportion}.")
        if self.loss_type not in {"pseudo_huber", "mse"}:
            raise ValueError(
                "loss_type must be one of {'pseudo_huber', 'mse'}, got "
                f"{self.loss_type!r}."
            )
        if self.pseudo_huber_delta <= 0:
            raise ValueError(f"pseudo_huber_delta must be > 0, got {self.pseudo_huber_delta}.")
        if self.action_latent_mode not in {"dct", "identity"}:
            raise ValueError(
                "action_latent_mode must be one of {'dct', 'identity'}, got "
                f"{self.action_latent_mode!r}."
            )
        if self.dct_loss_high_freq_weight < 0:
            raise ValueError(
                f"dct_loss_high_freq_weight must be >= 0, got {self.dct_loss_high_freq_weight}."
            )
        if self.dct_loss_freq_power <= 0:
            raise ValueError(f"dct_loss_freq_power must be > 0, got {self.dct_loss_freq_power}.")
        if self.semigroup_loss_weight < 0:
            raise ValueError(
                f"semigroup_loss_weight must be >= 0, got {self.semigroup_loss_weight}."
            )
        if self.semigroup_start_step < 0:
            raise ValueError(
                f"semigroup_start_step must be >= 0, got {self.semigroup_start_step}."
            )
        if self.semigroup_warmup_steps < 0:
            raise ValueError(
                f"semigroup_warmup_steps must be >= 0, got {self.semigroup_warmup_steps}."
            )
        if not (0.0 <= self.semigroup_min_time_delta < 1 / 3):
            raise ValueError(
                "semigroup_min_time_delta must satisfy 0 <= value < 1/3, got "
                f"{self.semigroup_min_time_delta}."
            )
        if not (0.0 <= self.semigroup_teacher_ema_decay < 1.0):
            raise ValueError(
                "semigroup_teacher_ema_decay must satisfy 0 <= value < 1, got "
                f"{self.semigroup_teacher_ema_decay}."
            )
        if self.imf_diagnostics_spike_loss_threshold < 0:
            raise ValueError(
                "imf_diagnostics_spike_loss_threshold must be >= 0, got "
                f"{self.imf_diagnostics_spike_loss_threshold}."
            )
        if not self.use_smolvlm_vl_encoder and not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"vision_backbone must be a torchvision ResNet name, got {self.vision_backbone}.")
        if self.vlm_tokenizer_max_length <= 0:
            raise ValueError(
                f"vlm_tokenizer_max_length must be a positive integer. Got {self.vlm_tokenizer_max_length}."
            )
        if self.vlm_image_forward_batch_size < 0:
            raise ValueError(
                "vlm_image_forward_batch_size must be >= 0, where 0 disables chunking. "
                f"Got {self.vlm_image_forward_batch_size}."
            )
        if self.vlm_text_encoder_mode not in {"embedding", "transformer"}:
            raise ValueError(
                "vlm_text_encoder_mode must be one of {'embedding', 'transformer'}, got "
                f"{self.vlm_text_encoder_mode!r}."
            )
        if self.vlm_conditioning_mode not in {"flat_tokens", "layerwise"}:
            raise ValueError(
                "vlm_conditioning_mode must be one of {'flat_tokens', 'layerwise'}, got "
                f"{self.vlm_conditioning_mode!r}."
            )
        if self.vlm_conditioning_mode == "layerwise":
            if not self.use_smolvlm_vl_encoder:
                raise ValueError("vlm_conditioning_mode='layerwise' requires use_smolvlm_vl_encoder=True.")
            if self.vlm_text_encoder_mode != "transformer":
                raise ValueError(
                    "vlm_conditioning_mode='layerwise' requires vlm_text_encoder_mode='transformer'."
                )
        if self.vlm_layerwise_attention_mode not in {"self_attn", "cross_attn"}:
            raise ValueError(
                "vlm_layerwise_attention_mode must be one of {'self_attn', 'cross_attn'}, got "
                f"{self.vlm_layerwise_attention_mode!r}."
            )
        if self.vlm_layerwise_self_attn_every_n_layers < 1:
            raise ValueError(
                "vlm_layerwise_self_attn_every_n_layers must be >= 1, got "
                f"{self.vlm_layerwise_self_attn_every_n_layers}."
            )
        if self.vlm_text_num_layers < 0:
            raise ValueError(
                f"vlm_text_num_layers must be non-negative. Got {self.vlm_text_num_layers}."
            )
        if self.vlm_text_encoder_mode == "transformer" and self.vlm_text_num_layers < 1:
            raise ValueError(
                "vlm_text_num_layers must be >= 1 when vlm_text_encoder_mode='transformer'. "
                f"Got {self.vlm_text_num_layers}."
            )
        if not (
            isinstance(self.vlm_resize_shape, tuple)
            and len(self.vlm_resize_shape) == 2
            and all(isinstance(d, int) and d > 0 for d in self.vlm_resize_shape)
        ):
            raise ValueError(
                f"vlm_resize_shape must be a pair of positive integers. Got {self.vlm_resize_shape}."
            )
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
        if self.backbone_type not in {"attnres_full", "attnres_diff", "vanilla", "diff_transformer"}:
            raise ValueError(
                "backbone_type must be one of "
                f"{'attnres_full', 'attnres_diff', 'vanilla', 'diff_transformer'}, "
                f"got {self.backbone_type!r}."
            )
        if self.backbone_type in {"attnres_full", "attnres_diff", "diff_transformer"} and (
            self.n_emb // self.n_head
        ) % 2 != 0:
            raise ValueError(
                f"{self.backbone_type} uses RoPE, which requires an even per-head dimension. "
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
