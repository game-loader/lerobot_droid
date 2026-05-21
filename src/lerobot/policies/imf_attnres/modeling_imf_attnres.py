#!/usr/bin/env python

from collections import deque
from contextlib import nullcontext
from copy import deepcopy
import math

import einops
import torch
import torchvision
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import SpatialSoftmax, _replace_submodules
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.import_utils import require_package

from .configuration_imf_attnres import IMFAttnResConfig
from .imf_transformer1d import IMFTransformer1D

try:
    from torch.func import jvp as torch_func_jvp
except ImportError:  # pragma: no cover
    torch_func_jvp = None


class IMFAttnResRgbEncoder(nn.Module):
    """RGB encoder compatible with DiffusionRgbEncoder but safe for tiny smoke-test images.

    DiffusionRgbEncoder performs a shape dry-run while the ResNet is in training mode.
    For 16x16 synthetic images this can create 1x1 BatchNorm activations and fail.
    This local encoder mirrors the same runtime path but evaluates the dry-run in eval
    mode, then restores the previous mode.
    """

    def __init__(self, config: IMFAttnResConfig):
        super().__init__()
        self.resize = (
            torchvision.transforms.Resize(config.resize_shape)
            if config.resize_shape is not None
            else None
        )

        crop_shape = config.crop_shape
        if crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(crop_shape)
                if config.crop_is_random
                else self.center_crop
            )
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("You can't replace BatchNorm in a pretrained model without ruining the weights!")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=max(1, x.num_features // 16), num_channels=x.num_features),
            )

        images_shape = next(iter(config.image_features.values())).shape
        if config.crop_shape is not None:
            dummy_shape_h_w = config.crop_shape
        elif config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        self.backbone.train(was_training)

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.resize is not None:
            x = self.resize(x)
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        return self.relu(self.out(x))


def _resolve_vlm_torch_dtype(dtype_name: str | None) -> torch.dtype | str | None:
    """Map config-friendly dtype names to values accepted by transformers."""
    if dtype_name is None or dtype_name == "auto":
        return dtype_name
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unsupported torch dtype for SmolVLM encoder: {dtype_name!r}.")
    return dtype


def _get_smolvlm_core_model(vlm: nn.Module) -> nn.Module:
    """Return the inner SmolVLM model containing vision/text/connector modules."""
    if hasattr(vlm, "model"):
        return vlm.model
    if all(hasattr(vlm, attr) for attr in ("vision_model", "connector", "text_model")):
        return vlm
    raise AttributeError("Could not locate SmolVLM core model with vision_model, connector, and text_model.")


class IMFAttnResSmolVLMVLEncoder(nn.Module):
    """SmolVLM visual-language prefix encoder for IMF-AttnRes conditioning.

    The module reuses SmolVLM's vision encoder, connector, and language token
    embeddings, then appends one trainable projected robot-state token. It does
    not instantiate or use SmolVLA's action expert.
    """

    def __init__(self, config: IMFAttnResConfig):
        super().__init__()
        require_package("transformers", extra="smolvla")
        from transformers import AutoConfig, AutoModelForImageTextToText, SmolVLMForConditionalGeneration

        self.model_name = getattr(config, "vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        self.freeze_vlm_encoder = getattr(config, "freeze_vlm_encoder", True)
        self.load_vlm_weights = getattr(config, "load_vlm_weights", True)
        self.tokenizer_max_length = int(getattr(config, "vlm_tokenizer_max_length", 48))
        self.padding_side = getattr(config, "vlm_tokenizer_padding_side", "right")
        self.truncate_language = getattr(config, "vlm_tokenizer_truncation", True)
        self.vlm_resize_shape = getattr(config, "vlm_resize_shape", (512, 512))
        self.image_forward_batch_size = int(getattr(config, "vlm_image_forward_batch_size", 0))
        self.text_encoder_mode = getattr(config, "vlm_text_encoder_mode", "embedding")
        self.text_num_layers = int(getattr(config, "vlm_text_num_layers", 16))
        self.state_in_text_layers = bool(getattr(config, "vlm_state_in_text_layers", True))
        self.num_images = len(getattr(config, "image_features", {}))

        if self.load_vlm_weights:
            from_pretrained_kwargs = {"low_cpu_mem_usage": True}
            resolved_dtype = _resolve_vlm_torch_dtype(getattr(config, "vlm_encoder_torch_dtype", "bfloat16"))
            if resolved_dtype is not None:
                from_pretrained_kwargs["torch_dtype"] = resolved_dtype
            self.vlm = AutoModelForImageTextToText.from_pretrained(self.model_name, **from_pretrained_kwargs)
        else:
            hf_config = AutoConfig.from_pretrained(self.model_name)
            self.vlm = SmolVLMForConditionalGeneration(config=hf_config)

        self.vlm_model = _get_smolvlm_core_model(self.vlm)
        self.vision_model = self.vlm_model.vision_model
        self.connector = self.vlm_model.connector
        self.text_model = self.vlm_model.text_model
        self.text_embeddings = self.text_model.get_input_embeddings()
        self._trim_unused_text_modules()

        text_config = getattr(getattr(self.vlm, "config", None), "text_config", None)
        self.feature_dim = getattr(text_config, "hidden_size", None)
        if self.feature_dim is None:
            self.feature_dim = getattr(getattr(self.vlm_model.text_model, "config", None), "hidden_size", None)
        if self.feature_dim is None and hasattr(self.text_embeddings, "embedding_dim"):
            self.feature_dim = self.text_embeddings.embedding_dim
        if self.feature_dim is None:
            fallback_hidden_size = getattr(config, "vlm_hidden_size", None)
            if fallback_hidden_size is None:
                raise ValueError("Could not infer SmolVLM text hidden size for IMF-AttnRes conditioning.")
            self.feature_dim = int(fallback_hidden_size)
        self.feature_dim = int(self.feature_dim)

        self.pad_token_id = getattr(text_config, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(getattr(self.vlm, "config", None), "pad_token_id", 0)
        if self.pad_token_id is None:
            self.pad_token_id = 0

        self.tokens_per_image = self._infer_tokens_per_image()
        self.tokens_per_step = self.num_images * self.tokens_per_image + self.tokenizer_max_length + 1

        state_dim = config.robot_state_feature.shape[0]
        env_dim = config.env_state_feature.shape[0] if config.env_state_feature else 0
        self.state_env_dim = state_dim + env_dim
        self.state_projection = nn.Linear(self.state_env_dim, self.feature_dim)

        if self.freeze_vlm_encoder:
            for param in self.vlm.parameters():
                param.requires_grad = False
            self.vlm.eval()

    def _trim_unused_text_modules(self) -> None:
        """Keep only the SmolVLM text modules needed by the configured text path."""
        if hasattr(self.vlm, "lm_head"):
            self.vlm.lm_head = nn.Identity()

        layers = getattr(self.text_model, "layers", None)
        if layers is None:
            if self.text_encoder_mode == "transformer":
                raise ValueError("SmolVLM text transformer mode requires text_model.layers.")
            return

        if self.text_encoder_mode == "embedding":
            self.text_model.layers = nn.ModuleList()
            return

        if self.text_encoder_mode != "transformer":
            raise ValueError(
                "vlm_text_encoder_mode must be one of {'embedding', 'transformer'}, got "
                f"{self.text_encoder_mode!r}."
            )
        if self.text_num_layers < 1:
            raise ValueError(
                "vlm_text_num_layers must be >= 1 when vlm_text_encoder_mode='transformer'. "
                f"Got {self.text_num_layers}."
            )
        if self.text_num_layers > len(layers):
            raise ValueError(
                "vlm_text_num_layers cannot exceed the loaded SmolVLM text layer count. "
                f"Got {self.text_num_layers}, but model has {len(layers)} layers."
            )
        self.text_model.layers = layers[: self.text_num_layers]
        text_model_config = getattr(self.text_model, "config", None)
        if text_model_config is not None and hasattr(text_model_config, "num_hidden_layers"):
            text_model_config.num_hidden_layers = self.text_num_layers

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vlm_encoder:
            self.vlm.eval()
        return self

    def _infer_tokens_per_image(self) -> int:
        vlm_config = getattr(self.vlm, "config", None)
        vision_config = getattr(vlm_config, "vision_config", None)
        patch_size = getattr(vision_config, "patch_size", None)
        scale_factor = getattr(vlm_config, "scale_factor", None)
        if self.vlm_resize_shape is not None and patch_size is not None and scale_factor is not None:
            resize_width, resize_height = self.vlm_resize_shape
            return int(((resize_width // patch_size) * (resize_height // patch_size)) / (scale_factor**2))

        image_seq_len = getattr(self.vlm_model, "image_seq_len", None)
        if image_seq_len is not None:
            return int(image_seq_len)

        image_size = getattr(vision_config, "image_size", None)
        if image_size is not None and patch_size is not None and scale_factor is not None:
            return int(((image_size // patch_size) ** 2) / (scale_factor**2))

        resize_width, resize_height = self.vlm_resize_shape
        patch_size = 16 if patch_size is None else patch_size
        scale_factor = 4 if scale_factor is None else scale_factor
        return int(((resize_width // patch_size) * (resize_height // patch_size)) / (scale_factor**2))

    def _pad_or_truncate_language(self, tokens: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        sequence_length = tokens.shape[1]
        if sequence_length > self.tokenizer_max_length:
            if not self.truncate_language:
                raise ValueError(
                    "SmolVLM language tokens are longer than vlm_tokenizer_max_length and truncation is disabled. "
                    f"Got {sequence_length=} and {self.tokenizer_max_length=}."
                )
            tokens = tokens[:, : self.tokenizer_max_length]
            masks = masks[:, : self.tokenizer_max_length]
        elif sequence_length < self.tokenizer_max_length:
            pad_length = self.tokenizer_max_length - sequence_length
            token_padding = torch.full(
                (tokens.shape[0], pad_length),
                int(self.pad_token_id),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            mask_padding = torch.zeros(
                (masks.shape[0], pad_length),
                dtype=masks.dtype,
                device=masks.device,
            )
            if self.padding_side == "left":
                tokens = torch.cat([token_padding, tokens], dim=1)
                masks = torch.cat([mask_padding, masks], dim=1)
            elif self.padding_side == "right":
                tokens = torch.cat([tokens, token_padding], dim=1)
                masks = torch.cat([masks, mask_padding], dim=1)
            else:
                raise ValueError(f"Unsupported SmolVLM tokenizer padding side: {self.padding_side!r}.")
        return tokens, masks

    def _preprocess_images(self, images: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError(f"SmolVLM images must have shape (B, N, C, H, W). Got {tuple(images.shape)}.")
        flat_images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        if self.vlm_resize_shape is not None:
            from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

            flat_images = resize_with_pad(flat_images, *self.vlm_resize_shape, pad_value=0)
        return flat_images * 2.0 - 1.0

    def _embed_images(self, images: Tensor) -> Tensor:
        flat_batch = images.shape[0]
        if self.num_images == 0:
            return images.new_zeros((flat_batch, 0, self.feature_dim))

        pixel_values = self._preprocess_images(images)
        vision_dtype = getattr(self.vision_model, "dtype", pixel_values.dtype)
        pixel_values = pixel_values.to(dtype=vision_dtype)
        image_forward_batch_size = self.image_forward_batch_size
        if image_forward_batch_size > 0 and pixel_values.shape[0] > image_forward_batch_size:
            image_hidden_states = torch.cat(
                [
                    self.vision_model(
                        pixel_values=pixel_values_chunk,
                        patch_attention_mask=None,
                    ).last_hidden_state
                    for pixel_values_chunk in pixel_values.split(image_forward_batch_size, dim=0)
                ],
                dim=0,
            )
        else:
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=None,
            ).last_hidden_state
        image_hidden_states = self.connector(image_hidden_states)
        return einops.rearrange(
            image_hidden_states,
            "(b n) t d -> b (n t) d",
            b=flat_batch,
            n=self.num_images,
        )

    def _embed_language_tokens(self, tokens: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        tokens, masks = self._pad_or_truncate_language(tokens, masks)
        lang_embeddings = self.text_embeddings(tokens)
        lang_embeddings = lang_embeddings * masks.to(
            device=lang_embeddings.device,
            dtype=lang_embeddings.dtype,
        ).unsqueeze(-1)
        return lang_embeddings, masks

    @staticmethod
    def _scale_vlm_token_embeddings(tokens: Tensor) -> Tensor:
        return tokens * math.sqrt(tokens.shape[-1])

    def _encode_vlm_prefix(
        self,
        image_tokens: Tensor,
        language_tokens: Tensor,
        language_masks: Tensor,
        state_token: Tensor | None = None,
    ) -> Tensor:
        if self.text_encoder_mode != "transformer":
            return torch.cat([image_tokens, language_tokens], dim=1)

        batch_size, image_token_count = image_tokens.shape[:2]
        image_masks = torch.ones(
            batch_size,
            image_token_count,
            dtype=torch.bool,
            device=image_tokens.device,
        )
        token_parts = [
            self._scale_vlm_token_embeddings(image_tokens),
            self._scale_vlm_token_embeddings(language_tokens),
        ]
        mask_parts = [image_masks, language_masks.to(device=image_tokens.device)]
        if state_token is not None:
            token_parts.append(state_token.to(device=image_tokens.device, dtype=image_tokens.dtype))
            mask_parts.append(
                torch.ones(
                    state_token.shape[:2],
                    dtype=torch.bool,
                    device=image_tokens.device,
                )
            )
        prefix_tokens = torch.cat(token_parts, dim=1)
        prefix_masks = torch.cat(mask_parts, dim=1)
        position_ids = torch.arange(
            prefix_tokens.shape[1],
            device=prefix_tokens.device,
            dtype=torch.long,
        ).unsqueeze(0)
        text_outputs = self.text_model(
            inputs_embeds=prefix_tokens,
            attention_mask=prefix_masks,
            position_ids=position_ids,
            use_cache=False,
        )
        prefix_tokens = (
            text_outputs.last_hidden_state
            if hasattr(text_outputs, "last_hidden_state")
            else text_outputs[0]
        )
        return prefix_tokens * prefix_masks.to(device=prefix_tokens.device, dtype=prefix_tokens.dtype).unsqueeze(-1)

    def forward(
        self,
        images: Tensor,
        state: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        env_state: Tensor | None = None,
    ) -> Tensor:
        device = self.state_projection.weight.device
        images = images.to(device=device)
        state = state.to(device=device, dtype=self.state_projection.weight.dtype)
        lang_tokens = lang_tokens.to(device=device, dtype=torch.long)
        lang_masks = lang_masks.to(device=device, dtype=torch.bool)
        if env_state is not None:
            env_state = env_state.to(device=device, dtype=self.state_projection.weight.dtype)

        vlm_context = torch.no_grad() if self.freeze_vlm_encoder else nullcontext()
        with vlm_context:
            image_tokens = self._embed_images(images)
            language_tokens, language_masks = self._embed_language_tokens(lang_tokens, lang_masks)

        state_parts = [state]
        if self.state_env_dim > state.shape[-1]:
            if env_state is None:
                raise ValueError("SmolVLM state projection expects env_state, but none was provided.")
            state_parts.append(env_state)
        state_token_input = torch.cat(state_parts, dim=-1)
        state_token = self.state_projection(state_token_input).unsqueeze(1)
        include_state_in_text_layers = self.text_encoder_mode == "transformer" and self.state_in_text_layers
        vlm_prefix_tokens = self._encode_vlm_prefix(
            image_tokens,
            language_tokens,
            language_masks,
            state_token=state_token if include_state_in_text_layers else None,
        )
        if include_state_in_text_layers:
            prefix_tokens = vlm_prefix_tokens.to(dtype=state_token.dtype)
        else:
            prefix_tokens = torch.cat(
                [
                    vlm_prefix_tokens.to(dtype=state_token.dtype),
                    state_token,
                ],
                dim=1,
            )
        if prefix_tokens.shape[1] != self.tokens_per_step:
            raise ValueError(
                "SmolVLM VL encoder produced an unexpected number of condition tokens. "
                f"Got {prefix_tokens.shape[1]}, expected {self.tokens_per_step}."
            )
        return prefix_tokens


class IMFAttnResPolicy(PreTrainedPolicy):
    """LeRobot-native IMF-AttnRes policy."""

    config_class = IMFAttnResConfig
    name = "imf-attnres"

    def __init__(self, config: IMFAttnResConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.model = IMFAttnResModel(config)
        if config.compile_model:
            self.model.head = torch.compile(self.model.head, mode=config.compile_mode)
        self.reset()

    def get_optim_params(self):
        if self.model.semigroup_teacher_head is None:
            return self.model.parameters()
        return (parameter for parameter in self.model.parameters() if parameter.requires_grad)

    @torch.no_grad()
    def update(self):
        self.model.update_semigroup_teacher()

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)
        if getattr(self.config, "use_smolvlm_vl_encoder", False):
            self._queues[OBS_LANGUAGE_TOKENS] = deque(maxlen=self.config.n_obs_steps)
            self._queues[OBS_LANGUAGE_ATTENTION_MASK] = deque(maxlen=self.config.n_obs_steps)

    def _stack_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.config.image_features:
            return batch
        out = dict(batch)
        for key in self.config.image_features:
            if self.config.n_obs_steps == 1 and out[key].ndim == 4:
                out[key] = out[key].unsqueeze(1)
        out[OBS_IMAGES] = torch.stack([out[key] for key in self.config.image_features], dim=-4)
        return out

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        # If called after select_action has populated queues, use queued observation history.
        if self._queues is not None and len(self._queues[OBS_STATE]) > 0:
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            return self.model.generate_actions(batch, noise=noise)

        # Direct batch-mode use: accept current-step observations and repeat to fill history.
        batch = dict(batch)
        if self.config.image_features:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        if batch[OBS_STATE].ndim == 2:
            batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1).expand(-1, self.config.n_obs_steps, -1)
        if OBS_IMAGES in batch and batch[OBS_IMAGES].ndim == 5:
            batch[OBS_IMAGES] = batch[OBS_IMAGES].unsqueeze(1).expand(-1, self.config.n_obs_steps, -1, -1, -1, -1)
        if self.config.env_state_feature and batch[OBS_ENV_STATE].ndim == 2:
            batch[OBS_ENV_STATE] = batch[OBS_ENV_STATE].unsqueeze(1).expand(-1, self.config.n_obs_steps, -1)
        return self.model.generate_actions(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor], current_step: int | None = None) -> tuple[Tensor, None]:
        batch = self._stack_images(batch)
        loss, diagnostics = self.model.compute_loss(batch, current_step=current_step)
        return loss, diagnostics or None


class IMFAttnResModel(nn.Module):
    def __init__(self, config: IMFAttnResConfig):
        super().__init__()
        self.config = config

        state_dim = config.robot_state_feature.shape[0]
        cond_dim_per_step = state_dim
        self.rgb_encoder = None
        self.vl_encoder = None
        self.condition_tokens_per_step = 1

        if getattr(config, "use_smolvlm_vl_encoder", False):
            self.vl_encoder = IMFAttnResSmolVLMVLEncoder(config)
            self.cond_dim = self.vl_encoder.feature_dim
            self.condition_tokens_per_step = self.vl_encoder.tokens_per_step
            cond_dim_per_step = self.cond_dim
        elif config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [IMFAttnResRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                image_cond_dim = encoders[0].feature_dim
            else:
                self.rgb_encoder = IMFAttnResRgbEncoder(config)
                image_cond_dim = self.rgb_encoder.feature_dim

            if config.output_tokens_per_camera:
                self.condition_tokens_per_step = num_images
                cond_dim_per_step += image_cond_dim
            else:
                cond_dim_per_step += image_cond_dim * num_images

        if not getattr(config, "use_smolvlm_vl_encoder", False) and config.env_state_feature:
            cond_dim_per_step += config.env_state_feature.shape[0]

        self.cond_dim = cond_dim_per_step
        self.head = IMFTransformer1D(
            input_dim=config.action_feature.shape[0],
            output_dim=config.action_feature.shape[0],
            horizon=config.horizon,
            n_obs_steps=config.n_obs_steps * self.condition_tokens_per_step,
            cond_dim=self.cond_dim,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_emb=config.n_emb,
            p_drop_emb=config.p_drop_emb,
            p_drop_attn=config.p_drop_attn,
            causal_attn=config.causal_attn,
            time_as_cond=config.time_as_cond,
            obs_as_cond=config.obs_as_cond,
            n_cond_layers=config.n_cond_layers,
            backbone_type=config.backbone_type,
            n_kv_head=config.n_kv_head,
            attn_res_ffn_mult=config.attn_res_ffn_mult,
            attn_res_eps=config.attn_res_eps,
            attn_res_rope_theta=config.attn_res_rope_theta,
        )
        self.semigroup_teacher_head = None
        if config.enable_semigroup_consistency:
            self._reset_semigroup_teacher()

    @staticmethod
    def _raw_module(module: nn.Module) -> nn.Module:
        return getattr(module, "_orig_mod", module)

    def _reset_semigroup_teacher(self) -> None:
        if self.semigroup_teacher_head is None:
            self.semigroup_teacher_head = deepcopy(self._raw_module(self.head))
        self.semigroup_teacher_head.load_state_dict(self._raw_module(self.head).state_dict())
        for parameter in self.semigroup_teacher_head.parameters():
            parameter.requires_grad_(False)
        self.semigroup_teacher_head.eval()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        teacher_prefix = "semigroup_teacher_head."
        has_teacher_state = any(key.startswith(teacher_prefix) for key in state_dict)
        if has_teacher_state and self.semigroup_teacher_head is None:
            self._reset_semigroup_teacher()
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.enable_semigroup_consistency and not has_teacher_state:
            self._reset_semigroup_teacher()
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        if self.semigroup_teacher_head is not None:
            self.semigroup_teacher_head.eval()
        return self

    @torch.no_grad()
    def update_semigroup_teacher(self) -> None:
        if not self.config.enable_semigroup_consistency:
            return
        decay = float(self.config.semigroup_teacher_ema_decay)
        online_head = self._raw_module(self.head)
        teacher_head = self._raw_module(self.semigroup_teacher_head)

        for teacher_parameter, online_parameter in zip(
            teacher_head.parameters(),
            online_head.parameters(),
            strict=True,
        ):
            teacher_parameter.mul_(decay).add_(
                online_parameter.detach().to(device=teacher_parameter.device, dtype=teacher_parameter.dtype),
                alpha=1.0 - decay,
            )
        for teacher_buffer, online_buffer in zip(teacher_head.buffers(), online_head.buffers(), strict=True):
            teacher_buffer.copy_(
                online_buffer.detach().to(device=teacher_buffer.device, dtype=teacher_buffer.dtype)
            )
        self.semigroup_teacher_head.eval()

    @staticmethod
    def _broadcast_batch_time(value: Tensor, reference: Tensor) -> Tensor:
        while value.ndim < reference.ndim:
            value = value.unsqueeze(-1)
        return value

    @staticmethod
    def _apply_conditioning(
        trajectory: Tensor,
        condition_data: Tensor | None = None,
        condition_mask: Tensor | None = None,
    ) -> Tensor:
        if condition_data is None or condition_mask is None:
            return trajectory
        conditioned = trajectory.clone()
        conditioned[condition_mask] = condition_data[condition_mask]
        return conditioned

    @staticmethod
    def _jvp_math_sdp_context(z_t: Tensor):
        if z_t.is_cuda:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
                enable_cudnn=False,
            )
        return nullcontext()

    @staticmethod
    def _jvp_tangents(v: Tensor, r: Tensor, t: Tensor):
        return v.detach(), torch.zeros_like(r), torch.ones_like(t)

    def _sample_logit_normal(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        normal = torch.randn(batch_size, device=device, dtype=dtype)
        return torch.sigmoid(normal * self.config.p_std + self.config.p_mean)

    def _sample_tr(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        t = self._sample_logit_normal(batch_size, device, dtype)
        r = self._sample_logit_normal(batch_size, device, dtype)

        data_size = int(batch_size * self.config.data_proportion)
        fm_mask = torch.arange(batch_size, device=device) < data_size
        r = torch.where(fm_mask, t, r)

        t_final = torch.maximum(t, r)
        r_final = torch.minimum(t, r)
        return t_final, r_final

    def _sample_semigroup_times(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        min_delta = float(self.config.semigroup_min_time_delta)
        base = torch.rand(batch_size, 3, device=device, dtype=dtype)
        weights = torch.nn.functional.softmax(base, dim=-1)
        remaining = 1.0 - 3.0 * min_delta
        segment_lengths = min_delta + remaining * weights
        r = segment_lengths[:, 0] - min_delta
        s = r + segment_lengths[:, 1]
        t = s + segment_lengths[:, 2]
        return r, s, t

    def fn(self, z: Tensor, r: Tensor, t: Tensor, cond: Tensor | None = None) -> Tensor:
        return self.head(z, r, t, cond=cond)

    def _semigroup_teacher_fn(
        self,
        z: Tensor,
        r: Tensor,
        t: Tensor,
        cond: Tensor | None = None,
    ) -> Tensor:
        if self.semigroup_teacher_head is None:
            raise RuntimeError("Semigroup teacher head is not initialized.")
        return self.semigroup_teacher_head(z, r, t, cond=cond)

    def _compute_u_and_du_dt(
        self,
        z_t: Tensor,
        r: Tensor,
        t: Tensor,
        cond: Tensor,
        v: Tensor,
        condition_data: Tensor | None = None,
        condition_mask: Tensor | None = None,
    ):
        tangents = self._jvp_tangents(v, r, t)

        def g(z, r_value, t_value):
            conditioned_z = self._apply_conditioning(z, condition_data, condition_mask)
            return self.fn(conditioned_z, r_value, t_value, cond=cond)

        with self._jvp_math_sdp_context(z_t):
            if torch_func_jvp is not None:
                try:
                    return torch_func_jvp(g, (z_t, r, t), tangents)
                except (RuntimeError, TypeError, NotImplementedError):
                    pass
            return torch.autograd.functional.jvp(
                g,
                (z_t, r, t),
                tangents,
                create_graph=True,
                strict=False,
            )

    def _compound_velocity(self, u: Tensor, du_dt: Tensor, r: Tensor, t: Tensor) -> Tensor:
        delta = self._broadcast_batch_time(t - r, u)
        return u + delta * du_dt.detach()

    def _velocity_loss_from_error(self, error: Tensor) -> Tensor:
        if self.config.loss_type == "mse":
            return error.square()
        if self.config.loss_type != "pseudo_huber":
            raise ValueError(f"Unsupported IMF-AttnRes loss_type: {self.config.loss_type!r}.")
        delta = float(self.config.pseudo_huber_delta)
        return delta**2 * (torch.sqrt(1 + (error / delta).square()) - 1)

    def _flow_map(
        self,
        z_t: Tensor,
        r: Tensor,
        t: Tensor,
        cond: Tensor,
        *,
        fn=None,
    ) -> Tensor:
        if fn is None:
            fn = self.fn
        u = fn(z_t, r, t, cond=cond)
        delta = self._broadcast_batch_time(t - r, z_t)
        return z_t - delta * u

    def _semigroup_loss_weight(self, current_step: int | None) -> float:
        if not self.config.enable_semigroup_consistency:
            return 0.0
        target_weight = float(self.config.semigroup_loss_weight)
        if target_weight <= 0.0:
            return 0.0
        if current_step is None:
            return 0.0
        start_step = int(self.config.semigroup_start_step)
        if current_step <= start_step:
            return 0.0
        warmup_steps = int(self.config.semigroup_warmup_steps)
        if warmup_steps <= 0:
            return target_weight
        progress = min(max((current_step - start_step) / warmup_steps, 0.0), 1.0)
        return target_weight * progress

    def _semigroup_consistency_loss(
        self,
        x_t: Tensor,
        cond: Tensor,
        r: Tensor | None = None,
        s: Tensor | None = None,
        t: Tensor | None = None,
    ) -> Tensor:
        if r is None or s is None or t is None:
            r, s, t = self._sample_semigroup_times(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
        if self.semigroup_teacher_head is None:
            self._reset_semigroup_teacher()
        semigroup_cond = cond.detach()
        direct = self._flow_map(x_t, r=r, t=t, cond=semigroup_cond)
        with torch.no_grad():
            teacher_x_t = x_t.detach()
            teacher_cond = semigroup_cond.detach()
            mid = self._flow_map(
                teacher_x_t,
                r=s,
                t=t,
                cond=teacher_cond,
                fn=self._semigroup_teacher_fn,
            )
            composed = self._flow_map(
                mid.detach(),
                r=r,
                t=s,
                cond=teacher_cond,
                fn=self._semigroup_teacher_fn,
            )
        return self._velocity_loss_from_error(direct - composed.detach()).mean()

    @staticmethod
    def _dct_matrix(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        positions = torch.arange(size, device=device, dtype=dtype)
        freqs = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
        basis = torch.cos(torch.pi / size * (positions + 0.5) * freqs)
        basis[0] *= (1.0 / size) ** 0.5
        if size > 1:
            basis[1:] *= (2.0 / size) ** 0.5
        return basis

    def _encode_action_latent(self, actions: Tensor) -> Tensor:
        if self.config.action_latent_mode == "identity":
            return actions
        if self.config.action_latent_mode != "dct":
            raise ValueError(f"Unsupported action_latent_mode: {self.config.action_latent_mode!r}.")
        dct = self._dct_matrix(actions.shape[1], actions.device, actions.dtype)
        return torch.einsum("kh,bhd->bkd", dct, actions)

    def _decode_action_latent(self, latents: Tensor) -> Tensor:
        if self.config.action_latent_mode == "identity":
            return latents
        if self.config.action_latent_mode != "dct":
            raise ValueError(f"Unsupported action_latent_mode: {self.config.action_latent_mode!r}.")
        dct = self._dct_matrix(latents.shape[1], latents.device, latents.dtype)
        return torch.einsum("kh,bkd->bhd", dct, latents)

    def _action_latent_loss_weights(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.config.action_latent_mode != "dct":
            return torch.ones(1, self.config.horizon, 1, device=device, dtype=dtype)
        if self.config.horizon == 1:
            normalized_freq = torch.zeros(1, device=device, dtype=dtype)
        else:
            normalized_freq = torch.linspace(0.0, 1.0, self.config.horizon, device=device, dtype=dtype)
        weights = 1.0 + float(self.config.dct_loss_high_freq_weight) * normalized_freq.pow(
            float(self.config.dct_loss_freq_power)
        )
        return weights.view(1, self.config.horizon, 1)

    @staticmethod
    def _vector_norm_per_sample(value: Tensor) -> Tensor:
        return value.detach().float().flatten(start_dim=1).norm(dim=1)

    @staticmethod
    def _scalar_per_sample(value: Tensor) -> Tensor:
        return value.detach().float().reshape(value.shape[0], -1).mean(dim=1)

    @staticmethod
    def _safe_std(value: Tensor) -> Tensor:
        if value.numel() <= 1:
            return torch.zeros((), device=value.device, dtype=value.dtype)
        return value.std(unbiased=False)

    @classmethod
    def _add_bucket_stats(
        cls,
        diagnostics: dict[str, float],
        prefix: str,
        name: str,
        values: Tensor,
    ) -> None:
        values = values.detach().float()
        diagnostics[f"{prefix}/{name}_count"] = float(values.numel())
        if values.numel() == 0:
            return
        diagnostics[f"{prefix}/{name}_mean"] = float(values.mean().item())
        diagnostics[f"{prefix}/{name}_std"] = float(cls._safe_std(values).item())
        diagnostics[f"{prefix}/{name}_max"] = float(values.max().item())

    def _attnres_depth_attention_diagnostics(self) -> dict[str, float]:
        weights = []
        backbone = getattr(self.head, "attnres_backbone", None)
        if backbone is None:
            return {}
        for layer in backbone.layers:
            layer_weights = getattr(layer.attn_res, "last_depth_attention_weights", None)
            if layer_weights is not None:
                weights.append(layer_weights.float())
        if not weights:
            return {}

        entropy_values = []
        max_weight_values = []
        for layer_weights in weights:
            safe_weights = layer_weights.clamp_min(torch.finfo(torch.float32).tiny)
            entropy_values.append((-(safe_weights * safe_weights.log()).sum(dim=0)).flatten())
            max_weight_values.append(layer_weights.max(dim=0).values.flatten())
        entropy = torch.cat(entropy_values)
        max_weight = torch.cat(max_weight_values)
        diagnostics: dict[str, float] = {}
        self._add_bucket_stats(diagnostics, "imf_diagnostics/attnres", "depth_attention_entropy", entropy)
        self._add_bucket_stats(diagnostics, "imf_diagnostics/attnres", "depth_attention_max_weight", max_weight)
        return diagnostics

    def _imf_training_diagnostics(
        self,
        *,
        loss: Tensor,
        target: Tensor,
        u: Tensor,
        du_dt: Tensor,
        delta_du_dt: Tensor,
        t: Tensor,
        r: Tensor,
    ) -> dict[str, float]:
        spike_loss_threshold = float(self.config.imf_diagnostics_spike_loss_threshold)
        is_spike = bool(loss.detach().float().item() > spike_loss_threshold)
        spike_mask = torch.full((target.shape[0],), is_spike, device=target.device, dtype=torch.bool)
        non_spike_mask = ~spike_mask

        values = {
            "target_norm": self._vector_norm_per_sample(target),
            "u_norm": self._vector_norm_per_sample(u),
            "du_dt_norm": self._vector_norm_per_sample(du_dt),
            "delta_du_dt_norm": self._vector_norm_per_sample(delta_du_dt),
            "delta": self._scalar_per_sample(t - r),
            "t": self._scalar_per_sample(t),
            "r": self._scalar_per_sample(r),
        }

        diagnostics: dict[str, float] = {
            "imf_diagnostics/spike_threshold": spike_loss_threshold,
            "imf_diagnostics/spike/is_spike": float(is_spike),
            "imf_diagnostics/loss": float(loss.detach().float().item()),
        }
        for name, tensor in values.items():
            self._add_bucket_stats(diagnostics, "imf_diagnostics/all", name, tensor)
            self._add_bucket_stats(diagnostics, "imf_diagnostics/spike", name, tensor[spike_mask])
            self._add_bucket_stats(diagnostics, "imf_diagnostics/non_spike", name, tensor[non_spike_mask])
        diagnostics.update(self._attnres_depth_attention_diagnostics())
        return diagnostics

    def _sample_one_step(
        self,
        z_t: Tensor,
        r: Tensor | None = None,
        t: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> Tensor:
        batch_size = z_t.shape[0]
        if t is None:
            t = torch.ones(batch_size, device=z_t.device, dtype=z_t.dtype)
        if r is None:
            r = torch.zeros(batch_size, device=z_t.device, dtype=z_t.dtype)
        return self._flow_map(z_t, r=r, t=t, cond=cond)

    def _encode_images(self, batch: dict[str, Tensor]) -> Tensor | None:
        if not self.config.image_features:
            return None
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        if self.config.use_separate_rgb_encoder_per_camera:
            images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
            encoded = [encoder(images) for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)]
            if self.config.output_tokens_per_camera:
                img_features = torch.stack(encoded, dim=1)
                return einops.rearrange(img_features, "(b s) n d -> b s n d", b=batch_size, s=n_obs_steps)
            img_features_list = torch.cat(encoded)
            return einops.rearrange(
                img_features_list,
                "(n b s) ... -> b s (n ...)",
                b=batch_size,
                s=n_obs_steps,
            )

        img_features = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
        if self.config.output_tokens_per_camera:
            return einops.rearrange(img_features, "(b s n) d -> b s n d", b=batch_size, s=n_obs_steps)
        return einops.rearrange(img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps)

    def _prepare_smolvlm_language(
        self,
        batch: dict[str, Tensor],
        batch_size: int,
        n_obs_steps: int,
    ) -> tuple[Tensor, Tensor]:
        missing_keys = {
            key
            for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)
            if key not in batch or batch[key] is None
        }
        if missing_keys:
            raise ValueError(
                "SmolVLM VL encoder requires language tokens and attention mask. "
                f"Missing tokenized task inputs: {sorted(missing_keys)}."
            )

        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        if lang_tokens.shape != lang_masks.shape:
            raise ValueError(
                "SmolVLM language tokens and attention mask must have matching shapes. "
                f"Got tokens {tuple(lang_tokens.shape)} and mask {tuple(lang_masks.shape)}."
            )

        def flatten_or_repeat(value: Tensor, name: str) -> Tensor:
            if value.ndim == 2:
                if value.shape[0] != batch_size:
                    raise ValueError(
                        f"{name} batch dimension must match observation batch size. "
                        f"Got {value.shape[0]=}, {batch_size=}."
                    )
                return value.repeat_interleave(n_obs_steps, dim=0)
            if value.ndim == 3:
                if value.shape[0] != batch_size or value.shape[1] != n_obs_steps:
                    raise ValueError(
                        f"{name} batch/time dimensions must match observations. "
                        f"Got {tuple(value.shape[:2])}, expected ({batch_size}, {n_obs_steps})."
                    )
                return value.reshape(batch_size * n_obs_steps, value.shape[-1])
            raise ValueError(
                f"{name} must have shape (B, L) or (B, n_obs_steps, L). Got {tuple(value.shape)}."
            )

        return (
            flatten_or_repeat(lang_tokens, "language tokens"),
            flatten_or_repeat(lang_masks, "language attention mask"),
        )

    def _prepare_smolvlm_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        if self.vl_encoder is None:
            raise RuntimeError("SmolVLM VL encoder path requested, but no VL encoder was created.")
        if OBS_IMAGES not in batch:
            raise ValueError("SmolVLM VL encoder requires stacked observation images in the batch.")

        state = batch[OBS_STATE]
        batch_size, n_obs_steps = state.shape[:2]
        images = batch[OBS_IMAGES]
        if images.ndim != 6:
            raise ValueError(
                f"`{OBS_IMAGES}` must have shape (B, n_obs_steps, N, C, H, W). Got {tuple(images.shape)}."
            )
        if images.shape[0] != batch_size or images.shape[1] != n_obs_steps:
            raise ValueError(
                "SmolVLM image batch/time dimensions must match observation state. "
                f"Got images {tuple(images.shape[:2])}, state {tuple(state.shape[:2])}."
            )

        flat_images = einops.rearrange(images, "b s n c h w -> (b s) n c h w")
        flat_state = einops.rearrange(state, "b s d -> (b s) d")
        env_state = None
        if self.config.env_state_feature:
            if OBS_ENV_STATE not in batch:
                raise ValueError("SmolVLM VL encoder expects env_state because config.env_state_feature is set.")
            env = batch[OBS_ENV_STATE]
            if env.shape[:2] != (batch_size, n_obs_steps):
                raise ValueError(
                    "SmolVLM env_state batch/time dimensions must match observation state. "
                    f"Got env_state {tuple(env.shape[:2])}, state {tuple(state.shape[:2])}."
                )
            env_state = einops.rearrange(env, "b s d -> (b s) d")

        lang_tokens, lang_masks = self._prepare_smolvlm_language(batch, batch_size, n_obs_steps)
        step_tokens = self.vl_encoder(flat_images, flat_state, lang_tokens, lang_masks, env_state=env_state)
        cond = einops.rearrange(
            step_tokens,
            "(b s) t d -> b (s t) d",
            b=batch_size,
            s=n_obs_steps,
        )
        head_dtype = get_dtype_from_parameters(self.head)
        return cond.to(dtype=head_dtype)

    def _prepare_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        if getattr(self.config, "use_smolvlm_vl_encoder", False):
            return self._prepare_smolvlm_conditioning(batch)

        state = batch[OBS_STATE]
        cond_parts = [state]
        image_features = self._encode_images(batch)
        if image_features is not None:
            if image_features.ndim == 4:
                token_count = image_features.shape[2]
                state_tokens = state.unsqueeze(2).expand(-1, -1, token_count, -1)
                token_parts = [image_features, state_tokens]
                if self.config.env_state_feature:
                    env_tokens = batch[OBS_ENV_STATE].unsqueeze(2).expand(-1, -1, token_count, -1)
                    token_parts.append(env_tokens)
                cond = torch.cat(token_parts, dim=-1)
                return cond.reshape(cond.shape[0], cond.shape[1] * cond.shape[2], cond.shape[3])
            cond_parts.insert(0, image_features)
        if self.config.env_state_feature:
            cond_parts.append(batch[OBS_ENV_STATE])
        return torch.cat(cond_parts, dim=-1)

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps
        cond = self._prepare_conditioning(batch)
        device = get_device_from_parameters(self.head)
        dtype = get_dtype_from_parameters(self.head)
        action_latent = noise if noise is not None else torch.randn(
            (batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            device=device,
            dtype=dtype,
        )
        time_grid = torch.linspace(
            1.0,
            0.0,
            steps=self.config.num_inference_steps + 1,
            device=device,
            dtype=dtype,
        )
        for step_index in range(self.config.num_inference_steps):
            t = torch.full((batch_size,), float(time_grid[step_index].item()), device=device, dtype=dtype)
            r = torch.full((batch_size,), float(time_grid[step_index + 1].item()), device=device, dtype=dtype)
            action_latent = self._sample_one_step(action_latent, r=r, t=t, cond=cond)
        action = self._decode_action_latent(action_latent)
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return action[:, start:end]

    def compute_loss(
        self,
        batch: dict[str, Tensor],
        current_step: int | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        assert set(batch).issuperset({OBS_STATE, ACTION})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        actions = batch[ACTION]
        batch_size = actions.shape[0]
        assert actions.shape[1] == self.config.horizon
        assert batch[OBS_STATE].shape[1] == self.config.n_obs_steps
        cond = self._prepare_conditioning(batch)

        x = self._encode_action_latent(actions)
        e = torch.randn_like(x)
        t, r = self._sample_tr(batch_size, device=x.device, dtype=x.dtype)

        t_broadcast = self._broadcast_batch_time(t, x)
        z_t = (1 - t_broadcast) * x + t_broadcast * e

        with torch.no_grad():
            v = self.fn(z_t, t, t, cond=cond)
        u, du_dt = self._compute_u_and_du_dt(z_t, r, t, cond=cond, v=v)
        du_dt = du_dt.detach()
        delta = self._broadcast_batch_time(t - r, du_dt)
        delta_du_dt = delta * du_dt
        compound_velocity = u + delta_du_dt
        target = e - x

        loss = self._velocity_loss_from_error(compound_velocity - target)
        loss = loss * self._action_latent_loss_weights(device=loss.device, dtype=loss.dtype)
        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            mask = (~batch["action_is_pad"]).unsqueeze(-1).to(loss.dtype)
            valid_count = mask.sum() * loss.shape[-1]
            scalar_loss = (loss * mask).sum() / valid_count.clamp_min(1.0)
        else:
            scalar_loss = loss.mean()

        diagnostics = {}
        semigroup_weight = self._semigroup_loss_weight(current_step)
        if semigroup_weight > 0.0:
            semigroup_loss = self._semigroup_consistency_loss(z_t, cond=cond)
            weighted_semigroup_loss = semigroup_loss * semigroup_loss.new_tensor(semigroup_weight)
            scalar_loss = scalar_loss + weighted_semigroup_loss
            diagnostics.update(
                {
                    "imf_diagnostics/semigroup/loss": float(semigroup_loss.detach().float().item()),
                    "imf_diagnostics/semigroup/weight": float(semigroup_weight),
                    "imf_diagnostics/semigroup/weighted_loss": float(
                        weighted_semigroup_loss.detach().float().item()
                    ),
                }
            )
        if self.config.enable_imf_diagnostics:
            diagnostics.update(
                self._imf_training_diagnostics(
                    loss=scalar_loss,
                    target=target,
                    u=u,
                    du_dt=du_dt,
                    delta_du_dt=delta_du_dt,
                    t=t,
                    r=r,
                )
            )
        return scalar_loss, diagnostics
