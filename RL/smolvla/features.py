"""Frozen prefix embeddings, contextual tokens and native Actor K/V from ONE VLM.

Critics read contextual tokens without flattening layers. Dynamics predicts
prefix embeddings; the SAME frozen VLM regenerates contextual tokens and K/V
together. No separately learned observation encoder or incompatible cache head.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lerobot.utils.import_utils import _transformers_available, require_package
from RL.types import ObservationBatch

if TYPE_CHECKING or _transformers_available:
    from transformers import DynamicCache
else:
    DynamicCache = None

TOKEN_KEY = "observation.smolvlm_tokens"
MASK_KEY = "observation.smolvlm_mask"


@dataclass(frozen=True)
class Condition:
    kv: Tensor  # [batch, layer, token, 2 * kv_heads * head_dim], RoPE already applied to keys
    mask: Tensor  # [batch, token], same task/layout across a modeled trajectory
    prefix: Tensor  # [batch, token, VLM hidden], embed_prefix output BEFORE the VLM text layers
    tokens: Tensor  # [batch, token, VLM hidden], final normalized VLM prefix hidden states
    dynamic_mask: Tensor  # visual/state slots only; never predict language, special tokens or padding

    def __post_init__(self):
        if self.kv.ndim != 4 or not self.kv.is_floating_point() or not torch.isfinite(self.kv).all():
            raise ValueError("K/V features must be finite [B,L,N,C] floats")
        if self.mask.dtype != torch.bool or self.mask.shape != (self.kv.shape[0], self.kv.shape[2]):
            raise ValueError("K/V padding mask must be bool [B,N]")
        if not self.mask.any(-1).all() or not self.mask[:, -1].all():
            raise ValueError("Each prefix must contain valid tokens and a final state token")
        for name, value in (("prefix", self.prefix), ("tokens", self.tokens)):
            if (
                value.ndim != 3
                or value.shape[:2] != self.mask.shape
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"{name} must be finite [B,N,D] floats matching the mask")
        if self.prefix.shape != self.tokens.shape:
            raise ValueError("Prefix and contextual token shapes must match")
        if self.dynamic_mask.dtype != torch.bool or self.dynamic_mask.shape != self.mask.shape:
            raise ValueError("Dynamic mask must be bool [B,N]")
        if (self.dynamic_mask & ~self.mask).any() or not self.dynamic_mask[:, -1].all():
            raise ValueError("Dynamic slots must be valid and include the final state token")
        if any(t.device != self.kv.device for t in (self.mask, self.prefix, self.tokens, self.dynamic_mask)):
            raise ValueError("All condition tensors must share a device")

    def observation(self):
        # Do not duplicate/float-convert the large native cache for IQL.
        return ObservationBatch({TOKEN_KEY: self.tokens.detach(), MASK_KEY: self.mask})

    def select(self, indices):
        return Condition(
            self.kv[indices],
            self.mask[indices],
            self.prefix[indices],
            self.tokens[indices],
            self.dynamic_mask[indices],
        )

    def cache(self, head_dim: int, dtype: torch.dtype):
        require_package("transformers", extra="smolvla")
        half = self.kv.shape[-1] // 2
        if self.kv.shape[-1] % 2 or half % head_dim:
            raise ValueError("Invalid K/V width")
        cache = DynamicCache()
        for layer in range(self.kv.shape[1]):
            cache.update(
                self.kv[:, layer, :, :half]
                .to(dtype)
                .reshape(self.kv.shape[0], self.kv.shape[2], -1, head_dim)
                .transpose(1, 2),
                self.kv[:, layer, :, half:]
                .to(dtype)
                .reshape(self.kv.shape[0], self.kv.shape[2], -1, head_dim)
                .transpose(1, 2),
                layer,
            )
        return cache


class FrozenSmolVLM:
    def __init__(self, policy: Any, preprocessor: Any):
        self.policy, self.preprocessor = policy, preprocessor
        config = policy.config
        if config.n_obs_steps != 1 or config.rtc_config is not None or config.adapt_to_pi_aloha:
            raise ValueError("Offline SmolVLA currently requires one observation, no RTC or ALOHA remapping")
        if not config.use_cache or config.pad_language_to != "max_length":
            raise ValueError("Shared K/V conditioning requires use_cache and fixed max_length tokenization")
        config.train_expert_only = True
        config.freeze_vision_encoder = True
        config.train_state_proj = False
        config.compile_model = False
        wrapper = policy.model.vlm_with_expert
        wrapper.train_expert_only = True
        wrapper.freeze_vision_encoder = True
        for module in (wrapper.vlm, policy.model.state_proj):
            module.eval()
            module.requires_grad_(False)
        policy.eval()

    @torch.no_grad()
    def encode(self, observation: dict[str, Any], task: str):
        self.policy.eval()
        batch = dict(observation)
        for key, value in batch.items():
            if key.startswith("observation.images.") and value.dtype == torch.uint8:
                batch[key] = value.float() / 255.0
        batch["task"] = task
        # The caller supplies batches, so expand task explicitly (AddBatch only wraps one string).
        batch["task"] = [task] * batch["observation.state"].shape[0]
        prepared = self.preprocessor(batch)
        missing = set(self.policy.config.image_features) - prepared.keys()
        if missing:
            raise ValueError(f"Missing policy cameras: {sorted(missing)}")
        images, masks = self.policy.prepare_images(prepared)
        prefix, pad, att = self.policy.model.embed_prefix(
            images,
            masks,
            prepared["observation.language.tokens"],
            prepared["observation.language.attention_mask"],
            self.policy.prepare_state(prepared),
        )
        expected_att = torch.zeros_like(pad)
        expected_att[:, -1] = True
        if not torch.equal(att.bool(), expected_att) or not pad[:, -1].all():
            raise ValueError("Offline token dynamics requires an unpadded final single state token")
        image_end = prefix.shape[1] - prepared["observation.language.tokens"].shape[1] - 1
        dynamic = pad.clone()
        dynamic[:, image_end:-1] = False
        if self.policy.model.add_image_special_tokens:
            block, remainder = divmod(image_end, len(images))
            if remainder:
                raise ValueError("Camera token layouts must have equal lengths")
            start_len = self.policy.model.global_image_start_token.numel()
            end_len = self.policy.model.image_end_token.numel()
            for i in range(len(images)):
                dynamic[:, i * block : i * block + start_len] = False
                dynamic[:, (i + 1) * block - end_len : (i + 1) * block] = False
        return self.from_prefix(prefix, pad, dynamic), prepared.get("action")

    @torch.no_grad()
    def from_prefix(self, prefix: Tensor, mask: Tensor, dynamic_mask: Tensor) -> Condition:
        """Re-encode a (real or predicted) prefix without images or a second VLM.

        Image/text tokens attend within their group; the final state token also
        attends to that group, exactly as native embed_prefix for n_obs_steps=1.
        """
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        self.policy.eval()
        if prefix.ndim != 3 or mask.shape != prefix.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Expected prefix [B,N,D] and bool mask [B,N]")
        if not mask[:, -1].all() or not torch.isfinite(prefix).all():
            raise ValueError("Expected finite prefix with final state token")
        att = torch.zeros_like(mask)
        att[:, -1] = True
        outputs, cache = self.policy.model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(mask, att),
            position_ids=mask.long().cumsum(1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )
        kv = torch.stack(
            [
                torch.cat(
                    (layer.keys.transpose(1, 2).flatten(2), layer.values.transpose(1, 2).flatten(2)), -1
                )
                for layer in cache.layers
            ],
            dim=1,
        ).detach()
        return Condition(kv, mask.detach(), prefix.detach(), outputs[0].detach(), dynamic_mask.detach())
