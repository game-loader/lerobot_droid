"""Action-query Transformer critic adapted from Q-Planning, with real DINOv3 features."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from lerobot.utils.import_utils import _transformers_available, require_package

from .config import FastWAMQConfig

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModel, AutoTokenizer, T5EncoderModel


class DINOv3ImageEncoder(nn.Module):
    """DINOv3 patch tokens with camera identity and decoder-side spatial embeddings."""

    def __init__(self, config: FastWAMQConfig, backbone=None):
        """Construct the component from its configuration and supplied dependencies."""
        super().__init__()
        require_package("transformers", extra="fastwam")
        self.config = config
        self.backbone = backbone if backbone is not None else AutoModel.from_pretrained(config.dino_model)
        bc = self.backbone.config
        # Native DINOv3 checkpoints retain BF16-rounded RoPE periods. Preserve
        # them when loading our lossless conversion instead of regenerating them.
        periods = getattr(bc, "native_rope_periods", None)
        if periods is not None:
            frequency = self.backbone.rope_embeddings.inv_freq
            frequency.copy_(1 / torch.tensor(periods, dtype=frequency.dtype, device=frequency.device))
        self.patch_size = bc.patch_size
        self.num_register_tokens = bc.num_register_tokens
        self.projection = (
            nn.Identity()
            if bc.hidden_size == config.dim_model
            else nn.Linear(bc.hidden_size, config.dim_model)
        )
        h, w = config.image_size
        self.patch_position = nn.Parameter(
            torch.randn(
                len(config.camera_keys), (h // self.patch_size) * (w // self.patch_size), config.dim_model
            )
            * 0.02
        )
        self.view_embedding = nn.Embedding(len(config.camera_keys), config.dim_model)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.backbone.requires_grad_(not config.freeze_dino)
        if config.gradient_checkpointing and not config.freeze_dino:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def train(self, mode=True):
        """Set training mode while keeping frozen encoders and targets deterministic."""
        super().train(mode)
        if self.config.freeze_dino:
            self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        # Float RGB is [0,1]; uint8 RGB is [0,255]. No batch-max heuristic or double normalization.
        """Compute the component output for the supplied batch."""
        b, v, c, h, w = images.shape
        pixels = images.reshape(b * v, c, h, w).float()
        if images.dtype == torch.uint8:
            pixels = pixels / 255
        pixels = functional.interpolate(
            pixels, self.config.image_size, mode="bilinear", align_corners=False, antialias=True
        )
        pixels = (pixels - self.image_mean) / self.image_std
        tokens = self.backbone(pixel_values=pixels).last_hidden_state[:, 1 + self.num_register_tokens :]
        tokens = self.projection(tokens).reshape(b, v, -1, self.config.dim_model)
        positions = self.patch_position + self.view_embedding.weight[:, None]
        return (tokens + positions[None]).flatten(1, 2)


class T5TextEncoder(nn.Module):
    """Frozen T5; cache only unpadded task tokens on CPU, then batch-pad with an attention mask."""

    def __init__(self, config, model=None, tokenizer=None):
        """Construct the component from its configuration and supplied dependencies."""
        super().__init__()
        require_package("transformers", extra="fastwam")
        self.model = model if model is not None else T5EncoderModel.from_pretrained(config.text_model)
        self.tokenizer = (
            tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(config.text_model)
        )
        self.model.requires_grad_(False).eval()
        self.hidden_size = self.model.config.d_model
        self.max_length = config.text_max_length
        self._cache = {}

    def train(self, mode=True):
        """Set training mode while keeping frozen encoders and targets deterministic."""
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, tasks, device):
        """Compute the component output for the supplied batch."""
        missing = list(dict.fromkeys(t for t in tasks if t not in self._cache))
        if missing:
            encoded = self.tokenizer(
                missing, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(next(self.model.parameters()).device)
            output = self.model(**encoded).last_hidden_state
            for task, features, mask in zip(missing, output, encoded.attention_mask, strict=True):
                self._cache[task] = features[mask.bool()].float().cpu()
        lengths = [len(self._cache[t]) for t in tasks]
        tokens = torch.zeros(len(tasks), max(lengths), self.hidden_size, device=device)
        padding = torch.ones(len(tasks), max(lengths), dtype=torch.bool, device=device)
        for i, task in enumerate(tasks):
            tokens[i, : lengths[i]] = self._cache[task].to(device)
            padding[i, : lengths[i]] = False
        return tokens, padding


class QKNormMultiheadAttention(nn.MultiheadAttention):
    """Critic-only batch-first attention with affine-free, post-projection per-head Q/K RMS norm.

    Keep MultiheadAttention parameter names and initialization so old checkpoints remain loadable.
    Unit-RMS Q/K with standard SDPA scaling gives sqrt(head_dim) times cosine similarity;
    there is no learned gain or temperature that could undo the normalization.
    """

    def __init__(self, config):
        """Reuse the existing attention projections without adding trainable parameters."""
        super().__init__(config.dim_model, config.n_heads, dropout=config.dropout, batch_first=True)
        self.qk_norm_eps = config.qk_norm_eps

    def normalize_heads(self, tokens):
        """Compute RMS in FP32, then return the original AMP dtype for SDPA."""
        value = tokens.float()
        return (value * (value.square().mean(-1, keepdim=True) + self.qk_norm_eps).rsqrt()).to(tokens.dtype)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        """Project and normalize Q/K for this critic's noncausal, need_weights=False calls."""
        batch = query.shape[0]
        projections = [
            functional.linear(tokens, weight, bias)
            .reshape(batch, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            for tokens, weight, bias in zip(
                (query, key, value), self.in_proj_weight.chunk(3), self.in_proj_bias.chunk(3), strict=True
            )
        ]
        query, key, value = projections
        mask = None if key_padding_mask is None else ~key_padding_mask[:, None, None, :]
        output = functional.scaled_dot_product_attention(
            self.normalize_heads(query),
            self.normalize_heads(key),
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, -1, self.embed_dim)
        return self.out_proj(output), None


class DecoderLayer(nn.Module):
    """Pre-norm action self-attention, image/text cross-attention, and FFN."""

    def __init__(self, config):
        """Construct the component from its configuration and supplied dependencies."""
        super().__init__()
        d = config.dim_model
        if config.qk_norm:
            self.self_attn = QKNormMultiheadAttention(config)
            self.cross_attn = QKNormMultiheadAttention(config)
        else:
            self.self_attn = nn.MultiheadAttention(
                d, config.n_heads, dropout=config.dropout, batch_first=True
            )
            self.cross_attn = nn.MultiheadAttention(
                d, config.n_heads, dropout=config.dropout, batch_first=True
            )
        self.ffn = nn.Sequential(
            nn.Linear(d, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, d),
        )
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, context, position, action_padding, context_padding):
        """Compute the component output for the supplied batch."""
        # BF16 efficient SDPA with dropout amplifies backward roundoff in deep critics.
        # Math SDPA keeps attention intermediates in FP32; inference retains fused kernels.
        with sdpa_kernel(SDPBackend.MATH) if self.training else nullcontext():
            z = self.norm1(x)
            x = x + self.dropout(
                self.self_attn(
                    z + position, z + position, z, key_padding_mask=action_padding, need_weights=False
                )[0]
            )
            x = x + self.dropout(
                self.cross_attn(
                    self.norm2(x) + position,
                    context,
                    context,
                    key_padding_mask=context_padding,
                    need_weights=False,
                )[0]
            )
        return x + self.dropout(self.ffn(self.norm3(x)))


class FastWAMQNetwork(nn.Module):
    """Score a complete action chunk with action queries and image/text context."""

    def __init__(self, config, image_encoder, text_dim=768):
        """Construct the component from its configuration and supplied dependencies."""
        super().__init__()
        self.config, self.image_encoder = config, image_encoder
        d = config.dim_model
        self.text_projection = nn.Linear(text_dim, d) if config.use_text else None
        self.action_projection = nn.Linear(config.action_dim, d)
        self.query_position = nn.Parameter(torch.randn(config.chunk_size, d) * 0.02)
        self.cls_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.output_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(d, config.num_bins)
        )

    def encode_context(self, images, text=None):
        """Encode visual and optional text tokens once for candidate reuse."""
        context = self.image_encoder(images)
        padding = torch.zeros(context.shape[:2], dtype=torch.bool, device=context.device)
        if text is not None:
            tokens, text_padding = text
            context = torch.cat([self.text_projection(tokens), context], dim=1)
            padding = torch.cat([text_padding, padding], dim=1)
        return context, padding

    def forward_with_context(self, context, actions, valid=None):
        """Score action queries against a previously encoded context."""
        tokens, context_padding = context
        b, h, _ = actions.shape
        queries = torch.cat([self.cls_query.expand(b, -1, -1), self.action_projection(actions)], dim=1)
        position = functional.pad(self.query_position[:h], (0, 0, 1, 0))[None]
        padding = None if valid is None else functional.pad(~valid, (1, 0), value=False)
        for layer in self.layers:
            args = (queries, tokens, position, padding, context_padding)
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                queries = checkpoint(layer, *args, use_reentrant=False)
            else:
                queries = layer(*args)
        return self.head(self.output_norm(queries[:, 0]))

    def forward(self, images, actions, text=None, valid=None):
        """Compute the component output for the supplied batch."""
        return self.forward_with_context(self.encode_context(images, text), actions, valid)


class FastWAMQFunction(nn.Module):
    """Online critic, frozen EMA target, and shared frozen text encoder."""

    def __init__(self, config, *, dino_backbone=None, text_encoder=None, with_target=True):
        """Construct the component from its configuration and supplied dependencies."""
        super().__init__()
        self.config = config
        self.text_encoder = (
            text_encoder if text_encoder is not None else T5TextEncoder(config) if config.use_text else None
        )
        text_dim = self.text_encoder.hidden_size if self.text_encoder is not None else 0
        self.online = FastWAMQNetwork(config, DINOv3ImageEncoder(config, dino_backbone), text_dim)
        self.target = copy.deepcopy(self.online).requires_grad_(False).eval() if with_target else None
        self.register_buffer("bin_centers", torch.linspace(config.v_min, config.v_max, config.num_bins))

    def train(self, mode=True):
        """Set training mode while keeping frozen encoders and targets deterministic."""
        super().train(mode)
        if self.target is not None:
            self.target.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def text_features(self, tasks, device):
        """Encode task strings through the frozen cached text encoder."""
        return None if self.text_encoder is None else self.text_encoder(tasks, device)

    def q_values(self, logits):
        """Decode categorical logits as expected discounted returns in FP32."""
        return (logits.float().softmax(-1) * self.bin_centers).sum(-1)

    @torch.no_grad()
    def score_candidates(self, images, candidates, tasks):
        """Score candidate chunks in bounded sub-batches using shared visual context."""
        b, n, h, d = candidates.shape
        context = self.online.encode_context(images, self.text_features(tasks, images.device))
        values = []
        # Bound the N-dependent decoder memory; encode images/text only once.
        for start in range(0, n, self.config.candidate_batch_size):
            actions = candidates[:, start : start + self.config.candidate_batch_size]
            count = actions.shape[1]
            expanded = tuple(x.repeat_interleave(count, dim=0) for x in context)
            logits = self.online.forward_with_context(expanded, actions.reshape(b * count, h, d))
            values.append(self.q_values(logits).view(b, count))
        return torch.cat(values, dim=1)

    @torch.no_grad()
    def polyak_update(self):
        """Update target parameters after an optimizer step and copy nonparameter buffers."""
        for online, target in zip(self.online.parameters(), self.target.parameters(), strict=True):
            target.lerp_(online, self.config.target_tau)
        for online, target in zip(self.online.buffers(), self.target.buffers(), strict=True):
            target.copy_(online)

    def td_loss(self, batch):
        """Compute masked chunk-TD targets and Gaussian-histogram cross entropy."""
        text = self.text_features(batch["tasks"], batch["images"].device)
        # Evaluate the target first, before allocating the online backward graph.
        with torch.no_grad():
            q_next = self.q_values(
                self.target(batch["next_images"], batch["next_action_chunk"], text, batch["next_valid"])
            )
            targets = chunk_targets(
                batch["rewards"], batch["valid"], batch["done"], q_next, self.config.gamma
            )
            target_dist = hl_gauss_target(targets, self.bin_centers, self.config.hl_gauss_sigma)
        logits = self.online(batch["images"], batch["action_chunk"], text, batch["valid"])
        loss = -(target_dist * logits.float().log_softmax(-1)).sum(-1).mean()
        with torch.no_grad():
            prediction = self.q_values(logits.detach())
            td_error = prediction - targets
            log_probs = logits.detach().float().log_softmax(-1)
            target_entropy = -(target_dist * target_dist.clamp_min(1e-30).log()).sum(-1).mean()
        metrics = {
            "loss": loss.detach(),
            "q_pred": prediction.mean(),
            "q_target": targets.mean(),
            "bootstrap_fraction": (~batch["done"]).float().mean(),
            "td_mae": td_error.abs().mean(),
            "td_mse": td_error.square().mean(),
            "td_abs_max": td_error.abs().max(),
            "logit_abs_max": logits.detach().float().abs().max(),
            "prediction_entropy": -(log_probs.exp() * log_probs).sum(-1).mean(),
            "target_entropy": target_entropy,
            "hl_kl": loss.detach() - target_entropy,
        }
        return loss, metrics


def chunk_targets(rewards, valid, done, q_next, gamma):
    """Accumulate primitive-step rewards and mask terminal chunk bootstrapping."""
    discounts = gamma ** torch.arange(rewards.shape[1], device=rewards.device, dtype=torch.float32)
    reward = (rewards.float() * valid * discounts).sum(-1)
    return reward + (~done).float() * gamma ** valid.sum(-1) * q_next.float()


def hl_gauss_target(values, bins, sigma):
    # Same Gaussian-on-bin-centres target as the released Q-Planning implementation.
    """Encode scalar returns with Gaussian probabilities over fixed bin centres."""
    values = values.float().clamp(bins[0], bins[-1]).unsqueeze(-1)
    return (-0.5 * ((bins[None] - values) / sigma).square()).softmax(-1)
