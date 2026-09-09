#!/usr/bin/env python

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

"""Transformer Diffusion denoiser for DP3 with prefix / joint self-attention conditioning.

This is a conventional Transformer Diffusion model: the objective stays the
standard DDPM/DDIM epsilon prediction owned by ``DiffusionModel``. Only the
denoiser changes. Observation conditioning is injected the way pi0 / pi0.5
condition their action experts: observation tokens and the diffusion timestep
token form a prefix, action tokens follow, and a single unified self-attention
runs over the whole sequence with a block mask::

                    K: prefix   K: action
    Q: prefix          yes         no
    Q: action          yes         yes

Action tokens read the observation prefix and each other (bidirectional inside
the chunk); the prefix never reads action tokens. There is no AdaLN / FiLM
modulation - conditioning enters only through the attention mask.

The reusable timestep and rotary embeddings come from the repository's
``multi_task_dit`` Transformer Diffusion implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (
    RotaryPositionalEmbedding,
    SinusoidalPosEmb,
)

from .configuration_dp3 import DP3Config


def build_prefix_attention_mask(prefix_len: int, action_len: int, device: torch.device) -> Tensor:
    """Return a boolean ``(T, T)`` mask where ``True`` means the query may attend to the key.

    ``T = prefix_len + action_len``. Prefix rows may only attend to prefix
    columns; action rows may attend to every column.
    """

    if prefix_len <= 0 or action_len <= 0:
        raise ValueError(f"prefix_len and action_len must be positive, got {prefix_len} and {action_len}.")
    total = prefix_len + action_len
    mask = torch.ones(total, total, dtype=torch.bool, device=device)
    mask[:prefix_len, prefix_len:] = False
    return mask


class PrefixCausalAttention(nn.Module):
    """Multi-head self-attention over ``[prefix | action]`` with an explicit block mask."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        dropout: float,
        use_rope: bool,
        max_seq_len: int,
        rope_base: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.rope = (
            RotaryPositionalEmbedding(head_dim=self.head_dim, max_seq_len=max_seq_len, base=rope_base)
            if use_rope
            else None
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope is not None:
            q, k = self.rope(q, k)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch, seq_len, self.hidden_size)
        return self.out_proj(out)


class PrefixCausalTransformerBlock(nn.Module):
    """Pre-norm transformer block: masked self-attention + MLP, no adaptive normalization."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        dropout: float,
        use_rope: bool,
        max_seq_len: int,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = PrefixCausalAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), attn_mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class DP3DiffusionTransformer(nn.Module):
    """Drop-in replacement for ``DiffusionConditionalUnet1d`` with the same forward contract.

    ``forward(x, timestep, global_cond)`` maps a noisy action chunk
    ``(B, horizon, action_dim)`` to a prediction of the same shape. The global
    conditioning vector is the DP3 observation encoder output flattened over
    ``n_obs_steps``; it is split back into one token per observation step.
    """

    def __init__(self, config: DP3Config, global_cond_dim: int) -> None:
        super().__init__()
        action_feature = config.action_feature
        if action_feature is None:
            raise ValueError("DP3 action feature must be validated before model construction.")
        if global_cond_dim <= 0 or global_cond_dim % config.n_obs_steps != 0:
            raise ValueError(
                "global_cond_dim must be a positive multiple of n_obs_steps, "
                f"got {global_cond_dim} and n_obs_steps={config.n_obs_steps}."
            )
        self.config = config
        self.action_dim = action_feature.shape[0]
        self.horizon = config.horizon
        self.n_obs_steps = config.n_obs_steps
        self.obs_dim = global_cond_dim // config.n_obs_steps
        self.hidden_size = config.transformer_hidden_dim
        # Prefix = one token per observation step plus one diffusion timestep token.
        self.prefix_len = config.n_obs_steps + 1
        self.seq_len = self.prefix_len + config.horizon

        self.obs_proj = nn.Linear(self.obs_dim, self.hidden_size)
        timestep_dim = config.transformer_timestep_embed_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(timestep_dim),
            nn.Linear(timestep_dim, 2 * timestep_dim),
            nn.GELU(),
            nn.Linear(2 * timestep_dim, self.hidden_size),
        )
        self.action_in = nn.Linear(self.action_dim, self.hidden_size)
        self.pos_embedding = (
            None
            if config.transformer_use_rope
            else nn.Parameter(torch.empty(1, self.seq_len, self.hidden_size).normal_(std=0.02))
        )
        self.blocks = nn.ModuleList(
            PrefixCausalTransformerBlock(
                self.hidden_size,
                config.transformer_num_heads,
                dropout=config.transformer_dropout,
                use_rope=config.transformer_use_rope,
                max_seq_len=self.seq_len,
                rope_base=config.transformer_rope_base,
            )
            for _ in range(config.transformer_num_layers)
        )
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.action_out = nn.Linear(self.hidden_size, self.action_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # GPT-2 / DiT style: residual branches start near identity so the
        # denoiser is well-conditioned at initialization.
        for block in self.blocks:
            nn.init.zeros_(block.attn.out_proj.weight)
            nn.init.zeros_(block.attn.out_proj.bias)
            nn.init.zeros_(block.mlp[-1].weight)
            nn.init.zeros_(block.mlp[-1].bias)
        nn.init.normal_(self.action_out.weight, std=0.02)
        nn.init.zeros_(self.action_out.bias)

    def attention_mask(self, device: torch.device) -> Tensor:
        return build_prefix_attention_mask(self.prefix_len, self.horizon, device)

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (B, horizon, action_dim) noisy action chunk.
            timestep: (B,) diffusion timesteps.
            global_cond: (B, n_obs_steps * obs_dim) flattened observation features.
        Returns:
            (B, horizon, action_dim) denoiser prediction.
        """

        if global_cond is None:
            raise ValueError("DP3DiffusionTransformer requires global_cond observation features.")
        batch, seq_len, action_dim = x.shape
        if seq_len != self.horizon or action_dim != self.action_dim:
            raise ValueError(
                f"expected action chunk of shape (B, {self.horizon}, {self.action_dim}), "
                f"got {tuple(x.shape)}."
            )
        if global_cond.shape != (batch, self.n_obs_steps * self.obs_dim):
            raise ValueError(
                f"global_cond must have shape {(batch, self.n_obs_steps * self.obs_dim)}, "
                f"got {tuple(global_cond.shape)}."
            )
        if not torch.is_tensor(timestep):
            timestep = torch.full((batch,), int(timestep), dtype=torch.long, device=x.device)
        elif timestep.ndim == 0:
            timestep = timestep.expand(batch)

        obs_tokens = self.obs_proj(global_cond.reshape(batch, self.n_obs_steps, self.obs_dim))
        time_token = self.time_mlp(timestep.to(dtype=torch.float32)).to(obs_tokens.dtype).unsqueeze(1)
        action_tokens = self.action_in(x)
        hidden = torch.cat((obs_tokens, time_token, action_tokens), dim=1)
        if self.pos_embedding is not None:
            hidden = hidden + self.pos_embedding

        mask = self.attention_mask(hidden.device)
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = self.final_norm(hidden[:, self.prefix_len :])
        return self.action_out(hidden)
