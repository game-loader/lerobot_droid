from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class RMSNormNoWeight(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype)


def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> Tensor:
    if dim % 2 != 0:
        raise ValueError(f'RoPE requires an even head dimension, got {dim}.')
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).reshape_as(x).to(x.dtype)


class GroupedQuerySelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model={d_model} must be divisible by n_heads={n_heads}.')
        if n_heads % n_kv_heads != 0:
            raise ValueError(f'n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}.')

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_kv_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.w_o = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        rope_freqs: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.w_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head)
        v = self.w_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        if self.n_kv_heads != self.n_heads:
            k = k.unsqueeze(3).expand(
                batch_size, seq_len, self.n_kv_heads, self.n_kv_groups, self.d_head
            )
            k = k.reshape(batch_size, seq_len, self.n_heads, self.d_head)
            v = v.unsqueeze(3).expand(
                batch_size, seq_len, self.n_kv_heads, self.n_kv_groups, self.d_head
            )
            v = v.reshape(batch_size, seq_len, self.n_heads, self.d_head)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_dropout(self.w_o(out))


class MultiheadDifferentialSelfAttention(nn.Module):
    """Differential self-attention without FlashAttention.

    This follows the practical Diff Transformer idea of forming two attention
    maps and subtracting a learned sigmoid-scaled second map from the first:
    ``Attn_1(Q_1,K) V - sigmoid(lambda(x)) * Attn_2(Q_2,K) V``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        causal_attn: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model={d_model} must be divisible by n_heads={n_heads}.')
        if n_heads % n_kv_heads != 0:
            raise ValueError(f'n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}.')

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_kv_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.causal_attn = causal_attn
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model, 2 * n_heads * self.d_head, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.lambda_proj = nn.Linear(d_model, n_heads, bias=False)
        self.w_o = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    @staticmethod
    def _build_causal_mask(seq_len: int, device: torch.device) -> Tensor:
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    def _repeat_kv(self, value: Tensor, batch_size: int, seq_len: int) -> Tensor:
        if self.n_kv_heads == self.n_heads:
            return value
        value = value.unsqueeze(3).expand(
            batch_size,
            seq_len,
            self.n_kv_heads,
            self.n_kv_groups,
            self.d_head,
        )
        return value.reshape(batch_size, seq_len, self.n_heads, self.d_head)

    def _attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        scale = 1.0 / math.sqrt(self.d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        return torch.matmul(attn_weights, v)

    def forward(
        self,
        x: Tensor,
        rope_freqs: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, 2 * self.n_heads, self.d_head)
        k = self.w_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head)
        v = self.w_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head)

        if rope_freqs is not None:
            q = apply_rope(q, rope_freqs)
            k = apply_rope(k, rope_freqs)

        q = q.view(batch_size, seq_len, 2, self.n_heads, self.d_head)
        q1 = q[:, :, 0]
        q2 = q[:, :, 1]
        k = self._repeat_kv(k, batch_size, seq_len)
        v = self._repeat_kv(v, batch_size, seq_len)

        q1 = q1.transpose(1, 2)
        q2 = q2.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is None and self.causal_attn:
            mask = self._build_causal_mask(seq_len, x.device)

        attn1 = self._attention(q1, k, v, mask)
        attn2 = self._attention(q2, k, v, mask)
        lambda_value = torch.sigmoid(self.lambda_proj(x)).transpose(1, 2).unsqueeze(-1)
        out = attn1 - lambda_value * attn2
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_dropout(self.w_o(out))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, mult: float = 2.667) -> None:
        super().__init__()
        raw = int(mult * d_model)
        d_ff = ((raw + 7) // 8) * 8
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class AttnResOperator(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))
        self.key_norm = RMSNormNoWeight(eps=eps)
        self.last_depth_attention_weights: Tensor | None = None

    def forward(self, sources: Tensor) -> Tensor:
        keys = self.key_norm(sources)
        logits = torch.einsum('d,nbtd->nbt', self.pseudo_query, keys)
        weights = F.softmax(logits, dim=0)
        self.last_depth_attention_weights = weights.detach()
        return torch.einsum('nbt,nbtd->btd', weights, sources)


class AttnResSubLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float,
        ffn_mult: float,
        eps: float,
        is_attention: bool,
        use_differential_attention: bool = False,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model, eps=eps)
        self.attn_res = AttnResOperator(d_model, eps=eps)
        self.is_attention = is_attention
        if self.is_attention:
            attention_cls = (
                MultiheadDifferentialSelfAttention if use_differential_attention else GroupedQuerySelfAttention
            )
            attention_kwargs = {
                'd_model': d_model,
                'n_heads': n_heads,
                'n_kv_heads': n_kv_heads,
                'dropout': dropout,
            }
            if use_differential_attention:
                attention_kwargs['causal_attn'] = False
            self.fn = attention_cls(**attention_kwargs)
        else:
            self.fn = SwiGLUFFN(d_model=d_model, dropout=dropout, mult=ffn_mult)

    def forward(self, sources: Tensor, rope_freqs: Tensor, mask: Tensor | None = None) -> Tensor:
        h = self.attn_res(sources)
        normed = self.norm(h)
        if self.is_attention:
            return self.fn(normed, rope_freqs, mask)
        return self.fn(normed)


class AttnResTransformerBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        ffn_mult: float = 2.667,
        eps: float = 1e-6,
        rope_theta: float = 10000.0,
        causal_attn: bool = False,
        use_differential_attention: bool = False,
    ) -> None:
        super().__init__()
        self.causal_attn = causal_attn
        self.layers = nn.ModuleList()
        for _ in range(n_blocks):
            self.layers.append(
                AttnResSubLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                    eps=eps,
                    is_attention=True,
                    use_differential_attention=use_differential_attention,
                )
            )
            self.layers.append(
                AttnResSubLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                    eps=eps,
                    is_attention=False,
                )
            )

        rope_freqs = precompute_rope_freqs(
            dim=d_model // n_heads,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )
        self.register_buffer('rope_freqs', rope_freqs, persistent=False)

    @staticmethod
    def _build_causal_mask(seq_len: int, device: torch.device) -> Tensor:
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        rope_freqs = self.rope_freqs[:seq_len]
        mask = None
        if self.causal_attn:
            mask = self._build_causal_mask(seq_len, x.device)

        layer_outputs = [x]
        for layer in self.layers:
            sources = torch.stack(layer_outputs, dim=0)
            output = layer(sources, rope_freqs, mask)
            layer_outputs.append(output)

        return torch.stack(layer_outputs, dim=0).sum(dim=0)


class DifferentialTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float,
        ffn_mult: float,
        eps: float,
        causal_attn: bool,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.attn = MultiheadDifferentialSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            causal_attn=causal_attn,
        )
        self.ffn_norm = RMSNorm(d_model, eps=eps)
        self.ffn = SwiGLUFFN(d_model=d_model, dropout=dropout, mult=ffn_mult)

    def forward(self, x: Tensor, rope_freqs: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs=rope_freqs, mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DifferentialTransformerBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        ffn_mult: float = 2.667,
        eps: float = 1e-6,
        rope_theta: float = 10000.0,
        causal_attn: bool = False,
    ) -> None:
        super().__init__()
        self.causal_attn = causal_attn
        self.layers = nn.ModuleList(
            [
                DifferentialTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                    eps=eps,
                    causal_attn=causal_attn,
                )
                for _ in range(n_blocks)
            ]
        )
        rope_freqs = precompute_rope_freqs(
            dim=d_model // n_heads,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )
        self.register_buffer('rope_freqs', rope_freqs, persistent=False)

    @staticmethod
    def _build_causal_mask(seq_len: int, device: torch.device) -> Tensor:
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        rope_freqs = self.rope_freqs[:seq_len]
        mask = None
        if self.causal_attn:
            mask = self._build_causal_mask(seq_len, x.device)
        for layer in self.layers:
            x = layer(x, rope_freqs=rope_freqs, mask=mask)
        return x
