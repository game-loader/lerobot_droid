# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2023 Tal Daniel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Extracted from modules/modules.py at LPWM 4cf53c403433e64c01652ac2adbec66231a46dea.
# ruff: noqa
# fmt: off

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Beta
from .utils import reparameterize, spatial_transform, create_masks_fast, create_masks_with_scale, modulate
from .vision_encoder import Encoder


class AlternativeSpatialSoftmaxKP(torch.nn.Module):
    """
    This module performs spatial-softmax (ssm) by performing marginalization over heatmaps.
    """

    def __init__(self, kp_range=(-1, 1)):
        super().__init__()
        self.kp_range = kp_range

    def forward(self, heatmap, probs=False, variance=False):
        batch_size, n_kp, height, width = heatmap.shape
        # p(x) = \int p(x,y)dy
        logits = heatmap.view(batch_size, n_kp, -1)  # [batch_size, n_kp, h * w]
        scores = torch.softmax(logits, dim=-1)  # [batch_size, n_kp, h * w]
        scores = scores.view(batch_size, n_kp, height, width)  # [batch_size, n_kp, h, w]
        y_axis = torch.linspace(self.kp_range[0], self.kp_range[1], height,
                                device=scores.device).type_as(scores).expand(1, 1, -1)  # [1, 1, features_dim_height]
        x_axis = torch.linspace(self.kp_range[0], self.kp_range[1], width,
                                device=scores.device).type_as(scores).expand(1, 1, -1)  # [1, 1, features_dim_width]

        # marginalize over x (width) and y (height)
        sm_h = scores.sum(dim=-1)  # [batch_size, n_kp, h]
        sm_w = scores.sum(dim=-2)  # [batch_size, n_kp, w]

        # # expected value: probability per coordinate * coordinate
        kp_h = torch.sum(sm_h * y_axis, dim=-1, keepdim=True)  # [batch_size, n_kp, 1]
        kp_h = kp_h.squeeze(-1)  # [batch_size, n_kp], y coordinate of each kp

        kp_w = torch.sum(sm_w * x_axis, dim=-1, keepdim=True)  # [batch_size, n_kp, 1]
        kp_w = kp_w.squeeze(-1)  # [batch_size, n_kp], x coordinate of each kp

        # stack keypoints
        kp = torch.stack([kp_h, kp_w], dim=-1)  # [batch_size, n_kp, 2], x, y coordinates of each kp

        if variance:
            # sigma^2 = E[x^2] - (E[x])^2
            y_sq = (scores * (y_axis.unsqueeze(-1) ** 2)).sum(dim=(-2, -1))  # [batch_size, n_kp]
            v_h = (y_sq - kp_h ** 2).clamp_min(1e-6)  # [batch_size, n_kp]
            x_sq = (scores * (x_axis.unsqueeze(-2) ** 2)).sum(dim=(-2, -1))  # [batch_size, n_kp]
            v_w = (x_sq - kp_w ** 2).clamp_min(1e-6)  # [batch_size, n_kp]

            # covariance: E[xy] - E[x]E[y]
            xy_sq = (scores * (y_axis.unsqueeze(-1) * x_axis.unsqueeze(-2))).sum(dim=(-2, -1))  # [batch_size, n_kp]
            cov = xy_sq - kp_h * kp_w

            var = torch.stack([v_h, v_w, cov], dim=-1)
            return kp, var
        if probs:
            return kp, sm_h, sm_w
        else:
            return kp


class ImagePatcher(nn.Module):
    """
    Author: Tal Daniel
    This module take an image of size B x cdim x H x W and return a patchified tesnor
    B x cdim x num_patches x patch_size x patch_size. It also gives you the global location of the patch
    w.r.t the original image. We use this module to extract prior KP from patches, and we need to know their
    global coordinates for the Chamfer-KL.
    """

    def __init__(self, cdim=3, image_size=64, patch_size=16):
        super(ImagePatcher, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.patch_size = patch_size
        self.kh, self.kw = self.patch_size, self.patch_size  # kernel size
        self.dh, self.dw = self.patch_size, patch_size  # stride
        self.unfold_shape = self.get_unfold_shape()
        self.patch_location_idx = self.get_patch_location_idx()
        # print(f'unfold shape: {self.unfold_shape}')
        # print(f'patch locations: {self.patch_location_idx}')

    def get_patch_location_idx(self):
        h = np.arange(0, self.image_size)[::self.patch_size]
        w = np.arange(0, self.image_size)[::self.patch_size]
        ww, hh = np.meshgrid(h, w)
        hw = np.stack((hh, ww), axis=-1)
        hw = hw.reshape(-1, 2)
        # return torch.from_numpy(hw).int()
        return torch.tensor(hw, dtype=torch.int)

    def get_patch_centers(self):
        mid = self.patch_size // 2
        patch_locations_idx = self.get_patch_location_idx()
        patch_locations_idx += mid
        return patch_locations_idx

    def get_unfold_shape(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        patches = dummy_input.unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        unfold_shape = patches.shape[1:]
        return unfold_shape

    def img_to_patches(self, x):
        patches = x.unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        patches = patches.contiguous().view(patches.shape[0], patches.shape[1], -1, self.kh, self.kw)
        return patches

    def patches_to_img(self, x):
        patches_orig = x.view(x.shape[0], *self.unfold_shape)
        output_h = self.unfold_shape[1] * self.unfold_shape[3]
        output_w = self.unfold_shape[2] * self.unfold_shape[4]
        patches_orig = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_orig = patches_orig.view(-1, self.cdim, output_h, output_w)
        return patches_orig

    def forward(self, x, patches=True):
        # x [batch_size, 3, image_size, image_size] or [batch_size, 3, num_patches, image_size, image_size]
        if patches:
            return self.img_to_patches(x)
        else:
            return self.patches_to_img(x)


class ParticleNorm(nn.Module):
    """
    experimental particle normalization module, not used in the code but left here for research
    """

    def __init__(self, particle_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.particle_dim = particle_dim
        self.a = nn.Parameter(torch.ones(1, 1, 1, self.particle_dim))
        self.g = nn.Parameter(torch.ones(1, 1, 1, self.particle_dim))
        self.s = nn.Parameter(torch.zeros(1, 1, 1, self.particle_dim))

    def forward(self, x):
        # [bs, n_particles, T, dim]
        dims = (1,)
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        if len(x.shape) == 3:
            d_n = (x - self.a.squeeze(2) * mean) / (var + self.eps).sqrt()
            out = d_n * self.g.squeeze(2) + self.s.squeeze(2)
        else:
            d_n = (x - self.a * mean) / (var + self.eps).sqrt()
            out = d_n * self.g + self.s
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # F.normalize: x = x / (x ** 2).sum(-1, keepdim=True).sqrt()
        return F.normalize(x, dim=-1) * self.scale * self.g


class SimpleRelativePositionalBias(nn.Module):
    # adapted from https://github.com/facebookresearch/mega
    def __init__(self, max_positions, num_heads=1, max_particles=None, layer_norm=False):
        super().__init__()
        self.max_positions = max_positions
        self.num_heads = num_heads
        self.max_particles = max_particles
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_positions - 1, self.num_heads))
        self.ln_t = nn.LayerNorm([2 * max_positions - 1, self.num_heads]) if layer_norm else nn.Identity()

        if self.max_particles is not None:
            self.particle_rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_particles - 1, self.num_heads))
            self.ln_p = nn.LayerNorm([2 * max_particles - 1, self.num_heads]) if layer_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.rel_pos_bias, mean=0.0, std=std)
        if self.max_particles is not None:
            nn.init.normal_(self.particle_rel_pos_bias, mean=0.0, std=std)

    def get_particle_rel_position(self, num_particles):
        if self.max_particles is None:
            return 0.0
        if num_particles > self.max_particles:
            raise ValueError('Num particles {} going beyond max particles {}'.format(num_particles, self.max_particles))

        # seq_len * 2 -1
        in_ln = self.ln_p(self.particle_rel_pos_bias)
        b = in_ln[(self.max_particles - num_particles):(self.max_particles + num_particles - 1)]
        # seq_len * 3 - 1
        t = F.pad(b, (0, 0, 0, num_particles))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (num_particles, 1))
        t = t[:-num_particles]
        # seq_len x (3 * seq_len - 2)
        t = t.view(num_particles, 3 * num_particles - 2, b.shape[-1])
        r = (2 * num_particles - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]  # [seq_len, seq_len, n_heads]
        t = t.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        return t

    def forward(self, seq_len, num_particles=None):
        if seq_len > self.max_positions:
            raise ValueError('Sequence length {} going beyond max length {}'.format(seq_len, self.max_positions))

        # seq_len * 2 -1
        in_ln = self.ln_t(self.rel_pos_bias)
        b = in_ln[(self.max_positions - seq_len):(self.max_positions + seq_len - 1)]
        # seq_len * 3 - 1
        t = F.pad(b, (0, 0, 0, seq_len))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (seq_len, 1))
        t = t[:-seq_len]
        # seq_len x (3 * seq_len - 2)
        t = t.view(seq_len, 3 * seq_len - 2, b.shape[-1])
        r = (2 * seq_len - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]  # [seq_len, seq_len, n_heads]
        t = t.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        p = None
        if num_particles is not None and self.max_particles is not None:
            p = self.get_particle_rel_position(num_particles)  # [1, n_heads, n_part, n_part]
            t = t[:, :, None, :, None, :]
            p = p[:, :, :, None, :, None]
        return t, p


class ParticleSelfAttention(nn.Module):
    """
    A particle-based multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1,
                 positional_bias=False, max_particles=None, linear_bias=False, torch_attn=False):
        super().__init__()
        assert n_embed % n_head == 0
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.torch_attn = torch_attn
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.query = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.value = nn.Linear(n_embed, n_embed, bias=linear_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop) if not self.torch_attn else nn.Identity()
        # output projection
        self.proj = nn.Linear(n_embed, n_embed, bias=linear_bias)

        self.resid_drop = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.positional_bias = positional_bias
        self.max_particles = max_particles
        if self.positional_bias:
            self.rel_pos_bias = SimpleRelativePositionalBias(block_size, n_head, max_particles=max_particles)
        else:
            self.rel_pos_bias = nn.Identity()

    def forward(self, x):
        B, N, T, C = x.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        q = self.query(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        v = self.value(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)

        if self.torch_attn:
            y = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False,
                                               dropout_p=self.attn_pdrop if self.training else 0.0)

        else:
            # causal self-attention; Self-attend: (B, nh, N * T, hs) x (B, nh, hs, N  *T) -> (B, nh, N * T, N *T )
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, N * T, N * T)
            if self.positional_bias:
                att = att.view(B, -1, N, T, N, T)  # (B, nh, N, T, N, T)
                if self.max_particles is not None:
                    bias_t, bias_p = self.rel_pos_bias(T, num_particles=N)
                    bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                    bias_p = bias_p.view(1, bias_p.shape[1], N, 1, N, 1)
                    att = att + bias_t + bias_p
                else:
                    bias_t, _ = self.rel_pos_bias(T)
                    bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                    att = att + bias_t
                att = att.view(B, -1, N * T, N * T)  # (B, nh, N * T, N * T)
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v  # (B, nh, N*T, N*T) x (B, nh, N*T, hs) -> (B, nh, N*T, hs)

        y = y.transpose(1, 2).contiguous().view(B, N * T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        y = y.view(B, N, T, -1)
        return y


class MLP(nn.Module):
    def __init__(self, n_embed, resid_pdrop=0.1, hidden_dim_multiplier=4, activation='gelu'):
        super().__init__()
        self.fc_1 = nn.Linear(n_embed, hidden_dim_multiplier * n_embed)
        if activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(True)
        self.proj = nn.Linear(hidden_dim_multiplier * n_embed, n_embed)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        x = self.dropout(self.proj(self.act(self.fc_1(x))))
        return x


class SelfBlock(nn.Module):
    """ self-attention Transformer block """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 positional_bias=False, activation='gelu', max_particles=None, norm_type='ln', context_cond=False,
                 residual_modulation=False, context_gate=False, attn_scale=1.0):
        super().__init__()
        self.max_particles = max_particles
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        self.ln1 = norm_layer(n_embed)
        self.ln2 = norm_layer(n_embed)
        self.attn = ParticleSelfAttention(n_embed, n_head, block_size, attn_pdrop, resid_pdrop,
                                          positional_bias=positional_bias, max_particles=max_particles)
        self.attn_scale = attn_scale
        self.mlp = MLP(n_embed, resid_pdrop, hidden_dim_multiplier, activation=activation)
        self.context_cond = context_cond
        self.residual_modulation = residual_modulation
        self.context_gate = context_gate
        self.c_multiplier = 6 if context_gate else 4
        if self.context_cond:
            self.c_proj = nn.Linear(n_embed, self.c_multiplier * n_embed)
            nn.init.constant_(self.c_proj.weight, 0.0)
            if self.residual_modulation:
                nn.init.constant_(self.c_proj.bias, 0.0)
            else:
                nn.init.constant_(self.c_proj.bias[:2 * n_embed], 1.0)  # identity
                nn.init.constant_(self.c_proj.bias[2 * n_embed: 4 * n_embed], 0.0)  # zero shift
                if self.context_gate:
                    nn.init.constant_(self.c_proj.bias[4 * n_embed:], 0.0)  # zero gate

    def forward(self, x, c=None):
        if self.context_cond and c is not None:
            c_proj = self.c_proj(c).chunk(self.c_multiplier, dim=-1)
            scale_a, scale_b, shift_a, shift_b = c_proj[0], c_proj[1], c_proj[2], c_proj[3]
            if self.context_gate:
                gate_a, gate_b = c_proj[4], c_proj[5]
            else:
                gate_a = gate_b = 1.0
            x = x + self.attn_scale * gate_a * self.attn(
                modulate(self.ln1(x), scale_a, shift_a, self.residual_modulation))
            x = x + gate_b * self.mlp(modulate(self.ln2(x), scale_b, shift_b, self.residual_modulation))
        else:
            x = x + self.attn_scale * self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x


class ParticleSelfAttTransformer(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, output_dim, attn_pdrop=0.1, resid_pdrop=0.1,
                 hidden_dim_multiplier=4, positional_bias=False, activation='gelu', max_particles=None,
                 norm_type='rms', n_registers=0, init_std=0.02):
        super().__init__()
        self.positional_bias = positional_bias
        self.max_particles = max_particles  # for positional bias
        self.n_registers = n_registers  # "vision transformers need registers", balances the attention matrix
        # input embedding stem
        if self.positional_bias:
            self.pos_emb = nn.Identity()
        else:
            self.pos_emb = nn.Parameter(init_std * torch.randn(1, block_size, n_embed))
        if self.n_registers > 0:
            self.registers = nn.Parameter(init_std * torch.randn(1, self.n_registers, 1, n_embed))
        else:
            self.registers = None
        # transformer
        self.blocks = nn.Sequential(*[SelfBlock(n_embed, n_head, block_size, attn_pdrop,
                                                resid_pdrop, hidden_dim_multiplier,
                                                positional_bias, activation=activation, max_particles=max_particles,
                                                norm_type=norm_type)
                                      for _ in range(n_layer)])
        # decoder head
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        self.ln_f = norm_layer(n_embed)
        self.head = nn.Linear(n_embed, output_dim, bias=False)

        self.block_size = block_size
        self.n_embed = n_embed
        self.n_layer = n_layer
        # print(f"particle transformer # parameters: {sum(p.numel() for p in self.parameters())}")

    def get_block_size(self):
        return self.block_size

    def init_weights(self):
        # initialize layers
        pass
        # self.apply(self._init_weights)
        # if self.positional_bias:
        #     for m in self.blocks:
        #         m.attn.rel_pos_bias.reset_parameters()

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        # elif isinstance(module, ParticleTransformer):
        #     if not self.positional_bias:
        #         torch.nn.init.normal_(module.pos_emb, mean=0.0, std=std)

    def forward(self, x):
        # x: [b, t, n, f]
        x = x.permute(0, 2, 1, 3)  # [b, n, t, f]
        b, n, t, f = x.size()
        # n is the number of particles
        assert t <= self.block_size, f"Cannot forward, model block size is exhausted: t:{t}, block_size: {self.block_size}"
        assert f == self.n_embed, "invalid particle feature dim"

        if self.n_registers > 0 and self.registers is not None:
            registers = self.registers.repeat(b, 1, t, 1)
            x = torch.cat([x, registers], dim=1)  # [b, n+n_reg, t, f]

        if not self.positional_bias:
            position_embeddings = self.pos_emb[:, None, :t, :]
            x = x + position_embeddings
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if self.n_registers > 0:
            logits, _ = logits.split([logits.shape[1] - self.n_registers, self.n_registers], dim=1)
        logits = logits.permute(0, 2, 1, 3)  # [b, t, n, f]

        return logits


class ParticleAttributesProjection(torch.nn.Module):
    def __init__(self, n_particles, in_features_dim, hidden_dim, output_dim, bg_features_dim, add_ctx_token=False,
                 base_dim=32, depth=True, obj_on=True, base_var=False, bg=True, activation='gelu', init_std=0.2,
                 cat_particle_num=False, norm_layer=True, particle_score=False,
                 mask_inputs=True, use_z_orig=False, obj_on_film=False, mask_obj_on=False):
        super().__init__()
        self.n_particles = n_particles
        self.in_features_dim = in_features_dim
        self.bg_features_dim = bg_features_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.with_depth = depth
        self.with_obj_on = obj_on
        self.with_var = base_var
        self.with_bg = bg
        self.with_score = particle_score
        self.add_ctx_token = add_ctx_token
        self.cat_particle_num = cat_particle_num
        self.norm_layer = norm_layer
        self.mask_inputs = mask_inputs
        self.mask_obj_on = mask_obj_on
        self.use_z_orig = use_z_orig
        self.obj_on_film = obj_on_film
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU
        # self.particle_dim = 2 + 2 + 2 + in_features_dim
        # [z, z_scale, z_features]
        self.base_dim = base_dim
        self.n_entities = 3
        if self.with_depth:
            self.n_entities += 1
        if self.with_obj_on and not self.obj_on_film:
            self.n_entities += 1
        if self.with_var:
            self.n_entities += 1
        if self.with_score:
            self.n_entities += 1
        if self.use_z_orig:
            self.n_entities += 1
        if self.cat_particle_num:
            self.n_entities += 1
            self.particle_num_embed = nn.Parameter(0.02 * torch.randn(1, self.n_particles, self.base_dim))
        self.particle_dim = self.base_dim * self.n_entities

        self.xy_projection = nn.Sequential(nn.Linear(2, hidden_dim),
                                           RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                           activation_f(),
                                           nn.Linear(hidden_dim, self.base_dim))
        self.scale_projection = nn.Sequential(nn.Linear(2, hidden_dim),
                                              RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                              activation_f(),
                                              nn.Linear(hidden_dim, self.base_dim))
        if self.with_var:
            self.var_projection = nn.Sequential(nn.Linear(5, hidden_dim),
                                                RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                activation_f(),
                                                nn.Linear(hidden_dim, self.base_dim))
        if self.with_obj_on:
            if self.obj_on_film:
                self.obj_on_projection = nn.Sequential(nn.Linear(1, hidden_dim),
                                                       activation_f(),
                                                       nn.Linear(hidden_dim, 2 * hidden_dim))
                nn.init.constant_(self.obj_on_projection[-1].weight, 0.0)
                nn.init.constant_(self.obj_on_projection[-1].bias[:hidden_dim], 1.0)
                nn.init.constant_(self.obj_on_projection[-1].bias[hidden_dim:], 0.0)
            else:
                self.obj_on_projection = nn.Sequential(nn.Linear(1, hidden_dim),
                                                       # ParticleNorm2(self.n_particles),
                                                       RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                       activation_f(),
                                                       nn.Linear(hidden_dim, self.base_dim))

            if self.mask_inputs:
                self.xy_mask = nn.Parameter(2 * torch.ones(2))
                self.scale_mask = nn.Parameter(0.1 * torch.ones(2))
                self.features_mask = nn.Parameter(init_std * torch.randn(in_features_dim))
                if self.mask_obj_on:
                    self.obj_on_mask = nn.Parameter(torch.zeros(1))
        if self.with_depth:
            self.depth_projection = nn.Sequential(nn.Linear(1, hidden_dim),
                                                  RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                  activation_f(),
                                                  nn.Linear(hidden_dim, self.base_dim))
            if self.with_obj_on and self.mask_inputs:
                self.depth_mask = nn.Parameter(init_std * torch.randn(1))
        self.features_projection = nn.Sequential(nn.Linear(in_features_dim, hidden_dim),
                                                 RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                 activation_f(),
                                                 nn.Linear(hidden_dim, self.base_dim))
        if self.with_score:
            self.score_projection = nn.Sequential(nn.Linear(1, hidden_dim),
                                                  RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                  activation_f(),
                                                  nn.Linear(hidden_dim, self.base_dim))
        if self.with_bg:
            self.bg_projection = nn.Sequential(nn.Linear(bg_features_dim, hidden_dim),
                                               RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                               activation_f(),
                                               nn.Linear(hidden_dim, output_dim))
        if self.use_z_orig:
            self.origin_projection = nn.Sequential(nn.Linear(4, hidden_dim),
                                                   RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                   activation_f(),
                                                   nn.Linear(hidden_dim, base_dim))
            if self.mask_inputs:
                self.orig_mask = nn.Parameter(2 * torch.ones(4))
        if self.obj_on_film:
            self.particle_projection_0 = nn.Sequential(nn.Linear(self.particle_dim, hidden_dim),
                                                       RMSNorm(hidden_dim))
            self.particle_projection = nn.Sequential(activation_f(),
                                                     nn.Linear(hidden_dim, output_dim))
        else:
            self.particle_projection = nn.Sequential(nn.Linear(self.particle_dim, hidden_dim),
                                                     RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                     activation_f(),
                                                     nn.Linear(hidden_dim, output_dim))
        if self.add_ctx_token:
            self.ctx_embedding = nn.Parameter(init_std * torch.randn(1, 1, 1, output_dim))

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None, z_score=None,
                z_orig=None):
        # def forward(self, z, z_scale, z_obj_on, z_features, z_base_var, z_bg_features):
        # z, z_scale, z_velocity: [bs, n_particles, 2]
        # z_depth, z_obj_on: [bs, n_particles, 1]
        # z_features: [bs, n_particles, in_features_dim]
        # z_bg_features: [bs, bg_features_dim]
        # z_context: [bs, context_dim]
        # bs, n_particles, feat_dim = z_features.shape

        # add origin and offset
        if self.use_z_orig and z_orig is not None:
            z_offset = z - z_orig
            z_orig_tot = torch.cat([z_orig, z_offset], dim=-1)
        else:
            z_orig_tot = z_orig

        if self.with_obj_on and self.mask_inputs:
            z_gate = torch.where(z_obj_on > 0.2, 1.0, 0.0)
            z = z_gate * z + (1 - z_gate) * self.xy_mask
            z_scale = z_gate * z_scale + (1 - z_gate) * self.scale_mask
            z_features = z_gate * z_features + (1 - z_gate) * self.features_mask
            if self.use_z_orig and z_orig is not None:
                z_orig_mask = self.orig_mask
                z_orig_tot = z_gate * z_orig_tot + (1 - z_gate) * z_orig_mask
            if self.mask_obj_on:
                z_obj_on = z_gate * z_obj_on + (1 - z_gate) * self.obj_on_mask

        z_proj = self.xy_projection(z)
        z_scale_proj = self.scale_projection(z_scale)
        z_features_proj = self.features_projection(z_features)
        z_all = torch.cat([z_proj, z_scale_proj, z_features_proj], dim=-1)
        if self.with_obj_on:
            z_obj_on_proj = self.obj_on_projection(z_obj_on)
            if not self.obj_on_film:
                z_all = torch.cat([z_all, z_obj_on_proj], dim=-1)
        if self.with_depth:
            if self.with_obj_on and self.mask_inputs:
                z_depth = z_gate * z_depth + (1 - z_gate) * self.depth_mask
            z_depth_proj = self.depth_projection(z_depth)
            z_all = torch.cat([z_all, z_depth_proj], dim=-1)
        if self.with_var and z_base_var is not None:
            z_var_proj = self.var_projection(z_base_var)
            z_all = torch.cat([z_all, z_var_proj], dim=-1)
        if self.with_score and z_score is not None:
            z_score_proj = self.score_projection(z_score)
            z_all = torch.cat([z_all, z_score_proj], dim=-1)
        if self.use_z_orig and z_orig is not None:
            z_orig_proj = self.origin_projection(z_orig_tot)
            z_all = torch.cat([z_all, z_orig_proj], dim=-1)
        if self.cat_particle_num:
            if len(z.shape) == 4:
                p_embed = self.particle_num_embed.unsqueeze(1).repeat(z.shape[0], z.shape[1], 1, 1)
            else:
                p_embed = self.particle_num_embed.repeat(z.shape[0], 1, 1)
            z_all = torch.cat([z_all, p_embed], dim=-1)

        # z_all: [bs, n_particles, 2 + 2 + in_features_dim]
        if self.with_obj_on and self.obj_on_film:
            oscale, oshift = z_obj_on_proj.chunk(2, dim=-1)
            z_all_proj = self.particle_projection(oscale * self.particle_projection_0(z_all) + oshift)
        else:
            z_all_proj = self.particle_projection(
                z_all)  # [bs, n_particles, output_dim]  or [bs, n_particle, hidden_dim]
        if self.with_bg:
            z_bg_features_proj = self.bg_projection(z_bg_features)  # [bs, output_dim]
            z_all_proj = torch.cat([z_all_proj, z_bg_features_proj.unsqueeze(-2)], dim=-2)
        # [bs, T,  n_particles + 1, output_dim]
        if self.add_ctx_token:
            z_all_proj = torch.cat([z_all_proj,
                                    self.ctx_embedding.repeat(z.shape[0], z.shape[1], 1, 1)], dim=-2)
            # [bs, T,  n_particles + 2, output_dim]
        return z_all_proj


class ParticleAttributeDecoder(nn.Module):
    def __init__(self, n_particles, input_dim, hidden_dim, features_dim, bg_features_dim=None,
                 depth=False, obj_on=False, features=False, bg_features=False,
                 offset_logvar=False,
                 activation='gelu', dropout=0.0, shared_logvar=False,
                 output_ctx_logvar=True, features_dist='gauss'):
        super().__init__()
        # decoder to map back from PTE's inner dim to the particle's original dimension
        self.n_particles = n_particles
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.features_dist = features_dist
        self.features_dim = features_dim
        self.bg_features_dim = bg_features_dim
        self.offset_logvar = offset_logvar
        self.with_depth = depth
        self.with_obj_on = obj_on
        self.with_features = features
        self.with_bg_features = bg_features
        self.use_fg_backbone = (self.with_obj_on or self.with_depth or self.with_features)
        self.shared_logvar = shared_logvar
        self.output_ctx_logvar = output_ctx_logvar
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU
        if self.use_fg_backbone:
            self.fg_backbone = nn.Identity()
            if self.with_obj_on:
                self.obj_on_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                 activation_f(),
                                                 nn.Linear(hidden_dim, 1)
                                                 )  # log_a, log_b
            if self.with_depth:
                self.depth_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                activation_f(),
                                                nn.Linear(hidden_dim, 2)
                                                )  # mu_z, logvar_z
            if self.with_features:
                output_feat_dim = 2 * features_dim if (features_dist != 'categorical') else features_dim
                self.features_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                   activation_f(),
                                                   nn.Linear(hidden_dim, output_feat_dim)
                                                   )  # mu_features, logvar_features
        if self.with_bg_features:
            output_bg_feat_dim = 2 * bg_features_dim if (features_dist != 'categorical') else bg_features_dim
            self.bg_backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                             activation_f(),
                                             )
            self.bg_features_head = nn.Linear(hidden_dim, output_bg_feat_dim)  # mu_features, logvar_features

        self.init_weights()

    def init_weights(self):
        if self.with_features and self.features_dist != 'categorical':
            nn.init.constant_(self.features_head[-1].weight[:self.features_dim], 0.0)
            nn.init.constant_(self.features_head[-1].bias[:self.features_dim], 0.0)
            nn.init.constant_(self.features_head[-1].weight[self.features_dim:], 0.0)
            nn.init.constant_(self.features_head[-1].bias[self.features_dim:], math.log(0.001 ** 2))
        if self.with_bg_features and self.features_dist != 'categorical':
            nn.init.constant_(self.bg_features_head.weight[:self.bg_features_dim], 0.0)
            nn.init.constant_(self.bg_features_head.bias[:self.bg_features_dim], 0.0)
            nn.init.constant_(self.bg_features_head.weight[self.bg_features_dim:], 0.0)
            nn.init.constant_(self.bg_features_head.bias[self.bg_features_dim:], math.log(0.001 ** 2))

    def forward(self, x):
        # x: [bs, n_particles, input_dim]
        # bs, n_particles, in_dim = x.shape
        bs, ts, n_particles = x.shape[0], x.shape[1], x.shape[2]
        # the following assumes fg_particles + bg_particle + context particle
        fg_particles = n_particles - 2 if self.with_bg_features else n_particles - 1
        if self.use_fg_backbone:
            x_fg = x[:, :, :fg_particles]
            fg_features = self.fg_backbone(x_fg)
            if self.with_depth:
                depth = self.depth_head(fg_features)
                mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)
            else:
                mu_depth = logvar_depth = None
            if self.with_obj_on:
                obj_on = self.obj_on_head(fg_features)
                lobj_on_a = lobj_on_b = obj_on
            else:
                lobj_on_a = lobj_on_b = None
            if self.with_features:
                features = self.features_head(fg_features)
                if self.features_dist != 'categorical':
                    mu_features, logvar_features = torch.chunk(features, 2, dim=-1)
                else:
                    mu_features = logvar_features = features
            else:
                mu_features = logvar_features = None
        else:
            mu_depth = logvar_depth = None
            lobj_on_a = lobj_on_b = None
            mu_features = logvar_features = None

        if self.with_bg_features:
            x_bg = x[:, :, fg_particles]
            bg_features = self.bg_backbone(x_bg)
            bg_features = self.bg_features_head(bg_features)
            if self.features_dist != 'categorical':
                mu_bg_features, logvar_bg_features = torch.chunk(bg_features, 2, dim=-1)
            else:
                mu_bg_features = logvar_bg_features = bg_features
        else:
            mu_bg_features = logvar_bg_features = None

        decoder_out = {'mu_depth': mu_depth, 'logvar_depth': logvar_depth,
                       'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b,
                       'mu_features': mu_features, 'logvar_features': logvar_features,
                       'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features}

        return decoder_out


class BgEncoder(nn.Module):
    def __init__(self, cdim=3, image_size=64, pad_mode='replicate', dropout=0.0,
                 learned_feature_dim=16, use_resblock=False, activation='gelu', cnn_mid_blocks=False,
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, interaction_features=False,
                 mlp_hidden_dim=256, timestep_horizon=1, add_particle_temp_embed=False, init_std=0.2,
                 features_dist='gauss', n_bg_categories=4, n_bg_classes=4,
                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_bg_std=0.005,  # std for conv bg normal dist (<fg -> prioritize fg in learning)
                 ):
        super(BgEncoder, self).__init__()
        """
        DLP Background Module -- encode a latent for the (masked) background, z_bg
        Basically, just a convolutional-based encoder used in standard VAEs
        cdim: channels of the input image (3...)
        enc_channels: channels for the posterior CNN (takes in the whole image)
        pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
        learned_feature_dim: the latent visual features dimensions extracted from glimpses.
        """
        self.image_size = image_size
        self.dropout = dropout
        self.output_feat_map_size = int(image_size // (2 ** (len(ch_mult) - 1)))
        self.features_dim = learned_feature_dim
        self.features_dist = features_dist
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.cdim = cdim
        self.n_kp_enc = final_cnn_ch
        self.interaction_features = interaction_features
        self.use_resblock = use_resblock
        self.activation = activation
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.add_particle_temp_embed = add_particle_temp_embed

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_bg_std = init_conv_bg_std  # std for conv bg normal dist

        attn_res = [max(self.image_size // 16, 1)]
        self.bg_cnn_enc = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                                  attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
                                  in_channels=self.cdim,
                                  resolution=self.image_size, z_channels=final_cnn_ch, double_z=False,
                                  padding_mode=pad_mode, residual=self.use_resblock, in_conv_kernel_size=3,
                                  mid_blocks=cnn_mid_blocks)
        self.cnn_out_shape = self.get_cnn_shape()

        # new cnn
        feature_map_size = self.output_feat_map_size ** 2
        output_logvar = (not self.interaction_features and self.features_dist != 'categorical')
        self.output_logvar = output_logvar

        # new - FCN
        if self.features_dim % feature_map_size == 0:
            self.ch_learned_feature_dim = math.ceil(max(self.features_dim / feature_map_size, 1))
            out_ch = 2 * self.ch_learned_feature_dim if output_logvar else self.ch_learned_feature_dim
            self.to_latent = nn.Conv2d(in_channels=final_cnn_ch,
                                       out_channels=out_ch, kernel_size=1)
            output_z_cnn = (self.ch_learned_feature_dim, self.cnn_out_shape[-2], self.cnn_out_shape[-1])
            flattened_z_cnn = np.prod(output_z_cnn)
            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(
                    init_std * torch.randn(1, self.timestep_horizon, final_cnn_ch, self.cnn_out_shape[-1],
                                           self.cnn_out_shape[-1]))
            else:
                self.temp_embed = None

            self.projection_mode = 'fcn'
            self.to_mu = nn.Identity()
            self.to_logvar = nn.Identity()
        else:
            self.ch_learned_feature_dim = final_cnn_ch
            self.to_latent = nn.Identity()
            output_z_cnn = (self.ch_learned_feature_dim, self.cnn_out_shape[-2], self.cnn_out_shape[-1])
            flattened_z_cnn = np.prod(output_z_cnn)

            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(init_std * torch.randn(1, self.timestep_horizon, flattened_z_cnn))
            else:
                self.temp_embed = None

            self.projection_mode = 'fc'
            self.to_mu = self.get_mlp(flattened_z_cnn, self.features_dim)
            self.to_logvar = self.get_mlp(flattened_z_cnn,
                                          self.features_dim) if output_logvar else nn.Identity()

        self.info = (f'BgEncoder: requested latent size: {self.features_dim}, '
                     f'cnn output (h*w): {feature_map_size}, (latent_size / h*w)={self.features_dim / feature_map_size} ->'
                     f' latent projection mode: {self.projection_mode},'
                     f' project {output_z_cnn} ({flattened_z_cnn}) -> {self.features_dim}')

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # pass
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0, self.init_conv_bg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # use pytorch's default
                pass
        # pass

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        else:
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.mlp_hidden_dim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                activation_f(),
                                nn.Linear(hidden_dim, out_dim))

            return mlp

    def get_cnn_shape(self):
        dummy_input = torch.rand(1, self.cdim, self.image_size, self.image_size)
        out = self.bg_cnn_enc(dummy_input)
        if isinstance(out, tuple):
            out = out[1]
        return out.shape[1:]

    def encode_bg_features(self, x, masks=None, timesteps=None):
        # x: [bs, ch, image_size, image_size]
        # masks: [bs, 1, image_size, image_size]
        batch_size, _, features_dim, _ = x.shape
        # bg features
        if masks is not None:
            x_in = x * masks
        else:
            x_in = x
        enc_out = self.bg_cnn_enc(x_in)
        if isinstance(enc_out, tuple):
            cnn_features = enc_out[1]
        else:
            cnn_features = enc_out

        # new cnn
        if self.projection_mode == 'fcn' and self.temp_embed is not None:
            orig_shape = cnn_features.shape  # [batch_size * n_kp, ch, patch_size, patch_size]
            new_feat = cnn_features.view(-1, timesteps, *cnn_features.shape[1:])
            new_feat = new_feat + self.temp_embed[:, :timesteps]
            cnn_features = new_feat.view(orig_shape)
        features = self.to_latent(cnn_features)
        features = features.view(features.shape[0], -1)
        if self.projection_mode == 'fc' and self.temp_embed is not None:
            orig_shape = features.shape  # [batch_size * n_kp, ch, patch_size, patch_size]
            new_feat = features.view(-1, timesteps, *features.shape[1:])
            new_feat = new_feat + self.temp_embed[:, :timesteps]
            features = new_feat.view(orig_shape)
        if self.interaction_features:
            mu_bg = features
            logvar_bg = None
            mu_bg = self.to_mu(mu_bg)
        else:
            mu_bg = self.to_mu(features)
            logvar_bg = self.to_logvar(features)

        return mu_bg, logvar_bg

    def encode_all(self, x, masks=None, deterministic=False, timesteps=None):
        # encode background
        mu_bg, logvar_bg = self.encode_bg_features(x, masks, timesteps)
        if self.interaction_features:
            z_bg = mu_bg
        else:
            z_bg = reparameterize(mu_bg, logvar_bg) if not deterministic else mu_bg
        z_kp = torch.zeros(mu_bg.shape[0], 1, 2, device=x.device, dtype=torch.float)
        encode_dict = {'mu_bg': mu_bg, 'logvar_bg': logvar_bg, 'z_bg': z_bg, 'z_kp': z_kp}
        return encode_dict

    def forward(self, x, masks=None, deterministic=False, timesteps=None):
        encoder_out = self.encode_all(x, masks, deterministic, timesteps)
        mu_bg = encoder_out['mu_bg']
        logvar_bg = encoder_out['logvar_bg']
        z_bg = encoder_out['z_bg']
        z_kp = encoder_out['z_kp']
        output_dict = {'mu_bg': mu_bg, 'logvar_bg': logvar_bg, 'z_bg': z_bg, 'z_kp': z_kp}
        return output_dict


class ParticleAttributeEncoder(nn.Module):
    """
    Glimpse-encoder: encodes patches visual features in a variational fashion (mu, log-variance).
    Useful for object-based scenes.
    """

    def __init__(self, anchor_size, image_size, n_particles, cnn_channels=(16, 16, 32), margin=0, ch=3, max_offset=1.0,
                 kp_activation='tanh', use_resblock=False, hidden_dim=512, pad_mode='replicate', depth=False,
                 obj_on=True, scale=True, activation='gelu',
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, cnn_mid_blocks=False,
                 timestep_horizon=1, add_particle_temp_embed=False, init_std=0.2,
                 obj_on_min=1e-4, obj_on_max=100.0,
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super().__init__()
        self.anchor_size = anchor_size
        self.channels = cnn_channels
        self.image_size = image_size
        self.n_particles = n_particles
        self.patch_size = np.round(anchor_size * (image_size - 1)).astype(int)
        self.margin = margin
        self.crop_size = self.patch_size + 2 * margin
        self.ch = ch
        self.use_resblock = use_resblock
        self.kp_activation = kp_activation
        self.max_offset = max_offset  # max offset of x-y, [-max_offset, +max_offset]
        self.hidden_dim = hidden_dim
        self.with_depth = depth
        self.with_obj_on = obj_on
        self.with_scale = scale
        self.cnn_mid_blocks = cnn_mid_blocks
        self.timestep_horizon = timestep_horizon
        self.add_particle_temp_embed = add_particle_temp_embed
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.init_std = init_std
        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        attn_res = [max(self.crop_size // 16, 1)]
        self.cnn = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True, in_channels=self.ch,
                           resolution=self.crop_size, z_channels=final_cnn_ch, double_z=False, padding_mode=pad_mode,
                           residual=self.use_resblock, mid_blocks=cnn_mid_blocks)

        feature_map_size = (self.crop_size // 2 ** (len(ch_mult) - 1)) ** 2
        fc_in_dim = final_cnn_ch * feature_map_size
        if self.add_particle_temp_embed and self.timestep_horizon > 1:
            self.temp_embed = nn.Parameter(init_std * torch.randn(1, self.timestep_horizon, 1, fc_in_dim))
        else:
            self.temp_embed = None
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU

        self.backbone = nn.Identity()
        self.xy_head = nn.Sequential(nn.Linear(fc_in_dim, self.hidden_dim),
                                     activation_f(),
                                     nn.Linear(self.hidden_dim, 4))  # mu_x, logvar_s, mu_y, logvar_y
        scale_output = 4 if self.with_scale else 2
        self.scale_xy_head = nn.Sequential(nn.Linear(fc_in_dim, self.hidden_dim),
                                           activation_f(),
                                           nn.Linear(self.hidden_dim,
                                                     scale_output))  # mu_sx, logvar_sx, mu_sy, logvar_sy
        if self.with_obj_on:
            self.obj_on_head = nn.Sequential(nn.Linear(fc_in_dim, self.hidden_dim),
                                             activation_f(),
                                             nn.Linear(self.hidden_dim, 1, bias=False))  # [log_obj_on_a, log_obj_on_b]

        else:
            self.obj_on_head = None
        if self.with_depth:
            self.depth_head = nn.Sequential(nn.Linear(fc_in_dim, self.hidden_dim),
                                            activation_f(),
                                            nn.Linear(self.hidden_dim, 2))  # mu_depth, logvar_depth
        else:
            self.depth_head = None
        self.init_weights()

    def init_weights(self):
        # pass
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # pass
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(self.xy_head[-1].weight[:2], 0.0)
        torch.nn.init.constant_(self.xy_head[-1].bias[:2], 0.0)
        torch.nn.init.constant_(self.xy_head[-1].weight[2:], 0.0)
        torch.nn.init.constant_(self.xy_head[-1].bias[2:], math.log(0.01 ** 2))
        if self.with_scale:
            torch.nn.init.constant_(self.scale_xy_head[-1].bias[:2], 0.0)
            torch.nn.init.constant_(self.scale_xy_head[-1].bias[2:], math.log(0.1 ** 2))
            torch.nn.init.constant_(self.scale_xy_head[-1].weight, 0.0)

        if self.with_obj_on:
            torch.nn.init.constant_(self.obj_on_head[-1].weight, 0.0)
            if self.obj_on_head[-1].bias is not None:
                torch.nn.init.constant_(self.obj_on_head[-1].bias, 0.0)  # beta(a,)

    def forward(self, x, kp, z_scale=None, timesteps=None, deterministic=False):
        # x: [bs, ch, image_size, image_size]
        # kp: [bs, n_kp, 2] in [-1, 1]
        batch_size, _, _, img_size = x.shape
        _, n_kp, _ = kp.shape
        x_repeated = x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1)  # [batch_size, n_kp, ch, image_size, image_size]
        x_repeated = x_repeated.view(-1, *x.shape[1:])  # [batch_size * n_kp, ch, image_size, image_size]
        if z_scale is None:
            z_scale = (self.patch_size / img_size) * torch.ones_like(kp)
        else:
            # assume unnormalized z_scale
            z_scale = torch.sigmoid(z_scale)
        z_pos = kp.reshape(-1, kp.shape[-1])
        z_scale = z_scale.view(-1, z_scale.shape[-1])
        out_dims = (batch_size * n_kp, x.shape[1], self.patch_size, self.patch_size)
        cropped_objects = spatial_transform(x_repeated, z_pos, z_scale, out_dims, inverse=False, padding_mode='border')
        # [batch_size * n_kp, ch, patch_size, patch_size]

        # encode objects - fc
        enc_out = self.cnn(cropped_objects)
        if isinstance(enc_out, tuple):
            cropped_objects_cnn = enc_out[1]
        else:
            cropped_objects_cnn = enc_out

        cropped_objects_flat = cropped_objects_cnn.reshape(batch_size, n_kp, -1)  # flatten
        # backbone features
        backbone_features = cropped_objects_flat
        # projection
        backbone_features = self.backbone(backbone_features)
        if timesteps is not None and self.temp_embed is not None:
            orig_shape = backbone_features.shape
            new_feat = backbone_features.view(-1, timesteps, *backbone_features.shape[1:]) + self.temp_embed[:,
            :timesteps]
            backbone_features = new_feat.view(orig_shape)

        if self.with_obj_on:
            obj_on_feat = backbone_features
            obj_on = self.obj_on_head(obj_on_feat)

            obj_on = obj_on.view(batch_size, n_kp, 1)
            lobj_on_a = lobj_on_b = obj_on
            obj_on_a_gate = lobj_on_a.sigmoid()
            obj_on_a = ((1 - obj_on_a_gate) * self.obj_on_min + obj_on_a_gate * self.obj_on_max).exp()
            obj_on_b_gate = 1 - (lobj_on_b * 0 + lobj_on_a).sigmoid()
            obj_on_b = ((1 - obj_on_b_gate) * self.obj_on_min + obj_on_b_gate * self.obj_on_max).exp()
            obj_on_beta_dist = torch.distributions.Beta(obj_on_a, obj_on_b)
            mu_obj_on = obj_on_beta_dist.mean
            if deterministic:
                z_obj_on = obj_on_beta_dist.mean
            else:
                z_obj_on = obj_on_beta_dist.rsample()
        else:
            lobj_on_a = lobj_on_b = obj_on = None
            obj_on_a = obj_on_b = z_obj_on = mu_obj_on = None

        xy = self.xy_head(backbone_features)
        xy = xy.view(batch_size, n_kp, -1)
        mu, logvar = torch.chunk(xy, chunks=2, dim=-1)

        scale_xy = self.scale_xy_head(backbone_features)
        scale_xy = scale_xy.view(batch_size, n_kp, -1)
        if self.with_scale:
            mu_scale, logvar_scale = torch.chunk(scale_xy, chunks=2, dim=-1)
        else:
            mu_scale = scale_xy
            logvar_scale = None

        if self.kp_activation == "tanh":
            mu = self.max_offset * torch.tanh(mu)
        elif self.kp_activation == "sigmoid":
            mu = self.max_offset * torch.sigmoid(mu)

        if self.with_depth:
            depth = self.depth_head(backbone_features)
            depth = depth.view(batch_size, n_kp, 2)
            mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)
        else:
            mu_depth = logvar_depth = None

        spatial_out = {'mu': mu, 'logvar': logvar, 'mu_scale': mu_scale, 'logvar_scale': logvar_scale,
                       'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b, 'obj_on': obj_on,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b,
                       'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on}
        return spatial_out


class ParticleFeaturesEncoder(nn.Module):
    """
    Glimpse-encoder: encodes patches visual features in a variational fashion (mu, log-variance).
    Useful for object-based scenes.
    """

    def __init__(self, anchor_size, features_dim, image_size, margin=0, ch=3,
                 use_resblock=False, hidden_dim=256, pad_mode='replicate', activation='gelu',
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, output_logvar=True,
                 cnn_mid_blocks=False, timestep_horizon=1, add_particle_temp_embed=False, init_std=0.2,
                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super().__init__()
        self.anchor_size = anchor_size
        self.image_size = image_size
        self.patch_size = np.round(anchor_size * (image_size - 1)).astype(int)
        self.margin = margin
        self.crop_size = self.patch_size + 2 * margin
        self.ch = ch
        self.use_resblock = use_resblock
        self.features_dim = features_dim
        self.output_logvar = output_logvar
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.cnn_mid_blocks = cnn_mid_blocks
        self.timestep_horizon = timestep_horizon
        self.add_particle_temp_embed = add_particle_temp_embed
        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        attn_res = [max(self.crop_size // 16, 1)]
        self.cnn = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True, in_channels=self.ch,
                           resolution=self.crop_size, z_channels=final_cnn_ch, double_z=False, padding_mode=pad_mode,
                           residual=self.use_resblock, mid_blocks=cnn_mid_blocks)

        self.cnn_out_shape = self.get_cnn_shape()
        feature_map_size = (self.crop_size // 2 ** (len(ch_mult) - 1)) ** 2
        # new - FCN
        if self.features_dim % feature_map_size == 0:
            self.ch_feature_dim = math.ceil(max(self.features_dim / feature_map_size, 1))
            z_out_channels = 2 * self.ch_feature_dim if self.output_logvar else self.ch_feature_dim
            self.to_latent = nn.Conv2d(in_channels=final_cnn_ch, out_channels=z_out_channels, kernel_size=1)
            output_z_cnn = (self.ch_feature_dim, self.cnn_out_shape[-2], self.cnn_out_shape[-1])
            flattened_z_cnn = np.prod(output_z_cnn)
            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(
                    init_std * torch.randn(1, self.timestep_horizon, 1, final_cnn_ch, self.cnn_out_shape[-1],
                                           self.cnn_out_shape[-1]))
            else:
                self.temp_embed = None

            self.projection_mode = 'fcn'
            self.to_mu = nn.Identity()
            self.to_logvar = nn.Identity()
        else:
            self.ch_feature_dim = final_cnn_ch
            self.to_latent = nn.Identity()
            output_z_cnn = (self.ch_feature_dim, self.cnn_out_shape[-2], self.cnn_out_shape[-1])
            flattened_z_cnn = np.prod(output_z_cnn)
            self.projection_mode = 'fc'
            self.to_mu = self.get_mlp(flattened_z_cnn, self.features_dim)
            self.to_logvar = self.get_mlp(flattened_z_cnn, self.features_dim) if self.output_logvar else nn.Identity()
            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(init_std * torch.randn(1, self.timestep_horizon, 1, flattened_z_cnn))
            else:
                self.temp_embed = None
        self.init_weights()

        self.info = (f'ParticleFeaturesEncoder: requested latent size: {self.features_dim}, '
                     f'cnn output (h*w): {feature_map_size}, (latent_size / h*w)={self.features_dim / feature_map_size} ->'
                     f' latent projection mode: {self.projection_mode},'
                     f' project {output_z_cnn} ({flattened_z_cnn}) -> {self.features_dim}')

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        else:
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.hidden_dim

            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                activation_f(),
                                nn.Linear(hidden_dim, out_dim))
            return mlp

    def get_cnn_shape(self):
        dummy_input = torch.rand(1, self.ch, self.patch_size, self.patch_size)
        out = self.cnn(dummy_input)
        if isinstance(out, tuple):
            out = out[1]
        return out.shape[1:]

    def forward(self, x, kp, z_scale=None, timesteps=None, obj_on=None):
        # x: [bs, ch, image_size, image_size]
        # kp: [bs, n_kp, 2] in [-1, 1]
        batch_size = x.shape[0]
        n_kp = kp.shape[1]
        img_size = x.shape[-1]
        x_repeated = x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1)  # [batch_size, n_kp, ch, image_size, image_size]
        x_repeated = x_repeated.view(-1, *x.shape[1:])  # [batch_size * n_kp, ch, image_size, image_size]
        if z_scale is None:
            z_scale = (self.patch_size / img_size) * torch.ones_like(kp)
        else:
            # assume unnormalized z_scale
            z_scale = torch.sigmoid(z_scale)
        z_pos = kp.reshape(-1, kp.shape[-1])
        z_scale = z_scale.view(-1, z_scale.shape[-1])
        out_dims = (batch_size * n_kp, x.shape[1], self.patch_size, self.patch_size)
        cropped_objects = spatial_transform(x_repeated, z_pos, z_scale, out_dims, inverse=False, padding_mode='border')
        # [batch_size * n_kp, ch, patch_size, patch_size]

        # encode objects - fc
        enc_out = self.cnn(cropped_objects)
        if isinstance(enc_out, tuple):
            cropped_objects_cnn = enc_out[1]
        else:
            cropped_objects_cnn = enc_out

        if obj_on is not None:
            obj_on = obj_on.view(-1)
            cropped_objects_cnn = cropped_objects_cnn * obj_on[:, None, None, None]

        # new with cnn
        if self.projection_mode == 'fcn' and self.temp_embed is not None:
            orig_shape = cropped_objects_cnn.shape  # [batch_size * n_kp, ch, patch_size, patch_size]
            new_feat = cropped_objects_cnn.view(-1, timesteps, n_kp, *cropped_objects_cnn.shape[1:])
            new_feat = new_feat + self.temp_embed[:, :timesteps]
            cropped_objects_cnn = new_feat.view(orig_shape)

        features = self.to_latent(cropped_objects_cnn)
        features = features.view(batch_size, n_kp, -1)
        if self.projection_mode == 'fc' and self.temp_embed is not None:
            orig_shape = features.shape  # [batch_size * n_kp, ch, patch_size, patch_size]
            new_feat = features.view(-1, timesteps, n_kp, *features.shape[2:])
            new_feat = new_feat + self.temp_embed[:, :timesteps]
            features = new_feat.view(orig_shape)
        if self.output_logvar:
            mu_features = self.to_mu(features)
            logvar_features = self.to_logvar(features)
        else:
            mu_features = features
            mu_features = self.to_mu(mu_features)
            logvar_features = None

        cropped_objects = cropped_objects.view(batch_size, -1, *cropped_objects.shape[1:])
        # [batch_size, n_kp, ch, crop_size, crop_size]
        spatial_out = {'mu_features': mu_features, 'logvar_features': logvar_features,
                       'cropped_objects': cropped_objects}
        return spatial_out


class DLPPrior(nn.Module):
    def __init__(self, cdim=3, image_size=64, n_kp=1,
                 pad_mode='replicate',
                 patch_size=16, n_kp_prior=64,
                 kp_range=(-1, 1),
                 use_resblock=False,
                 filtering_heuristic='none',
                 ch_mult=(1, 2, 3), base_ch=32, num_res_blocks=2, cnn_mid_blocks=False,
                 init_zero_bias=True,
                 init_ssm_last_layer=True,  # spatial softmax initialization
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super(DLPPrior, self).__init__()
        """
        DLP Prior Module -- extract object location proposals from an image via SSM
        cdim: channels of the input image (3...)
        prior_channels: channels for prior CNN (takes in patches)
        n_kp: number of kp to extract from each (!) patch
        n_kp_prior: number of kp to filter from the set of prior kp (of size n_kp x num_patches)
        pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
        patch_size: patch size for the prior KP proposals network (not to be confused with the glimpse size)
        kp_range: the range of keypoints, can be [-1, 1] (default) or [0,1]
        kp_activation: the type of activation to apply on the keypoints: "tanh" for kp_range [-1, 1], "sigmoid" for [0, 1]
        filtering heuristic: filtering heuristic to filter prior keypoints,['distance', 'variance', 'random', 'none']
        """
        self.image_size = image_size
        self.kp_range = kp_range
        self.num_patches = int((image_size // patch_size) ** 2)
        self.n_kp = n_kp
        self.n_kp_total = self.n_kp * self.num_patches
        self.n_kp_prior = min(self.n_kp_total, n_kp_prior)
        self.patch_size = patch_size
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.cnn_mid_blocks = cnn_mid_blocks
        assert filtering_heuristic in ['distance', 'variance',
                                       'random', 'none'], f'unknown filtering heuristic: {filtering_heuristic}'
        self.filtering_heuristic = filtering_heuristic

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_ssm_last_layer = init_ssm_last_layer  # spatial softmax initialization
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        # prior
        self.patcher = ImagePatcher(cdim=cdim, image_size=image_size, patch_size=patch_size)
        # self.features_dim = int(patch_size // (2 ** (len(prior_channels) - 1)))
        self.features_dim = int(patch_size // (2 ** (len(ch_mult) - 1)))
        attn_res = [max(self.patch_size // 16, 1)]
        self.enc = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True, in_channels=cdim,
                           resolution=patch_size, z_channels=n_kp, double_z=False, padding_mode='replicate',
                           residual=self.use_resblock, mid_blocks=cnn_mid_blocks)

        self.ssm = AlternativeSpatialSoftmaxKP(kp_range=kp_range)

        self.init_weights()

    def init_conv_with_spatial_priors(self, conv: nn.Conv2d, gaussian_sigma=0.4, noise_std=0.05):
        """
        Initializes a conv layer with spatially structured filters for RGB or single-channel input.
        Supports Sobel, Prewitt, Laplacian, and Gaussian blobs with noise.
        """
        out_channels, in_channels, H, W = conv.weight.shape

        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.T

        prewitt_x = torch.tensor([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]], dtype=torch.float32)
        prewitt_y = prewitt_x.T

        laplacian = torch.tensor([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]], dtype=torch.float32)

        edge_filters = [sobel_x, sobel_y, prewitt_x, prewitt_y, laplacian]

        def make_gaussian_blob(size, sigma, center):
            x = torch.linspace(-1, 1, size)
            y = torch.linspace(-1, 1, size)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack([xx, yy], dim=0)
            diff = grid - torch.tensor(center).view(2, 1, 1)
            return torch.exp(-torch.sum(diff ** 2, dim=0) / (2 * sigma ** 2))

        weight = torch.zeros_like(conv.weight)

        for i in range(out_channels):
            filter_type = i % (len(edge_filters) + 1)
            if filter_type < len(edge_filters):
                ef = edge_filters[filter_type]
                ef_padded = torch.zeros((H, W))
                h0 = (H - ef.shape[0]) // 2
                w0 = (W - ef.shape[1]) // 2
                ef_padded[h0:h0 + ef.shape[0], w0:w0 + ef.shape[1]] = ef
                base_filter = ef_padded
            else:
                cx, cy = np.random.uniform(-0.5, 0.5, size=2)
                base_filter = make_gaussian_blob(H, gaussian_sigma, (cx, cy))

            noisy_filter = base_filter + noise_std * torch.randn_like(base_filter)

            # Apply the same (or slightly varied) filter across input channels
            for c in range(in_channels):
                # Option 1: same for all channels
                weight[i, c] = noisy_filter.clone()
                # Option 2 (optional): add slight channel-specific noise
                # weight[i, c] = noisy_filter + noise_std * torch.randn_like(noisy_filter)

        with torch.no_grad():
            conv.weight.copy_(weight)
            if conv.bias is not None:
                conv.bias.zero_()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # pass
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # use pytorch's default
                pass
        # initialize input filters with spatial priors
        if self.init_ssm_last_layer:
            m = self.enc.conv_out
            # nn.init.normal_(m.weight, -0.2, 0.02)
            # d = -1 * math.sqrt(1 / (m.in_channels + m.out_channels))
            d = -1.0 * self.init_conv_fg_std
            nn.init.constant_(m.weight, d)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # self.init_conv_with_spatial_priors(self.enc.conv_in)

    def img_to_patches(self, x):
        return self.patcher.img_to_patches(x)

    def patches_to_img(self, x):
        return self.patcher.patches_to_img(x)

    def get_global_kp(self, local_kp):
        # local_kp: [batch_size, num_patches, n_kp, 2]
        # returns the global coordinates of a KP within the original image.
        batch_size, num_patches, n_kp, _ = local_kp.shape
        global_coor = self.patcher.get_patch_location_idx().to(local_kp.device)  # [num_patches, 2]
        global_coor = global_coor[:, None, :].repeat(1, n_kp, 1)
        global_coor = (((local_kp - self.kp_range[0]) / (self.kp_range[1] - self.kp_range[0])) * (
                self.patcher.patch_size - 1) + global_coor) / (self.image_size - 1)
        global_coor = global_coor * (self.kp_range[1] - self.kp_range[0]) + self.kp_range[0]
        return global_coor

    def get_patch_centers(self):
        # get the resepective coordinates of the patches
        centers = self.patcher.get_patch_centers() / (self.image_size - 1)
        return centers

    def get_distance_from_patch_centers(self, kp, global_kp=False):
        # calculates the distance of a KP from the center of its parent patch. This is useful to understand (and filter)
        # if SSM detected something, otherwise, the KP will probably land in the center of the patch
        # (e.g., a solid-color patch will have the same activation in all pixels).
        if not global_kp:
            global_coor = self.get_global_kp(kp).view(kp.shape[0], -1, 2)
        else:
            global_coor = kp
        centers = 0.5 * (self.kp_range[1] + self.kp_range[0]) * torch.ones_like(kp).to(kp.device)
        global_centers = self.get_global_kp(centers.view(kp.shape[0], -1, self.n_kp, 2)).view(kp.shape[0], -1, 2)
        return ((global_coor - global_centers) ** 2).sum(-1)

    def encode_prior(self, x, filtering_heuristic='none', k=None):
        # encodes prior keypoints by patchifying the image and applying spatial-softmax
        # x: [batch_size, cdim, image_size, image_size]
        # global_kp: set True to get the global coordinates within the image (instead of local KP inside the patch)
        batch_size, cdim, image_size, image_size = x.shape
        x_patches = self.img_to_patches(x)  # [batch_size, cdim, num_patches, patch_size, patch_size]
        x_patches = x_patches.permute(0, 2, 1, 3, 4)  # [batch_size, num_patches, cdim, patch_size, patch_size]
        x_patches = x_patches.contiguous().view(-1, cdim, self.patcher.patch_size, self.patcher.patch_size)
        enc_out = self.enc(x_patches)  # [batch_size*num_patches, n_kp, features_dim, features_dim]
        if isinstance(enc_out, tuple):
            z = enc_out[1]
        else:
            z = enc_out
        kp_p, var_kp = self.ssm(z, probs=False, variance=True)  # [batch_size * num_patches, n_kp, 2]
        kp_p = kp_p.view(batch_size, -1, self.n_kp, 2)  # [batch_size, num_patches, n_kp, 2]
        kp_p = self.get_global_kp(kp_p)
        var_kp = var_kp.view(batch_size, kp_p.shape[1], self.n_kp, -1)  # [batch_size, num_patches, n_kp, 3]

        if k is None:
            k = self.n_kp_prior
        kp_p = kp_p.view(x.shape[0], -1, 2)  # [batch_size, n_kp_total, 2]
        var_kp = var_kp.view(x.shape[0], kp_p.shape[1], -1)  # [batch_size, n_kp_total, 3]
        if filtering_heuristic == 'distance':
            # filter proposals by distance to the patches' center
            dist_from_center = self.prior.get_distance_from_patch_centers(kp_p, global_kp=True)
            _, indices = torch.topk(dist_from_center, k=k, dim=-1, largest=True)
            batch_indices = torch.arange(kp_p.shape[0], device=kp_p.device).view(-1, 1)
            kp_p = kp_p[batch_indices, indices]
            var_kp = var_kp[batch_indices, indices]
        elif filtering_heuristic == 'variance':
            total_var = var_kp.sum(-1)
            _, indices = torch.topk(total_var, k=k, dim=-1, largest=False)
            batch_indices = torch.arange(kp_p.shape[0], device=kp_p.device).view(-1, 1)
            kp_p = kp_p[batch_indices, indices]
        elif filtering_heuristic == 'none':
            return kp_p, var_kp
        else:
            # alternatively, just sample random kp
            kp_p = kp_p[:, torch.randperm(kp_p.shape[1])[:k]]
            var_kp = var_kp[:, torch.randperm(kp_p.shape[1])[:k]]
        return kp_p, var_kp

    def forward(self, x):
        # prior proposals
        kp_p, var_kp = self.encode_prior(x, filtering_heuristic=self.filtering_heuristic)
        return kp_p, var_kp


class ParticleInteractionEncoder(nn.Module):
    def __init__(self, n_kp_enc, dropout=0.0, learned_feature_dim=16, learned_bg_feature_dim=16, embed_init_std=0.2,
                 projection_dim=128, timestep_horizon=1, pte_layers=1, pte_heads=1,
                 attn_norm_type='rms', hidden_dim=256, use_resblock=True, pad_mode='replicate',
                 temporal_interaction=True, interaction_depth=False, interaction_obj_on=False, activation='gelu',
                 scale_anchor=None,
                 interaction_features=False, ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, cdim=3,
                 image_size=64, n_views=1, bg=True, use_img_input=True, cnn_mid_blocks=False,
                 particle_positional_embed=True,
                 particle_score=False, norm_layer=True, add_particle_temp_embed=False,
                 features_dist='gauss', n_fg_categories=8, n_fg_classes=4, n_bg_categories=4, n_bg_classes=4,
                 obj_on_min=1e-4, obj_on_max=100.0,
                 particle_anchors=None, use_z_orig=False,
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super(ParticleInteractionEncoder, self).__init__()
        """
        DLP Foreground Module -- extract objects from an image

        """
        self.n_kp_enc = n_kp_enc
        self.dropout = dropout
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.attn_norm_type = attn_norm_type
        self.hidden_dim = hidden_dim
        self.temporal_interaction = temporal_interaction
        self.interaction_depth = interaction_depth
        self.interaction_obj_on = interaction_obj_on
        self.interaction_features = interaction_features
        self.with_bg = bg
        self.use_img_input = use_img_input
        self.activation = activation
        self.cnn_mid_blocks = cnn_mid_blocks
        self.particle_score = particle_score
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.add_particle_temp_embed = add_particle_temp_embed
        self.scale_anchor = scale_anchor
        self.use_z_orig = use_z_orig
        self.n_views = n_views

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        if particle_anchors is None:
            self.register_buffer('particles_anchor', torch.zeros(1, 1, self.n_kp_enc))
            self.use_z_orig = False
        else:
            self.register_buffer('particles_anchor', particle_anchors)

        n_particles = self.n_kp_enc  # [n_kp_enc]

        if self.use_img_input:
            # cnn stuff
            self.ctx_pre_pte_latent_dim = projection_dim  # can also be ctx dim
            self.image_size = image_size
            self.output_feat_map_size = int(image_size // (2 ** (len(ch_mult) - 1)))
            self.cdim = cdim

            attn_res = [max(self.image_size // 16, 1)]
            self.ctx_cnn_enc = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                                       attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
                                       in_channels=self.cdim,
                                       resolution=self.image_size, z_channels=final_cnn_ch, double_z=False,
                                       padding_mode=pad_mode, residual=use_resblock, in_conv_kernel_size=3,
                                       mid_blocks=cnn_mid_blocks)
            self.cnn_out_shape = self.get_cnn_shape()
            feature_map_size = self.output_feat_map_size ** 2

            # FCN or Linear
            if self.ctx_pre_pte_latent_dim % feature_map_size == 0:
                self.ch_learned_feature_dim = math.ceil(max(self.ctx_pre_pte_latent_dim / feature_map_size, 1))
                out_ch = self.ch_learned_feature_dim
                self.to_latent = nn.Conv2d(in_channels=final_cnn_ch,
                                           out_channels=out_ch, kernel_size=1)
                output_z_cnn = (self.ch_learned_feature_dim, self.cnn_out_shape[-2], self.cnn_out_shape[-1])
                flattened_z_cnn = np.prod(output_z_cnn)

                self.projection_mode = 'fcn'
                self.to_latent_lin = nn.Identity()
            else:
                self.ch_learned_feature_dim = final_cnn_ch
                self.to_latent = nn.Identity()
                output_z_cnn = (self.ch_learned_feature_dim, self.cnn_out_shape[-2], self.cnn_out_shape[-1])
                flattened_z_cnn = np.prod(output_z_cnn)

                self.projection_mode = 'fc'
                self.to_latent_lin = self.get_mlp(flattened_z_cnn, self.ctx_pre_pte_latent_dim)

            self.info = (f'ParticleInteractionEncoder: requested latent size: {self.ctx_pre_pte_latent_dim}, '
                         f'cnn output (h*w): {feature_map_size}, (latent_size / h*w)={self.ctx_pre_pte_latent_dim / feature_map_size} ->'
                         f' latent projection mode: {self.projection_mode},'
                         f' project {output_z_cnn} ({flattened_z_cnn}) -> {self.ctx_pre_pte_latent_dim}')

            # end cnn stuff
            n_particles += 1  # [ctx + n_kp_enc]
            self.ctx_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
        else:
            self.info = f'ParticleInteractionEncoder: not using image as input context'
        if self.with_bg:
            n_particles += 1
            self.bg_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))

        # entities positional embeddings
        if particle_positional_embed:
            self.particle_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.n_kp_enc, projection_dim))
        else:
            self.particle_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))

        # interaction encoder
        self.basic_particle_proj = ParticleAttributesProjection(n_particles=self.n_kp_enc,
                                                                in_features_dim=self.learned_feature_dim,
                                                                hidden_dim=self.hidden_dim,
                                                                output_dim=projection_dim,
                                                                bg_features_dim=self.learned_bg_feature_dim,
                                                                add_ctx_token=False,
                                                                depth=not self.interaction_depth,
                                                                obj_on=not self.interaction_obj_on,
                                                                base_var=False, bg=self.with_bg,
                                                                particle_score=self.particle_score,
                                                                norm_layer=norm_layer,
                                                                use_z_orig=self.use_z_orig)
        if self.add_particle_temp_embed and not self.temporal_interaction:
            self.temp_embed = nn.Parameter(
                self.embed_init_std * torch.randn(1, self.timestep_horizon, 1, projection_dim))
        else:
            self.temp_embed = None

        if self.n_views > 1:
            self.view_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.n_views, 1, projection_dim))
        else:
            self.view_embeddings = None

        block_size = self.timestep_horizon if self.temporal_interaction else 1
        self.pte = ParticleSelfAttTransformer(n_embed=self.projection_dim, n_head=pte_heads,
                                              n_layer=pte_layers,
                                              block_size=block_size,
                                              output_dim=self.projection_dim, attn_pdrop=dropout,
                                              resid_pdrop=dropout,
                                              hidden_dim_multiplier=4, positional_bias=False,
                                              activation=activation,
                                              max_particles=None, norm_type=attn_norm_type,
                                              init_std=embed_init_std)

        self.particle_decoder = ParticleAttributeDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                         hidden_dim=self.hidden_dim,
                                                         features_dim=learned_feature_dim,
                                                         bg_features_dim=learned_bg_feature_dim,
                                                         depth=self.interaction_depth,
                                                         obj_on=self.interaction_obj_on,
                                                         features=self.interaction_features,
                                                         bg_features=(self.interaction_features and self.with_bg),
                                                         features_dist=self.features_dist)
        self.init_weights()

    def init_weights(self):
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.particle_decoder.init_weights()
        self.pte.init_weights()

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        else:
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.hidden_dim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                activation_f(),
                                nn.Linear(hidden_dim, out_dim))
            return mlp

    def get_cnn_shape(self):
        dummy_input = torch.rand(1, self.cdim, self.image_size, self.image_size)
        out = self.ctx_cnn_enc(dummy_input)
        if isinstance(out, tuple):
            out = out[1]
        return out.shape[1:]

    def encode_ctx_features(self, x, masks=None):
        # x: [bs, ch, image_size, image_size]
        # masks: [bs, 1, image_size, image_size]
        batch_size, _, features_dim, _ = x.shape
        # bg features
        if masks is not None:
            x_in = x * masks
        else:
            x_in = x
        enc_out = self.ctx_cnn_enc(x_in)
        if isinstance(enc_out, tuple):
            cnn_features = enc_out[1]
        else:
            cnn_features = enc_out

        # new cnn
        features = self.to_latent(cnn_features)
        features = features.view(features.shape[0], -1)
        features = self.to_latent_lin(features)
        return features

    def encode_all(self, x, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None,
                   z_score=None, patch_id_embed=None, deterministic=False, warmup=False,
                   detach_before_proj=False):
        """
        output order:
        if with_bg and ctx_pool_mode='token': [n_particles, bg, ctx, ctx_token*]
        else: [n_particles, ctx, ctx_token*]
        """
        # x: [bs * n_views, t, ch, h, w]
        bs, timestep_horizon = z.shape[0], z.shape[1]
        z_v = z.detach() if detach_before_proj else z
        z_scale_v = z_scale.detach() if detach_before_proj else z_scale
        z_obj_on_v = z_obj_on.detach() if (z_obj_on is not None and detach_before_proj) else z_obj_on
        z_depth_v = z_depth.detach() if (z_depth is not None and detach_before_proj) else z_depth
        z_features_v = z_features.detach() if detach_before_proj else z_features
        if not self.with_bg:
            z_bg_features = None
        z_bg_features_v = z_bg_features.detach() if (
                z_bg_features is not None and detach_before_proj) else z_bg_features
        z_base_var_v = z_base_var.detach() if z_base_var is not None else z_base_var
        z_score_v = z_score.detach() if z_score is not None else z_score
        if self.use_z_orig:
            z_orig_v = self.particles_anchor.unsqueeze(0).repeat(z_v.shape[0], z_v.shape[1], 1, 1)
        else:
            z_orig_v = None

        particle_projection = self.basic_particle_proj(z=z_v,
                                                       z_scale=z_scale_v,
                                                       z_obj_on=z_obj_on_v,
                                                       z_depth=z_depth_v,
                                                       z_features=z_features_v,
                                                       z_bg_features=z_bg_features_v,
                                                       z_base_var=z_base_var_v,
                                                       z_score=z_score_v,
                                                       z_orig=z_orig_v)
        # add entity pos embeddings
        if self.particle_embeddings.shape[2] == 1:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, z.shape[2], 1)
        else:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, 1, 1)
        if patch_id_embed is not None:
            p_embeddings = p_embeddings + patch_id_embed
        if self.with_bg:
            bg_embeddings = self.bg_embeddings.repeat(bs, timestep_horizon, 1, 1)
            p_embeddings = torch.cat([p_embeddings, bg_embeddings], dim=2)
        particle_projection = particle_projection + p_embeddings

        if self.use_img_input:
            # add context
            if len(x.shape) == 5:
                # x: [bs, t, ch, h, w]
                x_in = x.view(-1, *x.shape[2:])
            else:
                x_in = x
            ctx_features = self.encode_ctx_features(x_in)
            ctx_features = ctx_features.view(bs, timestep_horizon, 1, -1)  # [bs, T, 1, projection_dim]
            ctx_features = ctx_features + self.ctx_embeddings.repeat(bs, timestep_horizon, 1, 1)
            particle_projection = torch.cat([particle_projection, ctx_features], dim=2)
            # [bs, t, n_p + 2, proj_dim] if with_bg else [bs, t, n_p + 1, proj_dim]
        #     # [bs, t, n_p + 2, proj_dim]

        if self.n_views > 1:
            # [bs * n_views, t, n, d] -> [bs, t, n_views, n, d] -> [bs, t, n_views * n, d]
            particle_projection = particle_projection.view(-1, self.n_views, *particle_projection.shape[1:])
            particle_projection = particle_projection.permute(0, 2, 1, 3, 4)  # [bs, t, n_views, n, d]
            particle_projection = particle_projection + self.view_embeddings
            particle_projection = particle_projection.reshape(particle_projection.shape[0],
                                                              particle_projection.shape[1],
                                                              -1,
                                                              particle_projection.shape[-1])  # [bs, t, n_views * n, d]

        if timestep_horizon > 1 and not self.temporal_interaction:
            if self.add_particle_temp_embed:
                particle_projection = particle_projection + self.temp_embed[:, :timestep_horizon]
            particle_projection = particle_projection.view(-1, 1, *particle_projection.shape[2:])
            # [bs * ts, 1, n, f]
        particles_out = self.pte(particle_projection)
        particles_out = particles_out.view(-1, timestep_horizon, *particles_out.shape[2:])
        # [bs, ts, n, f]
        if self.n_views > 1:
            # [bs, t, n_views * n, d] -> [bs * n_views, t, n, d]
            particles_out = particles_out.view(particles_out.shape[0], timestep_horizon, self.n_views, -1,
                                               particles_out.shape[-1])
            particles_out = particles_out.permute(0, 2, 1, 3, 4)
            particles_out = particles_out.reshape(-1, *particles_out.shape[2:])
        particle_decoder_out = self.particle_decoder(particles_out)  # [bs * n_views, t, n, d]
        # unpack
        mu_depth = particle_decoder_out['mu_depth']
        logvar_depth = particle_decoder_out['logvar_depth']
        if self.interaction_depth:
            z_depth = reparameterize(mu_depth, logvar_depth) if not deterministic else mu_depth
        else:
            z_depth = None
        mu_features = particle_decoder_out['mu_features']
        logvar_features = particle_decoder_out['logvar_features']
        mu_bg_features = particle_decoder_out['mu_bg_features']
        logvar_bg_features = particle_decoder_out['logvar_bg_features']
        if self.interaction_features:
            mu_features = z_features + mu_features
            if self.features_dist == 'categorical':
                logits = mu_features.view(*mu_features.shape[:-1], self.n_fg_categories, self.n_fg_classes)
                # [bs, T, n_p, n_categories, n_classes]
                probs = logits.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                if deterministic:
                    samples = torch.argmax(probs.view(-1, probs.shape[-1]), dim=-1, keepdim=True)
                    samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                    samples = samples.view(probs.shape)
                    # straight-through
                    z_features = samples.detach() + (probs - probs.detach())
                    z_features = z_features.view(*mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]
                else:
                    samples = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
                    samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                    samples = samples.view(probs.shape)
                    # straight-through
                    z_features = samples.detach() + (probs - probs.detach())
                    z_features = z_features.view(*mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]
            else:
                # logvar_features = logvar_features.clamp_max(math.log(0.2 ** 2))
                z_features = reparameterize(mu_features, logvar_features) if not deterministic else mu_features
            if self.with_bg:
                mu_bg_features = z_bg_features + mu_bg_features
                if self.features_dist == 'categorical':
                    logits_bg = mu_bg_features.view(*mu_bg_features.shape[:-1], self.n_bg_categories, self.n_bg_classes)
                    # [bs, T, n_p, n_categories, n_classes]
                    probs_bg = logits_bg.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                    if deterministic:
                        samples_bg = torch.argmax(probs_bg.view(-1, probs_bg.shape[-1]), dim=-1, keepdim=True)
                        samples_bg = F.one_hot(samples_bg.squeeze(-1), num_classes=self.n_bg_classes)
                        samples_bg = samples_bg.view(probs_bg.shape)
                        # straight-through
                        z_bg_features = samples_bg.detach() + (probs_bg - probs_bg.detach())
                        z_bg_features = z_bg_features.view(
                            *mu_bg_features.shape)  # [bs, T, n_p, n_categories * n_classes]
                    else:
                        samples_bg = torch.multinomial(probs_bg.view(-1, probs_bg.shape[-1]), num_samples=1)
                        samples_bg = F.one_hot(samples_bg.squeeze(-1), num_classes=self.n_bg_classes)
                        samples_bg = samples_bg.view(probs_bg.shape)
                        # straight-through
                        z_bg_features = samples_bg.detach() + (probs_bg - probs_bg.detach())
                        z_bg_features = z_bg_features.view(
                            *mu_bg_features.shape)  # [bs, T, n_p, n_categories * n_classes]
                else:
                    # logvar_bg_features = logvar_bg_features.clamp_max(math.log(0.2 ** 2))
                    z_bg_features = reparameterize(mu_bg_features,
                                                   logvar_bg_features) if not deterministic else mu_bg_features
        else:
            z_features = z_bg_features = None
        lobj_on_a = particle_decoder_out['lobj_on_a']
        lobj_on_b = particle_decoder_out['lobj_on_b']
        if self.interaction_obj_on:
            obj_on_a_gate = (lobj_on_a).sigmoid()
            obj_on_a = ((1 - obj_on_a_gate) * self.obj_on_min + obj_on_a_gate * self.obj_on_max).exp()
            obj_on_b_gate = 1 - (lobj_on_b * 0 + lobj_on_a).sigmoid()
            obj_on_b = ((1 - obj_on_b_gate) * self.obj_on_min + obj_on_b_gate * self.obj_on_max).exp()
            obj_on_beta_dist = torch.distributions.Beta(obj_on_a, obj_on_b)
            mu_obj_on = obj_on_beta_dist.mean
            z_obj_on = obj_on_beta_dist.rsample() if not deterministic else obj_on_beta_dist.mean
        else:
            obj_on_a = obj_on_b = z_obj_on = mu_obj_on = None

        encode_dict = {'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
                       'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
                       'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
                       'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features,
                       'z_bg_features': z_bg_features, 'z_scale': z_scale, 'z': z}
        return encode_dict

    def forward(self, x, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None, z_score=None,
                patch_id_embed=None, deterministic=False, warmup=False):
        output_dict = self.encode_all(x, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_base_var, z_score,
                                      patch_id_embed, deterministic=deterministic, warmup=warmup)
        return output_dict


class ParticleEncoder(nn.Module):
    def __init__(self, cdim=3, image_size=64,
                 pad_mode='replicate', dropout=0.0, n_kp_per_patch=1, n_kp_prior=20,
                 patch_size=16, n_kp_enc=20, n_kp_dec=None, learned_feature_dim=16,
                 kp_range=(-1, 1), kp_activation="tanh", anchor_s=0.25,
                 use_resblock=True, embed_init_std=0.2, projection_dim=128, timestep_horizon=1,
                 filtering_heuristic='none', obj_ch_mult_prior=(1, 2),
                 obj_ch_mult=(1, 2, 3), obj_base_ch=32, obj_final_cnn_ch=32, num_res_blocks=2,
                 interaction_features=False, interaction_obj_on=False, interaction_depth=True,
                 temporal_interaction=True, cnn_mid_blocks=False, mlp_hidden_dim=256,
                 embed_prior_patch_pos=False, add_particle_temp_embed=False,
                 features_dist='gauss', n_fg_categories=8, n_fg_classes=4,
                 use_null_features_embed=True, obj_on_min=1e-4, obj_on_max=100.0, warmup_n_kp_ratio=0.35,
                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_ssm_last_layer=True,  # spatial softmax initialization
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super(ParticleEncoder, self).__init__()
        """
        DLP Foreground Module – Extracts objects from an image using keypoints and learned features. 
        Combines posterior CNN for full image processing and prior CNN for patch-based keypoint proposals.
        
        Args:
        cdim (int, default=3): Number of channels in the input image.
        image_size (int, default=64): Resolution of the input image (assumes square images).
        pad_mode (str, default='replicate'): Padding mode for CNNs, options are 'zeros' or 'replicate'.
        dropout (float, default=0.0): Dropout rate for CNNs (not used in practice).
        n_kp_per_patch (int, default=1): Number of keypoints proposed per patch.
        n_kp_prior (int, default=20): Number of keypoints filtered from prior proposals.
        patch_size (int, default=16): Size of patches for the prior keypoint proposal network.
        n_kp_enc (int, default=20): Number of posterior keypoints to learn.
        n_kp_dec (int, optional): Number of keypoints for decoder (if different from encoder).
        learned_feature_dim (int, default=16): Dimensionality of latent visual features for glimpses.
        kp_range (tuple, default=(-1, 1)): Range for keypoints; options are (-1, 1) or (0, 1).
        kp_activation (str, default='tanh'): Activation function for keypoints; 'tanh' for range (-1, 1), 'sigmoid' for range (0, 1).
        anchor_s (float, default=0.25): Glimpse size as a ratio of image size (e.g., 0.25 → glimpse size is 0.25 * image_size).
        use_resblock (bool, default=True): Whether to use residual blocks in CNNs.
        embed_init_std (float, default=0.2): Standard deviation for initializing learned tokens.
        projection_dim (int, default=128): Dimensionality of embeddings for transformer input.
        timestep_horizon (int, default=1): Maximum timesteps the model processes at once.
        filtering_heuristic (str, default='none'): Method for filtering prior keypoints. Options: 'distance', 'variance', 'random', 'none'.
        obj_ch_mult (tuple, default=(1, 2, 3)): Multiplicative factors for object feature channels at each CNN stage.
        obj_base_ch (int, default=32): Base number of channels in object feature extractor.
        obj_final_cnn_ch (int, default=32): Number of channels in the final object CNN layer.
        num_res_blocks (int, default=2): Number of residual blocks in object feature extractor.
        interaction_features (bool, default=False): Whether to compute interaction-based features.
        interaction_obj_on (bool, default=False): Whether to include "object-on" features for interactions.
        interaction_depth (bool, default=True): Whether to compute depth information for interactions.
        temporal_interaction (bool, default=True): Whether to model temporal interactions between features.
        cnn_mid_blocks (bool, default=False): Whether to include intermediate blocks in the CNN.
        mlp_hidden_dim (int, default=256): Hidden dimensionality for MLP layers.
        embed_prior_patch_pos (bool, default=False): Whether to embed positional information for prior patches.
        add_particle_temp_embed (bool, default=False): Whether to add temporal embeddings to particles.
        features_dist (str, default='gauss'): Distribution type for keypoint features. Options: 'gauss'.
        n_fg_categories (int, default=8): Number of foreground categories for classification.
        n_fg_classes (int, default=4): Number of foreground classes for classification.
        use_null_features_embed (bool, default=True): Whether to use a learned embedding for filtered-out particles.
        obj_on_min (float, default=1e-4): Minimum concentration value in Beta dist for transparency" probabilities.
        obj_on_max (float, default=100.0): Maximum concentration value in Beta dist for transparency" probabilities.
        """
        self.image_size = image_size
        self.dropout = dropout
        self.kp_range = kp_range
        self.n_kp_per_patch = n_kp_per_patch
        self.n_kp_enc = n_kp_enc
        self.n_kp_dec = self.n_kp_enc if n_kp_dec is None else n_kp_dec
        self.n_kp_prior = n_kp_prior
        self.kp_activation = kp_activation
        self.patch_size = patch_size
        self.anchor_patch_s = patch_size / image_size
        self.features_dim = int(image_size // (2 ** (len(obj_ch_mult) - 1)))
        self.learned_feature_dim = learned_feature_dim
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.anchor_s = anchor_s
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.num_patches = int((image_size // self.patch_size) ** 2)
        self.interaction_features = interaction_features
        self.interaction_depth = interaction_depth
        self.interaction_obj_on = interaction_obj_on
        self.temporal_interaction = temporal_interaction
        self.add_particle_temp_embed = add_particle_temp_embed
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim
        self.embed_prior_patch_pos = embed_prior_patch_pos
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.use_null_features_embed = use_null_features_embed
        self.warmup_n_kp_ratio = warmup_n_kp_ratio
        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_ssm_last_layer = init_ssm_last_layer  # spatial softmax initialization
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        self.prior_encoder = DLPPrior(cdim=cdim, image_size=image_size, n_kp=self.n_kp_per_patch,
                                      patch_size=patch_size, kp_range=kp_range, pad_mode=pad_mode,
                                      n_kp_prior=n_kp_prior,
                                      filtering_heuristic=filtering_heuristic,
                                      ch_mult=obj_ch_mult_prior, base_ch=obj_base_ch, num_res_blocks=num_res_blocks,
                                      use_resblock=use_resblock, cnn_mid_blocks=cnn_mid_blocks,
                                      init_ssm_last_layer=init_ssm_last_layer, init_conv_layers=init_conv_layers,
                                      init_conv_fg_std=init_conv_fg_std)

        # attribute encoder - anchor (z_a), offset (z_o), scale (z_s)
        anchor_s_att = patch_size / image_size
        self.particle_attribute_enc = ParticleAttributeEncoder(anchor_size=anchor_s, image_size=image_size,
                                                               n_particles=self.n_kp_prior,
                                                               margin=0, ch=cdim,
                                                               kp_activation=kp_activation,
                                                               use_resblock=use_resblock,
                                                               max_offset=1.0,
                                                               pad_mode=pad_mode, depth=not self.interaction_depth,
                                                               obj_on=not self.interaction_obj_on,
                                                               ch_mult=obj_ch_mult, base_ch=obj_base_ch,
                                                               final_cnn_ch=obj_final_cnn_ch,
                                                               num_res_blocks=num_res_blocks,
                                                               cnn_mid_blocks=cnn_mid_blocks,
                                                               hidden_dim=mlp_hidden_dim,
                                                               timestep_horizon=self.timestep_horizon,
                                                               add_particle_temp_embed=add_particle_temp_embed,
                                                               init_std=embed_init_std,
                                                               obj_on_min=self.obj_on_min,
                                                               obj_on_max=self.obj_on_max,
                                                               init_zero_bias=init_zero_bias,
                                                               init_conv_layers=init_conv_layers,
                                                               init_conv_fg_std=init_conv_fg_std)
        # appearance encoder - visual features encoder (z_f)
        output_logvar = (not self.interaction_features and self.features_dist != 'categorical')
        self.particle_features_enc = ParticleFeaturesEncoder(anchor_s, learned_feature_dim,
                                                             image_size,
                                                             margin=0, pad_mode=pad_mode,
                                                             ch_mult=obj_ch_mult, base_ch=obj_base_ch,
                                                             final_cnn_ch=obj_final_cnn_ch,
                                                             num_res_blocks=num_res_blocks,
                                                             output_logvar=output_logvar,
                                                             use_resblock=use_resblock, cnn_mid_blocks=cnn_mid_blocks,
                                                             hidden_dim=mlp_hidden_dim,
                                                             timestep_horizon=self.timestep_horizon,
                                                             add_particle_temp_embed=add_particle_temp_embed,
                                                             init_zero_bias=init_zero_bias,
                                                             init_conv_layers=init_conv_layers,
                                                             init_conv_fg_std=init_conv_fg_std
                                                             )
        # embed the source patch of the particles
        if self.embed_prior_patch_pos:
            self.patch_id_embed = nn.Parameter(self.embed_init_std * torch.randn(1, self.n_kp_prior, mlp_hidden_dim))
        else:
            self.patch_id_embed = None
        patch_centers = self.prior_encoder.get_patch_centers().unsqueeze(0) * (
                self.kp_range[1] - self.kp_range[0]) + self.kp_range[0]
        # append null particle
        patch_centers = torch.cat([patch_centers, torch.zeros(1, 1, 2)], dim=1)
        if self.n_kp_enc != self.n_kp_dec and self.interaction_features and self.use_null_features_embed:
            self.null_feature_embed = nn.Parameter(self.embed_init_std * torch.randn(1, 1, self.learned_feature_dim))
        self.register_buffer('patch_centers', patch_centers)
        self.register_buffer('mu_scale_prior', torch.tensor(np.log(self.anchor_s / (1 - self.anchor_s + 1e-5))))
        self.init_weights()

    def init_weights(self):
        self.prior_encoder.init_weights()

    def encode_prior(self, x):
        return self.prior_encoder(x)

    def encode_pos_scale_with_prior(self, x, deterministic=False, warmup=False, timesteps=None):
        batch_size, ch, h, w = x.shape
        kp_p, var_kp = self.encode_prior(x)
        # kp_init: [batch_size, n_kp, 2] in [-1, 1]
        kp_init = kp_p
        # 0. create or filter anchors
        if kp_init is None:
            # randomly sample n_kp_enc kp
            mu = torch.rand(batch_size, self.n_kp_prior, 2, device=x.device) * 2 - 1  # in [-1, 1]
        else:
            mu = kp_init
        logvar = torch.zeros_like(mu)
        z_base = mu + 0.0 * logvar  # deterministic value for chamfer-kl
        # 1. posterior offsets and scale, it is okay of scale_prev is None
        particle_stats_dict = self.particle_attribute_enc(x, z_base, timesteps=timesteps, deterministic=deterministic)

        mu_offset = particle_stats_dict['mu']
        logvar_offset = particle_stats_dict['logvar']
        mu_scale = particle_stats_dict['mu_scale']
        logvar_scale = particle_stats_dict['logvar_scale']
        if not self.interaction_obj_on:
            lobj_on_a = particle_stats_dict['lobj_on_a']
            lobj_on_b = particle_stats_dict['lobj_on_b']
            obj_on_a = particle_stats_dict['obj_on_a']
            obj_on_b = particle_stats_dict['obj_on_b']
            mu_obj_on = particle_stats_dict['mu_obj_on']
            z_obj_on = particle_stats_dict['z_obj_on']
        else:
            obj_on_a = obj_on_b = z_obj_on = mu_obj_on = None
        if not self.interaction_depth:
            mu_depth = particle_stats_dict['mu_depth']
            logvar_depth = particle_stats_dict['logvar_depth']
            if deterministic:
                z_depth = mu_depth
            else:
                z_depth = reparameterize(mu_depth, logvar_depth)
        else:
            mu_depth = logvar_depth = z_depth = None

        # final position
        mu_tot = z_base + mu_offset
        logvar_tot = logvar_offset
        mu_scale = self.mu_scale_prior + mu_scale

        # reparameterize
        if deterministic:
            z_offset = mu_offset
            z_scale = mu_scale
        else:
            z_offset = reparameterize(mu_offset, logvar_offset)
            z_scale = reparameterize(mu_scale, logvar_scale)

        z = z_base + z_offset
        z_base_var = var_kp.detach()
        confidence_score = particle_stats_dict['logvar'].detach()
        z_base_var = torch.cat([z_base_var, confidence_score], dim=-1)
        z_base_id = torch.arange(z_base.shape[-2], device=z_base.device)[None, :, None]  # [1, n_patches, 1]
        z_base_id = z_base_id.repeat(z_base.shape[0], 1, 1)  # [bs, n_patches, 1]

        if self.embed_prior_patch_pos:
            patch_id_embed = self.patch_id_embed.repeat(mu_tot.shape[0], 1, 1)
        else:
            patch_id_embed = None

        mu_score = (z_base_var.sum(-1, keepdim=True) / 30) * 2 - 1  # [bs * T, n_patches, 1]
        logvar_score = math.log(0.2 ** 2) * torch.ones_like(mu_score)  # 0.1, original without normalization: 1.0
        z_score = mu_score

        # variance filtering
        total_var = z_base_var.sum(-1)  # [bs * T, n_kp]
        # for single-image settings (self.timestep_horizon == 1), we can filter in the encoder
        # if self.n_kp_enc < self.n_kp_prior or (warmup and self.timestep_horizon == 1):
        if self.n_kp_enc < self.n_kp_prior:
            n_filter = self.n_kp_enc if not warmup else min(self.n_kp_enc,
                                                            int(self.warmup_n_kp_ratio * self.n_kp_prior))
            _, embed_ind = torch.topk(total_var, k=n_filter, dim=-1, largest=False)
            # make selection
            batch_ind = torch.arange(batch_size, device=x.device)[:, None]
            mu_tot = mu_tot[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            z_base = z_base[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            z_base_var = z_base_var[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            z_base_id = z_base_id[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
            mu_offset = mu_offset[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            logvar_offset = logvar_offset[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            z = z[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            z_offset = z_offset[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            z_scale = z_scale[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            mu_scale = mu_scale[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            mu_score = mu_score[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
            logvar_score = logvar_score[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
            z_score = z_score[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
            if logvar_scale is not None:
                logvar_scale = logvar_scale[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            if not self.interaction_obj_on:
                obj_on_a = obj_on_a[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
                obj_on_b = obj_on_b[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
                mu_obj_on = mu_obj_on[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
                z_obj_on = z_obj_on[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
            if not self.interaction_depth:
                z_depth = z_depth[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
                mu_depth = mu_depth[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
                logvar_depth = logvar_depth[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
            if self.embed_prior_patch_pos:
                patch_id_embed = patch_id_embed[batch_ind, embed_ind]

        out_dict = {'mu': mu, 'logvar': logvar, 'z_base': z_base, 'z': z, 'mu_tot': mu_tot,
                    'patch_id_embed': patch_id_embed,
                    'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
                    'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
                    'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset,
                    'kp_p': kp_p, 'var_kp': var_kp, 'z_base_var': z_base_var, 'total_var': total_var,
                    'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
                    'z_base_id': z_base_id, 'mu_score': mu_score, 'logvar_score': logvar_score, 'z_score': z_score}
        return out_dict

    def encode_appearance(self, x, z, z_scale, deterministic=False, timesteps=None, obj_on=None):
        # 2. posterior attributes: obj_on, depth and visual features
        obj_enc_out = self.particle_features_enc(x, z, z_scale=z_scale, timesteps=timesteps)

        mu_features = obj_enc_out['mu_features']
        logvar_features = obj_enc_out['logvar_features']
        cropped_objects = obj_enc_out['cropped_objects']

        if obj_on is not None:
            z_gate = torch.where(obj_on > 0.2, 1.0, 0.0)
            mu_features = z_gate * mu_features + (1 - z_gate) * self.null_feature_embed

        if not self.interaction_features:
            # reparameterize
            if self.features_dist == 'categorical':
                logits = mu_features.view(*mu_features.shape[:-1], self.n_fg_categories, self.n_fg_classes)
                # [bs, T, n_p, n_categories, n_classes]
                probs = logits.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                if deterministic:
                    samples = torch.argmax(probs.view(-1, probs.shape[-1]), dim=-1, keepdim=True)
                    samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                    samples = samples.view(probs.shape)
                    # straight-through
                    z_features = samples.detach() + (probs - probs.detach())
                    z_features = z_features.view(*mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]
                else:
                    samples = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
                    samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                    samples = samples.view(probs.shape)
                    # straight-through
                    z_features = samples.detach() + (probs - probs.detach())
                    z_features = z_features.view(*mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]
            else:
                z_features = reparameterize(mu_features, logvar_features) if not deterministic else mu_features
        else:
            z_features = mu_features

        out_dict = {'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
                    'cropped_objects': cropped_objects}
        return out_dict

    def encode_all(self, x, deterministic=False, warmup=False):
        # make sure x is [bs, T, ch, h, w]
        if len(x.shape) == 4:
            # that means x: [bs, ch, h, w]
            x = x.unsqueeze(1)  # -> [bs, T=1, ch, h, w]
        bs, timestep_horizon, ch, h, w = x.shape
        x = x.view(bs * timestep_horizon, *x.shape[2:])  # [bs * T, ch, h, w]
        # encode particles position and scale
        stage1_dict = self.encode_pos_scale_with_prior(x, deterministic=deterministic, warmup=warmup,
                                                       timesteps=timestep_horizon)
        # unpack
        kp_p = stage1_dict['kp_p']
        var_kp = stage1_dict['var_kp']
        z_base_var = stage1_dict['z_base_var']
        total_var = stage1_dict['total_var']
        patch_id_embed = stage1_dict['patch_id_embed']

        z_base = stage1_dict['z_base']
        mu_offset = stage1_dict['mu_offset']
        logvar_offset = stage1_dict['logvar_offset']
        z_offset = stage1_dict['z_offset']
        mu_tot = stage1_dict['mu_tot']
        z = stage1_dict['z']
        mu_scale = stage1_dict['mu_scale']
        logvar_scale = stage1_dict['logvar_scale']
        z_scale = stage1_dict['z_scale']
        # the following may be None if they are modeled by the interaction module
        mu_depth = stage1_dict['mu_depth']
        logvar_depth = stage1_dict['logvar_depth']
        z_depth = stage1_dict['z_depth']
        obj_on_a = stage1_dict['obj_on_a']
        obj_on_b = stage1_dict['obj_on_b']
        mu_obj_on = stage1_dict['mu_obj_on']
        z_obj_on = stage1_dict['z_obj_on']

        mu_score = stage1_dict['mu_score']
        logvar_score = stage1_dict['logvar_score']
        z_score = stage1_dict['z_score']

        if self.n_kp_enc != self.n_kp_dec and self.interaction_features and self.use_null_features_embed:
            total_var = z_base_var.sum(-1)
            n_filter = self.n_kp_dec if not warmup else min(self.n_kp_dec, int(self.warmup_n_kp_ratio * self.n_kp_enc))
            _, embed_ind = torch.topk(total_var, k=n_filter, dim=-1, largest=False)
            # make selection
            batch_ind = torch.arange(z.shape[0], device=z.device)[:, None]
            z_app = z[batch_ind, embed_ind].contiguous()
            z_scale_app = z_scale[batch_ind, embed_ind].contiguous()
            stage2_dict = self.encode_appearance(x, z_app, z_scale_app, deterministic=deterministic,
                                                 timesteps=timestep_horizon, obj_on=None)
            # unpack
            cropped_objects = stage2_dict['cropped_objects']
            mu_features_app = stage2_dict['mu_features']
            logvar_features = stage2_dict['logvar_features']  # None
            z_features_app = stage2_dict['z_features']

            mu_features = self.null_feature_embed.repeat(z.shape[0], self.n_kp_enc, 1)
            mu_features[batch_ind, embed_ind] = mu_features_app

            z_features = mu_features

        else:
            stage2_dict = self.encode_appearance(x, z, z_scale, deterministic=deterministic, timesteps=timestep_horizon,
                                                 obj_on=None)
            # unpack
            cropped_objects = stage2_dict['cropped_objects']
            mu_features = stage2_dict['mu_features']
            logvar_features = stage2_dict['logvar_features']
            z_features = stage2_dict['z_features']

        # reshape to [bs, T, ...]
        z_base = z_base.view(bs, timestep_horizon, *z_base.shape[1:])
        z_base_var = z_base_var.view(bs, timestep_horizon, *z_base_var.shape[1:])
        if patch_id_embed is not None:
            patch_id_embed = patch_id_embed.view(bs, timestep_horizon, *patch_id_embed.shape[1:])
        mu_offset = mu_offset.view(bs, timestep_horizon, *mu_offset.shape[1:])
        logvar_offset = logvar_offset.view(bs, timestep_horizon, *logvar_offset.shape[1:])
        z_offset = z_offset.view(bs, timestep_horizon, *z_offset.shape[1:])
        mu_tot = mu_tot.view(bs, timestep_horizon, *mu_tot.shape[1:])
        z = z.view(bs, timestep_horizon, *z.shape[1:])
        mu_scale = mu_scale.view(bs, timestep_horizon, *mu_scale.shape[1:])
        if logvar_scale is not None:
            logvar_scale = logvar_scale.view(bs, timestep_horizon, *logvar_scale.shape[1:])
        z_scale = z_scale.view(bs, timestep_horizon, *z_scale.shape[1:])
        if not self.interaction_features:
            mu_features = mu_features.view(bs, timestep_horizon, *mu_features.shape[1:])
            logvar_features = logvar_features.view(bs, timestep_horizon, *logvar_features.shape[1:])
        z_features = z_features.view(bs, timestep_horizon, *z_features.shape[1:])
        cropped_objects = cropped_objects.view(-1, *cropped_objects.shape[2:])
        if not self.interaction_depth:
            mu_depth = mu_depth.view(bs, timestep_horizon, *mu_depth.shape[1:])
            logvar_depth = logvar_depth.view(bs, timestep_horizon, *logvar_depth.shape[1:])
            z_depth = z_depth.view(bs, timestep_horizon, *z_depth.shape[1:])
        if not self.interaction_obj_on:
            obj_on_a = obj_on_a.view(bs, timestep_horizon, *obj_on_a.shape[1:])
            obj_on_b = obj_on_b.view(bs, timestep_horizon, *obj_on_b.shape[1:])
            mu_obj_on = mu_obj_on.view(bs, timestep_horizon, *mu_obj_on.shape[1:])
            z_obj_on = z_obj_on.view(bs, timestep_horizon, *z_obj_on.shape[1:])
        mu_score = mu_score.view(bs, timestep_horizon, *mu_score.shape[1:])
        logvar_score = logvar_score.view(bs, timestep_horizon, *logvar_score.shape[1:])
        z_score = z_score.view(bs, timestep_horizon, *z_score.shape[1:])

        encode_dict = {'mu_anchor': z_base, 'logvar_anchor': torch.zeros_like(z_base), 'z_base': z_base, 'z': z,
                       'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset, 'mu_tot': mu_tot,
                       'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
                       'cropped_objects': cropped_objects.detach(), 'patch_id_embed': patch_id_embed,
                       'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
                       'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
                       'kp_p': kp_p, 'var_kp': var_kp, 'z_base_var': z_base_var, 'mu_score': mu_score,
                       'logvar_score': logvar_score, 'z_score': z_score}
        return encode_dict

    def forward(self, x, deterministic=False, warmup=False):
        output_dict = self.encode_all(x, deterministic, warmup)
        return output_dict


class DLPEncoder(nn.Module):
    def __init__(self,
                 # Input configuration
                 cdim=3,  # Number of input image channels
                 image_size=64,  # Input image size (assumed square)
                 n_views=1,  # number of input views (e.g., multiple cameras)
                 pad_mode='replicate',  # Padding mode for CNNs
                 dropout=0.0,  # Dropout rate (not typically used)

                 # Keypoint and patch configuration
                 n_kp_per_patch=1,  # Number of keypoints per patch
                 n_kp_prior=20,  # Number of keypoints to filter from proposals
                 patch_size=16,  # Patch size for keypoint proposal network
                 n_kp_enc=20,  # Number of posterior keypoints to learn
                 n_kp_dec=None,  # Number of keypoints for decoder (if different from encoder)
                 warmup_n_kp_ratio=0.35,
                 mask_bg_in_enc=True,  # before encoding the bg, mask with the particles' obj_on

                 # Feature dimensions
                 learned_feature_dim=16,  # Dimension of learned visual features
                 learned_bg_feature_dim=16,  # Dimension of background features
                 kp_range=(-1, 1),  # Range for keypoint coordinates
                 kp_activation="tanh",  # Activation for keypoint coordinates
                 anchor_s=0.25,  # Glimpse size ratio

                 # Network architecture
                 use_resblock=True,  # Use residual blocks
                 embed_init_std=0.02,  # Standard deviation for embedding initialization
                 projection_dim=128,  # Embedding dimension for transformer

                 # Transformer configuration
                 timestep_horizon=1,  # Maximum timesteps to process at once
                 pte_layers=1,  # Number of particle transformer encoder layers
                 pte_heads=1,  # Number of particle transformer encoder heads
                 context_dim=16,  # Context latent dimension
                 filtering_heuristic='none',  # Method to filter prior keypoints
                 attn_norm_type='rms',  # Normalization type for attention

                 # Object encoder configuration
                 obj_ch_mult_prior=(1, 2,),  # Channel multipliers for prior patch encoder (kp proposals)
                 obj_ch_mult=(1, 2, 3),  # Channel multipliers for object encoder
                 obj_base_ch=32,  # Base channels for object encoder
                 obj_final_cnn_ch=32,  # Final CNN channels for object encoder
                 cnn_mid_blocks=False,  # Use middle blocks in CNN
                 mlp_hidden_dim=256,  # Hidden dimension for MLPs
                 pte_inner_dim=256,  # Inner dimension for particle transformer

                 # Background decoder configuration
                 bg_ch_mult=(1, 2, 3),  # Channel multipliers for background encoder
                 bg_base_ch=32,  # Base channels for background encoder
                 bg_final_cnn_ch=32,  # Final CNN channels for background encoder
                 num_res_blocks=2,  # Number of residual blocks

                 # Interaction configuration
                 ctx_pool_mode='none',  # Mode for pooling context features
                 interaction_depth=True,  # Enable depth interaction between particles
                 interaction_obj_on=False,  # Enable transparency interaction
                 interaction_features=True,  # Enable feature interaction
                 particle_score=False,  # Use particle confidence scores

                 # Embedding options
                 add_particle_temp_embed=False,  # Add temporal embeddings to particles
                 particle_positional_embed=True,  # Add positional embeddings to particles

                 # Context modeling
                 ctx_enc=None,
                 causal_ctx=True,  # Use causal attention for context
                 pte_ctx_layers=1,  # Number of context transformer layers
                 pte_ctx_heads=1,  # Number of context transformer heads
                 ctx_dist='gauss',  # Distribution type for context
                 n_ctx_categories=4,  # Number of context categories
                 n_ctx_classes=4,  # Number of context classes per category
                 global_ctx_pool=False,  # learn global latent context in addition to per-particle context
                 pool_ctx_dim=256,  # pool dimension for the global ctx latent
                 n_pool_ctx_categories=8,  # Number of global context categories (if categorical)
                 n_pool_ctx_classes=4,  # Number of global context classes per category
                 global_local_fuse_mode='none',  # concatenate/add global and local z_ctx to condition the dynamics
                 condition_local_on_global=True,  # condition z_context on z_context_global

                 # Distribution configuration
                 features_dist='gauss',  # Distribution type for features
                 n_fg_categories=8,  # Number of foreground categories, 'categorical' dist
                 n_fg_classes=4,  # Number of foreground classes per category, 'categorical' dist
                 n_bg_categories=4,  # Number of background categories, 'categorical' dist
                 n_bg_classes=4,  # Number of background classes per category, 'categorical' dist
                 obj_on_min=1e-4,  # Minimum concentration in Beta dist transparency value
                 obj_on_max=100,  # Maximum concentration in Beta dist transparency value
                 use_z_orig=True,  # Use original patch center coordinates as features

                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_ssm_last_layer=True,  # spatial softmax initialization
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 init_conv_bg_std=0.005,  # std for conv bg normal dist (<fg -> prioritize fg in learning)
                 ):
        """
        DLP Encoder Module

        A neural network module that extracts object-centric representations from images using
        the Deep Latent Particles (DLP) approach. This encoder processes images to identify
        and represent objects as particles with learned attributes.

        Args:
            cdim (int): Number of input image channels. Defaults to 3.
            image_size (int): Size of input images (assumed square). Defaults to 64.
            pad_mode (str): Padding mode for CNNs ('zeros' or 'replicate'). Defaults to 'replicate'.
            dropout (float): Dropout rate for CNNs (typically unused). Defaults to 0.0.
            n_kp_per_patch (int): Number of keypoints to extract per patch. Defaults to 1.
            n_kp_prior (int): Number of keypoints to filter from proposals. Defaults to 20.
            patch_size (int): Size of patches for keypoint proposal network. Defaults to 16.
            n_kp_enc (int): Number of posterior keypoints to learn. Defaults to 20.
            n_kp_dec (Optional[int]): Number of keypoints for decoder. If None, equals n_kp_enc. Defaults to None.
            learned_feature_dim (int): Dimension of learned visual features. Defaults to 16.
            learned_bg_feature_dim (int): Dimension of background features. Defaults to 16.
            kp_range (tuple): Range for keypoint coordinates, either (-1, 1) or (0, 1). Defaults to (-1, 1).
            kp_activation (str): Activation for keypoint coordinates ('tanh' or 'sigmoid'). Defaults to 'tanh'.
            anchor_s (float): Glimpse size as ratio of image_size. Defaults to 0.25.
            use_resblock (bool): Use residual blocks in network. Defaults to True.
            embed_init_std (float): Standard deviation for embedding initialization. Defaults to 0.02.
            projection_dim (int): Embedding dimension for transformer. Defaults to 128.
            timestep_horizon (int): Maximum number of timesteps to process at once. Defaults to 1.
            pte_layers (int): Number of particle transformer encoder layers. Defaults to 1.
            pte_heads (int): Number of particle transformer encoder heads. Defaults to 1.
            context_dim (int): Dimension of context latent space. Defaults to 16.
            filtering_heuristic (str): Method to filter prior keypoints. Defaults to 'none'.
            attn_norm_type (str): Normalization type for attention blocks. Defaults to 'rms'.
            obj_ch_mult_prior (tuple): Channel multipliers for prior patch encoder. Defaults to (1, 2, 3).
            obj_ch_mult (tuple): Channel multipliers for object encoder. Defaults to (1, 2, 3).
            obj_base_ch (int): Base channels for object encoder. Defaults to 32.
            obj_final_cnn_ch (int): Final CNN channels for object encoder. Defaults to 32.
            cnn_mid_blocks (bool): Use middle blocks in CNN. Defaults to False.
            mlp_hidden_dim (int): Hidden dimension for MLPs. Defaults to 256.
            pte_inner_dim (int): Inner dimension for particle transformer. Defaults to 256.
            bg_ch_mult (tuple): Channel multipliers for background encoder. Defaults to (1, 2, 3).
            bg_base_ch (int): Base channels for background encoder. Defaults to 32.
            bg_final_cnn_ch (int): Final CNN channels for background encoder. Defaults to 32.
            num_res_blocks (int): Number of residual blocks. Defaults to 2.
            ctx_pool_mode (str): Mode for pooling context features. Defaults to 'none'.
            interaction_depth (bool): Enable modeling depth by interaction between particles. Defaults to True.
            interaction_obj_on (bool): Enable modeling transparency by interaction. Defaults to False.
            interaction_features (bool): Enable modeling features by interaction. Defaults to True.
            particle_score (bool): Use particle confidence scores. Defaults to False.
            add_particle_temp_embed (bool): Add temporal embeddings to particles. Defaults to False.
            particle_positional_embed (bool): Add positional embeddings to particles. Defaults to True.
            causal_ctx (bool): Use causal attention for context. Defaults to True.
            pte_ctx_layers (int): Number of context transformer layers. Defaults to 1.
            pte_ctx_heads (int): Number of context transformer heads. Defaults to 1.
            ctx_dist (str): Distribution type for context ('gauss' or 'categorical'). Defaults to 'gauss'.
            n_ctx_categories (int): Number of context categories if categorical. Defaults to 4.
            n_ctx_classes (int): Number of context classes per category. Defaults to 4.
            features_dist (str): Distribution type for features ('gauss' or 'categorical'). Defaults to 'gauss'.
            n_fg_categories (int): Number of foreground categories if categorical. Defaults to 8.
            n_fg_classes (int): Number of foreground classes per category. Defaults to 4.
            n_bg_categories (int): Number of background categories if categorical. Defaults to 4.
            n_bg_classes (int): Number of background classes per category. Defaults to 4.
            obj_on_min (float): Minimum concentration value in Beta dist for transparency value. Defaults to 1e-4.
            obj_on_max (float): Maximum concentration value in Beta dist transparency value. Defaults to 100.
            use_z_orig (bool): Use original patch center coordinates. Defaults to True.

        Notes:
            The encoder operates in several stages:
            1. Patch Processing: Divides input image into patches and processes each
            2. Keypoint Proposal: Generates candidate keypoints using spatial softmax
            3. Feature Extraction: Learns visual features around each keypoint
            4. Particle Interaction: Models relationships between particles
            5. Context Modeling: Captures dynamics for the latent context (if enabled)

            The module supports both Gaussian and categorical distributions for
            features and context variables.

        The architecture uses a combination of CNNs and transformers:
            - CNNs for initial feature extraction from patches
            - Transformer encoders for modeling particle interactions
            - Separate pathways for foreground and background processing
            - Optional causal attention for temporal modeling
        """
        super(DLPEncoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.n_views = n_views
        self.dropout = dropout
        self.kp_range = kp_range
        self.n_kp_per_patch = n_kp_per_patch
        self.n_kp_enc = n_kp_enc
        self.n_kp_prior = n_kp_prior
        self.n_kp_dec = self.n_kp_enc if n_kp_dec is None else n_kp_dec
        self.warmup_n_kp_ratio = warmup_n_kp_ratio
        self.kp_activation = kp_activation
        self.patch_size = patch_size
        self.anchor_patch_s = patch_size / image_size
        self.features_dim = int(image_size // (2 ** (len(bg_ch_mult) - 1)))
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes

        self.context_dim = context_dim
        self.mask_bg_in_enc = mask_bg_in_enc  # before encoding the bg, mask with the particles' obj_on
        self.anchor_s = anchor_s
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.use_resblock = use_resblock
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.num_patches = int((image_size // self.patch_size) ** 2)
        self.attn_norm_type = attn_norm_type
        self.use_z_orig = use_z_orig
        self.interaction_depth = interaction_depth
        self.interaction_obj_on = interaction_obj_on
        self.interaction_features = interaction_features
        self.use_particle_inter_enc = (self.interaction_features or self.interaction_depth or self.interaction_obj_on)
        self.add_particle_temp_embed = add_particle_temp_embed
        self.temporal_interaction = False  # True=allow to attend over timesteps

        self.use_ctx_enc = (self.context_dim > 0)
        self.particle_score = particle_score
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_ssm_last_layer = init_ssm_last_layer  # spatial softmax initialization
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist
        self.init_conv_bg_std = init_conv_bg_std  # std for conv bg normal dist

        self.register_buffer('scale_anchor', torch.tensor(np.log(anchor_s / (1 - anchor_s + 1e-5))))
        use_norm_layer = True  # norm layer in the pre-attention projections modules
        self.particle_enc = ParticleEncoder(cdim=cdim,
                                            image_size=image_size,
                                            pad_mode=pad_mode,
                                            n_kp_per_patch=self.n_kp_per_patch,
                                            n_kp_prior=self.n_kp_prior,
                                            patch_size=self.patch_size, n_kp_enc=self.n_kp_enc, n_kp_dec=self.n_kp_dec,
                                            learned_feature_dim=learned_feature_dim,
                                            kp_range=kp_range, kp_activation=kp_activation, anchor_s=anchor_s,
                                            use_resblock=use_resblock, embed_init_std=embed_init_std,
                                            projection_dim=projection_dim, timestep_horizon=timestep_horizon,
                                            filtering_heuristic=filtering_heuristic,
                                            obj_ch_mult_prior=obj_ch_mult_prior,
                                            obj_ch_mult=obj_ch_mult,
                                            obj_base_ch=obj_base_ch,
                                            obj_final_cnn_ch=obj_final_cnn_ch, num_res_blocks=num_res_blocks,
                                            interaction_features=interaction_features,
                                            interaction_obj_on=interaction_obj_on,
                                            interaction_depth=interaction_depth,
                                            temporal_interaction=self.temporal_interaction,
                                            cnn_mid_blocks=cnn_mid_blocks,
                                            mlp_hidden_dim=mlp_hidden_dim, embed_prior_patch_pos=False,
                                            add_particle_temp_embed=self.add_particle_temp_embed,
                                            features_dist=self.features_dist, n_fg_categories=n_fg_categories,
                                            n_fg_classes=n_fg_classes, obj_on_min=self.obj_on_min,
                                            obj_on_max=self.obj_on_max, warmup_n_kp_ratio=self.warmup_n_kp_ratio,
                                            init_zero_bias=init_zero_bias,
                                            init_ssm_last_layer=init_ssm_last_layer,
                                            init_conv_layers=init_conv_layers,
                                            init_conv_fg_std=init_conv_fg_std)

        self.prior_encoder = self.particle_enc.prior_encoder
        self.bg_encoder = BgEncoder(cdim=cdim, image_size=image_size, pad_mode=pad_mode,
                                    learned_feature_dim=learned_bg_feature_dim, use_resblock=use_resblock,
                                    ch_mult=bg_ch_mult, base_ch=bg_base_ch, final_cnn_ch=bg_final_cnn_ch,
                                    num_res_blocks=num_res_blocks, interaction_features=interaction_features,
                                    cnn_mid_blocks=cnn_mid_blocks, mlp_hidden_dim=mlp_hidden_dim,
                                    timestep_horizon=timestep_horizon,
                                    add_particle_temp_embed=self.add_particle_temp_embed,
                                    features_dist=self.features_dist, n_bg_categories=n_bg_categories,
                                    n_bg_classes=n_bg_classes,
                                    init_zero_bias=init_zero_bias,
                                    init_conv_layers=init_conv_layers,
                                    init_conv_bg_std=init_conv_bg_std)

        patch_centers = self.prior_encoder.get_patch_centers().unsqueeze(0) * (
                self.kp_range[1] - self.kp_range[0]) + self.kp_range[0]
        # append null particle
        patch_centers = torch.cat([patch_centers, torch.zeros(1, 1, 2)], dim=1)
        # self.patch_centers = patch_centers
        self.register_buffer('patch_centers', patch_centers)
        particle_anchors = patch_centers[:, :-1]  # [1, 1, n_kp_enc], no need for (0,0)-the bg
        particle_anchors = particle_anchors.unsqueeze(-2).repeat(1, 1, self.n_kp_per_patch, 1).view(1, -1, 2)

        if self.use_particle_inter_enc:
            self.particle_inter_enc = ParticleInteractionEncoder(n_kp_enc=n_kp_enc, dropout=0.0,
                                                                 learned_feature_dim=learned_feature_dim,
                                                                 learned_bg_feature_dim=learned_bg_feature_dim,
                                                                 embed_init_std=embed_init_std,
                                                                 projection_dim=projection_dim,
                                                                 timestep_horizon=timestep_horizon,
                                                                 pte_layers=pte_layers,
                                                                 pte_heads=pte_heads,
                                                                 attn_norm_type=attn_norm_type, pad_mode=pad_mode,
                                                                 use_resblock=use_resblock,
                                                                 hidden_dim=mlp_hidden_dim,
                                                                 temporal_interaction=self.temporal_interaction,
                                                                 interaction_features=interaction_features,
                                                                 interaction_depth=interaction_depth,
                                                                 interaction_obj_on=interaction_obj_on,
                                                                 cdim=cdim, image_size=image_size, n_views=self.n_views,
                                                                 ch_mult=bg_ch_mult, base_ch=bg_base_ch,
                                                                 final_cnn_ch=bg_final_cnn_ch,
                                                                 num_res_blocks=num_res_blocks,
                                                                 bg=True, use_img_input=True,
                                                                 cnn_mid_blocks=cnn_mid_blocks,
                                                                 particle_score=True,
                                                                 particle_positional_embed=particle_positional_embed,
                                                                 norm_layer=use_norm_layer,
                                                                 add_particle_temp_embed=self.add_particle_temp_embed,
                                                                 features_dist=self.features_dist,
                                                                 n_fg_categories=n_fg_categories,
                                                                 n_fg_classes=n_fg_classes,
                                                                 n_bg_categories=n_bg_categories,
                                                                 n_bg_classes=n_bg_classes,
                                                                 scale_anchor=self.scale_anchor,
                                                                 obj_on_min=self.obj_on_min,
                                                                 obj_on_max=self.obj_on_max,
                                                                 particle_anchors=particle_anchors,
                                                                 use_z_orig=self.use_z_orig,
                                                                 init_zero_bias=init_zero_bias,
                                                                 init_conv_layers=init_conv_layers,
                                                                 init_conv_fg_std=init_conv_fg_std
                                                                 )
        else:
            self.particle_inter_enc = None

        self.ctx_enc = ctx_enc

        self.init_weights()

    def init_weights(self):
        self.particle_enc.init_weights()
        self.bg_encoder.init_weights()
        self.prior_encoder.init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            elif isinstance(m, nn.Linear):
                pass

    def get_bg_mask_from_particle_glimpses(self, z, z_obj_on, mask_size, z_scale=None, detach_grad=True):
        """
        generates a mask based on particles position and the scale. Masks are squares.
        """
        if detach_grad:
            with torch.no_grad():
                if z_scale is None:
                    obj_fmap_masks = create_masks_fast(z.detach(), anchor_s=self.anchor_s, feature_dim=mask_size)
                else:
                    obj_fmap_masks = create_masks_with_scale(z.detach(), anchor_s=self.anchor_s, image_size=mask_size,
                                                             scale=z_scale.detach())
                z_gate = torch.where(z_obj_on.detach() > 0.2, 1.0, 0.0)[:, :, None, None, None]
                obj_fmap_masks = obj_fmap_masks.clamp(0, 1) * z_gate
                # obj_fmap_masks = obj_fmap_masks.clamp(0, 1) * z_obj_on[:, :, None, None, None].detach()
                bg_mask = 1 - obj_fmap_masks.squeeze(2).sum(1, keepdim=True).clamp(0, 1)
        else:
            with torch.no_grad():
                if z_scale is None:
                    obj_fmap_masks = create_masks_fast(z, anchor_s=self.anchor_s, feature_dim=mask_size)
                else:
                    obj_fmap_masks = create_masks_with_scale(z, anchor_s=self.anchor_s, image_size=mask_size,
                                                             scale=z_scale)
            obj_fmap_masks = obj_fmap_masks.clamp(0, 1) * z_obj_on[:, :, None, None, None]
            bg_mask = 1 - obj_fmap_masks.squeeze(2).sum(1, keepdim=True).clamp(0, 1)
        return bg_mask

    def encode_all(self, x, deterministic=False, warmup=False, actions=None, actions_mask=None, lang_embed=None,
                   x_goal=None, deterministic_goal=True):
        """
        encoding steps:
        1. encode bg: x -> bg_enc -> [bs * T, projection_dim]
        2. encode patches: x -> patch_enc -> [bs * T, n_patches, projection_dim ]
        3. encode particles: [patches, bg, particle_tokens, bg_token, ctx_token] -> pte -> [bs, T, n_particles + 2, dim]
        """
        # make sure x is [bs, T, ch, h, w]
        if len(x.shape) == 4:
            # that means x: [bs, ch, h, w]
            x = x.unsqueeze(1)  # -> [bs, T=1, ch, h, w]
        bs, timestep_horizon, ch, h, w = x.shape
        if x_goal is not None:
            if len(x_goal.shape) == 4:
                # that means x: [bs, ch, h, w]
                x_goal = x_goal.unsqueeze(1)  # -> [bs, T=1, ch, h, w]
            x = torch.cat([x, x_goal], dim=1)  # [bs, T+1, ...]
        # x = x.view(bs * timestep_horizon, *x.shape[2:])  # [bs * T, ch, h, w]
        # encode particles
        particle_dict = self.particle_enc(x, deterministic, warmup)
        # unpack
        kp_p = particle_dict['kp_p']
        var_kp = particle_dict['var_kp']
        patch_id_embed = particle_dict['patch_id_embed']
        z_base = particle_dict['z_base']
        z = particle_dict['z']
        mu_offset = particle_dict['mu_offset']
        logvar_offset = particle_dict['logvar_offset']
        z_offset = particle_dict['z_offset']
        mu_tot = particle_dict['mu_tot']
        z_base_var = particle_dict['z_base_var']
        mu_scale = particle_dict['mu_scale']
        logvar_scale = particle_dict['logvar_scale']
        z_scale = particle_dict['z_scale']
        mu_depth = particle_dict['mu_depth']
        logvar_depth = particle_dict['logvar_depth']
        z_depth = particle_dict['z_depth']
        obj_on_a = particle_dict['obj_on_a']
        obj_on_b = particle_dict['obj_on_b']
        mu_obj_on = particle_dict['mu_obj_on']
        z_obj_on = particle_dict['z_obj_on']
        mu_features = particle_dict['mu_features']
        logvar_features = particle_dict['logvar_features']
        z_features = particle_dict['z_features']
        cropped_objects = particle_dict['cropped_objects']

        z_score = particle_dict['z_score']
        mu_score = particle_dict['mu_score']
        logvar_score = particle_dict['logvar_score']

        if x_goal is not None and deterministic_goal:
            z = torch.cat([z[:, :-1], mu_tot[:, -1:]], dim=1)
            if z_obj_on is not None:
                z_obj_on = torch.cat([z_obj_on[:, :-1], Beta(obj_on_a[:, -1:], obj_on_b[:, -1:]).mean], dim=1)
            z_scale = torch.cat([z_scale[:, :-1], mu_scale[:, -1:]], dim=1)
            if z_depth is not None:
                z_depth = torch.cat([z_depth[:, :-1], mu_depth[:, -1:]], dim=1)
            if not self.interaction_features:
                z_features = torch.cat([z_features[:, :-1], mu_features[:, -1:]], dim=1)

        # encode bg
        # x = x.view(bs * timestep_horizon, *x.shape[2:])  # [bs * T, ch, h, w]
        x = x.view(-1, *x.shape[2:])  # [bs * T, ch, h, w]
        z_v = z.view(-1, *z.shape[2:])
        if self.n_kp_dec != self.n_kp_enc:
            # variance filtering
            total_var = z_base_var.view(-1, *z_base_var.shape[2:]).sum(-1)
            n_filter = self.n_kp_dec if not warmup else min(self.n_kp_dec, int(self.warmup_n_kp_ratio * self.n_kp_enc))
            _, embed_ind = torch.topk(total_var, k=n_filter, dim=-1, largest=False)
            # make selection
            batch_ind = torch.arange(z_v.shape[0], device=z_v.device)[:, None]
            z_v = z_v[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]

        if self.interaction_obj_on:
            z_obj_on_v = torch.ones(-1, z_v.shape[1], device=x.device, dtype=torch.float)
        else:
            z_obj_on_v = z_obj_on.view(-1, *z_obj_on.shape[2:]).squeeze(-1)
            if self.n_kp_dec != self.n_kp_enc:
                z_obj_on_v = z_obj_on_v[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]

        if self.mask_bg_in_enc:
            bg_enc_mask = self.get_bg_mask_from_particle_glimpses(z_v, z_obj_on_v, mask_size=x.shape[-1])
            bg_dict = self.bg_encoder(x, bg_enc_mask, deterministic, timestep_horizon)
        else:
            bg_enc_mask = None
            bg_dict = self.bg_encoder(x, None, deterministic, timestep_horizon)  # unmasked bg
        mu_bg_features = bg_dict['mu_bg']
        mu_bg_features = mu_bg_features.view(bs, -1, mu_bg_features.shape[-1])
        logvar_bg_features = bg_dict['logvar_bg']
        if logvar_bg_features is not None:
            logvar_bg_features = logvar_bg_features.view(bs, -1, logvar_bg_features.shape[-1])
        z_bg_features = bg_dict['z_bg']
        z_bg_features = z_bg_features.view(bs, -1, z_bg_features.shape[-1])
        if x_goal is not None and deterministic_goal and not self.interaction_features:
            z_bg_features = torch.cat([z_bg_features[:, :-1], mu_bg_features[:, -1:]], dim=1)

        if self.use_particle_inter_enc:
            z_in_inter = z_base + z_offset  # so we can detach z_base (ssm) if more stable
            inter_dict = self.particle_inter_enc(x, z_in_inter, z_scale, z_obj_on, z_depth, z_features, z_bg_features,
                                                 z_base_var, z_score, patch_id_embed,
                                                 deterministic=deterministic, warmup=warmup)
            if self.interaction_features:
                mu_features = inter_dict['mu_features']
                logvar_features = inter_dict['logvar_features']
                z_features = inter_dict['z_features']

                if x_goal is not None and deterministic_goal:
                    z_features = torch.cat([z_features[:, :-1], mu_features[:, -1:]], dim=1)

                if inter_dict.get('mu_bg_features') is not None:
                    mu_bg_features = inter_dict['mu_bg_features']
                    logvar_bg_features = inter_dict['logvar_bg_features']
                    z_bg_features = inter_dict['z_bg_features']

                    if x_goal is not None and deterministic_goal:
                        z_bg_features = torch.cat([z_bg_features[:, :-1], mu_bg_features[:, -1:]], dim=1)
            if self.interaction_obj_on:
                obj_on_a = inter_dict['obj_on_a']
                obj_on_b = inter_dict['obj_on_b']
                mu_obj_on = inter_dict['mu_obj_on']
                z_obj_on = inter_dict['z_obj_on']
                if x_goal is not None and deterministic_goal:
                    z_obj_on = torch.cat([z_obj_on[:, :-1], Beta(obj_on_a[:, -1:], obj_on_b[:, -1:]).mean], dim=1)
            if self.interaction_depth:
                mu_depth = inter_dict['mu_depth']
                logvar_depth = inter_dict['logvar_depth']
                z_depth = inter_dict['z_depth']

                if x_goal is not None and deterministic_goal:
                    z_depth = torch.cat([z_depth[:, :-1], mu_depth[:, -1:]], dim=1)

        if self.use_ctx_enc:
            z_in_ctx = z_base + z_offset  # so we can detach z_base (ssm) if more stable
            if x_goal is not None and deterministic_goal:
                z_in_ctx = torch.cat([z_in_ctx[:, :-1], mu_tot[:, -1:]], dim=1)
            z_scale_in_ctx = z_scale
            z_obj_on_in_ctx = z_obj_on
            z_depth_in_ctx = z_depth
            z_features_in_ctx = z_features
            z_bg_features_in_ctx = z_bg_features

            ctx_dict = self.ctx_enc(z_in_ctx, z_scale_in_ctx, z_obj_on_in_ctx, z_depth_in_ctx,
                                    z_features_in_ctx, z_bg_features_in_ctx, z_base_var,
                                    z_score, patch_id_embed, deterministic=deterministic, warmup=warmup,
                                    actions=actions, actions_mask=actions_mask, lang_embed=lang_embed)
            z_goal_proj = ctx_dict['z_goal_proj']
            # global context
            mu_context_global = ctx_dict['mu_context_global']
            logvar_context_global = ctx_dict['logvar_context_global']
            z_context_global = ctx_dict['z_context_global']

            mu_context_global_dyn = ctx_dict['mu_context_global_dyn']
            logvar_context_global_dyn = ctx_dict['logvar_context_global_dyn']
            z_context_global_dyn = ctx_dict['z_context_global_dyn']

            # local context
            mu_context = ctx_dict['mu_context']
            logvar_context = ctx_dict['logvar_context']
            z_context = ctx_dict['z_context']

            mu_context_dyn = ctx_dict['mu_context_dyn']
            logvar_context_dyn = ctx_dict['logvar_context_dyn']
            z_context_dyn = ctx_dict['z_context_dyn']
        else:
            mu_context_global = logvar_context_global = z_context_global = None
            mu_context_global_dyn = logvar_context_global_dyn = z_context_global_dyn = None
            mu_context = logvar_context = z_context = None
            mu_context_dyn = logvar_context_dyn = z_context_dyn = None
            z_goal_proj = None

        if x_goal is not None:
            # remove last timestep
            z_base = z_base[:, :-1].contiguous()
            z = z[:, :-1].contiguous()
            mu_offset = mu_offset[:, :-1].contiguous()
            logvar_offset = logvar_offset[:, :-1].contiguous()
            z_offset = z_offset[:, :-1].contiguous()
            mu_tot = mu_tot[:, :-1].contiguous()
            mu_features = mu_features[:, :-1].contiguous()
            logvar_features = logvar_features[:, :-1].contiguous()
            z_features = z_features[:, :-1].contiguous()
            mu_bg_features = mu_bg_features[:, :-1].contiguous()
            logvar_bg_features = logvar_bg_features[:, :-1].contiguous()
            z_bg_features = z_bg_features[:, :-1].contiguous()
            obj_on_a = obj_on_a[:, :-1].contiguous()
            obj_on_b = obj_on_b[:, :-1].contiguous()
            z_obj_on = z_obj_on[:, :-1].contiguous()
            if mu_obj_on is not None:
                mu_obj_on = mu_obj_on[:, :-1].contiguous()
            z_base_var = z_base_var[:, :-1].contiguous()
            mu_depth = mu_depth[:, :-1].contiguous()
            logvar_depth = logvar_depth[:, :-1].contiguous()
            z_depth = z_depth[:, :-1].contiguous()
            mu_scale = mu_scale[:, :-1].contiguous()
            logvar_scale = logvar_scale[:, :-1].contiguous()
            z_scale = z_scale[:, :-1].contiguous()
            kp_p = kp_p.view(bs, -1, *kp_p.shape[1:])[:, :-1].reshape(-1, *kp_p.shape[1:])  # orig: [bs * T, N, 2]
            var_kp = var_kp.view(bs, -1, *var_kp.shape[1:])[:, :-1].reshape(-1,
                                                                            *var_kp.shape[1:])  # orig: [bs * T, N, 2]
            bg_enc_mask = bg_enc_mask.view(bs, -1, *bg_enc_mask.shape[1:])[:, :-1].reshape(-1, *bg_enc_mask.shape[
                1:])  # orig: [bs * t, 1, im_size, im_size]
            mu_score = mu_score[:, :-1].contiguous()
            logvar_score = logvar_score[:, :-1].contiguous()
            z_score = z_score[:, :-1].contiguous()

        encode_dict = {'mu_anchor': z_base, 'logvar_anchor': torch.zeros_like(z_base), 'z_base': z_base, 'z': z,
                       'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset, 'mu_tot': mu_tot,
                       'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
                       'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features,
                       'z_bg_features': z_bg_features, 'mu_context': mu_context, 'logvar_context': logvar_context,
                       'z_context': z_context,
                       'mu_context_global': mu_context_global, 'logvar_context_global': logvar_context_global,
                       'z_context_global': z_context_global,
                       'cropped_objects': cropped_objects.detach(), 'patch_id_embed': patch_id_embed,
                       'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
                       'z_base_var': z_base_var,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
                       'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
                       'kp_p': kp_p, 'var_kp': var_kp, 'bg_enc_mask': bg_enc_mask,
                       'mu_score': mu_score, 'logvar_score': logvar_score, 'z_score': z_score,
                       'mu_context_dyn': mu_context_dyn, 'logvar_context_dyn': logvar_context_dyn,
                       'z_context_dyn': z_context_dyn,
                       'mu_context_global_dyn': mu_context_global_dyn,
                       'logvar_context_global_dyn': logvar_context_global_dyn,
                       'z_context_global_dyn': z_context_global_dyn,
                       'z_goal_proj': z_goal_proj
                       }
        return encode_dict

    def forward(self, x, deterministic=False, warmup=False, actions=None, actions_mask=None, lang_embed=None,
                x_goal=None):
        output_dict = self.encode_all(x, deterministic, warmup, actions=actions, actions_mask=actions_mask,
                                      lang_embed=lang_embed, x_goal=x_goal)
        return output_dict
