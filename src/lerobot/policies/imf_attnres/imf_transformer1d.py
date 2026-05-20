"""Local IMF-AttnRes transformer head aligned with diffusion_policy@185ed659."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .attnres_transformer_components import (
    AttnResOperator,
    AttnResSubLayer,
    AttnResTransformerBackbone,
    GroupedQuerySelfAttention,
    RMSNorm,
    RMSNormNoWeight,
    SwiGLUFFN,
)
from .utils import ModuleAttrMixin, SinusoidalPosEmb

logger = logging.getLogger(__name__)


class IMFTransformer1D(ModuleAttrMixin):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int | None = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 8,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
        backbone_type: str = 'attnres_full',
        n_kv_head: int = 8,
        attn_res_ffn_mult: float = 2.667,
        attn_res_eps: float = 1e-6,
        attn_res_rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()

        if n_head < 1:
            raise ValueError(f'n_head must be >= 1, got {n_head}.')
        if n_kv_head < 1:
            raise ValueError(f'n_kv_head must be >= 1, got {n_kv_head}.')
        if n_emb % n_head != 0:
            raise ValueError(f'n_emb={n_emb} must be divisible by n_head={n_head}.')
        if n_head % n_kv_head != 0:
            raise ValueError(f'n_head={n_head} must be divisible by n_kv_head={n_kv_head}.')
        if backbone_type == 'attnres_full' and (n_emb // n_head) % 2 != 0:
            raise ValueError(
                'attnres_full uses RoPE, which requires an even per-head dimension. '
                f'Got n_emb={n_emb}, n_head={n_head}, head_dim={n_emb // n_head}.'
            )
        if n_obs_steps is None:
            n_obs_steps = horizon

        self.backbone_type = backbone_type

        t_seq = horizon
        t_cond = 2
        if not time_as_cond:
            t_seq += 2
            t_cond -= 2
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            t_cond += n_obs_steps

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.drop = nn.Dropout(p_drop_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb) if obs_as_cond else None
        self.time_token_proj = None
        self.cond_pos_emb = None
        self.pos_emb = None
        self.encoder = None
        self.decoder = None
        self.attnres_backbone = None
        encoder_only = False

        if backbone_type == 'attnres_full':
            if not time_as_cond:
                raise ValueError('attnres_full backbone requires time_as_cond=True.')
            if n_cond_layers != 0:
                raise ValueError('attnres_full backbone does not support n_cond_layers > 0.')

            self.time_token_proj = nn.Linear(n_emb, n_emb)
            self.attnres_backbone = AttnResTransformerBackbone(
                d_model=n_emb,
                n_blocks=n_layer,
                n_heads=n_head,
                n_kv_heads=n_kv_head,
                max_seq_len=t_seq + t_cond,
                dropout=p_drop_attn,
                ffn_mult=attn_res_ffn_mult,
                eps=attn_res_eps,
                rope_theta=attn_res_rope_theta,
                causal_attn=causal_attn,
            )
            self.ln_f = RMSNorm(n_emb, eps=attn_res_eps)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, t_seq, n_emb))
            if t_cond > 0:
                self.cond_pos_emb = nn.Parameter(torch.zeros(1, t_cond, n_emb))
                if n_cond_layers > 0:
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=n_emb,
                        nhead=n_head,
                        dim_feedforward=4 * n_emb,
                        dropout=p_drop_attn,
                        activation='gelu',
                        batch_first=True,
                        norm_first=True,
                    )
                    self.encoder = nn.TransformerEncoder(
                        encoder_layer=encoder_layer,
                        num_layers=n_cond_layers,
                    )
                else:
                    self.encoder = nn.Sequential(
                        nn.Linear(n_emb, 4 * n_emb),
                        nn.Mish(),
                        nn.Linear(4 * n_emb, n_emb),
                    )

                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
                self.decoder = nn.TransformerDecoder(
                    decoder_layer=decoder_layer,
                    num_layers=n_layer,
                )
            else:
                encoder_only = True
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_layer,
                )

            self.ln_f = nn.LayerNorm(n_emb)

        if causal_attn and backbone_type != 'attnres_full':
            sz = t_seq
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
            self.register_buffer('mask', mask)

            if time_as_cond and obs_as_cond:
                s_seq = t_cond
                t_idx, s_idx = torch.meshgrid(
                    torch.arange(t_seq),
                    torch.arange(s_seq),
                    indexing='ij',
                )
                mask = t_idx >= (s_idx - 2)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        self.head = nn.Linear(n_emb, output_dim)

        self.T = t_seq
        self.T_cond = t_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        self.apply(self._init_weights)
        logger.info('number of parameters: %e', sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            AttnResTransformerBackbone,
            AttnResSubLayer,
            GroupedQuerySelfAttention,
            SwiGLUFFN,
            RMSNormNoWeight,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ('in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight'):
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            for name in ('in_proj_bias', 'bias_k', 'bias_v'):
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if getattr(module, 'bias', None) is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, AttnResOperator):
            torch.nn.init.zeros_(module.pseudo_query)
        elif isinstance(module, IMFTransformer1D):
            if module.pos_emb is not None:
                torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f'Unaccounted module {module}')

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters(recurse=False):
                fpn = f'{mn}.{pn}' if mn else pn

                if pn.endswith('bias') or pn.startswith('bias') or pn == 'pseudo_query':
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        if self.pos_emb is not None:
            no_decay.add('pos_emb')
        no_decay.add('_dummy_variable')
        if self.cond_pos_emb is not None:
            no_decay.add('cond_pos_emb')

        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f'parameters {inter_params} made it into both decay/no_decay sets!'
        assert len(param_dict.keys() - union_params) == 0, (
            f'parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay sets!'
        )

        return [
            {
                'params': [param_dict[pn] for pn in sorted(decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [param_dict[pn] for pn in sorted(no_decay)],
                'weight_decay': 0.0,
            },
        ]

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def _prepare_time_input(self, value: torch.Tensor | float | int, sample: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(value):
            value = torch.tensor([value], dtype=sample.dtype, device=sample.device)
        elif value.ndim == 0:
            value = value[None].to(device=sample.device, dtype=sample.dtype)
        else:
            value = value.to(device=sample.device, dtype=sample.dtype)
        return value.expand(sample.shape[0])

    def _forward_attnres_full(
        self,
        sample: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sample_tokens = self.input_emb(sample)
        token_parts = [
            self.time_token_proj(self.time_emb(r)).unsqueeze(1),
            self.time_token_proj(self.time_emb(t)).unsqueeze(1),
        ]
        if self.obs_as_cond:
            if cond is None:
                raise ValueError('cond is required when obs_as_cond=True for attnres_full backbone.')
            token_parts.append(self.cond_obs_emb(cond))
        token_parts.append(sample_tokens)
        x = torch.cat(token_parts, dim=1)
        x = self.drop(x)
        x = self.attnres_backbone(x)
        x = x[:, -sample_tokens.shape[1]:, :]
        return x

    def _forward_vanilla(
        self,
        sample: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r_emb = self.time_emb(r).unsqueeze(1)
        t_emb = self.time_emb(t).unsqueeze(1)
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            token_embeddings = torch.cat([r_emb, t_emb, input_emb], dim=1)
            token_count = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :token_count, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 2:, :]
        else:
            cond_embeddings = torch.cat([r_emb, t_emb], dim=1)
            if self.obs_as_cond:
                cond_embeddings = torch.cat([cond_embeddings, self.cond_obs_emb(cond)], dim=1)
            token_count = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :token_count, :]
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x

            token_embeddings = input_emb
            token_count = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :token_count, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask,
            )
        return x

    def forward(
        self,
        sample: torch.Tensor,
        r: torch.Tensor | float | int,
        t: torch.Tensor | float | int,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = self.input_emb.weight.dtype
        sample = sample.to(dtype=dtype)
        if cond is not None:
            cond = cond.to(dtype=dtype)
        r = self._prepare_time_input(r, sample)
        t = self._prepare_time_input(t, sample)

        if self.backbone_type == 'attnres_full':
            x = self._forward_attnres_full(sample, r, t, cond=cond)
        else:
            x = self._forward_vanilla(sample, r, t, cond=cond)

        x = self.ln_f(x)
        x = self.head(x)
        return x
