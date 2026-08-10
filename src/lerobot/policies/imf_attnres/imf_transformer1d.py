"""Local IMF-AttnRes transformer head aligned with diffusion_policy@185ed659."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .attnres_transformer_components import (
    AttnResOperator,
    AttnResSubLayer,
    AttnResTransformerBackbone,
    DifferentialTransformerBackbone,
    DifferentialTransformerBlock,
    GroupedQuerySelfAttention,
    MultiheadDifferentialSelfAttention,
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
        backbone_type: str = "attnres_full",
        n_kv_head: int = 8,
        attn_res_ffn_mult: float = 2.667,
        attn_res_eps: float = 1e-6,
        attn_res_rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()

        if n_head < 1:
            raise ValueError(f"n_head must be >= 1, got {n_head}.")
        if n_kv_head < 1:
            raise ValueError(f"n_kv_head must be >= 1, got {n_kv_head}.")
        if n_emb % n_head != 0:
            raise ValueError(f"n_emb={n_emb} must be divisible by n_head={n_head}.")
        if n_head % n_kv_head != 0:
            raise ValueError(f"n_head={n_head} must be divisible by n_kv_head={n_kv_head}.")
        if (
            backbone_type in {"attnres_full", "attnres_diff", "diff_transformer"}
            and (n_emb // n_head) % 2 != 0
        ):
            raise ValueError(
                f"{backbone_type} uses RoPE, which requires an even per-head dimension. "
                f"Got n_emb={n_emb}, n_head={n_head}, head_dim={n_emb // n_head}."
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
        self.diff_transformer_backbone = None
        encoder_only = False

        if backbone_type in {"attnres_full", "attnres_diff", "diff_transformer"}:
            if not time_as_cond:
                raise ValueError(f"{backbone_type} backbone requires time_as_cond=True.")
            if n_cond_layers != 0:
                raise ValueError(f"{backbone_type} backbone does not support n_cond_layers > 0.")

            self.time_token_proj = nn.Linear(n_emb, n_emb)
            backbone_kwargs = {
                "d_model": n_emb,
                "n_blocks": n_layer,
                "n_heads": n_head,
                "n_kv_heads": n_kv_head,
                "max_seq_len": t_seq + t_cond,
                "dropout": p_drop_attn,
                "ffn_mult": attn_res_ffn_mult,
                "eps": attn_res_eps,
                "rope_theta": attn_res_rope_theta,
                "causal_attn": causal_attn,
            }
            if backbone_type in {"attnres_full", "attnres_diff"}:
                self.attnres_backbone = AttnResTransformerBackbone(
                    **backbone_kwargs,
                    use_differential_attention=backbone_type == "attnres_diff",
                )
            else:
                self.diff_transformer_backbone = DifferentialTransformerBackbone(**backbone_kwargs)
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
                        activation="gelu",
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
                    activation="gelu",
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
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_layer,
                )

            self.ln_f = nn.LayerNorm(n_emb)

        self.layerwise_attention_mode = "cross_attn"
        self.layerwise_self_attn_every_n_layers = 2
        self.layerwise_prefix_proj = nn.Linear(cond_dim, n_emb) if obs_as_cond else None
        self.layerwise_cross_attn = None
        if obs_as_cond:
            self.layerwise_cross_attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=n_emb,
                        num_heads=n_head,
                        dropout=p_drop_attn,
                        batch_first=True,
                    )
                    for _ in range(n_layer)
                ]
            )

        if causal_attn and backbone_type != "attnres_full":
            sz = t_seq
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                s_seq = t_cond
                t_idx, s_idx = torch.meshgrid(
                    torch.arange(t_seq),
                    torch.arange(s_seq),
                    indexing="ij",
                )
                mask = t_idx >= (s_idx - 2)
                mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
                self.register_buffer("memory_mask", mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        self.head = nn.Linear(n_emb, output_dim)

        self.T = t_seq
        self.T_cond = t_cond
        self.horizon = horizon
        self.n_layer = n_layer
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def set_layerwise_prefix_config(
        self,
        *,
        attention_mode: str = "cross_attn",
        self_attn_every_n_layers: int = 2,
    ) -> None:
        if attention_mode not in {"self_attn", "cross_attn"}:
            raise ValueError(f"Unsupported layerwise attention_mode: {attention_mode!r}.")
        if self_attn_every_n_layers < 1:
            raise ValueError(f"self_attn_every_n_layers must be >= 1, got {self_attn_every_n_layers}.")
        self.layerwise_attention_mode = attention_mode
        self.layerwise_self_attn_every_n_layers = self_attn_every_n_layers

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
            DifferentialTransformerBackbone,
            DifferentialTransformerBlock,
            AttnResSubLayer,
            GroupedQuerySelfAttention,
            MultiheadDifferentialSelfAttention,
            SwiGLUFFN,
            RMSNormNoWeight,
            nn.MultiheadAttention,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"):
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            for name in ("in_proj_bias", "bias_k", "bias_v"):
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if getattr(module, "bias", None) is not None:
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
            raise RuntimeError(f"Unaccounted module {module}")

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            torch.nn.Linear,
            torch.nn.MultiheadAttention,
            MultiheadDifferentialSelfAttention,
        )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias") or pn.startswith("bias") or pn == "pseudo_query":
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        if self.pos_emb is not None:
            no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay sets!"
        )

        return [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
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
                raise ValueError("cond is required when obs_as_cond=True for attnres_full backbone.")
            token_parts.append(self.cond_obs_emb(cond))
        token_parts.append(sample_tokens)
        x = torch.cat(token_parts, dim=1)
        x = self.drop(x)
        x = self.attnres_backbone(x)
        x = x[:, -sample_tokens.shape[1] :, :]
        return x

    def _forward_diff_transformer(
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
                raise ValueError("cond is required when obs_as_cond=True for diff_transformer backbone.")
            token_parts.append(self.cond_obs_emb(cond))
        token_parts.append(sample_tokens)
        x = torch.cat(token_parts, dim=1)
        x = self.drop(x)
        x = self.diff_transformer_backbone(x)
        x = x[:, -sample_tokens.shape[1] :, :]
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

    @staticmethod
    def _condition_value(cond, key: str, default=None):
        if isinstance(cond, dict):
            return cond.get(key, default)
        return getattr(cond, key, default)

    def _layerwise_additive_key_mask(self, key_mask: torch.Tensor, query_len: int) -> torch.Tensor:
        additive_mask = torch.zeros(
            key_mask.shape[0],
            1,
            query_len,
            key_mask.shape[1],
            device=key_mask.device,
            dtype=self.input_emb.weight.dtype,
        )
        return additive_mask.masked_fill(~key_mask[:, None, None, :], torch.finfo(additive_mask.dtype).min)

    def _run_single_layerwise_backbone_step(
        self,
        tokens: torch.Tensor,
        layer_idx: int,
        *,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.backbone_type in {"attnres_full", "attnres_diff"}:
            rope_freqs = self.attnres_backbone.rope_freqs[: tokens.shape[1]]
            mask = None
            if key_mask is not None:
                mask = self._layerwise_additive_key_mask(key_mask, tokens.shape[1])
            if self.attnres_backbone.causal_attn:
                causal_mask = self.attnres_backbone._build_causal_mask(tokens.shape[1], tokens.device)
                mask = causal_mask if mask is None else mask + causal_mask
            layer_outputs = [tokens]
            start = 2 * layer_idx
            for sublayer in self.attnres_backbone.layers[start : start + 2]:
                sources = torch.stack(layer_outputs, dim=0)
                layer_outputs.append(sublayer(sources, rope_freqs, mask))
            return torch.stack(layer_outputs, dim=0).sum(dim=0)

        if self.backbone_type == "diff_transformer":
            rope_freqs = self.diff_transformer_backbone.rope_freqs[: tokens.shape[1]]
            mask = None
            if key_mask is not None:
                mask = self._layerwise_additive_key_mask(key_mask, tokens.shape[1])
            if self.diff_transformer_backbone.causal_attn:
                causal_mask = self.diff_transformer_backbone._build_causal_mask(
                    tokens.shape[1], tokens.device
                )
                mask = causal_mask if mask is None else mask + causal_mask
            return self.diff_transformer_backbone.layers[layer_idx](tokens, rope_freqs=rope_freqs, mask=mask)

        if self.encoder_only and isinstance(self.encoder, nn.TransformerEncoder):
            return self.encoder.layers[layer_idx](tokens)
        if self.decoder is not None:
            return self.decoder.layers[layer_idx](tgt=tokens, memory=tokens)
        return self.encoder(tokens)

    def _layerwise_prefix_context_update(
        self,
        action_tokens: torch.Tensor,
        prefix_tokens: torch.Tensor,
        prefix_mask: torch.Tensor,
        layer_idx: int,
        attention_mode: str,
        self_attn_every_n_layers: int,
    ) -> torch.Tensor:
        use_action_context = attention_mode != "cross_attn" or layer_idx % self_attn_every_n_layers == 0
        if use_action_context:
            context_tokens = torch.cat([prefix_tokens, action_tokens], dim=1)
            context_mask = torch.cat(
                [
                    prefix_mask,
                    torch.ones(
                        action_tokens.shape[:2],
                        dtype=torch.bool,
                        device=action_tokens.device,
                    ),
                ],
                dim=1,
            )
        else:
            context_tokens = prefix_tokens
            context_mask = prefix_mask

        context_out, _ = self.layerwise_cross_attn[layer_idx](
            query=action_tokens,
            key=context_tokens,
            value=context_tokens,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        return context_out

    @staticmethod
    def _with_last_depth_source_update(
        layer_outputs: list[torch.Tensor],
        update: torch.Tensor,
    ) -> list[torch.Tensor]:
        if not layer_outputs:
            raise ValueError("layer_outputs must contain at least one tensor.")
        conditioned = list(layer_outputs)
        conditioned[-1] = conditioned[-1] + update
        return conditioned

    def _forward_layerwise_prefix_attnres(
        self,
        action_tokens: torch.Tensor,
        prefix_layers: torch.Tensor,
        prefix_mask: torch.Tensor,
        attention_mode: str,
        self_attn_every_n_layers: int,
    ) -> torch.Tensor:
        rope_freqs = self.attnres_backbone.rope_freqs[: action_tokens.shape[1]]
        mask = None
        if self.attnres_backbone.causal_attn:
            mask = self.attnres_backbone._build_causal_mask(action_tokens.shape[1], action_tokens.device)

        layer_outputs = [action_tokens]
        for layer_idx in range(self.n_layer):
            prefix_tokens = self.layerwise_prefix_proj(prefix_layers[:, layer_idx])
            current_action_tokens = torch.stack(layer_outputs, dim=0).sum(dim=0)
            context_update = self._layerwise_prefix_context_update(
                current_action_tokens,
                prefix_tokens,
                prefix_mask,
                layer_idx,
                attention_mode,
                self_attn_every_n_layers,
            )
            # Keep the AttnRes residual/depth-attention stack identical to the
            # original backbone: only AttnRes sublayer outputs become persistent
            # depth branches. SmolVLM context is a temporary action-shaped
            # conditioning update on the current layer input, not an extra branch.
            conditioned_layer_outputs = self._with_last_depth_source_update(layer_outputs, context_update)

            start = 2 * layer_idx
            for sublayer in self.attnres_backbone.layers[start : start + 2]:
                sources = torch.stack(conditioned_layer_outputs, dim=0)
                output = sublayer(sources, rope_freqs, mask)
                layer_outputs.append(output)
                conditioned_layer_outputs.append(output)

        return torch.stack(layer_outputs, dim=0).sum(dim=0)

    def _forward_layerwise_prefix(
        self,
        sample: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        cond,
    ) -> torch.Tensor:
        if not self.obs_as_cond or self.layerwise_prefix_proj is None or self.layerwise_cross_attn is None:
            raise ValueError("Layer-wise prefix conditioning requires cond_dim > 0.")
        prefix_layers = self._condition_value(cond, "prefix_layers")
        prefix_mask = self._condition_value(cond, "prefix_mask")
        if prefix_layers is None or prefix_mask is None:
            raise ValueError("Layer-wise conditioning requires 'prefix_layers' and 'prefix_mask'.")
        if prefix_layers.ndim != 4:
            raise ValueError(
                "prefix_layers must have shape (B, n_layers, prefix_tokens, cond_dim). "
                f"Got {tuple(prefix_layers.shape)}."
            )
        if prefix_mask.ndim != 2:
            raise ValueError(
                f"prefix_mask must have shape (B, prefix_tokens). Got {tuple(prefix_mask.shape)}."
            )
        if prefix_layers.shape[0] != sample.shape[0] or prefix_mask.shape[0] != sample.shape[0]:
            raise ValueError("Layer-wise prefix batch dimension must match sample batch dimension.")
        if prefix_layers.shape[2] != prefix_mask.shape[1]:
            raise ValueError("Layer-wise prefix token dimension must match prefix_mask.")

        prefix_layers = prefix_layers.to(device=sample.device, dtype=self.input_emb.weight.dtype)
        prefix_mask = prefix_mask.to(device=sample.device, dtype=torch.bool)
        attention_mode = self._condition_value(cond, "attention_mode", self.layerwise_attention_mode)
        self_attn_every_n_layers = int(
            self._condition_value(
                cond,
                "self_attn_every_n_layers",
                self.layerwise_self_attn_every_n_layers,
            )
        )
        if attention_mode not in {"self_attn", "cross_attn"}:
            raise ValueError(f"Unsupported layer-wise attention mode: {attention_mode!r}.")
        if self_attn_every_n_layers < 1:
            raise ValueError(f"self_attn_every_n_layers must be >= 1, got {self_attn_every_n_layers}.")

        sample_tokens = self.input_emb(sample)
        action_token_count = sample_tokens.shape[1]
        r_token = self.time_emb(r).unsqueeze(1)
        t_token = self.time_emb(t).unsqueeze(1)
        if self.time_token_proj is not None:
            r_token = self.time_token_proj(r_token)
            t_token = self.time_token_proj(t_token)
        action_tokens = torch.cat(
            [
                r_token,
                t_token,
                sample_tokens,
            ],
            dim=1,
        )
        action_tokens = self.drop(action_tokens)
        num_layers = self.n_layer
        if prefix_layers.shape[1] < num_layers:
            repeat_count = num_layers - prefix_layers.shape[1]
            prefix_layers = torch.cat(
                [prefix_layers, prefix_layers[:, -1:].expand(-1, repeat_count, -1, -1)],
                dim=1,
            )

        if self.backbone_type in {"attnres_full", "attnres_diff"}:
            action_tokens = self._forward_layerwise_prefix_attnres(
                action_tokens,
                prefix_layers,
                prefix_mask,
                attention_mode,
                self_attn_every_n_layers,
            )
            return action_tokens[:, -action_token_count:, :]

        for layer_idx in range(num_layers):
            prefix_tokens = self.layerwise_prefix_proj(prefix_layers[:, layer_idx])
            use_joint_self = attention_mode != "cross_attn" or layer_idx % self_attn_every_n_layers == 0
            if use_joint_self:
                joint = torch.cat([prefix_tokens, action_tokens], dim=1)
                joint_mask = torch.cat(
                    [
                        prefix_mask,
                        torch.ones(
                            action_tokens.shape[:2],
                            dtype=torch.bool,
                            device=action_tokens.device,
                        ),
                    ],
                    dim=1,
                )
                joint = self._run_single_layerwise_backbone_step(joint, layer_idx, key_mask=joint_mask)
                action_tokens = joint[:, -action_tokens.shape[1] :]
            else:
                key_padding_mask = ~prefix_mask
                cross_out, _ = self.layerwise_cross_attn[layer_idx](
                    query=action_tokens,
                    key=prefix_tokens,
                    value=prefix_tokens,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                action_tokens = action_tokens + cross_out
                action_tokens = self._run_single_layerwise_backbone_step(action_tokens, layer_idx)

        return action_tokens[:, -action_token_count:, :]

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
            if isinstance(cond, dict):
                cond = {
                    key: value.to(dtype=dtype)
                    if torch.is_tensor(value) and value.is_floating_point()
                    else value
                    for key, value in cond.items()
                }
            else:
                cond = cond.to(dtype=dtype)
        r = self._prepare_time_input(r, sample)
        t = self._prepare_time_input(t, sample)

        if isinstance(cond, dict) or hasattr(cond, "prefix_layers"):
            x = self._forward_layerwise_prefix(sample, r, t, cond)
        elif self.backbone_type in {"attnres_full", "attnres_diff"}:
            x = self._forward_attnres_full(sample, r, t, cond=cond)
        elif self.backbone_type == "diff_transformer":
            x = self._forward_diff_transformer(sample, r, t, cond=cond)
        else:
            x = self._forward_vanilla(sample, r, t, cond=cond)

        x = self.ln_f(x)
        x = self.head(x)
        return x
