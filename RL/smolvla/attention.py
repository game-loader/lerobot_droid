"""Small query-only Transformers over frozen observation and ordered action tokens."""

import torch
from torch import nn

from RL.algorithms.iql import ActionPacker, _positive_int
from RL.smolvla.features import MASK_KEY, TOKEN_KEY


class TokenMemory(nn.Module):
    """A lightweight projection, NOT a new visual/observation backbone.

    Frozen VLM outputs are explicitly detached. Fixed-layout position embeddings
    distinguish camera/patch slots and action order. Invalid keys are excluded
    from attention, not merely zero-filled and allowed into the softmax.
    """

    def __init__(self, token_shape, hidden, *, action_dim=None, execution_steps=None, active_mask=None):
        super().__init__()
        count, width = token_shape
        for name, value in (("token_count", count), ("token_width", width), ("hidden", hidden)):
            _positive_int(name, value)
        self.token_shape = (count, width)
        self.observation_norm = nn.LayerNorm(width)
        self.observation_proj = nn.Linear(width, hidden)
        self.observation_position = nn.Parameter(torch.randn(1, count, hidden) * 0.02)
        self.packer = None
        if action_dim is not None:
            self.packer = ActionPacker(
                action_dim=action_dim, chunk_size=execution_steps, active_action_mask=active_mask
            )
            self.active_dim = int(active_mask.sum())
            self.action_proj = nn.Linear(self.active_dim + 1, hidden)
            self.action_position = nn.Parameter(torch.randn(1, execution_steps, hidden) * 0.02)
            self.action_type = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

    def forward(self, observation, actions=None, valid=None):
        tokens = observation.features[TOKEN_KEY].detach().float()
        mask = observation.features[MASK_KEY]
        if tokens.shape[1:] != self.token_shape or mask.dtype != torch.bool or mask.shape != tokens.shape[:2]:
            raise ValueError("Frozen token shape/mask differs from the attention head contract")
        if not mask.any(-1).all():
            raise ValueError("Each observation needs at least one valid token")
        clean = torch.where(mask[..., None], tokens, 0)
        memory = self.observation_proj(self.observation_norm(clean)) + self.observation_position
        if self.packer is None:
            if actions is not None or valid is not None:
                raise ValueError("V is observation-only; action input is forbidden")
            return memory, mask
        # Reuse the validated active-dimension / contiguous-step contract. Only
        # this small action array is temporarily packed; observations never are.
        packed = self.packer(actions, valid)
        active = packed[:, : -self.packer.chunk_size].reshape(tokens.shape[0], -1, self.active_dim)
        action_tokens = self.action_proj(torch.cat((active, valid[..., None].float()), -1))
        action_tokens = action_tokens + self.action_position + self.action_type
        return torch.cat((memory, action_tokens), 1), torch.cat((mask, valid), 1)


class QueryTransformer(nn.Module):
    """Only query tokens are updated; frozen conditioning stays a read-only memory."""

    def __init__(self, hidden, layers, heads):
        super().__init__()
        for name, value in (("hidden", hidden), ("layers", layers), ("heads", heads)):
            _positive_int(name, value)
        if hidden % heads:
            raise ValueError("Transformer hidden size must be divisible by attention heads")
        # Instantiate separately: TransformerDecoder's default cloning would
        # otherwise initialize all layers with identical parameters.
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                hidden,
                heads,
                dim_feedforward=2 * hidden,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, queries, memory, memory_mask, query_mask=None):
        for layer in self.layers:
            queries = layer(
                queries,
                memory,
                memory_key_padding_mask=~memory_mask,
                tgt_key_padding_mask=None if query_mask is None else ~query_mask,
            )
        return self.norm(queries)
