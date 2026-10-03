"""Ordinary conditional FM policy sharing real DLP particles with GT-only LPWM dynamics.

No IMF, JVP, adaptive normalization, predicted-action dynamics, or latent alignment.
All public action APIs consume preprocessed inputs and return normalized actions.
"""

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_lpwm_fm import LPWMFMConfig
from .world_model import LPWMWorldModel

LANGUAGE = "observation.language.embedding"
LANGUAGE_MASK = "observation.language.attention_mask"
STATE = "observation.state"


class AttentionBlock(nn.Module):
    """Pre-norm attention and FFN; conditioning is exclusively through tokens."""

    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None, padding: Tensor | None = None) -> Tensor:
        h = self.norm1(x)
        x = x + self.attention(h, h, h, attn_mask=mask, key_padding_mask=padding, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class ParticleSceneEncoder(nn.Module):
    """Frame-causal attention over per-view particles/background and one state token/frame."""

    def __init__(self, config: LPWMFMConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.foreground = nn.Linear(config.latent_dim, d)
        self.background = nn.Linear(config.learned_bg_feature_dim, d)
        self.state = nn.Sequential(nn.Linear(config.state_dim, d), nn.GELU(), nn.Linear(d, d))
        self.view_embedding = nn.Embedding(len(config.image_features), d)
        self.time_embedding = nn.Embedding(config.n_obs_steps, d)
        self.type_embedding = nn.Embedding(3, d)
        self.blocks = nn.ModuleList(
            [AttentionBlock(d, config.n_heads, config.dropout) for _ in range(config.scene_n_layers)]
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, particles: Tensor, background: Tensor, state: Tensor) -> Tensor:
        b, t, v, n, _ = particles.shape
        if t > self.config.n_obs_steps or v != self.view_embedding.num_embeddings:
            raise ValueError("Particle time/view dimensions do not match policy configuration.")
        if state.shape != (b, t, self.config.state_dim):
            raise ValueError("State history must match the particle batch and time axes.")
        view = self.view_embedding.weight[None, None, :, None, :]
        fg = self.foreground(particles) + view + self.type_embedding.weight[0]
        bg = self.background(background).unsqueeze(-2) + view + self.type_embedding.weight[1]
        visual = torch.cat((fg, bg), dim=-2).reshape(b, t, v * (n + 1), -1)
        robot = self.state(state).unsqueeze(-2) + self.type_embedding.weight[2]
        tokens = torch.cat((visual, robot), dim=2)
        tokens = tokens + self.time_embedding.weight[:t][None, :, None, :]
        per_frame = tokens.shape[2]
        tokens = tokens.flatten(1, 2)
        # Time-major flattening; every token in a frame sees all tokens of that frame.
        times = torch.arange(t, device=tokens.device).repeat_interleave(per_frame)
        future_mask = times[None, :] > times[:, None]
        for block in self.blocks:
            tokens = block(tokens, mask=future_mask)
        return self.norm(tokens)


class FlowActionExpert(nn.Module):
    """Denoising action tokens attend to scene/language/time tokens and one another."""

    def __init__(self, config: LPWMFMConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.action_in = nn.Linear(config.action_dim, d)
        self.action_position = nn.Embedding(config.horizon, d)
        self.time_in = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(
            [AttentionBlock(d, config.n_heads, config.dropout) for _ in range(config.expert_n_layers)]
        )
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.action_dim))

    def time_embedding(self, tau: Tensor) -> Tensor:
        half = self.config.hidden_dim // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=tau.device, dtype=torch.float32) / max(half - 1, 1)
        )
        phase = tau.float()[:, None] * frequencies[None, :] * 1000
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        embedding = F.pad(embedding, (0, self.config.hidden_dim - embedding.shape[-1]))
        return self.time_in(embedding.to(dtype=self.action_in.weight.dtype))

    def forward(self, noisy_actions: Tensor, tau: Tensor, condition: Tensor, valid: Tensor) -> Tensor:
        k = noisy_actions.shape[1]
        action = self.action_in(noisy_actions) + self.action_position.weight[:k]
        prefix = torch.cat((condition, self.time_embedding(tau).unsqueeze(1)), dim=1)
        tokens = torch.cat((prefix, action), dim=1)
        prefix_size = prefix.shape[1]
        # Prefix cannot absorb noisy targets; action tokens may attend to every valid prefix and action.
        mask = torch.zeros(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device)
        mask[:prefix_size, prefix_size:] = True
        padding = torch.cat(
            (~valid, torch.zeros(valid.shape[0], k + 1, dtype=torch.bool, device=valid.device)), dim=1
        )
        for block in self.blocks:
            tokens = block(tokens, mask=mask, padding=padding)
        return self.output(tokens[:, -k:])


class LPWMFMPolicy(PreTrainedPolicy):
    """A: ordinary FM. B: ordinary FM plus clean-GT-action world-model training."""

    config_class = LPWMFMConfig
    name = "lpwm-fm"

    def __init__(self, config: LPWMFMConfig, **kwargs):
        super().__init__(config, **kwargs)
        config.validate_features()
        # One owner of the visual encoder: both encode and world_loss reach this same module.
        self.world_model = LPWMWorldModel(config)
        self.scene_encoder = ParticleSceneEncoder(config)
        self.language_projection = nn.Linear(config.language_dim, config.hidden_dim)
        self.expert = FlowActionExpert(config)
        self.reset()

    def get_optim_params(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def reset(self) -> None:
        self._observation_queues = {
            key: deque(maxlen=self.config.n_obs_steps) for key in (*self.config.image_features, STATE)
        }
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def _images(self, batch: dict[str, Tensor]) -> Tensor:
        views = []
        expected = None
        for key in self.config.image_features:
            if key not in batch:
                raise KeyError(f"Missing camera observation: {key}")
            image = batch[key]
            if image.ndim == 4:
                image = image.unsqueeze(1)
            if image.ndim != 5 or image.shape[2] != 3:
                raise ValueError(f"{key} must have shape [B,T,3,H,W] or [B,3,H,W].")
            if expected is not None and image.shape[:2] != expected:
                raise ValueError("All cameras must have identical batch/history axes.")
            expected = image.shape[:2]
            # Resize before stacking so cameras may have distinct source resolutions.
            if image.dtype == torch.uint8:
                image = image.float() / 255
            if not image.is_floating_point() or not torch.isfinite(image).all():
                raise ValueError("RGB must be finite uint8 or floating point in [0,1].")
            if image.amin() < 0 or image.amax() > 1:
                raise ValueError("Floating RGB must be in [0,1]; do not normalize visual features.")
            b, t = image.shape[:2]
            image = F.interpolate(
                image.flatten(0, 1),
                (self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).reshape(b, t, 3, self.config.image_size, self.config.image_size)
            views.append(image)
        return torch.stack(views, dim=2)

    def encode_condition(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Read only deployment observations; future world labels cannot enter this branch."""
        if STATE not in batch:
            raise KeyError("observation.state is required.")
        if LANGUAGE not in batch:
            raise KeyError(f"{LANGUAGE} is required; cache real frozen text embeddings before training.")
        images = self._images(batch)
        b, t = images.shape[:2]
        if not 1 <= t <= self.config.n_obs_steps:
            raise ValueError("Only configured past/current observation history may enter the policy.")
        state = batch[STATE]
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.shape != (b, t, self.config.state_dim):
            raise ValueError(f"observation.state must have shape [B,T,{self.config.state_dim}].")
        language = batch[LANGUAGE]
        if language.ndim != 3 or language.shape[0] != b or language.shape[2] != self.config.language_dim:
            raise ValueError(f"{LANGUAGE} must have shape [B,L,{self.config.language_dim}].")
        if language.shape[1] < 1 or not torch.isfinite(language).all():
            raise ValueError("Language embeddings must be nonempty and finite.")
        valid = batch.get(LANGUAGE_MASK, batch.get("observation.language.mask"))
        if valid is None:
            valid = torch.ones(language.shape[:2], dtype=torch.bool, device=language.device)
        if valid.shape != language.shape[:2]:
            raise ValueError("Language attention mask must have shape [B,L], with True for valid tokens.")
        valid = valid.bool()
        if not valid.any(dim=1).all():
            raise ValueError("Every sample needs at least one unmasked language token.")
        latents = self.world_model.encode(images, deterministic=True)
        scene = self.scene_encoder(latents["particles"], latents["background"], state)
        # Cached language is frozen data, never a trainable encoder hidden inside this policy.
        language = self.language_projection(language.detach().to(self.language_projection.weight.dtype))
        condition = torch.cat((scene, language), dim=1)
        condition_valid = torch.cat(
            (torch.ones(b, scene.shape[1], dtype=torch.bool, device=valid.device), valid), dim=1
        )
        return condition, condition_valid

    def flow_matching_loss(
        self, batch: dict[str, Tensor], *, noise: Tensor | None = None, tau: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute ordinary conditional path velocity regression, normalized over valid actions."""
        condition, valid = self.encode_condition(batch)
        actions = batch["action"]
        expected = (condition.shape[0], self.config.horizon, self.config.action_dim)
        if actions.shape != expected:
            raise ValueError(f"action must have shape {expected}; targets begin at current time.")
        padding = batch.get("action_is_pad")
        if padding is None:
            padding = torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
        if padding.shape != actions.shape[:2]:
            raise ValueError("action_is_pad must have shape [B,K].")
        keep = ~padding.bool()
        # Padded values never enter attention (even if padded target data are NaN/garbage).
        actions = torch.where(keep[..., None], actions, torch.zeros_like(actions))
        if not torch.isfinite(actions).all():
            raise ValueError("Non-padding actions must be finite.")
        noise = torch.randn_like(actions) if noise is None else noise
        tau = torch.rand(actions.shape[0], device=actions.device) if tau is None else tau
        if noise.shape != actions.shape or tau.shape != (actions.shape[0],):
            raise ValueError("noise must match action shape, and tau must have shape [B].")
        fraction = tau[:, None, None].to(actions.dtype)
        noisy = (1 - fraction) * actions + fraction * noise
        target = noise - actions
        prediction = self.expert(noisy, tau, condition, valid)
        error = (prediction.float() - target.float()).square()
        numerator = torch.where(keep[..., None], error, torch.zeros_like(error)).sum()
        denominator = (keep.sum() * self.config.action_dim).clamp_min(1)
        loss = numerator / denominator
        return loss, {"fm_loss": loss.detach(), "valid_action_fraction": keep.float().mean().detach()}

    def _world_scale(self, current_step: int | None) -> float:
        if current_step is None:
            if self.config.world_warmup_steps or self.config.world_ramp_steps:
                raise ValueError("current_step is required when world warmup/ramp is configured.")
            return self.config.world_weight
        if current_step < self.config.world_warmup_steps:
            return 0.0
        ramp = self.config.world_ramp_steps
        fraction = min(1.0, (current_step - self.config.world_warmup_steps + 1) / ramp) if ramp else 1.0
        return self.config.world_weight * fraction

    def forward(
        self, batch: dict[str, Tensor], current_step: int | None = None
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        loss, metrics = self.flow_matching_loss(batch)
        if self.config.variant == "B" and self.config.world_weight > 0:
            if "world.images" not in batch or "world.actions" not in batch:
                raise KeyError("Variant B requires world.images and clean GT world.actions.")
            if current_step is None and "current_step" in batch:
                current_step = int(batch["current_step"])
            scale = self._world_scale(current_step)
            world_images, world_actions = batch["world.images"], batch["world.actions"]
            if world_images.ndim != 6 or world_images.shape[2:4] != (len(self.config.image_features), 3):
                raise ValueError("world.images must be [B,T,V,3,H,W] in configured camera order.")
            if world_images.shape[0] != batch["action"].shape[0] or world_images.shape[1] < 2:
                raise ValueError(
                    "World windows must share the policy batch and include at least one transition."
                )
            if world_actions.shape != (
                world_images.shape[0],
                world_images.shape[1] - 1,
                self.config.action_dim,
            ):
                raise ValueError("world.actions must be [B,T-1,action_dim], aligned clean GT transitions.")
            # Never supervise transitions across padding/episode boundaries. Dataset should construct
            # complete windows. Reject unsupported padded windows rather than silently corrupting targets.
            for key in ("world_is_pad", "world.images_is_pad", "world.actions_is_pad"):
                if key in batch and batch[key].bool().any():
                    raise ValueError("World windows must contain valid within-episode transitions only.")
            if not torch.isfinite(world_actions).all():
                raise ValueError("GT world.actions must be finite.")
            world_loss, world_metrics = self.world_model.world_loss(
                world_images, world_actions.detach(), current_step=current_step
            )
            loss = loss + scale * world_loss
            metrics.update(
                {
                    f"world/{key}": value.detach() if isinstance(value, Tensor) else value
                    for key, value in world_metrics.items()
                }
            )
            metrics.update({"world_loss": world_loss.detach(), "world_weight": scale})
        metrics["loss"] = loss.detach()
        return loss, metrics

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        """Integrate noise at tau=1 to clean normalized actions at tau=0 with Euler."""
        condition, valid = self.encode_condition(batch)
        shape = (condition.shape[0], self.config.horizon, self.config.action_dim)
        if noise is not None and noise.shape != shape:
            raise ValueError(f"noise must have shape {shape}.")
        actions = (
            torch.randn(shape, device=condition.device, dtype=self.expert.action_in.weight.dtype)
            if noise is None
            else noise.clone()
        )
        steps = self.config.num_inference_steps
        for index in range(steps):
            tau = torch.full((shape[0],), 1 - index / steps, device=actions.device, dtype=torch.float32)
            actions = actions - self.expert(actions, tau, condition, valid) / steps
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Update observations every control tick; replan only when the action queue is empty."""
        history = dict(batch)
        for key, queue in self._observation_queues.items():
            if key not in batch:
                raise KeyError(f"Missing observation: {key}")
            value = batch[key]
            single_ndim = 2 if key == STATE else 4
            if value.ndim == single_ndim + 1:
                if not 1 <= value.shape[1] <= self.config.n_obs_steps:
                    raise ValueError("Explicit history length exceeds n_obs_steps.")
                queue.clear()
                queue.extend(value.unbind(dim=1))
            elif value.ndim == single_ndim:
                if queue and queue[-1].shape != value.shape:
                    raise ValueError(
                        "Observation batch changed; call reset() before a new environment batch."
                    )
                queue.append(value)
            else:
                raise ValueError(f"Unexpected rank for {key}.")
            while len(queue) < self.config.n_obs_steps:
                queue.appendleft(queue[0])
            history[key] = torch.stack(tuple(queue), dim=1)
        if not self._action_queue:
            chunk = self.predict_action_chunk(history, **kwargs)
            self._action_queue.extend(chunk[:, : self.config.n_action_steps].unbind(dim=1))
        return self._action_queue.popleft()
