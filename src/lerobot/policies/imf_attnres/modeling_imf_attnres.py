#!/usr/bin/env python

from collections import deque
from contextlib import nullcontext

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
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

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
        return self.model.parameters()

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

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

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = self._stack_images(batch)
        loss, diagnostics = self.model.compute_loss(batch)
        return loss, diagnostics or None


class IMFAttnResModel(nn.Module):
    def __init__(self, config: IMFAttnResConfig):
        super().__init__()
        self.config = config

        state_dim = config.robot_state_feature.shape[0]
        cond_dim_per_step = state_dim
        self.rgb_encoder = None
        self.condition_tokens_per_step = 1

        if config.image_features:
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

        if config.env_state_feature:
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

    def fn(self, z: Tensor, r: Tensor, t: Tensor, cond: Tensor | None = None) -> Tensor:
        return self.head(z, r, t, cond=cond)

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
                create_graph=False,
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
        u = self.fn(z_t, r, t, cond=cond)
        delta = self._broadcast_batch_time(t - r, z_t)
        return z_t - delta * u

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

    def _prepare_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
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
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        action = noise if noise is not None else torch.randn(
            (batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            device=device,
            dtype=dtype,
        )
        time_grid = torch.linspace(
            1.0,
            0.0,
            steps=self.config.num_inference_steps + 1,
            device=cond.device,
            dtype=cond.dtype,
        )
        for step_index in range(self.config.num_inference_steps):
            t = torch.full((batch_size,), float(time_grid[step_index].item()), device=cond.device, dtype=cond.dtype)
            r = torch.full((batch_size,), float(time_grid[step_index + 1].item()), device=cond.device, dtype=cond.dtype)
            action = self._sample_one_step(action, r=r, t=t, cond=cond)
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return action[:, start:end]

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        assert set(batch).issuperset({OBS_STATE, ACTION})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        actions = batch[ACTION]
        batch_size = actions.shape[0]
        assert actions.shape[1] == self.config.horizon
        assert batch[OBS_STATE].shape[1] == self.config.n_obs_steps
        cond = self._prepare_conditioning(batch)

        x = actions
        e = torch.randn_like(x)
        t, r = self._sample_tr(batch_size, device=x.device, dtype=x.dtype)

        t_broadcast = self._broadcast_batch_time(t, x)
        z_t = (1 - t_broadcast) * x + t_broadcast * e

        v = self.fn(z_t, t, t, cond=cond)
        u, du_dt = self._compute_u_and_du_dt(z_t, r, t, cond=cond, v=v)
        delta = self._broadcast_batch_time(t - r, du_dt)
        delta_du_dt = delta * du_dt.detach()
        compound_velocity = u + delta_du_dt
        target = e - x

        loss = self._velocity_loss_from_error(compound_velocity - target)
        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            mask = (~batch["action_is_pad"]).unsqueeze(-1).to(loss.dtype)
            valid_count = mask.sum() * loss.shape[-1]
            scalar_loss = (loss * mask).sum() / valid_count.clamp_min(1.0)
        else:
            scalar_loss = loss.mean()

        diagnostics = {}
        if self.config.enable_imf_diagnostics:
            diagnostics = self._imf_training_diagnostics(
                loss=scalar_loss,
                target=target,
                u=u,
                du_dt=du_dt,
                delta_du_dt=delta_du_dt,
                t=t,
                r=r,
            )
        return scalar_loss, diagnostics
