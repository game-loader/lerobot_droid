"""LPWM/DLPv3 RGB-to-particle encoding, without reconstruction, dynamics or action generation."""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy

from ._vendor.particle_encoder import DLPEncoder
from .configuration_lpwm_imf import LPWMIMFConfig

UPSTREAM_REVISION = "4cf53c403433e64c01652ac2adbec66231a46dea"
_ENCODER_ONLY = (
    "lpwm-imf currently implements only RGB -> DLPv3 visual latents. "
    "No IMF action head or training objective is implemented; use encode_observation() instead."
)


@dataclass
class LPWMVisualLatents:
    """Retain LPWM's structured foreground particles and separate background features.

    Foreground fields have shape (B, T, V, N, D_field); background has shape
    (B, T, V, D_background). Position coordinates are ordered (y, x), as upstream.
    Scale is the upstream *pre-sigmoid* latent, not a box in pixel units.
    Depth is a learned ordering attribute, not metric depth. No cross-view or
    temporal particle-identity matching is implied by these axes.
    """

    position: Tensor
    scale: Tensor
    depth: Tensor
    presence: Tensor
    features: Tensor
    background: Tensor

    @property
    def z(self) -> Tensor:
        """Pack foreground Z as [position, scale, depth, presence, appearance], without pooling."""
        return torch.cat((self.position, self.scale, self.depth, self.presence, self.features), dim=-1)


class LPWMVisualEncoder(nn.Module):
    """Run the upstream DLPv3 encoder with shared weights on independent RGB camera views."""

    def __init__(self, config: LPWMIMFConfig):
        super().__init__()
        self.config = config
        self.encoder = DLPEncoder(**config.encoder_kwargs())
        if config.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        """Keep a frozen encoder in evaluation mode when its parent is trained."""
        super().train(mode)
        if self.config.freeze_encoder:
            self.encoder.eval()
        return self

    def prepare_images(self, images: Tensor) -> Tensor:
        """Resize raw uint8 or [0, 1] floating RGB to the upstream encoder's image domain."""
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape) < 1:
            raise ValueError("Expected a nonempty RGB tensor of shape (batch, 3, height, width).")
        if images.dtype != torch.uint8 and not images.is_floating_point():
            raise TypeError("RGB input must be uint8 in [0, 255] or floating point in [0, 1].")
        parameter = next(self.encoder.parameters())
        if images.is_floating_point() and (
            not torch.isfinite(images).all() or images.amin() < 0 or images.amax() > 1
        ):
            raise ValueError(
                "Floating RGB must be finite and in [0, 1]; do not apply ImageNet normalization."
            )
        is_uint8 = images.dtype == torch.uint8
        images = images.to(device=parameter.device, dtype=parameter.dtype)
        if is_uint8:
            images = images / 255.0
        shape = (self.config.image_size, self.config.image_size)
        if images.shape[-2:] != shape:
            images = F.interpolate(images, size=shape, mode="bilinear", align_corners=False, antialias=True)
        return 2.0 * images - 1.0 if self.config.normalize_rgb else images

    def forward(self, images: Tensor, *, deterministic: bool = True) -> LPWMVisualLatents:
        """Encode (B,C,H,W), (B,T,C,H,W), or (B,T,V,C,H,W) RGB into structured Z."""
        if images.ndim == 4:
            images = images[:, None, None]
        elif images.ndim == 5:
            images = images[:, :, None]
        if images.ndim != 6 or images.shape[3] != 3 or min(images.shape) < 1:
            raise ValueError("Expected RGB with shape (B,3,H,W), (B,T,3,H,W), or (B,T,V,3,H,W).")
        batch, steps, views = images.shape[:3]
        if steps > self.config.n_obs_steps:
            raise ValueError(f"Received {steps} frames, exceeding n_obs_steps={self.config.n_obs_steps}.")
        flattened = images.permute(0, 2, 1, 3, 4, 5).reshape(batch * views * steps, *images.shape[-3:])
        prepared = self.prepare_images(flattened).reshape(
            batch * views, steps, 3, self.config.image_size, self.config.image_size
        )
        context = torch.no_grad() if self.config.freeze_encoder else nullcontext()
        with context:
            encoded = self.encoder(prepared, deterministic=deterministic, warmup=False)

        def restore(value: Tensor) -> Tensor:
            value = value.reshape(batch, views, steps, *value.shape[2:])
            return value.movedim(1, 2).contiguous()

        return LPWMVisualLatents(
            position=restore(encoded["z"]),
            scale=restore(encoded["z_scale"]),
            depth=restore(encoded["z_depth"]),
            presence=restore(encoded["obj_on"]),
            features=restore(encoded["z_features"]),
            background=restore(encoded["z_bg_features"]),
        )

    def load_lpwm_encoder(self, checkpoint: str | Path | Mapping[str, Tensor]) -> None:
        """Strictly load a local bare encoder or full LPWM state dict, excluding latent context.

        No download or random-weight fallback is performed. Full LPWM checkpoints
        use ``encoder_module.*`` keys; ``ctx_enc.*`` belongs to the excluded latent
        action/context stage. Architecture settings must match the checkpoint.
        """
        if isinstance(checkpoint, (str, Path)):
            path = Path(checkpoint)
            checkpoint = (
                load_file(str(path), device="cpu")
                if path.suffix == ".safetensors"
                else torch.load(path, map_location="cpu", weights_only=True)
            )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Expected an LPWM state dict or a checkpoint containing 'state_dict'.")
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
            raise ValueError("Checkpoint state_dict must be a string-keyed mapping.")
        state = dict(state)
        if state and all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        if any(key.startswith("encoder_module.") for key in state):
            state = {
                key.removeprefix("encoder_module."): value
                for key, value in state.items()
                if key.startswith("encoder_module.")
            }
        state = {key: value for key, value in state.items() if not key.startswith("ctx_enc.")}
        expected = self.encoder.state_dict()
        missing, unexpected = set(expected) - set(state), set(state) - set(expected)
        wrong_shapes = [
            key
            for key in set(expected) & set(state)
            if not isinstance(state[key], Tensor) or state[key].shape != expected[key].shape
        ]
        if missing or unexpected or wrong_shapes:
            raise ValueError(
                "LPWM encoder checkpoint does not match the configured architecture: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, shape_mismatch={sorted(wrong_shapes)}"
            )
        self.encoder.load_state_dict(state, strict=True)


class LPWMIMFPolicy(PreTrainedPolicy):
    """LeRobot-discoverable encoder-only LPWM-IMF scaffold, with no fake action outputs."""

    config_class = LPWMIMFConfig
    name = "lpwm-imf"

    def __init__(self, config: LPWMIMFConfig, **kwargs):
        super().__init__(config, **kwargs)
        config.validate_features()
        self.model = LPWMVisualEncoder(config)

    def encode_observation(
        self, batch: dict[str, Tensor], *, deterministic: bool = True
    ) -> LPWMVisualLatents:
        """Encode configured cameras in insertion order, preserving batch, time and view axes."""
        views = []
        batch_steps = None
        for key in self.config.image_features:
            if key not in batch:
                raise KeyError(f"Missing LPWM-IMF camera observation: {key}")
            images = batch[key]
            if images.ndim == 4:
                images = images.unsqueeze(1)
            if images.ndim != 5 or images.shape[2] != 3:
                raise ValueError(f"{key} must have shape (B,3,H,W) or (B,T,3,H,W).")
            if batch_steps is not None and images.shape[:2] != batch_steps:
                raise ValueError("All cameras must share batch and time dimensions.")
            batch_steps = images.shape[:2]
            # Each camera may have its own spatial resolution; do not stack raw images first.
            views.append(self.model(images, deterministic=deterministic))
        return LPWMVisualLatents(
            **{
                name: torch.cat([getattr(view, name) for view in views], dim=2)
                for name in ("position", "scale", "depth", "presence", "features", "background")
            }
        )

    def get_optim_params(self):
        """Expose trainable visual parameters for a separately supplied encoder objective."""
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def reset(self) -> None:
        """Do nothing: this encoder does not own action queues or hidden recurrent state."""

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Reject training until an actual objective and IMF action head are implemented."""
        raise NotImplementedError(_ENCODER_ONLY)

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Reject action generation rather than silently returning placeholders."""
        raise NotImplementedError(_ENCODER_ONLY)

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Reject robot control while only the visual encoder is available."""
        raise NotImplementedError(_ENCODER_ONLY)
