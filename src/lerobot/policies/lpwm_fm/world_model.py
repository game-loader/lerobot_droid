"""Real LPWM visual model and clean-action, frame-causal robot dynamics.

The native model is preserved separately. The robot model deliberately replaces
native context/AdaLN dynamics with ordinary self-attention over particle,
background and three identical clean-action tokens. No policy prediction is
accepted by ``world_loss``: actions must be recorded GT controls for t -> t+1.
"""

import inspect
import math
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from ._vendor.models import DLP, calc_dynamic_kl

UPSTREAM_REVISION = "4cf53c403433e64c01652ac2adbec66231a46dea"


def _native_kwargs(config: Any) -> dict:
    names = inspect.signature(DLP.__init__).parameters
    return {name: getattr(config, name) for name in names if name != "self" and hasattr(config, name)}


def native_reference(config: Any = None, **overrides) -> DLP:
    """Construct the complete native LPWM, including context, prior and dynamics.

    Uses native argument names/semantics (not the robot config's conventions).
    In native dynamics mode n_kp_enc is the *decoder* budget; the actual encoder
    retains n_kp_prior proposals. Sampling and native ELBO are available on DLP.
    No reference weights are downloaded or implied to be pretrained.
    """
    kwargs = _native_kwargs(config) if config is not None else {}
    if config is not None:
        kwargs["n_kp_enc"] = getattr(config, "n_kp_dec", config.n_kp_enc)
    kwargs.update(overrides)
    return DLP(**kwargs)


class ActionTokenDynamics(nn.Module):
    """Distributional particle dynamics with three identical action tokens/frame.

    Output at frame t predicts frame t+1. Frame t may attend to *all* tokens of
    frames <=t, including a_t, but never to tokens/actions at t+1. Different camera
    grids are distinguished, not interpreted as common physical coordinates.
    """

    action_token_repeat = 3

    def __init__(self, config: Any):
        super().__init__()
        dim = getattr(config, "world_hidden_dim", getattr(config, "hidden_dim", 256))
        self.n_particles = config.n_kp_enc
        self.feature_dim = config.learned_feature_dim
        self.bg_dim = config.learned_bg_feature_dim
        self.particle_projection = nn.Linear(6 + self.feature_dim, dim)
        self.background_projection = nn.Linear(self.bg_dim, dim)
        self.action_projection = nn.Linear(7, dim)
        self.slot_embedding = nn.Embedding(self.n_particles + 1, dim)
        self.camera_embedding = nn.Embedding(getattr(config, "max_cameras", 8), dim)
        self.action_type = nn.Parameter(torch.randn(1, 1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim,
            getattr(config, "world_n_heads", getattr(config, "n_heads", 8)),
            dim * 4,
            dropout=getattr(config, "dropout", 0.0),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            getattr(config, "world_n_layers", getattr(config, "world_layers", 4)),
            norm=nn.LayerNorm(dim),
            enable_nested_tensor=False,
        )
        # Native latent distributions: Gaussian position, scale, depth, features;
        # Beta visibility, plus Gaussian background. No point-estimate surrogate.
        self.gaussian_heads = nn.ModuleDict(
            {
                name: nn.Linear(dim, 2 * width)
                for name, width in (
                    ("position", 2),
                    ("scale", 2),
                    ("depth", 1),
                    ("features", self.feature_dim),
                )
            }
        )
        self.presence_head = nn.Linear(dim, 2)
        self.background_head = nn.Linear(dim, 2 * self.bg_dim)

    @staticmethod
    def causal_mask(steps: int, tokens_per_frame: int, device=None) -> Tensor:
        frame = torch.arange(steps, device=device).repeat_interleave(tokens_per_frame)
        return frame[None, :] > frame[:, None]  # True = masked (PyTorch Transformer).

    @staticmethod
    def _time_embedding(steps: int, dim: int, reference: Tensor) -> Tensor:
        frequency = torch.exp(
            -math.log(10000) * torch.arange(0, dim, 2, device=reference.device).float() / dim
        )
        phase = torch.arange(steps, device=reference.device).float()[:, None] * frequency
        result = torch.zeros(steps, dim, device=reference.device)
        result[:, 0::2] = phase.sin()
        result[:, 1::2] = phase[:, : dim // 2].cos()
        return result.to(reference.dtype)[None, :, None, :]

    def build_tokens(self, particles: Tensor, background: Tensor, actions: Tensor) -> Tensor:
        batch, steps, views, count = particles.shape[:4]
        if count != self.n_particles or actions.shape != (batch, steps, 7):
            raise ValueError("Expected matching particle history and GT actions [B,T,7].")
        if views > self.camera_embedding.num_embeddings:
            raise ValueError("Number of views exceeds configured max_cameras.")
        fg = self.particle_projection(particles)
        bg = self.background_projection(background).unsqueeze(-2)
        visual = torch.cat((fg, bg), dim=-2)
        visual = visual + self.slot_embedding.weight[None, None, None]
        visual = visual + self.camera_embedding.weight[:views][None, None, :, None]
        visual = visual.flatten(2, 3)
        # GT labels must not create a gradient route to a policy/predicted action.
        action = self.action_projection(actions.detach()).unsqueeze(2) + self.action_type
        action = action.expand(-1, -1, self.action_token_repeat, -1)
        tokens = torch.cat((visual, action), dim=2)
        return tokens + self._time_embedding(steps, tokens.shape[-1], tokens)

    def forward(self, particles: Tensor, background: Tensor, actions: Tensor) -> dict[str, Tensor]:
        batch, steps, views, count = particles.shape[:4]
        tokens = self.build_tokens(particles, background, actions)
        n_tokens = tokens.shape[2]
        hidden = self.transformer(
            tokens.flatten(1, 2), mask=self.causal_mask(steps, n_tokens, tokens.device)
        ).reshape(batch, steps, n_tokens, -1)
        visual = hidden[:, :, : views * (count + 1)].reshape(batch, steps, views, count + 1, -1)
        foreground = visual[..., :count, :]
        out = {}
        for name, head in self.gaussian_heads.items():
            mean, logvar = head(foreground).chunk(2, dim=-1)
            out[f"mu_{name}"] = mean
            out[f"logvar_{name}"] = logvar.clamp(-10, 6)
        a, b = self.presence_head(foreground).chunk(2, dim=-1)
        out["obj_on_a"] = F.softplus(a) + 1e-4
        out["obj_on_b"] = F.softplus(b) + 1e-4
        mean, logvar = self.background_head(visual[..., count, :]).chunk(2, dim=-1)
        out["mu_background"] = mean
        out["logvar_background"] = logvar.clamp(-10, 6)
        return out


class LPWMWorldModel(nn.Module):
    """Shared frame-independent DLP encoder, native decoder/prior, robot dynamics.

    ``encode`` and ``world_loss`` use exactly ``self.encoder``. RGB input is
    [0,1]; resizing is shared by both APIs. All native visual operations/KLs run
    in float32 even inside mixed-precision training for Beta/grid-sample safety.
    Loss weights rec/prior/dyn are applied here; outer world_weight belongs to
    the policy so it is never applied twice.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        if getattr(config, "action_token_repeat", 3) != 3:
            raise ValueError("Approved robot dynamics requires exactly 3 identical action tokens.")
        if getattr(config, "world_horizon", 1) != 1:
            raise ValueError("Robot world model currently supports GT one-step teacher forcing only.")
        kwargs = _native_kwargs(config)
        kwargs.update(timestep_horizon=1, context_dim=0, n_views=1, features_dist="gauss")
        self.visual = DLP(**kwargs)
        # Native static DLP normally decodes every particle. Retain the LPWM
        # decoder's variance-based budget and background masking for robot use.
        self.visual.n_kp_dec = config.n_kp_dec
        self.visual.filter_particles_in_decoder = True
        self.visual.encoder_module.n_kp_dec = config.n_kp_dec
        self.visual.decoder_module.n_kp_enc = config.n_kp_dec
        self.dynamics = ActionTokenDynamics(config)
        if getattr(config, "freeze_encoder", False):
            self.encoder.requires_grad_(False)

    @property
    def encoder(self) -> nn.Module:
        return self.visual.encoder_module

    @property
    def decoder(self) -> nn.Module:
        return self.visual.decoder_module

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self.config, "freeze_encoder", False):
            self.encoder.eval()
        return self

    def _prepare(self, images: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        if images.ndim != 6 or images.shape[3] != 3 or min(images.shape) < 1:
            raise ValueError("Expected nonempty float RGB images [B,T,V,3,H,W].")
        if not images.is_floating_point():
            raise TypeError("Expected floating RGB inputs in [0,1].")
        if not torch.isfinite(images).all() or images.amin() < 0 or images.amax() > 1:
            raise ValueError("RGB inputs must be finite and in [0,1].")
        shape = tuple(images.shape[:3])
        images = images.reshape(-1, *images.shape[-3:]).float()
        target = (self.config.image_size, self.config.image_size)
        if images.shape[-2:] != target:
            images = F.interpolate(images, target, mode="bilinear", align_corners=False, antialias=True)
        # One-frame sequences make causal independence an architectural fact,
        # not a reliance on how upstream PINT happens to flatten its tokens.
        return images.unsqueeze(1), shape

    @staticmethod
    def _restore(raw: dict, shape: tuple[int, int, int]) -> dict[str, Tensor]:
        names = (
            "z",
            "z_scale",
            "z_depth",
            "obj_on",
            "z_features",
            "z_bg_features",
            "mu_tot",
            "logvar_offset",
            "mu_scale",
            "logvar_scale",
            "mu_depth",
            "logvar_depth",
            "mu_features",
            "logvar_features",
            "mu_bg_features",
            "logvar_bg_features",
            "obj_on_a",
            "obj_on_b",
        )
        result = {name: raw[name].reshape(*shape, *raw[name].shape[2:]) for name in names}
        presence = result["obj_on"]
        if presence.ndim == result["z"].ndim - 1:
            presence = presence.unsqueeze(-1)
        result["particles"] = torch.cat(
            (result["z"], result["z_scale"], result["z_depth"], presence, result["z_features"]), -1
        )
        result["background"] = result["z_bg_features"]
        return result

    def encode(self, images: Tensor, deterministic: bool = True) -> dict[str, Tensor]:
        prepared, shape = self._prepare(images)
        with torch.autocast(device_type=images.device.type, enabled=False):
            if self.visual.normalize_rgb:
                prepared = 2 * prepared - 1
            raw = self.encoder(prepared, deterministic=deterministic, warmup=False)
            return self._restore(raw, shape)

    def _dynamic_loss(self, latent: dict[str, Tensor], prediction: dict[str, Tensor]) -> Tensor:
        def target(name):
            tensor = latent[name][:, 1:]
            return tensor.reshape(-1, *tensor.shape[3:]).contiguous()

        def prior(name):
            tensor = prediction[name]
            return tensor.reshape(-1, *tensor.shape[3:]).contiguous()

        # Directly reuse the native Gaussian/Beta dynamic KL, including its
        # posterior/prior gradients. Posterior positions = z_base + mu_offset.
        kwargs = {}
        for output, native in (
            ("position", "tot"),
            ("depth", "depth"),
            ("scale", "scale"),
            ("features", "features"),
            ("background", "bg_features"),
        ):
            suffix = "" if output == "position" else "_bg" if output == "background" else f"_{output}"
            kwargs[f"mu{suffix}_post"] = target(f"mu_{native}")
            kwargs[f"logvar{suffix}_post"] = target(
                "logvar_offset" if output == "position" else f"logvar_{native}"
            )
            kwargs[f"mu{suffix}_prior"] = prior(f"mu_{output}")
            kwargs[f"logvar{suffix}_prior"] = prior(f"logvar_{output}")
        for name in ("obj_on_a", "obj_on_b"):
            kwargs[f"{name}_post"] = target(name)
            kwargs[f"{name}_prior"] = prior(name)
        mask = target("obj_on").reshape(-1, self.config.n_kp_enc)
        result = calc_dynamic_kl(**kwargs, kl_mask=mask, reduce="mean", balance=0.5)
        return result["loss_kl"] / (self.config.n_kp_enc + 1)

    def world_loss(
        self, images: Tensor, actions: Tensor, current_step: int | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Native reconstruction/static priors + GT-action t -> t+1 latent KL.

        actions[:, t] MUST be the dataset clean control executed between images
        at t and t+1. No predicted-action branch, scheduled sampling, latent
        alignment, or cross-episode/padded transitions is implemented here.
        Callers must supply only valid within-episode windows.
        """
        del current_step  # Reserved for integration, not a silent schedule.
        prepared, shape = self._prepare(images)
        batch, steps, _ = shape
        if steps < 2 or actions.shape != (batch, steps - 1, 7):
            raise ValueError("Need T>=2 images and matching GT actions [B,T-1,7].")
        if not torch.isfinite(actions).all():
            raise ValueError("GT actions must be finite.")
        with torch.autocast(device_type=images.device.type, enabled=False):
            raw = self.visual(prepared, deterministic=False, with_loss=True, recon_loss_type="mse")
            latent = self._restore(raw, shape)
            prediction = self.dynamics(
                latent["particles"][:, :-1], latent["background"][:, :-1], actions.detach().float()
            )
            dynamic = self._dynamic_loss(latent, prediction)
            native_losses = raw["loss_dict"]
            # Native terms, normalized independently so weights do not depend
            # on image size or particle count. Not the upstream aggregate ELBO.
            reconstruction = native_losses["loss_rec"] / (3 * self.config.image_size**2)
            static_prior = native_losses["kl"] / (self.config.n_kp_enc + 1)
            loss = (
                getattr(self.config, "reconstruction_weight", getattr(self.config, "rec_weight", 1.0))
                * reconstruction
                + getattr(self.config, "prior_weight", 0.01) * static_prior
                + getattr(self.config, "dynamics_weight", getattr(self.config, "dyn_weight", 0.1)) * dynamic
            )
        metrics = {
            "world_loss": loss.detach(),
            "world_rec": reconstruction.detach(),
            "world_prior": static_prior.detach(),
            "world_dyn": dynamic.detach(),
            "world_psnr": native_losses["psnr"].detach(),
        }
        return loss, metrics
