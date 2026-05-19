#!/usr/bin/env python

"""Minimal RED tests for the IMF-AttnRes policy.

These tests intentionally describe the desired public API for a new LeRobot policy
without providing any production implementation. They should fail until
``src/lerobot/policies/imf_attnres`` and factory registration exist.
"""

import inspect
import os
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import (
    get_policy_class,
    make_policy,
    make_policy_config,
    make_pre_post_processors,
)
from lerobot.policies.imf_attnres.modeling_imf_attnres import IMFAttnResModel
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.scripts import lerobot_train as lerobot_train_script
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

POLICY_NAME = "imf-attnres"
STATE_DIM = 8
ACTION_DIM = 8
LIBERO_ACTION_DIM = 7
IMAGE_SIZE = 16
IMAGE_KEYS = (
    f"{OBS_IMAGES}.agentview",
    f"{OBS_IMAGES}.eye_in_hand",
)
LIBERO_IMAGE_KEYS = (
    f"{OBS_IMAGES}.image",
    f"{OBS_IMAGES}.image2",
)


def make_tiny_imf_attnres_config(state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
    """Create the small IMF-AttnRes config expected by fast policy tests."""
    config = make_policy_config(
        POLICY_NAME,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=2,
        n_layer=1,
        n_emb=32,
        spatial_softmax_num_keypoints=4,
        pretrained_backbone_weights=None,
        push_to_hub=False,
    )
    config.device = "cpu"
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        **{
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE))
            for key in IMAGE_KEYS
        },
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.IDENTITY,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,
    }
    return config


def make_tiny_libero_imf_attnres_config():
    """Create a tiny config with the canonical LIBERO vector dims and camera names."""
    config = make_tiny_imf_attnres_config(state_dim=STATE_DIM, action_dim=LIBERO_ACTION_DIM)
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        **{
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE))
            for key in LIBERO_IMAGE_KEYS
        },
    }
    return config


def make_libero_dataset_features() -> dict[str, dict]:
    return {
        ACTION: {
            "dtype": "float32",
            "shape": (LIBERO_ACTION_DIM,),
            "names": [f"joint_{idx}" for idx in range(LIBERO_ACTION_DIM)],
        },
        OBS_STATE: {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [f"state_{idx}" for idx in range(STATE_DIM)],
        },
        **{
            key: {
                "dtype": "image",
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channels"],
            }
            for key in LIBERO_IMAGE_KEYS
        },
    }


def add_libero_episode(dataset, num_frames: int = 5) -> None:
    """Populate a tiny local LeRobotDataset with deterministic LIBERO-like frames."""
    for frame_idx in range(num_frames):
        frame = {
            ACTION: np.linspace(-0.5, 0.5, LIBERO_ACTION_DIM, dtype=np.float32) + frame_idx * 0.01,
            OBS_STATE: np.linspace(-1.0, 1.0, STATE_DIM, dtype=np.float32) + frame_idx * 0.01,
            "task": "LIBERO smoke task",
        }
        for camera_idx, key in enumerate(LIBERO_IMAGE_KEYS):
            frame[key] = np.full(
                (IMAGE_SIZE, IMAGE_SIZE, 3),
                fill_value=32 + frame_idx + camera_idx * 16,
                dtype=np.uint8,
            )
        dataset.add_frame(frame)
    dataset.save_episode()
    dataset.finalize()


def make_libero_like_batch(
    *,
    batch_size: int = 2,
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
    image_keys: tuple[str, ...] = IMAGE_KEYS,
) -> dict[str, torch.Tensor]:
    """Synthetic LIBERO-like training batch with proprioception and two RGB cameras."""
    batch = {
        OBS_STATE: torch.randn(batch_size, 2, state_dim),
        ACTION: torch.randn(batch_size, 4, action_dim),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }
    batch.update(
        {
            key: torch.rand(batch_size, 2, 3, IMAGE_SIZE, IMAGE_SIZE)
            for key in image_keys
        }
    )
    return batch


def make_observation_batch(batch_size: int = 2, state_dim: int = STATE_DIM) -> dict[str, torch.Tensor]:
    """Current-step observation batch used by predict/select action tests."""
    batch = {
        OBS_STATE: torch.randn(batch_size, state_dim),
    }
    batch.update({key: torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE) for key in IMAGE_KEYS})
    return batch



def test_imf_attnres_default_normalization_matches_lerobot_diffusion_convention():
    """Default config should normalize visuals externally and avoid internal image normalization."""
    config = make_policy_config(POLICY_NAME, push_to_hub=False)

    assert config.normalization_mapping["VISUAL"] == NormalizationMode.MEAN_STD
    assert config.normalization_mapping["STATE"] == NormalizationMode.MIN_MAX
    assert config.normalization_mapping["ACTION"] == NormalizationMode.MIN_MAX


def test_imf_attnres_default_tr_sampling_config_matches_pmf_logit_normal_defaults():
    """IMF-AttnRes should expose the pMF Logit-Normal t/r sampling defaults, without uniform mixing."""
    config = make_policy_config(POLICY_NAME, push_to_hub=False)

    assert config.p_mean == -0.4
    assert config.p_std == 1.0
    assert config.data_proportion == 0.5


def test_imf_attnres_default_training_loss_is_pseudo_huber():
    """IMF-AttnRes should train with pseudo-Huber loss by default."""
    config = make_policy_config(POLICY_NAME, push_to_hub=False)

    assert config.loss_type == "pseudo_huber"
    assert config.pseudo_huber_delta == 1.0


def test_imf_attnres_default_action_latent_uses_dct_with_high_frequency_loss_weighting():
    """DCT action latent and high-frequency weighted loss should be enabled by default."""
    config = make_policy_config(POLICY_NAME, push_to_hub=False)

    assert config.action_latent_mode == "dct"
    assert config.dct_loss_high_freq_weight == 1.0
    assert config.dct_loss_freq_power == 2.0


def test_imf_attnres_factory_returns_registered_config_and_policy_class():
    """Factory helpers should expose the new IMF-AttnRes config and policy classes."""
    policy_cls = get_policy_class(POLICY_NAME)
    policy_cfg = make_tiny_imf_attnres_config()

    assert policy_cls.name == POLICY_NAME
    assert issubclass(policy_cls, PreTrainedPolicy)
    assert issubclass(
        policy_cfg.__class__,
        inspect.signature(policy_cls.__init__).parameters["config"].annotation,
    )
    assert policy_cfg.type == POLICY_NAME


def test_imf_attnres_sample_tr_uses_logit_normal_data_proportion_without_uniform(monkeypatch):
    """Sample t/r from Logit-Normal, force the FM half to r=t, and never draw uniform samples."""
    config = make_tiny_imf_attnres_config()
    model = IMFAttnResModel(config)
    logit_t = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    logit_r = torch.tensor([2.0, 1.0, 0.0, -1.0], dtype=torch.float32)
    randn_values = [logit_t, logit_r]

    def fake_randn(batch_size, *, device=None, dtype=None):
        assert batch_size == 4
        return randn_values.pop(0).to(device=device, dtype=dtype)

    def fail_uniform(*args, **kwargs):
        raise AssertionError("IMF-AttnRes t/r sampling should not use a uniform mixture")

    monkeypatch.setattr(torch, "randn", fake_randn)
    monkeypatch.setattr(torch, "rand", fail_uniform)

    t, r = model._sample_tr(batch_size=4, device=torch.device("cpu"), dtype=torch.float32)

    raw_t = torch.sigmoid(logit_t * config.p_std + config.p_mean)
    raw_r = torch.sigmoid(logit_r * config.p_std + config.p_mean)
    expected_r_before_sort = torch.cat([raw_t[:2], raw_r[2:]])
    expected_t = torch.maximum(raw_t, expected_r_before_sort)
    expected_r = torch.minimum(raw_t, expected_r_before_sort)

    torch.testing.assert_close(t, expected_t)
    torch.testing.assert_close(r, expected_r)
    assert randn_values == []
    assert torch.all(t >= r)
    torch.testing.assert_close(t[:2], r[:2])


def test_imf_attnres_compute_loss_uses_tr_sampler(monkeypatch):
    """Training loss should obtain t/r through the shared pMF-style sampler."""
    policy_cls = get_policy_class(POLICY_NAME)
    config = make_tiny_imf_attnres_config()
    policy = policy_cls(config)
    policy.train()
    calls = []

    def fake_sample_tr(batch_size, device, dtype):
        calls.append((batch_size, device, dtype))
        return (
            torch.full((batch_size,), 0.75, device=device, dtype=dtype),
            torch.full((batch_size,), 0.25, device=device, dtype=dtype),
        )

    monkeypatch.setattr(policy.model, "_sample_tr", fake_sample_tr, raising=False)

    loss, _ = policy.forward(make_libero_like_batch())

    assert calls == [(2, torch.device("cpu"), torch.float32)]
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_imf_attnres_forward_returns_scalar_loss():
    """Training forward should return a differentiable scalar loss."""
    policy_cls = get_policy_class(POLICY_NAME)
    config = make_tiny_imf_attnres_config()
    policy = policy_cls(config)
    policy.train()

    loss, output_dict = policy.forward(make_libero_like_batch())

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()
    assert loss.requires_grad
    assert output_dict is None or isinstance(output_dict, dict)


def test_imf_attnres_pseudo_huber_loss_matches_formula():
    """Pseudo-Huber loss should use delta^2 * (sqrt(1 + (error / delta)^2) - 1)."""
    config = make_tiny_imf_attnres_config()
    config.pseudo_huber_delta = 0.5
    model = IMFAttnResModel(config)
    error = torch.tensor([-1.0, 0.0, 0.25, 2.0])

    loss = model._velocity_loss_from_error(error)

    expected = config.pseudo_huber_delta**2 * (
        torch.sqrt(1 + (error / config.pseudo_huber_delta) ** 2) - 1
    )
    torch.testing.assert_close(loss, expected)


def test_imf_attnres_mse_loss_can_be_selected_for_ablation():
    """MSE should remain selectable for direct ablations against the previous objective."""
    config = make_tiny_imf_attnres_config()
    config.loss_type = "mse"
    model = IMFAttnResModel(config)
    error = torch.tensor([-1.0, 0.0, 0.25, 2.0])

    loss = model._velocity_loss_from_error(error)

    torch.testing.assert_close(loss, error.square())


def test_imf_attnres_dct_action_latent_roundtrip_recovers_actions():
    """Orthonormal DCT followed by IDCT along the horizon should recover actions."""
    config = make_tiny_imf_attnres_config()
    model = IMFAttnResModel(config)
    actions = torch.randn(2, config.horizon, ACTION_DIM)

    latents = model._encode_action_latent(actions)
    reconstructed = model._decode_action_latent(latents)

    torch.testing.assert_close(reconstructed, actions, rtol=1e-5, atol=1e-5)


def test_imf_attnres_identity_action_latent_is_noop():
    """Identity latent mode should preserve the previous action-space path for ablations."""
    config = make_tiny_imf_attnres_config()
    config.action_latent_mode = "identity"
    model = IMFAttnResModel(config)
    actions = torch.randn(2, config.horizon, ACTION_DIM)

    latents = model._encode_action_latent(actions)
    reconstructed = model._decode_action_latent(latents)

    torch.testing.assert_close(latents, actions)
    torch.testing.assert_close(reconstructed, actions)


def test_imf_attnres_dct_frequency_loss_weights_increase_for_high_frequencies():
    """DCT loss weights should be lowest at DC and highest at the last frequency bin."""
    config = make_tiny_imf_attnres_config()
    config.dct_loss_high_freq_weight = 3.0
    config.dct_loss_freq_power = 2.0
    model = IMFAttnResModel(config)

    weights = model._action_latent_loss_weights(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected = torch.tensor([1.0, 1.0 + 3.0 / 9.0, 1.0 + 12.0 / 9.0, 4.0])
    assert weights.shape == (1, config.horizon, 1)
    torch.testing.assert_close(weights.flatten(), expected)


def test_imf_attnres_identity_latent_loss_weights_are_uniform():
    """Non-DCT latent mode should not apply frequency-dependent weights."""
    config = make_tiny_imf_attnres_config()
    config.action_latent_mode = "identity"
    config.dct_loss_high_freq_weight = 3.0
    model = IMFAttnResModel(config)

    weights = model._action_latent_loss_weights(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(weights, torch.ones(1, config.horizon, 1))


def test_imf_attnres_generate_actions_decodes_dct_latent_before_slicing(monkeypatch):
    """Inference should sample in DCT latent space and IDCT back to actions before returning the chunk."""
    config = make_tiny_imf_attnres_config()
    model = IMFAttnResModel(config)
    batch = make_libero_like_batch()
    latent = torch.randn(2, config.horizon, ACTION_DIM)
    calls = {"decode": 0}

    monkeypatch.setattr(model, "_prepare_conditioning", lambda batch: torch.randn(2, 2, model.cond_dim))
    monkeypatch.setattr(model, "_sample_one_step", lambda z_t, r, t, cond: latent)
    original_decode = model._decode_action_latent

    def decode_spy(value):
        calls["decode"] += 1
        return original_decode(value)

    monkeypatch.setattr(model, "_decode_action_latent", decode_spy)

    actions = model.generate_actions(batch, noise=torch.zeros_like(latent))

    expected = original_decode(latent)[:, config.n_obs_steps - 1 : config.n_obs_steps - 1 + config.n_action_steps]
    assert calls["decode"] == 1
    torch.testing.assert_close(actions, expected)


def test_imf_attnres_forward_returns_wandb_diagnostics():
    """Training forward should expose scalar diagnostics for WandB variance debugging."""
    policy_cls = get_policy_class(POLICY_NAME)
    config = make_tiny_imf_attnres_config()
    config.enable_imf_diagnostics = True
    policy = policy_cls(config)
    policy.train()

    loss, output_dict = policy.forward(make_libero_like_batch())

    assert loss.shape == ()
    assert isinstance(output_dict, dict)
    expected_keys = {
        "imf_diagnostics/spike_threshold",
        "imf_diagnostics/spike/is_spike",
        "imf_diagnostics/all/target_norm_mean",
        "imf_diagnostics/all/u_norm_mean",
        "imf_diagnostics/all/du_dt_norm_mean",
        "imf_diagnostics/all/delta_du_dt_norm_mean",
        "imf_diagnostics/all/delta_mean",
        "imf_diagnostics/all/t_mean",
        "imf_diagnostics/all/r_mean",
        "imf_diagnostics/spike/target_norm_count",
        "imf_diagnostics/non_spike/target_norm_count",
        "imf_diagnostics/attnres/depth_attention_entropy_mean",
        "imf_diagnostics/attnres/depth_attention_max_weight_mean",
    }
    assert expected_keys.issubset(output_dict)
    assert output_dict["imf_diagnostics/spike_threshold"] == config.imf_diagnostics_spike_loss_threshold
    assert output_dict["imf_diagnostics/spike/is_spike"] == float(loss.detach().item() > 0.2)
    bucket_count = (
        output_dict["imf_diagnostics/spike/target_norm_count"]
        + output_dict["imf_diagnostics/non_spike/target_norm_count"]
    )
    assert bucket_count == 2.0
    for key, value in output_dict.items():
        assert isinstance(value, float), key
        assert np.isfinite(value), key


def test_imf_attnres_diagnostics_are_disabled_by_default():
    """IMF diagnostics should be explicit opt-in to avoid noisy per-step WandB logs."""
    policy_cls = get_policy_class(POLICY_NAME)
    config = make_tiny_imf_attnres_config()
    policy = policy_cls(config)
    policy.train()

    loss, output_dict = policy.forward(make_libero_like_batch())

    assert loss.shape == ()
    assert output_dict is None


def test_imf_attnres_diagnostics_spike_threshold_is_configurable():
    """The spike/non-spike split should use the policy config threshold."""
    policy_cls = get_policy_class(POLICY_NAME)
    config = make_tiny_imf_attnres_config()
    config.enable_imf_diagnostics = True
    config.imf_diagnostics_spike_loss_threshold = 1_000_000.0
    policy = policy_cls(config)
    policy.train()

    _, output_dict = policy.forward(make_libero_like_batch())

    assert isinstance(output_dict, dict)
    assert output_dict["imf_diagnostics/spike_threshold"] == 1_000_000.0
    assert output_dict["imf_diagnostics/spike/is_spike"] == 0.0
    assert output_dict["imf_diagnostics/spike/target_norm_count"] == 0.0
    assert output_dict["imf_diagnostics/non_spike/target_norm_count"] == 2.0


class _TinyAccelerator:
    num_processes = 1

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def unwrap_model(self, model, keep_fp32_wrapper=True):
        return model


def _make_train_metrics() -> MetricsTracker:
    return MetricsTracker(
        batch_size=2,
        num_frames=4,
        num_episodes=1,
        metrics={
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
        },
        accelerator=_TinyAccelerator(),
    )


def test_imf_attnres_update_policy_logs_gradient_norm_buckets():
    """Training update should log total, AttnRes, and main-DiT gradient norms."""
    policy_cls = get_policy_class(POLICY_NAME)
    config = make_tiny_imf_attnres_config()
    config.enable_imf_diagnostics = True
    policy = policy_cls(config)
    policy.train()
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=1e-4)

    _, output_dict = update_policy(
        _make_train_metrics(),
        policy,
        make_libero_like_batch(),
        optimizer,
        grad_clip_norm=10.0,
        accelerator=_TinyAccelerator(),
    )

    assert isinstance(output_dict, dict)
    for key in ("grad_norm/total", "grad_norm/attnres", "grad_norm/main_dit"):
        assert key in output_dict
        assert isinstance(output_dict[key], float)
        assert np.isfinite(output_dict[key])
        assert output_dict[key] >= 0.0


def test_imf_attnres_update_policy_omits_gradient_norm_buckets_when_diagnostics_disabled():
    """Gradient diagnostics should be disabled unless the policy opts in."""
    policy_cls = get_policy_class(POLICY_NAME)
    config = make_tiny_imf_attnres_config()
    policy = policy_cls(config)
    policy.train()
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=1e-4)

    _, output_dict = update_policy(
        _make_train_metrics(),
        policy,
        make_libero_like_batch(),
        optimizer,
        grad_clip_norm=10.0,
        accelerator=_TinyAccelerator(),
    )

    assert output_dict == {}


def test_lerobot_train_splits_imf_diagnostics_for_every_step_wandb_logging():
    """IMF and gradient diagnostics should be identifiable for every-step WandB logging."""
    every_step, regular = lerobot_train_script._split_every_step_wandb_diagnostics(
        {
            "imf_diagnostics/all/target_norm_mean": 1.0,
            "grad_norm/attnres": 2.0,
            "sample_weight_mean_weight": 3.0,
        }
    )

    assert every_step == {
        "imf_diagnostics/all/target_norm_mean": 1.0,
        "grad_norm/attnres": 2.0,
    }
    assert regular == {"sample_weight_mean_weight": 3.0}


def test_imf_attnres_predict_action_chunk_and_select_action_shapes():
    """Inference APIs should return action chunks and one selected action with expected shapes."""
    policy_cls = get_policy_class(POLICY_NAME)
    config = make_tiny_imf_attnres_config()
    policy = policy_cls(config)
    policy.eval()
    policy.reset()

    observation = make_observation_batch(batch_size=2)
    with torch.no_grad():
        action_chunk = policy.predict_action_chunk(observation)
    assert action_chunk.shape == (2, config.n_action_steps, ACTION_DIM)

    policy.reset()
    with torch.no_grad():
        selected_action = policy.select_action(observation)
    assert selected_action.shape == (2, ACTION_DIM)


def test_imf_attnres_save_and_load_preserves_outputs(tmp_path):
    """Saved and loaded IMF-AttnRes policies should have identical weights and outputs."""
    policy_cls = get_policy_class(POLICY_NAME)
    config = make_tiny_imf_attnres_config()
    policy = policy_cls(config)
    policy.eval()

    save_dir = tmp_path / "imf_attnres_policy"
    policy.save_pretrained(save_dir)

    loaded_policy = policy_cls.from_pretrained(save_dir, config=config)
    loaded_policy.eval()

    assert policy.state_dict().keys() == loaded_policy.state_dict().keys()
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, loaded_policy.state_dict()[key], rtol=0, atol=0)

    observation = make_observation_batch(batch_size=2)
    deterministic_noise = torch.randn(2, config.horizon, ACTION_DIM)
    policy.reset()
    loaded_policy.reset()
    with torch.no_grad():
        action = policy.select_action(dict(observation), noise=deterministic_noise.clone())
        loaded_action = loaded_policy.select_action(dict(observation), noise=deterministic_noise.clone())
    torch.testing.assert_close(action, loaded_action)


@pytest.mark.parametrize("state_dim,action_dim", [(7, 7), (8, 8)])
def test_imf_attnres_libero_style_training_smoke(state_dim: int, action_dim: int):
    """Run forward/backward/optimizer step on synthetic LIBERO-like 7/8-DoF features."""
    policy_cls = get_policy_class(POLICY_NAME)
    config = make_tiny_imf_attnres_config(state_dim=state_dim, action_dim=action_dim)
    policy = policy_cls(config)
    policy.train()
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=1e-4)

    batch = make_libero_like_batch(state_dim=state_dim, action_dim=action_dim)
    loss, _ = policy.forward(batch)
    assert loss.shape == ()
    assert torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )


def test_imf_attnres_canonical_libero_shapes_and_camera_keys_forward_backward():
    """Canonical LIBERO smoke: 8-D state, 7-D action, and image/image2 cameras."""
    policy_cls = get_policy_class(POLICY_NAME)
    config = make_tiny_libero_imf_attnres_config()

    assert config.input_features[OBS_STATE].shape == (STATE_DIM,)
    assert config.output_features[ACTION].shape == (LIBERO_ACTION_DIM,)
    assert set(config.input_features) == {OBS_STATE, *LIBERO_IMAGE_KEYS}

    policy = policy_cls(config)
    policy.train()
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=1e-4)

    batch = make_libero_like_batch(action_dim=LIBERO_ACTION_DIM, image_keys=LIBERO_IMAGE_KEYS)
    assert batch[OBS_STATE].shape == (2, config.n_obs_steps, STATE_DIM)
    assert batch[ACTION].shape == (2, config.horizon, LIBERO_ACTION_DIM)
    for key in LIBERO_IMAGE_KEYS:
        assert batch[key].shape == (2, config.n_obs_steps, 3, IMAGE_SIZE, IMAGE_SIZE)

    loss, _ = policy.forward(batch)
    assert loss.shape == ()
    assert torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )


def test_imf_attnres_libero_style_train_pipeline_smoke(tmp_path):
    """Exercise LeRobot dataset/policy factories on local LIBERO-shaped data without downloading."""
    pytest.importorskip("datasets", reason="LeRobotDataset local smoke requires the dataset extra")
    import datasets.config

    from lerobot.datasets.factory import make_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config = make_tiny_libero_imf_attnres_config()
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MIN_MAX,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    local_repo_id = "local/libero_imf_attnres_smoke"
    dataset_root = tmp_path / "libero_imf_attnres_smoke"
    created_dataset = LeRobotDataset.create(
        repo_id=local_repo_id,
        fps=10,
        features=make_libero_dataset_features(),
        root=dataset_root,
        use_videos=False,
    )
    add_libero_episode(created_dataset, num_frames=5)

    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=local_repo_id,
            root=str(dataset_root),
            episodes=[0],
            use_imagenet_stats=False,
        ),
        policy=config,
        steps=1,
        batch_size=2,
        num_workers=0,
        prefetch_factor=0,
        persistent_workers=False,
        use_policy_training_preset=True,
    )
    train_cfg.validate()

    hf_datasets_cache = tmp_path / "hf_datasets_cache"
    hf_downloaded_cache = hf_datasets_cache / "downloads"
    hf_extracted_cache = hf_downloaded_cache / "extracted"
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets_cache)
    os.environ["HF_HOME"] = str(tmp_path / "hf_home")

    with patch.object(datasets.config, "HF_DATASETS_CACHE", hf_datasets_cache), patch.object(
        datasets.config, "DOWNLOADED_DATASETS_PATH", hf_downloaded_cache
    ), patch.object(datasets.config, "EXTRACTED_DATASETS_PATH", hf_extracted_cache), patch(
        "lerobot.datasets.dataset_metadata.snapshot_download"
    ) as metadata_download, patch(
        "lerobot.datasets.lerobot_dataset.snapshot_download"
    ) as data_download:
        dataset = make_dataset(train_cfg)

    metadata_download.assert_not_called()
    data_download.assert_not_called()

    assert dataset.meta.shapes[OBS_STATE] == (STATE_DIM,)
    assert dataset.meta.shapes[ACTION] == (LIBERO_ACTION_DIM,)
    assert set(dataset.meta.camera_keys) == set(LIBERO_IMAGE_KEYS)
    for key in LIBERO_IMAGE_KEYS:
        assert dataset.meta.shapes[key] == (IMAGE_SIZE, IMAGE_SIZE, 3)

    item = dataset[0]
    assert item[OBS_STATE].shape == (config.n_obs_steps, STATE_DIM)
    assert item[ACTION].shape == (config.horizon, LIBERO_ACTION_DIM)
    assert item["action_is_pad"].shape == (config.horizon,)
    for key in LIBERO_IMAGE_KEYS:
        assert item[key].shape == (config.n_obs_steps, 3, IMAGE_SIZE, IMAGE_SIZE)

    policy = make_policy(config, ds_meta=dataset.meta)
    policy.train()
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    optimizer, scheduler = make_optimizer_and_scheduler(train_cfg, policy)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=True)
    batch = next(iter(dataloader))
    processed = preprocessor(batch)

    assert processed[OBS_STATE].shape == (2, config.n_obs_steps, STATE_DIM)
    assert processed[ACTION].shape == (2, config.horizon, LIBERO_ACTION_DIM)
    for key in LIBERO_IMAGE_KEYS:
        assert processed[key].shape == (2, config.n_obs_steps, 3, IMAGE_SIZE, IMAGE_SIZE)

    loss, _ = policy.forward(processed)
    assert loss.shape == ()
    assert torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )
