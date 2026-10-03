"""Offline tests of the real LPWM encoder (no mocked neural-network backbone)."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import (
    get_policy_class,
    make_policy,
    make_policy_config,
    make_pre_post_processors,
)
from lerobot.policies.lpwm_imf.configuration_lpwm_imf import LPWMIMFConfig
from lerobot.policies.lpwm_imf.modeling_lpwm_imf import LPWMIMFPolicy, LPWMVisualEncoder

CAMERA = "observation.images.front"


def tiny_config(**overrides):
    kwargs = {
        "device": "cpu",
        "push_to_hub": False,
        "n_obs_steps": 2,
        "image_size": 32,
        "patch_size": 16,
        "n_kp_prior": 4,
        "n_kp_enc": 4,
        "n_kp_dec": 4,
        "obj_ch_mult_prior": (1, 2),
        "obj_ch_mult": (1, 2),
        "bg_ch_mult": (1, 2, 4),
        "obj_base_ch": 8,
        "obj_final_cnn_ch": 8,
        "bg_base_ch": 8,
        "bg_final_cnn_ch": 8,
        "mlp_hidden_dim": 32,
        "dropout": 0.0,
        "input_features": {CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))},
    }
    kwargs.update(overrides)
    return LPWMIMFConfig(**kwargs)


def test_registered_config_and_policy_without_actions_or_state():
    config = make_policy_config("lpwm-imf", device="cpu")
    assert isinstance(config, LPWMIMFConfig)
    assert get_policy_class("lpwm-imf") is LPWMIMFPolicy
    assert config.n_kp_enc == config.n_kp_prior == 64
    assert config.n_kp_dec == 30
    assert config.latent_dim == 10
    assert config.action_delta_indices is None
    assert config.reward_delta_indices is None
    assert config.observation_delta_indices == [-1, 0]
    assert config.encoder_kwargs()["context_dim"] == 0
    LPWMIMFPolicy(tiny_config())


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((2, 3, 32, 32), (2, 1, 1, 4, 10)),
        ((2, 2, 3, 32, 32), (2, 2, 1, 4, 10)),
        ((2, 2, 3, 3, 32, 32), (2, 2, 3, 4, 10)),
    ],
)
def test_real_encoder_shapes_and_field_packing(shape, expected):
    encoder = LPWMVisualEncoder(tiny_config()).eval()
    result = encoder(torch.rand(shape))
    assert result.z.shape == expected
    assert result.background.shape == (*expected[:3], 4)
    torch.testing.assert_close(result.z[..., :2], result.position)
    torch.testing.assert_close(result.z[..., 2:4], result.scale)
    torch.testing.assert_close(result.z[..., 4:5], result.depth)
    torch.testing.assert_close(result.z[..., 5:6], result.presence)
    torch.testing.assert_close(result.z[..., 6:], result.features)
    assert torch.isfinite(result.z).all()
    assert torch.isfinite(result.background).all()
    assert ((result.presence >= 0) & (result.presence <= 1)).all()
    assert encoder.encoder.ctx_enc is None
    assert encoder.encoder.particle_inter_enc is not None


def test_factory_builds_encoder_from_dataset_features():
    config = tiny_config(input_features={})
    dataset_meta = SimpleNamespace(
        features={
            CAMERA: {"dtype": "image", "shape": (3, 32, 32), "names": ["channels", "height", "width"]},
            "action": {"dtype": "float32", "shape": (7,), "names": [f"action_{i}" for i in range(7)]},
        },
        stats={},
    )
    policy = make_policy(config, ds_meta=dataset_meta)
    assert isinstance(policy, LPWMIMFPolicy)
    assert policy.config.action_feature.shape == (7,)
    result = policy.eval().encode_observation({CAMERA: torch.rand(1, 3, 32, 32)})
    assert result.z.shape == (1, 1, 1, 4, 10)


def test_wrapper_preserves_upstream_fields_exactly():
    encoder = LPWMVisualEncoder(tiny_config()).eval()
    images = torch.rand(2, 2, 3, 32, 32)
    with torch.no_grad():
        upstream = encoder.encoder(images, deterministic=True)
        actual = encoder(images)
    for field, key in [
        ("position", "z"),
        ("scale", "z_scale"),
        ("depth", "z_depth"),
        ("presence", "obj_on"),
        ("features", "z_features"),
        ("background", "z_bg_features"),
    ]:
        torch.testing.assert_close(getattr(actual, field).squeeze(2), upstream[key], rtol=0, atol=0)


def test_multiview_axes_and_camera_order_are_preserved():
    encoder = LPWMVisualEncoder(tiny_config()).eval()
    images = torch.rand(2, 2, 3, 3, 32, 32)
    with torch.no_grad():
        combined = encoder(images)
        for view in range(3):
            separate = encoder(images[:, :, view])
            torch.testing.assert_close(combined.z[:, :, view], separate.z[:, :, 0], rtol=2e-4, atol=2e-6)
            torch.testing.assert_close(
                combined.background[:, :, view], separate.background[:, :, 0], rtol=2e-4, atol=2e-6
            )


def test_policy_supports_distinct_camera_resolutions_without_state():
    config = tiny_config(
        input_features={
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 24, 40)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 40, 24)),
        }
    )
    policy = LPWMIMFPolicy(config).eval()
    batch = {CAMERA: torch.rand(2, 2, 3, 24, 40), "observation.images.wrist": torch.rand(2, 2, 3, 40, 24)}
    result = policy.encode_observation(batch)
    assert result.z.shape == (2, 2, 2, 4, 10)
    torch.testing.assert_close(result.z[:, :, :1], policy.model(batch[CAMERA]).z)


def test_uint8_and_float_inputs_match_and_are_not_mutated():
    encoder = LPWMVisualEncoder(tiny_config()).eval()
    images = torch.randint(0, 256, (2, 3, 20, 40), dtype=torch.uint8)
    original = images.clone()
    with torch.no_grad():
        integer_result = encoder(images)
        float_result = encoder(images.float() / 255.0)
    torch.testing.assert_close(integer_result.z, float_result.z, rtol=0, atol=0)
    assert torch.equal(images, original)
    assert encoder.prepare_images(images).shape == (2, 3, 32, 32)


def test_optional_minus_one_one_preprocessing():
    encoder = LPWMVisualEncoder(tiny_config(normalize_rgb=True))
    assert torch.equal(encoder.prepare_images(torch.zeros(1, 3, 32, 32)), -torch.ones(1, 3, 32, 32))
    assert torch.equal(encoder.prepare_images(torch.ones(1, 3, 32, 32)), torch.ones(1, 3, 32, 32))


def test_deterministic_posterior_and_explicit_sampling():
    encoder = LPWMVisualEncoder(tiny_config()).eval()
    images = torch.rand(2, 2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(encoder(images).z, encoder(images).z, rtol=0, atol=0)
        torch.manual_seed(11)
        sample_1 = encoder(images, deterministic=False)
        torch.manual_seed(12)
        sample_2 = encoder(images, deterministic=False)
    assert not torch.equal(sample_1.features, sample_2.features)
    assert not torch.equal(sample_1.background, sample_2.background)


def test_encoder_and_rgb_receive_gradients():
    encoder = LPWMVisualEncoder(tiny_config()).train()
    images = torch.rand(2, 2, 3, 32, 32, requires_grad=True)
    result = encoder(images)
    (result.z.square().mean() + result.background.square().mean()).backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    for module in (
        encoder.encoder.particle_enc,
        encoder.encoder.bg_encoder,
        encoder.encoder.particle_inter_enc,
    ):
        grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert any(torch.count_nonzero(grad) for grad in grads)


def test_freeze_encoder_survives_parent_train():
    policy = LPWMIMFPolicy(tiny_config(freeze_encoder=True)).train()
    assert not policy.model.encoder.training
    assert not any(parameter.requires_grad for parameter in policy.parameters())
    assert list(policy.get_optim_params()) == []
    result = policy.encode_observation({CAMERA: torch.rand(2, 2, 3, 32, 32)})
    assert not result.z.requires_grad


@pytest.mark.parametrize("method", ["forward", "predict_action_chunk", "select_action"])
def test_incomplete_action_policy_fails_explicitly(method):
    policy = LPWMIMFPolicy(tiny_config())
    with pytest.raises(NotImplementedError, match="No IMF action head"):
        getattr(policy, method)({})


@pytest.mark.parametrize(
    "options",
    [
        {"image_size": 30},
        {"n_kp_prior": 3},
        {"n_kp_enc": 3},
        {"n_kp_dec": 5},
        {"n_obs_steps": 0},
        {"anchor_s": 0},
        {"anchor_s": 1},
        {"dropout": 1},
        {"pint_enc_heads": 3},
        {"obj_base_ch": 3},
        {"bg_ch_mult": ()},
        {"attn_norm_type": "bad"},
        {"normalization_mapping": {"VISUAL": NormalizationMode.MEAN_STD}},
    ],
)
def test_invalid_config_rejected(options):
    with pytest.raises(ValueError):
        tiny_config(**options)


@pytest.mark.parametrize(
    "images",
    [
        torch.ones(1, 3, 32, 32) * 255.0,
        torch.ones(1, 3, 32, 32) * -0.1,
        torch.full((1, 3, 32, 32), float("nan")),
        torch.zeros(1, 4, 32, 32),
        torch.zeros(1, 3, 3, 32, 32),  # Too many time steps.
        torch.zeros(0, 3, 32, 32),
    ],
)
def test_invalid_rgb_rejected(images):
    with pytest.raises(ValueError):
        LPWMVisualEncoder(tiny_config())(images)


def test_non_rgb_integer_dtype_rejected():
    with pytest.raises(TypeError):
        LPWMVisualEncoder(tiny_config())(torch.zeros(1, 3, 32, 32, dtype=torch.int64))


def test_missing_and_misaligned_camera_inputs_rejected():
    with pytest.raises(ValueError, match="VISUAL"):
        LPWMIMFPolicy(tiny_config(input_features={}))
    config = tiny_config()
    config.input_features["observation.images.wrist"] = config.input_features[CAMERA]
    policy = LPWMIMFPolicy(config)
    with pytest.raises(KeyError, match="Missing"):
        policy.encode_observation({})
    with pytest.raises(ValueError, match="batch and time"):
        policy.encode_observation(
            {CAMERA: torch.rand(2, 2, 3, 32, 32), "observation.images.wrist": torch.rand(1, 2, 3, 32, 32)}
        )


def test_processors_keep_raw_rgb_and_config_roundtrip(tmp_path):
    config = tiny_config()
    pre, post = make_pre_post_processors(config)
    image = torch.rand(3, 32, 32)
    batch = pre({CAMERA: image})
    torch.testing.assert_close(batch[CAMERA], image.unsqueeze(0), rtol=0, atol=0)
    assert LPWMIMFPolicy(config).eval().encode_observation(batch).z.shape == (1, 1, 1, 4, 10)
    config.save_pretrained(tmp_path)
    restored = LPWMIMFConfig.from_pretrained(tmp_path)
    assert restored.type == "lpwm-imf"
    assert restored.encoder_kwargs() == config.encoder_kwargs()
    assert post is not None


def test_policy_checkpoint_roundtrip(tmp_path):
    policy = LPWMIMFPolicy(tiny_config()).eval()
    images = {CAMERA: torch.rand(2, 2, 3, 32, 32)}
    with torch.no_grad():
        before = policy.encode_observation(images)
    policy.save_pretrained(tmp_path)
    restored = LPWMIMFPolicy.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        after = restored.encode_observation(images)
    torch.testing.assert_close(before.z, after.z, rtol=0, atol=0)
    torch.testing.assert_close(before.background, after.background, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["bare", "lpwm", "ddp", "safetensors"])
def test_strict_local_lpwm_encoder_loading(tmp_path, layout):
    source = LPWMVisualEncoder(tiny_config()).eval()
    target = LPWMVisualEncoder(tiny_config()).eval()
    state = source.encoder.state_dict()
    if layout in {"lpwm", "ddp"}:
        state = {f"encoder_module.{key}": value for key, value in state.items()}
        state["decoder_module.unused"] = torch.zeros(1)
        state["encoder_module.ctx_enc.unused"] = torch.zeros(1)
        if layout == "ddp":
            state = {f"module.{key}": value for key, value in state.items()}
        state = {"state_dict": state}
    if layout == "safetensors":
        from safetensors.torch import save_file

        path = tmp_path / "encoder.safetensors"
        save_file({key: value.clone() for key, value in state.items()}, str(path))
    else:
        path = tmp_path / "lpwm.pth"
        torch.save(state, path)
    target.load_lpwm_encoder(path)
    images = torch.rand(1, 2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(target(images).z, source(images).z, rtol=0, atol=0)


def test_bad_checkpoint_fails_without_partial_parameter_changes():
    encoder = LPWMVisualEncoder(tiny_config())
    before = {key: value.clone() for key, value in encoder.encoder.state_dict().items()}
    wrong = dict(before)
    wrong.pop(next(iter(wrong)))
    with pytest.raises(ValueError, match="missing="):
        encoder.load_lpwm_encoder(wrong)
    wrong = dict(before)
    wrong[next(iter(wrong))] = torch.zeros(100, 100)
    with pytest.raises(ValueError, match="shape_mismatch="):
        encoder.load_lpwm_encoder(wrong)
    for key, value in encoder.encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with pytest.raises(FileNotFoundError):
        encoder.load_lpwm_encoder("/nonexistent/lpwm-test-checkpoint.pth")


def test_vendor_excludes_world_model_and_heavy_runtime_imports():
    import lerobot.policies.lpwm_imf._vendor as vendor

    for path in Path(vendor.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert node.name not in {"DLPDynamics", "DLPContext", "DLPDecoder", "Decoder", "LPIPS"}
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".")[0] not in {"matplotlib", "cv2", "requests", "transformers"}
                    for alias in node.names
                )
