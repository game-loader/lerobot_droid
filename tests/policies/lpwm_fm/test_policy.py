"""CPU integration tests of the real LPWM/DLP FM policy (no placeholder visual encoders)."""

import inspect
import json

import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.policies.lpwm_fm.configuration_lpwm_fm import LPWMFMConfig
from lerobot.policies.lpwm_fm.modeling_lpwm_fm import LANGUAGE, LANGUAGE_MASK, STATE, LPWMFMPolicy
from lerobot.policies.lpwm_fm.processor_lpwm_fm import make_lpwm_fm_pre_post_processors

CAMERAS = ("observation.images.agent", "observation.images.wrist")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(**overrides):
    """Reduced channel counts, but the complete real DLP encoder/decoder and world model."""
    options = {
        "device": "cpu",
        "push_to_hub": False,
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
        "hidden_dim": 32,
        "n_heads": 4,
        "scene_n_layers": 2,
        "expert_n_layers": 1,
        "world_hidden_dim": 32,
        "world_n_heads": 4,
        "world_n_layers": 1,
        "language_dim": 12,
        "horizon": 4,
        "n_action_steps": 2,
        "num_inference_steps": 2,
        "input_features": {
            **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)) for key in CAMERAS},
            STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    }
    options.update(overrides)
    return LPWMFMConfig(**options)


def make_batch(b=2):
    return {
        **{key: torch.rand(b, 2, 3, 32, 32) for key in CAMERAS},
        STATE: torch.randn(b, 2, 8),
        LANGUAGE: torch.randn(b, 3, 12),
        LANGUAGE_MASK: torch.tensor([[True, True, False]]).expand(b, -1),
        "action": torch.randn(b, 4, 7),
        "action_is_pad": torch.tensor([[False, False, False, True]]).expand(b, -1),
        "world.images": torch.rand(b, 3, 2, 3, 32, 32),
        "world.actions": torch.randn(b, 2, 7),
    }


def test_config_and_shared_initialization():
    config = tiny_config()
    assert config.type == "lpwm-fm"
    assert config.action_delta_indices == [0, 1, 2, 3]
    assert config.observation_delta_indices == [-1, 0]
    assert config.normalization_mapping["ACTION"] == NormalizationMode.IDENTITY
    assert config.world_layers == config.world_n_layers
    assert config.rec_weight == config.reconstruction_weight
    assert config.dyn_weight == config.dynamics_weight
    torch.manual_seed(23)
    a = LPWMFMPolicy(tiny_config(variant="A", world_weight=0))
    torch.manual_seed(23)
    b = LPWMFMPolicy(tiny_config(variant="B", world_weight=1))
    assert a.state_dict().keys() == b.state_dict().keys()
    for name, parameter in a.state_dict().items():
        torch.testing.assert_close(parameter, b.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "options",
    [
        {"variant": "C"},
        {"hidden_dim": 31},
        {"n_action_steps": 5},
        {"world_weight": -1},
        {"dynamics_weight": float("nan")},
        {"freeze_encoder": True},
        {"num_inference_steps": 0},
        {"language_dim": 0},
    ],
)
def test_config_rejects_invalid(options):
    with pytest.raises(ValueError):
        tiny_config(**options)


def test_ordinary_fm_formula_and_padding_no_jvp(monkeypatch):
    policy = LPWMFMPolicy(tiny_config()).eval()
    batch = make_batch()
    tau = torch.tensor([0.2, 0.8])
    noise = torch.randn_like(batch["action"])
    captured = {}

    def capture(module, args):
        captured["noisy"] = args[0].detach().clone()
        captured["tau"] = args[1].detach().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Ordinary FM must not invoke JVP.")

    monkeypatch.setattr(torch.func, "jvp", forbidden)
    monkeypatch.setattr(torch.autograd.functional, "jvp", forbidden)
    policy.expert.output[-1].weight.data.zero_()
    policy.expert.output[-1].bias.data.zero_()
    hook = policy.expert.register_forward_pre_hook(capture)
    loss, _ = policy.flow_matching_loss(batch, noise=noise, tau=tau)
    hook.remove()
    keep = ~batch["action_is_pad"]
    actions = torch.where(keep[..., None], batch["action"], torch.zeros_like(batch["action"]))
    expected_noisy = (1 - tau[:, None, None]) * actions + tau[:, None, None] * noise
    torch.testing.assert_close(captured["noisy"], expected_noisy)
    expected_loss = ((noise - actions).square() * keep[..., None]).sum() / (keep.sum() * 7)
    torch.testing.assert_close(loss, expected_loss)
    batch["action"] = batch["action"].masked_fill(batch["action_is_pad"][..., None], float("nan"))
    second, _ = policy.flow_matching_loss(batch, noise=noise, tau=tau)
    torch.testing.assert_close(loss, second)
    loss.backward()
    assert policy.expert.output[-1].weight.grad is not None


@pytest.mark.parametrize("key", [STATE, LANGUAGE])
def test_state_and_language_required(key):
    policy = LPWMFMPolicy(tiny_config()).eval()
    batch = make_batch()
    del batch[key]
    with pytest.raises(KeyError, match=key):
        policy(batch)


def test_language_mask_and_no_future_label_leakage():
    policy = LPWMFMPolicy(tiny_config()).eval()
    batch = make_batch()
    noise = torch.randn(2, 4, 7)
    first = policy.predict_action_chunk(batch, noise=noise)
    modified = dict(batch)
    modified[LANGUAGE] = batch[LANGUAGE].clone()
    modified[LANGUAGE][:, 2] += 1000  # masked language must not affect output
    modified["world.images"] = 1 - batch["world.images"]
    modified["world.actions"] = -batch["world.actions"]
    modified["action"] = -batch["action"]
    second = policy.predict_action_chunk(modified, noise=noise)
    assert first.shape == (2, 4, 7)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    modified[LANGUAGE_MASK] = torch.zeros_like(batch[LANGUAGE_MASK])
    with pytest.raises(ValueError, match="unmasked"):
        policy.predict_action_chunk(modified)


def test_real_encoder_and_scene_frame_causality():
    policy = LPWMFMPolicy(tiny_config()).eval()
    batch = make_batch()
    images = policy._images(batch)
    with torch.no_grad():
        first = policy.world_model.encode(images)
        scene1 = policy.scene_encoder(first["particles"], first["background"], batch[STATE])
        images[:, 1] = 1 - images[:, 1]
        second = policy.world_model.encode(images)
        state2 = batch[STATE].clone()
        state2[:, 1] += 20
        scene2 = policy.scene_encoder(second["particles"], second["background"], state2)
    torch.testing.assert_close(first["particles"][:, 0], second["particles"][:, 0], rtol=0, atol=0)
    per_frame = scene1.shape[1] // 2
    torch.testing.assert_close(scene1[:, :per_frame], scene2[:, :per_frame], rtol=0, atol=0)
    assert not torch.allclose(scene1[:, per_frame:], scene2[:, per_frame:])


def test_camera_insertion_order_and_shared_weights():
    policy = LPWMFMPolicy(tiny_config()).eval()
    batch = make_batch()
    images = policy._images(batch)
    assert images.shape == (2, 2, 2, 3, 32, 32)
    torch.testing.assert_close(images[:, :, 0], batch[CAMERAS[0]])
    torch.testing.assert_close(images[:, :, 1], batch[CAMERAS[1]])
    with torch.no_grad():
        first = policy.world_model.encode(images)
        reversed_views = policy.world_model.encode(images.flip(2))
    torch.testing.assert_close(first["particles"], reversed_views["particles"].flip(2))
    torch.testing.assert_close(first["background"], reversed_views["background"].flip(2))


@pytest.mark.parametrize("variant,weight", [("A", 1.0), ("A", 0.0), ("B", 0.0)])
def test_a_or_zero_weight_skips_world_loss(monkeypatch, variant, weight):
    policy = LPWMFMPolicy(tiny_config(variant=variant, world_weight=weight))

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled world loss must not be evaluated.")

    monkeypatch.setattr(policy.world_model, "world_loss", forbidden)
    batch = make_batch()
    del batch["world.images"], batch["world.actions"]
    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert "world_loss" not in metrics


def test_gt_world_routing_and_both_losses_reach_same_real_encoder(monkeypatch):
    policy = LPWMFMPolicy(tiny_config(variant="B", world_weight=0.1)).train()
    batch = make_batch()
    batch["world.actions"].requires_grad_(True)
    real_world_loss = policy.world_model.world_loss
    captured = {}

    def checked(images, actions, current_step=None):
        torch.testing.assert_close(actions, batch["world.actions"], rtol=0, atol=0)
        assert not actions.requires_grad
        captured["step"] = current_step
        result = real_world_loss(images, actions, current_step=current_step)
        captured["loss"] = result[0]
        return result

    monkeypatch.setattr(policy.world_model, "world_loss", checked)
    loss, metrics = policy(batch, current_step=7)
    assert captured["step"] == 7
    torch.testing.assert_close(loss.detach(), metrics["fm_loss"] + 0.1 * metrics["world_loss"])
    encoder_parameters = [p for p in policy.world_model.encoder.parameters() if p.requires_grad]
    world_grads = torch.autograd.grad(
        captured["loss"], encoder_parameters, retain_graph=True, allow_unused=True
    )
    action_grads = torch.autograd.grad(
        loss - 0.1 * captured["loss"], encoder_parameters, retain_graph=True, allow_unused=True
    )
    for gradients in (world_grads, action_grads):
        assert any(g is not None and g.abs().max() > 0 for g in gradients)
        assert all(g is None or torch.isfinite(g).all() for g in gradients)
    loss.backward()
    assert batch["world.actions"].grad is None
    assert any(
        p.grad is not None and p.grad.abs().max() > 0 for p in policy.world_model.dynamics.parameters()
    )
    assert all(not value.requires_grad for value in metrics.values() if isinstance(value, torch.Tensor))


def test_b_world_input_required_and_episode_padding_rejected():
    policy = LPWMFMPolicy(tiny_config(variant="B"))
    batch = make_batch()
    batch["world.actions_is_pad"] = torch.tensor([[False, True], [False, False]])
    with pytest.raises(ValueError, match="within-episode"):
        policy(batch)
    del batch["world.actions"]
    with pytest.raises(KeyError, match="clean GT"):
        policy(batch)


def test_world_schedule():
    policy = LPWMFMPolicy(tiny_config(variant="B", world_warmup_steps=10, world_ramp_steps=5))
    assert policy._world_scale(9) == 0
    assert policy._world_scale(10) == pytest.approx(0.2)
    assert policy._world_scale(14) == 1
    assert policy._world_scale(100) == 1
    with pytest.raises(ValueError, match="current_step"):
        policy._world_scale(None)


def test_euler_sign_and_queue_updates(monkeypatch):
    policy = LPWMFMPolicy(tiny_config()).eval()
    policy.expert.output[-1].weight.data.zero_()
    policy.expert.output[-1].bias.data.fill_(2)
    batch = make_batch()
    torch.testing.assert_close(
        policy.predict_action_chunk(batch, noise=torch.ones(2, 4, 7)), -torch.ones(2, 4, 7)
    )
    original_predict = policy.predict_action_chunk
    calls = []

    def counted(history, **kwargs):
        calls.append({key: history[key].clone() for key in (*CAMERAS, STATE)})
        return original_predict(history, **kwargs)

    monkeypatch.setattr(policy, "predict_action_chunk", counted)
    single = {key: value[:, -1] if key in (*CAMERAS, STATE) else value for key, value in batch.items()}
    for step in range(3):
        single[STATE] = torch.full((2, 8), float(step))
        action = policy.select_action(single, noise=torch.ones(2, 4, 7))
        assert action.shape == (2, 7)
    assert len(calls) == 2
    torch.testing.assert_close(calls[0][STATE], torch.zeros(2, 2, 8))
    torch.testing.assert_close(calls[1][STATE][:, 0], torch.ones(2, 8))
    torch.testing.assert_close(calls[1][STATE][:, 1], torch.full((2, 8), 2.0))
    policy.reset()
    assert not policy._action_queue
    assert all(not queue for queue in policy._observation_queues.values())


def test_serialization_round_trip(tmp_path):
    policy = LPWMFMPolicy(tiny_config()).eval()
    batch = make_batch()
    noise = torch.randn(2, 4, 7)
    expected = policy.predict_action_chunk(batch, noise=noise)
    policy.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["type"] == "lpwm-fm"
    config = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(config, LPWMFMConfig)
    restored = LPWMFMPolicy.from_pretrained(tmp_path, local_files_only=True, strict=True).eval()
    torch.testing.assert_close(expected, restored.predict_action_chunk(batch, noise=noise), rtol=0, atol=0)


def test_processor_normalization_language_and_world_preservation():
    config = tiny_config()
    stats = {STATE: {"mean": torch.ones(8), "std": torch.full((8,), 2.0)}}
    pre, post = make_lpwm_fm_pre_post_processors(config, stats)
    batch = make_batch()
    batch[STATE] = torch.ones(2, 2, 8)
    processed = pre(batch)
    torch.testing.assert_close(processed[STATE], torch.zeros(2, 2, 8))
    for key in ("action", LANGUAGE, LANGUAGE_MASK, "world.images", "world.actions", *CAMERAS):
        torch.testing.assert_close(processed[key], batch[key])
    torch.testing.assert_close(post(batch["action"]), batch["action"])
    raw = {
        CAMERAS[0]: torch.rand(3, 32, 32),
        CAMERAS[1]: torch.rand(3, 32, 32),
        STATE: torch.ones(8),
        LANGUAGE: torch.randn(3, 12),
        LANGUAGE_MASK: torch.ones(3, dtype=torch.bool),
    }
    assert pre(raw)[LANGUAGE].shape == (1, 3, 12)


def test_expert_source_has_no_adaptive_conditioning():
    from lerobot.policies.lpwm_fm.modeling_lpwm_fm import AttentionBlock, FlowActionExpert

    source = inspect.getsource(AttentionBlock) + inspect.getsource(FlowActionExpert)
    assert "jvp(" not in source
    assert "AdaLN" not in source
    assert "FiLM" not in source
