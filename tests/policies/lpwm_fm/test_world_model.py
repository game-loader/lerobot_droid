"""CPU tests exercise native LPWM networks/losses, never mock visual backbones."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.lpwm_fm._vendor.modules import DLPContext, DLPDynamics
from lerobot.policies.lpwm_fm.world_model import ActionTokenDynamics, LPWMWorldModel, native_reference
from lerobot.policies.lpwm_imf.configuration_lpwm_imf import LPWMIMFConfig


def tiny_config(**updates):
    config = LPWMIMFConfig(
        device="cpu",
        push_to_hub=False,
        image_size=32,
        patch_size=16,
        n_kp_prior=4,
        n_kp_enc=4,
        n_kp_dec=4,
        obj_ch_mult_prior=(1, 2),
        obj_ch_mult=(1, 2),
        bg_ch_mult=(1, 2, 4),
        obj_base_ch=8,
        obj_final_cnn_ch=8,
        bg_base_ch=8,
        bg_final_cnn_ch=8,
        mlp_hidden_dim=32,
        dropout=0.0,
    )
    values = {
        **vars(config),
        "hidden_dim": 32,
        "world_layers": 1,
        "n_heads": 4,
        "rec_weight": 1.0,
        "prior_weight": 0.01,
        "dyn_weight": 0.1,
    }
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(old)


def gradient_sum(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    return sum(g.abs().sum().item() for g in grads)


def test_real_shapes_and_exact_native_field_packing():
    model = LPWMWorldModel(tiny_config()).eval()
    images = torch.rand(2, 3, 2, 3, 32, 32)
    result = model.encode(images)
    assert result["particles"].shape == (2, 3, 2, 4, 10)
    assert result["background"].shape == (2, 3, 2, 4)
    raw = model.encoder(images.reshape(-1, 1, 3, 32, 32), deterministic=True)
    expected = torch.cat([raw[name] for name in ("z", "z_scale", "z_depth", "obj_on", "z_features")], -1)
    torch.testing.assert_close(result["particles"], expected.reshape(2, 3, 2, 4, 10), rtol=0, atol=0)
    torch.testing.assert_close(result["background"], raw["z_bg_features"].reshape(2, 3, 2, 4), rtol=0, atol=0)
    assert model.encoder.ctx_enc is None


@pytest.mark.parametrize("normalized", [False, True])
def test_shared_encoder_receives_same_rgb_values_and_order(normalized):
    model = LPWMWorldModel(tiny_config(normalize_rgb=normalized)).eval()
    images = torch.rand(1, 3, 2, 3, 32, 32)
    seen = []
    hook = model.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    model.encode(images)
    model.world_loss(images, torch.rand(1, 2, 7))
    hook.remove()
    # Native DLP calls encode_all directly, so instrument its actual prior CNN
    # separately below; this hook is present for public encode.
    expected = images.reshape(-1, 1, 3, 32, 32)
    if normalized:
        expected = 2 * expected - 1
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
    seen = []
    hook = model.encoder.prior_encoder.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone())
    )
    model.encode(images)
    model.world_loss(images, torch.rand(1, 2, 7))
    hook.remove()
    assert len(seen) == 2
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)


def test_encode_has_no_future_or_cross_camera_leakage():
    model = LPWMWorldModel(tiny_config()).eval()
    images = torch.rand(1, 3, 2, 3, 32, 32)
    before = model.encode(images)
    changed = images.clone()
    changed[:, 2] = torch.rand_like(changed[:, 2])
    changed[:, :, 1] = torch.rand_like(changed[:, :, 1])
    after = model.encode(changed)
    for key in ("particles", "background"):
        torch.testing.assert_close(before[key][:, :2, 0], after[key][:, :2, 0], rtol=0, atol=0)
    alone = model.encode(images[:, :1, :1])
    torch.testing.assert_close(before["particles"][:, :1, :1], alone["particles"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_dec", [2, 4])
def test_real_world_loss_backpropagates_to_encoder_decoder_and_dynamics(n_dec):
    model = LPWMWorldModel(tiny_config(n_kp_dec=n_dec))
    loss, metrics = model.world_loss(torch.rand(1, 3, 2, 3, 32, 32), torch.rand(1, 2, 7))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert metrics["world_rec"] > 0 and metrics["world_dyn"] > 0 and metrics["world_prior"] > 0
    loss.backward()
    assert gradient_sum(model.encoder) > 0
    assert gradient_sum(model.decoder) > 0
    assert gradient_sum(model.dynamics) > 0


@pytest.mark.parametrize("component", ["rec", "prior", "dyn"])
def test_each_native_loss_component_trains_shared_encoder(component):
    weights = {"rec_weight": 0.0, "prior_weight": 0.0, "dyn_weight": 0.0}
    weights[f"{component}_weight"] = 1.0
    model = LPWMWorldModel(tiny_config(**weights))
    loss, _ = model.world_loss(torch.rand(1, 2, 1, 3, 32, 32), torch.rand(1, 1, 7))
    loss.backward()
    assert gradient_sum(model.encoder) > 0


def test_exactly_three_identical_action_tokens_and_no_modulated_normalization():
    model = LPWMWorldModel(tiny_config()).eval()
    latent = model.encode(torch.rand(1, 2, 2, 3, 32, 32))
    tokens = model.dynamics.build_tokens(latent["particles"], latent["background"], torch.rand(1, 2, 7))
    assert tokens.shape == (1, 2, 2 * 5 + 3, 32)
    torch.testing.assert_close(tokens[:, :, -1], tokens[:, :, -2], rtol=0, atol=0)
    torch.testing.assert_close(tokens[:, :, -1], tokens[:, :, -3], rtol=0, atol=0)
    assert all(
        "adaln" not in type(m).__name__.lower() and "film" not in type(m).__name__.lower()
        for m in model.modules()
    )
    with pytest.raises(ValueError, match="3 identical"):
        LPWMWorldModel(tiny_config(action_token_repeat=1))


def test_action_sensitivity_and_frame_causality():
    model = LPWMWorldModel(tiny_config()).eval()
    z = model.encode(torch.rand(1, 3, 2, 3, 32, 32))
    actions = torch.zeros(1, 3, 7)
    before = model.dynamics(z["particles"], z["background"], actions)
    actions[:, 1] = 1.0
    after = model.dynamics(z["particles"], z["background"], actions)
    for key in before:
        torch.testing.assert_close(before[key][:, 0], after[key][:, 0], rtol=0, atol=0)
    assert not torch.allclose(before["mu_position"][:, 1], after["mu_position"][:, 1])
    particles = z["particles"].clone()
    particles[:, 2] += 3
    changed = model.dynamics(particles, z["background"], actions)
    for key in changed:
        torch.testing.assert_close(changed[key][:, :2], after[key][:, :2], rtol=0, atol=0)
    mask = ActionTokenDynamics.causal_mask(3, 13)
    assert not mask[:13, :13].any() and mask[:13, 13:].all()


def test_gt_actions_exact_transition_indexing_and_no_action_gradient():
    model = LPWMWorldModel(tiny_config()).eval()
    images = torch.rand(1, 3, 2, 3, 32, 32)
    actions = torch.arange(14, dtype=torch.float).reshape(1, 2, 7).requires_grad_()
    captured = []
    hook = model.dynamics.register_forward_pre_hook(
        lambda module, args: captured.append(tuple(x.detach().clone() for x in args))
    )
    # Match the posterior random draws so source frame equality is exact.
    torch.manual_seed(89)
    source = model.encode(images, deterministic=False)
    torch.manual_seed(89)
    loss, _ = model.world_loss(images, actions)
    hook.remove()
    particles, background, clean = captured[0]
    torch.testing.assert_close(particles, source["particles"][:, :-1], rtol=0, atol=0)
    torch.testing.assert_close(background, source["background"][:, :-1], rtol=0, atol=0)
    torch.testing.assert_close(clean, actions.detach(), rtol=0, atol=0)
    loss.backward()
    assert actions.grad is None
    assert gradient_sum(model.dynamics.action_projection) > 0


def test_dynamics_targets_are_next_frame_native_posterior():
    model = LPWMWorldModel(tiny_config()).eval()
    z = model.encode(torch.rand(1, 3, 2, 3, 32, 32))
    # Matching q at t+1 exactly must make every Gaussian/Beta KL zero.
    prediction = {"mu_position": z["mu_tot"][:, 1:], "logvar_position": z["logvar_offset"][:, 1:]}
    for key, native in (
        ("scale", "scale"),
        ("depth", "depth"),
        ("features", "features"),
        ("background", "bg_features"),
    ):
        for statistic in ("mu", "logvar"):
            prediction[f"{statistic}_{key}"] = z[f"{statistic}_{native}"][:, 1:]
    for key in ("obj_on_a", "obj_on_b"):
        prediction[key] = z[key][:, 1:]
    assert model._dynamic_loss(z, prediction).abs() < 1e-6


def test_complete_native_context_prior_dynamics_loss_backward_and_sampling():
    model = native_reference(
        tiny_config(),
        timestep_horizon=2,
        context_dim=4,
        pint_dim=32,
        pint_dyn_layers=1,
        pint_dyn_heads=4,
        pint_ctx_layers=1,
        pint_ctx_heads=4,
    )
    assert isinstance(model.ctx_module, DLPContext)
    assert isinstance(model.dyn_module, DLPDynamics)
    images = torch.rand(1, 3, 3, 32, 32)
    output = model(images, deterministic=True, with_loss=True)
    assert output["mu_context"].shape == (1, 3, 5, 4)
    assert output["mu_dyn"].shape == (1, 2, 4, 2)
    losses = output["loss_dict"]
    assert losses["loss_kl_context"] > 0 and torch.isfinite(losses["loss"])
    losses["loss"].backward()
    for module in (model.encoder_module, model.decoder_module, model.ctx_module, model.dyn_module):
        assert gradient_sum(module) > 0
    with torch.no_grad():
        sample = model.sample_from_x(images, num_steps=2, cond_steps=1, deterministic=True)
    assert sample.shape == images.shape and torch.isfinite(sample).all()


def test_native_sources_retained_without_unrelated_plotting_dependencies():
    vendor = Path(__file__).parents[3] / "src/lerobot/policies/lpwm_fm/_vendor"
    assert "MIT License" in (vendor / "LICENSE").read_text()
    utilities = ast.parse((vendor / "util_func.py").read_text())
    imports = [n.module for n in ast.walk(utilities) if isinstance(n, ast.ImportFrom)]
    imports += [alias.name for n in ast.walk(utilities) if isinstance(n, ast.Import) for alias in n.names]
    assert not any(name and name.startswith(("matplotlib", "cv2", "imageio")) for name in imports)


def test_input_validation_and_resize():
    model = LPWMWorldModel(tiny_config())
    assert model.encode(torch.rand(1, 2, 1, 3, 48, 48))["particles"].shape == (1, 2, 1, 4, 10)
    with pytest.raises(ValueError, match="GT actions"):
        model.world_loss(torch.rand(1, 2, 1, 3, 32, 32), torch.rand(1, 2, 7))
    with pytest.raises(ValueError, match="\\[0,1\\]"):
        model.encode(torch.full((1, 2, 1, 3, 32, 32), 2.0))
    with pytest.raises(ValueError, match="one-step"):
        LPWMWorldModel(tiny_config(world_horizon=4))
