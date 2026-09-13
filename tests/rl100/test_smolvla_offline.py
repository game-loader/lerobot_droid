"""CPU tests use a tiny actor; optional real checkpoint smoke runs through the CLI."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

pytest.importorskip("transformers", exc_type=ModuleNotFoundError)

from transformers import DynamicCache

from RL.smolvla.adapter import make_adapters
from RL.smolvla.checkpoint import restore_checkpoint, save_checkpoint, validate_checkpoint
from RL.smolvla.data import verified_manifest_labels
from RL.smolvla.features import TOKEN_KEY
from RL.smolvla.flow import FlowConfig, flow_mean_std, gaussian_log_prob
from RL.smolvla.trainer import EncodedBatch, SmolVLAOfflineTrainer


class TinyVLMWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = nn.Linear(4, 4)
        self.lm_expert = nn.Linear(4, 4)
        self.config = SimpleNamespace(text_config=SimpleNamespace(head_dim=2))

    def forward(self, *, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache):
        prefix = inputs_embeds[0]
        tokens = self.vlm(prefix)
        states = tokens.reshape(*tokens.shape[:2], 2, 2)
        cache = DynamicCache()
        for i in range(2):
            cache.update((states + i).transpose(1, 2), (states - i).transpose(1, 2), i)
        return [tokens, None], cache


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm_with_expert = TinyVLMWrapper()
        self.state_proj = nn.Linear(2, 4)
        self.action_in_proj = nn.Linear(4, 4)
        self.action_out_proj = nn.Linear(4, 4)

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        context = past_key_values.layers[0].values.flatten(1).mean(1)[:, None, None]
        return self.action_out_proj(self.vlm_with_expert.lm_expert(self.action_in_proj(x_t))) + context * 0.05


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyModel()
        self.config = SimpleNamespace(
            n_obs_steps=1,
            rtc_config=None,
            adapt_to_pi_aloha=False,
            use_cache=True,
            pad_language_to="max_length",
            chunk_size=4,
            n_action_steps=2,
            action_feature=SimpleNamespace(shape=(2,)),
            max_action_dim=4,
        )

    def save_pretrained(self, path):
        path.mkdir(parents=True)
        torch.save(self.state_dict(), path / "model.safetensors")
        (path / "config.json").write_text("{}")


class DummyProcessor:
    def __init__(self, name):
        self.name = name

    def save_pretrained(self, path):
        (path / f"{self.name}.json").write_text("{}")


@pytest.fixture
def setup():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    encoder, current, behavior = make_adapters(TinyPolicy(), None, FlowConfig(3, 0.2))
    c = encoder.from_prefix(
        torch.randn(3, 4, 4),
        torch.tensor([[True, True, False, True]] * 3),
        torch.tensor([[True, False, False, True]] * 3),
    )
    n = encoder.from_prefix(c.prefix + 0.1 * c.dynamic_mask[..., None], c.mask, c.dynamic_mask)
    batch = EncodedBatch(
        c,
        n,
        torch.randn(3, 2, 2),
        torch.tensor([[True, True], [True, True], [True, False]]),
        torch.tensor([[0.0], [0.0], [1.0]]),
        torch.tensor([[False], [False], [True]]),
        torch.full((3, 1), 0.98),
    )
    trainer = SmolVLAOfflineTrainer(
        encoder,
        current,
        behavior,
        torch.ones(2, dtype=torch.bool),
        c,
        hidden=16,
        ensemble=2,
        dynamics_hidden=8,
        actor_lr=1e-5,
    )
    return trainer, batch


@pytest.mark.parametrize("steps,noise", [(0, 0.1), (True, 0.1), (3, 0), (3, -1), (3, float("nan"))])
def test_flow_validation(steps, noise):
    with pytest.raises(ValueError):
        FlowConfig(steps, noise)


def test_flow_formula_and_true_gaussian():
    x, v = torch.randn(4, 3, 2), torch.randn(4, 3, 2)
    cfg = FlowConfig(10, 0.2)
    mean, std = flow_mean_std(x, v, 0, cfg)
    torch.testing.assert_close(mean, x - 0.1 * (v + 0.02 * x))
    sample = mean + std * torch.randn_like(mean)
    torch.testing.assert_close(
        gaussian_log_prob(sample, mean, std), torch.distributions.Normal(mean, std).log_prob(sample)
    )
    for i in range(10):
        mean, std = flow_mean_std(x, v, i, cfg)
        assert std > 0 and torch.isfinite(mean).all()


def test_exact_shared_backbone_and_frozen_state(setup):
    t, b = setup
    assert t.current.policy.model.vlm_with_expert.vlm is t.behavior.policy.model.vlm_with_expert.vlm
    assert t.current.policy.model.state_proj is t.behavior.policy.model.state_proj
    assert (
        t.current.policy.model.vlm_with_expert.lm_expert
        is not t.behavior.policy.model.vlm_with_expert.lm_expert
    )
    for model in (t.current.policy.model.vlm_with_expert.vlm, t.current.policy.model.state_proj):
        assert not any(p.requires_grad for p in model.parameters())
    assert not any(p.requires_grad for p in t.behavior.policy.parameters())
    assert b.current.observation().features[TOKEN_KEY].data_ptr() == b.current.tokens.data_ptr()
    assert not b.current.tokens.requires_grad
    frozen_ids = {id(p) for p in t.current.policy.model.vlm_with_expert.vlm.parameters()}
    assert not frozen_ids.intersection(id(p) for p in t.iql.parameters())
    assert not frozen_ids.intersection(id(p) for p in t.dynamics.parameters())


def test_encode_preserves_special_tokens_and_task_layout(setup, monkeypatch):
    t, _ = setup
    policy = t.current.policy
    batch_size, length = 2, 14
    prefix = torch.randn(batch_size, length, 4)
    mask = torch.ones(batch_size, length, dtype=torch.bool)
    mask[:, 11:13] = False  # 2 cameras x (2 start, 2 visual, 1 end), 3 text, 1 state
    att = torch.zeros_like(mask)
    att[:, -1] = True
    policy.config.image_features = {"observation.images.head": None, "observation.images.wrist": None}
    policy.model.add_image_special_tokens = True
    policy.model.global_image_start_token = torch.tensor([1, 2])
    policy.model.image_end_token = torch.tensor([3])
    monkeypatch.setattr(policy.model, "embed_prefix", lambda *args: (prefix, mask, att), raising=False)
    monkeypatch.setattr(policy, "prepare_images", lambda batch: ([None, None], [None, None]), raising=False)
    monkeypatch.setattr(policy, "prepare_state", lambda batch: batch["observation.state"], raising=False)
    t.encoder.preprocessor = lambda batch: batch
    observation = {
        "observation.state": torch.randn(batch_size, 2),
        "observation.language.tokens": torch.ones(batch_size, 3, dtype=torch.long),
        "observation.language.attention_mask": torch.tensor([[True, False, False]] * batch_size),
        **{key: torch.zeros(batch_size, 3, 4, 4, dtype=torch.uint8) for key in policy.config.image_features},
    }
    c, _ = t.encoder.encode(observation, "pick cup and bowl")
    expected = torch.zeros_like(mask)
    expected[:, [2, 3, 7, 8, 13]] = True
    assert torch.equal(c.dynamic_mask, expected)
    torch.testing.assert_close(c.prefix, prefix, rtol=0, atol=0)
    att[:, -2] = True
    with pytest.raises(ValueError, match="single state token"):
        t.encoder.encode(observation, "pick cup and bowl")


def test_cache_roundtrip_and_masked_attention(setup):
    _, b = setup
    cache = b.current.cache(2, torch.float32)
    assert isinstance(cache, DynamicCache)
    assert cache.layers[0].keys.shape == (3, 2, 4, 2)
    restored = torch.stack(
        [
            torch.cat([value.transpose(1, 2).flatten(2) for value in (layer.keys, layer.values)], -1)
            for layer in cache.layers
        ],
        1,
    )
    torch.testing.assert_close(restored, b.current.kv, rtol=0, atol=0)
    t, _ = setup
    altered = b.current.tokens.clone()
    altered[:, 2] = 10000
    torch.testing.assert_close(
        t.iql.value(b.current.observation()),
        t.iql.value(replace(b.current, tokens=altered).observation()),
        rtol=0,
        atol=0,
    )


def test_native_smolvla_cache_prefill_and_repeated_denoising():
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    def text_model():
        model = nn.Module()
        model.layers = nn.ModuleList()
        model.norm = nn.Identity()
        for _ in range(2):
            layer = nn.Module()
            layer.input_layernorm = nn.Identity()
            layer.post_attention_layernorm = nn.Identity()
            layer.mlp = nn.Linear(4, 4)
            layer.self_attn = nn.Module()
            layer.self_attn.head_dim = 2
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(layer.self_attn, name, nn.Linear(4, 4))
            model.layers.append(layer)
        return model

    # Exercise the real upstream attention/cache path without downloading a VLM.
    wrapper = SmolVLMWithExpertModel.__new__(SmolVLMWithExpertModel)
    nn.Module.__init__(wrapper)
    wrapper.vlm = nn.Module()
    wrapper.vlm.model = nn.Module()
    wrapper.vlm.model.text_model = text_model()
    wrapper.vlm.model.vision_model = nn.Identity()
    wrapper.vlm.config = SimpleNamespace(text_config=SimpleNamespace(head_dim=2))
    wrapper.config = wrapper.vlm.config
    wrapper.lm_expert = text_model()
    wrapper.num_vlm_layers = wrapper.num_expert_layers = 2
    wrapper.num_attention_heads = wrapper.num_key_value_heads = 2
    wrapper.self_attn_every_n_layers = 1
    wrapper.attention_mode = "self_attn"
    policy = TinyPolicy()
    native_model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(native_model)
    native_model.config = policy.config
    native_model.vlm_with_expert = wrapper
    native_model.state_proj = policy.model.state_proj
    native_model.action_out_proj = nn.Linear(4, 4)
    native_model.embed_suffix = lambda latent, timestep: (
        latent,
        torch.ones(latent.shape[:2], dtype=torch.bool),
        torch.ones(latent.shape[:2], dtype=torch.bool),
    )
    policy.model = native_model
    encoder, adapter, _ = make_adapters(policy, None, FlowConfig(3, 0.2))
    mask = torch.tensor([[True, True, False, True]] * 2)
    condition = encoder.from_prefix(torch.randn(2, 4, 4), mask, mask.clone())
    cache = condition.cache(2, torch.float32)
    latent = torch.randn(2, 4, 4)
    kwargs = {"prefix_pad_masks": mask, "past_key_values": cache, "x_t": latent, "timestep": torch.ones(2)}
    first = native_model.denoise_step(**kwargs)
    second = native_model.denoise_step(**kwargs)
    assert all(layer.keys.shape[-2] == mask.shape[1] for layer in cache.layers)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(adapter.velocity(condition, latent, 0), first, rtol=0, atol=0)
    first.sum().backward()
    assert wrapper.lm_expert.layers[0].self_attn.q_proj.weight.grad is not None
    assert all(parameter.grad is None for parameter in wrapper.vlm.parameters())


def test_q_uses_action_order_and_v_cannot_see_actions(setup):
    t, b = setup
    obs = b.current.observation()
    q = t.iql.q1(obs, b.action, b.valid)
    reversed_q = t.iql.q1(obs, b.action.flip(1), b.valid)
    assert not torch.allclose(q[:2], reversed_q[:2])
    with pytest.raises(ValueError, match="observation-only"):
        t.iql.value(obs, b.action, b.valid)
    changed = b.action.clone()
    changed[~b.valid] = 1e5
    torch.testing.assert_close(q, t.iql.q1(obs, changed, b.valid), rtol=0, atol=0)
    all_valid = torch.ones_like(b.valid)
    assert not torch.allclose(q[-1], t.iql.q1(obs, changed, all_valid)[-1])


def test_inactive_action_channels_never_enter_q(setup):
    from RL.smolvla.critics import ValueTokenHead

    _, b = setup
    q = ValueTokenHead(
        (4, 4),
        hidden=16,
        action_dim=3,
        execution_steps=2,
        active_mask=torch.tensor([True, False, True]),
    )
    action = torch.randn(3, 2, 3)
    original = q(b.current.observation(), action, b.valid)
    action[:, :, 1] = 100000
    torch.testing.assert_close(original, q(b.current.observation(), action, b.valid), rtol=0, atol=0)
    with pytest.raises(ValueError, match="contiguous"):
        q(b.current.observation(), action, torch.tensor([[False, True]] * 3))


def test_value_query_gradients_and_explicit_source_detach(setup):
    t, b = setup
    from RL.types import ObservationBatch

    tokens = b.current.tokens.detach().clone().requires_grad_(True)
    features = dict(b.current.observation().features)
    features[TOKEN_KEY] = tokens
    obs = ObservationBatch(features)
    (t.iql.q1(obs, b.action, b.valid).sum() + t.iql.value(obs).sum()).backward()
    for head in (t.iql.q1, t.iql.value):
        assert head.value_tokens.grad.abs().sum() > 0
        assert head.memory.observation_proj.weight.grad.abs().sum() > 0
    assert tokens.grad is None
    assert not any(p.requires_grad for p in t.iql.target_q1.parameters())


def test_token_iql_matches_expectile_td_and_polyak_math(setup):
    from RL.algorithms.iql import compute_td_target, expectile_loss

    t, b = setup
    iql, obs = t.iql, b.current.observation()
    with torch.no_grad():
        bar = torch.minimum(iql.target_q1(obs, b.action, b.valid), iql.target_q2(obs, b.action, b.valid))
        v = iql.value(obs)
        td = compute_td_target(
            reward=b.reward,
            discount=b.discount,
            done=b.done,
            next_value=iql.value(b.following.observation()),
        )
        expected_v = expectile_loss(bar - v, expectile=iql.expectile).item()
        expected_q = (
            (iql.q1(obs, b.action, b.valid) - td).square().mean()
            + (iql.q2(obs, b.action, b.valid) - td).square().mean()
        ).item()
        old_target = iql.target_q1.value_tokens.clone()
    metrics = iql.update(b.decision())
    assert metrics["v_loss"] == pytest.approx(expected_v)
    assert metrics["q_loss"] == pytest.approx(expected_q)
    torch.testing.assert_close(iql.target_q1.value_tokens, old_target.lerp(iql.q1.value_tokens, iql.tau))
    torch.testing.assert_close(
        iql.advantage(obs, b.action, b.valid, normalize=False),
        iql.min_q(obs, b.action, b.valid) - iql.value(obs),
    )


def test_default_transformer_heads_reduce_parameters():
    from RL.smolvla.critics import TokenIQL
    from RL.smolvla.dynamics import TokenDynamics

    active = torch.ones(20, dtype=torch.bool)
    iql = TokenIQL((241, 960), 20, 32, active)
    dynamics = TokenDynamics((241, 960), 20, 32, active)
    qv_count = sum(p.numel() for p in iql.parameters() if p.requires_grad)
    dyn_count = sum(p.numel() for p in dynamics.parameters())
    old_qv_count = 2 * ((21152 + 1) * 256 + (256 + 1) * 256 + 257)
    old_qv_count += (20480 + 1) * 256 + (256 + 1) * 256 + 257
    assert old_qv_count == 16271619
    assert qv_count < 0.12 * old_qv_count
    assert qv_count + dyn_count < 0.16 * old_qv_count
    assert iql.q1.value_tokens.shape == (1, 2, 128)
    assert len(iql.q1.transformer.layers) == 2


@pytest.mark.parametrize("config", [{"hidden": 15}, {"layers": 0}, {"heads": 0}, {"value_tokens": 0}])
def test_attention_architecture_validation(config):
    from RL.smolvla.critics import ValueTokenHead

    with pytest.raises(ValueError):
        ValueTokenHead((4, 4), **config)


def test_prefix_dynamics_preserves_text_and_regenerates_cache(setup):
    t, b = setup
    with torch.no_grad():
        for i, head in enumerate(t.dynamics.heads):
            head.delta.bias.fill_(0.1 * (i + 1))
    with patch.object(t.encoder, "from_prefix", wraps=t.encoder.from_prefix) as encode:
        n, outcomes, disagreement = t.dynamics.predict(b.current, b.action, b.valid, encoder=t.encoder)
    assert encode.call_count == 1
    assert outcomes.shape == (3, 2) and (disagreement > 0).all()
    assert not torch.equal(n.prefix, b.current.prefix)
    static = ~b.current.dynamic_mask
    torch.testing.assert_close(n.prefix[static], b.current.prefix[static], rtol=0, atol=0)
    expected = t.encoder.from_prefix(n.prefix, n.mask, n.dynamic_mask)
    torch.testing.assert_close(n.tokens, expected.tokens, rtol=0, atol=0)
    torch.testing.assert_close(n.kv, expected.kv, rtol=0, atol=0)
    assert not torch.equal(n.kv, b.current.kv)
    assert not n.prefix.requires_grad and not n.tokens.requires_grad and not n.kv.requires_grad


def test_dynamics_attention_padding_and_actions(setup):
    t, b = setup
    t.dynamics.update(*b.dynamics_args())
    head = t.dynamics.heads[0]
    predicted, outcomes = head(b.current, b.action, b.valid)
    tokens = b.current.tokens.clone()
    tokens[~b.current.mask] = 100000
    action = b.action.clone()
    action[~b.valid] = 100000
    altered_p, altered_o = head(replace(b.current, tokens=tokens), action, b.valid)
    torch.testing.assert_close(predicted, altered_p, rtol=0, atol=0)
    torch.testing.assert_close(outcomes, altered_o, rtol=0, atol=0)
    reversed_p, reversed_o = head(b.current, b.action.flip(1), b.valid)
    assert not torch.allclose(outcomes[:2], reversed_o[:2])
    assert not torch.equal(predicted[:2], reversed_p[:2])


def test_dynamics_masks_terminal_targets_and_balances_state(setup):
    from RL.smolvla.dynamics import dynamic_error_mean

    t, b = setup
    p, o = t.dynamics.heads[0](b.current, b.action, b.valid)
    original = t.dynamics.losses(p, o, b.current, b.following, b.reward, b.done)[0]
    prefix = b.following.prefix.clone()
    prefix[~b.following.dynamic_mask] = 100000
    prefix[b.done.squeeze(-1)] = 100000
    changed = replace(b.following, prefix=prefix)
    actual = t.dynamics.losses(p, o, b.current, changed, b.reward, b.done)[0]
    torch.testing.assert_close(original, actual, rtol=0, atol=0)
    terminal = t.dynamics.losses(p, o, b.current, changed, b.reward, torch.ones_like(b.done))[0]
    assert terminal.item() == 0
    error = torch.ones(1, 193, 4)
    error[:, -1] = 3
    assert dynamic_error_mean(error, torch.ones(1, 193, dtype=torch.bool)).item() == 2


def test_old_mlp_rl_checkpoint_is_explicitly_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"kind": "smolvla_offline_rl_v1"}))
    with pytest.raises(ValueError, match="Old v1.*incompatible"):
        validate_checkpoint(tmp_path)


def test_trace_replay_identity_and_gradients(setup):
    t, b = setup
    trace = t.behavior.sample(b.current, torch.Generator().manual_seed(0))
    assert trace.actions.shape == (3, 4, 4)
    assert t.current.executed(trace.actions).shape == (3, 2, 2)
    for i in range(3):
        torch.testing.assert_close(
            t.current.replay_step(b.current, trace, i), trace.log_probs[i], rtol=0, atol=0
        )
    loss = -t.current.replay_step(b.current, trace, 2)[..., :2].mean()
    loss.backward()
    assert t.current.policy.model.action_out_proj.weight.grad is not None
    assert all(p.grad is None for p in t.current.policy.model.vlm_with_expert.vlm.parameters())


def test_training_updates_only_actor_and_heads(setup):
    t, b = setup
    frozen = {
        name: p.detach().clone() for name, p in t.current.policy.named_parameters() if not p.requires_grad
    }
    old = copy.deepcopy(t.behavior.policy.state_dict())
    actor = t.current.policy.model.action_out_proj.weight.detach().clone()
    q = t.iql.q1.memory.observation_proj.weight.detach().clone()
    t.iql.update(b.decision())
    assert not torch.equal(q, t.iql.q1.memory.observation_proj.weight)
    t.dynamics.update(*b.dynamics_args())
    fixed_iql = copy.deepcopy(t.iql.state_dict())
    fixed_dynamics = copy.deepcopy(t.dynamics.state_dict())
    metrics = t.update_actor(b)
    assert metrics["actor/replay_error"] == 0
    assert not torch.equal(actor, t.current.policy.model.action_out_proj.weight)
    for name, p in t.current.policy.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
    for name, p in t.behavior.policy.state_dict().items():
        torch.testing.assert_close(p, old[name], rtol=0, atol=0)
    for module, fixed in ((t.iql, fixed_iql), (t.dynamics, fixed_dynamics)):
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, fixed[name], rtol=0, atol=0)


def test_amq_consumes_regenerated_tokens_and_kv_not_starting_images(setup):
    t, b = setup
    calls = []
    critic_calls = []
    original_sample = t.current.sample
    original_q = t.iql.min_q

    def sample(condition, generator):
        calls.append(condition.kv.clone())
        return original_sample(condition, generator)

    def predict(condition, action, valid, *, encoder):
        following = encoder.from_prefix(condition.prefix + 1, condition.mask, condition.dynamic_mask)
        return following, torch.zeros(3, 2), torch.zeros(3)

    def min_q(observation, action, valid):
        critic_calls.append(observation.features[TOKEN_KEY].clone())
        return original_q(observation, action, valid)

    with (
        patch.object(t.current, "sample", side_effect=sample),
        patch.object(t.dynamics, "predict", side_effect=predict),
        patch.object(t.iql, "min_q", side_effect=min_q),
    ):
        t.amq_score(t.current, b.current, horizon=3)
    assert len(calls) == 3
    assert not torch.equal(calls[0], calls[1])
    assert not torch.equal(calls[1], calls[2])
    for i in range(3):
        expected = t.encoder.from_prefix(b.current.prefix + i, b.current.mask, b.current.dynamic_mask)
        torch.testing.assert_close(calls[i], expected.kv)
        torch.testing.assert_close(critic_calls[i], expected.tokens)


def test_gate_rejects_and_accepts_without_encoder_change(setup):
    t, b = setup
    with torch.no_grad():
        t.current.policy.model.action_out_proj.weight.add_(1)
    old = t.behavior.policy.model.action_out_proj.weight.detach().clone()
    with (
        patch.object(t.dynamics, "validation_loss", return_value=100),
        patch.object(t, "amq_score", side_effect=[(2.0, 0.0), (1.0, 0.0)]),
    ):
        result = t.evaluate_and_promote(b)
    assert not result["amq/promoted"]
    torch.testing.assert_close(t.behavior.policy.model.action_out_proj.weight, old)
    with (
        patch.object(t.dynamics, "validation_loss", return_value=0),
        patch.object(t, "amq_score", side_effect=[(2.0, 0.0), (1.0, 0.0)]),
    ):
        result = t.evaluate_and_promote(b)
    assert result["amq/promoted"]
    torch.testing.assert_close(
        t.behavior.policy.model.action_out_proj.weight, t.current.policy.model.action_out_proj.weight
    )


def test_checkpoint_candidate_and_accepted_roundtrip(setup, tmp_path):
    t, b = setup
    t.iql.update(b.decision())
    t.dynamics.update(*b.dynamics_args())
    t.update_actor(b)
    candidate = t.current.policy.model.action_out_proj.weight.detach().clone()
    accepted = t.behavior.policy.model.action_out_proj.weight.detach().clone()
    save_checkpoint(
        tmp_path / "checkpoint",
        t,
        DummyProcessor("pre"),
        DummyProcessor("post"),
        {"test": True},
        {"actor": 1},
    )
    assert validate_checkpoint(tmp_path / "checkpoint") == {"test": True}
    exported = torch.load(tmp_path / "checkpoint/pretrained_model/model.safetensors", weights_only=True)
    torch.testing.assert_close(exported["model.action_out_proj.weight"], accepted)
    with torch.no_grad():
        t.current.policy.model.action_out_proj.weight.zero_()
    assert restore_checkpoint(tmp_path / "checkpoint", t) == {"actor": 1}
    torch.testing.assert_close(t.current.policy.model.action_out_proj.weight, candidate)
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path / "checkpoint", t, None, None, {}, {})
    (tmp_path / "checkpoint/run.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        validate_checkpoint(tmp_path / "checkpoint")


def test_manifest_mapping_handles_source_episode_gaps():
    manifest = {
        "schema": "franka_duo_tele_data.mcap_to_lerobot.rgb20d.v1",
        "episodes": [
            {
                "episode": "episode_000002",
                "stats": {"frames_written": 5},
                "provenance": {"manifest_status": "complete", "manifest_reward": 1.0},
            },
            {
                "episode": "episode_000004",
                "stats": {"frames_written": 7},
                "provenance": {"manifest_status": "complete", "manifest_reward": 0.0},
            },
        ],
    }
    episodes = [{"episode_index": 0, "length": 5}, {"episode_index": 1, "length": 7}]
    assert verified_manifest_labels(manifest, episodes) == {0: True, 1: False}
    manifest["episodes"][0]["stats"]["frames_written"] = 6
    with pytest.raises(ValueError, match="order/length"):
        verified_manifest_labels(manifest, episodes)


def test_data_window_padding_rewards_and_holdout(tmp_path):
    import pandas as pd

    from RL.smolvla.data import SmolVLADecisionDataset

    class FakeHF:
        def __getitem__(self, index):
            if index == "task_index":
                return [0] * 7
            return {"action": list(torch.arange(14).reshape(7, 2)[index])}

    class FakeSource:
        def __init__(self, *args, **kwargs):
            self.hf_dataset = FakeHF()
            self.meta = SimpleNamespace(
                episodes=[
                    {"episode_index": 0, "length": 3, "dataset_from_index": 0, "dataset_to_index": 3},
                    {"episode_index": 1, "length": 4, "dataset_from_index": 3, "dataset_to_index": 7},
                ],
                camera_keys=["observation.images.head"],
                tasks=pd.DataFrame({"task_index": [0]}, index=["pick cup and bowl"]),
            )

        def __getitem__(self, index):
            return {
                "observation.state": torch.tensor([float(index), 0.0]),
                "observation.images.head": torch.zeros(3, 4, 4, dtype=torch.uint8),
            }

    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {"episodes": [{"episode_index": 0, "success": True}, {"episode_index": 1, "success": False}]}
        )
    )
    with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset", FakeSource):
        d = SmolVLADecisionDataset(tmp_path, "test", labels_path=labels, chunk_size=2, gamma=0.9)
        assert len(d) == 4
        assert d[0]["next_observation"]["observation.state"][0] == 2
        assert d[0]["reward"].item() == 0
        assert d[1]["reward"].item() == 1 and d[1]["done"].item()
        assert d[1]["valid"].tolist() == [True, False]
        assert d[3]["reward"].item() == 0 and d[3]["done"].item()
        assert d[3]["discount"].item() == pytest.approx(0.81)
        train, val, heldout = d.split()
        assert set(train).isdisjoint(val) and len(heldout) == 1
        assert {d.locations[i][0] for i in train}.isdisjoint({d.locations[i][0] for i in val})


@pytest.fixture
def cli_setup(tmp_path, monkeypatch):
    from RL.cli.train_smolvla_offline import parser
    from RL.smolvla.features import FrozenSmolVLM

    class TinyDataset:
        labels = {0: True, 1: True}

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 6

        def split(self, *args):
            return [0, 1, 2, 3], [4, 5], [1]

        def __getitem__(self, i):
            return {
                "observation": {"observation.state": torch.tensor([float(i), 0])},
                "next_observation": {"observation.state": torch.tensor([float(i) + 1, 0])},
                "action": torch.zeros(2, 2),
                "valid": torch.ones(2, dtype=torch.bool),
                "reward": torch.tensor([float(i % 2)]),
                "done": torch.tensor([bool(i % 2)]),
                "discount": torch.tensor([0.98]),
            }

    def encode(self, observation, task):
        state = observation["observation.state"]
        prefix = state[:, :1, None].expand(-1, 4, 4).contiguous() * 0.01
        mask = torch.ones(state.shape[0], 4, dtype=torch.bool)
        dynamic = mask.clone()
        dynamic[:, 1:3] = False
        return self.from_prefix(prefix, mask, dynamic), observation.get("action")

    monkeypatch.setattr(FrozenSmolVLM, "encode", encode)
    monkeypatch.setattr(
        "RL.smolvla.adapter.load_policy",
        lambda *args: (
            TinyPolicy(),
            DummyProcessor("pre"),
            DummyProcessor("post"),
            torch.ones(2, dtype=torch.bool),
        ),
    )
    monkeypatch.setattr("RL.smolvla.data.SmolVLADecisionDataset", TinyDataset)
    TinyPolicy().save_pretrained(tmp_path / "il")
    (tmp_path / "data/meta").mkdir(parents=True)
    (tmp_path / "data/meta/info.json").write_text("{}")
    return parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "il"),
            "--dataset-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--hidden",
            "16",
            "--ensemble",
            "2",
            "--dynamics-hidden",
            "8",
            "--torch-threads",
            "1",
            "--num-workers",
            "0",
            "--batch-size",
            "2",
            "--iql-steps",
            "2",
            "--dynamics-steps",
            "2",
            "--actor-steps",
            "1",
        ]
    )


@pytest.mark.parametrize(
    "stop,stages",
    [
        ("iql", ["iql"] * 2),
        ("dynamics", ["iql"] * 2 + ["dynamics"] * 2),
        ("actor", ["iql"] * 2 + ["dynamics"] * 2 + ["actor"]),
    ],
)
def test_cli_stages_are_ordered_and_stop_after_works(cli_setup, stop, stages):
    from RL.cli.train_smolvla_offline import run

    args = cli_setup
    args.smoke, args.stop_after = True, stop
    trainer = run(args)
    records = [json.loads(line) for line in (args.output_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in records] == stages
    assert trainer.actor_updates == int(stop == "actor")


def test_cli_does_not_start_actor_with_unvalidated_dynamics(cli_setup, monkeypatch):
    from RL.cli.train_smolvla_offline import run

    monkeypatch.setattr("RL.smolvla.dynamics.TokenDynamics.validation_loss", lambda *args: 2.0)
    with pytest.raises(RuntimeError, match="actor was NOT started"):
        run(cli_setup)
    records = [json.loads(line) for line in (cli_setup.output_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in records] == ["iql", "iql", "dynamics", "dynamics"]
    assert (cli_setup.output_dir / "checkpoints/dynamics_000002/manifest.json").is_file()


def test_cli_token_checkpoint_resume_and_architecture_guard(cli_setup):
    from RL.cli.train_smolvla_offline import run

    args = cli_setup
    args.smoke, args.stop_after = True, "iql"
    first = run(args)
    args.resume = args.output_dir / "checkpoints/iql_000002"
    args.output_dir = args.output_dir.with_name("resumed")
    args.stop_after = "dynamics"
    resumed = run(args)
    for key, value in first.iql.state_dict().items():
        torch.testing.assert_close(value, resumed.iql.state_dict()[key], rtol=0, atol=0)
    records = [json.loads(line) for line in (args.output_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in records] == ["dynamics", "dynamics"]
    args.output_dir = args.output_dir.with_name("bad_resume")
    args.value_tokens += 1
    with pytest.raises(ValueError, match="Resume configuration changed: value_tokens"):
        run(args)
