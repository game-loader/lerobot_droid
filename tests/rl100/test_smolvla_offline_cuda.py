"""Opt-in CUDA/real-data tests. Set SMOLVLA_TEST_CHECKPOINT and SMOLVLA_TEST_DATASET."""

import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

pytestmark = pytest.mark.skipif(
    not os.environ.get("SMOLVLA_TEST_CHECKPOINT"), reason="Real checkpoint CUDA tests are opt-in"
)


@pytest.fixture(scope="module")
def real():
    from RL.smolvla.adapter import load_policy, make_adapters
    from RL.smolvla.data import SmolVLADecisionDataset
    from RL.smolvla.flow import FlowConfig

    torch.set_num_threads(4)
    policy, pre, post, active = load_policy(Path(os.environ["SMOLVLA_TEST_CHECKPOINT"]), "cuda")
    encoder, current, old = make_adapters(policy, pre, FlowConfig())
    data = SmolVLADecisionDataset(
        os.environ["SMOLVLA_TEST_DATASET"],
        "local/franka_duo_lerobot_rgb20d_v1",
        confirmed_manifest_rewards=True,
    )
    raw = next(iter(DataLoader(data, batch_size=2)))
    return policy, pre, encoder, current, old, raw


def test_frozen_kv_exact_repeatability(real):
    policy, _, encoder, current, old, raw = real
    c1, _ = encoder.encode(raw["observation"], "pick cup and bowl")
    c2, _ = encoder.encode(raw["observation"], "pick cup and bowl")
    assert c1.kv.shape == (2, 16, 241, 640)
    assert c1.prefix.shape == c1.tokens.shape == (2, 241, 960)
    assert c1.dynamic_mask.sum(-1).tolist() == [193, 193]
    torch.testing.assert_close(c1.kv, c2.kv, rtol=0, atol=0)
    torch.testing.assert_close(c1.prefix, c2.prefix, rtol=0, atol=0)
    torch.testing.assert_close(c1.tokens, c2.tokens, rtol=0, atol=0)
    assert current.policy.model.vlm_with_expert.vlm is old.policy.model.vlm_with_expert.vlm
    assert not policy.model.state_proj.weight.requires_grad


def test_prefix_reencoding_is_exact_and_frozen(real):
    _, _, encoder, _, _, raw = real
    c, _ = encoder.encode(raw["observation"], "pick cup and bowl")
    prefix = c.prefix.clone().requires_grad_(True)
    rebuilt = encoder.from_prefix(prefix, c.mask, c.dynamic_mask)
    torch.testing.assert_close(c.kv, rebuilt.kv, rtol=0, atol=0)
    torch.testing.assert_close(c.tokens, rebuilt.tokens, rtol=0, atol=0)
    assert not rebuilt.prefix.requires_grad and not rebuilt.tokens.requires_grad
    assert not rebuilt.kv.requires_grad and prefix.grad is None


def test_real_prefix_dynamics_and_token_iql_updates(real):
    from RL.smolvla.trainer import SmolVLAOfflineTrainer

    policy, _, encoder, current, old, raw = real
    c, _ = encoder.encode(raw["observation"], "pick cup and bowl")
    trainer = SmolVLAOfflineTrainer(encoder, current, old, torch.ones(20, dtype=torch.bool, device="cuda"), c)
    frozen = {n: p.clone() for n, p in policy.named_parameters() if not p.requires_grad}
    batch = trainer.encode_batch(raw, "pick cup and bowl")
    trainer.iql.update(batch.decision())
    trainer.dynamics.update(*batch.dynamics_args())
    n, _, disagreement = trainer.dynamics.predict(c, batch.action, batch.valid, encoder=encoder)
    assert torch.isfinite(disagreement).all()
    assert not torch.equal(n.prefix, c.prefix)
    assert not torch.equal(n.kv, c.kv)
    torch.testing.assert_close(n.prefix[~c.dynamic_mask], c.prefix[~c.dynamic_mask], rtol=0, atol=0)
    rebuilt = encoder.from_prefix(n.prefix, n.mask, n.dynamic_mask)
    torch.testing.assert_close(n.kv, rebuilt.kv, rtol=0, atol=0)
    torch.testing.assert_close(n.tokens, rebuilt.tokens, rtol=0, atol=0)
    for name, parameter in policy.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
            assert parameter.grad is None


def test_deterministic_adapter_matches_native_il_sampler(real):
    policy, pre, encoder, current, _, raw = real
    condition, _ = encoder.encode(raw["observation"], "pick cup and bowl")
    obs = {
        k: v.float() / 255 if k.startswith("observation.images.") else v
        for k, v in raw["observation"].items()
    }
    batch = pre({**obs, "task": ["pick cup and bowl"] * 2})
    noise = torch.randn(2, 64, 32, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5))
    with torch.no_grad():
        original = policy.predict_action_chunk(batch, noise=noise)
        adapted = current.sample(condition, stochastic=False, noise=noise).actions[:, :, :20]
    torch.testing.assert_close(original, adapted, rtol=1e-5, atol=1e-5)


def test_stochastic_real_trace_replay(real):
    _, _, encoder, current, old, raw = real
    condition, _ = encoder.encode(raw["observation"], "pick cup and bowl")
    trace = old.sample(condition, torch.Generator(device="cuda").manual_seed(10))
    with torch.no_grad():
        for i in range(current.flow.steps):
            torch.testing.assert_close(
                current.replay_step(condition, trace, i), trace.log_probs[i], rtol=0, atol=0
            )
