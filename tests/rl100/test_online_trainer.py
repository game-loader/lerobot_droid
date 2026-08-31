# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Focused tests for decision-major online diffusion PPO."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from RL.adapters.checkpoint import CheckpointAdapter
from RL.checkpointing import RLProvenance, load_rl_checkpoint
from RL.cli.train_online import _checkpoint_root, _parser, _restore_online_state
from RL.config import RLConfig, TraceConfig
from RL.policy.diffusion_adapter import DiffusionRLAdapter
from RL.trainers.online import (
    OnlineTrainer,
    _clip_env_action,
    _raw_feature_mapping,
    _reset_history,
    _select_transition_frames,
    _validate_env_action,
)


def _adapter(root: Path) -> DiffusionRLAdapter:
    config = DiffusionConfig(
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(39,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        device="cpu",
        pretrained_backbone_weights=None,
        down_dims=(8,),
        kernel_size=3,
        n_groups=2,
        diffusion_step_embed_dim=8,
        num_train_timesteps=4,
        num_inference_steps=2,
    )
    policy = DiffusionPolicy(config)
    policy.save_pretrained(root)
    stats = {
        "observation.state": {
            "min": torch.full((39,), -20.0),
            "max": torch.full((39,), 20.0),
        },
        "action": {"min": torch.tensor([-2.0, 0.0]), "max": torch.tensor([2.0, 0.0])},
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(root)
    postprocessor.save_pretrained(root)
    return DiffusionRLAdapter(
        CheckpointAdapter.load(root, device="cpu"),
        TraceConfig(num_inference_steps=2, eta=1.0, sigma_min=0.01, sigma_max=0.1),
    )


class _AsyncDoneEnv:
    num_envs = 2

    def __init__(self) -> None:
        self.step_in_decision = 0
        self.actions: list[np.ndarray] = []

    def reset(self, *, seed=None):
        del seed
        self.step_in_decision = 0
        return {"agent_pos": np.zeros((2, 39), dtype=np.float32)}, {}

    def step(self, action: np.ndarray):
        self.actions.append(action.copy())
        phase = self.step_in_decision % 2
        self.step_in_decision += 1
        observation = np.full((2, 39), float(self.step_in_decision), dtype=np.float32)
        done = np.array([phase == 0, phase == 1], dtype=np.bool_)
        final_obs = np.empty(2, dtype=object)
        final_info = np.empty(2, dtype=object)
        for world in range(2):
            if done[world]:
                final_obs[world] = np.full(39, 3.0 + world, dtype=np.float32)
                final_info[world] = {"is_success": world == 0}
                observation[world] = 9.0
            else:
                final_obs[world] = None
                final_info[world] = None
        info = {
            "final_obs": final_obs,
            "_final_obs": done.copy(),
            "final_info": final_info,
            "_final_info": done.copy(),
            # A wrong reset/top-level value proves the trainer reads final_info.
            "is_success": np.logical_not(done),
        }
        return (
            {"agent_pos": observation},
            np.full(2, 100.0, dtype=np.float32),
            np.zeros(2, dtype=np.bool_),
            done,
            info,
        )


def test_collect_stores_partial_chunks_sparse_reward_and_terminal_state(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    old = _adapter(tmp_path / "old")
    old.policy.load_state_dict(current.policy.state_dict(), strict=True)
    trainer = OnlineTrainer(
        current_policy=current,
        old_policy=old,
        actor_optimizer=torch.optim.Adam(current.policy.parameters(), lr=1e-4),
        metrics_path=tmp_path / "metrics.jsonl",
        gamma=0.9,
        ppo_epochs=1,
    )

    rollout = trainer.collect(_AsyncDoneEnv(), decisions=2, seed=7)

    assert rollout.time_steps == 2
    assert rollout.num_envs == 2
    assert rollout.action.shape == (2, 2, 2, 2)
    assert rollout.action_valid[0].tolist() == [[True, False], [True, True]]
    assert torch.all(rollout.action[0, 0, 1] == 0.0)
    assert rollout.executed_steps[0].tolist() == [1, 2]
    torch.testing.assert_close(rollout.discount[0, :, 0], torch.tensor([0.9, 0.81]))
    assert rollout.reward[0, :, 0].tolist() == [1.0, 0.0]
    assert rollout.success[0, :, 0].tolist() == [True, False]
    terminal = rollout.next_observation.features["observation.state"][0]
    assert torch.all(terminal[0, -1] == 3.0)
    assert torch.all(terminal[1, -1] == 4.0)
    assert trainer.counters.environment_steps == 6


def test_collect_can_use_environment_rewards(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    old = _adapter(tmp_path / "old")
    old.policy.load_state_dict(current.policy.state_dict(), strict=True)
    trainer = OnlineTrainer(
        current_policy=current,
        old_policy=old,
        actor_optimizer=torch.optim.Adam(current.policy.parameters(), lr=1e-4),
        metrics_path=None,
        reward_mode="environment",
    )

    rollout = trainer.collect(_AsyncDoneEnv(), decisions=1, seed=7)

    # The fake environment returns 100 per executed step.  The environment
    # reward mode accumulates those scalars instead of replacing them with
    # terminal-success labels.
    assert rollout.reward[:, :, 0].tolist() == [[100.0, 200.0]]


def test_final_obs_selection_is_dimension_agnostic_for_dp3_features() -> None:
    base = {
        "observation.state": torch.zeros((2, 34), dtype=torch.float32),
        "observation.point_cloud": torch.zeros((2, 8, 3), dtype=torch.float32),
    }
    final_obs = np.empty(2, dtype=object)
    final_obs[0] = {
        "observation.state": np.ones(34, dtype=np.float32),
        "observation.point_cloud": np.ones((8, 3), dtype=np.float32),
    }
    final_obs[1] = None
    selected = _select_transition_frames(
        base,
        {"final_obs": final_obs, "_final_obs": np.asarray([True, False])},
        np.asarray([True, False]),
        num_envs=2,
    )
    torch.testing.assert_close(selected["observation.state"][0], torch.ones(34))
    torch.testing.assert_close(selected["observation.point_cloud"][0], torch.ones(8, 3))
    torch.testing.assert_close(selected["observation.state"][1], torch.zeros(34))


def test_done_world_history_advances_with_same_step_reset_frames(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    old = _adapter(tmp_path / "old")
    old.policy.load_state_dict(current.policy.state_dict(), strict=True)
    trainer = OnlineTrainer(
        current_policy=current,
        old_policy=old,
        metrics_path=None,
        gamma=0.9,
        ppo_epochs=1,
    )
    seen: list[torch.Tensor] = []
    sample_trace = trainer.old_policy.sample_trace

    def capture(observation, *, generator=None):
        seen.append(observation.features["observation.state"].detach().clone())
        return sample_trace(observation, generator=generator)

    trainer.old_policy.sample_trace = capture  # type: ignore[method-assign]
    trainer.collect(_AsyncDoneEnv(), decisions=2, seed=13)

    assert len(seen) == 2
    torch.testing.assert_close(seen[1][0, :, 0], torch.tensor([9.0, 2.0]))


def test_online_update_is_finite_and_syncs_rollout_policy(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    old = _adapter(tmp_path / "old")
    old.policy.load_state_dict(current.policy.state_dict(), strict=True)
    trainer = OnlineTrainer(
        current_policy=current,
        old_policy=old,
        actor_optimizer=torch.optim.Adam(current.policy.parameters(), lr=1e-4),
        metrics_path=tmp_path / "metrics.jsonl",
        gamma=0.9,
        ppo_epochs=1,
        minibatch_size=2,
        debug=True,
    )
    rollout = trainer.collect(_AsyncDoneEnv(), decisions=2, seed=11)

    metrics = trainer.update(rollout)

    assert math.isfinite(metrics["actor/loss"])
    assert math.isfinite(metrics["value/loss"])
    assert "info/actor/ratio_q05" in metrics
    assert "info/actor/denoise_00/ratio_q95" in metrics
    assert "info/actor/denoise_00/sigma_effective" in metrics
    assert "info/actor/old_replay_abs_delta_max" in metrics
    assert "rollout/chunk_success_rate" in metrics
    assert "rollout/success_count" in metrics
    assert "actor/ratio_q05" not in metrics
    assert trainer.counters.global_updates == 1
    assert trainer.counters.old_policy_syncs == 1
    for current_parameter, old_parameter in zip(
        current.policy.parameters(), old.policy.parameters(), strict=True
    ):
        torch.testing.assert_close(current_parameter, old_parameter)


def test_online_cli_parser_exposes_smoke_controls(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "output"),
            "--num-envs",
            "3",
            "--rollout-decisions",
            "99",
            "--ppo-epochs",
            "7",
            "--inference-steps",
            "50",
            "--env-factory",
            "tests.rl100.test_online_trainer:_AsyncDoneEnv",
            "--reward-mode",
            "environment",
            "--smoke",
        ]
    )

    assert args.smoke is True
    assert args.num_envs == 3
    assert args.rollout_decisions == 99
    assert args.ppo_epochs == 7
    assert args.inference_steps == 50
    assert args.env_factory.endswith(":_AsyncDoneEnv")
    assert args.reward_mode == "environment"


def test_online_cli_uses_conservative_actor_defaults(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert args.actor_lr == pytest.approx(1e-6)
    assert args.ppo_epochs == 1
    assert args.rollout_decisions == 30
    assert args.probability_sigma_min == pytest.approx(0.1)
    assert args.num_envs == 1
    assert args.env_device is None
    assert args.reward_mode == "terminal_success"


def test_online_cli_keeps_sim_device_as_non_simulation_factory_hint(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "output"),
            "--sim-device",
            "cpu",
        ]
    )

    assert args.env_device == "cpu"


def test_online_cli_resolves_run_directory_to_final_checkpoint(tmp_path: Path) -> None:
    final = tmp_path / "run" / "checkpoints" / "final"
    final.mkdir(parents=True)
    (final / "manifest.json").write_text("{}", encoding="utf-8")
    (final / "rl_state.pt").write_bytes(b"state")

    assert _checkpoint_root(tmp_path / "run") == final


def test_online_checkpoint_contains_reloadable_policy_and_value_state(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    old = _adapter(tmp_path / "old")
    old.policy.load_state_dict(current.policy.state_dict(), strict=True)
    trainer = OnlineTrainer(
        current_policy=current,
        old_policy=old,
        metrics_path=None,
        gamma=0.9,
        ppo_epochs=1,
    )
    trainer.update(trainer.collect(_AsyncDoneEnv(), decisions=1, seed=5))
    provenance = RLProvenance(
        stage="online",
        root_base_path="base",
        root_base_hash="a" * 64,
        input_checkpoint="input",
        input_checkpoint_hash="b" * 64,
        processor_fingerprint=current.checkpoint.processor_fingerprint(),
        policy_type="diffusion",
        policy_config_hash="c" * 64,
        dataset_root="dataset",
        dataset_repo_id="example/repo",
        dataset_summary_path="summary.json",
        dataset_summary_hash="d" * 64,
        feature_keys=("observation.state",),
        state_key="observation.state",
        state_dim=39,
        chunk_size=2,
        action_dim=2,
        active_action_mask=(True, False),
        lerobot_commit="e" * 40,
        rl100_url="https://github.com/Lei-Kun/RL-100",
    )
    config = RLConfig(
        trace=current.trace_config,
        state_dim=39,
        action_dim=2,
        chunk_size=2,
        n_obs_steps=2,
        gamma=0.9,
    )

    destination = trainer.save_checkpoint(
        tmp_path / "checkpoint", provenance=provenance, rl_config=config
    )
    loaded = load_rl_checkpoint(destination, device="cpu")

    assert (destination / "pretrained_model" / "model.safetensors").is_file()
    assert loaded.state.stage == "online"
    assert loaded.state.sampler_state is not None
    torch.testing.assert_close(
        loaded.state.sampler_state["generator"], trainer.sampler_state()["generator"]
    )
    saved_value = loaded.state.trainer_state["value_network"]
    for key, value in trainer.value_network.state_dict().items():
        torch.testing.assert_close(saved_value[key], value.cpu())

    resumed_current = DiffusionRLAdapter(loaded.current, current.trace_config)
    resumed_old = _adapter(tmp_path / "resumed-old")
    resumed_old.policy.load_state_dict(resumed_current.policy.state_dict(), strict=True)
    resumed = OnlineTrainer(
        current_policy=resumed_current,
        old_policy=resumed_old,
        metrics_path=None,
        gamma=0.9,
        ppo_epochs=1,
    )
    _restore_online_state(loaded, resumed)
    assert len(resumed._metrics_snapshot) == resumed.counters.metrics_rows == 1
    reloaded = load_rl_checkpoint(
        resumed.save_checkpoint(
            tmp_path / "resumed-checkpoint", provenance=provenance, rl_config=config
        ),
        device="cpu",
    )
    assert reloaded.state.counters.metrics_rows == 1


def test_nonfinite_value_loss_does_not_step_actor_or_value(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    old = _adapter(tmp_path / "old")
    old.policy.load_state_dict(current.policy.state_dict(), strict=True)
    trainer = OnlineTrainer(
        current_policy=current,
        old_policy=old,
        metrics_path=tmp_path / "metrics.jsonl",
        gamma=0.9,
        ppo_epochs=1,
    )
    rollout = trainer.collect(_AsyncDoneEnv(), decisions=1, seed=9)
    actor_before = [parameter.detach().clone() for parameter in current.policy.parameters()]
    value_before = [parameter.detach().clone() for parameter in trainer.value_network.parameters()]
    trainer._value = lambda observation: torch.full(  # type: ignore[method-assign]
        (observation.batch_size(), 1), float("nan"), device=trainer.value_device
    )

    with pytest.raises(ValueError, match="value loss"):
        trainer.update(rollout)

    for before, after in zip(actor_before, current.policy.parameters(), strict=True):
        torch.testing.assert_close(after, before)
    for before, after in zip(value_before, trainer.value_network.parameters(), strict=True):
        torch.testing.assert_close(after, before)


def test_sampler_state_restores_explicit_rollout_generator(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    trainer = OnlineTrainer(current_policy=current, metrics_path=tmp_path / "metrics.jsonl", seed=17)
    state = trainer.sampler_state()
    expected = torch.randn(5, generator=trainer._generator)
    trainer.restore_sampler_state(state)
    actual = torch.randn(5, generator=trainer._generator)
    torch.testing.assert_close(actual, expected)


def test_rgb_frame_is_not_mistaken_for_three_step_history() -> None:
    frame = torch.zeros(2, 3, 8, 8)
    history = _reset_history({"observation.images.front": frame}, n_obs_steps=3)
    assert history["observation.images.front"].shape == (2, 3, 3, 8, 8)


def test_generic_pixels_observation_is_preserved() -> None:
    state = np.zeros((2, 39), dtype=np.float32)
    pixels = np.zeros((2, 3, 8, 8), dtype=np.uint8)
    features = _raw_feature_mapping(
        {"agent_pos": state, "pixels": pixels}, num_envs=2
    )
    assert "observation.state" in features
    assert "pixels" in features


def test_action_space_bounds_are_checked_before_environment_step() -> None:
    class Space:
        shape = (2,)
        low = np.full(2, -1.0, dtype=np.float32)
        high = np.full(2, 1.0, dtype=np.float32)

    class Env:
        single_action_space = Space()

    with pytest.raises(ValueError, match="outside"):
        _validate_env_action(Env(), np.full((2, 2), 2.0, dtype=np.float32), num_envs=2)


def test_action_clipping_projects_to_box_bounds() -> None:
    class Space:
        shape = (2,)
        low = np.full(2, -1.0, dtype=np.float32)
        high = np.full(2, 1.0, dtype=np.float32)

    class Env:
        single_action_space = Space()

    clipped = _clip_env_action(
        Env(), np.asarray([[-1.5, 0.25], [0.5, 2.0]], dtype=np.float32), num_envs=2
    )
    np.testing.assert_array_equal(
        clipped, np.asarray([[-1.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    )


def test_failed_metric_logger_rolls_back_new_metrics_file(tmp_path: Path) -> None:
    current = _adapter(tmp_path / "current")
    trainer = OnlineTrainer(
        current_policy=current,
        metrics_path=tmp_path / "metrics.jsonl",
        logger=lambda _metrics: (_ for _ in ()).throw(RuntimeError("logger failed")),
    )
    rollout = trainer.collect(_AsyncDoneEnv(), decisions=1, seed=23)

    with pytest.raises(RuntimeError, match="logger failed"):
        trainer.update(rollout)

    assert not (tmp_path / "metrics.jsonl").exists()
    assert trainer.counters.metrics_rows == 0
