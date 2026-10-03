"""Evaluator contract tests, NOT evidence of policy success in the real simulator.

Only helper logic, a tiny real checkpoint, and explicitly mocked rollout control
are exercised. Native LIBERO/MuJoCo is deliberately unnecessary here.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "lpwm_ab_evaluate", Path(__file__).parents[2] / "scripts/lpwm_ab/evaluate.py"
)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)

CAMERAS = ["observation.images.image", "observation.images.image2"]


def robot_state():
    return {
        "eef": {"pos": np.array([1, 2, 3]), "quat": np.array([0, 0, 0, 1])},
        "gripper": {"qpos": np.array([0.1, -0.1])},
    }


def observation():
    return {
        "robot_state": robot_state(),
        "pixels": {
            CAMERAS[0]: np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3),
            CAMERAS[1]: np.full((4, 4, 3), 200, dtype=np.uint8),
        },
    }


def language_metadata():
    # Deliberately unlike simulator suite task order: selection must be by task text.
    return {
        "language": {"task_ids": [13, 7]},
        "tasks": {"7": "Pick up the Red Bowl!", "13": "open the drawer"},
    }


def language_cache():
    return evaluate.LanguageCache(
        language_metadata(),
        np.stack((np.zeros((3, 12)), np.ones((3, 12)))),
        np.array([[1, 1, 0], [1, 1, 1]]),
        12,
    )


def test_cli_defaults_and_final_namespace():
    args = evaluate.parse_args(["--checkpoint", "frozen", "--output", "result.json"])
    assert args.episodes_per_task == 10
    assert args.max_steps is None and args.task_ids is None
    assert args.seed_namespace == "validation"
    final = evaluate.parse_args(
        [
            "--checkpoint",
            "frozen",
            "--output",
            "result.json",
            "--task-ids",
            "2",
            "4",
            "--seed-namespace",
            "final",
            "--video",
        ]
    )
    assert final.task_ids == [2, 4] and final.seed_namespace == "final" and final.video


def test_language_selection_is_natural_description_not_task_id():
    tensors, info = language_cache().select(" PICK UP THE red bowl. ", "cpu")
    assert info["cache_task_id"] == 7 and info["cache_row"] == 1
    torch.testing.assert_close(tensors[evaluate.LANGUAGE], torch.ones(1, 3, 12))
    assert tensors[evaluate.LANGUAGE_MASK].dtype == torch.bool
    with pytest.raises(ValueError, match="No cached language"):
        language_cache().select("pick up the BLUE bowl", "cpu")
    with pytest.raises(ValueError, match="No cached language"):
        language_cache().select("7", "cpu")


@pytest.mark.parametrize("case", ["duplicate", "width", "masked", "nonfinite"])
def test_language_cache_validation(case):
    metadata = language_metadata()
    embeddings = np.ones((2, 3, 12), dtype=np.float32)
    masks = np.ones((2, 3), dtype=bool)
    if case == "duplicate":
        metadata["tasks"]["13"] = "pick up the red bowl"
    if case == "width":
        embeddings = embeddings[:, :, :-1]
    if case == "masked":
        masks[0] = False
    if case == "nonfinite":
        embeddings[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate.LanguageCache(metadata, embeddings, masks, 12)


def test_canonical8_quaternion_matches_repository_processor():
    from lerobot.processor.env_processor import LiberoProcessorStep

    state = robot_state()
    for quaternion in ([0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]):
        state["eef"]["quat"] = np.array(quaternion, dtype=np.float32)
        result = evaluate.canonical_state(state)
        expected_rotation = LiberoProcessorStep()._quat2axisangle(torch.tensor([quaternion]).float())[0]
        np.testing.assert_array_equal(result[3:6], expected_rotation.numpy())
        np.testing.assert_array_equal(result[:3], state["eef"]["pos"])
        np.testing.assert_allclose(result[6:], [0.1, -0.1])
        assert result.shape == (8,) and result.dtype == np.float32
    state["eef"]["quat"][0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate.canonical_state(state)


def test_exact_single_rotation_and_pil_training_resize():
    raw = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    actual = evaluate.rotate_resize_rgb(raw, 3)
    expected = np.asarray(
        Image.fromarray(raw[::-1, ::-1].copy()).convert("RGB").resize((3, 3), Image.Resampling.BILINEAR)
    )
    np.testing.assert_array_equal(actual, expected)
    # No resize: directly observe the 180-degree flip, not a transpose or double flip.
    square = observation()["pixels"][CAMERAS[0]]
    np.testing.assert_array_equal(evaluate.rotate_resize_rgb(square, 4), square[::-1, ::-1])


def test_camera_mapping_is_semantic_and_rejects_unknowns():
    mapping = evaluate.camera_mapping(CAMERAS[::-1])
    assert list(mapping) == ["robot0_eye_in_hand_image", "agentview_image"]
    assert mapping["agentview_image"] == CAMERAS[0]
    for keys in (["observation.images.front", CAMERAS[1]], [CAMERAS[0]], [CAMERAS[0]] * 2):
        with pytest.raises(ValueError):
            evaluate.camera_mapping(keys)


def test_batch_has_exact_saved_normalization_no_targets_and_correct_view_mapping():
    mean = np.arange(8, dtype=np.float32)
    std = np.arange(1, 9, dtype=np.float32)
    language, _ = language_cache().select("open the drawer", "cpu")
    obs = observation()
    batch = evaluate.observation_batch(
        obs,
        evaluate.camera_mapping(CAMERAS[::-1]),
        (mean, std),
        language,
        "cpu",
        image_size=4,
    )
    np.testing.assert_array_equal(
        batch[evaluate.STATE][0].numpy(), (evaluate.canonical_state(obs["robot_state"]) - mean) / std
    )
    np.testing.assert_allclose(batch[CAMERAS[1]].numpy(), 200 / 255)
    np.testing.assert_array_equal(
        (batch[CAMERAS[0]][0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
        obs["pixels"][CAMERAS[0]][::-1, ::-1],
    )
    assert "action" not in batch and not any(key.startswith("world.") for key in batch)
    assert batch[CAMERAS[0]].shape == (1, 3, 4, 4)


def test_state_statistics_reject_invalid_but_never_recompute():
    stats = {"mean": [2] * 8, "std": [1e-7] * 8}
    mean, std = evaluate.state_statistics(stats)
    np.testing.assert_array_equal(std, np.full(8, 1e-7, dtype=np.float32))
    assert (mean == 2).all()
    with pytest.raises(ValueError):
        evaluate.state_statistics({"mean": [0] * 8, "std": [0] * 8})


def test_native_action_clipping_and_no_gripper_inversion():
    action = torch.tensor([[2, -2, 0.5, -0.5, 0, 1, -0.25]])
    clipped, count = evaluate.clip_native_action(action)
    np.testing.assert_array_equal(clipped, [1, -1, 0.5, -0.5, 0, 1, -0.25])
    assert clipped.dtype == np.float32 and count == 2
    for invalid in (np.zeros((2, 7)), np.full(7, np.nan), np.full(7, np.inf)):
        with pytest.raises(ValueError, match="finite native"):
            evaluate.clip_native_action(invalid)


def test_paired_reproducible_seeds_disjoint_final_init_states():
    validation = evaluate.episode_plan(50, 10, 42, "validation", 3)
    assert validation == evaluate.episode_plan(50, 10, 42, "validation", 3)
    assert validation != evaluate.episode_plan(50, 10, 43, "validation", 3)
    final = evaluate.episode_plan(50, 10, 42, "final", 3)
    assert not {row["init_state_index"] for row in validation} & {row["init_state_index"] for row in final}
    assert not {row["seed"] for row in validation} & {row["seed"] for row in final}
    assert len({row["init_state_index"] for row in validation}) == 10
    with pytest.raises(ValueError, match="pool"):
        evaluate.episode_plan(10, 10, 42, "validation", 0)


class MockEnvironment:
    """Isolated evaluator-control test double; never counted as simulator performance evidence."""

    def __init__(self, succeed_after=None, terminate_after=None, **kwargs):
        self.succeed_after = succeed_after
        self.terminate_after = terminate_after
        self.steps = 0
        self.actions = []
        self.reset_calls = []
        self.closed = False
        self._init_states = np.zeros((50, 10))
        self.task_description = "open the drawer"

    def reset(self, seed):
        self.steps = 0
        self.reset_calls.append((seed, self.init_state_id))
        self.init_state_id += 1
        return observation(), {"is_success": False}

    def step(self, action):
        self.steps += 1
        self.actions.append(action.copy())
        return (
            observation(),
            100,
            self.steps == self.terminate_after,
            False,
            {"is_success": self.steps == self.succeed_after},
        )

    def close(self):
        self.closed = True


class MockQueuedPolicy:
    """Minimal action queue so tests catch accidental chunk-based counting or reset omissions."""

    def __init__(self):
        self.resets = 0
        self.remaining = 0
        self.generated_chunks = 0
        self.calls = 0
        self.config = SimpleNamespace(
            image_size=4, horizon=16, n_obs_steps=2, n_action_steps=8, num_inference_steps=10
        )

    def reset(self):
        self.resets += 1
        self.remaining = 0

    def select_action(self, batch):
        assert "action" not in batch and not any(key.startswith("world.") for key in batch)
        self.calls += 1
        if self.remaining == 0:
            self.generated_chunks += 1
            self.remaining = 8
        self.remaining -= 1
        return torch.tensor([[2.0, 0, 0, 0, 0, 0, -0.3]])


def run_mock(env, policy, episode=0, max_steps=10):
    language, _ = language_cache().select("open the drawer", "cpu")
    return evaluate.rollout_episode(
        env,
        policy,
        {"episode_index": episode, "seed": 23 + episode, "init_state_index": episode + 2},
        evaluate.camera_mapping(CAMERAS),
        (np.zeros(8, dtype=np.float32), np.ones(8, dtype=np.float32)),
        language,
        "cpu",
        max_steps=max_steps,
        image_size=4,
    )


def test_mock_episode_control_steps_success_not_reward_and_queue_resets():
    policy = MockQueuedPolicy()
    success = run_mock(MockEnvironment(succeed_after=9), policy)
    assert success["success"] and success["control_steps"] == 9
    assert success["clipped_action_components"] == 9
    assert success["action_clipping_fraction"] == pytest.approx(1 / 7)
    assert policy.generated_chunks == 2 and policy.resets == 2
    failure_env = MockEnvironment()
    failure = run_mock(failure_env, policy, episode=1, max_steps=2)
    assert not failure["success"] and failure["reached_step_limit"]
    assert policy.generated_chunks == 3 and policy.resets == 4
    assert failure_env.reset_calls == [(24, 3)]
    assert failure_env.actions[0][6] == pytest.approx(-0.3)
    summary = evaluate.summarize_episodes([success, failure])
    assert summary["num_episodes"] == 2  # NOT 3 chunks or 11 control steps
    assert summary["successes"] == 1 and summary["pc_success"] == 50 and summary["success_rate"] == 0.5


def test_mock_done_is_not_success_and_abort_clears_queue():
    policy = MockQueuedPolicy()
    row = run_mock(MockEnvironment(terminate_after=1), policy)
    assert row["terminated"] and not row["success"] and row["control_steps"] == 1
    policy.select_action = lambda batch: np.full(7, np.nan)
    with pytest.raises(ValueError, match="finite native"):
        run_mock(MockEnvironment(), policy)
    assert policy.resets == 4 and policy.remaining == 0


def test_atomic_result_json_and_no_empty_denominator(tmp_path):
    path = tmp_path / "subdir" / "result.json"
    evaluate.atomic_json(path, {"successes": 1, "num_episodes": 2})
    assert json.loads(path.read_text())["num_episodes"] == 2
    with pytest.raises(ValueError):
        evaluate.atomic_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text())["num_episodes"] == 2
    assert not list(path.parent.glob(".result.json.*"))
    with pytest.raises(ValueError, match="denominator"):
        evaluate.summarize_episodes([])


def test_missing_libero_config_fails_before_import_or_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="interactive setup is not permitted"):
        evaluate.simulator_api()


def test_mock_full_protocol_atomic_result_and_task_denominator(tmp_path, monkeypatch):
    policy = MockQueuedPolicy()
    suite = SimpleNamespace(tasks=[0, 1], get_task=lambda task: SimpleNamespace(language="open the drawer"))
    envs = []

    def environment_factory(**kwargs):
        env = MockEnvironment(succeed_after=1 if kwargs["task_id"] == 0 else None)
        envs.append(env)
        return env

    monkeypatch.setattr(evaluate, "simulator_api", lambda: (environment_factory, suite, 280))
    monkeypatch.setattr(
        evaluate,
        "load_checkpoint",
        lambda *args: {
            "policy": policy,
            "mapping": evaluate.camera_mapping(CAMERAS),
            "state_stats": (np.zeros(8, dtype=np.float32), np.ones(8, dtype=np.float32)),
            "language_cache": language_cache(),
            "checkpoint": {"step": 100, "model_sha256": "test-double-not-real-weights"},
        },
    )
    args = evaluate.parse_args(
        [
            "--checkpoint",
            str(tmp_path),
            "--output",
            str(tmp_path / "result.json"),
            "--device",
            "cpu",
            "--episodes-per-task",
            "2",
            "--max-steps",
            "2",
        ]
    )
    result = evaluate.run_evaluation(args)
    assert result["num_episodes"] == 4 and result["successes"] == 2
    assert result["pc_success"] == 50 and result["success_rate"] == 0.5
    assert [task["pc_success"] for task in result["per_task"]] == [100, 0]
    assert result["protocol"]["local_suite_max_steps"] == 280
    assert result["protocol"]["max_control_steps"] == 2
    assert all(env.closed for env in envs)
    assert json.loads(args.output.read_text())["checkpoint"]["step"] == 100


def test_load_actual_tiny_frozen_checkpoint_and_metadata(tmp_path):
    """Real weight loading only; this does not execute or certify simulator success."""
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.lpwm_fm.configuration_lpwm_fm import LPWMFMConfig
    from lerobot.policies.lpwm_fm.modeling_lpwm_fm import LPWMFMPolicy

    config = LPWMFMConfig(
        device="cpu",
        push_to_hub=False,
        image_size=32,
        patch_size=16,
        n_kp_enc=4,
        n_kp_dec=4,
        n_kp_prior=4,
        obj_base_ch=8,
        obj_final_cnn_ch=8,
        bg_base_ch=8,
        bg_final_cnn_ch=8,
        obj_ch_mult=(1, 2),
        obj_ch_mult_prior=(1, 2),
        bg_ch_mult=(1, 2, 4),
        mlp_hidden_dim=32,
        hidden_dim=32,
        n_heads=4,
        scene_n_layers=1,
        expert_n_layers=1,
        world_hidden_dim=32,
        world_n_heads=4,
        world_n_layers=1,
        language_dim=12,
        input_features={
            **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)) for key in CAMERAS},
            evaluate.STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    original = LPWMFMPolicy(config).eval()
    original.save_pretrained(tmp_path)
    for filename, payload in {
        "experiment.json": {
            "step": 20,
            "variant": "A",
            "seed": 42,
            "camera_keys": CAMERAS,
            "action_normalization": "identity",
        },
        "state_normalization.json": {"mean": [0.0] * 8, "std": [1.0] * 8},
        "language_metadata.json": language_metadata(),
    }.items():
        (tmp_path / filename).write_text(json.dumps(payload))
    np.save(tmp_path / "language_embeddings.npy", np.zeros((2, 3, 12), dtype=np.float32))
    np.save(tmp_path / "language_masks.npy", np.ones((2, 3), dtype=bool))
    bundle = evaluate.load_checkpoint(tmp_path, torch.device("cpu"))
    assert bundle["checkpoint"]["step"] == 20
    assert bundle["checkpoint"]["model_sha256"] == evaluate.file_sha256(tmp_path / "model.safetensors")
    assert not bundle["policy"].training and not any(p.requires_grad for p in bundle["policy"].parameters())
    for key, weights in original.state_dict().items():
        torch.testing.assert_close(bundle["policy"].state_dict()[key], weights, rtol=0, atol=0)


def test_init_state_offset_is_stable_skip_in_permutation():
    full = evaluate.episode_plan(50, 10, 42, "validation", 0)
    tail = evaluate.episode_plan(50, 4, 42, "validation", 0, init_state_offset=6)
    assert [row["init_state_index"] for row in tail] == [row["init_state_index"] for row in full[6:]]
    assert [row["seed"] for row in tail] == [row["seed"] for row in full[6:]]
    for offset in (-1, 24):
        with pytest.raises(ValueError):
            evaluate.episode_plan(50, 2, 42, "validation", 0, init_state_offset=offset)
    args = evaluate.parse_args(["--checkpoint", "x", "--output", "x.json", "--init-state-offset", "6"])
    assert args.init_state_offset == 6


def test_optional_missing_video_backend_is_explicit_not_rollout_failure(tmp_path, monkeypatch):
    def absent(path):
        raise ImportError("test: optional PyAV unavailable")

    monkeypatch.setattr(evaluate, "VideoWriter", absent)
    writer = evaluate.OptionalVideoWriter(tmp_path / "test.mp4")
    writer.append_data(np.zeros((4, 4, 3), dtype=np.uint8))
    writer.close()
    assert writer.error == "ImportError: test: optional PyAV unavailable"
