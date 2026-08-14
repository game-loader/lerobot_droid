#!/usr/bin/env python

import json

import pytest

from lerobot.envs import MoyaNewtonEnvConfig, PushtEnv
from lerobot.scripts import lerobot_eval
from lerobot.scripts.lerobot_eval import resolve_max_episodes_rendered


class CloseTrackingEnv:
    def __init__(self) -> None:
        self.close_calls = 0
        self.use_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _stub_run_one(task_group, task_id, _env, **_kwargs):
    _env.use_calls += 1
    return (
        task_group,
        task_id,
        {
            "sum_rewards": [1.0],
            "max_rewards": [1.0],
            "successes": [True],
            "video_paths": [],
        },
    )


def test_renderable_environment_preserves_requested_episode_count() -> None:
    assert resolve_max_episodes_rendered(PushtEnv(), 10) == 10


def test_headless_moya_environment_disables_rendered_episodes() -> None:
    cfg = MoyaNewtonEnvConfig()

    assert resolve_max_episodes_rendered(cfg, 10) == 0
    assert resolve_max_episodes_rendered(cfg, 0) == 0


def test_eval_environment_reuse_is_opt_in() -> None:
    assert PushtEnv().supports_eval_env_reuse is False
    assert MoyaNewtonEnvConfig().supports_eval_env_reuse is True


def test_negative_rendered_episode_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        resolve_max_episodes_rendered(PushtEnv(), -1)


def test_run_one_does_not_create_video_directory_when_rendering_is_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lerobot_eval, "eval_one", lambda *args, **kwargs: {})
    videos_dir = tmp_path / "videos"

    lerobot_eval.run_one(
        "task_group",
        0,
        None,
        policy=None,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=1,
        max_episodes_rendered=0,
        videos_dir=videos_dir,
        return_episode_data=False,
        start_seed=None,
    )

    assert not videos_dir.exists()


def test_save_eval_info_creates_output_directory(tmp_path) -> None:
    info = {"overall": {"pc_success": 100.0}}
    output_dir = tmp_path / "new" / "nested" / "output"

    lerobot_eval._save_eval_info(info, output_dir)

    assert json.loads((output_dir / "eval_info.json").read_text()) == info


@pytest.mark.parametrize("max_parallel_tasks", [1, 2])
def test_eval_policy_all_closes_environments_by_default(monkeypatch, max_parallel_tasks) -> None:
    envs = {task_id: CloseTrackingEnv() for task_id in range(2)}
    monkeypatch.setattr(lerobot_eval, "run_one", _stub_run_one)

    lerobot_eval.eval_policy_all(
        envs={"task_group": envs},
        policy=None,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=1,
        max_parallel_tasks=max_parallel_tasks,
    )

    assert [env.use_calls for env in envs.values()] == [1, 1]
    assert [env.close_calls for env in envs.values()] == [1, 1]


@pytest.mark.parametrize("max_parallel_tasks", [1, 2])
def test_eval_policy_all_can_reuse_environments(monkeypatch, max_parallel_tasks) -> None:
    envs = {task_id: CloseTrackingEnv() for task_id in range(2)}
    monkeypatch.setattr(lerobot_eval, "run_one", _stub_run_one)
    kwargs = {
        "envs": {"task_group": envs},
        "policy": None,
        "env_preprocessor": None,
        "env_postprocessor": None,
        "preprocessor": None,
        "postprocessor": None,
        "n_episodes": 1,
        "max_parallel_tasks": max_parallel_tasks,
        "close_envs_after_eval": False,
    }

    lerobot_eval.eval_policy_all(**kwargs)
    lerobot_eval.eval_policy_all(**kwargs)

    assert [env.use_calls for env in envs.values()] == [2, 2]
    assert [env.close_calls for env in envs.values()] == [0, 0]
