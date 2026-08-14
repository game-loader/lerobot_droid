#!/usr/bin/env python

import pytest

from lerobot.envs import MoyaNewtonEnvConfig, PushtEnv
from lerobot.scripts import lerobot_eval
from lerobot.scripts.lerobot_eval import resolve_max_episodes_rendered


def test_renderable_environment_preserves_requested_episode_count() -> None:
    assert resolve_max_episodes_rendered(PushtEnv(), 10) == 10


def test_headless_moya_environment_disables_rendered_episodes() -> None:
    cfg = MoyaNewtonEnvConfig()

    assert resolve_max_episodes_rendered(cfg, 10) == 0
    assert resolve_max_episodes_rendered(cfg, 0) == 0


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
