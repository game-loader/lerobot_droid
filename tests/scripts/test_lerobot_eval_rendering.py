#!/usr/bin/env python

import pytest

from lerobot.envs import MoyaNewtonEnvConfig, PushtEnv
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
