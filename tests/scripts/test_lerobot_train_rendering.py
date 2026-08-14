#!/usr/bin/env python

from unittest.mock import MagicMock

import pytest

from lerobot.scripts import lerobot_train


def test_log_eval_to_wandb_logs_metrics_without_video() -> None:
    wandb_logger = MagicMock(spec_set=lerobot_train.WandBLogger)
    wandb_log_dict = {"avg_sum_reward": 1.0}

    lerobot_train._log_eval_to_wandb(wandb_logger, wandb_log_dict, [], step=10)

    wandb_logger.log_dict.assert_called_once_with(wandb_log_dict, 10, mode="eval")
    wandb_logger.log_video.assert_not_called()


def test_log_eval_to_wandb_logs_first_video_when_available() -> None:
    wandb_logger = MagicMock(spec_set=lerobot_train.WandBLogger)
    wandb_log_dict = {"avg_sum_reward": 1.0}
    video_paths = ["first.mp4", "second.mp4"]

    lerobot_train._log_eval_to_wandb(wandb_logger, wandb_log_dict, video_paths, step=20)

    wandb_logger.log_dict.assert_called_once_with(wandb_log_dict, 20, mode="eval")
    wandb_logger.log_video.assert_called_once_with("first.mp4", 20, mode="eval")


@pytest.mark.parametrize("raise_error", [False, True])
def test_training_eval_env_scope_closes_once_on_exit(monkeypatch, raise_error) -> None:
    eval_env = {"task_group": {0: object()}}
    close_envs = MagicMock()
    monkeypatch.setattr(lerobot_train, "close_envs", close_envs)

    if raise_error:
        with (
            pytest.raises(RuntimeError, match="training failed"),
            lerobot_train._close_eval_envs_after_training(eval_env),
        ):
            raise RuntimeError("training failed")
    else:
        with lerobot_train._close_eval_envs_after_training(eval_env):
            pass

    close_envs.assert_called_once_with(eval_env)
