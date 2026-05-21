# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path
from types import SimpleNamespace

from lerobot.common import wandb_utils
from lerobot.common.wandb_utils import SwanLabLogger, make_train_logger
from lerobot.configs.default import DatasetConfig, SwanLabConfig, WandBConfig


class _FakeWandBLogger:
    def __init__(self, cfg):
        self.cfg = cfg


class _FakeSwanLabLogger:
    def __init__(self, cfg):
        self.cfg = cfg


def _minimal_cfg(**overrides):
    cfg = SimpleNamespace(
        logger="auto",
        wandb=WandBConfig(),
        swanlab=SwanLabConfig(),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_make_train_logger_preserves_wandb_auto_precedence(monkeypatch):
    cfg = _minimal_cfg()
    cfg.wandb.enable = True
    cfg.swanlab.enable = True
    monkeypatch.setattr(wandb_utils, "WandBLogger", _FakeWandBLogger)
    monkeypatch.setattr(wandb_utils, "SwanLabLogger", _FakeSwanLabLogger)

    logger = make_train_logger(cfg)

    assert isinstance(logger, _FakeWandBLogger)
    assert cfg.swanlab.enable is True


def test_make_train_logger_can_select_swanlab(monkeypatch):
    cfg = _minimal_cfg(logger="swanlab")
    cfg.wandb.enable = True
    cfg.swanlab.enable = True
    monkeypatch.setattr(wandb_utils, "WandBLogger", _FakeWandBLogger)
    monkeypatch.setattr(wandb_utils, "SwanLabLogger", _FakeSwanLabLogger)

    logger = make_train_logger(cfg)

    assert isinstance(logger, _FakeSwanLabLogger)


def test_make_train_logger_returns_none_when_logger_disabled():
    cfg = _minimal_cfg(logger="none")
    cfg.wandb.enable = True
    cfg.swanlab.enable = True

    assert make_train_logger(cfg) is None


def test_swanlab_logger_logs_nested_eval_metrics_and_skips_policy_artifact(monkeypatch, tmp_path):
    calls = {"init": None, "log": [], "video": []}

    class _FakeVideo:
        def __init__(self, path):
            calls["video"].append(path)
            self.path = path

    fake_swanlab = SimpleNamespace(
        init=lambda **kwargs: calls.__setitem__("init", kwargs) or SimpleNamespace(id="run-1"),
        log=lambda data, step=None: calls["log"].append((data, step)),
        Video=_FakeVideo,
        get_url=lambda: "https://example.test/run-1",
    )
    monkeypatch.setitem(sys.modules, "swanlab", fake_swanlab)
    monkeypatch.setattr(wandb_utils, "_convert_video_to_gif", lambda path, output_dir, fps=None: Path(path))

    cfg = SimpleNamespace(
        swanlab=SwanLabConfig(enable=True, project="proj"),
        wandb=WandBConfig(),
        output_dir=tmp_path,
        job_name="job",
        env=SimpleNamespace(fps=12, type="env"),
        dataset=DatasetConfig(repo_id="user/repo"),
        is_reward_model_training=False,
        policy=SimpleNamespace(type="act"),
        seed=0,
        resume=False,
        to_dict=lambda: {"cfg": "value"},
    )
    logger = SwanLabLogger(cfg)

    logger.log_dict({"metric": 1.0, "overall": {"video_paths": [tmp_path / "eval.mp4"]}}, step=3, mode="eval")
    logger.log_video(str(tmp_path / "eval.gif"), step=3, mode="eval")
    logger.log_policy(tmp_path / "checkpoint")

    assert calls["init"]["project"] == "proj"
    assert cfg.swanlab.run_id == "run-1"
    assert calls["log"][0][0]["eval/metric"] == 1.0
    assert "eval/overall/video_paths" not in calls["log"][0][0]
    assert calls["log"][0][1] == 3
    assert calls["video"] == [str(tmp_path / "eval.gif")]
    assert calls["log"][1][0]["eval/video"].path == str(tmp_path / "eval.gif")
    assert len(calls["log"]) == 2
