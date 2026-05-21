#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Mapping
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.utils.constants import PRETRAINED_MODEL_DIR

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


def cfg_to_group(
    cfg: TrainPipelineConfig, return_list: bool = False, truncate_tags: bool = False, max_tag_length: int = 64
) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""

    def _maybe_truncate(tag: str) -> str:
        """Truncate tag to max_tag_length characters if required.

        wandb rejects tags longer than 64 characters.
        See: https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py
        """
        if len(tag) <= max_tag_length:
            return tag
        return tag[:max_tag_length]

    if cfg.is_reward_model_training:
        trainable_tag = f"reward_model:{cfg.reward_model.type}"
    else:
        trainable_tag = f"policy:{cfg.policy.type}"
    lst = [
        trainable_tag,
        f"seed:{cfg.seed}",
    ]
    if cfg.dataset is not None:
        lst.append(f"dataset:{cfg.dataset.repo_id}")
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    if truncate_tags:
        lst = [_maybe_truncate(tag) for tag in lst]
    return lst if return_list else "-".join(lst)


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    # Get the WandB run ID.
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]
    return wandb_run_id


def get_safe_wandb_artifact_name(name: str):
    """WandB artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")


class TrainLogger(Protocol):
    """Common interface for experiment loggers used by the training pipeline."""

    def log_policy(self, checkpoint_dir: Path) -> None: ...

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ) -> None: ...

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None: ...


def make_train_logger(cfg: TrainPipelineConfig) -> TrainLogger | None:
    """Create the configured experiment logger.

    ``cfg.logger`` selects the backend: ``"auto"`` keeps the legacy behavior and uses WandB when
    ``wandb.enable=true``; otherwise it uses SwanLab when ``swanlab.enable=true``. Explicit values
    (``"wandb"``/``"swanlab"``) select that backend directly; ``"none"`` disables remote logging.
    """
    logger_backend = getattr(cfg, "logger", "auto")
    if logger_backend == "none":
        return None

    if logger_backend == "auto" and cfg.wandb.enable and cfg.swanlab.enable:
        logging.info("Both WandB and SwanLab are enabled; using WandB. Set logger=swanlab to switch.")

    if logger_backend == "wandb" or (logger_backend == "auto" and cfg.wandb.enable):
        if cfg.wandb.project:
            return WandBLogger(cfg)
        return None

    if logger_backend == "swanlab" or (logger_backend == "auto" and cfg.swanlab.enable):
        if cfg.swanlab.project:
            return SwanLabLogger(cfg)
        return None

    return None


class WandBLogger:
    """A helper class to log object using wandb."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        # Set up WandB.
        os.environ["WANDB_SILENT"] = "True"
        import wandb

        wandb_run_id = (
            cfg.wandb.run_id
            if cfg.wandb.run_id
            else get_wandb_run_id_from_filesystem(self.log_dir)
            if cfg.resume
            else None
        )
        wandb.init(
            id=wandb_run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.job_name,
            notes=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True, truncate_tags=True) if self.cfg.add_tags else None,
            dir=self.log_dir,
            config=cfg.to_dict(),
            # TODO(rcadene): try set to True
            save_code=False,
            # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
            job_type="train_eval",
            resume="must" if cfg.resume else None,
            mode=self.cfg.mode if self.cfg.mode in ["online", "offline", "disabled"] else "online",
        )
        run_id = wandb.run.id
        # NOTE: We will override the cfg.wandb.run_id with the wandb run id.
        # This is because we want to be able to resume the run from the wandb run id.
        cfg.wandb.run_id = run_id
        # Handle custom step key for rl asynchronous training.
        self._wandb_custom_step_key: set[str] | None = None
        logging.info(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
        self._wandb = wandb

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to wandb."""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_wandb_artifact_name(artifact_name)
        artifact = self._wandb.Artifact(artifact_name, type="model")
        pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

        # Check if this is a PEFT model (has adapter files instead of model.safetensors)
        adapter_model_file = pretrained_model_dir / "adapter_model.safetensors"
        standard_model_file = pretrained_model_dir / SAFETENSORS_SINGLE_FILE

        if adapter_model_file.exists():
            # PEFT model: add adapter files and configs
            artifact.add_file(adapter_model_file)
            adapter_config_file = pretrained_model_dir / "adapter_config.json"
            if adapter_config_file.exists():
                artifact.add_file(adapter_config_file)
            # Also add the policy config which is needed for loading
            config_file = pretrained_model_dir / "config.json"
            if config_file.exists():
                artifact.add_file(config_file)
        elif standard_model_file.exists():
            # Standard model: add the single safetensors file
            artifact.add_file(standard_model_file)
        else:
            logging.warning(
                f"No {SAFETENSORS_SINGLE_FILE} or adapter_model.safetensors found in {pretrained_model_dir}. "
                "Skipping model artifact upload to WandB."
            )
            return

        self._wandb.log_artifact(artifact)

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        # NOTE: This is not simple. Wandb step must always monotonically increase and it
        # increases with each wandb.log call, but in the case of asynchronous RL for example,
        # multiple time steps is possible. For example, the interaction step with the environment,
        # the training step, the evaluation step, etc. So we need to define a custom step key
        # to log the correct step for each metric.
        if custom_step_key is not None:
            if self._wandb_custom_step_key is None:
                self._wandb_custom_step_key = set()
            new_custom_key = f"{mode}/{custom_step_key}"
            if new_custom_key not in self._wandb_custom_step_key:
                self._wandb_custom_step_key.add(new_custom_key)
                self._wandb.define_metric(new_custom_key, hidden=True)

        for k, v in d.items():
            if not isinstance(v, (int | float | str)):
                logging.warning(
                    f'WandB logging of key "{k}" was ignored as its type "{type(v)}" is not handled by this wrapper.'
                )
                continue

            # Do not log the custom step key itself.
            if self._wandb_custom_step_key is not None and k in self._wandb_custom_step_key:
                continue

            if custom_step_key is not None:
                value_custom_step = d[custom_step_key]
                data = {f"{mode}/{k}": v, f"{mode}/{custom_step_key}": value_custom_step}
                self._wandb.log(data)
                continue

            self._wandb.log(data={f"{mode}/{k}": v}, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        wandb_video = self._wandb.Video(video_path, fps=self.env_fps, format="mp4")
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)


def _convert_video_to_gif(video_path: str | Path, output_dir: Path, fps: int | None = None) -> Path:
    """Convert a video file to GIF for SwanLab, which only accepts GIF videos."""
    video_path = Path(video_path)
    if video_path.suffix.lower() == ".gif":
        return video_path

    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{video_path.stem}.gif"
    if gif_path.exists() and gif_path.stat().st_mtime >= video_path.stat().st_mtime:
        return gif_path

    try:
        from lerobot.utils.import_utils import require_package

        require_package("av", extra="av-dep")
        import av
        from PIL import Image

        frames = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))

        if not frames:
            raise RuntimeError(f"No frames decoded from {video_path}.")

        duration_ms = int(1000 / fps) if fps else 100
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
    except Exception:
        if gif_path.exists():
            gif_path.unlink()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise
        import subprocess

        cmd = [ffmpeg, "-y", "-i", str(video_path)]
        if fps:
            cmd.extend(["-vf", f"fps={fps}"])
        cmd.append(str(gif_path))
        subprocess.run(cmd, check=True)

    return gif_path


def _flatten_swanlab_metrics(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        flat: dict[str, Any] = {}
        for key, nested_value in value.items():
            flat.update(_flatten_swanlab_metrics(f"{prefix}/{key}", nested_value))
        return flat
    if isinstance(value, (bool, int, float, str)):
        return {prefix: value}
    if isinstance(value, Path):
        return {prefix: str(value)}
    return {}


def _get_swanlab_run_url(swanlab_module: Any, run: Any) -> str | None:
    for obj in (run, swanlab_module):
        for attr in ("get_url", "get_run_url"):
            get_url = getattr(obj, attr, None)
            if callable(get_url):
                try:
                    url = get_url()
                except TypeError:
                    continue
                if url:
                    return str(url)
        for attr_path in (
            ("public", "cloud", "experiment_url"),
            ("url",),
            ("run_url",),
            ("cloud_url",),
            ("web_url",),
        ):
            value = obj
            for attr in attr_path:
                value = getattr(value, attr, None)
                if value is None:
                    break
            if value:
                return str(value)
    return None


class SwanLabLogger:
    """A helper class to log objects using SwanLab."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.swanlab
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None

        import swanlab

        init_kwargs = {
            "project": self.cfg.project,
            "workspace": self.cfg.workspace,
            "experiment_name": self.job_name,
            "description": self.cfg.description,
            "logdir": str(self.log_dir),
            "config": cfg.to_dict(),
            "job_type": "train_eval",
        }
        if cfg.resume:
            init_kwargs["resume"] = "must"
        if self.cfg.mode in ["cloud", "local", "offline", "disabled"]:
            init_kwargs["mode"] = self.cfg.mode
        if cfg.swanlab.run_id:
            init_kwargs["id"] = cfg.swanlab.run_id
        if self.cfg.add_tags:
            init_kwargs["tags"] = cfg_to_group(cfg, return_list=True)

        run = swanlab.init(**init_kwargs)

        run_id = getattr(run, "id", None) or getattr(run, "_id", None)
        if run_id:
            cfg.swanlab.run_id = run_id
        logging.info(colored("Logs will be synced with SwanLab.", "blue", attrs=["bold"]))
        url = _get_swanlab_run_url(swanlab, run)
        if url:
            logging.info(f"Track this run --> {colored(url, 'yellow', attrs=['bold'])}")
        self._swanlab = swanlab
        self._video_dir = self.log_dir / "swanlab_videos"

    def log_policy(self, checkpoint_dir: Path):
        """SwanLab mode intentionally does not upload checkpoint artifacts."""
        _ = checkpoint_dir
        return

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        if custom_step_key is not None:
            value_custom_step = d[custom_step_key]
            data: dict[str, Any] = {}
            for k, v in d.items():
                if k == custom_step_key:
                    continue
                data.update(_flatten_swanlab_metrics(f"{mode}/{k}", v))
            step_value = value_custom_step if isinstance(value_custom_step, int) else step
            self._swanlab.log(data, step=step_value)
            return

        data: dict[str, Any] = {}
        for k, v in d.items():
            data.update(_flatten_swanlab_metrics(f"{mode}/{k}", v))
        self._swanlab.log(data, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        gif_path = _convert_video_to_gif(video_path, self._video_dir, fps=self.env_fps)
        swanlab_video = self._swanlab.Video(str(gif_path))
        self._swanlab.log({f"{mode}/video": swanlab_video}, step=step)
