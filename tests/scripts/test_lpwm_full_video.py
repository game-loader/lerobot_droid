"""Video capture must not change task scope, episode ordering or simulator results."""

import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.lpwm_full import evaluate


@pytest.mark.parametrize("video,codec_error", [(False, False), (True, False), (True, True)])
def test_first_episode_video_keeps_ten_episode_plan(tmp_path, monkeypatch, video, codec_error):
    made, calls, writers = [], [], []

    class Env:
        _init_states = list(range(50))

        def __init__(self, **kwargs):
            self.closed = False
            made.append(self)

        def close(self):
            self.closed = True

    class Writer:
        def __init__(self, path):
            self.path = path
            self.error = "codec unavailable" if codec_error else None
            self.closed = False
            writers.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "lerobot.envs.libero",
        SimpleNamespace(TASK_SUITE_MAX_STEPS={"suite": 520}, LiberoEnv=Env),
    )
    monkeypatch.setattr(evaluate.native, "OptionalVideoWriter", Writer)

    def rollout(env, policy, plan, mapping, stats, language, device, max_steps, image_size, **kwargs):
        calls.append((plan.copy(), kwargs.get("video_writer"), max_steps, image_size))
        return {
            **plan,
            "success": plan["episode_index"] % 2 == 0,
            "control_steps": 5,
            "executed_action_components": 35,
            "clipped_action_components": 0,
        }

    monkeypatch.setattr(evaluate.native, "rollout_episode", rollout)
    bundle = {
        "policy": object(),
        "mapping": {},
        "state_stats": {},
        "language_cache": SimpleNamespace(select=lambda *a: ({}, {})),
    }
    item = {"suite": "suite", "task_id": 0, "global_task_id": 30}
    task = SimpleNamespace(name="task", language="instruction")
    result = evaluate.evaluate_task(
        item,
        bundle,
        {"suite": SimpleNamespace(get_task=lambda _: task)},
        False,
        "validation",
        torch.device("cpu"),
        tmp_path if video else None,
    )
    assert len(made) == 1 and made[0].closed
    assert [c[0] for c in calls] == evaluate.native.episode_plan(50, 10, 42, "validation", 30)
    assert all(c[2:] == (520, 128) for c in calls)
    assert result["num_episodes"] == 10 and result["successes"] == 5
    assert len(writers) == int(video)
    if video:
        assert writers[0].closed
        assert writers[0].path == tmp_path / "task_030_episode_000.mp4"
        assert calls[0][1] is writers[0] and all(c[1] is None for c in calls[1:])
        row = result["episodes"][0]
        assert ("video_error" in row) == codec_error
        assert ("video" in row) != codec_error
    else:
        assert all("video" not in row and "video_error" not in row for row in result["episodes"])


def test_serial_dispatch_passes_video_output_dir(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(evaluate, "evaluate_task", lambda *args: seen.append(args) or {})
    args = SimpleNamespace(
        workers=1, preflight=False, seed_namespace="validation", output=tmp_path / "replay.json", video=True
    )
    assert list(evaluate.task_results(args, [1], {}, {}, "cpu")) == [{}]
    assert seen[0][-1] == tmp_path / "replay_videos"


def test_spawn_worker_passes_video_dir(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "lerobot.envs.libero", SimpleNamespace(_get_suite=lambda s: s))
    monkeypatch.setattr(evaluate, "_WORKER_OPTIONS", (False, "validation", "cpu", tmp_path), raising=False)
    monkeypatch.setattr(evaluate, "_WORKER_BUNDLE", {}, raising=False)
    seen = []
    monkeypatch.setattr(evaluate, "evaluate_task", lambda *args: seen.append(args) or {})
    evaluate.evaluate_worker_task({"suite": "suite"})
    assert seen[0][-1] == tmp_path


def test_gallery_escapes_text_keeps_failures_and_requires_files(tmp_path):
    import json

    from scripts.lpwm_full.video_gallery import build_gallery

    folder = tmp_path / "full130"
    video_dir = folder / "formal_videos"
    video_dir.mkdir(parents=True)
    (video_dir / "task_000_episode_000.mp4").write_bytes(b"fixture")
    result = {
        "status": "complete",
        "checkpoint": {"step": 170000},
        "successes": 0,
        "num_episodes": 1,
        "success_rate": 0.0,
        "protocol": {"seed_namespace": "validation"},
        "per_task": [
            {
                "task_id": 0,
                "suite": "libero_spatial",
                "task_name": "task",
                "language": "<script>unsafe</script>",
                "successes": 0,
                "episodes": [
                    {
                        "episode_index": 0,
                        "success": False,
                        "control_steps": 280,
                        "seed": 42,
                        "init_state_index": 0,
                        "video": "/remote/task_000_episode_000.mp4",
                    }
                ],
            }
        ],
    }
    (folder / "formal.json").write_text(json.dumps(result))
    assert build_gallery(tmp_path)["video_count"] == 1
    page = (tmp_path / "index.html").read_text()
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in page
    assert 'data-outcome="failure"' in page
    assert 'src="full130/formal_videos/task_000_episode_000.mp4"' in page
    (video_dir / "task_000_episode_000.mp4").unlink()
    with pytest.raises(FileNotFoundError):
        build_gallery(tmp_path)
