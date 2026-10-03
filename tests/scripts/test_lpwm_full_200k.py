"""CPU-only contracts for fresh 200k full130/common40 task8 runs."""

import json
import sys
from types import SimpleNamespace

import pytest

from scripts.lpwm_ab import train_b_sweep as trainer
from scripts.lpwm_full import supervise
from scripts.lpwm_full.train import FULL_SCHEDULE_STEPS, validate_parallel_eval


def cli(tmp_path):
    data = tmp_path / "cache"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    runner = tmp_path / "evaluate.py"
    runner.write_text("# not executed\n")
    credential = tmp_path / "credential"
    credential.write_text("test-only")
    return [
        "--data",
        str(data),
        "--output",
        str(tmp_path / "run"),
        "--steps",
        "200000",
        "--schedule-steps",
        "200000",
        "--eval-runner",
        str(runner),
        "--eval-python",
        sys.executable,
        "--credential-file",
        str(credential),
        "--batch-size",
        "32",
        "--grad-accumulation",
        "1",
        "--eval-workers",
        "8",
    ]


def test_fresh_200k_schedule_preserves_save_and_eval_contract(tmp_path):
    args = trainer.parse_args(cli(tmp_path), allowed_schedule_steps=FULL_SCHEDULE_STEPS, allow_resume=True)
    assert args.steps == args.schedule_steps == 200000
    assert args.resume is None and not args.preflight and not args.disable_eval
    assert args.eval_workers == 8 and args.eval_episodes_per_task == 10
    assert args.batch_size == 32 and args.grad_accumulation == 1
    assert args.save_every == 5000 and args.validate_every == 500
    assert args.steps // args.save_every == 40
    assert trainer.cosine_lr(500, total_steps=200000) == pytest.approx(1e-4)
    assert trainer.cosine_lr(200000, total_steps=200000) == pytest.approx(1e-5)
    assert trainer.cosine_lr(80000, total_steps=200000) > 1e-5
    assert trainer.cosine_lr(1, total_steps=200000) == pytest.approx(2e-7)


def test_old_spatial_sweep_does_not_implicitly_allow_200k(tmp_path):
    with pytest.raises(SystemExit):
        trainer.parse_args(cli(tmp_path))


def test_short_200k_preflight_does_not_change_schedule_endpoint(tmp_path):
    values = cli(tmp_path)
    values[values.index("--steps") + 1] = "2"
    args = trainer.parse_args(values + ["--preflight"], allowed_schedule_steps=FULL_SCHEDULE_STEPS)
    assert args.steps == 2 and args.schedule_steps == 200000 and args.resume is None


@pytest.mark.parametrize("scope", [40, 130])
def test_supervisor_accepts_200k_task8_for_both_scopes(tmp_path, scope):
    args = supervise.parse_args(
        [
            "--root",
            str(tmp_path),
            "--steps",
            "200000",
            "--task-count",
            str(scope),
            "--eval-workers",
            "8",
            "--batch-size",
            "32",
            "--grad-accumulation",
            "1",
        ]
    )
    assert args.steps == 200000 and args.task_count == scope and args.eval_workers == 8
    assert args.steps // 5000 * scope * 10 == (16000 if scope == 40 else 52000)


def protocol(preflight=False):
    return {
        "protocol": {
            "parallel_workers": 8,
            "parallel_unit": "task",
            "within_task_episode_order": "serial original plan",
            "worker_start_method": "spawn",
            "actual_busy_workers": 4 if preflight else 8,
            "preflight": preflight,
        }
    }


@pytest.mark.parametrize("preflight", [False, True])
def test_task8_gate_preserves_original_episode_order(preflight):
    validate_parallel_eval(protocol(preflight), 8)


@pytest.mark.parametrize(
    "key,value",
    [
        ("parallel_workers", 1),
        ("parallel_unit", "episode"),
        ("within_task_episode_order", "random"),
        ("worker_start_method", "fork"),
        ("actual_busy_workers", 7),
    ],
)
def test_task8_gate_rejects_silent_serial_or_different_episode_scheduling(key, value):
    result = protocol()
    result["protocol"][key] = value
    with pytest.raises(ValueError, match="task8"):
        validate_parallel_eval(result, 8)


def test_legacy_serial_protocol_unchanged():
    validate_parallel_eval({"protocol": {}}, 1)


def test_200k_preflight_schedule_and_task8_must_match(tmp_path, monkeypatch):
    from scripts.lpwm_full import evaluate

    pre = tmp_path / "preflight"
    (pre / "eval").mkdir(parents=True)
    result = protocol(True)
    result["checkpoint"] = {"step": 2}
    result_path = pre / "eval/step_000002.json"
    result_path.write_text(json.dumps(result))
    (pre / "status.json").write_text(json.dumps({"status": "completed", "step": 2}))
    (pre / "experiment.json").write_text(
        json.dumps({"initial_weights_sha256": "expected", "cosine_endpoint_step": 80000})
    )
    result_path.with_suffix(".swanlab.json").write_text(
        json.dumps(
            {
                "verified_upload": True,
                "formal_result_eligible": False,
                "result_sha256": supervise.digest(result_path),
            }
        )
    )
    monkeypatch.setattr(evaluate, "validate_result", lambda *a, **kw: {})
    monkeypatch.setattr(evaluate, "protocol_suites", lambda _: evaluate.suites_for_count(40))
    args = SimpleNamespace(
        preflight_run=pre,
        preflight_step=2,
        task_count=40,
        expected_init_sha256="expected",
        steps=200000,
        eval_workers=8,
    )
    with pytest.raises(ValueError, match="200k"):
        supervise.verify_preflight(tmp_path, args)
    (pre / "experiment.json").write_text(
        json.dumps({"initial_weights_sha256": "expected", "cosine_endpoint_step": 200000})
    )
    supervise.verify_preflight(tmp_path, args)


@pytest.mark.parametrize("scope", [40, 130])
def test_supervisor_launches_fresh_200k_task8_and_requires_all40_evaluations(tmp_path, monkeypatch, scope):
    from scripts.lpwm_full import evaluate

    args = supervise.parse_args(
        [
            "--root",
            str(tmp_path),
            "--steps",
            "200000",
            "--task-count",
            str(scope),
            "--eval-workers",
            "8",
            "--batch-size",
            "32",
            "--grad-accumulation",
            "1",
        ]
    )
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache/manifest.json").write_text(json.dumps({"num_frames": 100, "episodes": [0, 1]}))
    (tmp_path / "frozen_source_sha256.json").write_text("{}")
    monkeypatch.setattr(supervise, "parse_args", lambda: args)
    monkeypatch.setattr(supervise, "verify_preflight", lambda *a: None)
    monkeypatch.setattr(supervise, "verify_profile", lambda *a: None)
    monkeypatch.setattr(supervise, "verify_cache", lambda *a: None)
    monkeypatch.setattr(supervise.shutil, "disk_usage", lambda _: SimpleNamespace(free=64 * 2**30))
    checked, calls = [], []
    monkeypatch.setattr(evaluate, "validate_result", lambda r: checked.append(r["checkpoint"]["step"]))
    monkeypatch.setattr(evaluate, "protocol_suites", lambda _: evaluate.suites_for_count(scope))

    def popen(command, **kwargs):
        calls.append(command)
        run = tmp_path / "run"
        (run / "eval").mkdir(parents=True)
        (run / "status.json").write_text(json.dumps({"status": "completed", "step": 200000}))
        for step in range(5000, 200001, 5000):
            result = protocol()
            result["checkpoint"] = {"step": step}
            output = run / "eval" / f"step_{step:06d}.json"
            output.write_text(json.dumps(result))
            output.with_suffix(".swanlab.json").write_text(
                json.dumps(
                    {
                        "verified_upload": True,
                        "formal_result_eligible": True,
                        "result_sha256": supervise.digest(output),
                    }
                )
            )
        return SimpleNamespace(pid=1, poll=lambda: 0, returncode=0)

    monkeypatch.setattr(supervise.subprocess, "Popen", popen)
    supervise.main()
    assert len(calls) == 1
    command = calls[0]
    assert not {"--resume", "--disable-eval", "--preflight"}.intersection(command)
    for key, value in (
        ("--steps", "200000"),
        ("--schedule-steps", "200000"),
        ("--eval-workers", "8"),
        ("--batch-size", "32"),
        ("--grad-accumulation", "1"),
    ):
        assert command[command.index(key) + 1] == value
    assert checked == list(range(5000, 200001, 5000))
    status = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert status["status"] == "completed" and status["evaluated_episodes"] == scope * 400
