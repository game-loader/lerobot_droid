import json
import os
from types import SimpleNamespace

from scripts.lpwm_monitor import watch


def make_suite(tmp_path, state="running"):
    suite = tmp_path / "suite"
    run = suite / "runs" / "test"
    (run / "eval").mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"status": state, "step": 1000}))
    (run / "experiment.json").write_text("{}")
    (suite / "queue_state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "runs": {
                    "test": {
                        "started_unix": 100,
                        "spec": {"world_weight": 0.03, "reconstruction_weight": 1, "dynamics_weight": 1},
                    }
                },
            }
        )
    )
    return suite, run


def test_monitor_does_not_modify_training_files(tmp_path, monkeypatch):
    suite, run = make_suite(tmp_path)
    monkeypatch.setattr(watch, "queue_alive", lambda _: True)
    before = {p: p.read_bytes() for p in suite.rglob("*") if p.is_file()}
    snap = watch.collect(suite, None, {})
    assert snap["alerts"] == []
    assert snap["runs"]["test"]["status"]["step"] == 1000
    assert before == {p: p.read_bytes() for p in suite.rglob("*") if p.is_file()}


def test_stale_training_alert_and_eval_grace(tmp_path, monkeypatch):
    suite, run = make_suite(tmp_path)
    monkeypatch.setattr(watch, "queue_alive", lambda _: True)
    os.utime(run / "status.json", (100, 100))
    assert "test:stale_running" in watch.collect(suite, None, {}, now=2001)["alerts"]
    (run / "status.json").write_text(json.dumps({"status": "evaluating", "step": 5000}))
    os.utime(run / "status.json", (100, 100))
    assert watch.collect(suite, None, {}, now=2001)["alerts"] == []


def test_result_waits_for_upload_and_verifies_once(tmp_path, monkeypatch):
    suite, run = make_suite(tmp_path)
    monkeypatch.setattr(watch, "queue_alive", lambda _: True)
    result = run / "eval/step_005000.json"
    result.write_text(json.dumps({"successes": 60, "num_episodes": 100, "success_rate": 0.6, "per_task": []}))
    assert watch.collect(suite, None, {})["evaluations"] == []
    result.with_suffix(".swanlab.json").write_text('{"status":"complete"}')
    calls = []
    gate = SimpleNamespace(verify_evaluation=lambda *a: calls.append(a) or {"ok": True})
    cache = {}
    for _ in range(2):
        snap = watch.collect(suite, gate, cache)
        assert snap["evaluations"][0]["successes"] == 60
    assert len(calls) == 1


def test_queue_missing_raises_alert_not_restart(tmp_path, monkeypatch):
    suite, _ = make_suite(tmp_path)
    monkeypatch.setattr(watch, "queue_alive", lambda _: False)
    assert "queue_process_missing" in watch.collect(suite, None, {})["alerts"]


def test_recovery_only_adopts_known_successful_completed_pair():
    from scripts.lpwm_monitor.recover_queue import KNOWN_ERROR, recovery_plan

    state = {
        "status": "failed",
        "runs": {
            name: {
                "status": "failed",
                "exit_code": 0,
                "error": KNOWN_ERROR + " receipt",
                "spec": {"id": name},
            }
            for name in ("phase1_world_0p03", "phase1_world_0p3")
        },
    }
    verified = []
    result = recovery_plan(
        state, lambda spec: verified.append(spec["id"]) or {"verified": True}, lambda r: False
    )
    assert len(result) == len(verified) == 2
    assert all(r["status"] == "failed" for r in state["runs"].values())


def test_recovery_refuses_other_failure_or_live_training():
    import pytest

    from scripts.lpwm_monitor.recover_queue import KNOWN_ERROR, recovery_plan

    state = {
        "status": "failed",
        "runs": {
            name: {
                "status": "failed",
                "exit_code": 0,
                "error": KNOWN_ERROR + " receipt",
                "spec": {"id": name},
            }
            for name in ("phase1_world_0p03", "phase1_world_0p3")
        },
    }
    with pytest.raises(ValueError, match="still alive"):
        recovery_plan(state, lambda _: None, lambda _: True)
    state["runs"]["phase1_world_0p03"]["error"] = "CUDA OOM"
    with pytest.raises(ValueError, match="Unrelated"):
        recovery_plan(state, lambda _: None, lambda _: False)
    state["runs"]["phase1_world_0p03"]["exit_code"] = 1
    with pytest.raises(ValueError, match="exit successfully"):
        recovery_plan(state, lambda _: None, lambda _: False)
