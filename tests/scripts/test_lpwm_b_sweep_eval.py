"""CPU-only wrapper contract tests: no evaluator, torch, simulator, SDK, or network import.

Run in isolation to avoid repository conftest GPU auto-detection:
    uv run --no-sync pytest --confcutdir=tests/scripts -p no:cacheprovider \
        tests/scripts/test_lpwm_b_sweep_eval.py
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location(
    "lpwm_b_sweep", Path(__file__).parents[2] / "scripts/lpwm_ab/eval_b_sweep.py"
)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)
SECRET = "dummy-test-key-never-use-online"


def result_fixture(task_ids=None, count=10, max_steps=None, namespace="validation", successes=True):
    task_ids = list(range(10)) if task_ids is None else task_ids
    result = {
        "schema_version": 1,
        "status": "complete",
        "checkpoint": {"variant": "B", "step": 5000},
        "protocol": {
            "name": "lpwm-libero-spatial-rollout-v1",
            "suite": "libero_spatial",
            "seed": 42,
            "seed_namespace": namespace,
            "episodes_per_task": count,
            "task_ids": task_ids,
            "init_state_offset": 0,
            "requested_max_steps": max_steps,
            "local_suite_max_steps": 280,
            "max_control_steps": max_steps or 280,
        },
        "per_task": [],
    }

    def summary(rows):
        success = sum(row["success"] for row in rows)
        return {
            "num_episodes": len(rows),
            "successes": success,
            "success_rate": success / len(rows),
            "pc_success": 100 * success / len(rows),
        }

    all_rows = []
    for task_id in task_ids:
        rows = [
            {
                "episode_index": index,
                "success": successes and index % 2 == 0,
                "control_steps": 100 + index,
                "num_chunks": 12,
            }
            for index in range(count)
        ]
        result["per_task"].append({"task_id": task_id, "episodes": rows, **summary(rows)})
        all_rows.extend(rows)
    result.update(summary(all_rows))
    return result


@pytest.fixture
def job(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "experiment.json").write_text(json.dumps({"variant": "B", "step": 5000}))
    credential = tmp_path / "credential"
    credential.write_text(SECRET + "\n")
    output = tmp_path / "result.json"
    state = SimpleNamespace(
        result=result_fixture(),
        output=output,
        checkpoint=checkpoint,
        credential=credential,
        metadata=output.with_suffix(".swanlab.json"),
    )
    state.args = [
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--credential-file",
        str(credential),
        "--project",
        "sweep",
        "--run-name",
        "B-5000",
    ]
    state.retry_args = [
        "--upload-only",
        str(output),
        "--credential-file",
        str(credential),
        "--project",
        "sweep",
        "--run-name",
        "B-5000-retry",
    ]

    def evaluate(args):
        output.write_text(json.dumps(state.result, indent=2) + "\n")
        return state.result

    state.evaluate = Mock(side_effect=evaluate)
    state.loader = Mock(return_value=SimpleNamespace(run_evaluation=state.evaluate))
    monkeypatch.setattr(wrapper, "load_evaluator", state.loader)
    monkeypatch.setattr(wrapper.time, "sleep", Mock())
    sdk = SimpleNamespace(
        login=Mock(return_value=True),
        has_run=Mock(return_value=False),
        log=Mock(return_value=None),
        finish=Mock(return_value=None),
        Settings=Mock(side_effect=lambda **kw: kw),
    )
    state.run = SimpleNamespace(id=None, mode="online", url="https://example.invalid/run")
    state.created_run_ids = []

    def init(**kwargs):
        # Match SwanLab 0.9.4: resume='never' must not receive a caller-owned ID.
        assert kwargs["resume"] == "never"
        assert "id" not in kwargs, "Run id should not be provided when resume=never."
        state.run.id = f"sdk{len(state.created_run_ids) + 1:05d}"
        state.created_run_ids.append(state.run.id)
        state.run.url = f"https://example.invalid/run/{state.run.id}"
        return state.run

    sdk.init = Mock(side_effect=init)

    def remote_metrics(**kwargs):
        logged = sdk.log.call_args.args[0]
        return {
            "list": [
                {"key": key, "metrics": [{"index": 1, "data": logged[key]}]}
                for key in kwargs["keys"]
                if key in logged
            ]
        }

    state.remote = SimpleNamespace(state="FINISHED", metrics=Mock(side_effect=remote_metrics))
    state.project = SimpleNamespace(project_id="project-id", visibility="PRIVATE")
    state.api = SimpleNamespace(
        username="test-user", project=Mock(return_value=state.project), run=Mock(return_value=state.remote)
    )
    sdk.Api = Mock(return_value=state.api)
    state.sdk = sdk
    monkeypatch.setitem(sys.modules, "swanlab", sdk)
    yield state


def read_meta(job):
    return json.loads(job.metadata.read_text())


def test_production_authentication_metrics_and_original_result(job):
    assert wrapper.main(job.args) == 0
    job.sdk.login.assert_called_once_with(api_key=SECRET, save=False, relogin=True)
    init = job.sdk.init.call_args.kwargs
    assert init["mode"] == "online" and init["public"] is False
    assert init["resume"] == "never"
    assert "id" not in init
    assert init["settings"]["terminal"] == {"proxy_type": "none"}
    assert all(value is False for value in init["settings"]["probe"].values())
    assert init["config"]["protocol"]["episodes_per_task"] == 10
    assert init["config"]["checkpoint_step"] == 5000
    assert init["config"]["formal_result_eligible"] is True
    job.evaluate.assert_called_once()
    args = job.evaluate.call_args.args[0]
    assert args.device == "cuda" and args.seed == 42 and args.seed_namespace == "validation"
    assert args.video is False and args.max_steps is None and args.init_state_offset == 0
    metrics = job.sdk.log.call_args.args[0]
    assert job.sdk.log.call_args.kwargs == {"step": 1}
    assert metrics["eval/num_episodes"] == 100
    assert metrics["eval/successes"] == 50 and metrics["eval/success_rate"] == 0.5
    for task in range(10):
        assert metrics[f"eval/task_{task:02d}/num_episodes"] == 10
        assert metrics[f"eval/task_{task:02d}/successes"] == 5
    job.sdk.finish.assert_called_once_with()
    metadata = read_meta(job)
    assert metadata["uploaded"] is True and metadata["status"] == "complete"
    assert metadata["verified_upload"] is True
    assert job.metadata.name == "result.swanlab.json"
    assert metadata["id"] == job.run.id == job.created_run_ids[0]
    assert metadata["id"] != metadata["attempt_id"][:8]
    assert metadata["url"] == job.run.url and metadata["result_sha256"]
    job.api.run.assert_called_once_with(f"test-user/sweep/{job.run.id}")
    assert json.loads(job.output.read_text()) == job.result
    assert SECRET not in job.metadata.read_text() and str(job.credential) not in job.metadata.read_text()
    assert SECRET not in repr(init) and str(job.credential) not in repr(init)


@pytest.mark.parametrize("gpu", ["0", "1"])
def test_preflight_handshake_preserves_physical_gpu(job, monkeypatch, gpu):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu)
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", gpu)
    monkeypatch.setenv("MUJOCO_GL", "existing-gl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "existing-platform")
    job.result = result_fixture([0], count=1, max_steps=8)
    assert (
        wrapper.main(
            job.args
            + ["--preflight", "--episodes-per-task", "1", "--task-ids", "0", "--max-steps", "8", "--video"]
        )
        == 0
    )
    args = job.evaluate.call_args.args[0]
    assert (args.episodes_per_task, args.task_ids, args.max_steps, args.video) == (1, [0], 8, True)
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == gpu
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == gpu
    assert os.environ["MUJOCO_GL"] == "existing-gl"
    assert os.environ["PYOPENGL_PLATFORM"] == "existing-platform"
    assert read_meta(job)["formal_result_eligible"] is False
    assert read_meta(job)["verified_upload"] is True
    assert read_meta(job)["protocol_role"] == "preflight"
    assert job.sdk.log.call_args.args[0]["eval/preflight"] == 1
    assert job.sdk.log.call_args.args[0]["eval/formal_result_eligible"] == 0
    assert job.sdk.init.call_args.kwargs["config"]["preflight"] is True


def test_egl_defaults_only_when_unset(job, monkeypatch):
    import os

    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    assert wrapper.main(job.args) == 0
    assert os.environ["MUJOCO_GL"] == os.environ["PYOPENGL_PLATFORM"] == "egl"
    assert "MUJOCO_EGL_DEVICE_ID" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize(
    "extra",
    [
        ["--episodes-per-task", "1"],
        ["--task-ids", "0"],
        ["--max-steps", "8"],
        ["--init-state-offset", "1"],
        ["--episodes-per-task", "11"],
        ["--task-ids", "0", "0"],
        ["--task-ids", "10"],
        ["--preflight", "--episodes-per-task", "0"],
        ["--preflight", "--max-steps", "0"],
    ],
)
def test_invalid_cli_does_not_allocate_or_authenticate(job, extra):
    with pytest.raises(SystemExit) as caught:
        wrapper.main(job.args + extra)
    assert caught.value.code == 2
    job.sdk.login.assert_not_called()
    job.loader.assert_not_called()


def test_100_completed_failures_is_valid_not_100_successes(job):
    job.result = result_fixture(successes=False)
    assert wrapper.main(job.args) == 0
    assert job.sdk.log.call_args.args[0]["eval/success_rate"] == 0


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda r: r.update(num_episodes=1200),
        lambda r: r.update(num_episodes=100.0),
        lambda r: r.update(successes=100),
        lambda r: r.update(success_rate=0.9),
        lambda r: r.update(pc_success=float("nan")),
        lambda r: r.update(status="partial"),
        lambda r: r["checkpoint"].update(variant="A"),
        lambda r: r["per_task"].pop(),
        lambda r: r["per_task"][0]["episodes"].pop(),
        lambda r: r["per_task"][0].update(num_episodes=120),
        lambda r: r["per_task"][0].update(success_rate=0.9),
        lambda r: r["per_task"][0]["episodes"][0].update(success="true"),
        lambda r: r["per_task"][0]["episodes"][0].update(episode_index=1),
        lambda r: r["per_task"][0]["episodes"][0].update(episode_index=False),
        lambda r: r["per_task"][0]["episodes"][0].update(control_steps=0),
        lambda r: r["protocol"].update(requested_max_steps=8),
        lambda r: r["protocol"].update(max_control_steps=8),
        lambda r: r["protocol"].update(init_state_offset=1),
    ],
)
def test_invalid_episode_results_never_uploaded(corrupt):
    result = result_fixture()
    corrupt(result)
    with pytest.raises((ValueError, KeyError)):
        wrapper.validate_result(result)


def test_partial_requires_preflight_even_in_upload_only(job):
    job.result = result_fixture([0], 1, 8)
    job.output.write_text(json.dumps(job.result))
    original = job.output.read_bytes()
    assert wrapper.main(job.retry_args) == 1
    job.sdk.login.assert_not_called()
    assert wrapper.main(job.retry_args + ["--preflight"]) == 0
    job.loader.assert_not_called()
    assert job.output.read_bytes() == original


def test_full_preflight_cannot_be_reclassified_on_retry(job):
    assert wrapper.main(job.args + ["--preflight"]) == 0
    original_metadata = job.metadata.read_bytes()
    assert wrapper.main(job.retry_args) == 1
    assert wrapper.main(job.retry_args) == 1
    assert job.metadata.read_bytes() == original_metadata
    assert wrapper.main(job.retry_args + ["--preflight"]) == 0


@pytest.mark.parametrize(
    "stage",
    [
        "authentication",
        "initialization",
        "start_logging",
        "rollout",
        "result_logging",
        "finish",
        "verification",
    ],
)
def test_failures_exit_nonzero_keep_result_and_do_not_leak_secrets(job, capsys, stage):
    error = RuntimeError(f"provider failure includes {SECRET}")
    if stage == "authentication":
        job.sdk.login.side_effect = error
    elif stage == "initialization":
        job.sdk.init.side_effect = error
    elif stage == "start_logging":
        job.sdk.log.side_effect = error
    elif stage == "rollout":
        job.evaluate.side_effect = error
    elif stage == "result_logging":
        job.sdk.log.side_effect = [None, error]
    elif stage == "finish":
        job.sdk.finish.side_effect = error
    else:
        job.remote.metrics.side_effect = error
    assert wrapper.main(job.args) == 1
    metadata = read_meta(job)
    assert metadata["status"] == "failed" and metadata["uploaded"] is False
    assert metadata["verified_upload"] is False
    assert metadata["failure_stage"] == stage
    if stage in ("result_logging", "finish", "verification"):
        assert json.loads(job.output.read_text()) == job.result
        assert metadata["evaluation_status"] == "complete"
    if stage in ("start_logging", "rollout", "result_logging"):
        assert job.sdk.finish.call_args.kwargs["state"] == "crashed"
    if stage == "authentication":
        job.evaluate.assert_not_called()
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err + job.metadata.read_text()


@pytest.mark.parametrize("failure", ["login", "log", "finish", "offline", "public", "invisible", "active"])
def test_nonraising_sdk_failures_are_rejected(job, failure):
    if failure == "login":
        job.sdk.login.return_value = False
    elif failure == "log":
        job.sdk.log.return_value = False
    elif failure == "finish":
        job.sdk.finish.return_value = False
    elif failure == "offline":
        job.run.mode = "offline"
    elif failure == "public":
        job.project.visibility = "PUBLIC"
    elif failure == "invisible":
        job.project.project_id = ""
    else:
        job.sdk.has_run.return_value = True
    assert wrapper.main(job.args) == 1
    assert read_meta(job)["uploaded"] is False
    assert read_meta(job)["verified_upload"] is False
    if failure != "finish":
        job.evaluate.assert_not_called()


def test_cleanup_failure_does_not_hide_rollout_failure(job):
    job.evaluate.side_effect = RuntimeError(SECRET)
    job.sdk.finish.side_effect = RuntimeError(SECRET)
    assert wrapper.main(job.args) == 1
    assert read_meta(job)["failure_stage"] == "rollout"
    assert read_meta(job)["cleanup_failed"] is True


def test_evaluator_writes_then_raises_keep_original(job):
    def fail_after_write(args):
        job.output.write_text(json.dumps(job.result))
        raise RuntimeError("rollout failure")

    job.evaluate.side_effect = fail_after_write
    assert wrapper.main(job.args) == 1
    assert json.loads(job.output.read_text()) == job.result


def test_upload_retry_no_rerollout_no_result_overwrite_own_run(job):
    job.sdk.log.side_effect = [None, RuntimeError(SECRET)]
    assert wrapper.main(job.args) == 1
    first_id = read_meta(job)["id"]
    original = job.output.read_bytes()
    job.sdk.log.side_effect = None
    assert wrapper.main(job.retry_args) == 0
    assert job.output.read_bytes() == original
    job.evaluate.assert_called_once()
    assert read_meta(job)["id"] != first_id
    assert job.created_run_ids == [first_id, read_meta(job)["id"]]
    assert read_meta(job)["id"] == job.run.id
    assert read_meta(job)["url"] == job.run.url
    for call in job.sdk.init.call_args_list:
        assert call.kwargs["resume"] == "never" and "id" not in call.kwargs
    assert read_meta(job)["upload_only"] is True


@pytest.mark.parametrize("missing_id", [None, ""])
def test_missing_sdk_returned_id_fails_before_rollout(job, missing_id):
    normal_init = job.sdk.init.side_effect

    def init_without_id(**kwargs):
        run = normal_init(**kwargs)
        run.id = missing_id
        return run

    job.sdk.init.side_effect = init_without_id
    assert wrapper.main(job.args) == 1
    assert read_meta(job)["failure_stage"] == "initialization"
    assert read_meta(job)["verified_upload"] is False
    job.evaluate.assert_not_called()
    job.api.run.assert_not_called()


def test_existing_result_never_rerolled_or_overwritten(job):
    job.output.write_text("original bytes")
    job.metadata.write_text("original metadata")
    assert wrapper.main(job.args) == 1
    assert job.output.read_text() == "original bytes"
    assert job.metadata.read_text() == "original metadata"
    job.sdk.login.assert_not_called()
    job.loader.assert_not_called()


@pytest.mark.parametrize("remote_failure", ["missing", "wrong_value", "wrong_step", "unfinished"])
def test_finish_returning_normally_is_not_proof_of_upload(job, remote_failure):
    if remote_failure == "unfinished":
        job.remote.state = "RUNNING"
    else:
        normal = job.remote.metrics.side_effect

        def broken_metrics(**kwargs):
            payload = normal(**kwargs)
            if remote_failure == "missing":
                payload["list"].pop()
            else:
                point = payload["list"][0]["metrics"][0]
                point["data" if remote_failure == "wrong_value" else "index"] = -10
            return payload

        job.remote.metrics.side_effect = broken_metrics
    assert wrapper.main(job.args) == 1
    assert read_meta(job)["failure_stage"] == "verification"
    assert read_meta(job)["uploaded"] is False
    assert read_meta(job)["verified_upload"] is False


def test_readback_allows_eventual_consistency(job):
    normal = job.remote.metrics.side_effect
    responses = iter([{"list": []}, None])

    def eventually_visible(**kwargs):
        return next(responses) or normal(**kwargs)

    job.remote.metrics.side_effect = eventually_visible
    assert wrapper.main(job.args) == 0
    assert job.remote.metrics.call_count == 2


def test_verified_upload_only_set_after_server_readback(job, capsys):
    normal = job.remote.metrics.side_effect

    def inspect_pending_metadata(**kwargs):
        metadata = read_meta(job)
        assert metadata["status"] == "running"
        assert metadata["evaluation_status"] == "complete"
        assert metadata["uploaded"] is False
        assert metadata["verified_upload"] is False
        return normal(**kwargs)

    job.remote.metrics.side_effect = inspect_pending_metadata
    assert wrapper.main(job.args) == 0
    assert read_meta(job)["verified_upload"] is True
    summary = json.loads(capsys.readouterr().out)
    assert summary["verified_upload"] is True
    assert summary["metadata"] == str(job.metadata)


def test_upload_only_uses_original_seed_namespace(job):
    job.result = result_fixture(namespace="final")
    job.output.write_text(json.dumps(job.result))
    assert wrapper.main(job.retry_args) == 0
    assert read_meta(job)["protocol_role"] == "final"
    assert job.sdk.init.call_args.kwargs["config"]["protocol"]["seed_namespace"] == "final"
    job.loader.assert_not_called()


def test_mismatched_requested_protocol_keeps_result_but_fails(job):
    job.result["protocol"]["seed"] = 43
    assert wrapper.main(job.args) == 1
    assert read_meta(job)["failure_stage"] == "result_validation"
    assert json.loads(job.output.read_text()) == job.result


def test_wrong_checkpoint_variant_fails_before_authentication(job):
    (job.checkpoint / "experiment.json").write_text('{"variant": "A"}')
    assert wrapper.main(job.args) == 1
    job.sdk.login.assert_not_called()
    job.loader.assert_not_called()


def test_external_credential_and_path_collision_guards(job):
    with pytest.raises(SystemExit):
        wrapper.main(job.args + ["--credential-file", str(wrapper.REPO_ROOT / "do-not-read")])
    with pytest.raises(SystemExit):
        wrapper.main(job.args + ["--output", str(job.credential)])
    job.metadata.symlink_to(job.credential)
    assert wrapper.main(job.args) == 1
    assert job.credential.read_text() == SECRET + "\n"
    job.loader.assert_not_called()


def test_upload_only_missing_file_fails_without_evaluator(job):
    assert wrapper.main(job.retry_args) == 1
    job.loader.assert_not_called()
    job.sdk.login.assert_not_called()


def test_result_validation_does_not_mutate_original():
    result = result_fixture()
    original = copy.deepcopy(result)
    wrapper.validate_result(result)
    assert result == original
