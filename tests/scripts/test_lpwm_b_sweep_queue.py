"""CPU-only orchestration tests: fake trainers, real immutable fixture artifacts, no GPU/SSH."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "lpwm_b_sweep_queue", Path(__file__).parents[2] / "scripts/lpwm_ab/run_b_sweep.py"
)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)

INIT_HASH = "a" * 64
SPLIT_HASH = "b" * 64


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def complete_run(path, run, args=None, successes=None, initial_hash=INIT_HASH, split_hash=SPLIT_HASH):
    """Emit the production trainer/evaluator schema without importing Torch."""
    path.mkdir(parents=True, exist_ok=False)
    experiment = {
        **queue.expected_settings(run),
        "preflight": False,
        "disable_eval": False,
        "project": args.project if args else "lpwm-fm-b-world-balance",
        "initial_weights_sha256": initial_hash,
        "split_sha256": split_hash,
    }
    # Trainer argparse stores these aliases rather than the CLI's long spelling.
    experiment["rec_weight"] = experiment.pop("reconstruction_weight")
    experiment["dyn_weight"] = experiment.pop("dynamics_weight")
    if args is not None:
        experiment.update(
            data=str(args.data), dataset_manifest_sha256=queue.file_hash(args.data / "manifest.json")
        )
    dump(path / "experiment.json", experiment)
    dump(path / "status.json", {"status": "completed", "step": 30_000})
    successes = successes or [40] * 6
    for step, count in zip(queue.STEPS, successes, strict=True):
        checkpoint = path / "checkpoints" / f"step_{step:06d}"
        checkpoint.mkdir(parents=True)
        metadata = {
            "step": step,
            "seed": 42,
            "variant": "B",
            "initial_weights_sha256": initial_hash,
            "split_sha256": split_hash,
        }
        for name in (*queue.ARTIFACTS, "optimizer.pt", "split.json"):
            (checkpoint / name).write_bytes(f"fixture {name} {step}".encode())
        dump(checkpoint / "experiment.json", metadata)
        all_hashes = {p.name: queue.file_hash(p) for p in checkpoint.iterdir()}
        dump(
            checkpoint / "complete.json",
            {
                "status": "complete",
                "step": step,
                "preflight": False,
                "initial_weights_sha256": initial_hash,
                "split_sha256": split_hash,
                "sha256": all_hashes,
            },
        )
        hashes = {name: all_hashes[name] for name in queue.ARTIFACTS}
        protocol = {
            "suite": "libero_spatial",
            "seed": 42,
            "seed_namespace": "validation",
            "task_ids": list(range(10)),
            "episodes_per_task": 10,
            "name": "lpwm-libero-spatial-rollout-v1",
            "init_state_offset": 0,
            "requested_max_steps": None,
            "max_control_steps": 280,
            "local_suite_max_steps": 280,
        }
        tasks = []
        for task_id in range(10):
            episodes = [
                {
                    "episode_index": i,
                    "init_state_index": i,
                    "seed": task_id * 10 + i,
                    "success": task_id * 10 + i < count,
                }
                for i in range(10)
            ]
            task_count = sum(e["success"] for e in episodes)
            tasks.append(
                {
                    "task_id": task_id,
                    "episodes": episodes,
                    "num_episodes": 10,
                    "successes": task_count,
                    "success_rate": task_count / 10,
                }
            )
        result = {
            "status": "complete",
            "num_episodes": 100,
            "successes": count,
            "success_rate": count / 100,
            "pc_success": count,
            "protocol": protocol,
            "protocol_sha256": queue.json_hash(protocol),
            "per_task": tasks,
            "checkpoint": {
                "path": str(checkpoint),
                "step": step,
                "variant": "B",
                "training_seed": 42,
                "split_sha256": split_hash,
                "artifact_sha256": hashes,
                "model_sha256": hashes["model.safetensors"],
                "bundle_sha256": queue.json_hash(hashes),
            },
        }
        result_path = path / "eval" / f"step_{step:06d}.json"
        dump(result_path, result)
        dump(
            result_path.with_suffix(".swanlab.json"),
            {
                "status": "complete",
                "uploaded": True,
                "verified_upload": True,
                "evaluation_status": "complete",
                "preflight": False,
                "formal_result_eligible": True,
                "protocol_role": "validation",
                "mode": "online",
                "public": False,
                "project": experiment["project"],
                "result": str(result_path),
                "result_sha256": queue.file_hash(result_path),
                "attempt_id": "c" * 32,
                "id": "c" * 8,
            },
        )
    return path


@pytest.fixture
def args(tmp_path):
    repo = tmp_path / "snapshot"
    (repo / "scripts/lpwm_ab").mkdir(parents=True)
    (repo / "src").mkdir()
    for name in ("train_b_sweep.py", "eval_b_sweep.py"):
        (repo / "scripts/lpwm_ab" / name).write_text("# trainer/evaluator stand-in; never executed\n")
    data = tmp_path / "data"
    dump(data / "manifest.json", {"tasks": {str(i): f"task{i}" for i in range(10)}})
    credential = tmp_path / "credential"
    credential.write_text("test-secret-not-for-queue-state")
    return queue.parse_args(
        [
            "--repo",
            str(repo),
            "--python",
            sys.executable,
            "--eval-python",
            sys.executable,
            "--root",
            str(tmp_path / "outputnew"),
            "--data",
            str(data),
            "--credential-file",
            str(credential),
            "--poll-seconds",
            "0.001",
        ]
    )


class FakeWorkers:
    """Every worker stays running until both of its round have been launched."""

    def __init__(self, monkeypatch, args, *, fail=None, missing=None, wrong_init=None, launch_fail=None):
        self.args = args
        self.fail, self.missing, self.wrong_init, self.launch_fail = fail, missing, wrong_init, launch_fail
        self.events, self.commands, self.environments, self.processes = [], [], [], []
        monkeypatch.setattr(queue.subprocess, "Popen", self.launch)
        monkeypatch.setattr(queue, "group_alive", lambda record: False)
        monkeypatch.setattr(queue.shutil, "disk_usage", lambda root: SimpleNamespace(free=100 * queue.GIB))

    def launch(self, command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        run_id = output.name
        if run_id == self.launch_fail:
            raise OSError("simulated spawn failure")
        phase1 = {r["id"]: r for pair in queue.phase1_rounds() for r in pair}
        world = float(command[command.index("--world-weight") + 1])
        phase2 = {r["id"]: r for r in queue.phase2_round(world)}
        run = {**phase1, **phase2}[run_id]
        assert not output.exists()
        assert kwargs["start_new_session"] is True and kwargs["pass_fds"]
        assert kwargs["stderr"] is subprocess.STDOUT
        self.events.append(("start", run_id))
        self.commands.append(command)
        self.environments.append(kwargs["env"])
        process = SimpleNamespace(
            pid=123456 + len(self.processes), finished=False, polled=False, waited=False
        )

        def poll():
            if not process.polled:
                process.polled = True
                return None
            if not process.finished:
                process.finished = True
                if run_id != self.fail:
                    complete_run(
                        output,
                        run,
                        self.args,
                        initial_hash="c" * 64 if run_id == self.wrong_init else INIT_HASH,
                    )
                    if run_id == self.missing:
                        (output / "eval/step_005000.json").unlink()
                self.events.append(("finish", run_id))
            return 9 if run_id == self.fail else 0

        def wait():
            process.waited = True
            return 9 if run_id == self.fail else 0

        process.poll, process.wait = poll, wait
        self.processes.append(process)
        return process


def execute(args):
    with queue.queue_lock(args.root, args.resume) as lock:
        return queue.SweepQueue(args, lock).execute()


def verified_candidates(tmp_path, counts):
    runs = [run for pair in queue.phase1_rounds() for run in pair]
    return [
        queue.verify_run(complete_run(tmp_path / run["id"], run, successes=values), run)
        for run, values in zip(runs, counts, strict=True)
    ]


def test_selects_late_mean_not_best_single_checkpoint(tmp_path):
    candidates = verified_candidates(
        tmp_path,
        [
            [100, 20, 20, 20, 20, 20],
            [40, 40, 40, 60, 60, 60],
            [30, 30, 30, 50, 50, 50],
            [99, 80, 80, 40, 40, 40],
        ],
    )
    result = queue.select_phase1(candidates)
    assert result["world_weight"] == 0.3
    assert result["ranking"][0]["score"] == 0.6
    assert result["best_checkpoint_is_descriptive_only"] is True
    descriptive = next(row for row in result["ranking"] if row["world_weight"] == 0.03)
    assert descriptive["best_checkpoint"]["step"] == 5000
    assert descriptive["best_checkpoint"]["success_rate"] == 1
    assert result["control"] == result["winner"]


def test_tie_breaks_by_final_then_smaller_world(tmp_path):
    candidates = verified_candidates(
        tmp_path,
        [
            [1, 1, 1, 60, 60, 60],
            [1, 1, 1, 50, 50, 80],
            [1, 1, 1, 50, 50, 80],
            [1, 1, 1, 50, 50, 80],
        ],
    )
    result = queue.select_phase1(candidates)
    assert [row["world_weight"] for row in result["ranking"]] == [0.1, 0.3, 1, 0.03]


def test_selection_requires_all_four_runs(tmp_path):
    candidates = verified_candidates(tmp_path, [[40] * 6] * 4)
    with pytest.raises(queue.QueueError, match="ALL FOUR"):
        queue.select_phase1(candidates[:3])
    with pytest.raises(queue.QueueError, match="ALL FOUR"):
        queue.select_phase1([candidates[0]] * 4)


@pytest.mark.parametrize(
    "field", ["initial_weights_sha256", "split_sha256", "protocol_sha256", "episode_plan_sha256"]
)
def test_selection_rejects_unpaired_runs(tmp_path, field):
    candidates = verified_candidates(tmp_path, [[40] * 6] * 4)
    candidates[3][field] = "e" * 64
    with pytest.raises(queue.QueueError, match="paired"):
        queue.select_phase1(candidates)


def test_phase_order_gpu_assignment_and_control(args, monkeypatch):
    monkeypatch.setenv("LIBERO_CONFIG_PATH", "/inherited/libero")
    workers = FakeWorkers(monkeypatch, args)
    state = execute(args)
    ids = [r["id"] for pair in queue.phase1_rounds() for r in pair] + [
        r["id"] for r in queue.phase2_round(0.03)
    ]
    assert workers.events == [
        (kind, run_id)
        for start in range(0, 6, 2)
        for kind in ("start", "finish")
        for run_id in ids[start : start + 2]
    ]
    assert state["status"] == "completed" and len(state["runs"]) == 6
    assert all(p.waited for p in workers.processes)
    assert state["selection"]["world_weight"] == 0.03
    assert state["selection"]["control"] == ids[0]
    comparison = queue.read_json(args.root / "phase2_comparison.json")
    assert comparison["control"] == ids[0] and len(comparison["runs"]) == 3
    for i, (command, env) in enumerate(zip(workers.commands, workers.environments, strict=True)):
        assert env["CUDA_VISIBLE_DEVICES"] == env["MUJOCO_EGL_DEVICE_ID"] == str(i % 2)
        assert env["MUJOCO_GL"] == env["PYOPENGL_PLATFORM"] == "egl"
        assert env["PYTHONPATH"] == str(args.repo / "src")
        assert env["LIBERO_CONFIG_PATH"] == "/inherited/libero"
        assert command[command.index("--steps") + 1] == "30000"
        assert command[command.index("--seed") + 1] == "42"
        assert command[command.index("--workers") + 1] == "4"
        assert "--project" in command and "--swanlab-project" not in command
        assert not any(flag in command for flag in ("--resume", "--preflight", "--disable-eval"))
        if i >= 2:
            assert command[command.index("--expected-init-sha256") + 1] == INIT_HASH
            assert command[command.index("--expected-split-sha256") + 1] == SPLIT_HASH
        if i >= 4:
            assert float(command[command.index("--world-weight") + 1]) == 0.03
            rec = float(command[command.index("--reconstruction-weight") + 1])
            dyn = float(command[command.index("--dynamics-weight") + 1])
            assert (rec, dyn) == [(0.5, 1.5), (1.5, 0.5)][i - 4] and rec + dyn == 2
    text = (args.root / "queue_state.json").read_text()
    assert args.credential_file.read_text() not in text
    for record in state["runs"].values():
        assert record["started_unix"] <= record["finished_unix"]
        assert record["exit_code"] == 0


@pytest.mark.parametrize("failure", ["fail", "missing", "wrong_init", "launch_fail"])
def test_failure_drains_round_and_stops_all_later_launches(args, monkeypatch, failure):
    first, second = queue.phase1_rounds()[0]
    target = second["id"] if failure == "launch_fail" else first["id"]
    workers = FakeWorkers(monkeypatch, args, **{failure: target})
    with pytest.raises(queue.QueueError):
        execute(args)
    state = queue.read_json(args.root / "queue_state.json")
    assert state["status"] == "failed"
    assert len(workers.commands) == (1 if failure == "launch_fail" else 2)
    assert all(p.waited for p in workers.processes)
    assert not (args.root / "phase1_selection.json").exists()
    assert all(
        record["spec"]["phase"] == 1 and record["spec"]["round"] == 1 for record in state["runs"].values()
    )
    if failure == "fail":
        assert state["runs"][second["id"]]["status"] == "completed"
        args.resume = True
        with pytest.raises(queue.QueueError, match="terminal"):
            execute(args)
        assert len(workers.commands) == 2


def test_second_round_failure_never_selects_or_launches_phase2(args, monkeypatch):
    workers = FakeWorkers(monkeypatch, args, missing=queue.phase1_rounds()[1][1]["id"])
    with pytest.raises(queue.QueueError, match="Round failed"):
        execute(args)
    assert len(workers.commands) == 4 and all(p.waited for p in workers.processes)
    assert not (args.root / "phase1_selection.json").exists()


def test_completed_queue_resume_reverifies_without_any_launch(args, monkeypatch):
    workers = FakeWorkers(monkeypatch, args)
    first_state = execute(args)
    args.resume = True
    second_state = execute(args)
    assert len(workers.commands) == 6
    assert second_state["selection"] == first_state["selection"]
    assert second_state["status"] == "completed"
    run_id = queue.phase1_rounds()[0][0]["id"]
    (args.root / "runs" / run_id / "eval/step_005000.json").unlink()
    with pytest.raises(queue.QueueError, match="Missing/unreadable"):
        execute(args)
    assert len(workers.commands) == 6


def test_resume_after_completed_round_does_not_retrain_it(args, monkeypatch):
    workers = FakeWorkers(monkeypatch, args)
    with queue.queue_lock(args.root, False) as lock:
        controller = queue.SweepQueue(args, lock)
        controller.run_round(queue.phase1_rounds()[0], future_runs=6)
    assert len(workers.commands) == 2
    args.resume = True
    state = execute(args)
    assert state["status"] == "completed" and len(workers.commands) == 6


def test_preexisting_even_empty_run_directory_is_never_reused(args, monkeypatch):
    workers = FakeWorkers(monkeypatch, args)
    with queue.queue_lock(args.root, False) as lock:
        controller = queue.SweepQueue(args, lock)
        first = queue.phase1_rounds()[0][0]
        (args.root / "runs" / first["id"]).mkdir()
        with pytest.raises(queue.QueueError, match="preexisting"):
            controller.execute()
    assert not workers.commands


def test_existing_suite_root_with_repo_and_metadata_is_allowed(args, monkeypatch):
    workers = FakeWorkers(monkeypatch, args)
    args.root.mkdir()
    (args.root / "repo").mkdir()
    (args.root / "SUITE_ROOT").write_text(str(args.root))
    state = execute(args)
    assert state["status"] == "completed" and len(workers.commands) == 6
    assert (args.root / "SUITE_ROOT").read_text() == str(args.root)
    with pytest.raises(queue.QueueError, match="already exists"), queue.queue_lock(args.root, False):
        pytest.fail("existing queue cannot be overwritten")


@pytest.mark.parametrize("name", ["runs", "logs"])
def test_existing_run_outputs_fail_before_queue_creation(args, name):
    (args.root / name / "existing").mkdir(parents=True)
    with pytest.raises(queue.QueueError, match="Conflicting"), queue.queue_lock(args.root, False):
        pytest.fail("existing outputs cannot be adopted by a fresh queue")
    assert not (args.root / "queue_state.json").exists()


def test_queue_lock_is_exclusive_and_inherited_by_own_child(args, tmp_path):
    # A tiny CPU process proves lock survival if the queue goes away before its
    # trainer; no CUDA environment, model imports, process signalling, or sleeps.
    script = "import os,sys; os.write(int(sys.argv[1]), b'R'); sys.stdin.buffer.read(1)"
    ready_r, ready_w = os.pipe()
    child = None
    try:
        with queue.queue_lock(args.root, False) as lock:
            with pytest.raises(queue.QueueError, match="lock is held"), queue.queue_lock(args.root, True):
                pytest.fail("lock acquired twice")
            child = subprocess.Popen(
                [sys.executable, "-c", script, str(ready_w)], stdin=subprocess.PIPE, pass_fds=(lock, ready_w)
            )
            assert os.read(ready_r, 1) == b"R"
        with pytest.raises(queue.QueueError, match="lock is held"), queue.queue_lock(args.root, True):
            pytest.fail("orphan must retain lock")
    finally:
        os.close(ready_r)
        os.close(ready_w)
        if child is not None:
            child.communicate(b"x", timeout=10)
    with queue.queue_lock(args.root, True):
        pass


def prepare_stale(args, *, complete=True, status="running"):
    run = queue.phase1_rounds()[0][0]
    with queue.queue_lock(args.root, False) as lock:
        controller = queue.SweepQueue(args, lock)
        if complete:
            complete_run(args.root / "runs" / run["id"], run, args)
        controller.state["runs"][run["id"]] = {
            "spec": run,
            "status": status,
            "pid": 98765,
            "pgid": 98765,
            "hostname": queue.socket.gethostname(),
            "boot_id": queue.boot_id(),
        }
        controller.save()
    args.resume = True
    return run


def test_stale_requires_explicit_recovery_even_if_outputs_look_complete(args, monkeypatch):
    workers = FakeWorkers(monkeypatch, args)
    run = prepare_stale(args)
    with pytest.raises(queue.QueueError, match="running/stale"):
        execute(args)
    assert not workers.commands
    args.recover_stale = [run["id"]]
    state = execute(args)
    assert len(workers.commands) == 5
    assert state["runs"][run["id"]]["recovered"] is True
    assert not any(command[command.index("--output") + 1].endswith(run["id"]) for command in workers.commands)


def test_stale_live_group_cannot_be_adopted(args, monkeypatch):
    run = prepare_stale(args)
    args.recover_stale = [run["id"]]
    workers = FakeWorkers(monkeypatch, args)
    monkeypatch.setattr(queue, "group_alive", lambda record: True)
    with pytest.raises(queue.QueueError, match="still alive"):
        execute(args)
    assert not workers.commands


def test_stale_incomplete_cannot_be_restarted_or_adopted(args, monkeypatch):
    run = prepare_stale(args, complete=False, status="starting")
    args.recover_stale = [run["id"]]
    workers = FakeWorkers(monkeypatch, args)
    with pytest.raises(queue.QueueError, match="Missing/unreadable"):
        execute(args)
    assert not workers.commands


def test_source_changes_block_resume_before_launch(args, monkeypatch):
    workers = FakeWorkers(monkeypatch, args)
    with queue.queue_lock(args.root, False) as lock:
        queue.SweepQueue(args, lock)
    args.resume = True
    (args.repo / "src/changed.py").write_text("# changed snapshot")
    with pytest.raises(queue.QueueError, match="configuration/source/data changed"):
        execute(args)
    assert not workers.commands


def test_low_disk_blocks_launch_and_future_budget_is_enforced(args, monkeypatch):
    workers = FakeWorkers(monkeypatch, args)
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda path: SimpleNamespace(free=2 * queue.GIB))
    with pytest.raises(queue.QueueError, match="Disk guard"):
        execute(args)
    assert not workers.commands
    args.estimated_run_gib = 1.3
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda path: SimpleNamespace(free=14 * queue.GIB))
    guard = queue.disk_guard(args, future_runs=6, round_runs=2)
    assert guard["required_bytes"] == pytest.approx(10.8 * queue.GIB, abs=1)
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 * queue.GIB))
    with pytest.raises(queue.QueueError, match="Disk guard"):
        queue.disk_guard(args, future_runs=6, round_runs=2)


def test_disk_rechecked_before_second_worker_and_first_is_drained(args, monkeypatch):
    workers = FakeWorkers(monkeypatch, args)
    space = iter([20 * queue.GIB, 2 * queue.GIB])
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda path: SimpleNamespace(free=next(space)))
    with pytest.raises(queue.QueueError, match="Disk guard"):
        execute(args)
    assert len(workers.commands) == 1 and workers.processes[0].waited


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_eval", "Missing/unreadable"),
        ("partial_train", "30000"),
        ("partial_eval", "not complete"),
        ("wrong_denominator", "100 complete"),
        ("wrong_rate", "success_rate"),
        ("missing_episodes", "Partial task"),
        ("duplicate_episode", "duplicate episode"),
        ("duplicate_task", "duplicate task"),
        ("wrong_checkpoint", "checkpoint identity"),
        ("changed_model", "artifact changed"),
        ("missing_manifest", "completion manifest"),
        ("changed_optimizer", "artifact changed"),
        ("wrong_protocol", "protocol"),
        ("preflight", "preflight"),
    ],
)
def test_output_verifier_fails_closed(tmp_path, mutation, match):
    run = queue.phase1_rounds()[0][0]
    path = complete_run(tmp_path / run["id"], run)
    result_path = path / "eval/step_005000.json"
    result = queue.read_json(result_path)
    if mutation == "missing_eval":
        result_path.unlink()
    elif mutation == "partial_train":
        dump(path / "status.json", {"status": "completed", "step": 25_000})
    elif mutation == "partial_eval":
        result["status"] = "running"
    elif mutation == "wrong_denominator":
        result["num_episodes"] = 99
    elif mutation == "wrong_rate":
        result["success_rate"] = 0.9
    elif mutation == "missing_episodes":
        result["per_task"][0]["episodes"].pop()
    elif mutation == "duplicate_episode":
        result["per_task"][0]["episodes"][1] = copy.deepcopy(result["per_task"][0]["episodes"][0])
    elif mutation == "duplicate_task":
        result["per_task"][1]["task_id"] = 0
    elif mutation == "wrong_checkpoint":
        result["checkpoint"]["step"] = 10_000
    elif mutation in {"changed_model", "changed_optimizer"}:
        name = "model.safetensors" if mutation == "changed_model" else "optimizer.pt"
        (path / "checkpoints/step_005000" / name).write_text("corrupted")
    elif mutation == "missing_manifest":
        (path / "checkpoints/step_005000/complete.json").unlink()
    elif mutation == "wrong_protocol":
        result["protocol"]["seed_namespace"] = "final"
        result["protocol_sha256"] = queue.json_hash(result["protocol"])
    elif mutation == "preflight":
        experiment = queue.read_json(path / "experiment.json")
        experiment["preflight"] = True
        dump(path / "experiment.json", experiment)
    if mutation != "missing_eval":
        dump(result_path, result)
    with pytest.raises(queue.QueueError, match=match):
        queue.verify_run(path, run)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "failed",
        "not_uploaded",
        "not_verified",
        "stale_hash",
        "preflight",
        "wrong_project",
        "public",
        "no_server_identity",
    ],
)
def test_requires_server_verified_upload_sidecar(tmp_path, mutation):
    run = queue.phase1_rounds()[0][0]
    path = complete_run(tmp_path / run["id"], run)
    receipt_path = path / "eval/step_005000.swanlab.json"
    receipt = queue.read_json(receipt_path)
    if mutation == "missing":
        receipt_path.unlink()
    else:
        if mutation == "failed":
            receipt["status"] = "failed"
        elif mutation == "not_uploaded":
            receipt["uploaded"] = False
        elif mutation == "not_verified":
            receipt["verified_upload"] = False
        elif mutation == "stale_hash":
            receipt["result_sha256"] = "0" * 64
        elif mutation == "preflight":
            receipt["preflight"] = True
        elif mutation == "wrong_project":
            receipt["project"] = "wrong-project"
        elif mutation == "public":
            receipt["public"] = True
        elif mutation == "no_server_identity":
            receipt.pop("id")
        dump(receipt_path, receipt)
    with pytest.raises(queue.QueueError):
        queue.verify_run(path, run)


def test_resume_does_not_trust_completed_training_when_upload_receipt_missing(args, monkeypatch):
    workers = FakeWorkers(monkeypatch, args)
    execute(args)
    run = queue.phase1_rounds()[0][0]
    (args.root / "runs" / run["id"] / "eval/step_030000.swanlab.json").unlink()
    args.resume = True
    with pytest.raises(queue.QueueError, match="Missing/unreadable"):
        execute(args)
    assert len(workers.commands) == 6


def test_uv_wrapper_and_venv_interpreter_symlink_are_preserved(args, tmp_path):
    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    args.python = interpreter
    args.uv = Path("/opt/conda/bin/uv")
    run = queue.phase1_rounds()[0][0]
    command = queue.train_command(args, run)
    assert command[:7] == [str(args.uv), "run", "--no-project", "--no-config", "--", str(interpreter), "-u"]
    argv = [
        "--repo",
        str(args.repo),
        "--python",
        str(interpreter),
        "--eval-python",
        str(interpreter),
        "--root",
        str(args.root),
        "--data",
        str(args.data),
        "--credential-file",
        str(args.credential_file),
    ]
    assert queue.parse_args(argv).python == interpreter
    with pytest.raises(SystemExit):
        queue.parse_args([*argv, "--min-free-gib", "2.99"])
    with pytest.raises(SystemExit):
        queue.parse_args([*argv, "--recover-stale", run["id"]])


def test_nonfinite_and_duplicate_json_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"status": "complete", "success_rate": NaN}')
    with pytest.raises(queue.QueueError, match="Nonfinite"):
        queue.read_json(path)
    path.write_text('{"status": "complete", "status": "running"}')
    with pytest.raises(queue.QueueError, match="Duplicate"):
        queue.read_json(path)


def test_server_generated_run_id_independent_of_request_id(tmp_path):
    run = queue.phase1_rounds()[0][0]
    path = complete_run(tmp_path / run["id"], run)
    for receipt_path in (path / "eval").glob("*.swanlab.json"):
        receipt = queue.read_json(receipt_path)
        receipt["id"] = "3h5w69fl"
        assert receipt["id"] != receipt["attempt_id"][:8]
        dump(receipt_path, receipt)
    assert queue.verify_run(path, run)["id"] == run["id"]
