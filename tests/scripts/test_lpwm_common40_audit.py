"""Offline audit regressions. No SSH, cloud credentials, GPU or simulator required."""

import io
import json
import shutil
import tarfile

import pytest

from scripts.lpwm_monitor import audit_common40 as audit


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    audit.write_json(path, value)


def make_catalog():
    tasks, installed = [], []
    for gid in range(40):
        suite, tid = audit.SUITES[gid // 10], gid % 10
        name = f"task_{gid:02d}"
        common = {
            "global_task_id": gid,
            "suite": suite,
            "task_id": tid,
            "name": name,
            "language": f"task {gid:02d}",
        }
        tasks.append(
            {
                **common,
                "source_task_index": 39 - gid,
                "official_global_task_id": gid if gid < 30 else gid + 90,
                "problem_folder": suite,
                "bddl_file": name + ".bddl",
                "init_states_file": name + ".pruned_init",
            }
        )
        installed.append({**common, "init_state_count": 50})
    return {"suites": dict.fromkeys(audit.SUITES, 10), "tasks": tasks}, {"tasks": installed}


def make_result(catalog, benchmark, marker, root, step):
    tasks, all_rows = [], []
    for task, installed in zip(catalog["tasks"], benchmark["tasks"], strict=True):
        gid, suite = task["global_task_id"], task["suite"]
        rows = []
        for index, state, seed in audit.expected_plan(installed["init_state_count"], gid):
            success = index < step // 5000
            control = 5 if success else audit.LIMITS[suite]
            rows.append(
                {
                    "episode_index": index,
                    "init_state_index": state,
                    "seed": seed,
                    "success": success,
                    "control_steps": control,
                    "terminated": success,
                    "truncated": False,
                    "reached_step_limit": not success,
                    "executed_action_components": control * 7,
                    "clipped_action_components": 0,
                    "action_clipping_fraction": 0.0,
                    "duration_seconds": 1.0,
                }
            )
        tasks.append(
            {
                "task_id": gid,
                "suite": suite,
                "suite_task_id": task["task_id"],
                "task_name": task["name"],
                "language": task["language"],
                "language_cache": {
                    "cache_task_id": gid,
                    "cache_row": gid,
                    "cache_description": task["language"],
                },
                "max_control_steps": audit.LIMITS[suite],
                "episodes": rows,
                **audit.summarize(rows),
            }
        )
        all_rows.extend(rows)
    per_suite = {
        suite: audit.summarize([e for t in tasks if t["suite"] == suite for e in t["episodes"]])
        for suite in audit.SUITES
    }
    artifacts = {k: marker["sha256"][k] for k in audit.EVAL_ARTIFACTS}
    protocol = audit.protocol_expected()
    return {
        "status": "complete",
        "checkpoint": {
            "step": step,
            "variant": "B",
            "training_seed": 42,
            "split_sha256": marker["split_sha256"],
            "path": f"{root}/checkpoints/step_{step:06d}",
            "artifact_sha256": artifacts,
            "bundle_sha256": audit.object_digest(artifacts),
            "model_sha256": artifacts["model.safetensors"],
        },
        "protocol": protocol,
        "protocol_sha256": audit.object_digest(protocol),
        "per_task": tasks,
        "per_suite": per_suite,
        "suite_macro_success_rate": sum(v["success_rate"] for v in per_suite.values()) / 4,
        "duration_seconds": 400.0,
        **audit.summarize(all_rows),
    }


def cloud_payload(expected, legacy=True):
    return {
        "list": [
            {
                "key": key,
                "metrics": [
                    {"index" if legacy else "step": step, "data" if legacy else "value": value}
                    for step, value in values.items()
                ],
            }
            for key, values in expected.items()
        ]
    }


def cloud_run(metadata, expected):
    return {
        "id": metadata["id"],
        "path": audit.run_path(metadata),
        "state": "FINISHED",
        "metrics": cloud_payload(expected),
    }


@pytest.fixture
def result_bundle():
    catalog, benchmark = make_catalog()
    marker = {"split_sha256": "a" * 64, "sha256": dict.fromkeys(audit.ARTIFACTS, "b" * 64)}
    root = "/read-only/run"
    result = make_result(catalog, benchmark, marker, root, 5000)
    return result, catalog, benchmark, marker, root


def validate(bundle):
    result, catalog, benchmark, marker, root = bundle
    return audit.validate_result(result, catalog, benchmark, 5000, marker, root)


def test_complete_episode_and_suite_counts(result_bundle):
    values, suites, tasks, episodes, plan = validate(result_bundle)
    assert (values["successes"], values["num_episodes"], values["pc_success"]) == (40, 400, 10.0)
    assert len(suites) == 4 and len(tasks) == 40 and len(episodes) == len(plan) == 400
    assert all(row["num_episodes"] == 100 for row in suites)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed_namespace", "final"),
        ("seed", 43),
        ("seed", True),
        ("preflight", True),
        ("formal_result_eligible", False),
        ("episodes_per_task", 9),
        ("task_count", 39),
        ("horizon", 8),
        ("control_freq_hz", 10),
        ("actual_preflight_max_steps", 2),
        ("flow_inference_steps", 1),
        ("n_obs_steps", 1),
        ("n_action_steps", 16),
        ("state_preprocessing", "different state"),
    ],
)
def test_reject_rehashed_wrong_protocol(result_bundle, field, value):
    result_bundle[0]["protocol"][field] = value
    result_bundle[0]["protocol_sha256"] = audit.object_digest(result_bundle[0]["protocol"])
    with pytest.raises(audit.AuditError, match="Protocol"):
        validate(result_bundle)


def test_reject_unrecomputed_protocol_hash(result_bundle):
    result_bundle[0]["protocol_sha256"] = "0" * 64
    with pytest.raises(audit.AuditError, match="Protocol SHA256"):
        validate(result_bundle)


@pytest.mark.parametrize(
    "field,value",
    [
        ("success", 1),
        ("success", "false"),
        ("control_steps", 0),
        ("control_steps", 281),
        ("control_steps", True),
        ("executed_action_components", 42),
        ("clipped_action_components", -1),
        ("action_clipping_fraction", float("nan")),
        ("episode_index", 1),
        ("init_state_index", 49),
        ("seed", 0),
        ("terminated", 1),
        ("reached_step_limit", True),
        ("duration_seconds", -1),
    ],
)
def test_reject_bad_episode(result_bundle, field, value):
    result_bundle[0]["per_task"][0]["episodes"][0][field] = value
    with pytest.raises(audit.AuditError):
        validate(result_bundle)


def test_reject_duplicate_episode(result_bundle):
    task = result_bundle[0]["per_task"][0]
    task["episodes"][1] = task["episodes"][0]
    with pytest.raises(audit.AuditError, match="Episode plan"):
        validate(result_bundle)


@pytest.mark.parametrize("target", ["task", "suite", "aggregate"])
def test_reject_inflated_success_count(result_bundle, target):
    result = result_bundle[0]
    record = {
        "task": result["per_task"][0],
        "suite": result["per_suite"]["libero_spatial"],
        "aggregate": result,
    }[target]
    record["successes"] += 1
    with pytest.raises(audit.AuditError, match="successes"):
        validate(result_bundle)


def test_reject_swapped_tasks_even_with_correct_denominator(result_bundle):
    tasks = result_bundle[0]["per_task"]
    tasks[0], tasks[1] = tasks[1], tasks[0]
    with pytest.raises(audit.AuditError, match="Task 0"):
        validate(result_bundle)


def test_reject_missing_suite(result_bundle):
    result_bundle[0]["per_suite"].pop("libero_10")
    with pytest.raises(audit.AuditError, match="suite set"):
        validate(result_bundle)


def test_reject_unbound_eval_model(result_bundle):
    result_bundle[0]["checkpoint"]["model_sha256"] = "0" * 64
    with pytest.raises(audit.AuditError, match="model SHA256"):
        validate(result_bundle)


def test_catalog_matches_installed_identities(result_bundle):
    _, catalog, benchmark, _, _ = result_bundle
    audit.validate_catalog(catalog, benchmark)
    catalog["tasks"][30]["official_global_task_id"] = 30
    with pytest.raises(audit.AuditError, match="Official task offset"):
        audit.validate_catalog(catalog, benchmark)


@pytest.mark.parametrize(
    "key,value",
    [
        ("global_task_id", 5),
        ("task_id", 1),
        ("suite", "libero_90"),
        ("source_task_index", 38),
        ("language", "wrong instruction"),
    ],
)
def test_reject_catalog_corruption(result_bundle, key, value):
    _, catalog, benchmark, _, _ = result_bundle
    catalog["tasks"][0][key] = value
    with pytest.raises(audit.AuditError):
        audit.validate_catalog(catalog, benchmark)


@pytest.mark.parametrize("legacy", [True, False])
def test_cloud_requires_exact_metrics_steps_and_values(legacy):
    expected = {"eval/successes": {1: 203}, "eval/success_rate": {1: 0.5075}}
    meta = {"id": "test", "url": "https://swanlab.cn/@owner/project/runs/test"}
    record = cloud_run(meta, expected)
    record["metrics"] = cloud_payload(expected, legacy)
    assert audit.verify_cloud_run(record, meta, expected) == 2
    record["metrics"]["list"][0]["metrics"][0]["index" if legacy else "step"] = 2
    with pytest.raises(audit.AuditError, match="step coverage"):
        audit.verify_cloud_run(record, meta, expected)


@pytest.mark.parametrize("corruption", ["id", "state", "duplicate", "missing", "value", "boolean"])
def test_reject_false_cloud_verification(corruption):
    expected = {"eval/successes": {1: 203}}
    meta = {"id": "test", "url": "https://swanlab.cn/@owner/project/runs/test"}
    record = cloud_run(meta, expected)
    if corruption == "id":
        record["id"] = "another"
    elif corruption == "state":
        record["state"] = "RUNNING"
    elif corruption == "missing":
        record["metrics"]["list"] = []
    elif corruption == "duplicate":
        record["metrics"]["list"][0]["metrics"] *= 2
    else:
        record["metrics"]["list"][0]["metrics"][0]["data"] = True if corruption == "boolean" else 202
    with pytest.raises(audit.AuditError):
        audit.verify_cloud_run(record, meta, expected)


@pytest.mark.parametrize("name", ["../secret", "/root/secret", "dir/../../secret"])
def test_unsafe_manifest_paths_rejected(tmp_path, name):
    with pytest.raises(audit.AuditError, match="Unsafe path"):
        audit.safe_child(tmp_path, name)


def test_symlink_escape_rejected(tmp_path):
    (tmp_path / "escape").symlink_to("/etc")
    with pytest.raises(audit.AuditError, match="Unsafe path"):
        audit.safe_child(tmp_path, "escape/passwd")


@pytest.mark.parametrize("raw", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}'])
def test_json_parser_rejects_ambiguous_or_nonfinite_data(tmp_path, raw):
    path = tmp_path / "input.json"
    path.write_text(raw)
    with pytest.raises(audit.AuditError):
        audit.read(path)


@pytest.mark.parametrize(
    "name,kind", [("../escape", "file"), ("raw/run/link", "link"), ("outside.json", "file")]
)
def test_snapshot_extraction_rejects_unsafe_members(tmp_path, name, kind):
    archive = tmp_path / "snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo(name)
        if kind == "link":
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            bundle.addfile(info)
        else:
            info.size = 2
            bundle.addfile(info, io.BytesIO(b"{}"))
    with pytest.raises(audit.AuditError, match="Unsafe archive member"):
        audit.extract_snapshot(archive, tmp_path / "extracted")


@pytest.fixture(scope="module")
def full_evidence(tmp_path_factory):
    base = tmp_path_factory.mktemp("common40")
    source, evidence = base / "source/run", base / "evidence"
    source.mkdir(parents=True)
    evidence.mkdir()
    root = str(source)
    catalog, benchmark = make_catalog()
    benchmark.update(root=root, captured_at="synthetic")
    split = {
        "seed": 42,
        "train_episode_ids": [0, 1],
        "validation_episode_ids": [2],
        "validation_fraction": 0.1,
    }
    split["sha256"] = audit.object_digest(split)
    data_manifest = {"task_catalog": catalog}
    save(source.parent / "cache/manifest.json", data_manifest)
    save(source.parent / "cache/task_catalog.json", catalog)
    save(source.parent / "cache/split.json", split)
    experiment = {
        "variant": "B",
        "seed": 42,
        "steps": 30000,
        "preflight": False,
        "swanlab_mode": "online",
        "full_task_count": 40,
        "eval_episodes_per_task": 10,
        "split_sha256": split["sha256"],
        "initial_weights_sha256": "e" * 64,
        "dataset_manifest_sha256": audit.digest(source.parent / "cache/manifest.json"),
        "rec_weight": 1.0,
        "dyn_weight": 1.0,
        "prior_weight": 0.001,
    }
    for name, value in (
        ("config", {}),
        ("task_catalog", catalog),
        ("split", split),
        ("state_normalization", {}),
        ("experiment", experiment),
    ):
        save(source / f"{name}.json", value)
    rows = []
    for step in (1, *range(10, 30001, 10)):
        row = {
            "step": step,
            "train/fm_loss": 1 / step,
            "train/world/world_rec": 0.1,
            "train/world/world_dyn": 0.2,
            "train/world/world_prior": 0.3,
            "train/world_loss": 0.3003,
        }
        rows.append(row)
        if step % 500 == 0:
            rows.append({k.replace("train/", "validation/"): v for k, v in row.items()})
    cloud = {"root": root, "captured_at": "synthetic", "visibility": "PRIVATE", "evaluations": {}}
    for step in audit.STEPS:
        folder = source / "checkpoints" / f"step_{step:06d}"
        folder.mkdir(parents=True)
        for name in audit.ARTIFACTS:
            if name == "experiment.json":
                save(folder / name, {**experiment, "step": step})
            elif (source / name).exists():
                shutil.copyfile(source / name, folder / name)
            else:
                (folder / name).write_bytes(b"synthetic artifact, not a real weight or optimizer")
        marker = {
            "status": "complete",
            "step": step,
            "schema_version": 1,
            "preflight": False,
            "initial_weights_sha256": experiment["initial_weights_sha256"],
            "split_sha256": split["sha256"],
            "sha256": {name: audit.digest(folder / name) for name in audit.ARTIFACTS},
        }
        save(folder / "complete.json", marker)
        result = make_result(catalog, benchmark, marker, root, step)
        path = source / "eval" / f"step_{step:06d}.json"
        save(path, result)
        receipt = {
            "status": "complete",
            "verified_upload": True,
            "uploaded": True,
            "formal_result_eligible": True,
            "protocol_role": "validation",
            "result_sha256": audit.digest(path),
            "id": f"eval{step}",
            "url": f"https://swanlab.cn/@owner/project/runs/eval{step}",
        }
        save(path.with_suffix(".swanlab.json"), receipt)
        metrics = audit.eval_metrics(result)
        rows.append({"step": step, **{k.replace("eval/", "rollout/"): float(v) for k, v in metrics.items()}})
        cloud["evaluations"][str(step)] = cloud_run(receipt, {k: {1: v} for k, v in metrics.items()})
    for name in ("best_loss", "final", "latest"):
        (source / "checkpoints" / name).symlink_to("step_030000")
    save(
        source / "status.json",
        {
            "status": "completed",
            "step": 30000,
            "preflight": False,
            "best_saved_validation_fm_loss": 1 / 30000,
        },
    )
    training = {"id": "training", "url": "https://swanlab.cn/@owner/project/runs/training"}
    save(source / "swanlab_run.json", training)
    (source / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    expected = {}
    for row in rows:
        for key, value in row.items():
            if key != "step":
                expected.setdefault(key, {})[row["step"]] = value
    cloud["training"] = cloud_run(training, expected)
    for name, path in audit.snapshot_files(source).items():
        dest = evidence / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    save(evidence / "remote_hashes.json", audit.remote_hashes(source))
    save(evidence / "benchmark_readback.json", benchmark)
    save(evidence / "cloud_readback.json", cloud)
    return source, evidence


def test_end_to_end_offline_audit_exports_all_curves(full_evidence):
    source, evidence = full_evidence
    report = audit.audit(evidence, str(source))
    counts = report["counts"]
    assert report["status"] == "passed"
    assert counts["checkpoint_payload_hashes"] == 60
    assert counts["optimizer_hashes"] == 6 and counts["episode_outcomes"] == 2400
    assert counts["task_summaries"] == 240 and counts["suite_summaries"] == 24
    assert counts["validation_rows"] == 60 and counts["training_rows"] == 3001
    assert counts["cloud_eval_metric_points"] == 294 and counts["cloud_runs"] == 7
    assert len((evidence / "audit/episode_outcomes.jsonl").read_text().splitlines()) == 2400
    assert len((evidence / "audit/episode_outcomes.csv").read_text().splitlines()) == 2401
    assert (evidence / "archive_sha256.json").is_file()
    assert all(p.suffix not in {".pt", ".npy", ".safetensors"} for p in (evidence / "raw/run").rglob("*"))
    assert any("Single training seed" in c for c in report["caveats"])
    assert any("NOT 2400 independent" in c for c in report["caveats"])


@pytest.mark.parametrize(
    "target", ["optimizer", "manifest", "snapshot", "receipt", "cloud", "alias", "missing_hash"]
)
def test_replay_fails_closed_on_tampered_evidence(full_evidence, tmp_path, target):
    source, original = full_evidence
    evidence = tmp_path / "evidence"
    shutil.copytree(original, evidence)
    hashes_path = evidence / "remote_hashes.json"
    hashes = audit.read(hashes_path)
    if target == "optimizer":
        hashes["checkpoints"]["5000"]["optimizer.pt"]["sha256"] = "0" * 64
        save(hashes_path, hashes)
    elif target == "manifest":
        hashes["checkpoints"]["5000"].pop("optimizer.pt")
        save(hashes_path, hashes)
    elif target == "missing_hash":
        hashes["files"].pop("raw/run/checkpoints/step_005000/complete.json")
        save(hashes_path, hashes)
    elif target == "alias":
        hashes["checkpoint_aliases"]["final"] = "step_025000"
        save(hashes_path, hashes)
    elif target == "cloud":
        path = evidence / "cloud_readback.json"
        record = audit.read(path)
        record["evaluations"]["5000"]["state"] = "RUNNING"
        save(path, record)
    else:
        path = evidence / (
            "raw/run/status.json" if target == "snapshot" else "raw/run/eval/step_005000.swanlab.json"
        )
        value = audit.read(path)
        value["status"] = "failed"
        save(path, value)
    with pytest.raises(audit.AuditError):
        audit.audit(evidence, str(source))


def test_remote_hash_worker_detects_optimizer_bit_change(full_evidence):
    source, _ = full_evidence
    optimizer = source / "checkpoints/step_005000/optimizer.pt"
    original = optimizer.read_bytes()
    try:
        optimizer.write_bytes(original + b"corruption")
        result = audit.remote_hashes(source)
        assert result["checkpoints"]["5000"]["optimizer.pt"]["matched"] is False
    finally:
        optimizer.write_bytes(original)


def test_collection_uses_only_inline_readonly_workers(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write(b"{}")
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    monkeypatch.setattr(audit, "extract_snapshot", lambda *args: None)
    audit.collect(tmp_path, "/read-only/run")
    assert len(calls) == 4
    assert [command[-1].split("--worker ")[1].split()[0] for command, _ in calls] == [
        "snapshot",
        "hashes",
        "benchmark",
        "cloud",
    ]
    for command, kwargs in calls:
        assert command[:-1] == list(audit.SSH)
        assert " -B - " in command[-1]
        assert audit.CREDENTIAL not in command[-1]
        assert isinstance(kwargs["input"], bytes)
        assert all(word not in command[-1] for word in ("kill", "nohup", "train.py", "evaluate.py"))
    with pytest.raises(audit.AuditError, match="already exists"):
        audit.collect(tmp_path, "/read-only/run")


def test_cloud_worker_failure_never_prints_secret(tmp_path, monkeypatch, capsys):
    def fail(*args):
        raise RuntimeError("API_KEY_DO_NOT_EXPOSE")

    monkeypatch.setattr(audit, "cloud_readback", fail)
    assert audit.main(["--worker", "cloud", "--remote-root", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert "API_KEY_DO_NOT_EXPOSE" not in output.err + output.out
    assert "RuntimeError" in output.err
