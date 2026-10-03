#!/usr/bin/env python3
"""Read-only common40 B30k completion audit; never launches training/eval or writes ARA.

Collect (SSH runs only this module's read-only workers; credentials stay remote)::

    uv run --no-sync python scripts/lpwm_monitor/audit_common40.py \
        --evidence /tmp/lpwm-common40-completion-20260922 --collect

Replay the audit entirely from the evidence directory by omitting --collect.
Weights/optimizer are streamed into SHA256 *on the remote*, never downloaded or
unpickled. Cloud access is Api.run/metrics only: no login/save/init/log/finish.
The report distinguishes local consistency from remote hash and cloud readback.
"""

import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import math
import re
import shlex
import subprocess
import sys
import tarfile
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

STEPS = tuple(range(5000, 30001, 5000))
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIMITS = dict(zip(SUITES, (280, 280, 300, 520), strict=True))
ARTIFACTS = {
    "config.json",
    "experiment.json",
    "language_embeddings.npy",
    "language_masks.npy",
    "language_metadata.json",
    "model.safetensors",
    "optimizer.pt",
    "split.json",
    "state_normalization.json",
    "task_catalog.json",
}
EVAL_ARTIFACTS = ARTIFACTS - {"optimizer.pt", "split.json", "task_catalog.json"}
REMOTE_ROOT = "/root/gpufree-data/lpwm_40/20260921/run"
SSH = (
    "ssh",
    "-F",
    "/dev/null",
    "-S",
    "/tmp/lpwm-full-remote-control",
    "-o",
    "UserKnownHostsFile=/tmp/lpwm-remote-known-hosts",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-p30429",
    "root@120.209.70.195",
)
REMOTE_PYTHON = "/root/lpwm_ab/.venv-eval/bin/python"
CREDENTIAL = "/root/lpwm_ab/.swanlab_api_key"
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".yaml", ".yml", ".txt", ".csv"}
CONTEXT_FILES = (
    "training.log",
    "launcher.log",
    "pipeline_status.json",
    "environment.json",
    "frozen_source_sha256.json",
    "cache/manifest.json",
    "cache/task_catalog.json",
    "cache/split.json",
    "cache/verification.json",
    "cache/source_manifest.json",
    "repo/source-manifest.json",
    "repo/scripts/lpwm_full/evaluate.py",
    "repo/scripts/lpwm_full/train.py",
    "repo/scripts/lpwm_full/prepare_common40.py",
    "repo/scripts/lpwm_ab/train_b_sweep.py",
    "repo/scripts/lpwm_ab/evaluate.py",
    "repo/scripts/lpwm_ab/eval_b_sweep.py",
    "repo/scripts/lpwm_ab/data.py",
    "repo/src/lerobot/envs/libero.py",
)
CAVEATS = [
    "Single training seed (42); no cross-seed uncertainty or generalization claim.",
    "All six checkpoints reuse the same 400 validation episode plans (40 tasks x 10); "
    "2400 outcome rows are NOT 2400 independent initial states or a held-out final test.",
    "Training includes all four suites/all40 task instances, including LIBERO_10; "
    "NOT a LIBERO90-to10 held-out-task experiment.",
    "Hash consistency and cloud scalar readback bind stored evidence, not historical simulator "
    "execution authenticity; no rollouts were rerun and no videos/actions were independently replayed.",
    "Official HDF5 references in the catalog are reference-only, not provenance for converted "
    "training rows. Dataset image/scalar payloads are not rehashed by this completion audit.",
    "Optimizer integrity means byte-level SHA256 only, not a deserialization/resumability test. "
    "Weights/optimizer and private simulator assets are not copied into the archive.",
]


class AuditError(ValueError):
    """An evidence invariant failed (also enforced with python -O)."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def now():
    return datetime.now(UTC).isoformat()


def read(path):
    def reject(value):
        raise AuditError(f"Nonfinite JSON constant: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=unique)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def integer(value):
    return type(value) is int


def equal(actual, expected, label):
    if isinstance(expected, float):
        require(
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
            label,
        )
    else:
        require(type(actual) is type(expected) and actual == expected, label)


def norm(text):
    require(isinstance(text, str) and bool(text), "Empty language")
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def safe_child(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe path: {name}")
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), f"Unsafe path: {name}")
    return path


def snapshot_files(root):
    """Allowlisted text evidence only, never secrets, weights or simulator assets."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.suffix in TEXT_SUFFIXES:
            name = str(path.relative_to(root))
            require(not any(part.startswith(".") for part in Path(name).parts), "Hidden evidence file")
            result[f"raw/run/{name}"] = safe_child(root, name)
    for name in CONTEXT_FILES:
        path = safe_child(root.parent, name)
        if path.is_file():
            result[f"raw/context/{name}"] = path
    return result


def hash_record(path):
    before = path.stat()
    sha = digest(path)
    after = path.stat()
    require(
        (before.st_size, before.st_mtime_ns, before.st_ino)
        == (after.st_size, after.st_mtime_ns, after.st_ino),
        f"File changed during hashing: {path.name}",
    )
    return {"sha256": sha, "bytes": after.st_size, "mtime_ns": after.st_mtime_ns}


def remote_hashes(root):
    """Hash ALL checkpoint manifest members, including optimizer; no pickle loading."""
    files = snapshot_files(root)
    checkpoints = {}
    require(
        {p.name for p in (root / "checkpoints").iterdir() if p.is_dir() and not p.is_symlink()}
        == {f"step_{step:06d}" for step in STEPS},
        "Unexpected checkpoint set",
    )
    for step in STEPS:
        folder = root / "checkpoints" / f"step_{step:06d}"
        marker = read(folder / "complete.json")
        require(set(marker["sha256"]) == ARTIFACTS, f"step {step}: incomplete hash manifest")
        records = {}
        for name, expected in marker["sha256"].items():
            path = safe_child(folder, name)
            record = hash_record(path)
            record.update(expected_sha256=expected, matched=record["sha256"] == expected)
            records[name] = record
            files[f"raw/run/checkpoints/{folder.name}/{name}"] = path
        checkpoints[str(step)] = records
    # Rehash the small manifest/result files after payloads; local audit binds their exact bytes.
    records = {
        name: hash_record(path)
        for name, path in files.items()
        if name.split("/")[-1]
        not in ("model.safetensors", "optimizer.pt", "language_embeddings.npy", "language_masks.npy")
    }
    return {
        "schema_version": 1,
        "root": str(root),
        "captured_at": now(),
        "checkpoints": checkpoints,
        "files": records,
        "checkpoint_aliases": {
            p.name: str(p.readlink()) for p in (root / "checkpoints").iterdir() if p.is_symlink()
        },
    }


def benchmark_evidence(root):
    """Read installed task map and trusted init tensors, not a simulator/environment."""
    import numpy as np
    import torch

    base = Path(sys.prefix) / "lib/python3.12/site-packages/libero/libero"
    mapping = base / "benchmark/libero_suite_task_map.py"
    tree = ast.parse(mapping.read_text())
    task_map = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in node.targets)
    )
    catalog = read(root / "task_catalog.json")
    rows = []
    # The installed trusted files contain numpy arrays; narrowly allowlist numpy reconstruction.
    allowed = [
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
    ]
    for task in catalog["tasks"]:
        suite, tid = task["suite"], task["task_id"]
        name = task_map[suite][tid]
        init = safe_child(base / "init_files", f"{suite}/{name}.pruned_init")
        bddl = safe_child(base / "bddl_files", f"{suite}/{name}.bddl")
        with torch.serialization.safe_globals(allowed):
            states = torch.load(init, map_location="cpu", weights_only=True)
        language_name = name
        if name[0].isupper():
            language_name = name[name.find("SCENE") + (8 if "SCENE10" in name else 7) :]
        rows.append(
            {
                "suite": suite,
                "task_id": tid,
                "global_task_id": len(rows),
                "name": name,
                "language": language_name.replace("_", " "),
                "init_state_count": len(states),
                "init_states_sha256": digest(init),
                "bddl_sha256": digest(bddl),
            }
        )
    return {
        "captured_at": now(),
        "root": str(root),
        "benchmark_root": str(base),
        "task_map_sha256": digest(mapping),
        "tasks": rows,
    }


def run_path(metadata):
    url = urlparse(metadata["url"])
    parts = url.path.strip("/").split("/")
    require(
        url.scheme == "https"
        and url.hostname == "swanlab.cn"
        and len(parts) == 4
        and parts[0].startswith("@")
        and parts[2] == "runs"
        and parts[3] == metadata["id"],
        "Invalid SwanLab run identity",
    )
    return "/".join((parts[0][1:], parts[1], parts[3]))


def metric_rows(root):
    rows = []
    for line in (root / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        require(integer(row.get("step")), "Noninteger training step")
        for key, value in row.items():
            require(type(value) in (int, float) and math.isfinite(value), f"Nonfinite metric {key}")
        rows.append(row)
    return rows


def eval_metrics(result):
    # Counts/rates are subsequently independently checked against episode rows.
    return {
        **{
            f"eval/{key}": result[key]
            for key in ("successes", "num_episodes", "success_rate", "pc_success", "suite_macro_success_rate")
        },
        **{f"eval/{name}/success_rate": result["per_suite"][name]["success_rate"] for name in SUITES},
        **{
            f"eval/{task['suite']}/task_{task['suite_task_id']:02d}/success_rate": task["success_rate"]
            for task in result["per_task"]
        },
    }


def cloud_readback(root, credential):
    """Only existing runs. Suppress SDK chatter; never emit credentials or raw errors."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import swanlab

        api = swanlab.Api(api_key=Path(credential).read_text().strip())
        training = read(root / "swanlab_run.json")
        project_path = run_path(training).rsplit("/", 1)[0]
        visibility = api.project(project_path).visibility
        keys = sorted({key for row in metric_rows(root) for key in row if key != "step"})

        def fetch(metadata, keys, full):
            path = run_path(metadata)
            require(path.rsplit("/", 1)[0] == project_path, "Different cloud project")
            remote = api.run(path)
            return {
                "id": metadata["id"],
                "path": path,
                "state": remote.state,
                "metrics": remote.metrics(keys=keys, all=full, sample=10, ignore_timestamp=True),
            }

        train = fetch(training, keys, True)
        evaluations = {}
        for step in STEPS:
            path = root / "eval" / f"step_{step:06d}.json"
            evaluations[str(step)] = fetch(
                read(path.with_suffix(".swanlab.json")), sorted(eval_metrics(read(path))), True
            )
        return {
            "captured_at": now(),
            "root": str(root),
            "sdk_version": swanlab.__version__,
            "project": project_path,
            "visibility": visibility,
            "training": train,
            "evaluations": evaluations,
        }


def protocol_expected():
    return {
        "name": "lpwm-libero-common40-rollout-v1",
        "suites": dict.fromkeys(SUITES, 10),
        "scope_task_count": 40,
        "formal_result_eligible": True,
        "seed": 42,
        "seed_namespace": "validation",
        "preflight": False,
        "episodes_per_task": 10,
        "task_count": 40,
        "control_freq_hz": 20,
        "suite_max_steps": LIMITS,
        "actual_preflight_max_steps": None,
        "state_pool": "validation firsthalf;final secondhalf of seeded no-replacement initstates; task seed uses globalcatalogid",
        "image_preprocessing": "raw OpenGL128 RGB rotate180 ONCE;PILbilinear128;CHWfloat/255",
        "state_preprocessing": "eef_pos3+xyzwquat_to_axisangle3+gripperqpos2; savedtrainmean/std",
        "action_execution": "native relativeOSC_POSE7,clip[-1,1],no gripperinversion",
        "horizon": 16,
        "n_action_steps": 8,
        "n_obs_steps": 2,
        "flow_inference_steps": 10,
        "language": "exact normalized text; duplicatedtext tokens must be byteidentical",
        "training_includes_all_suites": "NOT a held-out-task LIBERO90-to10 generalization experiment",
    }


def seed_for(gid, episode):
    text = f"lpwm-libero-rollout-v1:validation:42:{gid}:{episode}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def expected_plan(total_states, gid):
    import numpy as np

    require(integer(total_states) and total_states >= 20, "Too few init states")
    pool = np.arange(total_states // 2)
    states = np.random.default_rng(seed_for(gid, -1)).permutation(pool)[:10]
    return [(i, int(state), seed_for(gid, i)) for i, state in enumerate(states)]


def validate_catalog(catalog, benchmark):
    tasks = catalog["tasks"]
    require(catalog["suites"] == dict.fromkeys(SUITES, 10), "Catalog suite counts")
    require(len(tasks) == len(benchmark["tasks"]) == 40, "Catalog must contain 40 tasks")
    require(len({t["source_task_index"] for t in tasks}) == 40, "Duplicate source task mapping")
    for gid, (task, installed) in enumerate(zip(tasks, benchmark["tasks"], strict=True)):
        suite, tid = SUITES[gid // 10], gid % 10
        for key, value in {"suite": suite, "task_id": tid, "global_task_id": gid}.items():
            equal(task[key], value, f"Catalog {gid}: {key}")
            equal(installed[key], value, f"Installed catalog {gid}: {key}")
        equal(task["official_global_task_id"], gid if gid < 30 else gid + 90, "Official task offset")
        require(integer(task["source_task_index"]), "Invalid source task ID")
        require(
            task["name"] == installed["name"] and norm(task["language"]) == norm(installed["language"]),
            f"Installed catalog identity mismatch: {gid}",
        )
        for key, value in {
            "problem_folder": suite,
            "bddl_file": task["name"] + ".bddl",
            "init_states_file": task["name"] + ".pruned_init",
        }.items():
            equal(task[key], value, f"Catalog {gid}: {key}")


def summarize(rows):
    successes = sum(row["success"] for row in rows)
    components = sum(row["executed_action_components"] for row in rows)
    clipped = sum(row["clipped_action_components"] for row in rows)
    return {
        "successes": successes,
        "num_episodes": len(rows),
        "success_rate": successes / len(rows),
        "pc_success": 100.0 * successes / len(rows),
        "control_steps": sum(row["control_steps"] for row in rows),
        "clipped_action_components": clipped,
        "executed_action_components": components,
        "action_clipping_fraction": clipped / components,
    }


def validate_summary(record, rows, label):
    values = summarize(rows)
    for key, expected in values.items():
        equal(record.get(key), expected, f"{label}: {key}")
    return values


def validate_result(result, catalog, benchmark, step, manifest, root):
    require(result["status"] == "complete", "Incomplete result")
    protocol = result["protocol"]
    require(set(protocol) == set(protocol_expected()), "Protocol fields changed")
    for key, value in protocol_expected().items():
        equal(protocol[key], value, f"Protocol: {key}")
    equal(result["protocol_sha256"], object_digest(protocol), "Protocol SHA256 mismatch")
    cp = result["checkpoint"]
    for key, value in {
        "step": step,
        "variant": "B",
        "training_seed": 42,
        "split_sha256": manifest["split_sha256"],
        "path": f"{root}/checkpoints/step_{step:06d}",
    }.items():
        equal(cp[key], value, f"Checkpoint binding: {key}")
    artifacts = {name: manifest["sha256"][name] for name in EVAL_ARTIFACTS}
    equal(cp["artifact_sha256"], artifacts, "Eval artifact hash binding")
    equal(cp["bundle_sha256"], object_digest(artifacts), "Eval bundle SHA256")
    equal(cp["model_sha256"], artifacts["model.safetensors"], "Eval model SHA256")
    require(len(result["per_task"]) == 40, "Wrong task count")
    require(set(result["per_suite"]) == set(SUITES), "Wrong suite set")
    all_rows, outcomes, task_rows, plans = [], [], [], []
    for task, item, installed in zip(result["per_task"], catalog["tasks"], benchmark["tasks"], strict=True):
        gid, suite, tid = item["global_task_id"], item["suite"], item["task_id"]
        for key, value in {
            "task_id": gid,
            "suite": suite,
            "suite_task_id": tid,
            "task_name": item["name"],
            "max_control_steps": LIMITS[suite],
        }.items():
            equal(task[key], value, f"Task {gid}: {key}")
        require(norm(task["language"]) == norm(item["language"]), "Task language mismatch")
        info = task["language_cache"]
        require(norm(info["cache_description"]) == norm(item["language"]), "Cached language mismatch")
        cache_gid = info["cache_task_id"]
        require(integer(cache_gid) and cache_gid in range(40), "Invalid language cache task")
        require(
            norm(catalog["tasks"][cache_gid]["language"]) == norm(item["language"]), "Wrong cache identity"
        )
        rows = task["episodes"]
        require(len(rows) == 10, "Wrong episode count")
        plan = expected_plan(installed["init_state_count"], gid)
        for row, expected in zip(rows, plan, strict=True):
            for key, value in zip(("episode_index", "init_state_index", "seed"), expected, strict=True):
                equal(row[key], value, f"Episode plan {step}/{gid}: {key}")
            for key in ("success", "terminated", "truncated", "reached_step_limit"):
                require(type(row[key]) is bool, f"Nonboolean outcome: {key}")
            control = row["control_steps"]
            require(integer(control) and 1 <= control <= LIMITS[suite], "Invalid episode horizon")
            equal(row["executed_action_components"], control * 7, "Action component count")
            clipped = row["clipped_action_components"]
            require(integer(clipped) and 0 <= clipped <= control * 7, "Invalid clipped count")
            equal(row["action_clipping_fraction"], clipped / (control * 7), "Episode clipping fraction")
            equal(
                row["reached_step_limit"],
                control == LIMITS[suite] and not (row["success"] or row["terminated"] or row["truncated"]),
                "Step limit flag",
            )
            require(
                row["success"] or row["terminated"] or row["truncated"] or row["reached_step_limit"],
                "Unfinished episode",
            )
            require(
                type(row["duration_seconds"]) in (int, float)
                and math.isfinite(row["duration_seconds"])
                and row["duration_seconds"] > 0,
                "Episode duration",
            )
            outcomes.append({"step": step, "suite": suite, "task_id": gid, "suite_task_id": tid, **row})
            plans.append((gid, *expected))
        values = validate_summary(task, rows, f"Task {gid}")
        task_rows.append(
            {
                "step": step,
                "suite": suite,
                "task_id": gid,
                "suite_task_id": tid,
                "language": item["language"],
                **values,
            }
        )
        all_rows.extend(rows)
    suite_rows = []
    for suite in SUITES:
        rows = [r for r in outcomes if r["suite"] == suite]
        values = validate_summary(result["per_suite"][suite], rows, suite)
        suite_rows.append({"step": step, "suite": suite, **values})
    values = validate_summary(result, all_rows, "Aggregate")
    equal(result["suite_macro_success_rate"], sum(s["success_rate"] for s in suite_rows) / 4, "Suite macro")
    return values, suite_rows, task_rows, outcomes, plans


def cloud_points(payload):
    points = {}
    for row in payload["list"]:
        key = row["key"]
        require(key not in points, f"Duplicate cloud metric key: {key}")
        values = {}
        for point in row["metrics"]:
            step = point.get("step", point.get("index"))
            value = point.get("value", point.get("data"))
            require(integer(step) and step not in values, f"Duplicate/invalid cloud step: {key}")
            require(type(value) in (int, float) and math.isfinite(value), f"Invalid cloud scalar: {key}")
            values[step] = value
        points[key] = values
    return points


def verify_cloud_run(record, metadata, expected):
    equal(record["id"], metadata["id"], "Cloud run ID mismatch")
    equal(record["path"], run_path(metadata), "Cloud run path mismatch")
    equal(record["state"], "FINISHED", "Cloud run not FINISHED")
    actual = cloud_points(record["metrics"])
    require(set(actual) == set(expected), "Missing/unexpected cloud metric keys")
    count = 0
    for key, values in expected.items():
        require(set(actual[key]) == set(values), f"Cloud step coverage: {key}")
        for step, value in values.items():
            # API serializes scalar floats with finite precision.
            require(
                math.isclose(actual[key][step], value, rel_tol=1e-6, abs_tol=1e-6),
                f"Cloud scalar mismatch: {key}@{step}",
            )
            count += 1
    return count


def validate_training(root, experiment, rollout_metrics):
    rows = metric_rows(root)
    train = [r for r in rows if "train/fm_loss" in r]
    val = [r for r in rows if "validation/fm_loss" in r]
    rollout = [r for r in rows if "rollout/successes" in r]
    require(
        [r["step"] for r in train] == [1, *range(10, 30001, 10)], "Training curve missing/duplicate steps"
    )
    require(
        [r["step"] for r in val] == list(range(500, 30001, 500)), "Validation curve missing/duplicate steps"
    )
    require([r["step"] for r in rollout] == list(STEPS), "Rollout metric steps")
    for row in rollout:
        for key, value in rollout_metrics[row["step"]].items():
            equal(float(row[key.replace("eval/", "rollout/", 1)]), float(value), f"Training rollout {key}")
    for prefix, curve in (("train", train), ("validation", val)):
        for row in curve:
            rec, dyn, prior = (row[f"{prefix}/world/world_{k}"] for k in ("rec", "dyn", "prior"))
            world = (
                experiment["rec_weight"] * rec
                + experiment["dyn_weight"] * dyn
                + experiment["prior_weight"] * prior
            )
            require(
                math.isclose(world, row[f"{prefix}/world_loss"], rel_tol=1e-5, abs_tol=1e-7),
                f"World loss decomposition at {row['step']}",
            )
    expected = {}
    for row in rows:
        for key, value in row.items():
            if key != "step":
                values = expected.setdefault(key, {})
                require(row["step"] not in values, f"Duplicate local metric: {key}@{row['step']}")
                values[row["step"]] = value
    return train, val, expected


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def audit(evidence, remote_root=REMOTE_ROOT):
    evidence = Path(evidence)
    root = evidence / "raw/run"
    hashes, benchmark, cloud = (
        read(evidence / name)
        for name in ("remote_hashes.json", "benchmark_readback.json", "cloud_readback.json")
    )
    for record in (hashes, benchmark, cloud):
        equal(record["root"], remote_root, "Evidence root binding")
    verified_files = 0
    for name, record in hashes["files"].items():
        path = safe_child(evidence, name)
        require(
            path.is_file() and path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
            f"Snapshot/remote mismatch: {name}",
        )
        verified_files += 1
    # Refuse a self-consistent but incomplete remote hash report.
    required = {
        "raw/run/" + str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.suffix in TEXT_SUFFIXES
    }
    require(required <= set(hashes["files"]), "Missing remote snapshot hashes")
    require(set(hashes["checkpoints"]) == {str(s) for s in STEPS}, "Missing checkpoint hash records")
    require(
        {p.name for p in (root / "checkpoints").iterdir()} == {f"step_{s:06d}" for s in STEPS},
        "Unexpected local checkpoints",
    )
    expected_result_files = {f"step_{s:06d}.json" for s in STEPS}
    require(
        {
            p.name
            for p in (root / "eval").glob("step_*.json")
            if not p.name.endswith((".partial.json", ".swanlab.json"))
        }
        == expected_result_files,
        "Unexpected final eval files",
    )
    catalog, split, experiment = (
        read(root / name) for name in ("task_catalog.json", "split.json", "experiment.json")
    )
    validate_catalog(catalog, benchmark)
    equal(
        split["sha256"],
        object_digest({k: v for k, v in split.items() if k != "sha256"}),
        "Split object SHA256",
    )
    require(
        not set(split["train_episode_ids"]) & set(split["validation_episode_ids"]), "Split episode leakage"
    )
    for key, value in {
        "variant": "B",
        "seed": 42,
        "steps": 30000,
        "preflight": False,
        "swanlab_mode": "online",
        "full_task_count": 40,
        "eval_episodes_per_task": 10,
        "split_sha256": split["sha256"],
    }.items():
        equal(experiment[key], value, f"Training experiment: {key}")
    status = read(root / "status.json")
    require(
        status["status"] == "completed" and status["step"] == 30000 and status["preflight"] is False,
        "Training not completed B30k",
    )
    context = evidence / "raw/context/cache"
    equal(read(context / "task_catalog.json"), catalog, "Cache catalog binding")
    equal(read(context / "split.json"), split, "Cache split binding")
    equal(digest(context / "manifest.json"), experiment["dataset_manifest_sha256"], "Dataset manifest SHA256")
    manifest = read(context / "manifest.json")
    equal(manifest["task_catalog"], catalog, "Manifest catalog binding")
    aggregate, suites, tasks, episodes = [], [], [], []
    reference_plan, rollout_metrics, eval_ids = None, {}, []
    payload_count = cloud_eval_count = 0
    for step in STEPS:
        folder = root / "checkpoints" / f"step_{step:06d}"
        marker = read(folder / "complete.json")
        for key, value in {
            "status": "complete",
            "step": step,
            "preflight": False,
            "schema_version": 1,
            "split_sha256": split["sha256"],
            "initial_weights_sha256": experiment["initial_weights_sha256"],
        }.items():
            equal(marker[key], value, f"Complete marker {step}: {key}")
        remote = hashes["checkpoints"][str(step)]
        require(set(marker["sha256"]) == set(remote) == ARTIFACTS, "Checkpoint manifest coverage")
        for name, expected in marker["sha256"].items():
            record = remote[name]
            require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "Invalid hash")
            require(
                record["sha256"] == record["expected_sha256"] == expected and record["matched"] is True,
                f"Remote checkpoint hash mismatch: {step}/{name}",
            )
            require(integer(record["bytes"]) and record["bytes"] > 0, "Empty checkpoint artifact")
            if (folder / name).is_file():
                equal(digest(folder / name), expected, f"Local checkpoint hash: {step}/{name}")
            payload_count += 1
        for name in ("config.json", "task_catalog.json", "split.json", "state_normalization.json"):
            equal(read(folder / name), read(root / name), f"Checkpoint/root binding: {step}/{name}")
        cp_exp = read(folder / "experiment.json")
        for key in (
            "variant",
            "seed",
            "split_sha256",
            "dataset_manifest_sha256",
            "initial_weights_sha256",
            "preflight",
        ):
            equal(cp_exp[key], experiment[key], f"Checkpoint experiment {key}")
        equal(cp_exp["step"], step, "Checkpoint experiment step")
        result_path = root / "eval" / f"step_{step:06d}.json"
        result, receipt = read(result_path), read(result_path.with_suffix(".swanlab.json"))
        for key, value in {
            "status": "complete",
            "verified_upload": True,
            "uploaded": True,
            "formal_result_eligible": True,
            "protocol_role": "validation",
            "result_sha256": digest(result_path),
        }.items():
            equal(receipt[key], value, f"Receipt {step}: {key}")
        values, srows, trows, erows, plan = validate_result(
            result, catalog, benchmark, step, marker, remote_root
        )
        if reference_plan is None:
            reference_plan = plan
        equal(plan, reference_plan, "Different repeated validation pool")
        expected = eval_metrics(result)
        rollout_metrics[step] = expected
        cloud_eval_count += verify_cloud_run(
            cloud["evaluations"][str(step)], receipt, {k: {1: v} for k, v in expected.items()}
        )
        eval_ids.append(receipt["id"])
        aggregate.append(
            {
                "step": step,
                **values,
                "suite_macro_success_rate": result["suite_macro_success_rate"],
                "evaluation_run_id": receipt["id"],
                "result_sha256": receipt["result_sha256"],
                "protocol_sha256": result["protocol_sha256"],
                "episode_plan_sha256": object_digest(plan),
            }
        )
        suites.extend(srows)
        tasks.extend(trows)
        episodes.extend(erows)
    require(len(set(eval_ids)) == 6, "Repeated eval cloud IDs")
    equal(set(cloud["evaluations"]), {str(s) for s in STEPS}, "Cloud eval coverage")
    equal(cloud["visibility"], "PRIVATE", "Cloud project not private")
    train, val, expected = validate_training(root, experiment, rollout_metrics)
    training_metadata = read(root / "swanlab_run.json")
    require(training_metadata["id"] not in eval_ids, "Training/eval run ID collision")
    cloud_train_count = verify_cloud_run(cloud["training"], training_metadata, expected)
    best_saved = min((r for r in val if r["step"] in STEPS), key=lambda r: r["validation/fm_loss"])
    equal(
        status["best_saved_validation_fm_loss"],
        best_saved["validation/fm_loss"],
        "Best saved validation loss",
    )
    equal(
        hashes["checkpoint_aliases"],
        {"final": "step_030000", "latest": "step_030000", "best_loss": f"step_{best_saved['step']:06d}"},
        "Checkpoint aliases",
    )
    counts = {
        "checkpoints": 6,
        "checkpoint_payload_hashes": payload_count,
        "optimizer_hashes": 6,
        "complete_json_hashes": 6,
        "snapshot_file_hashes": verified_files,
        "catalogs": 6,
        "protocols": 6,
        "result_receipt_bindings": 6,
        "suite_summaries": len(suites),
        "task_summaries": len(tasks),
        "episode_outcomes": len(episodes),
        "unique_validation_episode_plans": len(reference_plan),
        "training_rows": len(train),
        "validation_rows": len(val),
        "cloud_runs": 7,
        "cloud_eval_metric_points": cloud_eval_count,
        "cloud_training_metric_points": cloud_train_count,
    }
    report = {
        "status": "passed",
        "audited_at": now(),
        "remote_root": remote_root,
        "catalog_file_sha256": digest(root / "task_catalog.json"),
        "catalog_object_sha256": object_digest(catalog),
        "protocol_sha256": aggregate[0]["protocol_sha256"],
        "counts": counts,
        "aggregate_curve": aggregate,
        "suite_curve": suites,
        "training_run_id": training_metadata["id"],
        "evaluation_run_ids": eval_ids,
        "best_saved_validation": best_saved,
        "final_validation": val[-1],
        "caveats": CAVEATS,
        "evidence_timestamps": {
            "hashes": hashes["captured_at"],
            "benchmark": benchmark["captured_at"],
            "cloud": cloud["captured_at"],
        },
    }
    out = evidence / "audit"
    out.mkdir(exist_ok=True)
    for name, rows in (
        ("aggregate_curve", aggregate),
        ("suite_curve", suites),
        ("task_curve", tasks),
        ("episode_outcomes", episodes),
        ("training_curve", train),
        ("validation_curve", val),
    ):
        write_csv(out / f"{name}.csv", rows)
    (out / "episode_outcomes.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in episodes)
    )
    write_json(out / "report.json", report)
    lines = [
        "# Common40 LPWM-FM B30k completion audit",
        "",
        "Status: PASSED",
        "",
        "| Step | Spatial /100 | Object /100 | Goal /100 | Long /100 | Aggregate /400 | Success |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        values = [
            next(s["successes"] for s in suites if s["step"] == row["step"] and s["suite"] == name)
            for name in SUITES
        ]
        lines.append(
            f"| {row['step']} | "
            + " | ".join(map(str, values))
            + f" | {row['successes']}/400 | {row['pc_success']:.2f}% |"
        )
    lines += [
        "",
        "## Verification counts",
        "",
        *[f"- {key}: {value}" for key, value in counts.items()],
        "",
        "## Sources",
        "",
        f"- Remote root: `{remote_root}`",
        "- `../raw/run/`: logs, complete manifests, catalog, configs, final evals and receipts (no weights).",
        "- `../raw/context/`: frozen evaluator/trainer source, dataset manifests and training log.",
        "- `../remote_hashes.json`: on-host streaming SHA256; exact copied bytes checked locally.",
        "- `../benchmark_readback.json`: installed canonical task map, BDDL/init hashes and init counts.",
        "- `../cloud_readback.json`: fresh read-only existing-run API responses, no uploaded scalars.",
        "- `../archive_sha256.json`: local evidence-file inventory; SHA256 is integrity, not a signature.",
        "",
        "## Limitations",
        "",
        *[f"- {c}" for c in CAVEATS],
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    inventory = {
        str(p.relative_to(evidence)): hash_record(p)
        for p in sorted(evidence.rglob("*"))
        if p.is_file() and p.name != "archive_sha256.json"
    }
    write_json(evidence / "archive_sha256.json", {"captured_at": now(), "files": inventory})
    return report


def extract_snapshot(archive, destination):
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            require(
                member.isfile()
                and member.name.startswith("raw/")
                and ".." not in Path(member.name).parts
                and not Path(member.name).is_absolute(),
                "Unsafe archive member",
            )
        bundle.extractall(destination, filter="data")


def collect(evidence, remote_root):
    evidence.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    for worker, filename in (
        ("snapshot", "snapshot-nonweights.tar.gz"),
        ("hashes", "remote_hashes.json"),
        ("benchmark", "benchmark_readback.json"),
        ("cloud", "cloud_readback.json"),
    ):
        command = shlex.join([REMOTE_PYTHON, "-B", "-", "--worker", worker, "--remote-root", remote_root])
        target = evidence / filename
        # Do not replace a prior completed collection silently.
        require(not target.exists(), f"Evidence already exists: {filename}; use a fresh directory")
        with target.open("xb") as stream:
            proc = subprocess.run(
                [*SSH, command], input=source, stdout=stream, stderr=subprocess.PIPE, check=False
            )
        require(
            proc.returncode == 0, f"Remote read-only {worker} worker failed; stderr suppressed (credentials)"
        )
        if worker == "snapshot":
            extract_snapshot(target, evidence)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("/tmp/lpwm-common40-completion-20260922"))
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument(
        "--worker", choices=("snapshot", "hashes", "benchmark", "cloud"), help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    try:
        if args.worker:
            root = Path(args.remote_root)
            if args.worker == "snapshot":
                with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as bundle:
                    for name, path in snapshot_files(root).items():
                        bundle.add(path, arcname=name, recursive=False)
            else:
                if args.worker == "hashes":
                    result = remote_hashes(root)
                elif args.worker == "benchmark":
                    result = benchmark_evidence(root)
                else:
                    result = cloud_readback(root, CREDENTIAL)
                print(json.dumps(result, sort_keys=True, allow_nan=False))
        else:
            if args.collect:
                collect(args.evidence, args.remote_root)
            report = audit(args.evidence, args.remote_root)
            print(json.dumps({"status": report["status"], "counts": report["counts"]}, indent=2))
        return 0
    except Exception as exc:
        # Do not disclose server/SDK exception text: it can contain request credentials.
        if args.worker:
            detail = type(exc).__name__ if args.worker == "cloud" else str(exc)
            print(f"Read-only {args.worker} worker failed: {detail}", file=sys.stderr)
        else:
            print(f"Audit FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
