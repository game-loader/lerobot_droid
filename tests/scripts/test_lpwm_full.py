"""Common40/all130 coverage gates and causal lossless cache regression tests."""

import json
import pickle
import sys
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
from torch.utils.data import DataLoader

from scripts.lpwm_full.data import FullCachedDataset, PackedImages
from scripts.lpwm_full.evaluate import (
    COMMON_SUITES,
    SUITES,
    ExactLanguageCache,
    catalog_suites,
    checkpoint_catalog,
    suites_for_count,
    validate_result,
)

GB_INIT_SHA256 = "ea7fa2f61a64c37332c9bf1778b4077be0d0ab4f2174a9142210b230c9ea6840"


def test_duplicate_language_requires_identical_real_tokens():
    metadata = {
        "language": {"task_ids": [0, 1, 2]},
        "tasks": {"0": "pick bowl", "1": "pick bowl", "2": "open drawer"},
    }
    embeddings = np.zeros((3, 2, 4), dtype=np.float32)
    masks = np.ones((3, 2), dtype=bool)
    c = ExactLanguageCache(metadata, embeddings, masks, 4)
    assert len(c.lookup) == 2
    embeddings[1, 0, 0] = 1
    with pytest.raises(ValueError, match="Conflicting"):
        ExactLanguageCache(metadata, embeddings, masks, 4)


def result_fixture(preflight=False, task_count=130, explicit_protocol=False):
    suites = suites_for_count(task_count)
    tasks = []
    offset = 0
    for suite, n in suites.items():
        for i in range(1 if preflight else n):
            rows = [
                {"episode_index": j, "success": j % 2 == 0, "control_steps": 2}
                for j in range(1 if preflight else 10)
            ]
            tasks.append(
                {
                    "task_id": offset + i,
                    "suite": suite,
                    "suite_task_id": i,
                    "episodes": rows,
                    "successes": sum(x["success"] for x in rows),
                    "success_rate": sum(x["success"] for x in rows) / len(rows),
                }
            )
        offset += n
    protocol = {"preflight": preflight}
    if explicit_protocol or task_count == 40:
        protocol.update(
            suites=suites,
            task_count=len(tasks),
            scope_task_count=task_count,
            episodes_per_task=1 if preflight else 10,
            actual_preflight_max_steps=2 if preflight else None,
            formal_result_eligible=not preflight,
        )
    n = sum(len(x["episodes"]) for x in tasks)
    success = sum(x["successes"] for x in tasks)
    return {
        "status": "complete",
        "protocol": protocol,
        "num_episodes": n,
        "successes": success,
        "success_rate": success / n,
        "suite_macro_success_rate": success / n,
        "per_task": tasks,
        "per_suite": {
            s: {
                "success_rate": success / n,
                "num_episodes": (1 if preflight else num) * (1 if preflight else 10),
                "successes": (1 if preflight else num) * (1 if preflight else 5),
            }
            for s, num in suites.items()
        },
    }


def test_full1300episode_denominator_and_suitecoverage():
    result = result_fixture()
    m = validate_result(result)
    assert m["eval/num_episodes"] == 1300 and m["eval/successes"] == 650
    assert len(result["per_task"]) == 130
    result["per_task"][32]["suite_task_id"] = 0
    with pytest.raises(ValueError, match="identity"):
        validate_result(result)


def test_preflight_not_formal_and_no_spatial_only_acceptance():
    result = result_fixture(True)
    assert validate_result(result, True)["eval/num_episodes"] == 5
    with pytest.raises(ValueError):
        validate_result(result, False)
    result = result_fixture()
    result["per_task"] = result["per_task"][:10]
    with pytest.raises(ValueError):
        validate_result(result)


@pytest.fixture
def packed(tmp_path):
    n = 48
    pixels = np.random.default_rng(42).integers(0, 256, (n, 2, 3, 128, 128), dtype=np.uint8)
    offsets = []
    lengths = []
    frames = []
    within = []
    with (tmp_path / "images.bin").open("wb") as f:
        for start in range(0, n, 8):
            x = pixels[start : start + 8]
            d = x.copy()
            d[1:] = x[1:] ^ x[:-1]
            b = zlib.compress(d.tobytes(), 1)
            for i in range(len(x)):
                offsets.append(f.tell())
                lengths.append(len(b))
                frames.append(len(x))
                within.append(i)
            f.write(b)
    np.savez(
        tmp_path / "index.npz", offsets=offsets, lengths=lengths, block_frames=frames, frame_in_block=within
    )
    manifest = {
        "num_frames": n,
        "image_shards": [{"start": 0, "end": n, "path": "images.bin", "index": "index.npz"}],
        "episodes": [
            {"episode_index": 0, "start": 0, "end": 24, "task_index": 0},
            {"episode_index": 1, "start": 24, "end": 48, "task_index": 0},
        ],
        "language": {"task_ids": [0]},
        "cameras": ["observation.images.image", "observation.images.image2"],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for name, values in [
        ("states", np.zeros((n, 8), np.float32)),
        ("actions", np.arange(n * 7, dtype=np.float32).reshape(n, 7)),
        ("task_index", np.zeros(n, dtype=np.int64)),
        ("language_embeddings", np.zeros((1, 2, 4), np.float32)),
        ("language_masks", np.ones((1, 2), bool)),
    ]:
        np.save(tmp_path / (name + ".npy"), values)
    return tmp_path, manifest, pixels


def test_packed_pixels_and_pickle(packed):
    root, m, pixels = packed
    images = PackedImages(root, m)
    np.testing.assert_array_equal(images[6:11], pixels[6:11])
    restored = pickle.loads(pickle.dumps(images))
    np.testing.assert_array_equal(restored[47], pixels[47])
    with pytest.raises(IndexError):
        images[48]


def test_workers_cannot_share_zip_offsets_and_episode_alignment(packed):
    root, _, pixels = packed
    ds = FullCachedDataset(root, [0, 1], {"mean": [0] * 8, "std": [1] * 8})
    assert ds.anchors.tolist() == list(range(1, 9)) + list(range(25, 33))
    for _ in range(2):
        observed = []
        for batch in DataLoader(ds, batch_size=2, num_workers=2):
            observed.extend(batch["action"][:, 0, 0].tolist())
        assert observed == [float(t * 7) for t in ds.anchors]
    item = ds[0]
    np.testing.assert_allclose(item["observation.images.image"], pixels[0:2, 0] / 255, atol=1e-7)
    np.testing.assert_array_equal(item["world.actions"][0], np.arange(7))
    assert tuple(item["world.images"].shape) == (3, 2, 3, 128, 128)


def test_suite_macro_and_count_recomputed_not_trusted():
    result = result_fixture()
    result["per_suite"]["libero_90"]["num_episodes"] = 100
    with pytest.raises(ValueError, match="Suite denominator"):
        validate_result(result)
    result = result_fixture()
    result["suite_macro_success_rate"] = 0.9
    with pytest.raises(ValueError, match="macro"):
        validate_result(result)


@pytest.mark.parametrize("task_count", [40, 130])
@pytest.mark.parametrize("preflight", [False, True])
def test_protocol_scope_episode_denominators(task_count, preflight):
    result = result_fixture(preflight, task_count, explicit_protocol=True)
    expected_tasks = len(suites_for_count(task_count)) if preflight else task_count
    metrics = validate_result(result, preflight)
    assert metrics["eval/num_episodes"] == expected_tasks * (1 if preflight else 10)
    assert set(result["per_suite"]) == set(suites_for_count(task_count))
    assert len(result["per_task"]) == expected_tasks
    assert result["protocol"]["formal_result_eligible"] is not preflight
    if task_count == 40:
        assert [t["task_id"] for t in result["per_task"]] == (
            [0, 10, 20, 30] if preflight else list(range(40))
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("suites", SUITES),
        ("suites", {"libero_spatial": 40}),
        ("suites", {**COMMON_SUITES, "libero_10": 9}),
        ("task_count", 130),
        ("episodes_per_task", 1),
        ("scope_task_count", 130),
        ("formal_result_eligible", False),
        ("actual_preflight_max_steps", 2),
        ("preflight", True),
    ],
)
def test_common40_rejects_wrong_protocol(key, value):
    result = result_fixture(task_count=40)
    result["protocol"][key] = value
    with pytest.raises(ValueError):
        validate_result(result)


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "global_id", "suite", "episode", "denominator"]
)
def test_common40_strict_task_and_episode_coverage(corruption):
    result = result_fixture(task_count=40)
    if corruption == "missing":
        result["per_task"].pop()
    elif corruption == "duplicate":
        result["per_task"][39] = result["per_task"][38]
    elif corruption == "global_id":
        result["per_task"][30]["task_id"] = 120  # Old130 LIBERO10 IDs must be remapped.
    elif corruption == "suite":
        result["per_task"][30]["suite"] = "libero_90"
    elif corruption == "episode":
        result["per_task"][0]["episodes"][1]["episode_index"] = 0
    else:
        result["per_suite"]["libero_10"]["num_episodes"] = 10
    with pytest.raises(ValueError):
        validate_result(result)


def test_common40_preflight_is_four_short_nonformal_episodes():
    result = result_fixture(True, 40)
    assert validate_result(result, True)["eval/num_episodes"] == 4
    with pytest.raises(ValueError, match="preflight"):
        validate_result(result)
    result["per_task"][0]["episodes"][0]["control_steps"] = 3
    with pytest.raises(ValueError, match="outcomes"):
        validate_result(result, True)
    result = result_fixture(True, 40)
    result["protocol"]["actual_preflight_max_steps"] = 3
    with pytest.raises(ValueError, match="max_steps"):
        validate_result(result, True)


def catalog_fixture(task_count=40):
    tasks = []
    for suite, count in suites_for_count(task_count).items():
        for tid in range(count):
            tasks.append(
                {
                    "global_task_id": len(tasks),
                    "suite": suite,
                    "task_id": tid,
                    "name": f"{suite}_task_{tid}",
                    "language": f"instruction {suite} {tid}",
                }
            )
    return {"tasks": tasks}


def manifest_fixture(task_count=40):
    catalog = catalog_fixture(task_count)
    return {
        "schema": "lpwm_libero_full_zlib_v1",
        "image_codec": "xor16_zlib_rgb128_v1",
        "tasks": {str(t["global_task_id"]): t["language"] for t in catalog["tasks"]},
        "task_catalog": catalog,
        "episodes": [{"task_index": i} for i in range(task_count)],
        "scalar_sha256": {},
    }


@pytest.mark.parametrize("task_count", [40, 130])
def test_catalog_and_training_scope_inferred(task_count):
    from scripts.lpwm_full.train import validate_manifest

    manifest = manifest_fixture(task_count)
    batches = (task_count + 7) // 8
    assert validate_manifest(manifest, validation_batches=batches) == suites_for_count(task_count)
    with pytest.raises(ValueError, match="Offline validation"):
        validate_manifest(manifest, validation_batches=batches - 1)
    manifest["episodes"].pop()
    with pytest.raises(ValueError, match="actual episodes"):
        validate_manifest(manifest)
    assert validate_manifest(manifest, preflight=True) == suites_for_count(task_count)


@pytest.mark.parametrize("corruption", ["partial", "suite", "local_id", "global_id", "language", "preflight"])
def test_training_rejects_bad_common_manifest(corruption):
    from scripts.lpwm_full.train import validate_manifest

    manifest = manifest_fixture()
    task = manifest["task_catalog"]["tasks"][30]
    if corruption == "partial":
        manifest["task_catalog"]["tasks"].pop()
    elif corruption == "suite":
        task["suite"] = "libero_90"
    elif corruption == "local_id":
        task["task_id"] = 1
    elif corruption == "global_id":
        task["global_task_id"] = 120
    elif corruption == "language":
        manifest["tasks"]["30"] = "not the catalog language"
    else:
        manifest["preflight_only"] = True
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_checkpoint_catalog_preflight_fallback_only(tmp_path):
    checkpoint = tmp_path / "checkpoints/step_000002"
    checkpoint.mkdir(parents=True)
    (tmp_path / "task_catalog.json").write_text(json.dumps(catalog_fixture()))
    assert checkpoint_catalog(checkpoint, preflight=True)[1] == COMMON_SUITES
    with pytest.raises(FileNotFoundError):
        checkpoint_catalog(checkpoint)
    (checkpoint / "task_catalog.json").write_text(json.dumps(catalog_fixture(130)))
    assert checkpoint_catalog(checkpoint, preflight=True)[1] == SUITES
    assert catalog_suites(catalog_fixture()) == COMMON_SUITES


@pytest.mark.parametrize("task_count", [40, 130])
@pytest.mark.parametrize("preflight", [False, True])
def test_evaluation_uses_checkpoint_scope_without_simulator(tmp_path, monkeypatch, task_count, preflight):
    from scripts.lpwm_full import evaluate

    catalog = catalog_fixture(task_count)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "task_catalog.json").write_text(json.dumps(catalog))
    by_suite = {}
    for task in catalog["tasks"]:
        by_suite.setdefault(task["suite"], []).append(
            SimpleNamespace(name=task["name"], language=task["language"])
        )
    requested_suites = []
    caps = {name: 520 if name == "libero_10" else 280 for name in SUITES}
    env_calls = []

    def get_suite(name):
        requested_suites.append(name)
        return SimpleNamespace(get_task=lambda tid: by_suite[name][tid])

    class FakeEnv:
        def __init__(self, **kwargs):
            self._init_states = list(range(50))
            env_calls.append(kwargs)

        def close(self):
            pass

    def rollout(env, policy, plan, mapping, stats, language, device, max_steps, size):
        assert max_steps == (2 if preflight else env_calls[-1]["episode_length"])
        return {
            "episode_index": plan["episode_index"],
            "success": False,
            "control_steps": 2,
            "executed_action_components": 14,
            "clipped_action_components": 0,
        }

    monkeypatch.setitem(
        sys.modules,
        "lerobot.envs.libero",
        SimpleNamespace(
            TASK_SUITE_MAX_STEPS=caps,
            LiberoEnv=FakeEnv,
            _get_suite=get_suite,
        ),
    )
    monkeypatch.setattr(evaluate.native, "LanguageCache", evaluate.native.LanguageCache)
    monkeypatch.setattr(
        evaluate.native,
        "load_checkpoint",
        lambda *args: {
            "policy": object(),
            "checkpoint": {"step": 2 if preflight else 5000},
            "mapping": {},
            "state_stats": {},
            "language_cache": SimpleNamespace(select=lambda *args: (None, {})),
        },
    )
    monkeypatch.setattr(evaluate.native, "rollout_episode", rollout)
    output = tmp_path / "result.json"
    result = evaluate.run_evaluation(
        SimpleNamespace(
            checkpoint=checkpoint,
            output=output,
            device="cpu",
            preflight=preflight,
            seed_namespace="validation",
        )
    )
    assert requested_suites == list(suites_for_count(task_count))
    assert len(env_calls) == (len(requested_suites) if preflight else task_count)
    assert result["num_episodes"] == (len(requested_suites) if preflight else task_count * 10)
    assert result["protocol"]["scope_task_count"] == task_count
    assert result["protocol"]["formal_result_eligible"] is not preflight
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize("task_count", [40, 130])
def test_train_passes_inferred_scope_to_trainer(tmp_path, monkeypatch, task_count):
    from scripts.lpwm_full import train

    data = tmp_path / "cache"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps(manifest_fixture(task_count)))
    args = SimpleNamespace(
        data=data,
        output=tmp_path / "run",
        preflight=False,
        validation_batches=task_count,
        world_weight=1.0,
        rec_weight=1.0,
        dyn_weight=1.0,
        prior_weight=0.001,
    )
    base = SimpleNamespace()
    calls = []
    monkeypatch.setattr(train.trainer, "parse_args", lambda **kwargs: args)
    monkeypatch.setattr(train.trainer, "training_helpers", lambda: base)
    monkeypatch.setattr(train.trainer, "run_checkpoint_eval", train.trainer.run_checkpoint_eval)
    monkeypatch.setattr(train.trainer, "train", lambda options: calls.append(options))
    train.main()
    assert calls == [args]
    assert args.full_task_count == task_count
    assert args.full_suites == list(suites_for_count(task_count))
    assert base.LPWMCachedDataset is FullCachedDataset
    assert json.loads((args.output / "task_catalog.json").read_text()) == catalog_fixture(task_count)


def test_supervisor_defaults_and_batch_configuration():
    from scripts.lpwm_full.supervise import parse_args

    args = parse_args(["--root", "/unused"])
    assert (args.task_count, args.batch_size, args.grad_accumulation) == (130, 8, 4)
    assert str(args.train_python) == "/root/lpwm_ab/.venv/bin/python"
    assert str(args.eval_python) == "/root/lpwm_ab/.venv-eval/bin/python"
    assert str(args.libero_config) == "/root/lpwm_ab/libero_config"
    assert str(args.credential_file) == "/root/lpwm_ab/.swanlab_api_key"
    assert args.expected_init_sha256 == "3116e036247b84884b6c259de7bf5a5fbdfa6737a8e763c8a2157f5c01d2e13e"
    assert args.run_name is None and args.workers == 4
    for batch, accumulation in [(8, 4), (16, 2), (32, 1)]:
        args = parse_args(
            [
                "--root",
                "/unused",
                "--task-count",
                "40",
                "--batch-size",
                str(batch),
                "--grad-accumulation",
                str(accumulation),
                "--preflight-run",
                "preflight_common",
            ]
        )
        assert (args.task_count, args.batch_size, args.grad_accumulation) == (40, batch, accumulation)
        assert str(args.preflight_run) == "preflight_common"
    for extra in [["--batch-size", "16"], ["--task-count", "10"], ["--preflight-step", "0"]]:
        with pytest.raises(SystemExit):
            parse_args(["--root", "/unused", *extra])


@pytest.mark.parametrize("value", ["", "none", "0" * 63, "0" * 65, "g" * 64, " " + "0" * 63])
def test_supervisor_rejects_invalid_init_hash(value):
    from scripts.lpwm_full.supervise import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--root", "/unused", "--expected-init-sha256", value])


def test_supervisor_workers_and_hash_normalization():
    from scripts.lpwm_full.supervise import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--root", "/unused", "--workers", "-1"])
    args = parse_args(
        ["--root", "/unused", "--workers", "0", "--expected-init-sha256", GB_INIT_SHA256.upper()]
    )
    assert args.workers == 0 and args.expected_init_sha256 == GB_INIT_SHA256


def profile_fixture():
    return {
        "status": "passed",
        "batch_size": 8,
        "grad_accumulation": 4,
        "effective_batch": 32,
        "world_scale_step": 1000,
        "formal_training": False,
    }


def test_supervisor_profile_gate_requires_matching_passed_full_world_probe(tmp_path):
    from scripts.lpwm_full.supervise import parse_args, verify_profile

    args = parse_args(["--root", str(tmp_path), "--profile-gate", "profile8.json"])
    path = tmp_path / "profile8.json"
    path.write_text(json.dumps(profile_fixture()))
    verify_profile(tmp_path, args)
    for key, value in [
        ("status", "oom"),
        ("batch_size", 16),
        ("grad_accumulation", 2),
        ("world_scale_step", 1),
        ("formal_training", True),
        ("effective_batch", 16),
    ]:
        path.write_text(json.dumps({**profile_fixture(), key: value}))
        with pytest.raises(ValueError, match="Profile gate"):
            verify_profile(tmp_path, args)


@pytest.mark.parametrize("hash_style", ["basename", "relative", "explicit", "complete"])
def test_common_cache_hashes_without_mandatory_complete_json(tmp_path, hash_style):
    from scripts.lpwm_full.supervise import digest, verify_cache

    manifest = manifest_fixture()
    manifest["num_frames"] = 40
    folder = tmp_path / "common"
    folder.mkdir()
    hashes = {}
    for name in ["images.bin", "index.npz"]:
        (folder / name).write_bytes(b"test " + name.encode())
        hashes[name] = digest(folder / name)
    for name in ["states", "actions", "task_index", "language_embeddings", "language_masks"]:
        path = tmp_path / f"{name}.npy"
        path.write_bytes(b"scalar")
        manifest["scalar_sha256"][path.name] = digest(path)
    shard = {"start": 0, "end": 40, "path": "common/images.bin", "index": "common/index.npz"}
    if hash_style == "basename":
        shard["files_sha256"] = hashes
    elif hash_style == "relative":
        shard["files_sha256"] = {f"common/{name}": value for name, value in hashes.items()}
    elif hash_style == "explicit":
        shard.update(sha256=hashes["images.bin"], index_sha256=hashes["index.npz"])
    else:
        shard["files_sha256"] = hashes
        (folder / "complete.json").write_text(json.dumps({"files_sha256": hashes}))
    manifest["image_shards"] = [shard]
    verify_cache(tmp_path, manifest, 40)
    with pytest.raises(ValueError, match="task scope"):
        verify_cache(tmp_path, manifest)  # Legacy default remains 130.
    (folder / "index.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cache file"):
        verify_cache(tmp_path, manifest, 40)


def write_preflight_fixture(root, directory="preflight_common", task_count=40):
    from scripts.lpwm_full.supervise import digest

    preflight = root / directory
    (preflight / "eval").mkdir(parents=True)
    (preflight / "status.json").write_text(json.dumps({"status": "completed", "step": 2}))
    (preflight / "experiment.json").write_text(
        json.dumps(
            {
                "initial_weights_sha256": "3116e036247b84884b6c259de7bf5a5fbdfa6737a8e763c8a2157f5c01d2e13e",
            }
        )
    )
    result = result_fixture(True, task_count)
    result["checkpoint"] = {"step": 2}
    output = preflight / "eval/step_000002.json"
    output.write_text(json.dumps(result))
    receipt = {"verified_upload": True, "formal_result_eligible": False, "result_sha256": digest(output)}
    output.with_suffix(".swanlab.json").write_text(json.dumps(receipt))
    return output


def test_supervisor_preflight_gate_scope_receipt_and_configurable_path(tmp_path):
    from scripts.lpwm_full.supervise import parse_args, verify_preflight

    output = write_preflight_fixture(tmp_path)
    args = parse_args(
        [
            "--root",
            str(tmp_path),
            "--task-count",
            "40",
            "--preflight-run",
            "preflight_common",
        ]
    )
    verify_preflight(tmp_path, args)
    args.preflight_run = tmp_path / "preflight_common"
    verify_preflight(tmp_path, args)
    args.task_count = 130
    with pytest.raises(ValueError, match="task scope"):
        verify_preflight(tmp_path, args)
    args.task_count = 40
    result = json.loads(output.read_text())
    result["duration_seconds"] = 99
    output.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="changed result"):
        verify_preflight(tmp_path, args)


@pytest.mark.parametrize("recorded_hash", [GB_INIT_SHA256, "0" * 64, None])
def test_supervisor_portable_preflight_init_still_fails_closed(tmp_path, recorded_hash):
    from scripts.lpwm_full.supervise import parse_args, verify_preflight

    write_preflight_fixture(tmp_path, task_count=130)
    args = parse_args(["--root", str(tmp_path), "--preflight-run", "preflight_common"])
    experiment_path = tmp_path / "preflight_common/experiment.json"
    experiment = json.loads(experiment_path.read_text())
    args.expected_init_sha256 = GB_INIT_SHA256
    with pytest.raises(ValueError, match="Architecture/init"):
        verify_preflight(tmp_path, args)  # The old host's preflight cannot certify the new initialization.
    experiment["initial_weights_sha256"] = recorded_hash
    experiment_path.write_text(json.dumps(experiment))
    if recorded_hash == GB_INIT_SHA256:
        verify_preflight(tmp_path, args)
        args = parse_args(["--root", str(tmp_path), "--preflight-run", "preflight_common"])
    with pytest.raises(ValueError, match="Architecture/init"):
        verify_preflight(tmp_path, args)


@pytest.mark.parametrize("conflict", [None, "shard", "complete", "missing"])
def test_shared_image_shards_hash_each_path_and_completion_once(tmp_path, monkeypatch, conflict):
    from collections import Counter

    from scripts.lpwm_full import supervise

    manifest = manifest_fixture()
    manifest["num_frames"] = 42
    manifest["image_shards"] = []
    folder = tmp_path / "task_000"
    folder.mkdir()
    images = folder / "images.bin"
    images.write_bytes(b"shared large images file")
    image_hash = supervise.digest(images)
    hashes = {"images.bin": image_hash}
    for index in range(42):
        path = folder / f"episode_{index:06d}.npz"
        path.write_bytes(str(index).encode())
        hashes[path.name] = supervise.digest(path)
        manifest["image_shards"].append(
            {
                "start": index,
                "end": index + 1,
                "path": "task_000/images.bin",
                "index": f"task_000/{path.name}",
                "files_sha256": {"images.bin": image_hash, path.name: hashes[path.name]},
            }
        )
    for name in ["states", "actions", "task_index", "language_embeddings", "language_masks"]:
        path = tmp_path / f"{name}.npy"
        path.write_bytes(b"scalar")
        manifest["scalar_sha256"][path.name] = supervise.digest(path)
    complete = folder / "complete.json"
    if conflict == "shard":
        manifest["image_shards"][-1]["files_sha256"]["images.bin"] = "f" * 64
    elif conflict == "complete":
        # Complete record is checked first; make the first shard disagree with it.
        manifest["image_shards"][0]["files_sha256"]["images.bin"] = "f" * 64
    elif conflict == "missing":
        del manifest["image_shards"][0]["files_sha256"]["images.bin"]
    complete.write_text(json.dumps({"files_sha256": hashes}))
    digest_calls, read_calls = Counter(), Counter()
    original_digest, original_read = supervise.digest, supervise.read

    def tracked_digest(path):
        digest_calls[path] += 1
        return original_digest(path)

    def tracked_read(path):
        read_calls[path] += 1
        return original_read(path)

    monkeypatch.setattr(supervise, "digest", tracked_digest)
    monkeypatch.setattr(supervise, "read", tracked_read)
    if conflict:
        with pytest.raises(ValueError, match="missing hash" if conflict == "missing" else "Conflicting"):
            supervise.verify_cache(tmp_path, manifest, 40)
    else:
        supervise.verify_cache(tmp_path, manifest, 40)
    assert digest_calls[images] == 1
    assert read_calls[complete] == 1
    assert max(digest_calls.values()) == 1


def test_adapter_cache_is_preflight_only_not_production_common40():
    from scripts.lpwm_full.train import validate_manifest

    manifest = manifest_fixture()
    manifest.update(
        preflight_only=True,
        source_note="Old official Spatial task0 RGB with40-catalog remapped language; not common40 data",
    )
    manifest["episodes"] = [{"task_index": 0}]
    assert validate_manifest(manifest, preflight=True, validation_batches=1) == COMMON_SUITES
    with pytest.raises(ValueError, match="Production requires"):
        validate_manifest(manifest)
    manifest["episodes"] = [{"task_index": i} for i in range(40)]
    with pytest.raises(ValueError, match="Production requires"):
        validate_manifest(manifest)  # Marker alone still prohibits production.


@pytest.mark.parametrize("steps", [30000, 80000])
@pytest.mark.parametrize("task_count", [40, 130])
@pytest.mark.parametrize(
    "deployment",
    ["legacy", "portable", "external-egl", "external-cuda", "empty-cuda", "invalid-formal"],
)
def test_supervisor_forwards_scope_batch_and_checks_all_results(
    tmp_path, monkeypatch, task_count, steps, deployment
):
    from scripts.lpwm_full import supervise

    args = supervise.parse_args(
        [
            "--steps",
            str(steps),
            "--root",
            str(tmp_path),
            "--task-count",
            str(task_count),
            "--preflight-run",
            "preflight_adapter_run",
        ]
    )
    portable = deployment != "legacy"
    if portable:
        args = supervise.parse_args(
            [
                "--root",
                str(tmp_path),
                "--steps",
                str(steps),
                "--task-count",
                str(task_count),
                "--preflight-run",
                "preflight_adapter_run",
                "--batch-size",
                "32",
                "--grad-accumulation",
                "1",
                "--workers",
                "0",
                "--run-name",
                "gb200-fresh",
                "--train-python",
                str(tmp_path / ".venv/bin/python"),
                "--eval-python",
                str(tmp_path / "eval-env/.venv/bin/python"),
                "--libero-config",
                str(tmp_path / "libero_config"),
                "--credential-file",
                str(tmp_path / ".swanlab_api_key"),
                "--expected-init-sha256",
                GB_INIT_SHA256,
                "--profile-gate",
                "profile32.json",
            ]
        )
        (tmp_path / "profile32.json").write_text(
            json.dumps({**profile_fixture(), "batch_size": 32, "grad_accumulation": 1})
        )
    graphics_keys = ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID", "MUJOCO_GL", "PYOPENGL_PLATFORM")
    for key in graphics_keys:
        monkeypatch.delenv(key, raising=False)
    external_env = {}
    if deployment == "external-egl":
        external_env = {
            "CUDA_VISIBLE_DEVICES": "GPU-test-uuid",
            "MUJOCO_EGL_DEVICE_ID": "3",
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
        }
    elif deployment == "external-cuda":
        external_env = {"CUDA_VISIBLE_DEVICES": "GPU-test-uuid"}
    elif deployment == "empty-cuda":
        external_env = {"CUDA_VISIBLE_DEVICES": ""}
    for key, value in external_env.items():
        monkeypatch.setenv(key, value)
    write_preflight_fixture(tmp_path, directory="preflight_adapter_run", task_count=task_count)
    (tmp_path / "preflight_adapter_run/experiment.json").write_text(
        json.dumps({"initial_weights_sha256": args.expected_init_sha256, "cosine_endpoint_step": steps})
    )
    (tmp_path / "cache").mkdir()
    manifest = manifest_fixture(task_count)
    manifest["num_frames"] = 1000
    (tmp_path / "cache/manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "frozen_source_sha256.json").write_text("{}")
    monkeypatch.setattr(supervise, "parse_args", lambda: args)
    monkeypatch.setattr(supervise, "verify_cache", lambda *args: None)
    monkeypatch.setattr(supervise.shutil, "disk_usage", lambda root: SimpleNamespace(free=10 * 2**30))
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        run = tmp_path / "run"
        (run / "eval").mkdir(parents=True)
        (run / "status.json").write_text(json.dumps({"status": "completed", "step": steps}))
        for step in range(5000, steps + 1, 5000):
            result = result_fixture(task_count=task_count)
            result["checkpoint"] = {"step": step}
            if deployment == "invalid-formal" and step == steps:
                result["protocol"]["preflight"] = True
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
        return SimpleNamespace(pid=12345, poll=lambda: 0, returncode=0)

    monkeypatch.setattr(supervise.subprocess, "Popen", fake_popen)
    if deployment == "invalid-formal":
        with pytest.raises(ValueError, match="preflight marker"):
            supervise.main()
        assert json.loads((tmp_path / "pipeline_status.json").read_text())["status"] == "failed"
        return
    supervise.main()
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0] == str(args.train_python)
    for flag, value in (
        ("--eval-python", args.eval_python),
        ("--credential-file", args.credential_file),
        ("--expected-init-sha256", args.expected_init_sha256),
        ("--workers", args.workers),
        ("--world-weight", 1),
        ("--rec-weight", 1),
        ("--dyn-weight", 1),
        ("--prior-weight", "0.001"),
    ):
        assert command[command.index(flag) + 1] == str(value)
    expected_name = (
        "gb200-fresh"
        if portable
        else (f"{'common40' if task_count == 40 else 'full130'}-B-w1-rec1-dyn1-cos{steps // 1000}k-seed42")
    )
    assert command[command.index("--run-name") + 1] == expected_name
    assert not {"--preflight", "--resume", "--disable-eval", "--eval-task-ids"}.intersection(command)
    env = kwargs["env"]
    assert kwargs["cwd"] == tmp_path / "repo"
    assert env["LIBERO_CONFIG_PATH"] == str(args.libero_config)
    assert env["PYTHONPATH"] == f"{tmp_path}/repo:{tmp_path}/repo/src"
    for key, value in external_env.items():
        assert env[key] == value
    if "CUDA_VISIBLE_DEVICES" not in external_env:
        assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["MUJOCO_EGL_DEVICE_ID"] == "0"
    elif "MUJOCO_EGL_DEVICE_ID" not in external_env:
        assert "MUJOCO_EGL_DEVICE_ID" not in env
    assert env["MUJOCO_GL"] == external_env.get("MUJOCO_GL", "egl")
    assert env["PYOPENGL_PLATFORM"] == external_env.get("PYOPENGL_PLATFORM", "egl")
    assert command[command.index("--steps") + 1] == str(steps)
    assert command[command.index("--schedule-steps") + 1] == str(steps)
    assert command[command.index("--batch-size") + 1] == ("32" if portable else "8")
    assert command[command.index("--grad-accumulation") + 1] == ("1" if portable else "4")
    assert command[command.index("--validation-batches") + 1] == str(task_count)
    status = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert status["status"] == "completed"
    assert status["task_count"] == task_count
    assert status["evaluated_episodes"] == task_count * 10 * (steps // 5000)


def test_supervisor_accepts_explicit_80k_budget(tmp_path):
    from scripts.lpwm_full.supervise import parse_args

    args = parse_args(["--root", str(tmp_path), "--steps", "80000", "--task-count", "130"])
    assert args.steps == 80000 and args.task_count == 130
    assert args.batch_size == 8 and args.grad_accumulation == 4
    assert parse_args(["--root", str(tmp_path)]).steps == 30000


def test_cache_overflow_requires_explicit_trusted_root(tmp_path):
    from scripts.lpwm_full.supervise import digest, verify_cache

    cache = tmp_path / "cache"
    cache.mkdir()
    external = tmp_path / "approved_overflow"
    external.mkdir()
    (cache / "task").symlink_to(external, target_is_directory=True)
    manifest = manifest_fixture()
    manifest["num_frames"] = 40
    for name in ["states", "actions", "task_index", "language_embeddings", "language_masks"]:
        p = cache / f"{name}.npy"
        p.write_bytes(b"scalar")
        manifest["scalar_sha256"][p.name] = digest(p)
    hashes = {}
    for name in ["images.bin", "index.npz"]:
        (external / name).write_bytes(b"test" + name.encode())
        hashes[name] = digest(external / name)
    (external / "complete.json").write_text(json.dumps({"files_sha256": hashes}))
    manifest["image_shards"] = [
        {"start": 0, "end": 40, "path": "task/images.bin", "index": "task/index.npz", "files_sha256": hashes}
    ]
    with pytest.raises(ValueError, match="outside cache root"):
        verify_cache(cache, manifest, 40)
    verify_cache(cache, manifest, 40, storage_roots=[external])
    with pytest.raises(ValueError, match="entire filesystem"):
        verify_cache(cache, manifest, 40, storage_roots=["/"])
    (external / "images.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cache file"):
        verify_cache(cache, manifest, 40, storage_roots=[external])


def test_80k_preflight_schedule_must_match_requested_production(tmp_path):
    from scripts.lpwm_full.supervise import parse_args, verify_preflight

    write_preflight_fixture(tmp_path, task_count=130)
    args = parse_args(["--root", str(tmp_path), "--steps", "80000", "--preflight-run", "preflight_common"])
    with pytest.raises(ValueError, match="cosine endpoint"):
        verify_preflight(tmp_path, args)
    path = tmp_path / "preflight_common/experiment.json"
    data = json.loads(path.read_text())
    data["cosine_endpoint_step"] = 80000
    path.write_text(json.dumps(data))
    verify_preflight(tmp_path, args)
