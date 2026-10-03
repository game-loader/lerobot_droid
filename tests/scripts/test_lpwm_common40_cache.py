"""Common40 conversion contracts, using only CPU and tiny embedded-PNG fixtures."""

import copy
import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from scripts.lpwm_ab.data import build_split
from scripts.lpwm_full.data import FullCachedDataset, PackedImages
from scripts.lpwm_full.prepare_common40 import (
    CAMERAS,
    SUITES,
    atomic,
    digest,
    encode_block,
    image_blocks,
    inspect_source,
    match_catalog,
    prepare,
    read_json,
    reuse_language,
    rgb_pair_preserved,
)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    reference = tmp_path / "reference"
    (reference / "cache").mkdir(parents=True)
    # Deliberately disagree with suite canonical order and language array row order.
    languages = ["task ten", "task goal", "task object", "task spatial"]
    reference_tasks = [
        {
            "global_task_id": official,
            "task_id": 0,
            "suite": suite,
            "name": language.replace(" ", "_"),
            "language": language,
            "bddl_file": f"{language}.bddl",
            "problem_folder": suite,
            "init_states_file": f"{language}.pruned_init",
            "demo": "raw.hdf5",
            "sha256": "not-local",
            "bytes": 999,
        }
        for official, suite, language in zip([120, 20, 10, 0], reversed(SUITES), languages, strict=True)
    ]
    catalog = {"tasks": reference_tasks}
    atomic(reference / "task_catalog.json", catalog)
    language_ids = [10, 0, 120, 20]
    embeddings = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
    np.save(reference / "cache/language_embeddings.npy", embeddings)
    np.save(reference / "cache/language_masks.npy", np.ones((4, 3), bool))
    atomic(
        reference / "cache/language_manifest.json",
        {
            "task_ids": language_ids,
            "hidden_dim": 5,
            "max_length": 3,
            "task_count": 4,
            "encoder": "fixture",
            "weights_sha256": "fixture-weights",
            "snapshot": "fixture",
        },
    )
    pq.write_table(
        pa.table({"task_index": np.arange(4), "__index_level_0__": languages}), root / "meta/tasks.parquet"
    )
    rows, episodes = [], []
    for eid, tid in enumerate([0, 1, 2, 3, 1, 0, 3, 2]):
        start = len(rows)
        length = 19 + eid
        for frame in range(length):
            row = {
                "index": len(rows),
                "episode_index": eid,
                "task_index": tid,
                "frame_index": frame,
                "observation.state": np.full(8, len(rows) + 0.25, np.float32).tolist(),
                "action": (np.arange(7, dtype=np.float32) + len(rows) * 10).tolist(),
            }
            for camera, key in enumerate(CAMERAS):
                image = np.zeros((256, 256, 3), np.uint8)
                image[:128, :128] = [eid * 25, frame * 5, 250 - camera * 100]
                image[128:, 128:] = [camera * 100, 255, 10]
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, format="PNG")
                row[key] = {"bytes": buffer.getvalue(), "path": "ignored.png"}
            rows.append(row)
        episodes.append(
            {
                "episode_index": eid,
                "dataset_from_index": start,
                "dataset_to_index": len(rows),
                "length": length,
                "tasks": [languages[tid]],
            }
        )
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    schema = pa.schema(
        [(key, pa.struct([("bytes", pa.binary()), ("path", pa.string())])) for key in CAMERAS]
        + [
            ("observation.state", pa.list_(pa.float32())),
            ("action", pa.list_(pa.float32())),
            ("index", pa.int64()),
            ("episode_index", pa.int64()),
            ("frame_index", pa.int64()),
            ("task_index", pa.int64()),
        ]
    )
    # A file boundary bisects both an episode and a 16-frame block.
    pq.write_table(pa.Table.from_pylist(rows[:23], schema=schema), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(rows[23:], schema=schema), root / "data/chunk-000/file-001.parquet")
    features = {k: {"dtype": "image", "shape": [256, 256, 3]} for k in CAMERAS}
    features.update(
        {
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [7]},
        }
    )
    atomic(
        root / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "fps": 10,
            "total_episodes": 8,
            "total_frames": len(rows),
            "total_tasks": 4,
            "features": features,
        },
    )
    return root, reference, rows, catalog


def run_fixture(source, output):
    root, reference, rows, _ = source
    return prepare(
        root,
        output,
        reference,
        workers=2,
        reserve_bytes=0,
        expected_counts=(8, len(rows), 4),
        expected_suites=dict.fromkeys(SUITES, 1),
    )


def test_roundtrip_rows_orientation_catalog_split_and_supervisor_hashes(source, tmp_path, capsys):
    root, reference, rows, _ = source
    output = tmp_path / "output"
    report = run_fixture(source, output)
    assert report["frames"] == len(rows)
    assert report["episodes"] == 8 and report["tasks"] == 4
    assert report["split"]["train"]["episodes"] == report["split"]["validation"]["episodes"] == 4
    manifest = read_json(output / "manifest.json")
    catalog = manifest["task_catalog"]["tasks"]
    assert [(t["global_task_id"], t["suite"], t["task_id"], t["source_task_index"]) for t in catalog] == [
        (i, suite, 0, 3 - i) for i, suite in enumerate(SUITES)
    ]
    assert "sha256" not in catalog[0]  # Do not mislabel raw HDF5 digest as local source provenance.
    assert read_json(output / "split.json") == build_split(manifest, 42, 0.1)
    np.testing.assert_array_equal(np.load(output / "states.npy"), [r["observation.state"] for r in rows])
    np.testing.assert_array_equal(np.load(output / "actions.npy"), [r["action"] for r in rows])
    np.testing.assert_array_equal(np.load(output / "source_task_index.npy"), [r["task_index"] for r in rows])
    np.testing.assert_array_equal(np.load(output / "task_index.npy"), [3 - r["task_index"] for r in rows])
    cached = PackedImages(output, manifest)
    for i, row in enumerate(rows):
        np.testing.assert_array_equal(cached[i], rgb_pair_preserved(row))
    assert not np.array_equal(cached[0], cached[0][:, :, ::-1, ::-1])
    for handle in cached.files.values():
        handle.close()
    original_language = np.load(reference / "cache/language_embeddings.npy")
    np.testing.assert_array_equal(
        np.load(output / "language_embeddings.npy"), original_language[[1, 0, 3, 2]]
    )
    for tid in range(4):
        folder = output / f"task_{tid:03d}"
        complete = read_json(folder / "complete.json")
        assert {"images.bin", "index.npz", "states.npy", "actions.npy"} <= complete["files_sha256"].keys()
        for name, sha256 in complete["files_sha256"].items():
            assert digest(folder / name) == sha256  # Identical to supervise's validation loop.
        with np.load(folder / "index.npz") as index:
            for ep in complete["episodes"]:
                assert index["frame_in_block"][ep["start"]] == 0
    dataset = FullCachedDataset(
        output, manifest["split"]["train_episode_ids"], read_json(output / "state_stats.json")
    )
    for i in (0, len(dataset) - 1):
        sample = dataset[i]
        anchor = int(dataset.anchors[i])
        np.testing.assert_array_equal(
            sample["action"].numpy(), np.load(output / "actions.npy")[anchor : anchor + 16]
        )
    for handle in dataset.images.files.values():
        handle.close()
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert len([m for m in messages if m["status"] == "task_complete"]) == 4
    assert report["all_pixels_verified_via_PackedImages"]


def test_existing_complete_cache_is_verified_not_rebuilt(source, tmp_path, monkeypatch):
    output = tmp_path / "output"
    first = run_fixture(source, output)

    def no_encoding(*args, **kwargs):
        pytest.fail("Resume tried to encode a completed task")

    monkeypatch.setattr("scripts.lpwm_full.prepare_common40.encode_block", no_encoding)
    assert run_fixture(source, output) == first
    packed = output / "task_000/images.bin"
    with packed.open("r+b") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        run_fixture(source, output)


def test_partial_resume_keeps_completed_tasks(source, tmp_path, monkeypatch):
    import scripts.lpwm_full.prepare_common40 as converter

    output = tmp_path / "output"
    original = converter.finish_task
    completed = []

    def fail_after_first(*args, **kwargs):
        result = original(*args, **kwargs)
        completed.append(result["task"]["global_task_id"])
        if len(completed) == 2:
            raise RuntimeError("injected interruption")
        return result

    monkeypatch.setattr(converter, "finish_task", fail_after_first)
    with pytest.raises(RuntimeError, match="interruption"):
        run_fixture(source, output)
    first = output / f"task_{completed[0]:03d}/complete.json"
    before = first.stat().st_mtime_ns
    monkeypatch.setattr(converter, "finish_task", original)
    report = run_fixture(source, output)
    assert report["frames"] == len(source[2]) and first.stat().st_mtime_ns == before


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("frame_index", 10, "Frame rows"),
        ("index", 10, "global row"),
        ("episode_index", 1, "Episode rows"),
        ("task_index", 2, "inside episode"),
        ("action", [float("nan")] * 7, "action values"),
        ("observation.state", [0.0] * 7, "row width"),
    ],
)
def test_rejects_scalar_misalignment_and_bad_values(source, field, value, match):
    root, _, rows, _ = source
    file = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(file)
    edited = table.to_pylist()
    edited[2][field] = value
    pq.write_table(pa.Table.from_pylist(edited, schema=table.schema), file)
    with pytest.raises(ValueError, match=match):
        inspect_source(root, expected_counts=(8, len(rows), 4))


def test_exact_language_mapping_rejects_fuzzy_or_ambiguous_match(source):
    _, _, _, reference = source
    tasks = {3 - i: reference["tasks"][i]["language"] for i in range(4)}
    expected = dict.fromkeys(SUITES, 1)
    changed = copy.deepcopy(reference)
    changed["tasks"][0]["language"] += " "
    with pytest.raises(ValueError, match="exact"):
        match_catalog(tasks, changed, expected_suites=expected)
    changed = copy.deepcopy(reference)
    changed["tasks"].append(changed["tasks"][0])
    with pytest.raises(ValueError, match="exact"):
        match_catalog(tasks, changed, expected_suites=expected)


def test_language_conflict_is_not_silently_reused(source, tmp_path):
    _, reference_root, _, reference = source
    tasks = {i: t["language"] for i, t in enumerate(reference["tasks"])}
    catalog = match_catalog(tasks, reference, expected_suites=dict.fromkeys(SUITES, 1))
    other = copy.deepcopy(reference)
    other["tasks"][0]["language"] = other["tasks"][1]["language"]
    with pytest.raises(ValueError, match="Conflicting|Missing"):
        reuse_language(tmp_path, catalog, other, reference_root / "cache")


def test_blocks_carry_across_file_boundaries_but_not_episodes(source):
    root, _, rows, _ = source
    blocks = list(image_blocks(sorted(root.glob("data/*/*.parquet"))))
    assert [r["index"] for block in blocks for r in block] == list(range(len(rows)))
    for block in blocks:
        assert len(block) <= 16 and len({r["episode_index"] for r in block}) == 1
    assert any(block[0]["index"] < 23 <= block[-1]["index"] for block in blocks)
    with pytest.raises(ValueError, match="crosses episode"):
        encode_block([rows[0], rows[-1]])


def test_source_changed_after_completion_fails_closed(source, tmp_path):
    output = tmp_path / "output"
    run_fixture(source, output)
    marker = output / "task_000/complete.json"
    original = marker.read_bytes()
    atomic(source[0] / "meta/extra.json", {"changed": True})
    with pytest.raises(ValueError, match="different source"):
        run_fixture(source, output)
    assert marker.read_bytes() == original


def test_disk_reserve_fails_before_committing_tasks(source, tmp_path):
    root, reference, rows, _ = source
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="disk reserve"):
        prepare(
            root,
            output,
            reference,
            workers=1,
            reserve_bytes=2**80,
            expected_counts=(8, len(rows), 4),
            expected_suites=dict.fromkeys(SUITES, 1),
        )
    assert not list(output.glob("task_*/complete.json"))
    assert not (output / "manifest.json").exists()
