"""No network/GPU: fail-closed manifest-last transfer commit tests."""

import hashlib
import json

import pytest

from scripts.lpwm_full.await_transfer import transferred


def verified(tmp_path):
    payload = b'{"manifest":"test"}'
    (tmp_path / "manifest.json").write_bytes(payload)
    report = {
        "status": "complete",
        "tasks": 40,
        "frames": 273465,
        "episodes": 1693,
        "all_source_scalars_exact": True,
        "all_pixels_verified_via_PackedImages": True,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }
    (tmp_path / "verification.json").write_text(json.dumps(report))
    return report


def test_requires_verified_exact_manifest(tmp_path):
    assert not transferred(tmp_path)
    verified(tmp_path)
    assert transferred(tmp_path)
    (tmp_path / "manifest.json").write_text('{"partial":')
    assert not transferred(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "verifying"),
        ("tasks", 130),
        ("frames", 2),
        ("episodes", 1),
        ("all_source_scalars_exact", False),
        ("all_pixels_verified_via_PackedImages", False),
    ],
)
def test_reject_wrong_scope_or_unverified_cache(tmp_path, field, value):
    report = verified(tmp_path)
    report[field] = value
    (tmp_path / "verification.json").write_text(json.dumps(report))
    assert not transferred(tmp_path)


def test_partial_verification_file_not_ready(tmp_path):
    verified(tmp_path)
    (tmp_path / "verification.json").write_text("{")
    assert not transferred(tmp_path)
