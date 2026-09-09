"""Compatibility of saved custom-branch configs with current training config."""

import json

import pytest

from lerobot.configs.train import TrainPipelineConfig, _migrate_legacy_eval_frequency


def test_legacy_eval_frequency_preserves_original_mapping():
    original = {"eval_freq": 5000, "steps": 10000}
    assert _migrate_legacy_eval_frequency(original) == {"env_eval_freq": 5000, "steps": 10000}
    assert original == {"eval_freq": 5000, "steps": 10000}


def test_conflicting_evaluation_frequency_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        _migrate_legacy_eval_frequency({"eval_freq": 5, "env_eval_freq": 10})


def test_checkpoint_config_migrates_legacy_eval_frequency(tmp_path):
    (tmp_path / "train_config.json").write_text(
        json.dumps(
            {
                "dataset": {"repo_id": "local/example"},
                "eval_freq": 1234,
                "preserve_pretrained_processor_stats": False,
            }
        )
    )
    cfg = TrainPipelineConfig.from_pretrained(tmp_path)
    assert cfg.env_eval_freq == 1234
