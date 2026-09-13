"""CPU integration coverage for the custom training lifecycle and processors."""

import json
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("datasets", exc_type=ModuleNotFoundError)
pytest.importorskip("accelerate", exc_type=ModuleNotFoundError)
pytest.importorskip("diffusers", exc_type=ModuleNotFoundError)

from lerobot.envs import MoyaNewtonEnvConfig
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.scripts import lerobot_train
from lerobot.utils.eval_provenance import EVAL_PROVENANCE_FILENAME, sha256_tree
from tests.training.test_ema import make_dummy_dataset, make_train_config


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    return make_dummy_dataset(tmp_path_factory.mktemp("custom-train-data"))


class TrackedEnv:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("fail_eval", [False, True])
def test_train_eval_environment_lifecycle(dataset_root, tmp_path, monkeypatch, reuse, fail_eval):
    cfg = make_train_config(dataset_root, tmp_path / "output", steps=2, ema_enable=False)
    cfg.env = MoyaNewtonEnvConfig(device="cpu")
    cfg.env.supports_eval_env_reuse = reuse
    cfg.env_eval_freq = 1
    cfg.eval.n_episodes = 1
    cfg.eval.batch_size = 1
    cfg.save_freq = 1
    cfg.wandb.enable = True
    cfg.wandb.project = "test-no-network"
    logger = Mock()
    monkeypatch.setattr(lerobot_train, "WandBLogger", lambda _: logger)
    monkeypatch.setattr(lerobot_train, "make_env_pre_post_processors", lambda **_: (None, None))
    created, evaluated = [], []

    def make_env(*args, **kwargs):
        env = TrackedEnv()
        created.append(env)
        return {"moya_newton": {0: env}}

    def evaluate(**kwargs):
        env = kwargs["envs"]["moya_newton"][0]
        assert env.close_calls == 0
        assert kwargs["max_episodes_rendered"] == 0
        assert kwargs["close_envs_after_eval"] is not reuse
        evaluated.append(env)
        if fail_eval and len(evaluated) == 2:
            raise RuntimeError("evaluation failed")
        return {
            "overall": {
                "avg_sum_reward": 1.0,
                "pc_success": 100.0,
                "eval_s": 0.01,
                "video_paths": [],
            }
        }

    monkeypatch.setattr(lerobot_train, "make_env", make_env)
    monkeypatch.setattr(lerobot_train, "eval_policy_all", evaluate)
    if fail_eval:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            lerobot_train.train(cfg)
    else:
        lerobot_train.train(cfg)

    assert len(evaluated) == 2
    assert len(created) == (1 if reuse else 2)
    assert all(env.close_calls == 1 for env in created)
    logger.log_video.assert_not_called()
    saved_evals = sorted((cfg.output_dir / "eval").glob("step_*/eval_info.json"))
    assert len(saved_evals) == (1 if fail_eval else 2)
    for path in saved_evals:
        assert json.loads(path.read_text())["overall"]["pc_success"] == 100.0
        provenance = json.loads((path.parent / EVAL_PROVENANCE_FILENAME).read_text())
        step_id = path.parent.name.removeprefix("step_")
        checkpoint_path = cfg.output_dir / "checkpoints" / step_id / "pretrained_model"
        assert provenance["checkpoint"] == str(checkpoint_path.resolve())
        assert provenance["checkpoint_sha256"] == sha256_tree(checkpoint_path)
        assert provenance["env_type"] == "moya_newton"


@pytest.fixture
def checkpoint(dataset_root, tmp_path):
    cfg = make_train_config(dataset_root, tmp_path / "initial", steps=1, ema_enable=False)
    lerobot_train.train(cfg)
    return cfg.output_dir / "checkpoints" / "000001" / "pretrained_model"


def _normalization_stats(pipeline, step_type):
    step = next(step for step in pipeline.steps if isinstance(step, step_type))
    return {
        feature: {name: torch.as_tensor(value).clone() for name, value in values.items()}
        for feature, values in step.stats.items()
    }


@pytest.mark.parametrize("preserve", [False, True])
def test_finetuning_preserves_or_replaces_saved_stats(
    dataset_root, tmp_path, monkeypatch, checkpoint, preserve
):
    cfg = make_train_config(dataset_root, tmp_path / "finetuned", steps=1, ema_enable=False)
    cfg.policy.pretrained_path = checkpoint
    cfg.preserve_pretrained_processor_stats = preserve
    factory = lerobot_train.make_pre_post_processors
    original_pre, original_post = factory(cfg.policy, pretrained_path=checkpoint)
    saved = (
        _normalization_stats(original_pre, NormalizerProcessorStep),
        _normalization_stats(original_post, UnnormalizerProcessorStep),
    )
    original_make_dataset = lerobot_train.make_train_eval_datasets
    replacement_stats, captured = {}, []

    def make_dataset(train_cfg):
        dataset, validation = original_make_dataset(train_cfg)
        for key in ("observation.state", "action"):
            for statistic, value in dataset.meta.stats[key].items():
                if statistic != "count":
                    dataset.meta.stats[key][statistic] = torch.as_tensor(value).numpy() + 10.0
        replacement_stats.update(dataset.meta.stats)
        return dataset, validation

    def capture_processors(*args, **kwargs):
        pre, post = factory(*args, **kwargs)
        captured.extend(
            [
                _normalization_stats(pre, NormalizerProcessorStep),
                _normalization_stats(post, UnnormalizerProcessorStep),
            ]
        )
        return pre, post

    monkeypatch.setattr(lerobot_train, "make_train_eval_datasets", make_dataset)
    monkeypatch.setattr(lerobot_train, "make_pre_post_processors", capture_processors)
    lerobot_train.train(cfg)

    assert len(captured) == 2
    if preserve:
        torch.testing.assert_close(captured, list(saved), rtol=0, atol=0)
    for index, actual in enumerate(captured):
        expected = saved[index] if preserve else replacement_stats
        for key in ("observation.state", "action"):
            if key not in actual:
                continue
            for statistic in ("min", "max"):
                torch.testing.assert_close(
                    actual[key][statistic], torch.as_tensor(expected[key][statistic]), rtol=0, atol=0
                )
    assert not torch.equal(saved[0]["action"]["min"], torch.as_tensor(replacement_stats["action"]["min"]))


@pytest.mark.parametrize("preserve", [False, True])
def test_resume_helper_does_not_override_saved_stats(checkpoint, preserve):
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(checkpoint)
    original_pre, original_post = lerobot_train.make_pre_post_processors(config, pretrained_path=checkpoint)
    pre_kwargs, post_kwargs = lerobot_train._policy_processor_factory_kwargs(
        preserve_pretrained_processor_stats=preserve,
        processor_pretrained_path=checkpoint,
        resume=True,
        dataset_stats={"action": {"min": [999.0], "max": [1000.0]}},
        device_type="cpu",
        rename_map={},
        input_features=config.input_features,
        output_features=config.output_features,
        normalization_mapping=config.normalization_mapping,
    )
    assert "dataset_stats" not in pre_kwargs
    for overrides in (
        pre_kwargs.get("preprocessor_overrides", {}),
        post_kwargs.get("postprocessor_overrides", {}),
    ):
        assert all("stats" not in value for value in overrides.values())
    loaded_pre, loaded_post = lerobot_train.make_pre_post_processors(
        config, pretrained_path=checkpoint, **pre_kwargs, **post_kwargs
    )
    for original, loaded, step_type in (
        (original_pre, loaded_pre, NormalizerProcessorStep),
        (original_post, loaded_post, UnnormalizerProcessorStep),
    ):
        expected = _normalization_stats(original, step_type)
        actual = _normalization_stats(loaded, step_type)
        for feature, values in expected.items():
            for statistic, value in values.items():
                torch.testing.assert_close(actual[feature][statistic], value, rtol=0, atol=0)
