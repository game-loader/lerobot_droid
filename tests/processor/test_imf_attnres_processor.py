#!/usr/bin/env python

"""Minimal RED tests for IMF-AttnRes policy processors."""

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import make_policy_config, make_pre_post_processors
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    EnvTransition,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    ProcessorStep,
    RenameObservationsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

POLICY_NAME = "imf-attnres"
STATE_DIM = 8
ACTION_DIM = 8
IMAGE_SIZE = 16
IMAGE_KEYS = (
    f"{OBS_IMAGES}.agentview",
    f"{OBS_IMAGES}.eye_in_hand",
)


def make_tiny_imf_attnres_config():
    config = make_policy_config(
        POLICY_NAME,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=2,
        n_layer=1,
        n_emb=32,
        spatial_softmax_num_keypoints=4,
        pretrained_backbone_weights=None,
        push_to_hub=False,
    )
    config.device = "cpu"
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        **{
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE))
            for key in IMAGE_KEYS
        },
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    return config


def make_dataset_stats():
    return {
        OBS_STATE: {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)},
        ACTION: {"min": torch.full((ACTION_DIM,), -1.0), "max": torch.ones(ACTION_DIM)},
        **{key: {} for key in IMAGE_KEYS},
    }


def make_single_transition_batch():
    observation = {
        OBS_STATE: torch.randn(STATE_DIM),
        **{key: torch.rand(3, IMAGE_SIZE, IMAGE_SIZE) for key in IMAGE_KEYS},
    }
    action = torch.randn(ACTION_DIM).clamp(-1, 1)
    return transition_to_batch(create_transition(observation, action))


def test_make_imf_attnres_processor_basic():
    """Factory should create the standard policy pre/post processor pipelines."""
    config = make_tiny_imf_attnres_config()
    stats = make_dataset_stats()

    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)

    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"

    assert len(preprocessor.steps) == 4
    assert isinstance(preprocessor.steps[0], RenameObservationsProcessorStep)
    assert isinstance(preprocessor.steps[1], AddBatchDimensionProcessorStep)
    assert isinstance(preprocessor.steps[2], DeviceProcessorStep)
    assert isinstance(preprocessor.steps[3], NormalizerProcessorStep)

    assert len(postprocessor.steps) == 2
    assert isinstance(postprocessor.steps[0], UnnormalizerProcessorStep)
    assert isinstance(postprocessor.steps[1], DeviceProcessorStep)


def test_imf_attnres_processor_handles_libero_like_multicamera_observation():
    """Preprocessor should batch/device/normalize LIBERO-like state, action, and multi-camera images."""
    config = make_tiny_imf_attnres_config()
    stats = make_dataset_stats()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)

    batch = make_single_transition_batch()
    processed = preprocessor(batch)

    assert processed[OBS_STATE].shape == (1, STATE_DIM)
    for key in IMAGE_KEYS:
        assert processed[key].shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert torch.allclose(processed[key][0], batch[key], rtol=1e-5)
    assert processed[TransitionKey.ACTION.value].shape == (1, ACTION_DIM)

    policy_action = torch.zeros(1, ACTION_DIM)
    postprocessed_action = postprocessor(policy_action)
    assert postprocessed_action.shape == (1, ACTION_DIM)
    assert postprocessed_action.device.type == "cpu"


def test_make_imf_attnres_processor_with_smolvlm_vl_encoder(monkeypatch):
    """SmolVLM-enabled IMF-AttnRes should tokenize task text using the SmolVLA-style flow."""

    class DummyTokenizerProcessorStep(ProcessorStep):
        def __init__(self, tokenizer_name, padding, padding_side, max_length, truncation):
            self.tokenizer_name = tokenizer_name
            self.padding = padding
            self.padding_side = padding_side
            self.max_length = max_length
            self.truncation = truncation

        def __call__(self, transition: EnvTransition) -> EnvTransition:
            return transition

        def transform_features(self, features):
            return features

    import lerobot.policies.imf_attnres.processor_imf_attnres as processor_imf_attnres

    monkeypatch.setattr(processor_imf_attnres, "TokenizerProcessorStep", DummyTokenizerProcessorStep)

    config = make_tiny_imf_attnres_config()
    config.use_smolvlm_vl_encoder = True
    config.vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    config.vlm_pad_language_to = "max_length"
    config.vlm_tokenizer_padding_side = "right"
    config.vlm_tokenizer_max_length = 48
    config.vlm_tokenizer_truncation = True
    stats = make_dataset_stats()

    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)

    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"

    assert len(preprocessor.steps) == 6
    assert isinstance(preprocessor.steps[0], RenameObservationsProcessorStep)
    assert isinstance(preprocessor.steps[1], AddBatchDimensionProcessorStep)
    assert isinstance(preprocessor.steps[2], NewLineTaskProcessorStep)
    assert isinstance(preprocessor.steps[3], DummyTokenizerProcessorStep)
    assert preprocessor.steps[3].tokenizer_name == config.vlm_model_name
    assert preprocessor.steps[3].padding == config.vlm_pad_language_to
    assert preprocessor.steps[3].padding_side == config.vlm_tokenizer_padding_side
    assert preprocessor.steps[3].max_length == config.vlm_tokenizer_max_length
    assert preprocessor.steps[3].truncation == config.vlm_tokenizer_truncation
    assert isinstance(preprocessor.steps[4], DeviceProcessorStep)
    assert isinstance(preprocessor.steps[5], NormalizerProcessorStep)

    assert len(postprocessor.steps) == 2
    assert isinstance(postprocessor.steps[0], UnnormalizerProcessorStep)
    assert isinstance(postprocessor.steps[1], DeviceProcessorStep)


def test_smolvlm_processor_forces_visual_identity_normalization(monkeypatch):
    """SmolVLM expects policy images to stay in [0, 1] before its internal [-1, 1] conversion."""

    class DummyTokenizerProcessorStep(ProcessorStep):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, transition: EnvTransition) -> EnvTransition:
            return transition

        def transform_features(self, features):
            return features

    import lerobot.policies.imf_attnres.processor_imf_attnres as processor_imf_attnres

    monkeypatch.setattr(processor_imf_attnres, "TokenizerProcessorStep", DummyTokenizerProcessorStep)

    config = make_tiny_imf_attnres_config()
    config.use_smolvlm_vl_encoder = True
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    stats = {
        OBS_STATE: {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)},
        ACTION: {"min": torch.full((ACTION_DIM,), -1.0), "max": torch.ones(ACTION_DIM)},
        **{
            key: {
                "mean": torch.tensor([0.485, 0.456, 0.406]),
                "std": torch.tensor([0.229, 0.224, 0.225]),
            }
            for key in IMAGE_KEYS
        },
    }

    preprocessor, _ = make_pre_post_processors(config, dataset_stats=stats)
    normalizer = preprocessor.steps[-1]

    assert isinstance(normalizer, NormalizerProcessorStep)
    assert normalizer.norm_map[FeatureType.VISUAL] == NormalizationMode.IDENTITY
    assert normalizer.norm_map[FeatureType.STATE] == NormalizationMode.MEAN_STD
    assert normalizer.norm_map[FeatureType.ACTION] == NormalizationMode.MIN_MAX

    batch = make_single_transition_batch()
    processed = preprocessor(batch)
    for key in IMAGE_KEYS:
        assert torch.allclose(processed[key][0], batch[key], rtol=1e-5)
