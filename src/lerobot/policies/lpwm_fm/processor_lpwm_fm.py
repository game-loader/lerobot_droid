"""Normalize robot state, preserve native actions/RGB and cached frozen language."""

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    TransitionKey,
    batch_to_transition,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_lpwm_fm import LPWMFMConfig


def lpwm_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """Keep explicit GT world windows (the generic converter drops unknown keys)."""
    transition = batch_to_transition(batch)
    complementary = dict(transition[TransitionKey.COMPLEMENTARY_DATA] or {})
    complementary.update({key: value for key, value in batch.items() if key.startswith("world.")})
    if "current_step" in batch:
        complementary["current_step"] = batch["current_step"]
    transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
    return transition


@ProcessorStepRegistry.register(name="lpwm_fm_language_batch")
@dataclass
class LPWMFMLanguageBatchStep(ObservationProcessorStep):
    """Only batch unbatched [L,D] cached language; never synthesize embeddings."""

    def observation(self, observation):
        observation = dict(observation)
        key = "observation.language.embedding"
        if key in observation and observation[key].ndim == 2:
            observation[key] = observation[key].unsqueeze(0)
            for mask_key in ("observation.language.attention_mask", "observation.language.mask"):
                if mask_key in observation and observation[mask_key].ndim == 1:
                    observation[mask_key] = observation[mask_key].unsqueeze(0)
        return observation

    def transform_features(self, features):
        return features


def make_lpwm_fm_pre_post_processors(
    config: LPWMFMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Export processor for raw inputs; standalone training already processes tensors.

    World actions MUST arrive preprocessed in the same units as the policy actions
    (native LIBERO units with the default IDENTITY mapping). World RGB stays [0,1].
    World windows are preserved/device-moved, not normalized a second time. When
    restoring a serialized processor, pass ``lpwm_batch_to_transition`` as its
    ``to_transition`` argument if auxiliary world batches will be processed.
    """
    steps = make_default_policy_processor_steps(config, dataset_stats)
    pre, post = make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            LPWMFMLanguageBatchStep(),
            steps.to_device,
            steps.normalize,
        ],
        output_steps=[steps.unnormalize, steps.to_cpu],
    )
    pre.to_transition = lpwm_batch_to_transition
    return pre, post
