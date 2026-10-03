"""Raw-RGB processors for the encoder-only LPWM-IMF stage."""

from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_lpwm_imf import LPWMIMFConfig


def make_lpwm_imf_pre_post_processors(
    config: LPWMIMFConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Batch/move observations without ImageNet normalization; resizing lives in the encoder.

    The postprocessor is only a factory-compatible identity/device adapter. It
    does not make action prediction available in this encoder-only policy.
    """
    return (
        PolicyProcessorPipeline(
            steps=[
                RenameObservationsProcessorStep(rename_map={}),
                AddBatchDimensionProcessorStep(),
                DeviceProcessorStep(device=config.device),
            ],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline(
            steps=[DeviceProcessorStep(device="cpu")],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
