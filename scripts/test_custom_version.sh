#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# The two OpenPI reference tests require gated PaliGemma access. All other
# tests in those files still run; no network access is needed for this suite.
exec uv run --no-sync pytest \
  tests/rl100 \
  tests/policies/test_dp3.py tests/policies/test_diffusion_state_only.py \
  tests/envs/test_moya_newton.py \
  tests/test_franka_duo_real_recorder.py tests/test_franka_duo_manual_recorder.py \
  tests/test_franka_duo_validator.py tests/test_franka_duo_eval.py \
  tests/scripts/test_smolvla_server.py tests/scripts/test_lerobot_eval_rendering.py \
  tests/scripts/test_custom_train_integration.py \
  tests/datasets/test_camera_cache.py tests/datasets/test_dataset_reader.py \
  tests/datasets/test_dataset_tools.py tests/configs tests/processor \
  tests/distributed/test_checkpoint_wrappers.py tests/distributed/test_parallel_dims_and_factory.py \
  tests/common/test_checkpoint_legacy_contracts.py tests/common/test_checkpoint_save_resume.py \
  tests/training/test_ema.py tests/scripts/test_lerobot_train_batch_preprocessing.py \
  tests/scripts/test_train_remote_dispatch.py \
  --deselect=tests/processor/test_pi0_processor.py::test_pi0_processor_inputs_match_openpi_reference \
  --deselect=tests/processor/test_pi05_processor.py::test_pi05_processor_inputs_match_openpi_reference \
  -q --maxfail=5 "$@"
