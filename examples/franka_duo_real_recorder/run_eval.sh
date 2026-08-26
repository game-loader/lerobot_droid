#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ROS and camera drivers stay on the host; this script only starts the Python
# evaluator and never creates a container.
if [[ -z "${ROS_DISTRO:-}" && -f /opt/ros/jazzy/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
fi

CONFIG="${FRANKA_EVAL_CONFIG:-${SCRIPT_DIR}/eval_config.yaml}"
if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/model_bundle [eval options...]" >&2
  exit 2
fi
BUNDLE="$1"
shift
cd "${REPO_ROOT}"
exec uv run --project "${REPO_ROOT}" python -m examples.franka_duo_real_recorder.eval_franka_duo \
  --bundle "${BUNDLE}" --config "${CONFIG}" "$@"
