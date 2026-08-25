#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source ROS on the host when this script is launched from a plain shell.
if [[ -z "${ROS_DISTRO:-}" && -f /opt/ros/jazzy/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
fi

CONFIG="${FRANKA_RECORDER_CONFIG:-${SCRIPT_DIR}/config.yaml}"
cd "${REPO_ROOT}"
exec uv run --project "${REPO_ROOT}" python "${SCRIPT_DIR}/record_franka_duo.py" \
  --config "${CONFIG}" "$@"
