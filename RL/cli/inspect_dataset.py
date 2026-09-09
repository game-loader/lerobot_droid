# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inspect sparse-reward decision construction for a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from RL.adapters.lerobot_v3 import LeRobotV3DecisionDataset
from RL.config import RLConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--state-dim", type=int, default=39)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = RLConfig(
        state_key=args.state_key,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        n_obs_steps=args.n_obs_steps,
        chunk_size=args.chunk_size,
        gamma=args.gamma,
    )
    dataset = LeRobotV3DecisionDataset.from_root(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        summary_path=args.summary,
        config=config,
    )
    print(json.dumps(dataset.inspection_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
