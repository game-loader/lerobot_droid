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

from .configuration_dp3 import DP3Config
from .pointcloud import depth_to_point_cloud

__all__ = ["DP3Config", "DP3Policy", "depth_to_point_cloud", "make_dp3_pre_post_processors"]


def __getattr__(name: str):
    if name == "DP3Policy":
        from .modeling_dp3 import DP3Policy

        return DP3Policy
    if name == "make_dp3_pre_post_processors":
        from .processor_dp3 import make_dp3_pre_post_processors

        return make_dp3_pre_post_processors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
