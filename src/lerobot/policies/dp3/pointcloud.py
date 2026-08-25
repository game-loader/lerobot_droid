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

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def _as_vec3(value: Sequence[float] | NDArray[np.floating] | None, name: str) -> NDArray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    return array


def depth_to_point_cloud(
    depth: NDArray,
    camera_matrix: Sequence[float] | NDArray,
    *,
    depth_scale: float = 1.0,
    rgb: NDArray | None = None,
    extrinsics: NDArray | None = None,
    workspace_min: Sequence[float] | NDArray[np.floating] | None = None,
    workspace_max: Sequence[float] | NDArray[np.floating] | None = None,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
    num_points: int | None = 512,
    seed: int | None = None,
) -> NDArray[np.float32]:
    """Deproject one depth image into a fixed-size XYZ or XYZRGB point set.

    ``camera_matrix`` accepts either ROS ``CameraInfo.k`` flattened row-major
    or a 3x3 matrix. ``extrinsics``, when supplied, transforms homogeneous
    camera optical-frame points into the desired stable training frame.
    """

    depth_array = np.asarray(depth)
    if depth_array.ndim != 2:
        raise ValueError(f"depth must have shape (height, width), got {depth_array.shape}.")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive.")
    if not 0 <= min_depth < max_depth:
        raise ValueError("Expected 0 <= min_depth < max_depth.")
    if num_points is not None and num_points <= 0:
        raise ValueError("num_points must be positive or None.")

    intrinsics = np.asarray(camera_matrix, dtype=np.float32)
    if intrinsics.size != 9:
        raise ValueError(f"camera_matrix must contain 9 values, got shape {intrinsics.shape}.")
    intrinsics = intrinsics.reshape(3, 3)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if not np.isfinite(intrinsics).all() or fx <= 0 or fy <= 0:
        raise ValueError("camera_matrix must contain finite positive focal lengths.")

    depth_m = depth_array.astype(np.float32, copy=False) * float(depth_scale)
    valid = np.isfinite(depth_m) & (depth_m >= min_depth) & (depth_m <= max_depth)
    rows, columns = np.nonzero(valid)
    z = depth_m[rows, columns]
    points = np.column_stack(((columns - cx) * z / fx, (rows - cy) * z / fy, z)).astype(
        np.float32,
        copy=False,
    )

    if extrinsics is not None:
        transform = np.asarray(extrinsics, dtype=np.float32)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"extrinsics must be a finite 4x4 matrix, got {transform.shape}.")
        points = points @ transform[:3, :3].T + transform[:3, 3]

    lower = _as_vec3(workspace_min, "workspace_min")
    upper = _as_vec3(workspace_max, "workspace_max")
    if lower is not None:
        keep = np.all(points >= lower, axis=1)
        points, rows, columns = points[keep], rows[keep], columns[keep]
    if upper is not None:
        keep = np.all(points <= upper, axis=1)
        points, rows, columns = points[keep], rows[keep], columns[keep]

    if points.shape[0] == 0:
        raise ValueError("No valid depth points remain after filtering.")

    features = points
    if rgb is not None:
        rgb_array = np.asarray(rgb)
        if rgb_array.shape != (*depth_array.shape, 3):
            raise ValueError(f"rgb must have shape {(*depth_array.shape, 3)}, got {rgb_array.shape}.")
        colors = rgb_array[rows, columns].astype(np.float32)
        if np.issubdtype(rgb_array.dtype, np.integer):
            colors /= np.iinfo(rgb_array.dtype).max
        features = np.concatenate((points, colors), axis=1)

    if num_points is not None and features.shape[0] > num_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(features.shape[0], size=num_points, replace=False)
        features = features[indices]
    elif num_points is not None and features.shape[0] < num_points:
        rng = np.random.default_rng(seed)
        padding = rng.choice(features.shape[0], size=num_points - features.shape[0], replace=True)
        indices = np.concatenate((np.arange(features.shape[0]), padding))
        rng.shuffle(indices)
        features = features[indices]

    return np.ascontiguousarray(features, dtype=np.float32)
