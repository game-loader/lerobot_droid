#!/usr/bin/env python3
"""Explicit model-bundle loading for the Franka Duo real-robot evaluator.

There are two intentionally separate backends:

``lerobot``
    A LeRobot ``pretrained_model`` directory containing ``config.json``,
    weights, and the saved policy processor pipelines.

``rl100_native``
    An RL-100 export with a manifest-owned Python factory.  RL-100 checkpoints
    are not self-describing (the scheduler, shape metadata, normalizer, and
    encoder are required), so this loader refuses to infer those values from a
    directory containing ``model.pt``.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

from .action_spec import FrankaDuoActionSpec
from .franka_duo_eval_io import PointCloudConfig

LOGGER = logging.getLogger("franka_duo_eval.policy")
MANIFEST_NAME = "manifest.json"
SUPPORTED_BACKENDS = {"lerobot", "rl100_native"}


class _Predictor(Protocol):
    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray: ...

    def reset(self) -> None: ...


@dataclasses.dataclass
class PolicyBundle:
    manifest: dict[str, Any]
    action_spec: FrankaDuoActionSpec
    pointcloud_config: PointCloudConfig
    predictor: _Predictor
    required_observation_keys: tuple[str, ...]
    requires_state: bool = False

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        missing = [key for key in self.required_observation_keys if key not in observation]
        if missing:
            raise ValueError(f"Policy observation is missing required key(s): {missing}")
        action = np.asarray(self.predictor.predict(observation), dtype=np.float32)
        if action.shape != (self.action_spec.dimension,):
            raise ValueError(
                "Policy predictor must return one flat action vector with shape "
                f"({self.action_spec.dimension},), got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise ValueError("Policy predictor returned non-finite action values")
        return np.ascontiguousarray(action)

    def reset(self) -> None:
        self.predictor.reset()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                raise ValueError(f"Requested {device}, but CUDA is unavailable")
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for policy evaluation") from exc
    return device


def _validate_inputs(
    manifest: Mapping[str, Any], pointcloud: PointCloudConfig
) -> tuple[dict[str, Any], bool, tuple[str, ...], tuple[str, ...]]:
    raw = manifest.get("inputs")
    if not isinstance(raw, Mapping):
        raise ValueError("manifest.inputs is required and must be a mapping")
    point_key = raw.get("point_cloud_key")
    if not isinstance(point_key, str) or not point_key:
        raise ValueError("manifest.inputs.point_cloud_key must be a non-empty string")
    image_keys = raw.get("image_keys")
    if not isinstance(image_keys, Mapping) or not image_keys:
        raise ValueError("manifest.inputs.image_keys must map model image keys to wrist/head sources")
    sources: list[str] = []
    for model_key, source in image_keys.items():
        if not isinstance(model_key, str) or not model_key:
            raise ValueError("manifest.inputs.image_keys contains an invalid model key")
        if isinstance(source, str):
            sources.append(source)
        else:
            raise ValueError(
                f"image source for {model_key} must be one physical source string; "
                "map wrist_left and wrist_right to separate model image keys"
            )
    missing_wrist = {"wrist_left", "wrist_right"} - set(sources)
    if missing_wrist:
        raise ValueError(
            "manifest.inputs.image_keys must consume both wrist RGB streams; "
            f"missing {sorted(missing_wrist)}"
        )
    raw_action_spec = manifest.get("action_spec")
    if not isinstance(raw_action_spec, Mapping):
        raise ValueError("manifest.action_spec is required and must be a mapping")
    action_dim = manifest.get("action_dim", raw_action_spec.get("dimension", -1))
    try:
        action_dim = int(action_dim)
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest action_dim must be an integer") from exc
    if action_dim != 20:
        raise ValueError("manifest action_dim must be exactly 20")
    raw_pointcloud = manifest.get("pointcloud")
    if not isinstance(raw_pointcloud, Mapping):
        raise ValueError("manifest.pointcloud is required and must be a mapping")
    try:
        manifest_channels = int(raw_pointcloud.get("channels", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest.pointcloud.channels must be an integer") from exc
    if manifest_channels != pointcloud.channels:
        raise ValueError("manifest pointcloud channels disagree with parsed pointcloud config")
    image_model_keys = tuple(str(key) for key in image_keys)
    required_keys = (point_key,) + ((str(raw["state_key"]),) if raw.get("state_key") else ()) + image_model_keys
    if len(set(required_keys)) != len(required_keys):
        raise ValueError("manifest.inputs contains duplicate observation keys")
    return dict(raw), bool(raw.get("state_key")), required_keys, image_model_keys


def _as_action(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().to("cpu").numpy()
    except ImportError:
        pass
    array = np.asarray(value, dtype=np.float32)
    if array.ndim > 1:
        # A policy may return [batch, action_dim] or [batch, horizon, action_dim].
        array = array.reshape(-1, array.shape[-1])[0]
    return array.reshape(-1)


def _prepare_native_input(
    observation: Mapping[str, np.ndarray],
    device: str,
    *,
    image_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Convert the manifest-keyed numpy observation to batched torch tensors."""

    import torch

    result: dict[str, Any] = {}
    for key, value in observation.items():
        array = np.asarray(value)
        tensor = torch.from_numpy(array)
        if key in image_keys or "image" in key:
            if tensor.ndim != 3 or tensor.shape[-1] != 3:
                raise ValueError(f"Native image input {key} must be HWC RGB, got {tuple(tensor.shape)}")
            tensor = tensor.to(torch.float32).permute(2, 0, 1).div(255.0)
        else:
            tensor = tensor.to(torch.float32)
        result[key] = tensor.unsqueeze(0).to(device)
    return result


class _LeRobotPredictor:
    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        device: str,
        required_observation_keys: tuple[str, ...],
    ):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.required_observation_keys = required_observation_keys

    def reset(self) -> None:
        self.policy.reset()
        for processor in (self.preprocessor, self.postprocessor):
            reset = getattr(processor, "reset", None)
            if callable(reset):
                reset()

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        import torch

        missing = [key for key in self.required_observation_keys if key not in observation]
        if missing:
            raise ValueError(f"Policy observation is missing required key(s): {missing}")

        # LeRobot's standard policy processors expect CHW image tensors and a
        # leading batch dimension, while the ROS reader intentionally returns
        # HWC uint8 arrays for camera-independent testing.
        prepared = prepare_observation_for_inference(
            {key: np.asarray(value).copy() for key, value in observation.items()},
            torch.device(self.device),
        )
        batch = self.preprocessor(prepared)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)
        return _as_action(action)


def _load_lerobot(bundle_dir: Path, manifest: dict[str, Any], device: str) -> PolicyBundle:
    policy_relative = manifest.get("policy_dir")
    if policy_relative is None:
        # A bundle may itself be the LeRobot ``pretrained_model`` directory;
        # the parent-directory layout is only a convenience for adding the
        # manifest without modifying a training checkpoint.
        policy_dir = bundle_dir if (bundle_dir / "config.json").is_file() else bundle_dir / "pretrained_model"
    else:
        policy_dir = (bundle_dir / str(policy_relative)).resolve()
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"LeRobot policy directory does not exist: {policy_dir}")
    pointcloud = PointCloudConfig.from_manifest(manifest)
    input_spec, requires_state, required_keys, _image_keys = _validate_inputs(manifest, pointcloud)
    config = PreTrainedConfig.from_pretrained(policy_dir)
    resolved_device = _resolve_device(device)
    config.device = resolved_device
    action_feature = config.action_feature
    if action_feature is None or tuple(action_feature.shape) != (20,):
        shape = None if action_feature is None else action_feature.shape
        raise ValueError(f"LeRobot bundle action feature must be (20,), got {shape}")
    input_features = config.input_features or {}
    point_key = str(input_spec["point_cloud_key"])
    point_feature = input_features.get(point_key)
    if point_feature is None or tuple(point_feature.shape) != (
        pointcloud.num_points,
        pointcloud.channels,
    ):
        raise ValueError(f"LeRobot bundle point-cloud feature does not match manifest for {point_key}")
    expected_image_keys = {str(key) for key in dict(input_spec["image_keys"])}
    missing_images = expected_image_keys - set(input_features)
    if missing_images:
        raise ValueError(f"LeRobot bundle is missing image feature(s): {sorted(missing_images)}")
    if requires_state:
        state_feature = input_features.get(str(input_spec["state_key"]))
        if state_feature is None:
            raise ValueError(f"LeRobot bundle is missing state feature {input_spec['state_key']}")
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(policy_dir, config=config, strict=True)
    try:
        preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(policy_dir))
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            f"LeRobot bundle must include saved policy_preprocessor.json and policy_postprocessor.json: {policy_dir}"
        ) from exc
    return PolicyBundle(
        manifest=manifest,
        action_spec=FrankaDuoActionSpec.from_manifest(manifest),
        pointcloud_config=pointcloud,
        predictor=_LeRobotPredictor(
            policy,
            preprocessor,
            postprocessor,
            resolved_device,
            required_observation_keys=required_keys,
        ),
        required_observation_keys=required_keys,
        requires_state=requires_state,
    )


class _NativePredictor:
    def __init__(
        self,
        model: Any,
        device: str,
        manifest: Mapping[str, Any],
        required_observation_keys: tuple[str, ...],
        image_keys: tuple[str, ...],
    ):
        self.model = model
        self.device = device
        self.manifest = manifest
        self.required_observation_keys = required_observation_keys
        self.image_keys = frozenset(image_keys)
        self._history: dict[str, list[Any]] = {}
        native = manifest.get("native", {})
        self.n_obs_steps = int(native.get("n_obs_steps", 1)) if isinstance(native, Mapping) else 1

    def reset(self) -> None:
        self._history.clear()
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def _history_batch(self, observation: Mapping[str, np.ndarray]) -> dict[str, Any]:
        import torch

        missing = [key for key in self.required_observation_keys if key not in observation]
        if missing:
            raise ValueError(f"Policy observation is missing required key(s): {missing}")
        batch: dict[str, Any] = {}
        for key, value in observation.items():
            tensor = _prepare_native_input(
                {key: value}, self.device, image_keys=self.image_keys
            )[key]
            history = self._history.setdefault(key, [])
            history.append(tensor)
            del history[:-self.n_obs_steps]
            if self.n_obs_steps > 1:
                while len(history) < self.n_obs_steps:
                    history.insert(0, history[0])
                batch[key] = torch.cat(history, dim=1)
            else:
                batch[key] = tensor
        return batch

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        batch = self._history_batch(observation)
        if callable(getattr(self.model, "predict", None)):
            value = self.model.predict(batch)
        elif callable(getattr(self.model, "predict_action", None)):
            value = self.model.predict_action(batch)
        elif callable(getattr(self.model, "select_action", None)):
            value = self.model.select_action(batch)
        else:
            raise TypeError("native factory result must expose predict, predict_action, or select_action")
        if isinstance(value, Mapping):
            value = value.get("action", value.get("actions"))
        if value is None:
            raise ValueError("native policy returned no action")
        return _as_action(value)


def _load_native(bundle_dir: Path, manifest: dict[str, Any], device: str) -> PolicyBundle:
    native = manifest.get("native")
    if not isinstance(native, Mapping):
        raise ValueError("rl100_native bundle requires manifest.native")
    factory_path = native.get("factory")
    if not isinstance(factory_path, str) or ":" not in factory_path:
        raise ValueError(
            "rl100_native manifest.native.factory must be 'python.module:function'; "
            "RL-100 model.pt is not self-describing and cannot be guessed"
        )
    module_name, function_name = factory_path.split(":", 1)
    python_root = native.get("python_root")
    if python_root:
        root = (bundle_dir / str(python_root)).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"native python_root does not exist: {root}")
        sys.path.insert(0, str(root))
    try:
        factory = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"Cannot import RL-100 native factory {factory_path}") from exc
    if not callable(factory):
        raise TypeError(f"RL-100 native factory {factory_path} is not callable")
    resolved_device = _resolve_device(device)
    kwargs = native.get("kwargs", {})
    if not isinstance(kwargs, Mapping):
        raise ValueError("manifest.native.kwargs must be a mapping")
    model = factory(bundle_dir=bundle_dir, device=resolved_device, **dict(kwargs))
    pointcloud = PointCloudConfig.from_manifest(manifest)
    input_spec, requires_state, required_keys, image_keys = _validate_inputs(manifest, pointcloud)
    del input_spec
    return PolicyBundle(
        manifest=manifest,
        action_spec=FrankaDuoActionSpec.from_manifest(manifest),
        pointcloud_config=pointcloud,
        predictor=_NativePredictor(
            model,
            resolved_device,
            manifest,
            required_observation_keys=required_keys,
            image_keys=image_keys,
        ),
        required_observation_keys=required_keys,
        requires_state=requires_state,
    )


def load_policy_bundle(path: Path, *, device: str = "auto") -> PolicyBundle:
    """Load and validate an exported bundle before ROS subscriptions start."""

    bundle_dir = Path(path).expanduser().resolve()
    manifest = _read_json(bundle_dir / MANIFEST_NAME, "policy bundle manifest")
    version = int(manifest.get("manifest_version", 1))
    if version != 1:
        raise ValueError(f"Unsupported policy bundle manifest_version={version}")
    backend = manifest.get("backend")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"manifest.backend must be one of {sorted(SUPPORTED_BACKENDS)}, got {backend!r}")
    # Parse the contract before importing either a heavy policy or an RL-100
    # repository.  This keeps a malformed export harmless on a robot host.
    action_spec = FrankaDuoActionSpec.from_manifest(manifest)
    pointcloud = PointCloudConfig.from_manifest(manifest)
    _validate_inputs(manifest, pointcloud)
    if backend == "lerobot":
        bundle = _load_lerobot(bundle_dir, manifest, device)
    else:
        bundle = _load_native(bundle_dir, manifest, device)
    if bundle.action_spec != action_spec or bundle.pointcloud_config != pointcloud:
        raise RuntimeError("Policy backend changed the validated action or point-cloud contract")
    LOGGER.info(
        "Loaded %s policy bundle: action_dim=%d pointcloud=(%d,%d) state=%s",
        backend,
        action_spec.dimension,
        pointcloud.num_points,
        pointcloud.channels,
        bundle.requires_state,
    )
    return bundle
