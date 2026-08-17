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

"""Configuration contracts for the RL migration."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


def _reject_unknown_fields(data: dict[str, Any], config_type: type[Any]) -> None:
    known_fields = {item.name for item in fields(config_type)}
    unknown_fields = sorted(data.keys() - known_fields)
    if unknown_fields:
        label = "field" if len(unknown_fields) == 1 else "fields"
        raise ValueError(f"Unknown {config_type.__name__} {label}: {', '.join(unknown_fields)}")


def _parse_json_object(payload: str | bytes | bytearray, config_name: str) -> dict[str, Any]:
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid {config_name} JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{config_name} JSON must contain an object, got {type(data).__name__}")
    return data


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return converted


@dataclass(frozen=True)
class TraceConfig:
    num_inference_steps: int = 10
    eta: float = 1.0
    sigma_min: float = 0.0067
    sigma_max: float = 0.1

    def __post_init__(self) -> None:
        _positive_int("num_inference_steps", self.num_inference_steps)
        eta = _finite_float("eta", self.eta)
        sigma_min = _finite_float("sigma_min", self.sigma_min)
        sigma_max = _finite_float("sigma_max", self.sigma_max)
        if eta < 0:
            raise ValueError(f"eta must be nonnegative, got {self.eta!r}")
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be positive, got {self.sigma_min!r}")
        if sigma_max < sigma_min:
            raise ValueError(f"sigma_max must be at least sigma_min ({sigma_min}), got {self.sigma_max!r}")
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "sigma_min", sigma_min)
        object.__setattr__(self, "sigma_max", sigma_max)

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> TraceConfig:
        data = _parse_json_object(payload, cls.__name__)
        _reject_unknown_fields(data, cls)
        return cls(**data)


@dataclass(frozen=True)
class RLConfig:
    trace: TraceConfig = field(default_factory=TraceConfig)
    state_key: str = "observation.state"
    n_obs_steps: int = 2
    state_dim: int = 39
    action_dim: int = 14
    chunk_size: int = 32
    gamma: float = 0.99

    def __post_init__(self) -> None:
        if not isinstance(self.trace, TraceConfig):
            raise ValueError(f"trace must be a TraceConfig, got {type(self.trace).__name__}")
        if not isinstance(self.state_key, str) or not self.state_key:
            raise ValueError(f"state_key must be a nonempty string, got {self.state_key!r}")
        _positive_int("n_obs_steps", self.n_obs_steps)
        _positive_int("state_dim", self.state_dim)
        _positive_int("action_dim", self.action_dim)
        _positive_int("chunk_size", self.chunk_size)
        gamma = _finite_float("gamma", self.gamma)
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in the interval [0, 1], got {self.gamma!r}")
        object.__setattr__(self, "gamma", gamma)

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> RLConfig:
        data = _parse_json_object(payload, cls.__name__)
        _reject_unknown_fields(data, cls)
        trace_data = data.get("trace")
        if trace_data is not None:
            if not isinstance(trace_data, dict):
                raise ValueError(f"trace must be a JSON object, got {type(trace_data).__name__}")
            _reject_unknown_fields(trace_data, TraceConfig)
            data["trace"] = TraceConfig(**trace_data)
        return cls(**data)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(f"{self.to_json()}\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> RLConfig:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
