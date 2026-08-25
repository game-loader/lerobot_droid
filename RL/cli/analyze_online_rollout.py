"""Roll out an online diffusion checkpoint and diagnose Moya failures.

The analysis keeps the online trainer's stochastic DDIM/action-chunk path, then
records terminal diagnostics and runs a separate native-IK feasibility probe for
each initial charger target.  It does not record video.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from RL.adapters.checkpoint import CheckpointAdapter
from RL.adapters.moya_newton import create_moya_env
from RL.config import TraceConfig
from RL.policy.diffusion_adapter import DiffusionRLAdapter
from RL.trainers.online import OnlineTrainer

_RECORD_KEYS = (
    "charger_position",
    "charger_lift_height",
    "charger_table_contacts",
    "right_hand_charger_contacts",
    "right_palm_position",
    "right_palm_target",
    "right_wrist_position",
    "right_wrist_target",
    "right_palm_distance_to_charger",
    "grasp_reference_distance",
    "right_arm_min_normalized_joint_margin",
    "right_arm_normalized_joint_positions",
    "right_thumb_actual_close_t",
    "right_four_fingers_actual_close_t",
    "success_hold_count",
    "elapsed_steps",
    "effective_action",
    "policy_action",
    "success",
    "charger_success",
    "native_success",
    "reward_components",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _world_value(value: Any, world: int, num_envs: int) -> Any:
    """Select one world from collated Moya info, preserving nested mappings."""

    if isinstance(value, Mapping):
        return {str(key): _world_value(item, world, num_envs) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == num_envs:
        return value[world].copy()
    return _copy_value(value)


def _scalar(value: Any, default: float = 0.0) -> float:
    try:
        array = np.asarray(value)
        if array.size == 0:
            return float(default)
        result = float(array.reshape(-1)[0])
        return result if math.isfinite(result) else float(default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        return bool(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return default


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("analysis contains a non-finite scalar")
        return result
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"analysis value is not JSON serializable: {type(value).__name__}")


class _RecordingEnv:
    """Thin proxy that records only the fields needed for failure diagnosis."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self.num_envs = int(env.num_envs)
        self.single_action_space = getattr(env, "single_action_space", None)
        self.action_space = getattr(env, "action_space", None)
        self.reset_info: Mapping[str, Any] = {}
        self.steps: list[dict[str, Any]] = []

    def reset(self, *, seed: Any = None, options: Any = None) -> Any:
        result = self._env.reset(seed=seed, options=options)
        if isinstance(result, tuple) and len(result) == 2:
            self.reset_info = _copy_value(result[1])
        else:
            self.reset_info = {}
        self.steps = []
        return result

    def step(self, action: np.ndarray) -> Any:
        result = self._env.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError("Moya step must return five values")
        _observation, _reward, terminated, truncated, info = result
        fields = {
            key: _copy_value(info[key]) for key in _RECORD_KEYS if key in info
        }
        self.steps.append(
            {
                "action": np.asarray(action, dtype=np.float32).copy(),
                "terminated": np.asarray(terminated, dtype=np.bool_).copy(),
                "truncated": np.asarray(truncated, dtype=np.bool_).copy(),
                "info": fields,
                "final_info": _copy_value(info.get("final_info")),
                "final_info_mask": _copy_value(info.get("_final_info")),
            }
        )
        return result

    def close(self) -> None:
        self._env.close()


def _resolve_policy_root(path: Path) -> Path:
    root = path.resolve(strict=True)
    if (root / "model.safetensors").is_file():
        return root
    nested = root / "pretrained_model"
    if (nested / "model.safetensors").is_file():
        return nested
    final = root / "checkpoints" / "final" / "pretrained_model"
    if (final / "model.safetensors").is_file():
        return final
    raise ValueError(f"could not find pretrained_model under {path}")


def _terminal_info(step: dict[str, Any], world: int, num_envs: int) -> dict[str, Any]:
    raw = step.get("final_info")
    mask = step.get("final_info_mask")
    if mask is not None and not _bool(np.asarray(mask)[world]):
        return {}
    if raw is None:
        return {}
    selected = _world_value(raw, world, num_envs)
    return dict(selected) if isinstance(selected, Mapping) else {}


def _trajectory_value(trajectory: list[dict[str, Any]], key: str) -> np.ndarray:
    values = [item.get(key) for item in trajectory if item.get(key) is not None]
    if not values:
        return np.empty((0,), dtype=np.float32)
    try:
        return np.asarray(values)
    except (TypeError, ValueError):
        return np.asarray(values, dtype=object)


def _episode_summary(
    *,
    episode_index: int,
    batch_index: int,
    world_index: int,
    seed: int,
    reset_info: Mapping[str, Any],
    steps: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    num_envs = int(np.asarray(reset_info["charger_initial_position"]).shape[0])
    initial_charger = np.asarray(reset_info["charger_initial_position"])[world_index].astype(np.float32)
    initial_grasp = np.asarray(reset_info.get("grasp_reference_position"))[world_index].astype(np.float32)
    done_index = next(
        (
            index
            for index, item in enumerate(steps)
            if bool(item["terminated"][world_index] or item["truncated"][world_index])
        ),
        len(steps) - 1,
    )
    trajectory: list[dict[str, Any]] = []
    terminal: dict[str, Any] = {}
    for item in steps[: done_index + 1]:
        world_fields = {
            key: _world_value(value, world_index, num_envs)
            for key, value in item["info"].items()
        }
        world_fields["action"] = item["action"][world_index].copy()
        trajectory.append(world_fields)
        if bool(item["terminated"][world_index] or item["truncated"][world_index]):
            terminal = _terminal_info(item, world_index, num_envs)

    def values(key: str) -> np.ndarray:
        return _trajectory_value(trajectory, key)

    palm = values("right_palm_position").astype(np.float32, copy=False)
    palm_target = values("right_palm_target").astype(np.float32, copy=False)
    wrist = values("right_wrist_position").astype(np.float32, copy=False)
    wrist_target = values("right_wrist_target").astype(np.float32, copy=False)
    palm_residual = (
        np.linalg.norm(palm - palm_target, axis=1) if len(palm) and len(palm_target) else np.empty(0)
    )
    wrist_residual = (
        np.linalg.norm(wrist - wrist_target, axis=1) if len(wrist) and len(wrist_target) else np.empty(0)
    )
    lift = values("charger_lift_height").astype(np.float32, copy=False)
    palm_distance = values("right_palm_distance_to_charger").astype(np.float32, copy=False)
    margins = values("right_arm_min_normalized_joint_margin").astype(np.float32, copy=False)
    contacts = values("right_hand_charger_contacts").astype(np.int32, copy=False)
    table_contacts = values("charger_table_contacts").astype(np.int32, copy=False)
    success = _bool(terminal.get("is_success", terminal.get("success", False)))
    true_grasp = _bool(terminal.get("true_grasp_ever", False))
    clear_table = _bool(terminal.get("clear_table_ever", False))
    final_lift = _scalar(terminal.get("charger_lift_height", lift[-1] if len(lift) else 0.0))
    final_table = int(round(_scalar(terminal.get("charger_table_contacts", table_contacts[-1] if len(table_contacts) else 0))))
    final_hand = int(round(_scalar(terminal.get("right_hand_charger_contacts", contacts[-1] if len(contacts) else 0))))
    terminal_palm = np.asarray(terminal.get("right_palm_position", []), dtype=np.float32)
    terminal_palm_target = np.asarray(terminal.get("right_palm_target", []), dtype=np.float32)
    terminal_wrist = np.asarray(terminal.get("right_wrist_position", []), dtype=np.float32)
    terminal_wrist_target = np.asarray(terminal.get("right_wrist_target", []), dtype=np.float32)
    terminal_palm_residual = (
        float(np.linalg.norm(terminal_palm - terminal_palm_target))
        if terminal_palm.shape == (3,) and terminal_palm_target.shape == (3,)
        else (float(palm_residual[-1]) if len(palm_residual) else None)
    )
    terminal_wrist_residual = (
        float(np.linalg.norm(terminal_wrist - terminal_wrist_target))
        if terminal_wrist.shape == (3,) and terminal_wrist_target.shape == (3,)
        else (float(wrist_residual[-1]) if len(wrist_residual) else None)
    )
    reasons: list[str] = []
    if not true_grasp:
        reasons.append("true_grasp_never_established")
    if not clear_table:
        reasons.append("clear_table_never_established")
    if final_lift < 0.015:
        reasons.append("final_lift_below_15mm")
    if final_table != 0:
        reasons.append("final_table_contact")
    if final_hand <= 0:
        reasons.append("no_final_hand_contact")
    summary = {
        "episode_index": episode_index,
        "batch_index": batch_index,
        "world_index": world_index,
        "seed": seed,
        "steps": len(trajectory),
        "success": success,
        "failure_reasons": reasons,
        "charger_initial_position_m": initial_charger,
        "grasp_reference_position_m": initial_grasp,
        "final_lift_height_m": final_lift,
        "final_table_contacts": final_table,
        "final_hand_contacts": final_hand,
        "true_grasp_ever": true_grasp,
        "clear_table_ever": clear_table,
        "min_palm_distance_to_charger_m": float(np.min(palm_distance)) if len(palm_distance) else None,
        "max_lift_height_m": float(np.max(lift)) if len(lift) else None,
        "min_arm_joint_margin": float(np.min(margins)) if len(margins) else None,
        "max_palm_target_residual_m": float(np.max(palm_residual)) if len(palm_residual) else None,
        "final_palm_target_residual_m": terminal_palm_residual,
        "max_wrist_target_residual_m": float(np.max(wrist_residual)) if len(wrist_residual) else None,
        "final_wrist_target_residual_m": terminal_wrist_residual,
        "terminal_info": terminal,
    }
    arrays = {
        "action": np.asarray([item["action"] for item in trajectory], dtype=np.float32),
        "charger_position": values("charger_position").astype(np.float32, copy=False),
        "charger_lift_height": lift,
        "charger_table_contacts": table_contacts,
        "right_hand_charger_contacts": contacts,
        "right_palm_position": palm,
        "right_palm_target": palm_target,
        "right_wrist_position": wrist,
        "right_wrist_target": wrist_target,
        "right_palm_distance_to_charger": palm_distance,
        "right_arm_min_normalized_joint_margin": margins,
    }
    return summary, arrays


def _ik_probe(
    targets: np.ndarray,
    *,
    env: Any,
) -> list[dict[str, float]]:
    """Run native LM IK against each target and measure FK residual/margins."""

    import newton

    results: list[dict[str, float]] = []
    capacity = int(env.num_envs)
    for start in range(0, len(targets), capacity):
        batch = targets[start : start + capacity]
        padded = np.repeat(batch[-1:, :], capacity, axis=0)
        padded[: len(batch)] = batch
        env.reset(seed=910000 + start)
        sim = env.unwrapped._batched_sim
        sim._right_palm_target_positions[...] = np.asarray(padded, dtype=np.float32)
        sim._sync_right_palm_and_wrist_targets()
        sim._solve_and_apply_native_ik()
        qik = sim.joint_q_ik.numpy().copy()
        state = sim.model.state()
        state.joint_q.assign(qik.reshape(-1))
        state.joint_qd.assign(np.zeros_like(state.joint_qd.numpy()))
        newton.eval_fk(sim.model, state.joint_q, state.joint_qd, state)
        body = state.body_q.numpy().reshape(sim.world_count, sim.layout.body_count_per_world, 7)
        from moya_batched_env import RIGHT_ARM_DOFS, RIGHT_WRIST_BODY, _quat_rotate_vector_xyzw

        wrist = body[:, RIGHT_WRIST_BODY, :3]
        wrist_target = sim._right_target_positions.copy()
        wrist_residual = np.linalg.norm(wrist - wrist_target, axis=1)
        lower = sim._joint_limit_lower[:, np.asarray(RIGHT_ARM_DOFS)]
        upper = sim._joint_limit_upper[:, np.asarray(RIGHT_ARM_DOFS)]
        q = qik[:, np.asarray(RIGHT_ARM_DOFS)]
        margin = np.min(
            np.minimum(
                (q - lower) / np.maximum(upper - lower, 1.0e-8),
                (upper - q) / np.maximum(upper - lower, 1.0e-8),
            ),
            axis=1,
        )
        palm_base = body[:, 37, :3]
        palm_quat = body[:, 37, 3:]
        palm_offset = np.stack(
            [_quat_rotate_vector_xyzw(palm_quat[i], sim._right_palm_centroid_local) for i in range(capacity)],
            axis=0,
        )
        palm = palm_base + palm_offset
        palm_residual = np.linalg.norm(palm - sim._right_palm_target_positions, axis=1)
        for i in range(len(batch)):
            results.append(
                {
                    "ik_wrist_position_residual_m": float(wrist_residual[i]),
                    "ik_palm_position_residual_m": float(palm_residual[i]),
                    "ik_min_joint_margin": float(margin[i]),
                    "ik_position_reachable_2mm": bool(wrist_residual[i] <= 0.002),
                    "ik_near_joint_limit_1pct": bool(margin[i] < 0.01),
                }
            )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--episode-length", type=int, default=930)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--probability-sigma-min", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=810000)
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.episodes <= 0 or args.num_envs <= 0 or args.episode_length <= 0:
        raise ValueError("episodes, num_envs, and episode_length must be positive")
    if args.episodes < args.num_envs:
        raise ValueError("episodes must be at least num_envs for stable batch ordering")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    policy_root = _resolve_policy_root(args.checkpoint)
    adapter = CheckpointAdapter.load(policy_root, device=args.device)
    trace = TraceConfig(
        num_inference_steps=args.inference_steps,
        probability_sigma_min=args.probability_sigma_min,
    )
    current = DiffusionRLAdapter(adapter, trace)
    old = DiffusionRLAdapter(CheckpointAdapter.load(policy_root, device=args.device), trace)
    trainer = OnlineTrainer(current_policy=current, old_policy=old, metrics_path=None, seed=args.seed)
    decisions = math.ceil(args.episode_length / int(current.policy.config.n_action_steps))
    summaries: list[dict[str, Any]] = []
    trajectories: dict[str, np.ndarray] = {}
    batch_index = 0
    env = _RecordingEnv(
        create_moya_env(
            num_envs=args.num_envs,
            device=args.sim_device,
            episode_length=args.episode_length,
            headless=True,
        )
    )
    try:
        while len(summaries) < args.episodes:
            trainer.collect(env, decisions=decisions, seed=args.seed + batch_index)
            reset_info = env.reset_info
            for world in range(args.num_envs):
                if len(summaries) >= args.episodes:
                    break
                summary, arrays = _episode_summary(
                    episode_index=len(summaries),
                    batch_index=batch_index,
                    world_index=world,
                    seed=args.seed + batch_index,
                    reset_info=reset_info,
                    steps=env.steps,
                )
                summaries.append(summary)
                if not summary["success"]:
                    for key, value in arrays.items():
                        trajectories[f"episode_{summary['episode_index']:04d}/{key}"] = value
            print(f"batch={batch_index} analyzed={len(summaries)}/{args.episodes}", flush=True)
            batch_index += 1

        targets = np.asarray([item["grasp_reference_position_m"] for item in summaries], dtype=np.float32)
        ik = _ik_probe(targets, env=env._env)
        for summary, probe in zip(summaries, ik, strict=True):
            summary.update(probe)
    finally:
        env.close()

    successes = [item for item in summaries if item["success"]]
    failures = [item for item in summaries if not item["success"]]
    reachable_failures = [item for item in failures if item["ik_position_reachable_2mm"]]
    output.mkdir(parents=True)
    (output / "episodes.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), sort_keys=True) + "\n" for item in summaries),
        encoding="utf-8",
    )
    with (output / "failed_trajectories.npz").open("wb") as handle:
        np.savez_compressed(handle, **trajectories)
    payload = {
        "checkpoint": str(policy_root),
        "checkpoint_model_sha256": _sha256(policy_root / "model.safetensors"),
        "episode_count": len(summaries),
        "successes": len(successes),
        "failures": len(failures),
        "success_rate_percent": 100.0 * len(successes) / len(summaries),
        "failure_with_2mm_ik_reachable": len(reachable_failures),
        "failure_with_2mm_ik_unreachable": len(failures) - len(reachable_failures),
        "ik_probe_thresholds": {
            "position_residual_m": 0.002,
            "near_joint_limit_margin": 0.01,
        },
        "config": {
            "num_envs": args.num_envs,
            "episode_length": args.episode_length,
            "decisions": decisions,
            "inference_steps": args.inference_steps,
            "probability_sigma_min": args.probability_sigma_min,
            "seed": args.seed,
            "device": args.device,
            "sim_device": args.sim_device,
        },
        "failure_reason_counts": {
            reason: sum(reason in item["failure_reasons"] for item in failures)
            for reason in sorted({reason for item in failures for reason in item["failure_reasons"]})
        },
        "aggregate": {
            "failure_min_palm_distance_m": float(np.nanmin([item["min_palm_distance_to_charger_m"] for item in failures])) if failures else None,
            "failure_median_final_palm_residual_m": float(np.nanmedian([item["final_palm_target_residual_m"] for item in failures])) if failures else None,
            "failure_median_ik_wrist_residual_m": float(np.nanmedian([item["ik_wrist_position_residual_m"] for item in failures])) if failures else None,
            "failure_median_ik_joint_margin": float(np.nanmedian([item["ik_min_joint_margin"] for item in failures])) if failures else None,
            "success_median_ik_wrist_residual_m": float(np.nanmedian([item["ik_wrist_position_residual_m"] for item in successes])) if successes else None,
        },
        "episodes": summaries,
    }
    (output / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("episode_count", "successes", "failures", "failure_with_2mm_ik_reachable", "failure_with_2mm_ik_unreachable")}, indent=2, sort_keys=True))
    return output


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
