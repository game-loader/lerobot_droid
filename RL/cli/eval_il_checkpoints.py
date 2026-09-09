"""Evaluate saved IL checkpoints in Newton and select the best one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from RL.cli.train_il_warmstart import processor_fingerprint
from RL.cli.train_iterative_offline import (
    _il_checkpoint_specs,
    _il_eval_inputs,
    _il_eval_payload,
    _run_il_evaluation_stage,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--eval-every-steps", type=int, default=5_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--env-device", default="cuda:0")
    parser.add_argument("--env-type", default="moya_newton")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    specs = _il_checkpoint_specs(
        args.train_output,
        steps=args.steps,
        save_freq=args.save_freq,
        eval_every_steps=args.eval_every_steps,
    )
    # Build this before running so a mismatched source normalizer fails before
    # any expensive Newton rollout starts.
    inputs = _il_eval_inputs(
        train_output=args.train_output,
        source_checkpoint=args.source_checkpoint,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        steps=args.steps,
        il_save_freq=args.save_freq,
        eval_every_steps=args.eval_every_steps,
        episodes=args.episodes,
        batch_size=args.batch_size,
        inference_steps=args.inference_steps,
        policy_device=args.policy_device,
        env_device=args.env_device,
        env_type=args.env_type,
        seed=args.seed,
        specs=specs,
    )
    best = _run_il_evaluation_stage(
        eval_output=args.train_output,
        specs=specs,
        expected_episodes=args.episodes,
        batch_size=args.batch_size,
        inference_steps=args.inference_steps,
        policy_device=args.policy_device,
        env_device=args.env_device,
        env_type=args.env_type,
        seed=args.seed,
        log_root=args.log_root,
    )
    payload = {
        "complete": True,
        "inputs": inputs,
        "processor_fingerprint": processor_fingerprint(args.source_checkpoint),
        "best": _il_eval_payload(best),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["best"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
