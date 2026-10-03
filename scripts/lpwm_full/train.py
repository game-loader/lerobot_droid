"""Use the winning B30k recipe with common40/all130 cache and evaluation adapters."""

import json
import subprocess

from scripts.lpwm_ab import train_b_sweep as trainer
from scripts.lpwm_full.data import FullCachedDataset
from scripts.lpwm_full.evaluate import catalog_suites, protocol_suites, validate_result

FULL_SCHEDULE_STEPS = (30000, 80000, 200000)


def validate_parallel_eval(result, workers):
    """Never let an explicitly requested task8 evaluation silently run serial."""
    if workers == 1:
        return
    protocol = result["protocol"]
    expected = {
        "parallel_workers": workers,
        "parallel_unit": "task",
        "within_task_episode_order": "serial original plan",
        "worker_start_method": "spawn",
    }
    if workers != 8 or any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Evaluation did not use the requested task8 protocol")
    if not protocol.get("preflight") and protocol.get("actual_busy_workers") != workers:
        raise ValueError("Formal task8 evaluation did not use all eight workers")


def full_checkpoint_eval(args, checkpoint, step):
    before = trainer.verify_checkpoint(checkpoint, step)
    output = args.output / "eval" / f"step_{step:06d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".swanlab.json").exists():
        raise FileExistsError(output)
    command = [
        str(args.eval_python),
        "-m",
        "scripts.lpwm_full.evaluate",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output",
        str(output.resolve()),
        "--credential-file",
        str(args.credential_file),
        "--project",
        args.project,
        "--run-name",
        f"{args.run_name}-eval-{step:06d}",
        "--workers",
        str(getattr(args, "eval_workers", 1)),
    ]
    if args.preflight:
        command += ["--preflight"]
    with output.with_suffix(".log").open("x") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    result = json.loads(output.read_text())
    metrics = validate_result(result, args.preflight)
    validate_parallel_eval(result, getattr(args, "eval_workers", 1))
    expected_suites = catalog_suites(json.loads((checkpoint / "task_catalog.json").read_text()))
    if protocol_suites(result["protocol"]) != expected_suites:
        raise ValueError("Evaluated wrong task scope")
    if (
        result["checkpoint"]["step"] != step
        or result["checkpoint"]["model_sha256"] != before["sha256"]["model.safetensors"]
    ):
        raise ValueError("Evaluated wrong checkpoint")
    trainer.verify_checkpoint(checkpoint, step)
    trainer.verify_eval_upload(output, preflight=args.preflight)
    prefix = "preflight_rollout/" if args.preflight else "rollout/"
    return {k.replace("eval/", prefix, 1): v for k, v in metrics.items()}


def validate_manifest(manifest, preflight=False, validation_batches=None):
    if manifest["schema"] != "lpwm_libero_full_zlib_v1":
        raise ValueError("Full lossless cache schema required")
    suites = catalog_suites(manifest["task_catalog"])
    task_count = sum(suites.values())
    expected = {str(t["global_task_id"]): t["language"] for t in manifest["task_catalog"]["tasks"]}
    if manifest["tasks"] != expected:
        raise ValueError("Manifest tasks differ from task catalog")
    observed_tasks = {ep["task_index"] for ep in manifest["episodes"]}
    if not observed_tasks <= set(range(task_count)):
        raise ValueError("Unknown episode task IDs")
    if not preflight and (observed_tasks != set(range(task_count)) or manifest.get("preflight_only")):
        raise ValueError(f"Production requires actual episodes for all{task_count}tasks, not a partial cache")
    # The trainer's balanced offline validation uses batch size 8, independently of microbatch.
    if not preflight and validation_batches is not None and validation_batches * 8 < task_count:
        raise ValueError(f"Offline validation must coverall{task_count}tasks")
    return suites


def main():
    args = trainer.parse_args(allowed_schedule_steps=FULL_SCHEDULE_STEPS, allow_resume=True)
    manifest = json.loads((args.data / "manifest.json").read_text())
    suites = validate_manifest(manifest, args.preflight, args.validation_batches)
    task_count = sum(suites.values())
    if (args.world_weight, args.rec_weight, args.dyn_weight, args.prior_weight) != (1.0, 1.0, 1.0, 0.001):
        raise ValueError("Use verified winning B losscoefficients")
    for file, value in manifest["scalar_sha256"].items():
        if trainer.file_sha256(args.data / file) != value:
            raise ValueError("Changed data/language cache")
    base = trainer.training_helpers()
    base.LPWMCachedDataset = FullCachedDataset
    trainer.run_checkpoint_eval = full_checkpoint_eval
    trainer.claim_output(args.output)
    trainer.atomic_json(args.output / "task_catalog.json", manifest["task_catalog"])
    args.full_task_count = task_count
    args.full_suites = list(suites)
    args.cache_codec = manifest["image_codec"]
    args.training_scope = f"all{task_count}taskinstances;noheldout-taskgeneralizationclaim"
    try:
        trainer.train(args)
    except BaseException as error:
        p = args.output / "status.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        trainer.atomic_json(
            p, {"status": "failed", "step": old.get("step"), "error_type": type(error).__name__}
        )
        raise


if __name__ == "__main__":
    main()
