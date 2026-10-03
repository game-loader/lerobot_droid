"""Read-only summary of exported B sweep evidence; never trains or evaluates a policy."""

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STEPS = (5000, 10000, 15000, 20000, 25000, 30000)
LATE = (20000, 25000, 30000)


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(directory):
    raw = directory / "raw"
    queue = read(raw / "queue_state.json")
    verified = read(directory / "remote-verification.json")
    assert queue["status"] == verified["queue_status"] == "completed"
    assert verified["checkpoint_count"] == verified["evaluation_count"] == 36
    assert verified["paired_contract_verified"]
    curve, episodes, tasks, summaries, diagnostic_rows, validation_rows = [], [], [], [], [], []
    run_ids = sorted(
        queue["runs"],
        key=lambda name: (
            queue["runs"][name]["spec"]["phase"],
            queue["runs"][name]["spec"]["world_weight"],
            queue["runs"][name]["spec"]["reconstruction_weight"],
        ),
    )
    reference_plan = None
    for name in run_ids:
        root = raw / "runs" / name
        spec = queue["runs"][name]["spec"]
        experiment = read(root / "experiment.json")
        recovered = root / "recovery_20260919"
        assert read(root / "status.json")["step"] == 30000
        successes = {}
        for step in STEPS:
            path = root / "eval" / f"step_{step:06d}.json"
            result = read(path)
            receipt = read(path.with_suffix(".swanlab.json"))
            assert receipt["result_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert result["status"] == "complete" and result["num_episodes"] == 100
            assert result["checkpoint"]["step"] == step
            assert result["protocol"]["seed_namespace"] == "validation"
            plan = []
            count = 0
            for task in result["per_task"]:
                assert task["num_episodes"] == 10
                assert sum(bool(ep["success"]) for ep in task["episodes"]) == task["successes"]
                tasks.append(
                    {
                        "run": name,
                        "step": step,
                        "task_id": task["task_id"],
                        "language": task["language"],
                        "successes": task["successes"],
                        "episodes": 10,
                    }
                )
                for ep in task["episodes"]:
                    plan.append((task["task_id"], ep["episode_index"], ep["init_state_index"], ep["seed"]))
                    count += int(ep["success"])
                    episodes.append(
                        {
                            "run": name,
                            "step": step,
                            "task_id": task["task_id"],
                            "episode_index": ep["episode_index"],
                            "init_state_index": ep["init_state_index"],
                            "seed": ep["seed"],
                            "success": int(ep["success"]),
                            "control_steps": ep["control_steps"],
                            "action_clipping_fraction": ep["action_clipping_fraction"],
                        }
                    )
            plan = sorted(plan)
            if reference_plan is None:
                reference_plan = plan
            assert plan == reference_plan and count == result["successes"]
            successes[step] = count
            curve.append(
                {
                    "run": name,
                    "world_weight": spec["world_weight"],
                    "rec_weight": spec["reconstruction_weight"],
                    "dyn_weight": spec["dynamics_weight"],
                    "step": step,
                    "successes": count,
                    "episodes": 100,
                    "success_percent": count,
                    "action_clipping_fraction": result["action_clipping_fraction"],
                    "evaluation_run_id": receipt["id"],
                    "result_sha256": receipt["result_sha256"],
                }
            )
        best_step = max(STEPS, key=lambda step: (successes[step], -step))
        metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        # The recovery script rewrote canonical metrics to checkpoint boundary then appended resumed steps.
        # Superseded interrupted_metrics.jsonl remains in raw/ for history, not mixed into this curve.
        val = [row for row in metrics if "validation/fm_loss" in row]
        assert len({row["step"] for row in val}) == len(val) == 60
        assert sorted(row["step"] for row in val) == list(range(500, 30001, 500))
        for row in val:
            validation_rows.append(
                {
                    "run": name,
                    "step": row["step"],
                    **{key: row[key] for key in sorted(row) if key.startswith("validation/")},
                }
            )
        diagnostic = [row for row in metrics if "gradient/world_to_fm_ratio" in row and row["step"] >= 20000]
        assert len(diagnostic) == 21
        for row in diagnostic:
            diagnostic_rows.append(
                {
                    "run": name,
                    "step": row["step"],
                    **{key: row[key] for key in sorted(row) if key.startswith("gradient/")},
                }
            )
        final = next(row for row in val if row["step"] == 30000)
        rec = final["validation/world/world_rec"]
        dyn = final["validation/world/world_dyn"]
        prior = final["validation/world/world_prior"]
        rec_contrib = spec["reconstruction_weight"] * rec
        dyn_contrib = spec["dynamics_weight"] * dyn
        prior_contrib = experiment["prior_weight"] * prior
        assert abs(rec_contrib + dyn_contrib + prior_contrib - final["validation/world_loss"]) < 1e-6
        def median(key, rows=diagnostic):
            return statistics.median(row[key] for row in rows)
        summaries.append(
            {
                "run": name,
                "phase": spec["phase"],
                "world_weight": spec["world_weight"],
                "rec_weight": spec["reconstruction_weight"],
                "dyn_weight": spec["dynamics_weight"],
                **{f"success_{step // 1000}k": successes[step] for step in STEPS},
                "late_successes_sum": sum(successes[s] for s in LATE),
                "late_mean_success_percent": sum(successes[s] for s in LATE) / 3,
                "best_checkpoint_step": best_step,
                "best_success_percent": successes[best_step],
                "resumed_from_25k": recovered.exists(),
                "validation_points": len(val),
                "final_val_fm": final["validation/fm_loss"],
                "final_val_action_l1": final["validation/sampled_action_l1"],
                "final_val_rec": rec,
                "final_val_dyn": dyn,
                "final_val_psnr": final["validation/world/world_psnr"],
                "weighted_rec_contribution": rec_contrib,
                "weighted_dyn_contribution": dyn_contrib,
                "weighted_prior_contribution": prior_contrib,
                "late_encoder_world_to_fm_median": median("gradient/world_to_fm_ratio"),
                "late_encoder_cosine_median": median("gradient/cosine"),
                "late_encoder_diagnostic_points": len(diagnostic),
                "final_checkpoint": str(
                    Path(queue["configuration"]["root"]) / "runs" / name / "checkpoints/step_030000"
                ),
            }
        )
    assert len(episodes) == 3600 and len(tasks) == 360 and len(curve) == 36
    phase1 = sorted(
        (s for s in summaries if s["phase"] == 1),
        key=lambda s: (-s["late_successes_sum"], -s["success_30k"], s["world_weight"]),
    )
    assert phase1[0]["run"] == verified["phase1_winner_recomputed"] == queue["selection"]["winner"]
    best = max(summaries, key=lambda s: (s["late_successes_sum"], s["success_30k"]))
    reference = {
        (ep["task_id"], ep["episode_index"]): ep["success"]
        for ep in episodes
        if ep["run"] == best["run"] and ep["step"] == 30000
    }
    paired = []
    for summary in summaries:
        candidate = {
            (ep["task_id"], ep["episode_index"]): ep["success"]
            for ep in episodes
            if ep["run"] == summary["run"] and ep["step"] == 30000
        }
        paired.append(
            {
                "reference": best["run"],
                "candidate": summary["run"],
                "reference_only_success": sum(reference[k] and not candidate[k] for k in reference),
                "candidate_only_success": sum(candidate[k] and not reference[k] for k in reference),
                "both_success": sum(reference[k] and candidate[k] for k in reference),
                "both_failure": sum(not reference[k] and not candidate[k] for k in reference),
            }
        )
    out = directory / "analysis"
    out.mkdir(exist_ok=True)
    for filename, rows in (
        ("success_curve.csv", curve),
        ("per_episode_success.csv", episodes),
        ("per_task_success.csv", tasks),
        ("run_summary.csv", summaries),
        ("validation_curve.csv", validation_rows),
        ("late_encoder_gradients.csv", diagnostic_rows),
        ("paired_final_counts.csv", paired),
    ):
        write_csv(out / filename, rows)
    report = {
        "queue_status": "completed",
        "completed_local": datetime.fromtimestamp(
            queue["finished_unix"], ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "runs": summaries,
        "phase1_selected": phase1[0]["run"],
        "selected_overall": best["run"],
        "evaluation_count": len(curve),
        "episode_evaluation_count": len(episodes),
        "unique_validation_initializations": len(reference_plan),
        "validation_metric_points": len(validation_rows),
        "all_episode_plans_paired": True,
        "training_seeds": [42],
        "final_pool_evaluated": False,
        "paired_final_counts": paired,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# L40S LPWM-FM B sweep results",
        "",
        f"Completed: {report['completed_local']}",
        "",
        "|run|world|rec|dyn|5k|10k|15k|20k|25k|30k|late mean|best|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        values = "|".join(str(s[f"success_{step // 1000}k"]) for step in STEPS)
        lines.append(
            f"|{s['run']}|{s['world_weight']}|{s['rec_weight']}|{s['dyn_weight']}|{values}|{s['late_mean_success_percent']:.2f}%|{s['best_success_percent']}% @{s['best_checkpoint_step']}|"
        )
    lines += [
        "",
        "Each checkpoint=100episodes (10tasks×10). Late mean uses20k,25k,30k as preregistered.",
        "All3600episode evaluations use the same100paired validation initializations; they are NOT3600independent held-out states.",
        "Single trainingseed42. No final-pool evaluation. No statistical superiority established.",
        "Initial pair0.03/0.3 resumed from25kafterbilling interruption; original/resumed histories retained.",
        "",
        "## Encoder gradient diagnostic (median of21last-microbatch samples at20k..30k)",
        "|run|weighted-world / FM encoder norm|cosine|val FM30k|val rec30k|val dyn30k|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"|{s['run']}|{s['late_encoder_world_to_fm_median']:.4f}|{s['late_encoder_cosine_median']:.4f}|{s['final_val_fm']:.6f}|{s['final_val_rec']:.6f}|{s['final_val_dyn']:.8f}|"
        )
    lines += [
        "",
        "Gradient samples compare combined weighted(rec+dyn+prior)world vsFM encoder gradients, NOT separate rec/dyn gradient contributions.",
        "Scalar losses differ in scale; equal coefficients are not equal gradient strength.",
        "Low dynamicKL alone is not proof of useful dynamics or latent collapse; no standalone collapse/forecast-quality diagnosis was run.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = summarize(args.directory)
    print(json.dumps({k: v for k, v in report.items() if k not in ("runs", "paired_final_counts")}, indent=2))


if __name__ == "__main__":
    main()
