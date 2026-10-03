"""Summarize the predeclared stability screen without promoting any production training."""

import argparse
import json
import time
from pathlib import Path


def collect(root):
    """Report completed and running arms together; partial results remain explicitly partial."""
    plan = json.loads((root / "plan/plan.json").read_text())
    rows = []
    for name in plan["cases"]:
        directory = root / "results" / name
        summary, failed, progress = (
            directory / "summary.json",
            directory / "FAILED.json",
            directory / "progress.json",
        )
        if summary.exists():
            row = json.loads(summary.read_text())
        elif failed.exists():
            row = {"case": name, "status": "failed", **json.loads(failed.read_text())}
        elif progress.exists():
            row = {"case": name, **json.loads(progress.read_text())}
        else:
            row = {"case": name, "status": "queued_or_initializing"}
        if (directory / "swanlab_run.json").exists():
            row["swanlab"] = json.loads((directory / "swanlab_run.json").read_text())
        rows.append(row)
    terminal = {"completed_window", "guard_stopped", "failed"}
    result = {
        "status": "complete" if all(r["status"] in terminal for r in rows) else "running",
        "arms": rows,
        "passes_gradient_screen": [r["case"] for r in rows if r.get("passes_gradient_screen")],
        "production_restarted": False,
        "limits": "one seed, continuation from5k to9k; fixed panel is training data; not proof of45k stability or policy quality",
    }
    temporary = root / "comparison.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(root / "comparison.json")
    lines = [
        "# Q stability comparison",
        "",
        f"Status: {result['status']}",
        "",
        "| Case | Status | Final step | Tail loss | Tail gradient p95 | Tail clip fraction | Fixed-target MAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    def fmt(value):
        return f"{value:.5g}" if isinstance(value, (int, float)) else "—"

    for row in rows:
        values = [
            row.get("final_step", row.get("step")),
            row.get("tail500_loss_median"),
            row.get("tail500_grad_p95"),
            row.get("tail500_clip_fraction"),
            row.get("final_monitor", {}).get("monitor/fixed_target_mae"),
        ]
        lines.append(f"| {row['case']} | {row['status']} | " + " | ".join(fmt(v) for v in values) + " |")
    lines.extend(["", result["limits"], "", "No production restart or automatic winner selection."])
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    """Take a snapshot or wait for all queues, producing a final report without retraining."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    while True:
        result = collect(args.root)
        if not args.watch or result["status"] == "complete":
            print(json.dumps(result, indent=2), flush=True)
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
