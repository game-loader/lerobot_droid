"""Render per-observation attention on its actual cached RGB, with an explicit common scale."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

TASK_NAMES = {0: "Drawer", 1: "Stack bowls", 2: "Fold towel", 3: "Dual-arm sorting"}


def main():
    """Render RGB references, CLS maps and all-query maps without changing their probabilities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--vmax-percent", type=float, default=1.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.input / "metadata.json").read_text())
    probabilities = np.load(args.input / "probabilities.npz")
    side = meta["patch_grid"][0]
    cmap = LinearSegmentedColormap.from_list("muted_heat", ["#f4e4c8", "#cda46b", "#9b594a"])
    norm = Normalize(0, args.vmax_percent, clip=True)
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})
    statistics = []
    for example_index in (0, 1):
        selected = [
            (i, sample) for i, sample in enumerate(meta["examples"]) if sample["example"] == example_index
        ]
        for mode in ("rgb", "cls", "valid_query_mean"):
            fig, axes = plt.subplots(len(selected), 3, figsize=(12.5, 11.5), layout="constrained")
            for row, (sample_index, sample) in enumerate(selected):
                if mode != "rgb":
                    values = probabilities[f"layer{args.layer}_{mode}"][sample_index].mean(0)
                    maps = values[meta["text_slots"] :].reshape(3, side, side)
                for view, image_info in enumerate(sample["images"]):
                    ax = axes[row, view]
                    pixels = np.asarray(Image.open(args.input / image_info["file"]))
                    height, width = pixels.shape[:2]
                    ax.imshow(pixels)
                    camera = image_info["camera"].split(".")[-1]
                    label = f"{camera} | episode {sample['episode']}, frame {sample['frame']}"
                    if mode != "rgb":
                        weights = maps[view] * 100
                        # Interpolation is for display only; the actual probabilities remain 14x14.
                        high = np.asarray(
                            Image.fromarray(weights.astype(np.float32)).resize(
                                (width, height), Image.Resampling.BILINEAR
                            )
                        )
                        rgba = cmap(norm(high))
                        rgba[..., 3] = 0.65 * np.clip(high / args.vmax_percent, 0, 1) ** 0.65
                        ax.imshow(rgba)
                        top = np.argsort(weights.flatten())[-3:][::-1]
                        for rank, position in enumerate(top, start=1):
                            patch_row, patch_col = np.unravel_index(position, weights.shape)
                            left, bottom = patch_col * width / side, patch_row * height / side
                            ax.add_patch(
                                Rectangle(
                                    (left, bottom),
                                    width / side,
                                    height / side,
                                    fill=False,
                                    edgecolor="#943d37",
                                    linewidth=1.6 if rank == 1 else 0.9,
                                )
                            )
                            ax.text(
                                left + 2,
                                bottom + 2,
                                str(rank),
                                color="white",
                                fontsize=8,
                                va="top",
                                bbox={"facecolor": "#943d37", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                            )
                        label += f"\nview mass {weights.sum():.1f}% | peak patch {weights.max():.2f}%"
                        statistics.append(
                            {
                                "sample_index": sample_index,
                                "task_index": sample["task_index"],
                                "episode": sample["episode"],
                                "frame": sample["frame"],
                                "mode": mode,
                                "layer": args.layer,
                                "camera": camera,
                                "view_attention_mass": float(maps[view].sum()),
                                "top_patches": [
                                    {
                                        "row": int(position // side),
                                        "column": int(position % side),
                                        "attention_weight": float(maps[view].flatten()[position]),
                                    }
                                    for position in top
                                ],
                            }
                        )
                    ax.set_title(label, fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    if view == 0:
                        ax.set_ylabel(TASK_NAMES[sample["task_index"]], fontsize=12)
            if mode == "rgb":
                title = f"Actual observations | representative set {example_index + 1}"
            else:
                query = "CLS query" if mode == "cls" else "all valid queries"
                title = f"40k critic | layer {args.layer} cross-attention | {query}, mean over 16 heads"
                fig.colorbar(
                    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=axes.ravel().tolist(),
                    shrink=0.6,
                    label=f"Mean probability per patch (%) | common scale, clipped at {args.vmax_percent:g}%",
                )
                fig.supxlabel(
                    "Boxes 1-3 mark the largest patch weights in each camera, not all attention. "
                    "Attention weights are not causal Q attribution.",
                    fontsize=8,
                )
            fig.suptitle(title, fontsize=13)
            fig.savefig(args.output / f"layer{args.layer}_{mode}_examples_{example_index + 1}.png", dpi=150)
            plt.close(fig)
    (args.output / f"layer{args.layer}_spatial_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
