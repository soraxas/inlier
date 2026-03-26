#!/usr/bin/env python3
"""
Render the rigid-transform example alignment with headless matplotlib.

Usage:
    cargo run --example rigid_transform
    python python/demo_rigid_transform_scene.py

This script uses the Agg backend so it can render offscreen and save directly to
an image file without requiring a display server.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the rigid transform alignment")
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "examples"
        / "assets"
        / "rigid_transform",
        help="Directory containing generated rigid-transform report files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path (defaults to rigid_transform_alignment.png under asset-dir)",
    )
    return parser.parse_args()


def read_report(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = stripped.split("\t")
        rows.append(
            {
                "src": np.array([float(values[0]), float(values[1]), float(values[2])]),
                "dst": np.array([float(values[3]), float(values[4]), float(values[5])]),
                "pred": np.array(
                    [float(values[6]), float(values[7]), float(values[8])]
                ),
                "expected_inlier": values[9] == "true",
                "estimated_inlier": values[10] == "true",
                "residual": float(values[11]),
            }
        )
    return rows


def draw_projection(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    x_index: int,
    y_index: int,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    dst = np.array([row["dst"] for row in rows])
    pred = np.array([row["pred"] for row in rows])
    mask = np.array([row["estimated_inlier"] for row in rows], dtype=bool)

    ax.scatter(
        dst[:, x_index],
        dst[:, y_index],
        s=24,
        c="#94a3b8",
        alpha=0.55,
        label="Observed target points",
    )
    ax.scatter(
        pred[mask, x_index],
        pred[mask, y_index],
        s=34,
        c="#22c55e",
        edgecolors="white",
        linewidths=0.6,
        label="Transformed source (estimated inliers)",
    )
    if np.any(~mask):
        ax.scatter(
            pred[~mask, x_index],
            pred[~mask, y_index],
            s=30,
            c="#ef4444",
            edgecolors="white",
            linewidths=0.6,
            label="Transformed source (rejected matches)",
        )

    line_rows = [row for row in rows if row["estimated_inlier"]]
    for row in line_rows[:20]:
        ax.plot(
            [row["pred"][x_index], row["dst"][x_index]],
            [row["pred"][y_index], row["dst"][y_index]],
            color="#22c55e",
            alpha=0.25,
            linewidth=1.0,
        )

    ax.set_title(title, color="#e2e8f0", fontsize=13)
    ax.set_xlabel(xlabel, color="#cbd5e1")
    ax.set_ylabel(ylabel, color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.grid(color="#1e293b", alpha=0.65)
    for spine in ax.spines.values():
        spine.set_color("#334155")


def main() -> int:
    args = parse_args()
    asset_dir = args.asset_dir
    report_path = asset_dir / "estimated_correspondence_report.tsv"
    output_path = args.output or asset_dir / "rigid_transform_alignment.png"

    if not report_path.exists():
        raise SystemExit(
            f"Missing generated rigid-transform report file in {asset_dir}. "
            "Run `cargo run --example rigid_transform` first."
        )

    rows = read_report(report_path)
    if not rows:
        raise SystemExit(f"No correspondence rows found in {report_path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="#0b1020")
    for ax in axes:
        ax.set_facecolor("#0b1020")

    draw_projection(axes[0], rows, 0, 1, "Aligned rigid transform (XY)", "x", "y")
    draw_projection(axes[1], rows, 0, 2, "Aligned rigid transform (XZ)", "x", "z")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False, labelcolor="#cbd5e1"
    )

    mean_residual = np.mean(
        [row["residual"] for row in rows if row["estimated_inlier"]]
    )
    fig.suptitle("Rigid transform estimation showcase", color="#f8fafc", fontsize=18)
    fig.text(
        0.5,
        0.03,
        f"Green points are transformed source inliers after RANSAC. "
        f"Red points are rejected tentative matches. Mean inlier residual: {mean_residual:.4f}.",
        ha="center",
        color="#94a3b8",
        fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)
    print(f"Saved rigid-transform showcase to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
