#!/usr/bin/env python3
"""
Render the homography example scene with matplotlib and real images from inlier-data.

Usage:
    cargo run --example homography_estimation
    python python/demo_homography_scene.py

This script uses the Agg backend so it can render offscreen and save directly to
an image file without requiring a display server.

Development workflow:
    1. Clone the separate inlier-data repository.
    2. Add or update large binary image assets there and push them.
    3. Run this script from inlier; it will use the local clone when available
       and otherwise fall back to the pooch-managed fetch path.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the homography estimation scene")
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "examples"
        / "assets"
        / "homography",
        help="Directory containing generated homography report files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path (defaults to homography_matches.png under asset-dir)",
    )
    return parser.parse_args()


def import_inlier_data():
    root = Path(__file__).resolve().parent.parent
    fallback_dirs = [
        root.parent / "inlier-data",
        root / "inlier-data",
    ]
    for candidate in fallback_dirs:
        testdata_dir = candidate / "testdata"
        if testdata_dir.exists():
            os.environ.setdefault("INLIER_DATA_DIR", str(testdata_dir))
            break

    try:
        return importlib.import_module("inlier_data")
    except ImportError:
        for candidate in fallback_dirs:
            module_file = candidate / "inlier_data.py"
            if module_file.exists():
                sys.path.insert(0, str(candidate))
                return importlib.import_module("inlier_data")
        raise


def fetch_images() -> tuple[Path, Path]:
    module = import_inlier_data()
    local_base = os.environ.get("INLIER_DATA_DIR")
    if local_base:
        local_a = Path(local_base) / "homography_sacre_coeur_a.jpg"
        local_b = Path(local_base) / "homography_sacre_coeur_b.jpg"
        if local_a.exists() and local_b.exists():
            return local_a, local_b
    return (
        Path(module.TEST_DATA.fetch("homography_sacre_coeur_a.jpg")),
        Path(module.TEST_DATA.fetch("homography_sacre_coeur_b.jpg")),
    )


def read_matrix(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            rows.append([float(x) for x in stripped.split()])
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError(f"invalid homography matrix file: {path}")
    return rows


def read_report(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        (
            label,
            src_x,
            src_y,
            dst_x,
            dst_y,
            expected_inlier,
            estimated_inlier,
            reproj_error,
        ) = stripped.split("\t")
        rows.append(
            {
                "label": label,
                "src": (float(src_x), float(src_y)),
                "dst": (float(dst_x), float(dst_y)),
                "expected_inlier": expected_inlier == "true",
                "estimated_inlier": estimated_inlier == "true",
                "reproj_error": float(reproj_error),
            }
        )
    return rows


def project(h: list[list[float]], point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    den = h[2][0] * x + h[2][1] * y + h[2][2]
    return (
        (h[0][0] * x + h[0][1] * y + h[0][2]) / den,
        (h[1][0] * x + h[1][1] * y + h[1][2]) / den,
    )


def setup_axis(ax: plt.Axes, image: Image.Image, title: str) -> None:
    width, height = image.size
    ax.imshow(np.asarray(image))
    ax.set_title(title, color="#e2e8f0", fontsize=13)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def annotate_points(ax: plt.Axes, rows: list[dict[str, object]], side: str) -> None:
    xs = [row[side][0] for row in rows]
    ys = [row[side][1] for row in rows]
    colors = ["#22c55e" if row["estimated_inlier"] else "#ef4444" for row in rows]
    ax.scatter(xs, ys, s=28, c=colors, edgecolors="white", linewidths=0.7)
    for row in rows:
        x, y = row[side]
        ax.text(
            x + 6,
            y - 6,
            str(row["label"]),
            color="#f8fafc",
            fontsize=7.5,
            family="monospace",
            bbox={
                "facecolor": "#0f172a",
                "alpha": 0.55,
                "pad": 1.5,
                "edgecolor": "none",
            },
        )


def main() -> int:
    args = parse_args()
    asset_dir = args.asset_dir
    matrix_path = asset_dir / "estimated_homography_matrix.txt"
    report_path = asset_dir / "estimated_match_report.tsv"
    output_path = args.output or asset_dir / "homography_matches.png"

    if not matrix_path.exists() or not report_path.exists():
        raise SystemExit(
            f"Missing generated homography report files in {asset_dir}. "
            "Run `cargo run --example homography_estimation` first."
        )

    image_a_path, image_b_path = fetch_images()
    image_a = Image.open(image_a_path).convert("RGB")
    image_b = Image.open(image_b_path).convert("RGB")
    matrix = read_matrix(matrix_path)
    rows = read_report(report_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#0b1020")
    for ax in axes:
        ax.set_facecolor("#0b1020")

    setup_axis(axes[0], image_a, image_a_path.name)
    setup_axis(axes[1], image_b, image_b_path.name)
    annotate_points(axes[0], rows, "src")
    annotate_points(axes[1], rows, "dst")

    fig.suptitle("Homography estimation showcase", color="#f8fafc", fontsize=18)
    fig.text(
        0.5,
        0.02,
        "Green arrows are estimated inliers. Red dashed arrows are rejected tentative matches. "
        "Input images are fetched from inlier-data via pooch.",
        ha="center",
        color="#94a3b8",
        fontsize=10,
    )

    for row in rows:
        src_disp = axes[0].transData.transform(row["src"])
        dst_disp = axes[1].transData.transform(row["dst"])
        src_fig = fig.transFigure.inverted().transform(src_disp)
        dst_fig = fig.transFigure.inverted().transform(dst_disp)
        color = "#22c55e" if row["estimated_inlier"] else "#ef4444"
        linestyle = "-" if row["estimated_inlier"] else (0, (4, 3))
        arrow = FancyArrowPatch(
            src_fig,
            dst_fig,
            transform=fig.transFigure,
            arrowstyle="-",
            linewidth=1.5,
            linestyle=linestyle,
            color=color,
            alpha=0.75,
        )
        fig.add_artist(arrow)

    source_corners = [
        (0.0, 0.0),
        (image_a.width - 1.0, 0.0),
        (image_a.width - 1.0, image_a.height - 1.0),
        (0.0, image_a.height - 1.0),
    ]
    projected_outline = [project(matrix, point) for point in source_corners]
    axes[1].add_patch(
        Polygon(
            projected_outline,
            closed=True,
            fill=False,
            edgecolor="#38bdf8",
            linewidth=2.2,
            linestyle=(0, (6, 4)),
        )
    )
    axes[1].text(
        14,
        image_b.height - 18,
        "projected source image footprint",
        color="#38bdf8",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "#0f172a", "alpha": 0.6, "pad": 2.0, "edgecolor": "none"},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)
    print(f"Fetched image A from: {image_a_path}")
    print(f"Fetched image B from: {image_b_path}")
    print(f"Saved homography showcase to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
