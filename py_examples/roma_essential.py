#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import inlier

try:
    from romatch import roma_outdoor
except ImportError as exc:
    raise SystemExit("romatch is required. Install with: uv pip install romatch") from exc


def load_matches(img_a: Path, img_b: Path, device: str):
    model = roma_outdoor(device=device)
    warp, certainty = model.match(str(img_a), str(img_b), device=device)
    matches, certainty = model.sample(warp, certainty)
    w_a, h_a = Image.open(img_a).size
    w_b, h_b = Image.open(img_b).size
    kpts_a, kpts_b = model.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
    return kpts_a.cpu().numpy(), kpts_b.cpu().numpy()


def load_intrinsics(path: Path):
    k = np.loadtxt(path)
    if k.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics in {path}")
    return k


def normalize_points(kpts, k):
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    x = (kpts[:, 0] - cx) / fx
    y = (kpts[:, 1] - cy) / fy
    return np.column_stack([x, y])


def main():
    parser = argparse.ArgumentParser(description="RoMa + inlier essential matrix example")
    parser.add_argument(
        "--img-a",
        default="data/02085496_6952371977.jpg",
        help="Path to first image",
    )
    parser.add_argument(
        "--img-b",
        default="data/02928139_3448003521.jpg",
        help="Path to second image",
    )
    parser.add_argument("--k1", required=True, help="Path to intrinsics for image A (3x3)")
    parser.add_argument("--k2", required=True, help="Path to intrinsics for image B (3x3)")
    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu/cuda)")
    parser.add_argument("--threshold", type=float, default=1e-3, help="RANSAC threshold (norm.)")
    args = parser.parse_args()

    kpts_a, kpts_b = load_matches(Path(args.img_a), Path(args.img_b), args.device)
    k1 = load_intrinsics(Path(args.k1))
    k2 = load_intrinsics(Path(args.k2))
    kpts_a = normalize_points(kpts_a, k1)
    kpts_b = normalize_points(kpts_b, k2)

    result = inlier.estimate_essential_matrix_py(kpts_a, kpts_b, threshold=args.threshold)

    print("Essential matrix result:")
    print(f"  inliers: {len(result['inliers'])}")
    print(f"  score: {result['score']:.4f}")
    print("  E:")
    for row in result["model"]:
        print("   ", row)


if __name__ == "__main__":
    main()
