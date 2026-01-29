#!/usr/bin/env python3
import argparse
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="RoMa + inlier fundamental matrix example")
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
    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu/cuda)")
    parser.add_argument("--threshold", type=float, default=1.0, help="RANSAC threshold (px)")
    args = parser.parse_args()

    kpts_a, kpts_b = load_matches(Path(args.img_a), Path(args.img_b), args.device)
    result = inlier.estimate_fundamental_matrix_py(kpts_a, kpts_b, threshold=args.threshold)

    print("Fundamental matrix result:")
    print(f"  inliers: {len(result['inliers'])}")
    print(f"  score: {result['score']:.4f}")
    print("  F:")
    for row in result["model"]:
        print("   ", row)


if __name__ == "__main__":
    main()
