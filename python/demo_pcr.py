#!/usr/bin/env python3
"""Example: Non-rigid point cloud registration from Python.

This script demonstrates how to use the Python API for non-rigid
point cloud registration with scale variation.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inlier.pcr import register_rigid, register_nonrigid, load_ply


def demo_synthetic():
    """Demo with synthetic data - rigid transformation."""
    print("=" * 60)
    print("Demo 1: Rigid Registration (Synthetic Data)")
    print("=" * 60)

    # Generate random source point cloud
    np.random.seed(42)
    src = np.random.rand(500, 3) * 10.0

    # Apply known transformation: rotation + translation + scale
    angle = np.pi / 6  # 30 degrees
    R_true = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    t_true = np.array([2.0, 1.0, 0.5])
    scale_true = 1.1

    dst = (src @ R_true.T) * scale_true + t_true

    # Add some noise
    dst += np.random.randn(*dst.shape) * 0.01

    print(f"Source: {src.shape[0]} points")
    print(f"Target: {dst.shape[0]} points")
    print("\nGround truth:")
    print(f"  Scale: {scale_true}")
    print(f"  Translation: {t_true}")

    # Register
    result = register_rigid(
        src,
        dst,
        voxel_size=0.2,
        feature_radius=1.0,
        normal_radius=0.5,
    )

    if result:
        print("\n✓ Registration succeeded!")
        print(f"  Estimated scale: {result['scale']:.4f}")
        print(f"  Estimated translation: {result['translation']}")
        print(
            f"  Inliers: {result['inlier_count']}/{result['total_correspondences']} "
            f"({100 * result['inlier_count'] / result['total_correspondences']:.1f}%)"
        )

        R_est = result["rotation"]
        print(f"\n  Rotation error: {np.linalg.norm(R_est - R_true):.4f}")
        print(
            f"  Translation error: {np.linalg.norm(result['translation'] - t_true):.4f}"
        )
        print(f"  Scale error: {abs(result['scale'] - scale_true):.4f}")
    else:
        print("\n✗ Registration failed!")

    return result is not None


def demo_nonrigid_synthetic():
    """Demo with synthetic non-rigid deformation."""
    print("\n" + "=" * 60)
    print("Demo 2: Non-Rigid Registration (Synthetic Deformation)")
    print("=" * 60)

    # Generate source point cloud
    np.random.seed(123)
    src = np.random.rand(800, 3) * 10.0

    # Apply spatially-varying scale (simulates thermal expansion)
    # Scale varies from 0.95 to 1.15 based on x-coordinate
    scale_field = 0.95 + 0.2 * (src[:, 0] / 10.0)
    dst = src * scale_field[:, None]

    # Add rigid transformation
    dst = dst @ np.eye(3) + np.array([1.0, 0.5, 0.0])

    # Add noise
    dst += np.random.randn(*dst.shape) * 0.02

    print(f"Source: {src.shape[0]} points")
    print(f"Target: {dst.shape[0]} points")
    print(
        f"\nGround truth scale range: {scale_field.min():.3f} - {scale_field.max():.3f}"
    )
    print(f"Ground truth scale std: {scale_field.std():.3f}")

    # Register with SIPFH (scale-invariant features)
    result = register_nonrigid(
        src,
        dst,
        voxel_size=0.3,
        feature_radius=1.5,
        normal_radius=0.8,
        use_scale_invariant_features=True,
    )

    if result:
        print("\n✓ Registration succeeded!")
        print(f"  Mean scale: {result['mean_scale']:.4f}")
        print(f"  Scale std: {result['scale_std']:.4f}")
        print(f"  Translation: {result['translation']}")
        print(
            f"  Inliers: {result['inlier_count']}/{result['total_correspondences']} "
            f"({100 * result['inlier_count'] / result['total_correspondences']:.1f}%)"
        )

        print(
            f"\n  Scale std error: {abs(result['scale_std'] - scale_field.std()):.4f}"
        )
    else:
        print("\n✗ Registration failed!")

    return result is not None


def demo_ply_files(src_path: str, dst_path: str):
    """Demo with real PLY files."""
    print("\n" + "=" * 60)
    print("Demo 3: Non-Rigid Registration (PLY Files)")
    print("=" * 60)

    try:
        print(f"Loading {src_path}...")
        src = load_ply(src_path)
        print(f"Loading {dst_path}...")
        dst = load_ply(dst_path)
    except Exception as e:
        print(f"✗ Failed to load PLY files: {e}")
        print("  Install plyfile: pip install plyfile")
        return False

    print(f"\nSource: {src.shape[0]} points")
    print(f"Target: {dst.shape[0]} points")

    # Register with SIPFH
    result = register_nonrigid(
        src,
        dst,
        voxel_size=0.05,
        feature_radius=0.3,
        normal_radius=0.15,
        use_scale_invariant_features=True,
    )

    if result:
        print("\n✓ Registration succeeded!")
        print(f"  Mean scale: {result['mean_scale']:.4f}")
        print(
            f"  Scale variation: {result['scale_std']:.4f} ({100 * result['scale_std'] / result['mean_scale']:.1f}%)"
        )
        print(
            f"  Inliers: {result['inlier_count']}/{result['total_correspondences']} "
            f"({100 * result['inlier_count'] / result['total_correspondences']:.1f}%)"
        )
    else:
        print("\n✗ Registration failed!")

    return result is not None


def main():
    """Run all demos."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  Point Cloud Registration - Python Examples            ║")
    print("╚" + "═" * 58 + "╝")

    success_count = 0

    # Demo 1: Rigid registration with synthetic data
    if demo_synthetic():
        success_count += 1

    # Demo 2: Non-rigid registration with synthetic deformation
    if demo_nonrigid_synthetic():
        success_count += 1

    # Demo 3: Real PLY files (if provided)
    if len(sys.argv) >= 3:
        if demo_ply_files(sys.argv[1], sys.argv[2]):
            success_count += 1
    else:
        print("\n" + "=" * 60)
        print("Demo 3: Skipped (no PLY files provided)")
        print("=" * 60)
        print("Usage: python demo_pcr.py <source.ply> <target.ply>")

    # Summary
    print("\n" + "=" * 60)
    print(f"Summary: {success_count}/2+ demos completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
