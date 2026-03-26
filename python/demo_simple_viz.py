#!/usr/bin/env python3
"""
Simple visualization of point cloud registration results using Open3D.

This demonstrates how to:
1. Load two point clouds
2. Perform registration (rigid or non-rigid) using inlier library
3. Visualize before/after with Open3D

Usage:
    python demo_simple_viz.py <source.ply> <target.ply> [--nonrigid]

Requirements:
    pip install open3d numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from inlier.pcr import PCRConfig, register_nonrigid, register_rigid


def visualize_point_clouds(
    geometries, window_name="Point Clouds", width=1280, height=720
):
    """Simple wrapper for Open3D visualization."""
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=width,
        height=height,
        point_show_normal=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud registration")
    parser.add_argument("source", type=str, help="Source point cloud (.ply)")
    parser.add_argument("target", type=str, help="Target point cloud (.ply)")
    parser.add_argument(
        "--nonrigid", action="store_true", help="Use non-rigid registration"
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.05, help="Voxel size (default: 0.05)"
    )
    parser.add_argument(
        "--noise-bound",
        type=float,
        default=0.2,
        help="Noise bound - increase to retain more inliers (default: 0.2, try 0.3-0.5 for noisy data)",
    )

    args = parser.parse_args()

    # Load point clouds
    print(f"Loading: {args.source}")
    source = o3d.io.read_point_cloud(args.source)
    print(f"  → {len(source.points)} points")

    print(f"Loading: {args.target}")
    target = o3d.io.read_point_cloud(args.target)
    print(f"  → {len(target.points)} points")

    # Convert to numpy
    source_np = np.asarray(source.points)
    target_np = np.asarray(target.points)

    # Configure registration
    config = PCRConfig(
        voxel_size=args.voxel_size,
        normal_radius=args.voxel_size * 3,
        feature_radius=args.voxel_size * 6,
        noise_bound=args.noise_bound,
    )

    # Show BEFORE registration
    print("\n" + "=" * 60)
    print("BEFORE registration (press Q to continue)")
    print("=" * 60)
    print("  Blue   = Source")
    print("  Orange = Target")

    source_vis = o3d.geometry.PointCloud(source)
    target_vis = o3d.geometry.PointCloud(target)
    source_vis.paint_uniform_color([0.0, 0.5, 1.0])  # Blue
    target_vis.paint_uniform_color([1.0, 0.5, 0.0])  # Orange

    visualize_point_clouds([source_vis, target_vis], "BEFORE Registration")

    # Perform registration
    method = "non-rigid" if args.nonrigid else "rigid"
    print(f"\nRunning {method} registration...")

    if args.nonrigid:
        result = register_nonrigid(source_np, target_np, config)
    else:
        result = register_rigid(source_np, target_np, config)

    if result is None:
        print("❌ Registration failed!")
        return 1

    # Print results
    print(f"\n✅ {method.capitalize()} registration succeeded!")
    print(f"  Initial correspondences: {result['total_correspondences']}")
    print(f"  Final inliers: {result['inlier_count']}")
    print(
        f"  Inlier ratio: {result['inlier_count'] / result['total_correspondences'] * 100:.1f}%"
    )

    if args.nonrigid:
        print(f"\n  Mean scale: {result['mean_scale']:.6f}")
        print(f"  Scale std dev: {result['scale_std']:.6f}")
        print(
            f"  Scale variation: {result['scale_std'] / result['mean_scale'] * 100:.1f}%"
        )
    else:
        # Extract transformation for rigid registration
        R = np.array(result["rotation"]).reshape(3, 3)
        t = np.array(result["translation"])
        s = result["scale"]

        print(f"\n  Scale: {s:.6f}")
        print(f"  Translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
        print("  Rotation (degrees):")
        # Compute Euler angles for display
        import math

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        print(f"    X: {math.degrees(x):.2f}°")
        print(f"    Y: {math.degrees(y):.2f}°")
        print(f"    Z: {math.degrees(z):.2f}°")

        # Apply transformation to source
        T = np.eye(4)
        T[:3, :3] = s * R
        T[:3, 3] = t
        source_transformed = o3d.geometry.PointCloud(source)
        source_transformed.transform(T)

        # Show AFTER registration
        print("\n" + "=" * 60)
        print("AFTER rigid registration (press Q to close)")
        print("=" * 60)
        print("  Blue   = Transformed source")
        print("  Orange = Target")

        source_vis = o3d.geometry.PointCloud(source_transformed)
        target_vis = o3d.geometry.PointCloud(target)
        source_vis.paint_uniform_color([0.0, 0.5, 1.0])
        target_vis.paint_uniform_color([1.0, 0.5, 0.0])

        visualize_point_clouds([source_vis, target_vis], "AFTER Rigid Registration")

    if args.nonrigid:
        print("\nNote: Non-rigid transformation cannot be visualized as a single")
        print("      transformation matrix. Each point has its own local scale.")
        print("      To visualize non-rigid results, the PCR API would need to")
        print("      return the transformed source points directly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
