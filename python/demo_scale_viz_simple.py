#!/usr/bin/env python3
"""
Simple scale field visualization for non-rigid registration.

For REAL point cloud data (PLY files), shows:
1. Source keypoints colored by RBF scale estimate
2. Correspondence lines connecting matches

For SYNTHETIC data, use this pattern:
- Add small jitter: points += np.random.randn(*points.shape) * 0.001
- Use larger voxel_size (0.2-0.3) to get enough features
- Or use RIGID registration instead (nonrigid needs rich geometry)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent))

from inlier.pcr import PCRConfig, register_nonrigid
import matplotlib.pyplot as plt


def visualize_scale_field(
    source: np.ndarray,  # N x 3
    keypoints: np.ndarray,  # M x 3 (subset of source, or downsampled)
    scales: np.ndarray,  # M scales, one per keypoint
    title: str = "Scale Field Visualization",
):
    """Visualize scale field on keypoints with color mapping."""

    # Create point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)
    source_pcd.paint_uniform_color([0.8, 0.8, 0.8])  # Gray background

    # Create keypoint spheres colored by scale
    min_scale = np.percentile(scales, 5)
    max_scale = np.percentile(scales, 95)
    mean_scale = scales.mean()

    # Normalize to [0, 1]
    norm_scales = (scales - min_scale) / (max_scale - min_scale + 1e-8)
    norm_scales = np.clip(norm_scales, 0, 1)

    # Apply colormap
    cmap = plt.cm.coolwarm
    colors = cmap(norm_scales)[:, :3]

    # Create spheres
    spheres = []
    for i, kp in enumerate(keypoints):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(kp)
        sphere.paint_uniform_color(colors[i])
        sphere.compute_vertex_normals()
        spheres.append(sphere)

    print("\n📊 Scale Field Statistics:")
    print(f"   Keypoints: {len(keypoints)}")
    print(f"   Scale range: [{scales.min():.3f}, {scales.max():.3f}]")
    print(f"   Mean: {mean_scale:.3f} ± {scales.std():.3f}")
    print(
        f"   Color map: {min_scale:.3f} (blue) → {mean_scale:.3f} (white) → {max_scale:.3f} (red)"
    )
    print("\n   Blue = shrinking, White = neutral, Red = expanding")

    # Visualize
    geometries = [source_pcd] + spheres[:100]  # Limit to 100 spheres for performance
    o3d.visualization.draw_geometries(
        geometries, window_name=title, width=1280, height=720
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize RBF scale field from non-rigid registration"
    )
    parser.add_argument("source", type=str, help="Source point cloud (.ply)")
    parser.add_argument("target", type=str, help="Target point cloud (.ply)")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.1,
        help="Voxel size for downsampling (default: 0.1)",
    )
    parser.add_argument(
        "--noise-bound",
        type=float,
        default=0.3,
        help="ROBIN noise bound (default: 0.3)",
    )

    args = parser.parse_args()

    # Load point clouds
    print("Loading point clouds...")
    source_pcd = o3d.io.read_point_cloud(args.source)
    target_pcd = o3d.io.read_point_cloud(args.target)

    source_np = np.asarray(source_pcd.points, dtype=np.float64)
    target_np = np.asarray(target_pcd.points, dtype=np.float64)

    print(f"Source: {len(source_np)} points")
    print(f"Target: {len(target_np)} points")

    # Configure non-rigid registration
    config = PCRConfig(
        voxel_size=args.voxel_size,
        feature_radius=args.voxel_size * 3,
        noise_bound=args.noise_bound,
        use_scale_invariant_features=True,  # SIPFH for nonrigid
    )

    print("\nRunning non-rigid registration...")
    print(f"  voxel_size={args.voxel_size}")
    print(f"  feature_radius={config.feature_radius}")
    print(f"  noise_bound={args.noise_bound}")

    result = register_nonrigid(source_np, target_np, config=config)

    if result is None:
        print("\n❌ Registration failed!")
        print("\nTips:")
        print("  - Try larger --noise-bound (0.5 or 1.0)")
        print("  - Try larger --voxel-size for faster computation")
        print("  - Check if point clouds have sufficient overlap")
        return 1

    print("\n✅ Registration successful!")
    print(
        f"   Inliers: {result['inlier_count']}/{result['total_correspondences']} "
        + f"({result['inlier_count'] / result['total_correspondences'] * 100:.1f}%)"
    )

    if "source_scales" not in result:
        print("\n❌ No scale field data in result!")
        return 1

    # Extract data
    keypoints = result["source_keypoints"]  # Keypoints that have scales
    scales = result["source_scales"]

    # Visualize
    visualize_scale_field(
        source_np, keypoints, scales, title=f"Scale Field - {Path(args.source).name}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
