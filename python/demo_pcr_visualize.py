#!/usr/bin/env python3
"""
Visualize point cloud registration results with Open3D.

Shows before/after registration with correspondence lines connecting
matched keypoints between source and target point clouds.

This script uses the ACTUAL correspondences from the inlier registration
pipeline (not Open3D's feature matching).

For NON-RIGID registration: source keypoints are colored by RBF scale field!
  - Blue = shrinking (scale < 1.0)
  - White = neutral (scale ≈ 1.0)
  - Red = expanding (scale > 1.0)
See SCALE_VISUALIZATION.md for interpretation guide.

Usage:
    # Rigid (uniform blue source)
    python demo_pcr_visualize.py <source.ply> <target.ply> --method rigid

    # Non-rigid (scale-colored source)
    python demo_pcr_visualize.py <source.ply> <target.ply> --method nonrigid

Requirements:
    pip install open3d numpy matplotlib
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from inlier.pcr import PCRConfig, register_nonrigid, register_rigid


def load_point_cloud(filepath: str) -> o3d.geometry.PointCloud:
    """Load point cloud from file."""
    pcd = o3d.io.read_point_cloud(filepath)
    if not pcd.has_points():
        raise ValueError(f"Failed to load point cloud from {filepath}")
    print(f"Loaded {len(pcd.points)} points from {filepath}")
    return pcd


def create_correspondence_lines(
    source_points: np.ndarray,
    target_points: np.ndarray,
    correspondences: np.ndarray,
    color: tuple = (1, 0, 0),
    use_cylinders: bool = False,
    line_radius: float = 0.01,
) -> o3d.geometry.LineSet:
    """
    Create LineSet showing correspondences between source and target.

    Args:
        source_points: Source keypoints (N, 3)
        target_points: Target keypoints (M, 3)
        correspondences: Correspondence indices (K, 2) where each row is [src_idx, tgt_idx]
        color: RGB color for lines (default: red)
        use_cylinders: Use cylinders instead of lines for thickness (slower but visible)
        line_radius: Radius of cylinder lines (only if use_cylinders=True)

    Returns:
        Open3D LineSet or list of cylinder meshes
    """
    # Combine points: first N are source, next M are target
    all_points = np.vstack([source_points, target_points])
    n_source = len(source_points)

    # Create line indices
    lines = []
    for src_idx, tgt_idx in correspondences:
        lines.append([src_idx, tgt_idx + n_source])

    if use_cylinders:
        # Create thick cylinders for better visibility
        cylinders = []
        for src_idx, tgt_idx in correspondences:
            p1 = source_points[src_idx]
            p2 = target_points[tgt_idx]

            # Create cylinder between points
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue

            # Create cylinder
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=line_radius, height=length
            )
            cylinder.paint_uniform_color(color)

            # Orient cylinder from p1 to p2
            direction = direction / length
            z_axis = np.array([0, 0, 1])

            # Rotation axis and angle
            rotation_axis = np.cross(z_axis, direction)
            rotation_axis_norm = np.linalg.norm(rotation_axis)

            if rotation_axis_norm > 1e-6:
                rotation_axis = rotation_axis / rotation_axis_norm
                angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))

                # Rodrigues' rotation formula
                K = np.array(
                    [
                        [0, -rotation_axis[2], rotation_axis[1]],
                        [rotation_axis[2], 0, -rotation_axis[0]],
                        [-rotation_axis[1], rotation_axis[0], 0],
                    ]
                )
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            else:
                # Already aligned or opposite
                R = np.eye(3) if np.dot(z_axis, direction) > 0 else np.diag([1, 1, -1])

            # Apply transformation
            cylinder.rotate(R, center=[0, 0, 0])
            cylinder.translate(p1 + direction * length / 2)

            cylinders.append(cylinder)

        return cylinders
    else:
        # Standard thin LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(all_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Set color
        colors = np.tile(color, (len(lines), 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set


def visualize_registration_result(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    source_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    correspondences: np.ndarray,
    transformation: np.ndarray = None,
    title: str = "Registration Result",
    source_scales: np.ndarray = None,
):
    """
    Visualize registration result with correspondence lines.

    Args:
        source: Source point cloud
        target: Target point cloud
        source_keypoints: Source keypoint coordinates (N, 3)
        target_keypoints: Target keypoint coordinates (M, 3)
        correspondences: Correspondence indices (K, 2)
        transformation: Optional 4x4 transformation matrix
        title: Window title
        source_scales: Optional per-keypoint scale values for coloring (nonrigid only)
    """
    # Clone and color point clouds
    source_vis = o3d.geometry.PointCloud(source)
    target_vis = o3d.geometry.PointCloud(target)

    # Color source keypoints by scale if provided (nonrigid), otherwise uniform blue
    if source_scales is not None:
        # Create a point cloud from keypoints to color by scale
        source_kp_pcd = o3d.geometry.PointCloud()
        source_kp_pcd.points = o3d.utility.Vector3dVector(source_keypoints)

        # Map scales to colors: blue (shrink) -> white (1.0) -> red (expand)
        # Use a coolwarm colormap
        import matplotlib.pyplot as plt

        # Compute scale range
        min_scale = max(0.5, np.percentile(source_scales, 5))  # Clamp to [0.5, 2.0]
        max_scale = min(2.0, np.percentile(source_scales, 95))
        mean_scale = np.mean(source_scales)

        # Normalize scales to [0, 1] range centered at mean
        norm_scales = (source_scales - min_scale) / (max_scale - min_scale)
        norm_scales = np.clip(norm_scales, 0, 1)

        # Apply coolwarm colormap
        cmap = plt.cm.coolwarm
        colors = cmap(norm_scales)[:, :3]  # Drop alpha

        # Color the source cloud uniformly (light blue)
        source_vis.paint_uniform_color([0.7, 0.85, 1.0])

        # Show scale distribution
        print("\n📊 Scale Field Statistics:")
        print(f"   Range: [{source_scales.min():.3f}, {source_scales.max():.3f}]")
        print(f"   Mean: {mean_scale:.3f}")
        print(f"   Std: {source_scales.std():.3f}")
        print(
            f"   Color mapping: {min_scale:.3f} (blue) → {mean_scale:.3f} (white) → {max_scale:.3f} (red)"
        )
    else:
        source_vis.paint_uniform_color([0.0, 0.5, 1.0])  # Blue

    target_vis.paint_uniform_color([1.0, 0.5, 0.0])  # Orange

    # Apply transformation if provided
    if transformation is not None:
        source_vis.transform(transformation)
        # Also transform source keypoints
        source_keypoints_h = np.hstack(
            [source_keypoints, np.ones((len(source_keypoints), 1))]
        )
        source_keypoints = (transformation @ source_keypoints_h.T).T[:, :3]

    # Create correspondence lines (green)
    # Use thick cylinders if few correspondences, thin lines otherwise
    n_corr = len(correspondences)
    use_thick = n_corr < 800  # Use cylinders for <100 correspondences
    line_radius = 0.02 if n_corr < 50 else 0.01  # Thicker for very few

    line_geom = create_correspondence_lines(
        source_keypoints,
        target_keypoints,
        correspondences,
        color=(0, 1, 0),
        use_cylinders=use_thick,
        line_radius=line_radius,
    )

    # Create keypoint visualizations (small spheres, limit to 100 for performance)
    keypoint_vis = []
    n_show = min(100, len(source_keypoints))

    if source_scales is not None:
        # Color source keypoints by scale
        for i, kp in enumerate(source_keypoints[:n_show]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere.translate(kp)
            sphere.paint_uniform_color(colors[i])  # Scale-based color
            keypoint_vis.append(sphere)
    else:
        # Uniform blue for source keypoints
        for kp in source_keypoints[:n_show]:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(kp)
            sphere.paint_uniform_color([0, 0, 1])  # Blue
            keypoint_vis.append(sphere)

    for kp in target_keypoints[: min(100, len(target_keypoints))]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(kp)
        sphere.paint_uniform_color([1, 0.5, 0])  # Orange
        keypoint_vis.append(sphere)

    # Visualize
    print(f"\n{title}")
    print("=" * len(title))
    if source_scales is not None:
        print(
            "Source keypoints colored by RBF scale (blue=shrink, white=1.0, red=expand)"
        )
        print("Orange: Target | Green: Correspondences")
    else:
        print("Blue: Source | Orange: Target | Green: Correspondences")
    print(f"Showing {len(correspondences)} correspondences")
    if use_thick:
        print(f"Using thick cylinders (radius={line_radius:.3f}) for better visibility")

    # Build geometry list
    geometries = [source_vis, target_vis]

    # Add lines (either LineSet or list of cylinders)
    if isinstance(line_geom, list):
        geometries.extend(line_geom)  # Cylinders
    else:
        geometries.append(line_geom)  # LineSet

    geometries.extend(keypoint_vis)

    o3d.visualization.draw_geometries(
        geometries, window_name=title, width=1280, height=720
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize PCR with correspondences")
    parser.add_argument("source", type=str, help="Source point cloud (.ply)")
    parser.add_argument("target", type=str, help="Target point cloud (.ply)")
    parser.add_argument(
        "--method",
        choices=["rigid", "nonrigid"],
        default="rigid",
        help="Registration method (default: rigid)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="Voxel size for downsampling (default: 0.05)",
    )
    parser.add_argument(
        "--normal-radius",
        type=float,
        default=0.15,
        help="Normal estimation radius (default: 0.15)",
    )
    parser.add_argument(
        "--feature-radius",
        type=float,
        default=0.3,
        help="Feature descriptor radius (default: 0.3)",
    )
    parser.add_argument(
        "--noise-bound",
        type=float,
        default=0.2,
        help="ROBIN/GNC noise bound - increase to retain more inliers (default: 0.2, try 0.3-0.5 for noisy data)",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.9,
        help="Feature matching ratio test (lower=stricter, default: 0.9)",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.5,
        help="Minimum allowed scale (nonrigid only, default: 0.5 = 50%% shrink)",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=2.0,
        help="Maximum allowed scale (nonrigid only, default: 2.0 = 200%% expand)",
    )

    args = parser.parse_args()

    # Load point clouds
    print("Loading point clouds...")
    source_o3d = load_point_cloud(args.source)
    target_o3d = load_point_cloud(args.target)

    # Convert to numpy
    source_np = np.asarray(source_o3d.points)
    target_np = np.asarray(target_o3d.points)

    source_np += np.random.randn(*source_np.shape) * 0.0001
    target_np += np.random.randn(*target_np.shape) * 0.0001

    # Configure registration
    if args.method == "nonrigid":
        # Non-rigid needs scale-invariant features (SIPFH)
        config = PCRConfig(
            voxel_size=args.voxel_size,
            normal_radius=args.normal_radius,
            feature_radius=args.feature_radius,
            ratio_threshold=args.ratio_threshold,
            noise_bound=args.noise_bound,
            use_scale_invariant_features=True,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )
    else:
        # Rigid uses FasterPFH
        config = PCRConfig(
            voxel_size=args.voxel_size,
            normal_radius=args.normal_radius,
            feature_radius=args.feature_radius,
            ratio_threshold=args.ratio_threshold,
            noise_bound=args.noise_bound,
        )

    print(f"\nRunning {args.method} registration...")
    print(f"  Voxel size: {config.voxel_size}")
    print(f"  Normal radius: {config.normal_radius}")
    print(f"  Feature radius: {config.feature_radius}")
    print(f"  Noise bound: {config.noise_bound}")
    if args.method == "nonrigid":
        print(f"  Scale range: [{config.min_scale:.2f}, {config.max_scale:.2f}]")
    print(f"  Feature radius: {config.feature_radius}")
    print(f"  Ratio threshold: {config.ratio_threshold}")
    print(f"  Noise bound: {config.noise_bound}")

    # Perform registration
    if args.method == "rigid":
        result = register_rigid(source_np, target_np, config)
    else:
        result = register_nonrigid(source_np, target_np, config)

    if result is None:
        print("\n❌ Registration failed!")
        return 1

    # Print results
    print("\n✅ Registration succeeded!")
    print(f"  Initial correspondences: {result['total_correspondences']}")
    print(f"  Final inliers: {result['inlier_count']}")
    inlier_ratio = result["inlier_count"] / result["total_correspondences"] * 100
    print(f"  Inlier ratio: {inlier_ratio:.1f}%")

    # Warn if too few inliers
    if inlier_ratio < 10:
        print(f"\n⚠️  WARNING: Very few inliers ({inlier_ratio:.1f}%)!")
        print(f"   Current noise_bound: {args.noise_bound}")
        print(f"   Try increasing: --noise-bound {args.noise_bound * 2:.2f}")
        print(f"   Or even higher: --noise-bound {args.noise_bound * 3:.2f}")
        print("   Typical building scans: 0.2-0.5 (20-50cm tolerance)")

    if args.method == "nonrigid":
        print(f"  Mean scale: {result['mean_scale']:.6f}")
        print(f"  Scale std: {result['scale_std']:.6f}")
        print(
            f"  Scale variation: {result['scale_std'] / result['mean_scale'] * 100:.1f}%"
        )
    else:
        R = np.array(result["rotation"]).reshape(3, 3)
        t = np.array(result["translation"])
        s = result["scale"]
        print("\nTransformation:")
        print(f"  Scale: {s:.6f}")
        print(f"  Translation: {t}")

    # Extract keypoints and correspondences from registration
    source_kp = result["source_keypoints"]
    target_kp = result["target_keypoints"]
    initial_corr = result["initial_correspondences"]
    inlier_corr = result["inlier_correspondences"]

    # Get source scales if nonrigid (for color-coding by RBF scale field)
    source_scales = result.get("source_scales", None)

    print(f"\nKeypoints: {len(source_kp)} source, {len(target_kp)} target")
    print(f"Correspondences: {len(initial_corr)} initial → {len(inlier_corr)} refined")

    # Visualize BEFORE (initial correspondences)
    print("\n" + "=" * 60)
    print("BEFORE registration - Initial correspondences (press Q)")
    print("=" * 60)
    visualize_registration_result(
        source_o3d,
        target_o3d,
        source_kp[initial_corr[:, 0]],
        target_kp[initial_corr[:, 1]],
        np.arange(len(initial_corr)).reshape(-1, 1) @ np.ones((1, 2), dtype=int),
        transformation=None,
        title=f"BEFORE - {len(initial_corr)} Initial Correspondences",
        source_scales=source_scales[initial_corr[:, 0]]
        if source_scales is not None
        else None,
    )

    # Visualize AFTER (refined inliers)
    print("\n" + "=" * 60)
    print("AFTER registration - Refined inliers (press Q)")
    print("=" * 60)

    if args.method == "rigid":
        R = np.array(result["rotation"]).reshape(3, 3)
        t = np.array(result["translation"])
        s = result["scale"]
        transformation = np.eye(4)
        transformation[:3, :3] = s * R
        transformation[:3, 3] = t
    else:
        transformation = None
        print("Note: Non-rigid uses per-point transformations")

    visualize_registration_result(
        source_o3d,
        target_o3d,
        source_kp[inlier_corr[:, 0]],
        target_kp[inlier_corr[:, 1]],
        np.arange(len(inlier_corr)).reshape(-1, 1) @ np.ones((1, 2), dtype=int),
        transformation=transformation,
        title=f"AFTER - {len(inlier_corr)} Refined Inliers",
        source_scales=source_scales[inlier_corr[:, 0]]
        if source_scales is not None
        else None,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
