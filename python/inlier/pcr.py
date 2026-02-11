"""High-level Point Cloud Registration (PCR) API for Python.

This module provides easy-to-use functions for registering point clouds
with both rigid and non-rigid transformations.

Examples:
    Rigid registration:
    >>> import numpy as np
    >>> from inlier.pcr import register_rigid
    >>>
    >>> src = np.random.rand(1000, 3)
    >>> dst = np.random.rand(1000, 3)
    >>> result = register_rigid(src, dst)
    >>> if result:
    ...     print(f"Rotation: {result['rotation']}")
    ...     print(f"Translation: {result['translation']}")
    ...     print(f"Inliers: {result['inlier_count']}/{result['total_correspondences']}")

    Non-rigid registration:
    >>> from inlier.pcr import register_nonrigid
    >>>
    >>> result = register_nonrigid(src, dst)
    >>> if result:
    ...     print(f"Mean scale: {result['mean_scale']:.3f}")
    ...     print(f"Scale variation: {result['scale_std']:.3f}")
    ...     print(f"Inliers: {result['inlier_count']}/{result['total_correspondences']}")
"""

from typing import Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray

from ._inlier_rs import PCRConfig, register_rigid_py, register_nonrigid_py


def register_rigid(
    src: NDArray[np.float64],
    dst: NDArray[np.float64],
    voxel_size: float = 0.05,
    normal_radius: float = 0.15,
    feature_radius: float = 0.3,
    ratio_threshold: float = 0.9,
    noise_bound: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Register point clouds with rigid transformation.

    Estimates rotation, translation, and uniform scale between two point clouds
    using KISS-Matcher with FasterPFH features and ROBIN outlier rejection.

    Args:
        src: Source point cloud (N × 3 array)
        dst: Target point cloud (M × 3 array)
        voxel_size: Voxel size for downsampling (0 to disable)
        normal_radius: Radius for normal estimation
        feature_radius: Radius for feature descriptors
        ratio_threshold: Feature matching ratio test (0-1, lower = stricter)
        noise_bound: ROBIN outlier rejection threshold

    Returns:
        Dictionary with keys:
            - rotation: 3×3 rotation matrix (flattened to 9 elements)
            - translation: 3-element translation vector
            - scale: Uniform scale factor
            - inlier_count: Number of inlier correspondences
            - total_correspondences: Total initial correspondences
        Returns None if registration fails.

    Example:
        >>> src = np.random.rand(1000, 3)
        >>> dst = src @ np.eye(3) + np.array([1, 0, 0])  # Translate by [1,0,0]
        >>> result = register_rigid(src, dst)
        >>> if result:
        ...     R = np.array(result['rotation']).reshape(3, 3)
        ...     t = np.array(result['translation'])
        ...     print(f"Detected translation: {t}")
    """
    if src.shape[1] != 3 or dst.shape[1] != 3:
        raise ValueError(f"Point clouds must be N×3, got {src.shape} and {dst.shape}")

    config = PCRConfig(
        voxel_size=voxel_size,
        normal_radius=normal_radius,
        feature_radius=feature_radius,
        ratio_threshold=ratio_threshold,
        noise_bound=noise_bound,
        use_scale_invariant_features=False,
    )

    result = register_rigid_py(src, dst, config)

    if result is not None:
        # Convert rotation matrix to numpy and reshape
        result["rotation"] = np.array(result["rotation"]).reshape(3, 3)
        result["translation"] = np.array(result["translation"])

    return result


def register_nonrigid(
    src: NDArray[np.float64],
    dst: NDArray[np.float64],
    voxel_size: float = 0.05,
    normal_radius: float = 0.15,
    feature_radius: float = 0.3,
    ratio_threshold: float = 0.9,
    noise_bound: float = 0.05,
    use_scale_invariant_features: bool = True,
) -> Optional[Dict[str, Any]]:
    """Register point clouds with non-rigid transformation.

    Estimates spatially-varying scale field + rigid transformation using
    SIPFH scale-invariant features and RBF interpolation. Suitable for:
    - Thermal expansion/contraction
    - Biological growth/deformation
    - Non-uniform scaling

    Args:
        src: Source point cloud (N × 3 array)
        dst: Target point cloud (M × 3 array)
        voxel_size: Voxel size for downsampling (0 to disable)
        normal_radius: Radius for normal estimation
        feature_radius: Radius for feature descriptors
        ratio_threshold: Feature matching ratio test (0-1, lower = stricter)
        noise_bound: ROBIN outlier rejection threshold
        use_scale_invariant_features: Use SIPFH (recommended) vs FasterPFH

    Returns:
        Dictionary with keys:
            - mean_scale: Average scale factor across control points
            - scale_std: Standard deviation of scale field
            - rotation: 3×3 rotation matrix (flattened to 9 elements)
            - translation: 3-element translation vector
            - inlier_count: Number of inlier correspondences
            - total_correspondences: Total initial correspondences
        Returns None if registration fails.

    Example:
        >>> # Create point cloud with non-uniform scaling
        >>> src = np.random.rand(1000, 3)
        >>> scale = np.linspace(0.9, 1.1, 1000)[:, None]
        >>> dst = src * scale + np.array([0, 0, 1])
        >>>
        >>> result = register_nonrigid(src, dst)
        >>> if result:
        ...     print(f"Mean scale: {result['mean_scale']:.3f}")
        ...     print(f"Scale variation: {result['scale_std']:.3f}")
        ...     print(f"Inliers: {result['inlier_count']}")
    """
    if src.shape[1] != 3 or dst.shape[1] != 3:
        raise ValueError(f"Point clouds must be N×3, got {src.shape} and {dst.shape}")

    config = PCRConfig(
        voxel_size=voxel_size,
        normal_radius=normal_radius,
        feature_radius=feature_radius,
        ratio_threshold=ratio_threshold,
        noise_bound=noise_bound,
        use_scale_invariant_features=use_scale_invariant_features,
    )

    result = register_nonrigid_py(src, dst, config)

    if result is not None:
        # Convert rotation matrix to numpy and reshape
        result["rotation"] = np.array(result["rotation"]).reshape(3, 3)
        result["translation"] = np.array(result["translation"])

    return result


def load_ply(filepath: str) -> NDArray[np.float64]:
    """Load point cloud from PLY file.

    Args:
        filepath: Path to PLY file

    Returns:
        N×3 numpy array of point coordinates

    Example:
        >>> src = load_ply("bunny.ply")
        >>> dst = load_ply("bunny_deformed.ply")
        >>> result = register_nonrigid(src, dst)
    """
    from plyfile import PlyData

    plydata = PlyData.read(filepath)
    vertex = plydata["vertex"]
    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])
    return points


__all__ = [
    "register_rigid",
    "register_nonrigid",
    "load_ply",
    "PCRConfig",
]
