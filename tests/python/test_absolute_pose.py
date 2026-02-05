from __future__ import annotations


import numpy as np

import inlier
from inlier_data import TEST_DATA


def _rotation_error_deg(r_gt: np.ndarray, r_est: np.ndarray) -> float:
    cos_angle = (np.trace(r_est @ r_gt.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _mean_reprojection_error(
    rotation: np.ndarray,
    translation: np.ndarray,
    points_3d: np.ndarray,
    points_2d_norm: np.ndarray,
) -> float:
    projected = rotation @ points_3d.T + translation.reshape(3, 1)
    projected_norm = (projected[:2] / projected[2]).T
    return float(np.linalg.norm(projected_norm - points_2d_norm, axis=1).mean())


def test_absolute_pose():
    """Run the absolute pose fitting example on the bundled pose6dscene data."""

    correspondences = np.loadtxt(TEST_DATA.fetch("pose6dscene_points.txt"))
    gt_pose = np.loadtxt(TEST_DATA.fetch("pose6dscene_gt.txt"))
    intrinsics = np.loadtxt(TEST_DATA.fetch("pose6dscene.K"))

    points_2d = correspondences[:, :2]
    points_3d = correspondences[:, 2:]

    inv_k = np.linalg.inv(intrinsics)
    points_2d_norm = np.array(
        [(inv_k @ np.array([u, v, 1.0]))[:2] for u, v in points_2d]
    )

    settings = inlier.RansacSettings(
        min_iterations=100,
        max_iterations=900,
        inlier_threshold=0.5,
        confidence=0.999,
        rng_seed=0,
        sampler="uniform",
    )

    result = inlier.estimate_absolute_pose_py(
        points_3d.tolist(),
        points_2d_norm.tolist(),
        threshold=0.5,
        settings=settings,
    )

    r_est = np.asarray(result["rotation"], dtype=float)
    t_est = np.asarray(result["translation"], dtype=float).reshape(3)

    rot_err = _rotation_error_deg(gt_pose[:, :3], r_est)
    trans_err = float(np.linalg.norm(t_est - gt_pose[:, 3]))
    reproj_err = _mean_reprojection_error(r_est, t_est, points_3d, points_2d_norm)
    inliers = len(result["inliers"])

    # With optional PnP/P3P solvers enabled and a larger iteration budget,
    # a single run should closely match the notebook’s ground truth.
    assert inliers > 25
    assert rot_err < 2.0, (
        f"rot_err: {rot_err}, trans_err: {trans_err}, reproj_err: {reproj_err}"
    )
    assert trans_err < 10.0
    assert reproj_err < 0.1
