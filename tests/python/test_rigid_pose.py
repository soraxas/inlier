import numpy as np

import inlier
from inlier_data import TEST_DATA


def _rotation_error_deg(r_gt: np.ndarray, r_est: np.ndarray) -> float:
    cos_angle = (np.trace(r_est @ r_gt.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _mean_alignment_error(
    rotation: np.ndarray,
    translation: np.ndarray,
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
) -> float:
    residuals = rotation @ pts_src.T + translation.reshape(3, 1) - pts_dst.T
    return float(np.linalg.norm(residuals, axis=0).mean())


def test_rigid_pose():
    """Replicate the SuperRANSAC rigid transform on the bundled example."""

    correspondences = np.loadtxt(TEST_DATA.fetch("rigid_pose_example_points.txt"))
    gt_pose = np.loadtxt(TEST_DATA.fetch("rigid_pose_example_gt.txt"))

    pts_src = correspondences[:, :3]
    pts_dst = correspondences[:, 3:]
    r_gt = gt_pose[:3, :3]
    t_gt = gt_pose[:3, 3]

    settings = inlier.RansacSettings(
        min_iterations=5000,
        max_iterations=5000,
        inlier_threshold=0.1,
        rng_seed=2,
        confidence=0.9999,
    )
    result = inlier.estimate_rigid_transform_py(
        pts_src.tolist(), pts_dst.tolist(), threshold=0.1, settings=settings
    )

    r_est = np.asarray(result["rotation"], dtype=float)
    t_est = np.asarray(result["translation"], dtype=float).reshape(3)

    rot_err = _rotation_error_deg(r_gt, r_est)
    trans_err = float(np.linalg.norm(t_est - t_gt))
    fit_err = _mean_alignment_error(r_est, t_est, pts_src, pts_dst)

    assert len(result["inliers"]) > 2800, result
    assert rot_err < 1.9, f"{rot_err} {trans_err} {fit_err}"
    assert trans_err < 0.095
    assert fit_err < 0.088
