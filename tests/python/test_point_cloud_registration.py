from __future__ import annotations

import numpy as np
import pytest

import inlier


@pytest.mark.filterwarnings("ignore:.*numpy.*")
def test_point_cloud_registration_synthetic():
    rng = np.random.default_rng(42)
    pts_src = rng.normal(size=(200, 3)).astype(np.float64)

    # Apply known transform
    angle = np.deg2rad(10.0)
    axis = np.array([0.2, 0.7, -0.3])
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    R_gt = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    t_gt = np.array([0.1, -0.05, 0.2])

    pts_tgt = (R_gt @ pts_src.T).T + t_gt
    pts_tgt += rng.normal(scale=0.005, size=pts_tgt.shape)  # small noise

    res = inlier.estimate_rigid_transform_py(
        pts_src.tolist(),
        pts_tgt.tolist(),
        threshold=0.05,
        settings=inlier.RansacSettings(min_iterations=64, max_iterations=256),
    )

    r_est = np.array(res["rotation"], dtype=float)
    t_est = np.array(res["translation"], dtype=float).reshape(3)

    rot_err = np.degrees(np.arccos(((np.trace(r_est @ R_gt.T) - 1) / 2).clip(-1, 1)))
    trans_err = np.linalg.norm(t_est - t_gt)
    fit_err = np.linalg.norm(
        (r_est @ pts_src.T + t_est.reshape(3, 1) - pts_tgt.T), axis=0
    ).mean()

    assert rot_err < 2.0
    assert trans_err < 0.05
    assert fit_err < 0.05
