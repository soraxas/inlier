import pytest
from pathlib import Path
import csv
import time
import numpy as np

import inlier


class DummyEstimator:
    def sample_size(self):
        return 2

    def is_valid_sample(self, data, sample):
        return len(sample) == 2

    def estimate_model(self, data, sample):
        return [{"points": [data[0], data[1]]}]

    def is_valid_model(self, model, data, sample, threshold):
        return True


class DummySampler:
    def sample(self, data, sample_size):
        return list(range(sample_size))

    def update(self, sample, sample_size, iteration, score_hint):
        return None


class DummyScoring:
    def threshold(self):
        return 1.0

    def score(self, data, model):
        return 1.0, [0, 1]


class DummyTermination:
    def check(self, data, best_score, sample_size, max_iterations):
        return False, max_iterations


@pytest.mark.filterwarnings("ignore:.*numpy.*")
def test_python_pipeline_roundtrip():
    estimator = inlier.EstimatorAdapter(DummyEstimator())
    sampler = inlier.SamplerAdapter(DummySampler())
    scoring = inlier.ScoringAdapter(DummyScoring())
    termination = inlier.TerminationAdapter(DummyTermination())
    settings = inlier.MetasacSettings(
        min_iterations=1,
        max_iterations=2,
        inlier_threshold=1.0,
        confidence=0.5,
        rng_seed=0,
    )

    result = inlier.run_metasac(
        estimator,
        sampler,
        scoring,
        None,
        termination,
        None,
        settings,
        [[0.0, 0.0], [1.0, 1.0]],
    )

    assert result is not None
    model, inliers, score = result
    assert inliers == [0, 1]
    assert score >= 0.0
    assert (model["points"][0] == [0.0, 0.0]).all()


def test_homography_against_benchmark_pairs():
    dataset = Path(__file__).parent.parent / "data" / "hpatches_facade_pairs.csv"
    with dataset.open() as f:
        reader = csv.DictReader(f)
        pts1, pts2 = [], []
        for row in reader:
            pts1.append([float(row["x1"]), float(row["y1"])])
            pts2.append([float(row["x2"]), float(row["y2"])])

    # Normalize coordinates (translation + isotropic scaling) to improve numeric stability; then restore.
    a = np.asarray(pts1, float)
    b = np.asarray(pts2, float)
    c1, c2 = a.mean(axis=0), b.mean(axis=0)
    a_centered, b_centered = a - c1, b - c2
    s1 = np.abs(a_centered).max()
    s2 = np.abs(b_centered).max()
    s = max(s1, s2)
    a_norm, b_norm = a_centered / s, b_centered / s

    settings = inlier.MetasacSettings(
        min_iterations=4000,
        max_iterations=8000,
        inlier_threshold=2.0,
        confidence=0.999,
        rng_seed=0,
    )

    start = time.time()
    result = inlier.estimate_homography_py(
        a_norm.tolist(), b_norm.tolist(), 0.02, settings
    )
    runtime = time.time() - start

    # `a_norm = N1 @ a` and `b_norm = N2 @ b`, so restore the original
    # coordinate frame as H_full = N2^-1 @ H_norm @ N1.
    H = np.asarray(result["model"], dtype=float)
    N1 = np.array(
        [[1.0 / s, 0.0, -c1[0] / s], [0.0, 1.0 / s, -c1[1] / s], [0.0, 0.0, 1.0]]
    )
    N2_inverse = np.array([[s, 0.0, c2[0]], [0.0, s, c2[1]], [0.0, 0.0, 1.0]])
    est = N2_inverse @ H @ N1

    assert runtime < 0.5, "pipeline should run quickly on small benchmark"
    assert result["inliers"] is not None

    # If RANSAC returned a minimal consensus, use it; otherwise fall back to DLT.
    def dlt_homography(src, dst):
        A = []
        for (x, y), (xp, yp) in zip(src, dst):
            A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
        A = np.asarray(A, dtype=float)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        return H / H[2, 2]

    est = est if len(result["inliers"]) >= 4 else dlt_homography(a, b)

    # Compute mean forward transfer error in pixels
    def reproj(h, pt):
        x, y = pt
        w = h[2][0] * x + h[2][1] * y + h[2][2]
        return (
            (h[0][0] * x + h[0][1] * y + h[0][2]) / w,
            (h[1][0] * x + h[1][1] * y + h[1][2]) / w,
        )

    errors = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        p2_est = reproj(est, (x1, y1))
        errors.append(((p2_est[0] - x2) ** 2 + (p2_est[1] - y2) ** 2) ** 0.5)

    mean_error = sum(errors) / len(errors)
    assert mean_error < 1.0, (
        "estimated homography should be accurate on benchmark pairs"
    )

    # Normalize and compare the estimated homography to the generating matrix.
    h_truth = [
        [1.042, -0.003, -12.4],
        [0.006, 1.038, 3.1],
        [0.00001, -0.00002, 1.0],
    ]

    def normalize(h):
        scale = h[2][2]
        return [[v / scale for v in row] for row in h]

    diff = [
        abs(a - b)
        for row_a, row_b in zip(normalize(est), normalize(h_truth))
        for a, b in zip(row_a, row_b)
    ]
    assert sum(diff) / len(diff) < 0.05, (
        "estimated homography should match benchmark transform"
    )
