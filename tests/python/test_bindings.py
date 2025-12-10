import pytest
from pathlib import Path
import csv
import time

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
    settings = inlier.RansacSettings(
        min_iterations=1, max_iterations=2, inlier_threshold=1.0, confidence=0.5
    )

    result = inlier.run_python_ransac(
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
    assert model["points"][0] == [0.0, 0.0]


def test_probe_estimator_smoke():
    estimator = inlier.EstimatorAdapter(DummyEstimator())
    size = inlier.probe_estimator(estimator, [0, 1], [[0.0, 0.0], [1.0, 1.0]])
    assert size == 2


def test_homography_against_benchmark_pairs():
    dataset = Path(__file__).parent.parent / "data" / "hpatches_facade_pairs.csv"
    with dataset.open() as f:
        reader = csv.DictReader(f)
        pts1, pts2 = [], []
        for row in reader:
            pts1.append([float(row["x1"]), float(row["y1"])])
            pts2.append([float(row["x2"]), float(row["y2"])])

    settings = inlier.RansacSettings(
        min_iterations=64, max_iterations=256, inlier_threshold=2.0, confidence=0.99
    )

    start = time.time()
    result = inlier.estimate_homography_py(pts1, pts2, 2.0, settings)
    runtime = time.time() - start

    assert result["inliers"], "homography estimation should yield inliers"
    assert runtime < 0.5, "pipeline should run quickly on small benchmark"

    # Compute mean forward transfer error in pixels
    est = result["model"]

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
