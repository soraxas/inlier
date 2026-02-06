"""Demonstrate building a RANSAC pipeline from scratch in Python.

This example fits a 1D line y = m*x + b using pure-Python estimator and scoring
classes, wired through the `Pipeline` builder exposed by the `inlier` crate.
"""

from __future__ import annotations

import sys
import numpy as np

import inlier


class RandomSampler:
    """Simple Python sampler: draws unique indices uniformly."""

    def sample(self, data, sample_size: int):
        if len(data) < sample_size:
            return []
        return list(
            np.random.default_rng().choice(len(data), size=sample_size, replace=False)
        )

    def update(self, sample, sample_size, iteration, score_hint):
        return None


class LineEstimator:
    def __init__(self, rng: np.random.Generator):
        # Randomize minimal/non-minimal sample size (2 or 3 points) to show effect.
        self._sample_size = int(rng.integers(2, 4))

    def sample_size(self) -> int:
        return self._sample_size

    def non_minimal_sample_size(self) -> int:
        return self._sample_size

    def is_valid_sample(self, data, sample) -> bool:
        (x1, _), (x2, _) = data[sample[0]], data[sample[1]]
        return abs(x2 - x1) > 1e-9

    def estimate_model(self, data, sample):
        (x1, y1), (x2, y2) = data[sample[0]], data[sample[1]]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return [[float(m), float(b)]]  # list of models

    def is_valid_model(self, model, data, sample, threshold) -> bool:  # noqa: ARG002
        return True


class LineScoring:
    def __init__(self, tau: float):
        self._tau = tau

    def threshold(self) -> float:
        return self._tau

    def score(self, data: np.ndarray, model: np.ndarray):
        m, b = model
        # print(data)

        print(type(data))
        # compute residuals for each point with numpy
        # residuals = np.abs(data[:, 1] - (model[0] * data[:, 0] + model[1]))
        # print(residuals)

        residuals = [abs(y - (m * x + b)) for x, y in data]
        # print(residuals)
        # exit()

        inliers = [i for i, r in enumerate(residuals) if r <= self._tau]
        return float(len(inliers)), inliers


def main() -> None:
    if len(sys.argv) > 1:
        n_sample = int(sys.argv[1])
    else:
        n_sample = 50

    # Sample a seed, then generate ground-truth params and noisy data.
    seed = np.random.default_rng().integers(0, 2**32 - 1, dtype=np.uint32)
    rng = np.random.default_rng(seed)

    slope_gt = rng.uniform(-10.0, 10.0)
    intercept_gt = rng.uniform(-10, 10)
    noise_sigma = rng.uniform(0.00, 0.5)

    xs = np.linspace(-1.0, 1.0, n_sample)
    ys = slope_gt * xs + intercept_gt + rng.normal(scale=noise_sigma, size=xs.shape)
    data = list(zip(xs.tolist(), ys.tolist()))

    pipe = inlier.Pipeline(
        estimator=inlier.EstimatorAdapter(LineEstimator(rng)),
        sampler=inlier.SamplerAdapter(RandomSampler()),
        scoring=inlier.ScoringAdapter(LineScoring(tau=0.1)),
        settings=inlier.MetasacSettings(
            min_iterations=500, max_iterations=2000, confidence=0.999
        ),
    )

    result = pipe.run(data)
    if result is None:
        raise RuntimeError("RANSAC failed")

    model, inliers, score = result
    m_est, b_est = model
    slope_err = abs(m_est - slope_gt) / max(abs(slope_gt), 1e-9)
    intercept_err = abs(b_est - intercept_gt) / max(abs(intercept_gt), 1e-9)

    print(f"Seed: {seed}")
    print(
        f"Ground truth: slope={slope_gt:.4f}, intercept={intercept_gt:.4f}, noise_sigma={noise_sigma:.4f}"
    )
    print(f"Estimated:    slope={m_est:.4f}, intercept={b_est:.4f}")
    print(f"Relative error: slope={slope_err:.4%}, intercept={intercept_err:.4%}")
    print(f"Inliers: {len(inliers)}, Score: {score}")


if __name__ == "__main__":
    main()
