//! Scoring primitives for SupeRANSAC.
//!
//! This module provides a `Score` type analogous to the C++ implementation
//! in `include/scoring/score.h` and a couple of basic scoring strategies:
//! - Pure inlier-count RANSAC scoring.
//! - MSAC-style scoring (truncated squared residual cost turned into a
//!   maximization objective).
//!
//! More advanced variants (MAGSAC, ACRANSAC) can be built on this interface
//! in later phases.

use crate::core::Scoring;
use crate::types::DataMatrix;

/// Scalar score storing an inlier count and a floating-point quality value.
///
/// This mirrors the C++ `scoring::Score` class: scores are compared primarily
/// by their `value` field, not the raw inlier count.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Score {
    pub inlier_count: usize,
    pub value: f64,
}

impl Score {
    pub fn new(inlier_count: usize, value: f64) -> Self {
        Self { inlier_count, value }
    }
}

/// RANSAC-style scoring that counts inliers using a user-provided residual
/// function.
///
/// The residual function takes `(data, model, row_index)` and returns a
/// non-negative residual value.
pub struct RansacInlierCountScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    threshold: f64,
    residual_fn: F,
    _marker: std::marker::PhantomData<M>,
}

impl<M, F> RansacInlierCountScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    pub fn new(threshold: f64, residual_fn: F) -> Self {
        Self {
            threshold,
            residual_fn,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<M, F> Scoring<M> for RansacInlierCountScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    type Score = Score;

    fn threshold(&self) -> f64 {
        self.threshold
    }

    fn score(
        &self,
        data: &DataMatrix,
        model: &M,
        inliers_out: &mut Vec<usize>,
    ) -> Self::Score {
        let n = data.nrows();
        let thresh_sq = self.threshold * self.threshold;
        inliers_out.clear();

        let mut inlier_count = 0usize;
        for i in 0..n {
            let r = (self.residual_fn)(data, model, i);
            if r * r <= thresh_sq {
                inliers_out.push(i);
                inlier_count += 1;
            }
        }

        // For a pure inlier-count RANSAC objective, we just use the inlier
        // count as both the quality value and count.
        Score::new(inlier_count, inlier_count as f64)
    }
}

/// MSAC-style scoring: minimizes a truncated squared-residual cost.
///
/// We convert the cost into a *maximization* objective by storing
/// `value = -cost` in the returned `Score`, so that larger is better.
pub struct MsacScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    threshold: f64,
    residual_fn: F,
    _marker: std::marker::PhantomData<M>,
}

impl<M, F> MsacScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    pub fn new(threshold: f64, residual_fn: F) -> Self {
        Self {
            threshold,
            residual_fn,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<M, F> Scoring<M> for MsacScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    type Score = Score;

    fn threshold(&self) -> f64 {
        self.threshold
    }

    fn score(
        &self,
        data: &DataMatrix,
        model: &M,
        inliers_out: &mut Vec<usize>,
    ) -> Self::Score {
        let n = data.nrows();
        let thresh_sq = self.threshold * self.threshold;
        inliers_out.clear();

        let mut inlier_count = 0usize;
        let mut cost = 0.0f64;

        for i in 0..n {
            let r = (self.residual_fn)(data, model, i);
            let r2 = r * r;
            if r2 <= thresh_sq {
                inliers_out.push(i);
                inlier_count += 1;
                cost += r2;
            } else {
                cost += thresh_sq;
            }
        }

        // Larger scores should be better, so we negate the MSAC cost.
        Score::new(inlier_count, -cost)
    }
}

/// MAGSAC-style scoring: uses a soft inlier/outlier threshold with marginalization.
///
/// This is a simplified implementation. A full MAGSAC implementation would
/// marginalize over the inlier threshold and use a more sophisticated cost function.
pub struct MagsacScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    threshold: f64,
    residual_fn: F,
    _marker: std::marker::PhantomData<M>,
}

impl<M, F> MagsacScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    pub fn new(threshold: f64, residual_fn: F) -> Self {
        Self {
            threshold,
            residual_fn,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<M, F> Scoring<M> for MagsacScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    type Score = Score;

    fn threshold(&self) -> f64 {
        self.threshold
    }

    fn score(
        &self,
        data: &DataMatrix,
        model: &M,
        inliers_out: &mut Vec<usize>,
    ) -> Self::Score {
        let n = data.nrows();
        let thresh_sq = self.threshold * self.threshold;
        inliers_out.clear();

        let mut inlier_count = 0usize;
        let mut cost = 0.0f64;

        // MAGSAC uses a soft threshold: points contribute based on their residual
        // with a smooth transition rather than a hard cutoff
        for i in 0..n {
            let r = (self.residual_fn)(data, model, i);
            let r2 = r * r;

            // Soft threshold: use exponential decay for outliers
            if r2 <= thresh_sq {
                inliers_out.push(i);
                inlier_count += 1;
                cost += r2;
            } else {
                // Soft penalty for outliers (simplified MAGSAC)
                let excess = r2 - thresh_sq;
                cost += thresh_sq + excess * f64::exp(-excess / (thresh_sq * 2.0));
            }
        }

        // Larger scores should be better
        Score::new(inlier_count, -cost)
    }
}

/// ACRANSAC-style scoring: adaptive consensus RANSAC.
///
/// This is a simplified implementation that adapts the threshold based on
/// the distribution of residuals.
pub struct AcransacScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    initial_threshold: f64,
    residual_fn: F,
    _marker: std::marker::PhantomData<M>,
}

impl<M, F> AcransacScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    pub fn new(initial_threshold: f64, residual_fn: F) -> Self {
        Self {
            initial_threshold,
            residual_fn,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<M, F> Scoring<M> for AcransacScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    type Score = Score;

    fn threshold(&self) -> f64 {
        self.initial_threshold
    }

    fn score(
        &self,
        data: &DataMatrix,
        model: &M,
        inliers_out: &mut Vec<usize>,
    ) -> Self::Score {
        let n = data.nrows();
        inliers_out.clear();

        // First pass: collect all residuals
        let mut residuals: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let r = (self.residual_fn)(data, model, i);
            residuals.push(r);
        }

        // Adaptive threshold: use median absolute deviation (MAD)
        // A simplified ACRANSAC approach
        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if n > 0 {
            residuals[n / 2]
        } else {
            self.initial_threshold
        };

        // Adaptive threshold based on median
        let adaptive_threshold = f64::max(self.initial_threshold, median * 1.5);
        let thresh_sq = adaptive_threshold * adaptive_threshold;

        let mut inlier_count = 0usize;
        for i in 0..n {
            let r = residuals[i];
            if r * r <= thresh_sq {
                inliers_out.push(i);
                inlier_count += 1;
            }
        }

        Score::new(inlier_count, inlier_count as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::{MsacScoring, RansacInlierCountScoring, Score};
    use crate::core::Scoring;
    use crate::types::DataMatrix;

    #[derive(Clone, Debug)]
    struct UnitModel;

    #[test]
    fn ransac_inlier_count_scoring_counts_correctly() {
        // Data matrix where each row contains a single scalar we treat as a residual.
        let mut data = DataMatrix::zeros(5, 1);
        data[(0, 0)] = 0.1;
        data[(1, 0)] = 0.4;
        data[(2, 0)] = 0.6;
        data[(3, 0)] = 1.0;
        data[(4, 0)] = 0.3;

        let model = UnitModel;

        let scoring = RansacInlierCountScoring::new(0.5, |d, _m, row| d[(row, 0)]);
        let mut inliers = Vec::new();
        let s: Score = scoring.score(&data, &model, &mut inliers);

        assert_eq!(s.inlier_count, 3);
        assert_eq!(s.value, 3.0);
        assert_eq!(inliers, vec![0, 1, 4]);
    }

    #[test]
    fn msac_scoring_penalizes_large_residuals() {
        // Same residual layout as the RANSAC test.
        let mut data = DataMatrix::zeros(5, 1);
        data[(0, 0)] = 0.1;
        data[(1, 0)] = 0.4;
        data[(2, 0)] = 0.6;
        data[(3, 0)] = 1.0;
        data[(4, 0)] = 0.3;

        let model = UnitModel;

        let scoring = MsacScoring::new(0.5, |d, _m, row| d[(row, 0)]);
        let mut inliers = Vec::new();
        let s: Score = scoring.score(&data, &model, &mut inliers);

        // Same inlier set as pure RANSAC.
        assert_eq!(s.inlier_count, 3);
        assert_eq!(inliers, vec![0, 1, 4]);

        // MSAC score should be negative and finite (since it's -cost).
        assert!(s.value.is_finite());
        assert!(s.value < 0.0);
    }
}
