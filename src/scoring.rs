//! Scoring primitives for SupeRANSAC.
//!
//! This module provides a basic score type and a simple RANSAC-style
//! inlier-count scoring implementation that plugs into the generic
//! `Scoring` trait from `core`.

use crate::core::Scoring;
use crate::types::DataMatrix;

/// Simple scalar score storing an inlier count and an optional quality value.
///
/// This is intentionally minimal; it mirrors the idea of the C++ `Score`
/// class while remaining generic enough for early experimentation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Score {
    pub inlier_count: usize,
    pub value: usize,
}

impl Score {
    pub fn new(inlier_count: usize, value: usize) -> Self {
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

        Score::new(inlier_count, inlier_count)
    }
}

#[cfg(test)]
mod tests {
    use super::{RansacInlierCountScoring, Score};
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
        assert_eq!(s.value, 3);
        assert_eq!(inliers, vec![0, 1, 4]);
    }
}
