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
        Self {
            inlier_count,
            value,
        }
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

    fn score(&self, data: &DataMatrix, model: &M, inliers_out: &mut Vec<usize>) -> Self::Score {
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

    fn score(&self, data: &DataMatrix, model: &M, inliers_out: &mut Vec<usize>) -> Self::Score {
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
/// This implementation uses an improved approximation of MAGSAC's marginalization
/// approach. A full MAGSAC implementation would use incomplete gamma functions
/// from special function libraries (like boost::math in C++).
pub struct MagsacScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    threshold: f64,
    residual_fn: F,
    sigma_max: f64,
    degrees_of_freedom: usize,
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
            sigma_max: threshold * 2.0, // Default: 2x threshold
            degrees_of_freedom: 2,      // Default: 2D residuals
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_sigma_max(mut self, sigma_max: f64) -> Self {
        self.sigma_max = sigma_max;
        self
    }

    pub fn with_degrees_of_freedom(mut self, dof: usize) -> Self {
        self.degrees_of_freedom = dof;
        self
    }

    /// Approximate upper incomplete gamma function using series expansion.
    /// For MAGSAC, we need Γ(a, x) where a = (dof-1)/2 and x = r^2/(2*σ_max^2)
    /// This is a simplified approximation that works reasonably well.
    fn approximate_upper_incomplete_gamma(&self, a: f64, x: f64) -> f64 {
        // For small x, use series expansion
        if x < 1.0 {
            // Γ(a, x) ≈ Γ(a) - x^a * e^(-x) * (1 + x/(a+1) + ...)
            let gamma_a = self.approximate_gamma(a);
            let x_pow_a = x.powf(a);
            let exp_neg_x = (-x).exp();
            let series = 1.0 + x / (a + 1.0) + x * x / ((a + 1.0) * (a + 2.0));
            gamma_a - x_pow_a * exp_neg_x * series
        } else {
            // For large x, use asymptotic expansion
            // Γ(a, x) ≈ x^(a-1) * e^(-x) * (1 + (a-1)/x + ...)
            let x_pow_a_minus_1 = x.powf(a - 1.0);
            let exp_neg_x = (-x).exp();
            let asymptotic = 1.0 + (a - 1.0) / x;
            x_pow_a_minus_1 * exp_neg_x * asymptotic
        }
    }

    /// Approximate gamma function using Stirling's approximation for a > 0
    fn approximate_gamma(&self, a: f64) -> f64 {
        if a <= 0.0 {
            return 1.0;
        }
        if a < 0.5 {
            // Use reflection formula: Γ(1-z) = π / (Γ(z) * sin(πz))
            let z = 1.0 - a;
            let gamma_z = self.approximate_gamma(z);
            std::f64::consts::PI / (gamma_z * (std::f64::consts::PI * a).sin())
        } else {
            // Stirling's approximation: Γ(a) ≈ sqrt(2π/a) * (a/e)^a * (1 + 1/(12a) + ...)
            let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
            let base = (sqrt_2pi / a).sqrt() * (a / std::f64::consts::E).powf(a);
            base * (1.0 + 1.0 / (12.0 * a))
        }
    }

    /// Compute MAGSAC-style loss for a given squared residual
    fn compute_loss(&self, r_sq: f64) -> f64 {
        let thresh_sq = self.threshold * self.threshold;
        let sigma_max_sq = self.sigma_max * self.sigma_max;
        let two_sigma_max_sq = 2.0 * sigma_max_sq;

        if r_sq < thresh_sq {
            // Inlier: compute marginalization loss
            let n_plus_1_per_2 = (self.degrees_of_freedom + 1) as f64 / 2.0;
            let n_minus_1_per_2 = (self.degrees_of_freedom - 1) as f64 / 2.0;

            let residual_norm = r_sq / two_sigma_max_sq;

            // Approximate lower incomplete gamma: γ(a, x) = Γ(a) - Γ(a, x)
            // For small x: γ(a, x) ≈ x^a / a * (1 - x/(a+1) + ...)
            let gamma_a = self.approximate_gamma(n_plus_1_per_2);
            let upper_gamma_lower =
                self.approximate_upper_incomplete_gamma(n_plus_1_per_2, residual_norm);
            let lower_gamma = gamma_a - upper_gamma_lower;

            // Upper incomplete gamma approximation
            let upper_gamma =
                self.approximate_upper_incomplete_gamma(n_minus_1_per_2, residual_norm);
            let value0 = self.approximate_upper_incomplete_gamma(n_minus_1_per_2, 0.0);

            // MAGSAC loss formula (simplified)
            sigma_max_sq / 2.0 * lower_gamma + sigma_max_sq / 4.0 * (upper_gamma - value0)
        } else {
            // Outlier: fixed loss
            self.threshold * self.threshold
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

    fn score(&self, data: &DataMatrix, model: &M, inliers_out: &mut Vec<usize>) -> Self::Score {
        let n = data.nrows();
        let thresh_sq = self.threshold * self.threshold;
        inliers_out.clear();

        let mut inlier_count = 0usize;
        let mut cost = 0.0f64;

        // MAGSAC: marginalize over threshold and compute loss for each point
        for i in 0..n {
            let r = (self.residual_fn)(data, model, i);
            let r_sq = r * r;

            let loss = self.compute_loss(r_sq);
            cost += loss;

            // Still track inliers for compatibility
            if r_sq <= thresh_sq {
                inliers_out.push(i);
                inlier_count += 1;
            }
        }

        // Larger scores should be better (negate cost)
        Score::new(inlier_count, -cost)
    }
}

/// ACRANSAC-style scoring: adaptive consensus RANSAC with NFA (Number of False Alarms).
///
/// This implementation adapts the threshold by iterating through residual thresholds
/// and selecting the one with the best NFA score.
pub struct AcransacScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    initial_threshold: f64,
    residual_fn: F,
    step_number: usize,
    minimal_sample_size: usize,
    mult_error: f64,
    log_alpha0: f64,
    _marker: std::marker::PhantomData<M>,
}

impl<M, F> AcransacScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    pub fn new(
        initial_threshold: f64,
        residual_fn: F,
        minimal_sample_size: usize,
        mult_error: f64,
        log_alpha0: f64,
    ) -> Self {
        Self {
            initial_threshold,
            residual_fn,
            step_number: 10,
            minimal_sample_size,
            mult_error,
            log_alpha0,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn set_step_number(&mut self, steps: usize) {
        self.step_number = steps;
    }

    /// Compute log10 of binomial coefficient C(n, k) using logarithms
    fn log_combi(&self, k: usize, n: usize, log10_table: &[f64]) -> f64 {
        if k >= n {
            return 0.0;
        }
        let k = if n - k < k { n - k } else { k };
        let mut r = 0.0;
        for i in 1..=k {
            if n >= i && i < log10_table.len() && (n - i + 1) < log10_table.len() {
                r += log10_table[n - i + 1] - log10_table[i];
            }
        }
        r
    }

    /// Precompute log10 table for [0, n+1]
    fn make_log10_table(&self, n: usize) -> Vec<f64> {
        let mut table = Vec::with_capacity(n + 1);
        for i in 0..=n {
            table.push((i as f64).log10());
        }
        table
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

    fn score(&self, data: &DataMatrix, model: &M, inliers_out: &mut Vec<usize>) -> Self::Score {
        let n = data.nrows();
        inliers_out.clear();

        if n < self.minimal_sample_size {
            return Score::new(0, f64::NEG_INFINITY);
        }

        // Collect residuals for points within initial threshold
        let thresh_sq = self.initial_threshold * self.initial_threshold;
        let mut residuals: Vec<(f64, usize)> = Vec::new();

        for i in 0..n {
            let r = (self.residual_fn)(data, model, i);
            let r_sq = r * r;
            if r_sq < thresh_sq {
                residuals.push((r, i));
            }
        }

        if residuals.is_empty() {
            return Score::new(0, f64::NEG_INFINITY);
        }

        // Sort by residual value
        residuals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let max_residual = residuals.last().unwrap().0;
        let residual_step = max_residual / self.step_number as f64;

        if residual_step < f64::EPSILON {
            return Score::new(0, f64::NEG_INFINITY);
        }

        // Precompute log10 table and binomial coefficients
        let log10_table = self.make_log10_table(n);
        let mut logc_n = vec![0.0; n + 1];
        let mut logc_k = vec![0.0; n + 1];

        for k in 0..=n {
            logc_n[k] = self.log_combi(k, n, &log10_table);
        }
        for k_val in 0..=n {
            logc_k[k_val] = self.log_combi(self.minimal_sample_size, k_val, &log10_table);
        }

        // Compute log epsilon 0
        let log_e0 = ((n - self.minimal_sample_size) as f64).log10();

        // Iterate through thresholds to find best NFA
        let mut best_nfa = f64::INFINITY;
        let mut best_threshold = 0.0;
        let mut best_inlier_count = self.minimal_sample_size;

        let mut current_max_idx = self.minimal_sample_size;

        for current_threshold in (0..=self.step_number).map(|i| (i as f64 + 1.0) * residual_step) {
            if current_threshold > max_residual {
                break;
            }

            // Count inliers up to current threshold
            while current_max_idx < residuals.len()
                && residuals[current_max_idx].0 <= current_threshold
            {
                current_max_idx += 1;
            }

            if current_max_idx <= self.minimal_sample_size {
                continue;
            }

            // Compute NFA: log_e0 + log_alpha * (k - m) + logc_n[k] + logc_k[k]
            let log_alpha =
                self.log_alpha0 + self.mult_error * (current_threshold + f64::EPSILON).log10();
            let k = current_max_idx;
            let nfa = log_e0
                + log_alpha * (k - self.minimal_sample_size) as f64
                + logc_n[k]
                + logc_k[k.min(logc_k.len() - 1)];

            // Keep best NFA (must be < 0 to be meaningful)
            if nfa < best_nfa && nfa < 0.0 {
                best_nfa = nfa;
                best_threshold = current_threshold;
                best_inlier_count = k;
            }
        }

        // Collect inliers for best threshold
        for &(r, idx) in &residuals[..best_inlier_count.min(residuals.len())] {
            if r <= best_threshold {
                inliers_out.push(idx);
            }
        }

        Score::new(best_inlier_count, -best_nfa)
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
