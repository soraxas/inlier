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
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

type SigmaLutMap = HashMap<(usize, u64), Vec<(f64, f64)>>;

// Precomputed gamma tables generated at build time for common (dof, k) pairs.
include!(concat!(env!("OUT_DIR"), "/sigma_lut.rs"));

#[allow(clippy::type_complexity)]
static SIGMA_LUT: OnceLock<Mutex<SigmaLutMap>> = OnceLock::new();

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
    priors: Option<Vec<f64>>,
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
            priors: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_priors(mut self, priors: &[f64]) -> Self {
        self.priors = Some(priors.to_vec());
        self
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
        let mut weighted = 0.0f64;
        for i in 0..n {
            let w = self
                .priors
                .as_ref()
                .and_then(|p| p.get(i))
                .copied()
                .unwrap_or(1.0);
            let r = (self.residual_fn)(data, model, i);
            if r * r <= thresh_sq {
                inliers_out.push(i);
                inlier_count += 1;
                weighted += w;
            }
        }

        // Weighted inlier support is used as the score value.
        Score::new(inlier_count, weighted)
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
    priors: Option<Vec<f64>>,
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
            priors: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_priors(mut self, priors: &[f64]) -> Self {
        self.priors = Some(priors.to_vec());
        self
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
            let w = self
                .priors
                .as_ref()
                .and_then(|p| p.get(i))
                .copied()
                .unwrap_or(1.0);
            let r = (self.residual_fn)(data, model, i);
            let r2 = r * r;
            if r2 <= thresh_sq {
                inliers_out.push(i);
                inlier_count += 1;
                cost += w * r2;
            } else {
                cost += w * thresh_sq;
            }
        }

        // Larger scores should be better, so we negate the MSAC cost.
        Score::new(inlier_count, -cost)
    }
}

/// Weighted inlier-count scoring (MinPRAN-style) using optional per-point priors.
pub struct MinpranScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    threshold: f64,
    residual_fn: F,
    priors: Option<Vec<f64>>,
    _marker: std::marker::PhantomData<M>,
}

impl<M, F> MinpranScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    pub fn new(threshold: f64, residual_fn: F) -> Self {
        Self {
            threshold,
            residual_fn,
            priors: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_priors(mut self, priors: &[f64]) -> Self {
        self.priors = Some(priors.to_vec());
        self
    }
}

impl<M, F> Scoring<M> for MinpranScoring<M, F>
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
        let mut weighted = 0.0f64;

        for i in 0..n {
            let w = self
                .priors
                .as_ref()
                .and_then(|p| p.get(i))
                .copied()
                .unwrap_or(1.0);
            let r = (self.residual_fn)(data, model, i);
            if r * r <= thresh_sq {
                inliers_out.push(i);
                inlier_count += 1;
                weighted += w;
            }
        }

        Score::new(inlier_count, weighted)
    }
}

/// Gaussian log-likelihood scoring (approximate GAU) with optional priors.
pub struct GaussianScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    sigma: f64,
    residual_fn: F,
    priors: Option<Vec<f64>>,
    _marker: std::marker::PhantomData<M>,
}

impl<M, F> GaussianScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    pub fn new(sigma: f64, residual_fn: F) -> Self {
        Self {
            sigma,
            residual_fn,
            priors: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_priors(mut self, priors: &[f64]) -> Self {
        self.priors = Some(priors.to_vec());
        self
    }
}

impl<M, F> Scoring<M> for GaussianScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    type Score = Score;

    fn threshold(&self) -> f64 {
        self.sigma * 3.0
    }

    fn score(&self, data: &DataMatrix, model: &M, inliers_out: &mut Vec<usize>) -> Self::Score {
        let n = data.nrows();
        inliers_out.clear();

        let var2 = 2.0 * self.sigma * self.sigma;
        let norm = -(self.sigma * (2.0 * std::f64::consts::PI).sqrt()).ln();

        let mut loglik = 0.0f64;
        let mut inlier_count = 0usize;

        for i in 0..n {
            let w = self
                .priors
                .as_ref()
                .and_then(|p| p.get(i))
                .copied()
                .unwrap_or(1.0);
            let r = (self.residual_fn)(data, model, i);
            let ll = norm - r * r / var2;
            loglik += w * ll;
            if r.abs() < self.threshold() {
                inliers_out.push(i);
                inlier_count += 1;
            }
        }

        Score::new(inlier_count, loglik)
    }
}

/// Simple ML-style scoring using a two-component mixture (inlier Gaussian + uniform outlier).
pub struct MlScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    threshold: f64,
    sigma: f64,
    inlier_prob: f64,
    residual_fn: F,
    priors: Option<Vec<f64>>,
    _marker: std::marker::PhantomData<M>,
}

impl<M, F> MlScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    pub fn new(threshold: f64, sigma: f64, residual_fn: F) -> Self {
        Self {
            threshold,
            sigma,
            inlier_prob: 0.5,
            residual_fn,
            priors: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_inlier_prob(mut self, p: f64) -> Self {
        self.inlier_prob = p.clamp(1e-6, 1.0 - 1e-6);
        self
    }

    pub fn with_priors(mut self, priors: &[f64]) -> Self {
        self.priors = Some(priors.to_vec());
        self
    }
}

impl<M, F> Scoring<M> for MlScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    type Score = Score;

    fn threshold(&self) -> f64 {
        self.threshold
    }

    fn score(&self, data: &DataMatrix, model: &M, inliers_out: &mut Vec<usize>) -> Self::Score {
        let n = data.nrows();
        inliers_out.clear();

        let var2 = 2.0 * self.sigma * self.sigma;
        let gauss_norm = 1.0 / (self.sigma * (2.0 * std::f64::consts::PI).sqrt());
        let outlier_density = 1.0 / self.threshold.max(1e-9);

        let mut loglik = 0.0f64;
        let mut inlier_count = 0usize;

        for i in 0..n {
            let w_prior = self
                .priors
                .as_ref()
                .and_then(|p| p.get(i))
                .copied()
                .unwrap_or(1.0);
            let r = (self.residual_fn)(data, model, i);
            let gauss = gauss_norm * (-r * r / var2).exp();
            let mix = self.inlier_prob * gauss + (1.0 - self.inlier_prob) * outlier_density;
            loglik += w_prior * mix.max(1e-20).ln();
            if r.abs() <= self.threshold {
                inliers_out.push(i);
                inlier_count += 1;
            }
        }

        Score::new(inlier_count, loglik)
    }
}

/// Grid-based scoring that rewards spatial coverage (approximate to C++ grid scoring).
pub struct GridScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    threshold: f64,
    residual_fn: F,
    cell_size: f64,
    priors: Option<Vec<f64>>,
    _marker: std::marker::PhantomData<M>,
}

impl<M, F> GridScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    pub fn new(threshold: f64, cell_size: f64, residual_fn: F) -> Self {
        Self {
            threshold,
            residual_fn,
            cell_size,
            priors: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_priors(mut self, priors: &[f64]) -> Self {
        self.priors = Some(priors.to_vec());
        self
    }
}

impl<M, F> Scoring<M> for GridScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    type Score = Score;

    fn threshold(&self) -> f64 {
        self.threshold
    }

    fn score(&self, data: &DataMatrix, model: &M, inliers_out: &mut Vec<usize>) -> Self::Score {
        let n = data.nrows();
        inliers_out.clear();
        if data.ncols() < 2 || self.cell_size <= 0.0 {
            return Score::new(0, f64::NEG_INFINITY);
        }

        let mut inlier_count = 0usize;
        let mut weighted = 0.0f64;
        let mut grid: std::collections::HashMap<(i64, i64), f64> = std::collections::HashMap::new();

        for i in 0..n {
            let w = self
                .priors
                .as_ref()
                .and_then(|p| p.get(i))
                .copied()
                .unwrap_or(1.0);
            let r = (self.residual_fn)(data, model, i);
            if r.abs() <= self.threshold {
                inliers_out.push(i);
                inlier_count += 1;
                weighted += w;

                let x = (data[(i, 0)] / self.cell_size).floor() as i64;
                let y = (data[(i, 1)] / self.cell_size).floor() as i64;
                *grid.entry((x, y)).or_insert(0.0) += w;
            }
        }

        // Coverage bonus: sum of cell weights encourages spatial spread.
        let coverage: f64 = grid.values().sum();
        Score::new(inlier_count, weighted + coverage)
    }
}

/// Lanczos approximation of ln Γ(x) for x > 0.
fn ln_gamma(x: f64) -> f64 {
    // Coefficients from Numerical Recipes.
    const COEFFS: [f64; 9] = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
        -0.0,
    ];
    if x < 0.5 {
        // Reflection formula
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - ln_gamma(1.0 - x);
    }
    let z = x - 1.0;
    let mut sum = 0.999_999_999_999_809_9;
    for (i, c) in COEFFS.iter().enumerate() {
        sum += c / (z + (i as f64) + 1.0);
    }
    let t = z + COEFFS.len() as f64 - 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
}

fn gamma_fn(x: f64) -> f64 {
    ln_gamma(x).exp()
}

/// Regularized lower incomplete gamma P(a, x) via series expansion or continued fraction.
fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    // Series expansion for x < a + 1
    if x < a + 1.0 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..200 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-14 {
                break;
            }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        // Continued fraction for upper; then 1 - Q
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / 1e-30;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..200 {
            let an = -(i as f64) * ((i as f64) - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = b + an / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-14 {
                break;
            }
        }
        1.0 - (-x + a * x.ln() - ln_gamma(a)).exp() * h
    }
}

fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    regularized_lower_gamma(a, x) * gamma_fn(a)
}

fn upper_incomplete_gamma(a: f64, x: f64) -> f64 {
    gamma_fn(a) - lower_incomplete_gamma(a, x)
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
    priors: Option<Vec<f64>>,
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
            priors: None,
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

    pub fn with_priors(mut self, priors: &[f64]) -> Self {
        self.priors = Some(priors.to_vec());
        self
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
            let gamma_a = gamma_fn(n_plus_1_per_2);
            let upper_gamma_lower = upper_incomplete_gamma(n_plus_1_per_2, residual_norm);
            let lower_gamma = gamma_a - upper_gamma_lower;

            // Upper incomplete gamma approximation
            let upper_gamma = upper_incomplete_gamma(n_minus_1_per_2, residual_norm);
            let value0 = upper_incomplete_gamma(n_minus_1_per_2, 0.0);

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
            let w = self
                .priors
                .as_ref()
                .and_then(|p| p.get(i))
                .copied()
                .unwrap_or(1.0);
            let r = (self.residual_fn)(data, model, i);
            let r_sq = r * r;

            let loss = self.compute_loss(r_sq);
            cost += w * loss;

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

/// σ-consensus++ (MAGSAC++) style scoring using closed-form weights/ρ with optional priors.
pub struct SigmaConsensusScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    sigma_max: f64,
    degrees_of_freedom: usize,
    k_quantile: f64,
    residual_fn: F,
    priors: Option<Vec<f64>>,
    _marker: std::marker::PhantomData<M>,
}

impl<M, F> SigmaConsensusScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    pub fn new(sigma_max: f64, degrees_of_freedom: usize, residual_fn: F) -> Self {
        Self {
            sigma_max,
            degrees_of_freedom,
            k_quantile: 3.64, // ~0.99 chi quantile (common choice in the paper)
            residual_fn,
            priors: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_k(mut self, k: f64) -> Self {
        self.k_quantile = k;
        self
    }

    pub fn with_priors(mut self, priors: &[f64]) -> Self {
        self.priors = Some(priors.to_vec());
        self
    }

    fn c_n(&self) -> f64 {
        let n = self.degrees_of_freedom as f64;
        (2.0f64.powf(n / 2.0) * gamma_fn(n / 2.0)).recip()
    }

    fn lut_lookup(&self, t_norm: f64) -> (f64, f64, f64) {
        let key = (self.degrees_of_freedom, self.k_quantile.to_bits());
        let map = SIGMA_LUT.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = map.lock().expect("gamma LUT lock poisoned");
        let table = guard.entry(key).or_insert_with(|| {
            let step = self.k_quantile / (SIGMA_LUT_SAMPLES as f64 - 1.0);
            let mut table = Vec::with_capacity(SIGMA_LUT_SAMPLES);
            for i in 0..SIGMA_LUT_SAMPLES {
                let t = i as f64 * step;
                let x = t * t / 2.0;
                let upper = upper_incomplete_gamma((self.degrees_of_freedom as f64 - 1.0) / 2.0, x);
                let lower = lower_incomplete_gamma((self.degrees_of_freedom as f64 + 1.0) / 2.0, x);
                table.push((upper, lower));
            }
            table
        });
        let clamped = t_norm.clamp(0.0, self.k_quantile);
        let pos = clamped / self.k_quantile * (table.len() as f64 - 1.0);
        let idx = pos.floor() as usize;
        let frac = pos - idx as f64;
        let (u0, l0) = table[idx];
        let (u1, l1) = if idx + 1 < table.len() {
            table[idx + 1]
        } else {
            (u0, l0)
        };
        let upper = u0 + frac * (u1 - u0);
        let lower = l0 + frac * (l1 - l0);
        let upper_k = table.last().map(|p| p.0).unwrap_or(upper);
        (upper, lower, upper_k)
    }

    fn rho(&self, r: f64) -> f64 {
        let k_sigma = self.k_quantile * self.sigma_max;
        let r_clamped = r.min(k_sigma);
        let n = self.degrees_of_freedom as f64;
        let c_n = self.c_n();
        let power = 2.0f64.powf((n + 1.0) / 2.0);
        let sigma_sq = self.sigma_max * self.sigma_max;
        let t_norm = r_clamped / self.sigma_max;
        let (upper_term, lower, upper_k) = self.lut_lookup(t_norm);

        let term1 = (sigma_sq / 2.0) * lower;
        let term2 = (r_clamped * r_clamped / 4.0) * (upper_term - upper_k);

        (1.0 / self.sigma_max) * c_n * power * (term1 + term2)
    }
}

impl<M, F> Scoring<M> for SigmaConsensusScoring<M, F>
where
    F: Fn(&DataMatrix, &M, usize) -> f64,
{
    type Score = Score;

    fn threshold(&self) -> f64 {
        self.k_quantile * self.sigma_max
    }

    fn score(&self, data: &DataMatrix, model: &M, inliers_out: &mut Vec<usize>) -> Self::Score {
        let n = data.nrows();
        inliers_out.clear();

        let mut inlier_count = 0usize;
        let mut loss = 0.0f64;

        for i in 0..n {
            let w_prior = self
                .priors
                .as_ref()
                .and_then(|p| p.get(i))
                .copied()
                .unwrap_or(1.0);
            let r = (self.residual_fn)(data, model, i).abs();
            let rho_val = self.rho(r);
            loss += w_prior * rho_val;

            if r <= self.threshold() {
                inliers_out.push(i);
                inlier_count += 1;
            }
        }

        Score::new(inlier_count, -loss)
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

        for (k, logc_n_val) in logc_n.iter_mut().enumerate() {
            *logc_n_val = self.log_combi(k, n, &log10_table);
        }
        for (k_val, logc_k_val) in logc_k.iter_mut().enumerate() {
            *logc_k_val = self.log_combi(self.minimal_sample_size, k_val, &log10_table);
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
