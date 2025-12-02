//! Miscellaneous utilities shared across the SupeRANSAC Rust port.
//!
//! This module starts with a small wrapper around `rand` that mirrors the
//! behavior of the C++ `UniformRandomGenerator` in
//! `include/utils/uniform_random_generator.h`.

use rand::distributions::Uniform;
use rand::prelude::*;

/// Uniform integer random-number generator similar to the C++ utility.
///
/// By default this uses a randomly seeded RNG, but test code can construct
/// it from a fixed seed for reproducible behavior.
pub struct UniformRandomGenerator<T>
where
    T: Copy + rand::distributions::uniform::SampleUniform + PartialOrd,
{
    rng: StdRng,
    dist: Option<Uniform<T>>,
}

impl<T> UniformRandomGenerator<T>
where
    T: Copy + rand::distributions::uniform::SampleUniform + PartialOrd,
{
    /// Construct with a random seed (suitable for production use).
    pub fn new() -> Self {
        let rng = StdRng::from_rng(thread_rng()).expect("failed to seed StdRng");
        Self { rng, dist: None }
    }

    /// Construct with a fixed seed (useful for tests).
    pub fn from_seed(seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self { rng, dist: None }
    }

    /// Reset the distribution range.
    pub fn reset(&mut self, min: T, max: T) {
        self.dist = Some(Uniform::new_inclusive(min, max));
    }

    /// Draw a single random value using the current distribution.
    pub fn next(&mut self) -> T {
        let dist = self
            .dist
            .as_ref()
            .expect("UniformRandomGenerator: distribution not initialized");
        self.rng.sample(dist)
    }

    /// Generate a set of unique random integers in `[min, max]` into `out`.
    ///
    /// This is a straightforward port of the C++ `generateUniqueRandomSet`, and
    /// is suitable for small sample sizes typical of minimal solvers.
    pub fn gen_unique(&mut self, out: &mut [T], min: T, max: T)
    where
        T: Eq,
    {
        self.reset(min, max);
        let n = out.len();
        for i in 0..n {
            loop {
                let candidate = self.next();
                // Check uniqueness among already-filled entries.
                if out[..i].iter().all(|&v| v != candidate) {
                    out[i] = candidate;
                    break;
                }
            }
        }
    }

    /// Generate a set of unique random integers using the *current* distribution.
    ///
    /// This mirrors the C++ overload that relies on the previously configured
    /// range in `reset`.
    pub fn gen_unique_current(&mut self, out: &mut [T])
    where
        T: Eq,
    {
        let n = out.len();
        for i in 0..n {
            loop {
                let candidate = self.next();
                if out[..i].iter().all(|&v| v != candidate) {
                    out[i] = candidate;
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::UniformRandomGenerator;

    #[test]
    fn unique_samples_within_bounds() {
        let mut rng = UniformRandomGenerator::<u32>::from_seed(1234);
        let mut buf = [0u32; 5];
        rng.gen_unique(&mut buf, 0, 10);

        // All within range
        assert!(buf.iter().all(|&v| v <= 10));

        // All unique
        for i in 0..buf.len() {
            for j in (i + 1)..buf.len() {
                assert_ne!(buf[i], buf[j]);
            }
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut rng1 = UniformRandomGenerator::<u32>::from_seed(42);
        let mut rng2 = UniformRandomGenerator::<u32>::from_seed(42);

        rng1.reset(0, 100);
        rng2.reset(0, 100);

        let a1: Vec<u32> = (0..10).map(|_| rng1.next()).collect();
        let a2: Vec<u32> = (0..10).map(|_| rng2.next()).collect();

        assert_eq!(a1, a2);
    }
}

/// Gaussian elimination with partial pivoting to solve A * x = b.
/// The matrix `augmented` should be [A | b] where A is n x n and b is n x 1.
/// The result is stored in `result`.
/// This matches the C++ `gaussElimination` implementation.
pub fn gauss_elimination(augmented: &mut nalgebra::DMatrix<f64>, result: &mut nalgebra::DVector<f64>) -> bool {
    let n = augmented.nrows();
    if n != augmented.ncols() - 1 || n != result.len() {
        return false;
    }

    // Pivotisation: find row with largest pivot element
    for i in 0..n {
        let mut max_row = i;
        let mut max_val = augmented[(i, i)].abs();

        for k in (i + 1)..n {
            let val = augmented[(k, i)].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..augmented.ncols() {
                let temp = augmented[(i, j)];
                augmented[(i, j)] = augmented[(max_row, j)];
                augmented[(max_row, j)] = temp;
            }
        }

        // Check for singular matrix
        if augmented[(i, i)].abs() < 1e-10 {
            return false;
        }

        // Elimination process
        for k in (i + 1)..n {
            let factor = augmented[(k, i)] / augmented[(i, i)];
            for j in i..augmented.ncols() {
                augmented[(k, j)] -= factor * augmented[(i, j)];
            }
        }
    }

    // Back-substitution
    for i in (0..n).rev() {
        result[i] = augmented[(i, n)];
        for j in (i + 1)..n {
            result[i] -= augmented[(i, j)] * result[j];
        }
        result[i] /= augmented[(i, i)];
    }

    true
}

/// Solve a cubic equation: x^3 + c2*x^2 + c1*x + c0 = 0
/// Returns the number of real roots found (1 or 3) and stores them in `roots`.
/// This matches the C++ `solveCubicReal` implementation.
pub fn solve_cubic_real(c2: f64, c1: f64, c0: f64, roots: &mut [f64; 3]) -> usize {
    let a = c1 - c2 * c2 / 3.0;
    let b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    let mut c = b * b / 4.0 + a * a * a / 27.0;

    let n_roots = if c > 0.0 {
        c = c.sqrt();
        let b_neg = -0.5 * b;
        roots[0] = (b_neg + c).cbrt() + (b_neg - c).cbrt() - c2 / 3.0;
        1
    } else {
        c = 3.0 * b / (2.0 * a) * (-3.0 / a).sqrt();
        let d = 2.0 * (-a / 3.0).sqrt();
        let acos_c = c.acos();
        roots[0] = d * (acos_c / 3.0).cos() - c2 / 3.0;
        roots[1] = d * (acos_c / 3.0 - 2.09439510239319526263557236234192).cos() - c2 / 3.0; // 2*pi/3
        roots[2] = d * (acos_c / 3.0 - 4.18879020478639052527114472468384).cos() - c2 / 3.0; // 4*pi/3
        3
    };

    // Single Newton iteration for refinement
    for i in 0..n_roots {
        let x = roots[i];
        let x2 = x * x;
        let x3 = x * x2;
        let dx = -(x3 + c2 * x2 + c1 * x + c0) / (3.0 * x2 + 2.0 * c2 * x + c1);
        roots[i] += dx;
    }

    n_roots
}

/// Sturm sequence-based polynomial root finder for high-degree polynomials.
/// This is a simplified implementation for degree 10 polynomials (used in 5-point Nister solver).
pub mod sturm {
    use super::*;

    /// Evaluate polynomial using Horner's method.
    /// Assumes that coeffs[degree] = 1.0 (monic polynomial)
    pub fn polyval<const N: usize>(coeffs: &[f64], x: f64) -> f64 {
        let mut fx = x + coeffs[N - 1];
        for i in (0..N - 1).rev() {
            fx = x * fx + coeffs[i];
        }
        fx
    }

    /// Count sign changes in Sturm sequence at point x.
    /// This is a simplified version for degree 10.
    pub fn sign_changes<const N: usize>(svec: &[f64], x: f64) -> usize {
        let mut f = vec![0.0; N + 1];
        f[N] = svec[3 * N - 1];
        f[N - 1] = svec[3 * N - 3] + x * svec[3 * N - 2];

        for i in (0..N - 1).rev() {
            f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2];
        }

        // Count sign changes
        let mut count = 0;
        for i in 0..N {
            if (f[i] < 0.0) != (f[i + 1] < 0.0) {
                count += 1;
            }
        }
        count
    }

    /// Get Cauchy bound on real roots
    pub fn get_bounds<const N: usize>(fvec: &[f64]) -> f64 {
        let mut max = 0.0f64;
        for i in 0..N {
            max = max.max(fvec[i].abs());
        }
        1.0 + max
    }

    /// Solve degree 10 polynomial using Sturm bracketing (simplified).
    /// Returns number of roots found.
    pub fn bisect_sturm_10(coeffs: &[f64; 11], roots: &mut [f64; 10], tol: f64) -> usize {
        if coeffs[10].abs() < 1e-10 {
            return 0;
        }

        // Normalize polynomial
        let c_inv = 1.0 / coeffs[10];
        let mut fvec = [0.0; 21]; // polynomial + derivative
        for i in 0..10 {
            fvec[i] = coeffs[i] * c_inv;
        }
        fvec[10] = 1.0;

        // Compute derivative
        for i in 0..9 {
            fvec[11 + i] = fvec[i + 1] * ((i + 1) as f64 / 10.0);
        }
        fvec[20] = 1.0;

        // Simplified root finding: use a basic approach
        // For a full implementation, we'd need the full Sturm sequence construction
        // This is a placeholder that uses a simpler method
        let bounds = get_bounds::<10>(&fvec[..10]);
        let a = -bounds;
        let b = bounds;

        // Use a simple grid search and Newton refinement
        let mut n_roots = 0;
        let step = (b - a) / 1000.0;

        for i in 0..1000 {
            let x0 = a + i as f64 * step;
            let x1 = x0 + step;

            let f0 = polyval::<10>(&fvec[..10], x0);
            let f1 = polyval::<10>(&fvec[..10], x1);

            // Check for sign change (root in interval)
            if (f0 < 0.0) != (f1 < 0.0) {
                // Refine using Newton's method
                let mut x = (x0 + x1) / 2.0;
                for _iter in 0..20 {
                    let fx = polyval::<10>(&fvec[..10], x);
                    if fx.abs() < tol {
                        break;
                    }
                    // Approximate derivative
                    let fpx = (polyval::<10>(&fvec[..10], x + 1e-8) - fx) / 1e-8;
                    if fpx.abs() < 1e-10 {
                        break;
                    }
                    let dx = fx / fpx;
                    x -= dx;
                    if dx.abs() < tol {
                        break;
                    }
                }

                if n_roots < 10 {
                    roots[n_roots] = x;
                    n_roots += 1;
                }
            }
        }

        n_roots
    }
}
