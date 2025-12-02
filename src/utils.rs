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
