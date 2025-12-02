//! RANSAC configuration types for the SupeRANSAC Rust port.
//!
//! These mirror the semantics of the C++ `RANSACSettings` struct and the
//! associated enum classes found in:
//! - `include/settings.h`
//! - `include/scoring/types.h`
//! - `include/samplers/types.h`
//! - `include/neighborhood/types.h`
//! - `include/local_optimization/types.h`
//! - `include/termination/types.h`
//! - `include/inlier_selectors/types.h`

/// Scoring strategy used to evaluate models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringType {
    Ransac,
    Msac,
    Magsac,
    Minpran,
    Acransac,
    Gau,
    Ml,
    Grid,
}

/// Sampling strategy for minimal sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerType {
    Uniform,
    Prosac,
    Napsac,
    ProgressiveNapsac,
    ImportanceSampler,
    ArSampler,
    Exhaustive,
}

/// Neighborhood graph type used for spatial reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborhoodType {
    Grid,
    BruteForce,
    FlannKnn,
    FlannRadius,
}

/// Local optimization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalOptimizationType {
    None,
    Lsq,
    Irls,
    NestedRansac,
    GcRansac,
    IteratedLmeds,
    CrossValidation,
}

/// Termination criterion type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationType {
    Ransac,
}

/// Inlier selector type (e.g., space-partitioning RANSAC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InlierSelectorType {
    None,
    SpacePartitioningRansac,
}

/// Settings specific to the adaptive reordering (AR) sampler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArSamplerSettings {
    pub estimator_variance: f64,
    pub randomness: f64,
}

impl Default for ArSamplerSettings {
    fn default() -> Self {
        Self {
            estimator_variance: 0.9765,
            randomness: 0.01,
        }
    }
}

/// Settings controlling local optimization procedures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalOptimizationSettings {
    pub max_iterations: usize,
    pub graph_cut_number: usize,
    pub sample_size_multiplier: f64,
    pub spatial_coherence_weight: f64,
}

impl Default for LocalOptimizationSettings {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            graph_cut_number: 20,
            sample_size_multiplier: 7.0,
            spatial_coherence_weight: 0.1,
        }
    }
}

/// Settings for constructing and querying neighborhood graphs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeighborhoodSettings {
    pub neighborhood_size: f64,
    pub neighborhood_grid_density: f64,
    pub nearest_neighbor_number: usize,
}

impl Default for NeighborhoodSettings {
    fn default() -> Self {
        Self {
            neighborhood_size: 20.0,
            neighborhood_grid_density: 4.0,
            nearest_neighbor_number: 6,
        }
    }
}

/// Main configuration object for the SupeRANSAC pipeline.
///
/// This is a direct Rust analogue of the C++ `superansac::RANSACSettings`
/// with the same default values.
#[derive(Debug, Clone, PartialEq)]
pub struct RansacSettings {
    /// Minimum number of iterations.
    pub min_iterations: usize,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Inlier threshold in the chosen residual domain.
    pub inlier_threshold: f64,
    /// Desired confidence level in \[0, 1\].
    pub confidence: f64,

    pub scoring: ScoringType,
    pub sampler: SamplerType,
    pub neighborhood: NeighborhoodType,
    pub local_optimization: LocalOptimizationType,
    pub final_optimization: LocalOptimizationType,
    pub termination_criterion: TerminationType,
    pub inlier_selector: InlierSelectorType,

    pub ar_sampler_settings: ArSamplerSettings,
    pub local_optimization_settings: LocalOptimizationSettings,
    pub final_optimization_settings: LocalOptimizationSettings,
    pub neighborhood_settings: NeighborhoodSettings,
}

impl Default for RansacSettings {
    fn default() -> Self {
        Self {
            min_iterations: 1000,
            max_iterations: 5000,
            inlier_threshold: 1.5,
            confidence: 0.99,
            scoring: ScoringType::Magsac,
            sampler: SamplerType::Prosac,
            neighborhood: NeighborhoodType::Grid,
            local_optimization: LocalOptimizationType::NestedRansac,
            final_optimization: LocalOptimizationType::Irls,
            termination_criterion: TerminationType::Ransac,
            inlier_selector: InlierSelectorType::None,
            ar_sampler_settings: ArSamplerSettings::default(),
            local_optimization_settings: LocalOptimizationSettings::default(),
            final_optimization_settings: LocalOptimizationSettings::default(),
            neighborhood_settings: NeighborhoodSettings::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ransac_settings_match_cpp_defaults() {
        let cfg = RansacSettings::default();
        assert_eq!(cfg.min_iterations, 1000);
        assert_eq!(cfg.max_iterations, 5000);
        assert!((cfg.inlier_threshold - 1.5).abs() < 1e-12);
        assert!((cfg.confidence - 0.99).abs() < 1e-12);

        assert_eq!(cfg.scoring, ScoringType::Magsac);
        assert_eq!(cfg.sampler, SamplerType::Prosac);
        assert_eq!(cfg.neighborhood, NeighborhoodType::Grid);
        assert_eq!(cfg.local_optimization, LocalOptimizationType::NestedRansac);
        assert_eq!(cfg.final_optimization, LocalOptimizationType::Irls);
        assert_eq!(cfg.termination_criterion, TerminationType::Ransac);
        assert_eq!(cfg.inlier_selector, InlierSelectorType::None);
    }

    #[test]
    fn default_nested_settings_match_cpp_defaults() {
        let cfg = RansacSettings::default();
        assert_eq!(cfg.ar_sampler_settings, ArSamplerSettings::default());
        assert_eq!(
            cfg.local_optimization_settings,
            LocalOptimizationSettings::default()
        );
        assert_eq!(
            cfg.final_optimization_settings,
            LocalOptimizationSettings::default()
        );
        assert_eq!(cfg.neighborhood_settings, NeighborhoodSettings::default());
    }
}
