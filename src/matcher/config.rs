//! Configuration for KISS-Matcher

/// Configuration parameters for KISS-Matcher
#[derive(Debug, Clone)]
pub struct KISSMatcherConfig {
    // Voxel downsampling
    pub use_voxel_sampling: bool,
    pub voxel_size: f64,

    // Feature extraction
    pub normal_radius: f64, // Typically 2.5-3.0 * voxel_size
    pub fpfh_radius: f64,   // Typically 5.0 * voxel_size
    pub fpfh_bins: usize,   // Histogram bins (typically 11)

    // Geometric filtering
    pub the_linearity: f64,   // 1.0 = disabled, < 1.0 = filter planar regions
    pub min_neighbors: usize, // Minimum neighbors for valid points

    // Matching parameters
    pub use_ratio_test: bool,
    pub ratio_threshold: f64, // Lowe's ratio test threshold (typically 0.8)
    pub num_max_corr: usize,  // Max correspondences to keep

    // ROBIN outlier rejection
    pub robin_noise_bound: f64, // Typically voxel_size * 1.0
    pub robin_mode: String,     // "max_core" or "progressive"
    pub tuple_scale: f64,       // Scale tolerance (typically 0.95)

    // GNC solver
    pub solver_noise_bound: f64,   // Typically voxel_size * 0.75
    pub gnc_max_iterations: usize, // GNC iterations
    pub gnc_mu_step: f64,          // GNC continuation parameter
}

impl KISSMatcherConfig {
    /// Create default configuration for given voxel size
    pub fn new(voxel_size: f64) -> Self {
        if voxel_size < 0.005 {
            panic!("Voxel size too small: {voxel_size}");
        }

        let normal_radius = 3.0 * voxel_size;
        let fpfh_radius = 5.0 * voxel_size;
        let robin_noise_bound = voxel_size * 1.0;
        let solver_noise_bound = voxel_size * 0.75;

        Self {
            use_voxel_sampling: true,
            voxel_size,
            normal_radius,
            fpfh_radius,
            fpfh_bins: 11,
            the_linearity: 1.0, // Disabled by default
            min_neighbors: 5,
            use_ratio_test: true,
            ratio_threshold: 0.8,
            num_max_corr: 5000,
            robin_noise_bound: robin_noise_bound.min(1.0), // Clamp to 1.0
            robin_mode: "max_core".to_string(),
            tuple_scale: 0.95,
            solver_noise_bound: solver_noise_bound.min(1.0), // Clamp to 1.0
            gnc_max_iterations: 50,
            gnc_mu_step: 1.4,
        }
    }

    /// Create configuration with custom parameters
    pub fn with_params(
        voxel_size: f64,
        normal_r_gain: f64,
        fpfh_r_gain: f64,
        robin_noise_bound_gain: f64,
        solver_noise_bound_gain: f64,
    ) -> Self {
        if voxel_size < 0.005 {
            panic!("Voxel size too small: {voxel_size}");
        }

        if robin_noise_bound_gain < solver_noise_bound_gain {
            panic!(
                "solver_noise_bound_gain ({solver_noise_bound_gain}) should be <= robin_noise_bound_gain ({robin_noise_bound_gain})"
            );
        }

        let normal_radius = normal_r_gain * voxel_size;
        let fpfh_radius = fpfh_r_gain * voxel_size;
        let robin_noise_bound = (voxel_size * robin_noise_bound_gain).min(1.0);
        let solver_noise_bound = (voxel_size * solver_noise_bound_gain).min(1.0);

        Self {
            use_voxel_sampling: true,
            voxel_size,
            normal_radius,
            fpfh_radius,
            fpfh_bins: 11,
            the_linearity: 1.0,
            min_neighbors: 5,
            use_ratio_test: true,
            ratio_threshold: 0.8,
            num_max_corr: 5000,
            robin_noise_bound,
            robin_mode: "max_core".to_string(),
            tuple_scale: 0.95,
            solver_noise_bound,
            gnc_max_iterations: 50,
            gnc_mu_step: 1.4,
        }
    }
}

impl Default for KISSMatcherConfig {
    fn default() -> Self {
        Self::new(0.3) // Default 0.3m voxel size
    }
}
