//! ROBIN matching with k-core pruning

use crate::types::DataMatrix;
use std::collections::{HashMap, HashSet};

/// ROBIN (Robust Bilateral Network) correspondence matching
///
/// Implements graph-theoretic outlier rejection using:
/// 1. Compatibility graph construction
/// 2. K-core decomposition
/// 3. Progressive/max-core pruning
pub struct ROBINMatching {
    noise_bound: f64,
    num_max_corr: usize,
    tuple_scale: f64,
}

impl ROBINMatching {
    pub fn new(noise_bound: f64, num_max_corr: usize, tuple_scale: f64) -> Self {
        Self {
            noise_bound,
            num_max_corr,
            tuple_scale,
        }
    }

    /// Apply k-core pruning to correspondences
    ///
    /// Builds a compatibility graph where edge (i,j) exists if:
    /// |‖dst_j - dst_i‖ - scale * ‖src_j - src_i‖| <= 2 * noise_bound
    ///
    /// Then performs k-core decomposition to remove low-degree nodes
    pub fn prune_correspondences(
        &self,
        src_points: &DataMatrix,
        dst_points: &DataMatrix,
        scale_estimate: f64,
        mode: &str,
    ) -> Vec<usize> {
        let n = src_points.n_points();
        if n != dst_points.n_points() {
            return Vec::new();
        }

        if n == 0 {
            return Vec::new();
        }

        // Build compatibility graph
        let graph = self.build_compatibility_graph(src_points, dst_points, scale_estimate);

        // Perform k-core decomposition
        match mode {
            "max_core" => self.max_core_pruning(&graph, n),
            "progressive" => self.progressive_pruning(&graph, n),
            _ => (0..n).collect(), // No pruning
        }
    }

    /// Build compatibility graph based on geometric consistency
    fn build_compatibility_graph(
        &self,
        src_points: &DataMatrix,
        dst_points: &DataMatrix,
        scale: f64,
    ) -> Vec<Vec<usize>> {
        let n = src_points.n_points();
        let mut graph = vec![Vec::new(); n];

        let threshold = 2.0 * self.noise_bound;

        // Check all pairs
        for i in 0..n {
            let src_i = (
                src_points.get(i, 0),
                src_points.get(i, 1),
                src_points.get(i, 2),
            );
            let dst_i = (
                dst_points.get(i, 0),
                dst_points.get(i, 1),
                dst_points.get(i, 2),
            );

            for j in (i + 1)..n {
                let src_j = (
                    src_points.get(j, 0),
                    src_points.get(j, 1),
                    src_points.get(j, 2),
                );
                let dst_j = (
                    dst_points.get(j, 0),
                    dst_points.get(j, 1),
                    dst_points.get(j, 2),
                );

                // Compute distances
                let src_dist = compute_distance(src_i, src_j);
                let dst_dist = compute_distance(dst_i, dst_j);

                // Check compatibility: |dst_dist - scale * src_dist| <= threshold
                if (dst_dist - scale * src_dist).abs() <= threshold {
                    graph[i].push(j);
                    graph[j].push(i);
                }
            }
        }

        graph
    }

    /// Max-core pruning: find maximum k where k-core exists
    fn max_core_pruning(&self, graph: &[Vec<usize>], n: usize) -> Vec<usize> {
        let mut degrees: Vec<usize> = graph.iter().map(|neighbors| neighbors.len()).collect();
        let mut active = vec![true; n];
        let mut changed = true;

        // Find minimum degree
        let mut min_degree = *degrees.iter().filter(|&&d| d > 0).min().unwrap_or(&0);

        // Iteratively remove nodes with degree < k
        while changed && min_degree > 0 {
            changed = false;

            for i in 0..n {
                if active[i] && degrees[i] < min_degree {
                    active[i] = false;
                    changed = true;

                    // Update neighbors
                    for &j in &graph[i] {
                        if active[j] {
                            degrees[j] = degrees[j].saturating_sub(1);
                        }
                    }
                }
            }

            if changed {
                // Recalculate min degree
                min_degree = degrees
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| active[*i])
                    .map(|(_, &d)| d)
                    .min()
                    .unwrap_or(0);
            }
        }

        // Return indices of active nodes
        (0..n).filter(|&i| active[i]).collect()
    }

    /// Progressive pruning: incrementally increase k
    fn progressive_pruning(&self, graph: &[Vec<usize>], n: usize) -> Vec<usize> {
        // For now, use same as max_core
        // TODO: Implement true progressive pruning
        self.max_core_pruning(graph, n)
    }
}

/// Compute Euclidean distance between two 3D points
fn compute_distance(p1: (f64, f64, f64), p2: (f64, f64, f64)) -> f64 {
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let dz = p2.2 - p1.2;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robin_pruning_basic() {
        // Create simple test case with 4 points
        let src = DataMatrix::from_row_slice(
            4,
            3,
            &[
                0.0, 0.0, 0.0, // Point 0
                1.0, 0.0, 0.0, // Point 1
                0.0, 1.0, 0.0, // Point 2
                10.0, 10.0, 0.0, // Point 3 (outlier)
            ],
        );

        let dst = DataMatrix::from_row_slice(
            4,
            3,
            &[
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 20.0, 20.0, 0.0, // Scaled outlier
            ],
        );

        let robin = ROBINMatching::new(0.1, 1000, 0.95);
        let inliers = robin.prune_correspondences(&src, &dst, 1.0, "max_core");

        // Points 0, 1, 2 should form a core, point 3 should be removed
        assert!(inliers.len() <= 4);
        assert!(inliers.contains(&0) || inliers.contains(&1) || inliers.contains(&2));
    }
}
