//! Preprocessing utilities for point clouds

use crate::types::DataMatrix;
use std::collections::HashMap;

#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use std::sync::Mutex;

/// Voxel downsample a point cloud
///
/// Groups points into voxel grid cells and keeps one representative point per voxel.
/// This reduces density while preserving overall structure.
///
/// # Arguments
/// * `points` - Input point cloud (N×3)
/// * `voxel_size` - Size of voxel grid cells
///
/// # Returns
/// Downsampled point cloud with at most one point per voxel
pub fn voxel_downsample(points: &DataMatrix, voxel_size: f64) -> DataMatrix {
    if voxel_size <= 0.0 {
        return points.clone();
    }

    let n_points = points.n_points();
    if n_points == 0 {
        return points.clone();
    }

    // Map voxel coordinates to point index
    #[cfg(feature = "rayon")]
    let voxel_map: Mutex<HashMap<(i32, i32, i32), Vec<usize>>> = Mutex::new(HashMap::new());
    #[cfg(not(feature = "rayon"))]
    let mut voxel_map: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

    #[cfg(feature = "progress")]
    let pb = crate::progress::create_progress_bar_if_large(n_points as u64, "Voxelizing", 10000);

    // Assign points to voxels
    #[cfg(feature = "rayon")]
    {
        (0..n_points).into_par_iter().for_each(|i| {
            #[cfg(feature = "progress")]
            if let Some(ref pb) = pb {
                pb.inc(1);
            }

            let x = points.get(i, 0);
            let y = points.get(i, 1);
            let z = points.get(i, 2);

            let vx = (x / voxel_size).floor() as i32;
            let vy = (y / voxel_size).floor() as i32;
            let vz = (z / voxel_size).floor() as i32;

            voxel_map
                .lock()
                .unwrap()
                .entry((vx, vy, vz))
                .or_default()
                .push(i);
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for i in 0..n_points {
            #[cfg(feature = "progress")]
            if let Some(ref pb) = pb {
                pb.inc(1);
            }

            let x = points.get(i, 0);
            let y = points.get(i, 1);
            let z = points.get(i, 2);

            let vx = (x / voxel_size).floor() as i32;
            let vy = (y / voxel_size).floor() as i32;
            let vz = (z / voxel_size).floor() as i32;

            voxel_map
                .entry((vx, vy, vz))
                .or_insert_with(Vec::new)
                .push(i);
        }
    }

    #[cfg(feature = "progress")]
    if let Some(pb) = pb {
        pb.finish();
    }

    #[cfg(feature = "rayon")]
    let voxel_map = voxel_map.into_inner().unwrap();

    // For each voxel, compute centroid of all points in that voxel
    let mut downsampled_points = Vec::with_capacity(voxel_map.len() * 3);

    for indices in voxel_map.values() {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_z = 0.0;

        for &idx in indices {
            sum_x += points.get(idx, 0);
            sum_y += points.get(idx, 1);
            sum_z += points.get(idx, 2);
        }

        let n = indices.len() as f64;
        downsampled_points.push(sum_x / n);
        downsampled_points.push(sum_y / n);
        downsampled_points.push(sum_z / n);
    }

    let n_out = downsampled_points.len() / 3;
    DataMatrix::from_row_slice(n_out, 3, &downsampled_points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_voxel_downsample_exact_grid() {
        // Create 8 points at corners of a unit cube
        let points = DataMatrix::from_row_slice(
            8,
            3,
            &[
                0.0, 0.0, 0.0, // corner 1
                1.0, 0.0, 0.0, // corner 2
                0.0, 1.0, 0.0, // corner 3
                1.0, 1.0, 0.0, // corner 4
                0.0, 0.0, 1.0, // corner 5
                1.0, 0.0, 1.0, // corner 6
                0.0, 1.0, 1.0, // corner 7
                1.0, 1.0, 1.0, // corner 8
            ],
        );

        // Voxel size 0.5 should merge each pair
        let downsampled = voxel_downsample(&points, 0.5);
        assert_eq!(downsampled.n_points(), 8); // Still 8 because they're at corners

        // Voxel size 2.0 should merge all into one
        let downsampled = voxel_downsample(&points, 2.0);
        assert_eq!(downsampled.n_points(), 1);

        // Check centroid is at (0.5, 0.5, 0.5)
        assert_relative_eq!(downsampled.get(0, 0), 0.5, epsilon = 1e-6);
        assert_relative_eq!(downsampled.get(0, 1), 0.5, epsilon = 1e-6);
        assert_relative_eq!(downsampled.get(0, 2), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_voxel_downsample_duplicates() {
        // Create points with duplicates
        let points = DataMatrix::from_row_slice(
            6,
            3,
            &[
                0.0, 0.0, 0.0, 0.01, 0.01, 0.01, // near (0,0,0)
                1.0, 0.0, 0.0, 1.01, 0.01, 0.01, // near (1,0,0)
                0.0, 1.0, 0.0, 0.01, 1.01, 0.01, // near (0,1,0)
            ],
        );

        // Voxel size 0.1 should merge pairs
        let downsampled = voxel_downsample(&points, 0.1);
        assert_eq!(downsampled.n_points(), 3);
    }

    #[test]
    fn test_voxel_downsample_no_change() {
        let points =
            DataMatrix::from_row_slice(3, 3, &[0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0]);

        // Very small voxel size - no merging
        let downsampled = voxel_downsample(&points, 0.01);
        assert_eq!(downsampled.n_points(), 3);
    }

    #[test]
    fn test_voxel_downsample_zero_size() {
        let points =
            DataMatrix::from_row_slice(3, 3, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

        let downsampled = voxel_downsample(&points, 0.0);
        assert_eq!(downsampled.n_points(), points.n_points());
    }
}
