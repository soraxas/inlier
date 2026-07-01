//! Neighborhood graph implementations for spatial sampling strategies.

use crate::types::DataMatrix;
use kiddo::{KdTree, SquaredEuclidean};

/// Minimal neighborhood graph abstraction used by NAPSAC.
///
/// This trait allows you to implement custom spatial neighborhood structures
/// for samplers like NAPSAC that require spatial relationships between points.
///
/// # Example: Custom KD-Tree Neighborhood Graph
///
/// ```rust
/// use inlier::samplers::NeighborhoodGraph;
/// use inlier::types::DataMatrix;
///
/// struct KDTreeNeighborhoodGraph {
///     neighbors: Vec<Vec<usize>>,
///     k_neighbors: usize,
/// }
///
/// impl KDTreeNeighborhoodGraph {
///     fn new(k_neighbors: usize) -> Self {
///         Self {
///             neighbors: Vec::new(),
///             k_neighbors,
///         }
///     }
///
///     fn build_kdtree(&mut self, data: &DataMatrix) {
///         // Build KD-tree and find k-nearest neighbors for each point
///         // (Implementation would use a KD-tree library like kdtree or rstar)
///         let n = data.n_points();
///         self.neighbors.clear();
///         self.neighbors.resize(n, Vec::new());
///
///         for i in 0..n {
///             // Find k nearest neighbors (simplified)
///             for j in 0..n {
///                 if i != j && self.neighbors[i].len() < self.k_neighbors {
///                     self.neighbors[i].push(j);
///                 }
///             }
///         }
///     }
/// }
///
/// impl NeighborhoodGraph for KDTreeNeighborhoodGraph {
///     fn neighbors(&self, index: usize) -> &[usize] {
///         if index < self.neighbors.len() {
///             &self.neighbors[index]
///         } else {
///             &[]
///         }
///     }
///
///     fn initialize(&mut self, data: &DataMatrix) {
///         self.build_kdtree(data);
///     }
/// }
/// ```
pub trait NeighborhoodGraph {
    /// Return the neighbor indices of a given point.
    ///
    /// # Arguments
    /// * `index` - Index of the point in the data matrix
    ///
    /// # Returns
    /// A slice of neighbor point indices. Empty if the point has no neighbors.
    fn neighbors(&self, index: usize) -> &[usize];

    /// Initialize the neighborhood graph with data.
    ///
    /// This method should build the internal data structures (e.g., grid,
    /// KD-tree, or nearest neighbor index) based on the provided data.
    ///
    /// # Arguments
    /// * `data` - The data matrix to build the graph from
    fn initialize(&mut self, data: &DataMatrix);
}

/// Grid-based neighborhood graph that partitions points into spatial cells.
///
/// Points in the same cell are considered neighbors. This is a simplified
/// 2D implementation for image coordinates.
pub struct GridNeighborhoodGraph {
    /// Grid cells: maps cell index to point indices
    grid: std::collections::HashMap<usize, Vec<usize>>,
    /// Cell index for each point
    point_to_cell: Vec<usize>,
    /// Cell size along x and y axes
    cell_size_x: f64,
    cell_size_y: f64,
    /// Number of cells along each axis
    cells_per_axis: usize,
    /// Empty vector for points with no neighbors
    empty: Vec<usize>,
}

impl GridNeighborhoodGraph {
    /// Create a new grid neighborhood graph.
    ///
    /// `cell_size_x` and `cell_size_y` are the sizes of each cell.
    /// `cells_per_axis` is the number of cells along each axis.
    pub fn new(cell_size_x: f64, cell_size_y: f64, cells_per_axis: usize) -> Self {
        Self {
            grid: std::collections::HashMap::new(),
            point_to_cell: Vec::new(),
            cell_size_x,
            cell_size_y,
            cells_per_axis,
            empty: Vec::new(),
        }
    }

    /// Initialize the grid with data points.
    ///
    /// Assumes data has at least 2 columns (x, y coordinates).
    pub fn initialize(&mut self, data: &DataMatrix) -> bool {
        if data.n_dims() < 2 {
            return false;
        }

        self.grid.clear();
        self.point_to_cell = vec![0; data.n_points()];

        for row in 0..data.n_points() {
            let x = data.get(row, 0);
            let y = data.get(row, 1);

            if x < 0.0 || y < 0.0 {
                continue; // Skip negative coordinates
            }

            // Compute cell indices
            let cell_x = (x / self.cell_size_x).floor() as usize;
            let cell_y = (y / self.cell_size_y).floor() as usize;

            // Clamp to valid range
            let cell_x = cell_x.min(self.cells_per_axis.saturating_sub(1));
            let cell_y = cell_y.min(self.cells_per_axis.saturating_sub(1));

            // Compute linear cell index
            let cell_idx = cell_y * self.cells_per_axis + cell_x;

            // Add point to cell
            self.grid.entry(cell_idx).or_default().push(row);
            self.point_to_cell[row] = cell_idx;
        }

        !self.grid.is_empty()
    }
}

impl NeighborhoodGraph for GridNeighborhoodGraph {
    fn neighbors(&self, index: usize) -> &[usize] {
        if index >= self.point_to_cell.len() {
            return &self.empty;
        }

        let cell_idx = self.point_to_cell[index];
        self.grid
            .get(&cell_idx)
            .map(|v| v.as_slice())
            .unwrap_or(&self.empty)
    }

    fn initialize(&mut self, data: &DataMatrix) {
        // GridNeighborhoodGraph::initialize is already implemented
        let _ = GridNeighborhoodGraph::initialize(self, data);
    }
}

/// KD-tree-based neighborhood graph using exact k-nearest-neighbor search.
///
/// Backed by the pure-Rust [`kiddo`] crate, this computes the exact `k` nearest
/// neighbors of every point. The spatial dimension `D` is fixed at compile time
/// (`2` for planar data, `3` for point clouds), matching the KD-tree idiom used
/// elsewhere in the crate. It is a drop-in `NeighborhoodGraph` for NAPSAC-style
/// samplers and compiles cleanly to `wasm32` (no C/C++ toolchain required).
///
/// # Example
/// ```
/// use inlier::samplers::{KdTreeNeighborhoodGraph, NeighborhoodGraph};
/// use inlier::types::DataMatrix;
///
/// let mut graph = KdTreeNeighborhoodGraph::<2>::new(10); // 10 neighbors, 2D data
/// let data = DataMatrix::zeros(100, 2);
/// graph.initialize(&data);
/// let neighbors = graph.neighbors(0);
/// ```
pub struct KdTreeNeighborhoodGraph<const D: usize> {
    /// Neighbors for each point: `neighbors[i]` contains the neighbor indices for point `i`
    neighbors: Vec<Vec<usize>>,
    /// Number of neighbors to find for each point
    k_neighbors: usize,
    /// Empty vector for points with no neighbors
    empty: Vec<usize>,
}

impl<const D: usize> KdTreeNeighborhoodGraph<D> {
    /// Create a new KD-tree neighborhood graph.
    ///
    /// # Arguments
    /// * `k_neighbors` - Number of nearest neighbors to find for each point
    pub fn new(k_neighbors: usize) -> Self {
        Self {
            neighbors: Vec::new(),
            k_neighbors,
            empty: Vec::new(),
        }
    }

    /// Build the neighborhood graph from data points.
    fn build_neighbors(&mut self, data: &DataMatrix) {
        let n = data.n_points();
        self.neighbors.clear();
        self.neighbors.resize(n, Vec::new());

        if n == 0 || self.k_neighbors == 0 {
            return;
        }
        debug_assert!(
            data.n_dims() >= D,
            "KdTreeNeighborhoodGraph::<{D}> requires data with at least {D} dimensions, got {}",
            data.n_dims()
        );

        // Build a KD-tree over all points, keyed by point index.
        let mut tree: KdTree<f64, D> = KdTree::with_capacity(n);
        for i in 0..n {
            let mut point = [0.0f64; D];
            for (j, coord) in point.iter_mut().enumerate() {
                *coord = data.get(i, j);
            }
            tree.add(&point, i as u64);
        }

        // For each point, query the k+1 nearest (the first is the point itself).
        let k = (self.k_neighbors + 1).min(n);
        for i in 0..n {
            let mut query = [0.0f64; D];
            for (j, coord) in query.iter_mut().enumerate() {
                *coord = data.get(i, j);
            }

            for nn in tree.nearest_n::<SquaredEuclidean>(&query, k) {
                let idx = nn.item as usize;
                if idx != i {
                    self.neighbors[i].push(idx);
                    if self.neighbors[i].len() >= self.k_neighbors {
                        break;
                    }
                }
            }
        }
    }
}

impl<const D: usize> NeighborhoodGraph for KdTreeNeighborhoodGraph<D> {
    fn neighbors(&self, index: usize) -> &[usize] {
        if index >= self.neighbors.len() {
            return &self.empty;
        }
        &self.neighbors[index]
    }

    fn initialize(&mut self, data: &DataMatrix) {
        self.build_neighbors(data);
    }
}

/// Dummy neighborhood graph for testing.
///
/// This is a simple implementation that considers points within a window
/// as neighbors. Useful for testing NAPSAC-like samplers.
pub struct DummyNeighborhood {
    /// For each point i, store a small set of neighbors around it.
    neighbors: Vec<Vec<usize>>,
}

impl DummyNeighborhood {
    pub fn new(num_points: usize, window: usize) -> Self {
        let mut neighbors = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let start = i.saturating_sub(window);
            let end = (i + window).min(num_points - 1);
            neighbors.push((start..=end).collect());
        }
        Self { neighbors }
    }
}

impl NeighborhoodGraph for DummyNeighborhood {
    fn neighbors(&self, index: usize) -> &[usize] {
        &self.neighbors[index]
    }

    fn initialize(&mut self, _data: &DataMatrix) {
        // Dummy neighborhood is pre-initialized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kdtree_neighbors_are_exact_and_exclude_self() {
        // Five collinear points on the x-axis at x = 0,1,2,3,4.
        let coords = [0.0, 1.0, 2.0, 3.0, 4.0];
        let mut data = DataMatrix::zeros(coords.len(), 2);
        for (i, &x) in coords.iter().enumerate() {
            data.set(i, 0, x);
            data.set(i, 1, 0.0);
        }

        let mut graph = KdTreeNeighborhoodGraph::<2>::new(2);
        graph.initialize(&data);

        // Interior point 2 (x=2) has the two immediate neighbors 1 and 3.
        let mut mid = graph.neighbors(2).to_vec();
        mid.sort_unstable();
        assert_eq!(mid, vec![1, 3]);
        assert!(!graph.neighbors(2).contains(&2), "must exclude self");

        // Endpoint 0 (x=0) has the two nearest: 1 then 2.
        let mut end = graph.neighbors(0).to_vec();
        end.sort_unstable();
        assert_eq!(end, vec![1, 2]);
        assert!(graph.neighbors(0).iter().all(|&n| n != 0));
    }

    #[test]
    fn kdtree_out_of_range_index_is_empty() {
        let graph = KdTreeNeighborhoodGraph::<3>::new(4);
        assert!(graph.neighbors(999).is_empty());
    }
}
