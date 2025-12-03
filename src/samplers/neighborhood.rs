//! Neighborhood graph implementations for spatial sampling strategies.

use crate::types::DataMatrix;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

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
///         let n = data.nrows();
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
        if data.ncols() < 2 {
            return false;
        }

        self.grid.clear();
        self.point_to_cell = vec![0; data.nrows()];

        for row in 0..data.nrows() {
            let x = data[(row, 0)];
            let y = data[(row, 1)];

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

/// USearch-based neighborhood graph using approximate nearest neighbor search.
///
/// This implementation uses the `usearch` crate for fast approximate nearest neighbor
/// search, which is more flexible than grid-based approaches and works well for
/// high-dimensional data or irregular point distributions.
///
/// # Example
/// ```
/// use inlier::samplers::{UsearchNeighborhoodGraph, NeighborhoodGraph};
/// use inlier::types::DataMatrix;
///
/// let mut graph = UsearchNeighborhoodGraph::new(10, 2); // 10 neighbors, 2D data
/// let data = DataMatrix::zeros(100, 2);
/// graph.initialize(&data);
/// let neighbors = graph.neighbors(0);
/// ```
pub struct UsearchNeighborhoodGraph {
    /// USearch index for nearest neighbor queries
    index: Option<Index>,
    /// Neighbors for each point: `neighbors[i]` contains the neighbor indices for point `i`
    neighbors: Vec<Vec<usize>>,
    /// Number of neighbors to find for each point
    k_neighbors: usize,
    /// Number of dimensions (extracted from data)
    dimensions: usize,
    /// Empty vector for points with no neighbors
    empty: Vec<usize>,
}

impl UsearchNeighborhoodGraph {
    /// Create a new USearch neighborhood graph.
    ///
    /// # Arguments
    /// * `k_neighbors` - Number of nearest neighbors to find for each point
    /// * `dimensions` - Number of dimensions in the data (will be set from data if 0)
    pub fn new(k_neighbors: usize, dimensions: usize) -> Self {
        Self {
            index: None,
            neighbors: Vec::new(),
            k_neighbors,
            dimensions,
            empty: Vec::new(),
        }
    }

    /// Build the neighborhood graph from data points.
    fn build_neighbors(&mut self, data: &DataMatrix) -> bool {
        let n = data.nrows();
        let dims = if self.dimensions > 0 {
            self.dimensions
        } else {
            data.ncols()
        };

        if n == 0 || dims == 0 {
            return false;
        }

        // Create USearch index
        let options = IndexOptions {
            dimensions: dims,
            // Squared L2 distance (faster)
            metric: MetricKind::L2sq,
            quantization: ScalarKind::F32,
            ..Default::default()
        };

        let index = match Index::new(&options) {
            Ok(idx) => idx,
            Err(_) => return false,
        };

        // Reserve capacity
        if index.reserve(n).is_err() {
            return false;
        }

        // Add all points to the index
        for i in 0..n {
            let mut vector = Vec::<f32>::with_capacity(dims);
            for j in 0..dims {
                vector.push(data[(i, j)] as f32);
            }
            if index.add(i as u64, &vector).is_err() {
                return false;
            }
        }

        // Find neighbors for each point
        self.neighbors.clear();
        self.neighbors.resize(n, Vec::new());

        for i in 0..n {
            let mut query = Vec::<f32>::with_capacity(dims);
            for j in 0..dims {
                query.push(data[(i, j)] as f32);
            }

            // Search for k+1 neighbors (including the point itself)
            let k = (self.k_neighbors + 1).min(n);
            match index.search(&query, k) {
                Ok(results) => {
                    // Filter out the point itself and store neighbors
                    for (key, _distance) in results.keys.iter().zip(results.distances.iter()) {
                        let key_usize = *key as usize;
                        if key_usize != i {
                            self.neighbors[i].push(key_usize);
                            if self.neighbors[i].len() >= self.k_neighbors {
                                break;
                            }
                        }
                    }
                }
                Err(_) => {
                    // If search fails, leave neighbors empty for this point
                }
            }
        }

        self.index = Some(index);
        true
    }
}

impl NeighborhoodGraph for UsearchNeighborhoodGraph {
    fn neighbors(&self, index: usize) -> &[usize] {
        if index >= self.neighbors.len() {
            return &self.empty;
        }
        &self.neighbors[index]
    }

    fn initialize(&mut self, data: &DataMatrix) {
        if self.dimensions == 0 {
            self.dimensions = data.ncols();
        }
        let _ = self.build_neighbors(data);
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
