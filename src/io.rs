//! I/O utilities for loading point cloud data

#[cfg(feature = "io")]
use crate::types::DataMatrix;
#[cfg(feature = "io")]
use ply_rs::{parser::Parser, ply::Property};
#[cfg(feature = "io")]
use std::fs::File;
#[cfg(feature = "io")]
use std::io::BufReader;

#[cfg(feature = "io")]
#[derive(Debug, Clone)]
struct Vertex {
    x: f32,
    y: f32,
    z: f32,
}

#[cfg(feature = "io")]
impl ply_rs::ply::PropertyAccess for Vertex {
    fn new() -> Self {
        Vertex {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    fn set_property(&mut self, key: String, property: Property) {
        match (key.as_ref(), property) {
            ("x", Property::Float(v)) => self.x = v,
            ("y", Property::Float(v)) => self.y = v,
            ("z", Property::Float(v)) => self.z = v,
            ("x", Property::Double(v)) => self.x = v as f32,
            ("y", Property::Double(v)) => self.y = v as f32,
            ("z", Property::Double(v)) => self.z = v as f32,
            _ => {}
        }
    }
}

/// Load point cloud from PLY file
///
/// Returns a DataMatrix with points stored internally as N_points x 3 (point per row).
#[cfg(feature = "io")]
pub fn load_ply(path: &str) -> Result<DataMatrix, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut buf_reader = BufReader::new(file);

    let parser = Parser::<Vertex>::new();
    let ply = parser.read_ply(&mut buf_reader)?;

    let vertices = ply
        .payload
        .get("vertex")
        .ok_or("No vertex element in PLY file")?;

    let n_points = vertices.len();
    if n_points == 0 {
        return Err("No points in PLY file".into());
    }

    // DataMatrix::from_row_slice expects: rows=n_points, cols=3 (dims per point)
    // data = [x0, y0, z0, x1, y1, z1, ...]
    let mut data = Vec::with_capacity(n_points * 3);

    for vertex in vertices.iter() {
        data.push(vertex.x as f64);
        data.push(vertex.y as f64);
        data.push(vertex.z as f64);
    }

    Ok(DataMatrix::from_row_slice(n_points, 3, &data))
}

#[cfg(all(test, feature = "io"))]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_load_ply() {
        let result = load_ply("tests/data/sample_src.ply");
        assert!(result.is_ok());

        let points = result.unwrap();
        assert_eq!(points.n_dims(), 3);
        assert_eq!(points.n_points(), 5);

        assert_relative_eq!(points.get(0, 0), 0.0, epsilon = 1e-6);
        assert_relative_eq!(points.get(0, 1), 0.3402897714, epsilon = 1e-6);
        assert_relative_eq!(points.get(0, 2), 0.1251951159, epsilon = 1e-6);
        assert_relative_eq!(points.get(4, 0), -1.25, epsilon = 1e-6);
        assert_relative_eq!(points.get(4, 1), 0.5, epsilon = 1e-6);
        assert_relative_eq!(points.get(4, 2), 2.75, epsilon = 1e-6);
    }

    #[test]
    fn test_load_ply_dst() {
        let result = load_ply("tests/data/sample_dst.ply");
        assert!(result.is_ok());

        let points = result.unwrap();
        assert_eq!(points.n_dims(), 3);
        assert_eq!(points.n_points(), 5);
        assert_relative_eq!(points.get(0, 0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(points.get(0, 1), -2.0, epsilon = 1e-6);
        assert_relative_eq!(points.get(0, 2), 0.5, epsilon = 1e-6);
        assert_relative_eq!(points.get(4, 0), -0.25, epsilon = 1e-6);
        assert_relative_eq!(points.get(4, 1), -1.5, epsilon = 1e-6);
        assert_relative_eq!(points.get(4, 2), 3.25, epsilon = 1e-6);
    }

    #[test]
    fn load_ply_rejects_empty_vertex_payload() {
        let result = load_ply("tests/data/empty_vertices.ply");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No points"));
    }

    #[test]
    fn load_ply_rejects_missing_file() {
        let result = load_ply("tests/data/does_not_exist.ply");
        assert!(result.is_err());
    }
}
