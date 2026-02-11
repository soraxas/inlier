use nalgebra::DMatrix;

fn main() {
    // NumPy: [[1,2,3], [4,5,6]] stored as [1,2,3,4,5,6]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    println!("Memory: {data:?}\n");

    // Create 3×2 column-major (what we do now)
    let matrix = DMatrix::from_column_slice(3, 2, &data);
    println!("3×2 matrix (dims × points):");
    println!("{matrix}");

    // Access points as COLUMNS
    println!("Accessing points:");
    println!("  Point 0 (column 0): {:?}", matrix.column(0).as_slice());
    println!("  Point 1 (column 1): {:?}", matrix.column(1).as_slice());

    println!("\n✓ Points are stored correctly as COLUMNS!");
    println!("  p0 = [1,2,3] ✓");
    println!("  p1 = [4,5,6] ✓");
}
