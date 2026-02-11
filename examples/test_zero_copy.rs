//! Test if we can do zero-copy transpose from NumPy layout

use nalgebra::DMatrix;

fn main() {
    // NumPy N×3 row-major: [p0.x, p0.y, p0.z, p1.x, p1.y, p1.z]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    println!("Original data (N×3 row-major):");
    println!("  {data:?}\n");

    // Current approach: from_row_slice then transpose (COPY!)
    let m1 = DMatrix::from_row_slice(2, 3, &data);
    println!("Step 1: from_row_slice(2, 3):");
    println!("{m1}");

    let transposed = m1.transpose();
    println!("Step 2: transpose() [ALLOCATES NEW MATRIX]:");
    println!("{transposed}\n");

    // Zero-copy approach: from_column_slice
    let m2 = DMatrix::from_column_slice(3, 2, &data);
    println!("Zero-copy: from_column_slice(3, 2) [NO ALLOCATION]:");
    println!("{m2}\n");

    println!("Are they equal? {}", transposed == m2);

    if transposed == m2 {
        println!("\n✓ ZERO-COPY IS POSSIBLE!");
        println!("  We can replace: from_row_slice(N, 3).transpose()");
        println!("  With:           from_column_slice(3, N)");
    }
}
