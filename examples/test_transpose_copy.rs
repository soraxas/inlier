use nalgebra::DMatrix;

fn main() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let matrix = DMatrix::from_column_slice(3, 2, &data);
    println!("Original matrix pointer: {:p}", matrix.as_ptr());

    let transposed = matrix.transpose();
    println!("Transposed pointer:      {:p}", transposed.as_ptr());

    if matrix.as_ptr() == transposed.as_ptr() {
        println!("\n✓ Zero-copy transpose (view)");
    } else {
        println!("\n✗ Transpose allocated new memory (copy)");
    }
}
