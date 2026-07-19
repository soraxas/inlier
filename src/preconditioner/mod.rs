use crate::types::DataMatrix;

/// Optional normalization step applied before running RANSAC.
pub trait Preconditioner<M> {
    type Normalization;

    /// Normalize the input data, returning the normalized data and any state
    /// needed to reverse the transformation.
    fn normalize(&self, data: &DataMatrix) -> (DataMatrix, Self::Normalization);

    /// Map the model back to the original coordinate system using the stored normalization.
    fn denormalize(&self, model: &M, norm: &Self::Normalization) -> M;
}

/// Identity preconditioner; leaves data and model unchanged.
pub struct IdentityPreconditioner;

impl<M> Preconditioner<M> for IdentityPreconditioner
where
    M: Clone,
{
    type Normalization = ();

    fn normalize(&self, data: &DataMatrix) -> (DataMatrix, Self::Normalization) {
        (data.clone(), ())
    }

    fn denormalize(&self, model: &M, _norm: &Self::Normalization) -> M {
        (*model).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Line;

    #[test]
    fn identity_preconditioner_is_exactly_idempotent() {
        let data = DataMatrix::from_row_slice(2, 2, &[1.0, -2.0, 3.5, 4.25]);
        let model = Line::new(1.0, -2.0, 3.0);
        let preconditioner = IdentityPreconditioner;

        let (normalized, normalization) =
            <IdentityPreconditioner as Preconditioner<Line>>::normalize(&preconditioner, &data);
        let (normalized_twice, _) = <IdentityPreconditioner as Preconditioner<Line>>::normalize(
            &preconditioner,
            &normalized,
        );
        let restored = <IdentityPreconditioner as Preconditioner<Line>>::denormalize(
            &preconditioner,
            &model,
            &normalization,
        );

        for matrix in [&normalized, &normalized_twice] {
            assert_eq!(matrix.n_points(), data.n_points());
            assert_eq!(matrix.n_dims(), data.n_dims());
            for point in 0..data.n_points() {
                for dimension in 0..data.n_dims() {
                    assert_eq!(matrix.get(point, dimension), data.get(point, dimension));
                }
            }
        }
        assert_eq!(restored.params, model.params);
    }
}
