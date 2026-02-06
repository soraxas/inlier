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
