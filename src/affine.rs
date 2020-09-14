


/// A trait for implementing two-dimensional affine transformations.
pub trait AffineTransformation2D<P, V> where Self: Sized {
    /// The output associated type for points. This allows us to use both pointers and values
    /// on the inputs.
    type OutPoint;
    /// The output associated type for vectors. This allows us to use both pointers and values
    /// on the inputs.
    type OutVector;

    /// The identity transformation for this type.
    fn identity() -> Self;

    /// Compute the inverse of an affine transformation.
    fn inverse(&self) -> Option<Self>;

    /// Apply the affine transformation to a vector.
    fn apply_vector(&self, vector: V) -> Self::OutVector;

    /// Apply the affine transformation to a point.
    fn apply_point(&self, point: P) -> Self::OutPoint;

    /// Apply the inverse of the affine transformation to a vector.
    fn apply_inverse_vector(&self, vector: V) -> Option<Self::OutVector> {
        self.inverse()
            .and_then(|inverse| Some(inverse.apply_vector(vector)))
    }
}

/// A trait for implementing three-dimensional affine transformations.
pub trait AffineTransformation3D<P, V> where Self: Sized {
    /// The output associated type for points. This allows us to use both pointers and values
    /// on the inputs.
    type OutPoint;
    /// The output associated type for vectors. This allows us to use both pointers and values
    /// on the inputs.
    type OutVector;

    /// The identity transformation for this type.
    fn identity() -> Self;

    /// Compute the inverse of an affine transformation.
    fn inverse(&self) -> Option<Self>;

    /// Apply the affine transformation to a vector.
    fn apply_vector(&self, vector: V) -> Self::OutVector;

    /// Apply the affine transformation to a point.
    fn apply_point(&self, point: P) -> Self::OutPoint;

    /// Apply the inverse of the affine transformation to a vector.
    fn apply_inverse_vector(&self, vector: V) -> Option<Self::OutVector> {
        self.inverse()
            .and_then(|inverse| Some(inverse.apply_vector(vector)))
    }
}

