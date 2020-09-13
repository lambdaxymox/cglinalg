


/// A trait for implementing two-dimensional affine transformations.
pub trait AffineTransformation2D<V> where Self: Sized {
    /// The result of applying an affine transformation. This allows use to handle vectors, points,
    /// and pointers to them interchangeably.
    type Applied;

    /// The identity transformation for this type.
    fn identity() -> Self;

    /// Compute the inverse of an affine transformation.
    fn inverse(&self) -> Option<Self>;

    /// Apply the affine transformation to the input.
    fn apply(&self, point: V) -> Self::Applied;

    /// Apply the inverse of the affine transformation to the input.
    fn apply_inverse(&self, point: V) -> Option<Self::Applied>;
}

/// A trait for implementing three-dimensional affine transformations.
pub trait AffineTransformation3D<V> where Self: Sized {
    /// The result of applying an affine transformation. This allows use to handle vectors, points,
    /// and pointers to them interchangeably.
    type Applied;

    /// The identity transformation for this type.
    fn identity() -> Self;
    
    /// Compute the inverse of an affine transformation.
    fn inverse(&self) -> Option<Self>;
    
    /// Apply the affine transformation to the input.
    fn apply(&self, point: V) -> Self::Applied;
    
    /// Apply the inverse of the affine transformation to the input.
    fn apply_inverse(&self, point: V) -> Option<Self::Applied>;
}

