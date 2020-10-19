use crate::scalar::{
    ScalarFloat,   
};
use num_traits::{
    Float,
};


/// This trait enables one to assign lengths to vectors.
pub trait Magnitude where Self: Sized {
    type Output: ScalarFloat;

    /// Compute the norm (length) of a vector.
    fn magnitude(&self) -> Self::Output;

    /// Compute the squared length of a vector.
    fn magnitude_squared(&self) -> Self::Output;

    /// Convert a vector into a unit vector.
    fn normalize(&self) -> Self;

    /// Normalize a vector with a specified magnitude.
    fn normalize_to(&self, magnitude: Self::Output) -> Self;

    /// Attempt to normalize a vector, but give up if the norm
    /// is too small.
    fn try_normalize(&self, threshold: Self::Output) -> Option<Self>;

    /// Compute the squared Eucliean distance between two vectors.
    fn distance_squared(&self, other: &Self) -> Self::Output;

    /// Compute the Euclidean distance between two vectors.
    fn distance(&self, other: &Self) -> Self::Output {
        self.distance_squared(other).sqrt()
    }
}

