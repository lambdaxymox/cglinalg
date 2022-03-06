use crate::common::scalar::{
    ScalarFloat,   
};


/// A type with this trait acts as a vector with a notion of magnitude (length)
/// in a Euclidean vector space.
///
/// Examples of types that can be made into Euclidean normed spaces include 
/// vectors, quaternions, complex numbers, points, and scalar numbers. In the 
/// scalar case, the Euclidean magnitude is the absolute value of the scalar.
///
/// # Examples
///
/// ```
/// # use cglinalg::{
/// #     Vector4,
/// #     Magnitude,  
/// # };
/// #
/// // The magnitude of the vector.
/// let vector = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
/// assert_eq!(vector.magnitude_squared(), 30_f64);
/// assert_eq!(vector.magnitude(), 30_f64.sqrt());
/// 
/// // Nomalizing a vector.
/// let vector = Vector4::new(1_f64, 1_f64, 1_f64, 1_f64);
/// let expected = vector / 2_f64;
/// let result = vector.normalize();
/// assert_eq!(result, expected);
///
/// // Normalize a vector to a specific magnitude.
/// let magnitude = 5_f64;
/// let expected = (5_f64 / 2_f64) * vector;
/// let result = vector.normalize_to(magnitude);
/// assert_eq!(result, expected);
///
/// // Normalize a vector whose magnitude is close to zero.
/// let vector = Vector4::new(1e-11_f64, 1e-11_f64, 1e-11_f64, 1e-11_f64);
/// let threshold = 1e-10;
/// assert!(vector.try_normalize(threshold).is_none());
///
/// // The Euclidean distance between two vectors.
/// let vector1 = Vector4::new(1_f64, 1_f64, 1_f64, 1_f64);
/// let vector2 = Vector4::new(2_f64, 2_f64, 2_f64, 2_f64);
/// assert_eq!(vector1.distance_squared(&vector2), 4_f64);
/// assert_eq!(vector1.distance(&vector2), 2_f64);
/// ```
pub trait Magnitude 
where 
    Self: Sized
{
    type Output: ScalarFloat;

    /// Compute the Euclidean squared magnitude of a vector.
    fn magnitude_squared(&self) -> Self::Output;

    /// Compute the Euclidean magnitude of a vector.
    fn magnitude(&self) -> Self::Output;

    /// Normalize a vector to a unit vector.
    fn normalize(&self) -> Self;

    /// Normalize a vector to a specified magnitude.
    fn normalize_to(&self, magnitude: Self::Output) -> Self;

    /// Attempt to normalize a vector, but give up if the norm
    /// is too small.
    ///
    /// If the magnitude of the vector is smaller than the threshold
    /// `threshold`, the function returns `None`.
    fn try_normalize(&self, threshold: Self::Output) -> Option<Self>;

    /// Compute the squared Euclidean distance between two vectors.
    fn distance_squared(&self, other: &Self) -> Self::Output;

    /// Compute the Euclidean distance between two vectors.
    fn distance(&self, other: &Self) -> Self::Output;
}

