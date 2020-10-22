use crate::scalar::{
    ScalarFloat,   
};
use num_traits::{
    Float,
};


/// This trait enables one to assign lengths to vectors.
pub trait Magnitude where Self: Sized {
    type Output: ScalarFloat;

    /// Compute the Euclidean squared magnitude of a vector.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4,
    /// #     Magnitude,
    /// # };
    /// #
    /// let vector = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
    /// 
    /// assert_eq!(vector.magnitude_squared(), 30_f64);
    /// ```
    fn magnitude_squared(&self) -> Self::Output;

    /// Compute the Euclidean magnitude of a vector.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4,
    /// #     Magnitude,
    /// # };
    /// #
    /// let vector = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
    /// 
    /// assert_eq!(vector.magnitude(), 30_f64.sqrt());
    /// ```
    fn magnitude(&self) -> Self::Output;

    /// Normalize a vector to a unit vector.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4,
    /// #     Magnitude, 
    /// # };
    /// # use core::f64;
    /// #
    /// let vector = Vector4::new(1_f64, 1_f64, 1_f64, 1_f64);
    /// let expected = vector / 2_f64;
    /// let result = vector.normalize();
    ///
    /// assert_eq!(result, expected);
    /// ```
    fn normalize(&self) -> Self;

    /// Normalize a vector to a specified magnitude.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4,
    /// #     Magnitude, 
    /// # };
    /// #
    /// let vector = Vector4::new(1_f64, 1_f64, 1_f64, 1_f64);
    /// let magnitude = 5_f64;
    /// let expected = (5_f64 / 2_f64) * vector;
    /// let result = vector.normalize_to(magnitude);
    ///
    /// assert_eq!(result, expected);
    /// ```
    fn normalize_to(&self, magnitude: Self::Output) -> Self;

    /// Attempt to normalize a vector, but give up if the norm
    /// is too small.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4,
    /// #     Magnitude, 
    /// # };
    /// #
    /// let vector = Vector4::new(1e-11_f64, 1e-11_f64, 1e-11_f64, 1e-11_f64);
    /// let threshold = 1e-10;
    ///
    /// assert!(vector.try_normalize(threshold).is_none());
    /// ```
    fn try_normalize(&self, threshold: Self::Output) -> Option<Self>;

    /// Compute the squared Eucliean distance between two vectors.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4,
    /// #     Magnitude,
    /// # };
    /// #
    /// let vector1 = Vector4::new(1_f64, 1_f64, 1_f64, 1_f64);
    /// let vector2 = Vector4::new(2_f64, 2_f64, 2_f64, 2_f64);
    ///
    /// assert_eq!(vector1.distance_squared(&vector2), 4_f64);
    /// ```
    fn distance_squared(&self, other: &Self) -> Self::Output;

    /// Compute the Euclidean distance between two vectors.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4,
    /// #     Magnitude,
    /// # };
    /// #
    /// let vector1 = Vector4::new(1_f64, 1_f64, 1_f64, 1_f64);
    /// let vector2 = Vector4::new(2_f64, 2_f64, 2_f64, 2_f64);
    ///
    /// assert_eq!(vector1.distance(&vector2), 2_f64);
    /// ```
    fn distance(&self, other: &Self) -> Self::Output;
}

