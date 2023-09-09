use cglinalg_numeric::{
    SimdScalarSigned,
    SimdScalarFloat,   
};


/// A type with this trait acts as a vector with a notion of norm (magnitude, length).
///
/// Examples of types that can be made into Euclidean normed spaces include 
/// vectors, quaternions, complex numbers, points, and scalar numbers. In the 
/// scalar case, the Euclidean norm is the absolute value of the scalar.
///
/// # Examples
///
/// ```
/// # use cglinalg_core::{
/// #     Vector4,
/// #     Normed,  
/// # };
/// #
/// // The norm of the vector.
/// let vector = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
/// assert_eq!(vector.norm_squared(), 30_f64);
/// assert_eq!(vector.norm(), 30_f64.sqrt());
/// 
/// // Nomalizing a vector.
/// let vector = Vector4::new(1_f64, 1_f64, 1_f64, 1_f64);
/// let expected = vector / 2_f64;
/// let result = vector.normalize();
/// assert_eq!(result, expected);
///
/// // Scale a vector to a specific norm.
/// let scale = 5_f64;
/// let expected = vector * 5_f64;
/// let result = vector.scale(scale);
/// assert_eq!(result, expected);
///
/// // Normalize a vector whose norm is close to zero.
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
pub trait Normed 
where 
    Self: Sized
{
    type Output: SimdScalarFloat;

    /// Compute the squared **L2** norm of a vector.
    /// 
    /// # Examples
    /// 
    /// Computing the squared **L2** norm of a vector.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// #
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// 
    /// assert_eq!(vector.norm_squared(), 14_f64);
    /// ```
    /// 
    /// Computing the squared **Frobenius** norm of a matrix.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Radians,
    /// #     Unit,
    /// #     Normed,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let matrix = Matrix3x3::from_axis_angle(&axis, angle);
    /// 
    /// assert_eq!(matrix.norm_squared(), 3_f64);
    /// ```
    fn norm_squared(&self) -> Self::Output;

    /// Compute the **L2** norm of a vector.
    /// 
    /// # Examples
    /// 
    /// Computing the **L2** norm of a vector.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// #
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// 
    /// assert_eq!(vector.norm(), f64::sqrt(14_f64));
    /// ```
    /// 
    /// Computing the **Frobenius** norm of a matrix.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Radians,
    /// #     Unit,
    /// #     Normed,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let matrix = Matrix3x3::from_axis_angle(&axis, angle);
    /// 
    /// assert_eq!(matrix.norm(), f64::sqrt(3_f64));
    /// ```
    fn norm(&self) -> Self::Output;

    /// Scale a vector by a factor `scale`.
    /// 
    /// This function multiples each element of a vector by `scale.`
    /// 
    /// # Examples
    /// 
    /// Scaling a vector.
    /// ```
    /// # use cglinalg_core::{
    /// #      Vector3,
    /// #      Normed,
    /// # };
    /// #
    /// let scale = f64::sqrt(2_f64);
    /// let norm = f64::sqrt(2_f64);
    /// let vector = Vector3::new(0_f64, norm, 0_f64);
    /// let expected = Vector3::new(0_f64, scale * norm, 0_f64);
    /// let result = vector.scale(scale);
    /// 
    /// assert_eq!(result, expected);
    /// assert_eq!(result.norm(), norm * scale);
    /// ```
    /// 
    /// Scaling a matrix.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Normed,
    /// # };
    /// #
    /// let scale = 3_f64;
    /// let matrix = Matrix3x3::identity();
    /// let expected = Matrix3x3::new(
    ///     3_f64, 0_f64, 0_f64,
    ///     0_f64, 3_f64, 0_f64,
    ///     0_f64, 0_f64, 3_f64
    /// );
    /// let result = matrix.scale(scale);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    fn scale(&self, scale: Self::Output) -> Self;

    /// Scale a vector mutably in place to a specified norm `norm`.
    /// 
    /// This function multiples each element of a vector by `scale.`
    /// 
    /// # Examples
    /// 
    /// Mutably scaling a vector.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// #
    /// let scale = f64::sqrt(2_f64);
    /// let norm = f64::sqrt(2_f64);
    /// let mut result = Vector3::new(0_f64, norm, 0_f64);
    /// let expected = Vector3::new(0_f64, norm * scale, 0_f64);
    /// result.scale_mut(scale);
    /// 
    /// assert_eq!(result, expected);
    /// assert_eq!(result.norm(), scale * norm);
    /// ```
    /// 
    /// Mutably scaling a matrix.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Normed,
    /// # };
    /// #
    /// let scale = 3_f64;
    /// let expected = Matrix3x3::new(
    ///     3_f64, 0_f64, 0_f64,
    ///     0_f64, 3_f64, 0_f64,
    ///     0_f64, 0_f64, 3_f64
    /// );
    /// let mut result = Matrix3x3::identity();
    /// result.scale_mut(scale);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    fn scale_mut(&mut self, scale: Self::Output);

    /// Scale a vector to by a factor `1 / scale`.
    /// 
    /// This function multiples each element of a vector by `1 / scale.`
    /// 
    /// # Examples
    /// 
    /// Unscaling a vector.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let scale = 3_f64;
    /// let norm = f64::sqrt(3_f64);
    /// let vector = Vector3::new(0_f64, 0_f64, norm);
    /// let expected = Vector3::new(0_f64, 0_f64, norm / scale);
    /// let result = vector.unscale(scale);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_eq!(result.norm(), norm / scale);
    /// ```
    /// 
    /// Unscaling a matrix.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 5_f64;
    /// let matrix = Matrix3x3::new(
    ///     1_f64,  9_f64,   74_f64, 
    ///     98_f64, 75_f64,  28_f64, 
    ///     36_f64, 100_f64, 86_f64
    /// );
    /// let expected = Matrix3x3::new(
    ///     0.2_f64,  1.8_f64,  14.8_f64, 
    ///     19.6_f64, 15.0_f64, 5.6_f64, 
    ///     7.2_f64,  20.0_f64, 17.2_f64
    /// );
    /// let result = matrix.unscale(scale);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn unscale(&self, scale: Self::Output) -> Self;

    /// Scale a vector mutably in place to a specified norm `1 / norm`.
    /// 
    /// This function multiples each element of a vector by `1 / scale.`
    /// 
    /// # Examples
    /// 
    /// Mutably unscaling a vector.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let scale = 3_f64;
    /// let norm = f64::sqrt(3_f64);
    /// let mut result = Vector3::new(0_f64, 0_f64, norm);
    /// let expected = Vector3::new(0_f64, 0_f64, norm / scale);
    /// result.unscale_mut(scale);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_eq!(result.norm(), norm / scale);
    /// ```
    /// 
    /// Mutably unscaling a matrix.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 5_f64;
    /// let mut result = Matrix3x3::new(
    ///     1_f64,  9_f64,   74_f64, 
    ///     98_f64, 75_f64,  28_f64, 
    ///     36_f64, 100_f64, 86_f64
    /// );
    /// let expected = Matrix3x3::new(
    ///     0.2_f64,  1.8_f64,  14.8_f64, 
    ///     19.6_f64, 15.0_f64, 5.6_f64, 
    ///     7.2_f64,  20.0_f64, 17.2_f64
    /// );
    /// result.unscale_mut(scale);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn unscale_mut(&mut self, scale: Self::Output);

    /// Normalize a vector to a unit vector.
    /// 
    /// # Examples
    /// 
    /// Normalizing a vector.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// #
    /// let vector = Vector3::new(1_f64, 1_f64, 1_f64);
    /// let expected = Vector3::new(
    ///     1_f64 / f64::sqrt(3_f64), 
    ///     1_f64 / f64::sqrt(3_f64), 
    ///     1_f64 / f64::sqrt(3_f64)
    /// );
    /// let result = vector.normalize();
    /// 
    /// assert_eq!(result, expected);
    /// assert_eq!(result.norm(), 1_f64);
    /// ```
    /// 
    /// Normalizing a matrix.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Radians,
    /// #     Unit,
    /// #     Normed,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(-f64::consts::FRAC_PI_3);
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let matrix = Matrix3x3::from_axis_angle(&axis, angle);
    /// 
    /// // The norm of the matrix is not the same as the determinant.
    /// assert_eq!(matrix.determinant(), 1_f64);
    /// assert_eq!(matrix.norm(), f64::sqrt(3_f64));
    /// 
    /// let normalized_matrix = matrix.normalize();
    /// let expected = 1_f64;
    /// let result = normalized_matrix.norm();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    fn normalize(&self) -> Self;

    /// Normalize a vector to a unit vector mutably in place and return its norm 
    /// prior to normalization.
    /// 
    /// # Example
    /// 
    /// Mutably normalizing a vector.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// #
    /// let mut result = Vector3::new(1_f64, 1_f64, 1_f64);
    /// let expected = Vector3::new(
    ///     1_f64 / f64::sqrt(3_f64), 
    ///     1_f64 / f64::sqrt(3_f64), 
    ///     1_f64 / f64::sqrt(3_f64)
    /// );
    /// let old_norm = result.normalize_mut();
    /// 
    /// assert_eq!(result, expected);
    /// assert_eq!(result.norm(), 1_f64);
    /// assert_eq!(old_norm, f64::sqrt(3_f64));
    /// ```
    /// 
    /// Mutably normalizing a matrix.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Radians,
    /// #     Unit,
    /// #     Normed,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(-f64::consts::FRAC_PI_3);
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let matrix = Matrix3x3::from_axis_angle(&axis, angle);
    /// 
    /// // The norm of the matrix is not the same as the determinant.
    /// assert_eq!(matrix.determinant(), 1_f64);
    /// assert_eq!(matrix.norm(), f64::sqrt(3_f64));
    /// 
    /// let mut normalized_matrix = matrix.clone();
    /// let unnormalized_norm = normalized_matrix.normalize_mut();
    /// 
    /// assert_eq!(unnormalized_norm, matrix.norm());
    /// 
    /// let expected = 1_f64;
    /// let result = normalized_matrix.norm();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    fn normalize_mut(&mut self) -> Self::Output;

    /// Attempt to normalize a vector, but give up if the norm
    /// is too small.
    ///
    /// If the norm of the vector is smaller than the threshold
    /// `threshold`, the function returns `None`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// # use core::f64;
    /// #
    /// let threshold = 1e-6;
    /// let failing_vector = Vector3::new(threshold - f64::EPSILON, 0_f64, 0_f64);
    /// let successful_vector = Vector3::new(threshold + f64::EPSILON, 0_f64, 0_f64);
    /// 
    /// assert!(failing_vector.try_normalize(threshold).is_none());
    /// assert!(successful_vector.try_normalize(threshold).is_some());
    /// ```
    fn try_normalize(&self, threshold: Self::Output) -> Option<Self>;

    /// Attempt to normalize a vector in place, but give up if the norm is too small.
    /// 
    /// If the norm of the vector is smaller than the threshold 
    /// `threshold`, the function does not mutate `self` and returns `None`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// # use core::f64;
    /// #
    /// let threshold = 1e-6;
    /// 
    /// let mut failing_vector = Vector3::new(threshold - f64::EPSILON, 0_f64, 0_f64);
    /// let old_failing_vector = failing_vector.clone();
    /// let failing_result = failing_vector.try_normalize_mut(threshold);
    /// 
    /// assert!(failing_result.is_none());
    /// // Nothing was changed because the normalization attempt failed.
    /// assert_eq!(failing_vector, old_failing_vector);
    /// 
    /// let mut successful_vector = Vector3::new(threshold + f64::EPSILON, 0_f64, 0_f64);
    /// let old_successful_vector = successful_vector.clone();
    /// let successful_result = successful_vector.try_normalize_mut(threshold);
    /// 
    /// assert!(successful_result.is_some());
    /// // The normalize attempt succeeded and mutated the old vector.
    /// assert_ne!(successful_vector, old_successful_vector);
    /// // A successful result returns the old norm before normalizing.
    /// assert_eq!(successful_result.unwrap(), old_successful_vector.norm());
    /// ```
    fn try_normalize_mut(&mut self, threshold: Self::Output) -> Option<Self::Output>;

    /// Compute the squared Euclidean distance between two vectors.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// #
    /// let vector1 = Vector3::new(0_f64, 1_f64, 0_f64);
    /// let vector2 = Vector3::new(0_f64, 5_f64, 0_f64);
    /// let expected = 16_f64;
    /// let result = vector1.distance_squared(&vector2);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    fn distance_squared(&self, other: &Self) -> Self::Output;

    /// Compute the Euclidean distance between two vectors.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// #
    /// let vector1 = Vector3::new(0_f64, 1_f64, 0_f64);
    /// let vector2 = Vector3::new(0_f64, 5_f64, 0_f64);
    /// let expected = 4_f64;
    /// let result = vector1.distance(&vector2);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    fn distance(&self, other: &Self) -> Self::Output;
}

/// A type this this trait acts as a vector norm on instances of its input types.
/// 
/// We say that a type `Self` is a norm on elements of a vector type `V` over 
/// a set of scalars `S` if each instance `norm` of the type `Self` satisfies the 
/// following properties.
/// ```text
/// (1) forall v in V. norm.norm(v) >= 0
/// (2) forall v in V. norn.norm(v) == 0 ==> v == 0
/// (3) forall v in V. forall c in S. norm.norm(v * c) == norm.norm(v) * abs(c)
/// (4) forall v1, v2 in V. norm.norm(v1 + v2) <= norm.norm(v1) + norm.norm(v2)
/// ```
pub trait Norm<V>
where
{
    type Output: SimdScalarSigned;

    /// Compute the norm of a normed vector type with respect to the norm `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Norm,
    /// #     L1Norm,
    /// # };
    /// # use core::f64;
    /// #
    /// let l1_norm = L1Norm::new();
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = 6_f64;
    /// let result = l1_norm.norm(&vector);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    fn norm(&self, rhs: &V) -> Self::Output;

    /// Compute the distance between two instances of a normed vector type
    /// with respect to the metric induced by the norm `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Norm,
    /// #     L2Norm,  
    /// # };
    /// # use core::f64;
    /// #
    /// let l2_norm = L2Norm::new();
    /// let vector1 = Vector3::new(1_f64, 1_f64, 1_f64);
    /// let vector2 = Vector3::new(0_f64, 1_f64, 0_f64);
    /// let expected = f64::sqrt(2_f64);
    /// let result = l2_norm.metric_distance(&vector1, &vector2);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    fn metric_distance(&self, lhs: &V, rhs: &V) -> Self::Output;
}

