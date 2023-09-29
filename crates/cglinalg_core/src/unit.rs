use cglinalg_numeric::{
    SimdScalarFloat,
};
use crate::normed::{
    Normed,
};

use core::fmt;
use core::ops;


/// A type that represents unit normalized values. 
///
/// This type enforces the requirement that values have a unit norm. This 
/// is useful when one needs to know the direction a vector is pointing for 
/// operations like constructing rotations. The unit type statically enforces the 
/// requirement that an input argument be normalized. This reduces the chance of
/// errors from passing an unnormalized vector or a zero vector into a calculation
/// involving unit vectors.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Unit<T> {
    value: T,
}

impl<T> Unit<T> {
    /// Unwraps the underlying value.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let vector = Vector3::new(3_f64, 5_f64, 7_f64);
    /// let expected = Vector3::new(
    ///     3_f64 / f64::sqrt(83_f64), 
    ///     5_f64 / f64::sqrt(83_f64), 
    ///     7_f64 / f64::sqrt(83_f64),
    /// );
    /// let result = Unit::from_value(vector).into_inner();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Wraps an object into a unit type, assuming that it is normalized without
    /// checking.
    #[inline]
    pub const fn from_value_unchecked(value: T) -> Self {
        Self { value }
    }
}

impl<T> AsRef<T> for Unit<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<T> ops::Deref for Unit<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { 
            &*(self as *const Unit<T> as *const T) 
        }
    }
}

impl<T> fmt::Display for Unit<T> 
where
    T: fmt::Display
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.value.fmt(formatter)
    }
}

impl<T> Unit<T> 
where 
    T: Normed
{
    /// Construct a new unit value, normalizing the input value.
    #[inline]
    pub fn from_value(value: T) -> Self {
        Self::from_value_with_norm(value).0
    }

    /// Construct a new normalized unit value and return its original norm.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Normed,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// #
    /// let vector: Vector3<f64> = Vector3::new(0_f64, 2_f64, 0_f64);
    /// let (wrapped, norm) = Unit::from_value_with_norm(vector);
    /// let unit_vector: &Vector3<f64> = &wrapped;
    /// 
    /// assert_eq!(norm, 2_f64);
    /// assert_eq!(unit_vector.norm_squared(), 1_f64, "unit_vector = {}", unit_vector);
    /// assert_eq!(unit_vector.norm(), 1_f64, "unit_vector = {}", unit_vector);
    /// ```
    #[inline]
    pub fn from_value_with_norm(value: T) -> (Self, T::Output) {
        let norm = value.norm();
        let normalized_value = value.normalize();
        let unit = Unit {
            value: normalized_value,
        };
        
        (unit, norm)
    }

    /// Construct a new normalized unit value and return its original norm, 
    /// provided that its norm is larger than `threshold`. 
    ///
    /// The argument `threshold` argument exists to check for vectors that may be
    /// very close to zero length.
    ///
    /// # Example
    ///
    /// Here is an example where the function returns `None` because the vector 
    /// norm is too small.
    /// ```
    /// # use cglinalg_core::{
    /// #     Normed,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// #
    /// let vector: Vector3<f64> = Vector3::new(0_f64, 1e-20_f64, 0_f64);
    /// let threshold = 1e-10_f64;
    /// let result = Unit::try_from_value_with_norm(vector, threshold);
    /// 
    /// assert!(result.is_none());
    /// ```
    #[inline]
    pub fn try_from_value_with_norm(value: T, threshold: T::Output) -> Option<(Self, T::Output)>
    where 
        T::Output: SimdScalarFloat, 
    {
        let norm_squared = value.norm_squared();

        if norm_squared > threshold * threshold {
            let norm = norm_squared.sqrt();
            let normalized_value = value.normalize();
            let unit = Unit {
                value: normalized_value,
            };

            Some((unit, norm))
        } else {
            None
        }
    }

    /// Construct a new normalized unit value, provided that its norm is 
    /// larger than `threshold`. 
    ///
    /// The argument `threshold` argument exists to check for vectors that may be
    /// very close to zero length.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Normed,
    /// #     Unit,
    /// #     Quaternion,
    /// # };
    /// # use core::f64;
    /// #
    /// let quaternion = Quaternion::new(0_f64, 1_f64, 1_f64, 1_f64);
    /// let expected = Some(Unit::from_value(quaternion / f64::sqrt(3_f64)));
    /// let result = Unit::try_from_value(quaternion, 0_f64);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn try_from_value(value: T, threshold: T::Output) -> Option<Self>
    where
        T::Output: SimdScalarFloat,
    {
        Self::try_from_value_with_norm(value, threshold).map(|(unit, _)| unit)
    }
}

impl<T> approx::AbsDiffEq for Unit<T>
where
    T: approx::AbsDiffEq
{
    type Epsilon = <T as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        T::abs_diff_eq(&self.value, &other.value, epsilon)
    }
}

impl<T> approx::RelativeEq for Unit<T>
where 
    T: approx::RelativeEq
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        T::relative_eq(&self.value, &other.value, epsilon, max_relative)
    }
}

impl<T> approx::UlpsEq for Unit<T>
where 
    T: approx::UlpsEq
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        T::ulps_eq(&self.value, &other.value, epsilon, max_ulps)
    }
}

