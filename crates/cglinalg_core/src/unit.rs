use crate::normed::Normed;
use cglinalg_numeric::SimdScalarFloat;

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
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Unit,
    /// #     Vector3,
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
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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
        unsafe { &*(self as *const Unit<T> as *const T) }
    }
}

impl<T> fmt::Display for Unit<T>
where
    T: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.value.fmt(formatter)
    }
}

impl<T> Unit<T>
where
    T: Normed,
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
        let unit = Unit { value: normalized_value };

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
            let unit = Unit { value: normalized_value };

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
/*
impl<T> approx::AbsDiffEq for Unit<T>
where
    T: approx::AbsDiffEq,
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
    T: approx::RelativeEq,
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
    T: approx::UlpsEq,
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
*/
impl<T> approx_cmp::AbsDiffEq for Unit<T>
where
    T: approx_cmp::AbsDiffEq,
{
    type Tolerance = <T as approx_cmp::AbsDiffEq>::Tolerance;

    #[inline]
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> bool {
        approx_cmp::AbsDiffEq::abs_diff_eq(&self.value, &other.value, max_abs_diff)
    }
}

impl<T> approx_cmp::AbsDiffAllEq for Unit<T>
where
    T: approx_cmp::AbsDiffAllEq,
{
    type AllTolerance = <T as approx_cmp::AbsDiffAllEq>::AllTolerance;

    #[inline]
    fn abs_diff_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> bool {
        approx_cmp::AbsDiffAllEq::abs_diff_all_eq(&self.value, &other.value, max_abs_diff)
    }
}

impl<T> approx_cmp::AssertAbsDiffEq for Unit<T>
where
    T: approx_cmp::AssertAbsDiffEq,
{
    type DebugAbsDiff = <T as approx_cmp::AssertAbsDiffEq>::DebugAbsDiff;
    type DebugTolerance = <T as approx_cmp::AssertAbsDiffEq>::DebugTolerance;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        approx_cmp::AssertAbsDiffEq::debug_abs_diff(&self.value, &other.value)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        approx_cmp::AssertAbsDiffEq::debug_abs_diff_tolerance(&self.value, &other.value, max_abs_diff)
    }
}

impl<T> approx_cmp::AssertAbsDiffAllEq for Unit<T>
where
    T: approx_cmp::AssertAbsDiffAllEq,
{
    type AllDebugTolerance = <T as approx_cmp::AssertAbsDiffAllEq>::AllDebugTolerance;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        approx_cmp::AssertAbsDiffAllEq::debug_abs_diff_all_tolerance(&self.value, &other.value, max_abs_diff)
    }
}

impl<T> approx_cmp::RelativeEq for Unit<T>
where
    T: approx_cmp::RelativeEq,
{
    type Tolerance = <T as approx_cmp::RelativeEq>::Tolerance;

    #[inline]
    fn relative_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_relative: &Self::Tolerance) -> bool {
        approx_cmp::RelativeEq::relative_eq(&self.value, &other.value, max_abs_diff, max_relative)
    }
}

impl<T> approx_cmp::RelativeAllEq for Unit<T>
where
    T: approx_cmp::RelativeAllEq,
{
    type AllTolerance = <T as approx_cmp::RelativeAllEq>::AllTolerance;

    #[inline]
    fn relative_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_relative: &Self::AllTolerance) -> bool {
        approx_cmp::RelativeAllEq::relative_all_eq(&self.value, &other.value, max_abs_diff, max_relative)
    }
}

impl<T> approx_cmp::AssertRelativeEq for Unit<T>
where
    T: approx_cmp::AssertRelativeEq,
{
    type DebugAbsDiff = <T as approx_cmp::AssertRelativeEq>::DebugAbsDiff;
    type DebugTolerance = <T as approx_cmp::AssertRelativeEq>::DebugTolerance;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        approx_cmp::AssertRelativeEq::debug_abs_diff(&self.value, &other.value)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        approx_cmp::AssertRelativeEq::debug_abs_diff_tolerance(&self.value, &other.value, max_abs_diff)
    }

    #[inline]
    fn debug_relative_tolerance(&self, other: &Self, max_relative: &Self::Tolerance) -> Self::DebugTolerance {
        approx_cmp::AssertRelativeEq::debug_relative_tolerance(&self.value, &other.value, max_relative)
    }
}

impl<A> approx_cmp::AssertRelativeAllEq for Unit<A>
where
    A: approx_cmp::AssertRelativeAllEq,
{
    type AllDebugTolerance = <A as approx_cmp::AssertRelativeAllEq>::AllDebugTolerance;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        approx_cmp::AssertRelativeAllEq::debug_abs_diff_all_tolerance(&self.value, &other.value, max_abs_diff)
    }

    #[inline]
    fn debug_relative_all_tolerance(&self, other: &Self, max_relative: &Self::AllTolerance) -> Self::AllDebugTolerance {
        approx_cmp::AssertRelativeAllEq::debug_relative_all_tolerance(&self.value, &other.value, max_relative)
    }
}

impl<T> approx_cmp::UlpsEq for Unit<T>
where
    T: approx_cmp::UlpsEq,
{
    type Tolerance = <T as approx_cmp::UlpsEq>::Tolerance;
    type UlpsTolerance = <T as approx_cmp::UlpsEq>::UlpsTolerance;

    #[inline]
    fn ulps_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_ulps: &Self::UlpsTolerance) -> bool {
        approx_cmp::UlpsEq::ulps_eq(&self.value, &other.value, max_abs_diff, max_ulps)
    }
}

impl<T> approx_cmp::UlpsAllEq for Unit<T>
where
    T: approx_cmp::UlpsAllEq,
{
    type AllTolerance = <T as approx_cmp::UlpsAllEq>::AllTolerance;
    type AllUlpsTolerance = <T as approx_cmp::UlpsAllEq>::AllUlpsTolerance;

    #[inline]
    fn ulps_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_ulps: &Self::AllUlpsTolerance) -> bool {
        approx_cmp::UlpsAllEq::ulps_all_eq(&self.value, &other.value, max_abs_diff, max_ulps)
    }
}

impl<T> approx_cmp::AssertUlpsEq for Unit<T>
where
    T: approx_cmp::AssertUlpsEq,
{
    type DebugAbsDiff = <T as approx_cmp::AssertUlpsEq>::DebugAbsDiff;
    type DebugUlpsDiff = <T as approx_cmp::AssertUlpsEq>::DebugUlpsDiff;
    type DebugTolerance = <T as approx_cmp::AssertUlpsEq>::DebugTolerance;
    type DebugUlpsTolerance = <T as approx_cmp::AssertUlpsEq>::DebugUlpsTolerance;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        approx_cmp::AssertUlpsEq::debug_abs_diff(&self.value, &other.value)
    }

    #[inline]
    fn debug_ulps_diff(&self, other: &Self) -> Self::DebugUlpsDiff {
        approx_cmp::AssertUlpsEq::debug_ulps_diff(&self.value, &other.value)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        approx_cmp::AssertUlpsEq::debug_abs_diff_tolerance(&self.value, &other.value, max_abs_diff)
    }

    #[inline]
    fn debug_ulps_tolerance(&self, other: &Self, max_ulps: &Self::UlpsTolerance) -> Self::DebugUlpsTolerance {
        approx_cmp::AssertUlpsEq::debug_ulps_tolerance(&self.value, &other.value, max_ulps)
    }
}

impl<T> approx_cmp::AssertUlpsAllEq for Unit<T>
where
    T: approx_cmp::AssertUlpsAllEq,
{
    type AllDebugTolerance = <T as approx_cmp::AssertUlpsAllEq>::AllDebugTolerance;
    type AllDebugUlpsTolerance = <T as approx_cmp::AssertUlpsAllEq>::AllDebugUlpsTolerance;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        approx_cmp::AssertUlpsAllEq::debug_abs_diff_all_tolerance(&self.value, &other.value, max_abs_diff)
    }

    #[inline]
    fn debug_ulps_all_tolerance(&self, other: &Self, max_ulps: &Self::AllUlpsTolerance) -> Self::AllDebugUlpsTolerance {
        approx_cmp::AssertUlpsAllEq::debug_ulps_all_tolerance(&self.value, &other.value, max_ulps)
    }
}
