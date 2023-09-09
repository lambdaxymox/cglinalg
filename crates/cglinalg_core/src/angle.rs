use cglinalg_numeric::{
    SimdScalar,
    SimdScalarSigned,
    SimdScalarFloat,
};

use core::f64;
use core::fmt;
use core::ops;


/// Implement trigonometry for typed angles. 
///
/// Making the units of the angles strongly typed enables us to make a careful 
/// distinction between different units of angles to prevent trigonometric 
/// errors that arise from using incorrect angular units. For example, adding
/// radians to degrees, or passing an angle in degrees to a trigonometric 
/// function when one meant to pass an angle in units of radians.
pub trait Angle 
where 
    Self: Copy + Clone,
    Self: fmt::Debug + fmt::Display,
    Self: PartialEq + PartialOrd,
    Self: num_traits::Zero,
    Self: ops::Neg<Output = Self>,
    Self: ops::Add<Self, Output = Self>,
    Self: ops::Sub<Self, Output = Self>,
    Self: ops::Mul<<Self as Angle>::Dimensionless, Output = Self>,
    Self: ops::Div<Self, Output = <Self as Angle>::Dimensionless>,
    Self: ops::Div<<Self as Angle>::Dimensionless, Output = Self>,
    Self: ops::Rem<Self, Output = Self>,
    Self: approx::AbsDiffEq<Epsilon = <Self as Angle>::Dimensionless>,
    Self: approx::RelativeEq<Epsilon = <Self as Angle>::Dimensionless>,
    Self: approx::UlpsEq<Epsilon = <Self as Angle>::Dimensionless>,
{
    type Dimensionless: SimdScalarFloat;

    /// The value of a full rotation around the unit circle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// #     Degrees,
    /// # };
    /// # use core::f64;
    /// #
    /// assert_eq!(Radians(2_f64 * f64::consts::PI), Radians::full_turn());
    /// assert_eq!(Degrees(360_f64), Degrees::full_turn());
    /// ```
    fn full_turn() -> Self;

    /// Compute the sine of a typed angle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let pi_over_4 = Radians(f64::consts::FRAC_PI_4);
    /// let expected = 1_f64 / f64::sqrt(2_f64);
    /// let result = pi_over_4.sin();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn sin(self) -> Self::Dimensionless;

    /// Compute the cosine of a typed angle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let pi_over_4 = Radians(f64::consts::FRAC_PI_4);
    /// let expected = 1_f64 / f64::sqrt(2_f64);
    /// let result = pi_over_4.cos();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn cos(self) -> Self::Dimensionless;

    /// Compute the tangent of a typed angle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let pi_over_4 = Radians(f64::consts::FRAC_PI_4);
    /// let expected = 1_f64;
    /// let result = pi_over_4.tan();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn tan(self) -> Self::Dimensionless;

    /// Compute the arcsine of a scalar value, returning a typed angle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let sin_pi_over_3 = f64::sqrt(3_f64) / 2_f64;
    /// let expected = Radians(f64::consts::FRAC_PI_3);
    /// let result = Radians::asin(sin_pi_over_3);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn asin(ratio: Self::Dimensionless) -> Self;

    /// Compute the arccosine of a scalar value, returning a typed angle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let cos_pi_over_3 = 1_f64 / 2_f64;
    /// let expected = Radians(f64::consts::FRAC_PI_3);
    /// let result = Radians::acos(cos_pi_over_3);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn acos(ratio: Self::Dimensionless) -> Self;

    /// Compute the arctangent of a scalar value, returning a typed angle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let tan_pi_over_3 = f64::sqrt(3_f64);
    /// let expected = Radians(f64::consts::FRAC_PI_3);
    /// let result = Radians::atan(tan_pi_over_3);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn atan(ratio: Self::Dimensionless) -> Self;

    /// Compute the four quadrant arctangent of two arguments, returning a 
    /// typed angle.
    /// 
    /// The return value is the arctangent of the quotient of the two input values. 
    /// That is, given inputs `x` and `y`, and an angle `theta` whose tangent 
    /// satisfies
    /// ```text
    /// tan2(theta) := y / x
    /// ```
    /// The `atan2` function is defined as
    /// ```text
    /// atan2(y, x) := atan(y / x) == theta
    /// ```
    ///
    /// The return values fall into the following value ranges.
    /// ```text
    /// x = 0 and y = 0 -> 0
    /// x >= 0          ->  arctan(y / x)       in [-pi / 2, pi / 2]
    /// y >= 0          -> (arctan(y / x) + pi) in (pi / 2, pi]
    /// y < 0           -> (arctan(y / x) - pi) in (-pi, -pi / 2)
    /// ```
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Radians,
    /// #     Angle,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let pi = f64::consts::PI;
    /// let x1 = 3_f64;
    /// let y1 = -3_f64;
    /// let x2 = -3_f64;
    /// let y2 = 3_f64;
    /// let expected1 = Radians(-pi / 4_f64);
    /// let expected2 = Radians(3_f64 * pi / 4_f64);
    /// let result1 = Radians::atan2(y1, x1);
    /// let result2 = Radians::atan2(y2, x2);
    /// 
    /// assert_relative_eq!(result1, expected1, epsilon = 1e-10);
    /// assert_relative_eq!(result2, expected2, epsilon = 1e-10);
    /// ```
    fn atan2(y: Self::Dimensionless, x: Self::Dimensionless) -> Self;

    /// Simultaneously compute the sine and cosine of a typed angle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Radians,
    /// #     Angle,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let pi_over_4 = Radians(f64::consts::FRAC_PI_4);
    /// let expected = (1_f64/ f64::sqrt(2_f64), 1_f64 / f64::sqrt(2_f64));
    /// let result = pi_over_4.sin_cos();
    /// 
    /// assert_relative_eq!(result.0, expected.0, epsilon = 1e-10);
    /// assert_relative_eq!(result.1, expected.1, epsilon = 1e-10);
    /// ```
    #[inline]
    fn sin_cos(self) -> (Self::Dimensionless, Self::Dimensionless) {
        (Self::sin(self), Self::cos(self))
    }

    /// The value of half of a full turn around the unit circle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// #     Degrees,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let expected_radians = Radians(f64::consts::PI);
    /// let expected_degrees = Degrees(180_f64);
    /// let result_radians = Radians::full_turn_div_2();
    /// let result_degrees = Degrees::full_turn_div_2();
    /// 
    /// assert_relative_eq!(result_radians, expected_radians, epsilon = 1e-10);
    /// assert_relative_eq!(result_degrees, expected_degrees, epsilon = 1e-10);
    /// ```
    #[inline]
    fn full_turn_div_2() -> Self {
        let denominator: Self::Dimensionless = cglinalg_numeric::cast(2);
        Self::full_turn() / denominator
    }

    /// The value of a one fourth of a full turn around the unit circle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// #     Degrees,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let expected_radians = Radians(f64::consts::FRAC_PI_2);
    /// let expected_degrees = Degrees(90_f64);
    /// let result_radians = Radians::full_turn_div_4();
    /// let result_degrees = Degrees::full_turn_div_4();
    /// 
    /// assert_relative_eq!(result_radians, expected_radians, epsilon = 1e-10);
    /// assert_relative_eq!(result_degrees, expected_degrees, epsilon = 1e-10);
    /// ```
    #[inline]
    fn full_turn_div_4() -> Self {
        let denominator: Self::Dimensionless = cglinalg_numeric::cast(4);
        Self::full_turn() / denominator
    }

    /// The value of one sixth of a full turn around the unit circle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// #     Degrees,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let expected_radians = Radians(f64::consts::FRAC_PI_3);
    /// let expected_degrees = Degrees(60_f64);
    /// let result_radians = Radians::full_turn_div_6();
    /// let result_degrees = Degrees::full_turn_div_6();
    /// 
    /// assert_relative_eq!(result_radians, expected_radians, epsilon = 1e-10);
    /// assert_relative_eq!(result_degrees, expected_degrees, epsilon = 1e-10);
    /// ```
    #[inline]
    fn full_turn_div_6() -> Self {
        let denominator: Self::Dimensionless = cglinalg_numeric::cast(6);
        Self::full_turn() / denominator
    }

    /// The value of one eighth of a full turn around the unit circle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// #     Degrees,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let expected_radians = Radians(f64::consts::FRAC_PI_4);
    /// let expected_degrees = Degrees(45_f64);
    /// let result_radians = Radians::full_turn_div_8();
    /// let result_degrees = Degrees::full_turn_div_8();
    /// 
    /// assert_relative_eq!(result_radians, expected_radians, epsilon = 1e-10);
    /// assert_relative_eq!(result_degrees, expected_degrees, epsilon = 1e-10);
    /// ```
    #[inline]
    fn full_turn_div_8() -> Self {
        let denominator: Self::Dimensionless = cglinalg_numeric::cast(8);
        Self::full_turn() / denominator
    }

    /// Map an angle to its smallest congruent angle in the range `[0, full_turn)`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(15_f64 * f64::consts::FRAC_PI_6);
    /// let expected = Radians(f64::consts::FRAC_PI_2);
    /// let result = angle.normalize();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    fn normalize(self) -> Self {
        let remainder = self % Self::full_turn();
        if remainder < Self::zero() {
            remainder + Self::full_turn()
        } else {
            remainder
        }
    }

    /// Map an angle to its smallest congruent angle in the range `[-full_turn / 2, full_turn / 2)`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(15_f64 * f64::consts::FRAC_PI_6);
    /// let expected = Radians(f64::consts::FRAC_PI_2);
    /// let result = angle.normalize_signed();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    fn normalize_signed(self) -> Self {
        let remainder = self.normalize();
        if remainder > Self::full_turn_div_2() {
            remainder - Self::full_turn()
        } else {
            remainder
        }
    }

    /// Compute the angle rotated by half of a turn.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Degrees(45_f64);
    /// let expected = Degrees(225_f64);
    /// let result = angle.opposite();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    fn opposite(self) -> Self {
        Self::normalize(self + Self::full_turn_div_2())
    }

    /// Compute the interior bisector of `self` and `other`, normalized to the
    /// range `[0, full_turn)`.
    /// 
    /// The interior bisector between two congruent angles `angle1` and `angle2` is
    /// given by
    /// ```text
    /// bisect(angle1, angle2) := angle1 + (1 / 2) * (angle2 - angle1)
    /// ```
    /// That is, the interior bisector between two angles is the angle that is 
    /// interpolated half-way between `angle1` and `angle2`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle1 = Degrees(0_f64);
    /// let angle2 = Degrees(120_f64);
    /// let expected = Degrees(60_f64);
    /// let result = Degrees::bisect(angle1, angle2);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    fn bisect(self, other: Self) -> Self {
        let one_half = cglinalg_numeric::cast(0.5);
        Self::normalize((other - self) * one_half + self)
    }

    /// Compute the cosecant of a typed angle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_6);
    /// let expected = 2_f64;
    /// let result = angle.csc();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    fn csc(self) -> Self::Dimensionless {
        Self::sin(self).recip()
    }

    /// Compute the cotangent of a typed angle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_6);
    /// let expected = f64::sqrt(3_f64);
    /// let result = angle.cot();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    #[inline]
    fn cot(self) -> Self::Dimensionless {
        Self::tan(self).recip()
    }

    /// Compute the secant of a typed angle.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_6);
    /// let expected = 2_f64 / f64::sqrt(3_f64);
    /// let result = angle.sec();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    #[inline]
    fn sec(self) -> Self::Dimensionless {
        Self::cos(self).recip()
    }
}

/// The angle (arc length) along the unit circle in units of radians.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Hash)]
pub struct Radians<S>(pub S);

impl<S> Radians<S> 
where 
    S: SimdScalar
{
    /// Construct a typed angle of zero radians.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let zero_radians: Radians<f64> = Radians::zero();
    /// 
    /// assert!(zero_radians.is_zero());
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self(S::zero())
    }

    /// Determine whether a typed angle is zero radians.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let zero_radians: Radians<f64> = Radians::zero();
    /// 
    /// assert!(zero_radians.is_zero());
    /// 
    /// let pi_radians: Radians<f64> = Radians(f64::consts::PI);
    /// 
    /// assert!(!pi_radians.is_zero());
    /// ```
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<S> Radians<S> 
where 
    S: SimdScalarFloat
{
    /// Returns `true` if the underlying floating point number of the typed
    /// angle is finite.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Radians,  
    /// # };
    /// # use core::f64;
    /// #
    /// let angle: Radians<f64> = Radians(f64::consts::PI / 4_f64);
    /// 
    /// assert!(angle.is_finite());
    /// ```
    #[inline]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

/// The angle (arc length) along the unit circle in units of degrees.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Hash)]
pub struct Degrees<S>(pub S);

impl<S> Degrees<S> 
where 
    S: SimdScalar 
{
    /// Construct a typed angle of zero degrees.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// # use core::f64;
    /// #
    /// let zero_degrees: Degrees<f64> = Degrees::zero();
    /// 
    /// assert!(zero_degrees.is_zero());
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self(S::zero())
    }

    /// Determine whether a typed angle is zero degrees.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// # use core::f64;
    /// #
    /// let zero_degrees: Degrees<f64> = Degrees::zero();
    /// 
    /// assert!(zero_degrees.is_zero());
    /// 
    /// let one_eighty_degrees: Degrees<f64> = Degrees(180_f64);
    /// 
    /// assert!(!one_eighty_degrees.is_zero());
    /// ```
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<S> Degrees<S> 
where 
    S: SimdScalarFloat 
{
    /// Returns `true` if the underlying floating point number of the typed
    /// angle is finite.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Angle,
    /// #     Degrees,  
    /// # };
    /// #
    /// let angle = Degrees(45_f64);
    ///
    /// assert!(angle.is_finite()); 
    /// ```
    #[inline]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

impl<S> Default for Radians<S>
where
    S: SimdScalar
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<S> Default for Degrees<S>
where
    S: SimdScalar
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<S> From<Degrees<S>> for Radians<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn from(degrees: Degrees<S>) -> Self {
        Self(degrees.0 * cglinalg_numeric::cast(f64::consts::PI / 180_f64))
    }
}

impl<S> From<Radians<S>> for Degrees<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn from(radians: Radians<S>) -> Self {
        Self(radians.0 * cglinalg_numeric::cast(180_f64 / f64::consts::PI))
    }
}

impl<S> fmt::Display for Degrees<S> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{} degrees", self.0)
    }
}

impl<S> fmt::Display for Radians<S> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{} radians", self.0)
    }
}

impl<S> ops::Add<Degrees<S>> for Degrees<S> 
where 
    S: SimdScalar
{
    type Output = Degrees<S>;

    #[inline]
    fn add(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<'a, S> ops::Add<&'a Degrees<S>> for Degrees<S> 
where 
    S: SimdScalar
{
    type Output = Degrees<S>;

    #[inline]
    fn add(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<'a, S> ops::Add<Degrees<S>> for &'a Degrees<S> 
where 
    S: SimdScalar
{
    type Output = Degrees<S>;

    #[inline]
    fn add(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<'a, 'b, S> ops::Add<&'b Degrees<S>> for &'a Degrees<S> 
where 
    S: SimdScalar
{
    type Output = Degrees<S>;

    #[inline]
    fn add(self, other: &'b Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<S> ops::Sub<Degrees<S>> for Degrees<S> 
where 
    S: SimdScalar
{
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 - other.0)
    } 
}

impl<'a, S> ops::Sub<&'a Degrees<S>> for Degrees<S> 
where 
    S: SimdScalar
{
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 - other.0)
    } 
}

impl<'a, S> ops::Sub<Degrees<S>> for &'a Degrees<S> 
where 
    S: SimdScalar
{
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 - other.0)
    } 
}

impl<'a, 'b, S> ops::Sub<&'b Degrees<S>> for &'a Degrees<S> 
where 
    S: SimdScalar
{
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: &'b Degrees<S>) -> Self::Output {
        Degrees(self.0 - other.0)
    } 
}

impl<S> ops::Mul<S> for Degrees<S> 
where
    S: SimdScalar
{
    type Output = Degrees<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Degrees(self.0 * other)
    }
}

impl<'a, S> ops::Mul<S> for &'a Degrees<S> 
where
    S: SimdScalar
{
    type Output = Degrees<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Degrees(self.0 * other)
    }
}

impl<S> ops::Div<S> for Degrees<S> 
where 
    S: SimdScalarFloat
{
    type Output = Degrees<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Degrees(self.0 / other)
    }
}

impl<'a, S> ops::Div<S> for &'a Degrees<S> 
where 
    S: SimdScalarFloat
{
    type Output = Degrees<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Degrees(self.0 / other)
    }
}

impl<S> ops::Div<Degrees<S>> for Degrees<S> 
where 
    S: SimdScalarFloat
{
    type Output = S;

    #[inline]
    fn div(self, other: Degrees<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<'a, S> ops::Div<&'a Degrees<S>> for Degrees<S> 
where 
    S: SimdScalarFloat
{
    type Output = S;

    #[inline]
    fn div(self, other: &'a Degrees<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<'a, S> ops::Div<Degrees<S>> for &'a Degrees<S> 
where 
    S: SimdScalarFloat
{
    type Output = S;

    #[inline]
    fn div(self, other: Degrees<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<'a, 'b, S> ops::Div<&'b Degrees<S>> for &'a Degrees<S> 
where 
    S: SimdScalarFloat
{
    type Output = S;

    #[inline]
    fn div(self, other: &'b Degrees<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<S> ops::Rem<Degrees<S>> for Degrees<S> 
where 
    S: SimdScalarFloat
{
    type Output = Degrees<S>;

    #[inline]
    fn rem(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 % other.0)
    }
}

impl<'a, S> ops::Rem<&'a Degrees<S>> for Degrees<S> 
where 
    S: SimdScalarFloat
{
    type Output = Degrees<S>;

    #[inline]
    fn rem(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 % other.0)
    }
}

impl<'a, S> ops::Rem<Degrees<S>> for &'a Degrees<S> 
where 
    S: SimdScalarFloat
{
    type Output = Degrees<S>;

    #[inline]
    fn rem(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 % other.0)
    }
}

impl<'a, 'b, S> ops::Rem<&'b Degrees<S>> for &'a Degrees<S> 
where 
    S: SimdScalarFloat 
{
    type Output = Degrees<S>;

    #[inline]
    fn rem(self, other: &'b Degrees<S>) -> Self::Output {
        Degrees(self.0 % other.0)
    }
}

impl<S> ops::Neg for Degrees<S> 
where 
    S: SimdScalarSigned 
{
    type Output = Degrees<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Degrees(-self.0)
    }
}

impl<'a, S> ops::Neg for &'a Degrees<S> 
where 
    S: SimdScalarSigned
{
    type Output = Degrees<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Degrees(-self.0)
    }
}

impl<S> ops::AddAssign<Degrees<S>> for Degrees<S> 
where 
    S: SimdScalar 
{
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    } 
}

impl<S> ops::SubAssign<Degrees<S>> for Degrees<S> 
where 
    S: SimdScalar 
{
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    } 
}

impl<S> ops::MulAssign<S> for Degrees<S> 
where 
    S: SimdScalar 
{
    #[inline]
    fn mul_assign(&mut self, other: S) {
        *self = *self * other;
    } 
}

impl<S> ops::DivAssign<S> for Degrees<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn div_assign(&mut self, other: S) {
        *self = *self / other;
    } 
}

impl<S> ops::RemAssign<Degrees<S>> for Degrees<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn rem_assign(&mut self, other: Self) {
        *self = *self % other;
    } 
}

impl<S> num_traits::Zero for Degrees<S> 
where 
    S: SimdScalar 
{
    #[inline]
    fn zero() -> Self {
        Self(S::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<S> approx::AbsDiffEq for Degrees<S> 
where 
    S: SimdScalarFloat 
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        S::abs_diff_eq(&self.0, &other.0, epsilon)
    }
}

impl<S> approx::RelativeEq for Degrees<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.0, &other.0, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Degrees<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.0, &other.0, epsilon, max_ulps)
    }
}

impl<S> ops::Add<Radians<S>> for Radians<S> 
where 
    S: SimdScalar 
{
    type Output = Radians<S>;

    #[inline]
    fn add(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 + other.0)
    } 
}

impl<'a, S> ops::Add<&'a Radians<S>> for Radians<S> 
where 
    S: SimdScalar
{
    type Output = Radians<S>;

    #[inline]
    fn add(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 + other.0)
    } 
}

impl<'a, S> ops::Add<Radians<S>> for &'a Radians<S> 
where 
    S: SimdScalar 
{
    type Output = Radians<S>;

    #[inline]
    fn add(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 + other.0)
    } 
}

impl<'a, 'b, S> ops::Add<&'b Radians<S>> for &'a Radians<S> 
where 
    S: SimdScalar
{
    type Output = Radians<S>;

    #[inline]
    fn add(self, other: &'b Radians<S>) -> Self::Output {
        Radians(self.0 + other.0)
    } 
}

impl<S> ops::Sub<Radians<S>> for Radians<S> 
where 
    S: SimdScalar
{
    type Output = Radians<S>;

    #[inline]
    fn sub(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 - other.0)
    } 
}

impl<'a, S> ops::Sub<&'a Radians<S>> for Radians<S> 
where 
    S: SimdScalar
{
    type Output = Radians<S>;

    #[inline]
    fn sub(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 - other.0)
    } 
}

impl<'a, S> ops::Sub<Radians<S>> for &'a Radians<S> 
where 
    S: SimdScalar
{
    type Output = Radians<S>;

    #[inline]
    fn sub(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 - other.0)
    } 
}

impl<'a, 'b, S> ops::Sub<&'b Radians<S>> for &'a Radians<S> 
where 
    S: SimdScalar
{
    type Output = Radians<S>;

    #[inline]
    fn sub(self, other: &'b Radians<S>) -> Self::Output {
        Radians(self.0 - other.0)
    } 
}

impl<S> ops::Mul<S> for Radians<S> 
where 
    S: SimdScalar
{
    type Output = Radians<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Radians(self.0 * other)
    }
}

impl<'a, S> ops::Mul<S> for &'a Radians<S> 
where 
    S: SimdScalar
{
    type Output = Radians<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Radians(self.0 * other)
    }
}

impl<S> ops::Div<S> for Radians<S> 
where 
    S: SimdScalarFloat 
{
    type Output = Radians<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Radians(self.0 / other)
    }
}

impl<'a, S> ops::Div<S> for &'a Radians<S> 
where 
    S: SimdScalarFloat 
{
    type Output = Radians<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Radians(self.0 / other)
    }
}

impl<S> ops::Div<Radians<S>> for Radians<S> 
where 
    S: SimdScalarFloat
{
    type Output = S;

    #[inline]
    fn div(self, other: Radians<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<'a, S> ops::Div<&'a Radians<S>> for Radians<S> 
where 
    S: SimdScalarFloat
{
    type Output = S;

    #[inline]
    fn div(self, other: &'a Radians<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<'a, S> ops::Div<Radians<S>> for &'a Radians<S> 
where 
    S: SimdScalarFloat
{
    type Output = S;

    #[inline]
    fn div(self, other: Radians<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<'a, 'b, S> ops::Div<&'b Radians<S>> for &'a Radians<S> 
where 
    S: SimdScalarFloat
{
    type Output = S;

    #[inline]
    fn div(self, other: &'b Radians<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<S> ops::Rem<Radians<S>> for Radians<S> 
where 
    S: SimdScalarFloat
{
    type Output = Radians<S>;

    #[inline]
    fn rem(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 % other.0)
    }
}

impl<'a, S> ops::Rem<&'a Radians<S>> for Radians<S> 
where 
    S: SimdScalarFloat
{
    type Output = Radians<S>;

    #[inline]
    fn rem(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 % other.0)
    }
}

impl<'a, S> ops::Rem<Radians<S>> for &'a Radians<S> 
where 
    S: SimdScalarFloat
{
    type Output = Radians<S>;

    #[inline]
    fn rem(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 % other.0)
    }
}

impl<'a, 'b, S> ops::Rem<&'b Radians<S>> for &'a Radians<S> 
where 
    S: SimdScalarFloat
{
    type Output = Radians<S>;

    #[inline]
    fn rem(self, other: &'b Radians<S>) -> Self::Output {
        Radians(self.0 % other.0)
    }
}

impl<S> ops::Neg for Radians<S> 
where 
    S: SimdScalarSigned
{
    type Output = Radians<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Radians(-self.0)
    }
}

impl<'a, S> ops::Neg for &'a Radians<S> 
where 
    S: SimdScalarSigned
{
    type Output = Radians<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Radians(-self.0)
    }
}

impl<S> ops::AddAssign<Radians<S>> for Radians<S> 
where 
    S: SimdScalar
{
    #[inline]
    fn add_assign(&mut self, other: Radians<S>) {
        self.0 += other.0;
    } 
}

impl<S> ops::SubAssign<Radians<S>> for Radians<S> 
where 
    S: SimdScalar
{
    #[inline]
    fn sub_assign(&mut self, other: Radians<S>) {
        self.0 -= other.0;
    } 
}

impl<S> ops::MulAssign<S> for Radians<S> 
where 
    S: SimdScalar
{
    #[inline]
    fn mul_assign(&mut self, other: S) {
        self.0 *= other;
    } 
}

impl<S> ops::DivAssign<S> for Radians<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn div_assign(&mut self, other: S) {
        self.0 /= other;
    } 
}

impl<S> ops::RemAssign<Radians<S>> for Radians<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn rem_assign(&mut self, other: Self) {
        self.0 = self.0 % other.0;
    } 
}

impl<S> num_traits::Zero for Radians<S> 
where 
    S: SimdScalar 
{
    #[inline]
    fn zero() -> Self {
        Self(S::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<S> approx::AbsDiffEq for Radians<S> 
where 
    S: SimdScalarFloat
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        S::abs_diff_eq(&self.0, &other.0, epsilon)
    }
}

impl<S> approx::RelativeEq for Radians<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.0, &other.0, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Radians<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.0, &other.0, epsilon, max_ulps)
    }
}

impl<S> Angle for Radians<S> 
where 
    S: SimdScalarFloat
{
    type Dimensionless = S;

    #[inline]
    fn full_turn() -> Self {
        Self(cglinalg_numeric::cast(2_f64 * f64::consts::PI))
    }

    #[inline]
    fn sin(self) -> Self::Dimensionless {
        S::sin(self.0)
    }

    #[inline]
    fn cos(self) -> Self::Dimensionless {
        S::cos(self.0)
    }

    #[inline]
    fn tan(self) -> Self::Dimensionless {
        S::tan(self.0)
    }

    #[inline]
    fn asin(ratio: Self::Dimensionless) -> Self {
        Self(Self::Dimensionless::asin(ratio))
    }

    #[inline]
    fn acos(ratio: Self::Dimensionless) -> Self {
        Self(Self::Dimensionless::acos(ratio))
    }

    #[inline]
    fn atan(ratio: Self::Dimensionless) -> Self {
        Self(Self::Dimensionless::atan(ratio))
    }

    #[inline]
    fn atan2(a: Self::Dimensionless, b: Self::Dimensionless) -> Self {
        Self(Self::Dimensionless::atan2(a, b))
    }
}

impl<S> Angle for Degrees<S> 
where 
    S: SimdScalarFloat
{
    type Dimensionless = S;

    #[inline]
    fn full_turn() -> Self {
        Self(cglinalg_numeric::cast(360))
    }

    #[inline]
    fn sin(self) -> Self::Dimensionless {
        Radians::from(self).sin()
    }

    #[inline]
    fn cos(self) -> Self::Dimensionless {
        Radians::from(self).cos()
    }

    #[inline]
    fn tan(self) -> Self::Dimensionless {
        Radians::from(self).tan()
    }

    #[inline]
    fn asin(ratio: Self::Dimensionless) -> Self {
        Radians(ratio.asin()).into()
    }

    #[inline]
    fn acos(ratio: Self::Dimensionless) -> Self {
        Radians(ratio.acos()).into()
    }

    #[inline]
    fn atan(ratio: Self::Dimensionless) -> Self {
        Radians(ratio.atan()).into()
    }

    #[inline]
    fn atan2(a: Self::Dimensionless, b: Self::Dimensionless) -> Self {
        Radians(Self::Dimensionless::atan2(a, b)).into()
    }
}

