use num_traits::{
    Num, 
    NumCast,
    Signed,
};

use core::fmt;
use core::ops;


/// A data type with this trait has the properties of a 
/// set of scalar numbers underlying vector and matrix 
/// data types.
pub trait SimdScalar 
where
    Self: Copy,
    Self: Clone,
    Self: fmt::Debug,
    Self: fmt::Display,
    Self: Num,
    Self: NumCast,
    Self: PartialOrd, 
    Self: ops::AddAssign,
    Self: ops::SubAssign,
    Self: ops::MulAssign,
    Self: ops::DivAssign,
    Self: ops::RemAssign,
{
}

/// A trait representing numbers with subtraction and additive inverses.
pub trait SimdScalarSigned 
where
    Self: SimdScalar + Signed
{
    /// Determine whether the sign of the number is positive.
    /// 
    /// # Examples
    /// 
    /// Examples of using `is_sign_positive` with floating point numbers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarSigned,
    /// # };
    /// # use core::f64;
    /// #
    /// let nan = f64::NAN;
    /// let neg_nan = -f64::NAN;
    /// let value = 7_f64;
    /// let neg_value = -7_f64;
    /// 
    /// assert!(value.is_sign_positive());
    /// assert!(!neg_value.is_sign_positive());
    /// assert!(nan.is_sign_positive());
    /// assert!(!neg_nan.is_sign_positive());
    /// ```
    /// 
    /// Examples of using `is_sign_positive` with integers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarSigned,
    /// # };
    /// # use core::i32;
    /// #
    /// let value = 7_i32;
    /// let neg_value = -7_i32;
    /// 
    /// assert!(value.is_sign_positive());
    /// assert!(!neg_value.is_sign_positive());
    /// ```
    fn is_sign_positive(self) -> bool;

    /// Determine whether the sign of the number is negative.
    /// 
    /// # Examples
    /// 
    /// Examples of using `is_sign_negative` with floating point numbers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarSigned,
    /// # };
    /// # use core::f64;
    /// #
    /// let nan = f64::NAN;
    /// let neg_nan = -f64::NAN;
    /// let value = 7_f64;
    /// let neg_value = -7_f64;
    /// 
    /// assert!(!value.is_sign_negative());
    /// assert!(neg_value.is_sign_negative());
    /// assert!(!nan.is_sign_negative());
    /// assert!(neg_nan.is_sign_negative());
    /// ```
    /// 
    /// Examples of using `is_sign_negative` with integers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarSigned,
    /// # };
    /// # use core::i32;
    /// #
    /// let value = 7_i32;
    /// let neg_value = -7_i32;
    /// 
    /// assert!(!value.is_sign_negative());
    /// assert!(neg_value.is_sign_negative());
    /// ```
    fn is_sign_negative(self) -> bool;

    /// Copy the sign of `sign` to `self`.
    /// 
    /// The `copysign` function is defined as follows. Given a number `x` and a number `sign`
    /// ```text
    /// copysign(x) = signum(sign) * abs(x)
    /// ```
    /// 
    /// # Examples
    /// 
    /// Examples of using `copysign` with floating point numbers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarSigned,
    /// # };
    /// # use core::f64;
    /// #
    /// let value = 3.5_f64;
    /// let sign = 0.5;
    /// 
    /// assert_eq!(value.copysign(sign),      value);
    /// assert_eq!(value.copysign(-sign),    -value);
    /// assert_eq!((-value).copysign(sign),   value);
    /// assert_eq!((-value).copysign(-sign), -value);
    /// 
    /// assert!(f64::NAN.copysign(1_f64).is_nan());
    /// assert!(f64::NAN.copysign(-1_f64).is_nan());
    /// ```
    /// 
    /// Examples of using `copysign` with integers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarSigned,
    /// # };
    /// # use core::i32;
    /// #
    /// let value = 3_i32;
    /// let sign = 7_i32;
    /// 
    /// assert_eq!(value.copysign(sign),      value);
    /// assert_eq!(value.copysign(-sign),    -value);
    /// assert_eq!((-value).copysign(sign),   value);
    /// assert_eq!((-value).copysign(-sign), -value);
    /// ```
    fn copysign(self, sign: Self) -> Self;

    /// Calculate the signum of the number.
    /// 
    /// The signum of a number is the number of the same type that represents its sign, 
    /// such that given a number `x`
    /// ```text
    /// signum(x) * abs(x) = x
    /// ```
    /// NOTE: this is a more general condition than the definition
    /// ```text
    /// signum(x) = if x > 0 { 1 } else if x < 0 { -1 } else { 0 }
    /// ```
    /// For floating point number types, the number `0` is also 
    /// signed: `signum(+0.0) == +1.0` and `signum(-0.0) == -1.0`. Moreover, the 
    /// value `NaN` evaluates to `NaN`. Indeed, the direct definition for `signum` does 
    /// not have a value for `NaN` in floating point types. For nonzero, non-`NaN`
    /// floating point numbers, `signum` satisfies either relation.
    /// 
    /// # Examples
    /// 
    /// Examples of using `signum` with floating point numbers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarSigned,
    /// # };
    /// # use core::f64;
    /// #
    /// let value = 6.9_f64;
    /// 
    /// assert_eq!(value.signum(), 1_f64);
    /// assert_eq!((-value).signum(), -1_f64);
    /// 
    /// assert_eq!(0_f64.signum(), 1_f64);
    /// assert_eq!((-0_f64).signum(), -1_f64);
    /// 
    /// assert!(f64::NAN.signum().is_nan());
    /// 
    /// assert_eq!(f64::INFINITY.signum(), 1_f64);
    /// assert_eq!(f64::NEG_INFINITY.signum(), -1_f64);
    /// ```
    /// 
    /// Examples of using `signum` with integers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarSigned,
    /// # };
    /// # use core::i32;
    /// #
    /// let value = 6_i32;
    /// 
    /// assert_eq!(value.signum(), 1_i32);
    /// assert_eq!((-value).signum(), -1_i32);
    /// 
    /// assert_eq!(0_i32.signum(), 0_i32);
    /// assert_eq!((-0_i32).signum(), 0_i32);
    /// ```
    fn signum(self) -> Self;

    /// Calculate the absolute value of the number.
    /// 
    /// The absolute value of a number `x` is the number of the same type that satisfies
    /// ```text
    /// abs(x) = signum(x) * x
    /// ```
    /// For integer types, this corresponds to the conventional mathematical formula
    /// ```text
    /// abs(x) = if x >= 0 { x } else { -x }
    /// ```
    /// whereas for floating point types, the first relation accounts for the case where
    /// `x` is `NaN`.
    /// 
    /// # Examples
    /// 
    /// Examples of using `abs` with floating point numbers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarSigned,
    /// # };
    /// # use core::f64;
    /// #
    /// let value1 = 6.9_f64;
    /// let value2 = -6.9_f64;
    /// 
    /// assert_eq!(value1.abs(), value1);
    /// assert_eq!(value2.abs(), value1);
    /// 
    /// assert!(f64::NAN.abs().is_nan());
    /// ```
    /// 
    /// Examples of using `abs` with integers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarSigned,
    /// # };
    /// # use core::i32;
    /// #
    /// let value1 = 6_i32;
    /// let value2 = -6_i32;
    /// 
    /// assert_eq!(value1.abs(), value1);
    /// assert_eq!(value2.abs(), value1);
    /// ```
    fn abs(self) -> Self;
}

pub trait SimdScalarOrd
where
    Self: SimdScalar + PartialOrd
{
    /// Calculate the maximum value of two numbers.
    /// 
    /// # Examples
    /// 
    /// Examples of using `max` with floating point numbers.
    /// ```
    /// # use cglinalg_core::{
    /// #      SimdScalarOrd,
    /// # };
    /// # use core::f64;
    /// #
    /// assert_eq!(SimdScalarOrd::max(1_f64, 2_f64), 2_f64);
    /// assert_eq!(SimdScalarOrd::max(-1_f64, -2_f64), -1_f64);
    /// ```
    /// 
    /// Examples of using `max` with integers.
    /// ```
    /// # use cglinalg_core::{
    /// #      SimdScalarOrd,
    /// # };
    /// # // use core::i32;
    /// #
    /// assert_eq!(SimdScalarOrd::max(1_i32, 2_i32), 2_i32);
    /// assert_eq!(SimdScalarOrd::max(-1_i32, -2_i32), -1_i32);
    /// ```
    fn max(self, other: Self) -> Self;

    /// Calculate the minimum value of two numbers.
    /// 
    /// # Examples
    /// 
    /// Examples of using `min` with floating point numbers.
    /// ```
    /// # use cglinalg_core::{
    /// #      SimdScalarOrd,
    /// # };
    /// # use core::f64;
    /// #
    /// assert_eq!(SimdScalarOrd::min(1_f64, 2_f64), 1_f64);
    /// assert_eq!(SimdScalarOrd::min(-1_f64, -2_f64), -2_f64);
    /// ```
    /// 
    /// Examples of using `min` with integers.
    /// ```
    /// # use cglinalg_core::{
    /// #      SimdScalarOrd,
    /// # };
    /// # use core::i32;
    /// #
    /// assert_eq!(SimdScalarOrd::min(1_i32, 2_i32), 1_i32);
    /// assert_eq!(SimdScalarOrd::min(-1_i32, -2_i32), -2_i32);
    /// ```
    fn min(self, other: Self) -> Self;

    /// Clamp the scalar to the range `[min_value, max_value]`.
    /// 
    /// This functions returns `min_value` if `self` < `min_value`, and it returns
    /// `max_value` if `self` > `max_value`.
    /// 
    /// # Examples
    /// 
    /// Examples of using `clamp` with floating point numbers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarOrd,
    /// # };
    /// # use core::f64;
    /// #
    /// assert_eq!(SimdScalarOrd::clamp(-3_f64, -2_f64, 1_f64), -2_f64);
    /// assert_eq!(SimdScalarOrd::clamp(0_f64, -2_f64, 1_f64), 0_f64);
    /// assert_eq!(SimdScalarOrd::clamp(2_f64, -2_f64, 1_f64), 1_f64);
    /// assert!(SimdScalarOrd::clamp(f64::NAN, -2_f64, 1_f64).is_nan());
    /// ```
    /// 
    /// Examples of using `clamp` with integers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarOrd,
    /// # };
    /// # use core::i32;
    /// #
    /// assert_eq!(SimdScalarOrd::clamp(-3_i32, -2_i32, 1_i32), -2_i32);
    /// assert_eq!(SimdScalarOrd::clamp(0_i32, -2_i32, 1_i32), 0_i32);
    /// assert_eq!(SimdScalarOrd::clamp(2_i32, -2_i32, 1_i32), 1_i32);
    /// ```
    fn clamp(self, min_value: Self, max_value: Self) -> Self;
}

/// A trait representing numbers that have finite minimum and maximum values
/// that they can represent.
pub trait SimdScalarBounded
where
    Self: SimdScalar + SimdScalarOrd
{
    /// Returns the smallest finite value of a number type.
    /// 
    /// # Examples
    /// 
    /// An example of using `min_value` with floating point numbers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarBounded,
    /// # };
    /// # use core::f64;
    /// # 
    /// let min_value_f64 = f64::min_value();
    /// 
    /// assert_eq!(min_value_f64, f64::MIN);
    /// ```
    /// 
    /// An example of using `min_value` with integers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarBounded,
    /// # };
    /// # use core::i32;
    /// #
    /// let min_value_i32 = i32::min_value();
    /// 
    /// assert_eq!(min_value_i32, i32::MIN);
    /// ```
    fn min_value() -> Self;

    /// Returns the largest finite value of a number type.
    /// 
    /// # Examples
    /// 
    /// An example of using `max_value` with floating point numbers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarBounded,
    /// # };
    /// # use core::f64;
    /// # 
    /// let max_value_f64 = f64::max_value();
    /// 
    /// assert_eq!(max_value_f64, f64::MAX);
    /// ```
    /// 
    /// An example of using `max_value` with integers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarBounded,
    /// # };
    /// # use core::i32;
    /// #
    /// let max_value_i32 = i32::max_value();
    /// 
    /// assert_eq!(max_value_i32, i32::MAX);
    /// ```
    fn max_value() -> Self;
}

/// A trait representing numbers that have the properties of finite precision
/// floating point arithmetic.
pub trait SimdScalarFloat:
      SimdScalarSigned + SimdScalarOrd + SimdScalarBounded
    + approx::AbsDiffEq<Epsilon = Self>
    + approx::RelativeEq<Epsilon = Self>
    + approx::UlpsEq<Epsilon = Self>
    + ops::Neg<Output = Self> 
{    
    /// Return the largest integer less than or equal to `self`.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// assert_eq!(SimdScalarFloat::floor(3.01_f64), 3_f64);
    /// assert_eq!(SimdScalarFloat::floor(3_f64), 3_f64);
    /// assert_eq!(SimdScalarFloat::floor(-3.01_f64), -4_f64);
    /// ```
    fn floor(self) -> Self;

    /// Return the smallest integer greater than or equal to `self`.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// assert_eq!(SimdScalarFloat::ceil(3.01_f64), 4_f64);
    /// assert_eq!(SimdScalarFloat::ceil(3_f64), 3_f64);
    /// assert_eq!(SimdScalarFloat::ceil(-3.01_f64), -3_f64);
    /// ```
    fn ceil(self) -> Self;

    /// Return the nearest integer to `self`, rounding half-way cases away from `0.0`.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// assert_eq!(SimdScalarFloat::round(3.3_f64), 3_f64);
    /// assert_eq!(SimdScalarFloat::round(-3.3_f64), -3_f64);
    /// assert_eq!(SimdScalarFloat::round(-3.7_f64), -4_f64);
    /// assert_eq!(SimdScalarFloat::round(3.5_f64), 4_f64);
    /// assert_eq!(SimdScalarFloat::round(4.5_f64), 5_f64);
    /// ```
    fn round(self) -> Self;

    /// Return the integer part of `self`, where non-integer numbers are always 
    /// truncated towards zero.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// # 
    /// assert_eq!(SimdScalarFloat::trunc(3.7_f64), 3_f64);
    /// assert_eq!(SimdScalarFloat::trunc(3_f64), 3_f64);
    /// assert_eq!(SimdScalarFloat::trunc(-3.7_f64), -3_f64);
    /// ```
    fn trunc(self) -> Self;

    /// Return the fractional part of `self`.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// # 
    /// assert_relative_eq!(SimdScalarFloat::fract(3.6_f64), 0.6_f64, epsilon = 1e-10);
    /// assert_relative_eq!(SimdScalarFloat::fract(-3.6_f64), -0.6_f64, epsilon = 1e-10);
    /// ```
    fn fract(self) -> Self;
    
    /// Compute the fused multiply-add of `self` with `a` and `b`.
    /// 
    /// Given a floating point number `self`, and floating point numbers `a` and 
    /// `b`, the fused multiply-add operation is given by
    /// ```text
    /// mul_add(self, a, b) = (self * a) + b 
    /// ```
    /// where the entire operation is done with only one rounding error, yielding 
    /// a more accurate result than doing one multiply and one add separately. A
    /// fused multiply-add operation can also be more performant than an unfused one 
    /// on hardware architectures with a dedicated `fma` instruction.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let m = 10_f64;
    /// let a = 4_f64;
    /// let b = 60_f64;
    /// 
    /// assert_relative_eq!(m.mul_add(a, b), m * a + b, epsilon = 1e-10);
    /// ```
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Compute the length of the hypotenuse of a right triangle with legs of 
    /// length `self` and `other`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let x = 3_f64;
    /// let y = 4_f64;
    /// let expected = 5_f64;
    /// let result = x.hypot(y);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    fn hypot(self, other: Self) -> Self;

    /// Compute the reciprocal (inverse) of a number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let x = 2_f64;
    /// 
    /// assert_eq!(x.recip(), 1_f64 / x);
    /// ```
    fn recip(self) -> Self;
    
    /// Compute the four quadrant arctangent of two arguments.
    /// 
    /// The return value is the arc tangent of the quotient of the two input values. 
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
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let pi = f64::pi();
    /// let x1 = 3_f64;
    /// let y1 = -3_f64;
    /// let x2 = -3_f64;
    /// let y2 = 3_f64;
    /// 
    /// assert_relative_eq!(SimdScalarFloat::atan2(y1, x1), -pi / 4_f64, epsilon = 1e-10);
    /// assert_relative_eq!(SimdScalarFloat::atan2(y2, x2), 3_f64 * pi / 4_f64, epsilon = 1e-10);
    /// ```
    fn atan2(self, other: Self) -> Self;

    /// Simultaneously compute the sine and cosine of `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let pi_over_4 = f64::frac_pi_4();
    /// let expected = (f64::frac_1_sqrt_2(), f64::frac_1_sqrt_2());
    /// let result = SimdScalarFloat::sin_cos(pi_over_4);
    /// 
    /// assert_relative_eq!(result.0, expected.0, epsilon = 1e-10);
    /// assert_relative_eq!(result.1, expected.1, epsilon = 1e-10);
    /// ```
    fn sin_cos(self) -> (Self, Self);

    /// Compute the sine of the number `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let pi_over_4 = f64::frac_pi_4();
    /// let expected = f64::frac_1_sqrt_2();
    /// let result = SimdScalarFloat::sin(pi_over_4);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn sin(self) -> Self;

    /// Compute the cosine of the number `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let pi_over_4 = f64::frac_pi_4();
    /// let expected = f64::frac_1_sqrt_2();
    /// let result = SimdScalarFloat::cos(pi_over_4);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn cos(self) -> Self;

    /// Compute the tangent of the number `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let pi_over_4 = f64::frac_pi_4();
    /// let expected = 1_f64;
    /// let result = SimdScalarFloat::tan(pi_over_4);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn tan(self) -> Self;

    /// Compute the arcsine of `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let sin_pi_over_2 = 1_f64;
    /// let expected = f64::frac_pi_2();
    /// let result = SimdScalarFloat::asin(sin_pi_over_2);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    fn asin(self) -> Self;
    
    /// Compute the arccosine of `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let cos_pi_over_4 = f64::frac_1_sqrt_2();
    /// let expected = f64::frac_pi_4();
    /// let result = SimdScalarFloat::acos(cos_pi_over_4);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    fn acos(self) -> Self;

    /// Compute the arctangent of `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let tan_pi_over_4 = 1_f64;
    /// let expected = f64::frac_pi_4();
    /// let result = SimdScalarFloat::atan(tan_pi_over_4);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    fn atan(self) -> Self;
    
    /// Compute the hyperbolic sine of `self`.
    /// 
    /// Given a floating point number `x`, the hyperbolic sine of `x` is given
    /// by
    /// ```text
    /// sinh(x) = (1 / 2) * (exp(x) - exp(-x))
    /// ```
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// // Golden ratio.
    /// let phi = (1_f64 + f64::sqrt(5_f64)) / 2_f64;
    /// let expected = 1_f64 / 2_f64;
    /// let result = SimdScalarFloat::sinh(SimdScalarFloat::ln(phi));
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn sinh(self) -> Self;

    /// Compute the hyperbolic cosine of `self`.
    /// 
    /// Given a floating point number `x`, the hyperbolic cosine of `x` is given
    /// by
    /// ```text
    /// cosh(x) = (1 / 2) * (exp(x) + exp(-x))
    /// ```
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// // Golden ratio.
    /// let phi = (1_f64 + f64::sqrt(5_f64)) / 2_f64;
    /// let expected = (1_f64 / 2_f64) * f64::sqrt(5_f64);
    /// let result = SimdScalarFloat::cosh(SimdScalarFloat::ln(phi));
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn cosh(self) -> Self;

    /// Compute the hyperbolic tangent of `self`
    /// 
    /// Given a floating point number `x`, the hyperbolic tangent of `x` is given 
    /// by
    /// ```text
    /// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    /// ```
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// // Golden ratio.
    /// let phi = (1_f64 + f64::sqrt(5_f64)) / 2_f64;
    /// let expected = (1_f64 / 5_f64) * f64::sqrt(5_f64);
    /// let result = SimdScalarFloat::tanh(SimdScalarFloat::ln(phi));
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn tanh(self) -> Self;

    /// Compute the inverse hyperbolic sine of `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let x = 1_f64;
    /// let expected = 1_f64;
    /// let result = x.sinh().asinh();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn asinh(self) -> Self;

    /// Compute the inverse hyperbolic cosine of `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let x = 1_f64;
    /// let expected = 1_f64;
    /// let result = x.cosh().acosh();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn acosh(self) -> Self;

    /// Compute the inverse hyperbolic tangent of `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let e = f64::e();
    /// let expected = e;
    /// let result = e.tanh().atanh();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn atanh(self) -> Self;
    
    /// Simultaneously compute the hyperbolic sine and hyperbolic cosine of `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// // Golden ratio.
    /// let phi = (1_f64 + f64::sqrt(5_f64)) / 2_f64;
    /// let sinh_ln_phi = 1_f64 / 2_f64;
    /// let cosh_ln_phi = (1_f64 / 2_f64) * f64::sqrt(5_f64);
    /// let expected = (sinh_ln_phi, cosh_ln_phi);
    /// let result = SimdScalarFloat::sinh_cosh(SimdScalarFloat::ln(phi));
    /// 
    /// assert_eq!(result.0, expected.0);
    /// assert_eq!(result.1, expected.1);
    /// ```
    fn sinh_cosh(self) -> (Self, Self) {
        (SimdScalarFloat::sinh(self), SimdScalarFloat::cosh(self))
    }
    
    /// Compute the logarithm of a number `self` with respect to a base `base`.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let value = 27_f64;
    /// let base = 3_f64;
    /// let expected = 3_f64;
    /// let result = value.log(base);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn log(self, base: Self) -> Self;

    /// Compute the logarithm of a number `self` with respect to a base of 2.
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let value = 32_f64;
    /// let expected = 5_f64;
    /// let result = value.log2();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn log2(self) -> Self;

    /// Compute the logarithm of a number `self` with respect to a base of 10.
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #      SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let value = 100_f64;
    /// let expected = 2_f64;
    /// let result = value.log10();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn log10(self) -> Self;

    /// Compute the natural logarithm of the number `self`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #      SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let value = SimdScalarFloat::exp(1_f64);
    /// let expected = 1_f64;
    /// let result = value.ln();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn ln(self) -> Self;

    /// Compute the value `ln(1 + self)` more accurately than if the operations
    /// we performed separately.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #      SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let value = f64::e() - 1_f64;
    /// let expected = 1_f64;
    /// let result = value.ln_1p();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn ln_1p(self) -> Self;
    
    /// Calculate the square root of the number `self`.
    /// 
    /// Returns `NaN` if `self` is a negative number other than `0.0`.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// #
    /// let positive = 4_f64;
    /// let negative = -4_f64;
    /// let negative_zero = -0_f64;
    ///
    /// assert_eq!(positive.sqrt(), 2_f64);
    /// assert!(negative.sqrt().is_nan());
    /// assert!(negative_zero.sqrt() == negative_zero);
    /// ```
    fn sqrt(self) -> Self;

    /// Calculate the cube root of the number `self`.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// #
    /// let value = 8_f64;
    /// 
    /// assert_eq!(value.cbrt(), 2_f64);
    /// assert_eq!((-value).cbrt(), -2_f64);
    /// ```
    fn cbrt(self) -> Self;
    
    /// Compute the exponential of the number `self`.
    /// 
    /// The exponential function is defined as follows. Given a floating point
    /// number `x`
    /// ```text
    /// exp(x) = e^x
    /// ```
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let expected = 1_f64;
    /// let e = SimdScalarFloat::exp(1_f64);
    /// let result = SimdScalarFloat::ln(e);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn exp(self) -> Self;

    /// Compute the exponential of the number `self` with respect to base 2.
    /// 
    /// Given a floating point number `x`
    /// ```text
    /// exp2(x) = 2^x
    /// ```
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let value = 2_f64;
    /// let expected = 4_f64;
    /// let result = SimdScalarFloat::exp2(value);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn exp2(self) -> Self;

    /// Compute the value `exp(self) - 1` in a way that remains accurate even 
    /// when `self` is close to zero.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let value = 7_f64;
    /// let expected = 6_f64;
    /// let result = SimdScalarFloat::exp_m1(SimdScalarFloat::ln(value));
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn exp_m1(self) -> Self;

    /// Compute the power of `self` with respect to an integer power `n`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let x = 2_f64;
    /// let expected = x * x;
    /// let result = x.powi(2_i32);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn powi(self, n: i32) -> Self;

    /// Compute the power of `self` with respect to a floating point power `n`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let x = 2_f64;
    /// let expected = x * x;
    /// let result = x.powf(2_f64);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    fn powf(self, n: Self) -> Self;
        
    /// Return `true` if `self` is neither infinite, nor `NaN`.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let value = 7_f64;
    /// let infinity = f64::INFINITY;
    /// let neg_infinity = f64::NEG_INFINITY;
    /// let nan = f64::NAN;
    ///
    /// assert!(value.is_finite());
    ///
    /// assert!(!nan.is_finite());
    /// assert!(!infinity.is_finite());
    /// assert!(!neg_infinity.is_finite());
    /// ```
    fn is_finite(self) -> bool;

    /// Return `true` if `self` is either positive infinity, or negative 
    /// infinity, and `false` otherwise.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let value = 7_f64;
    /// let infinity = f64::INFINITY;
    /// let neg_infinity = f64::NEG_INFINITY;
    /// let nan = f64::NAN;
    ///
    /// assert!(!value.is_infinite());
    ///
    /// assert!(!nan.is_infinite());
    /// assert!(infinity.is_infinite());
    /// assert!(neg_infinity.is_infinite());
    /// ```
    fn is_infinite(self) -> bool;

    /// Return `true` if `self` is `NaN`, and `false` otherwise.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let value = 7_f64;
    /// let infinity = f64::INFINITY;
    /// let neg_infinity = f64::NEG_INFINITY;
    /// let nan = f64::NAN;
    ///
    /// assert!(!value.is_nan());
    ///
    /// assert!(nan.is_nan());
    /// assert!(!infinity.is_nan());
    /// assert!(!neg_infinity.is_nan());
    /// ```
    fn is_nan(self) -> bool;

    /// Returns `true` if the floating point number is neither zero, 
    /// infinite, subnormal, or `NaN`. The function returns `false` otherwise.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let min_positive_value = f64::MIN_POSITIVE;
    /// let max_value = f64::MAX;
    /// let less_than_min_value = 1e-308;
    /// let zero = 0_f64;
    /// let nan = f64::NAN;
    /// let infinity = f64::INFINITY;
    /// 
    /// assert!(min_positive_value.is_normal());
    /// assert!(max_value.is_normal());
    /// 
    /// assert!(!zero.is_normal());
    /// assert!(!nan.is_normal());
    /// assert!(!infinity.is_normal());
    /// 
    /// assert!(!less_than_min_value.is_normal());
    /// ```
    fn is_normal(self) -> bool;

    /// Returns `true` if the floating point number is subnormal, and `false` 
    /// otherwise.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let min_positive_value = f64::MIN_POSITIVE;
    /// let max_value = f64::MAX;
    /// let zero = 0_f64;
    /// let nan = f64::NAN;
    /// let infinity = f64::INFINITY;
    /// let neg_infinity = f64::NEG_INFINITY;
    /// let less_than_min_value = 1e-308;
    /// 
    /// assert!(!min_positive_value.is_subnormal());
    /// assert!(!max_value.is_subnormal());
    /// assert!(!zero.is_subnormal());
    /// assert!(!nan.is_subnormal());
    /// assert!(!infinity.is_subnormal());
    /// assert!(!neg_infinity.is_subnormal());
    /// 
    /// assert!(less_than_min_value > zero);
    /// assert!(less_than_min_value < min_positive_value);
    /// assert!(less_than_min_value.is_subnormal());
    /// ```
    fn is_subnormal(self) -> bool;

    /// Returns the smallest positive value that this type can represent.
    /// 
    /// # Example
    /// 
    /// An example of using `min_positive` with floating point numbers.
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let min_positive = f64::min_positive_value();
    /// 
    /// assert_eq!(min_positive, f64::MIN_POSITIVE);
    /// ```
    fn min_positive_value() -> Self;

    /// Returns a representation of the constant `π`.
    fn pi() -> Self;

    /// Returns a representation of the constant `2π`.
    fn two_pi() -> Self;

    /// Returns a representation of the constant `π / 2`.
    fn frac_pi_2() -> Self;

    /// Returns a representation of the constant `π / 3`.
    fn frac_pi_3() -> Self;

    /// Returns a representation of the constant `π / 4`.
    fn frac_pi_4() -> Self;

    /// Returns a representation of the constant `π / 6`.
    fn frac_pi_6() -> Self;

    /// Returns a representation of the constant `π / 8`.
    fn frac_pi_8() -> Self;

    /// Returns a representation of the constant `1 / π`.
    fn frac_1_pi() -> Self;

    /// Returns a representation of the constant `2 / π`.
    fn frac_2_pi() -> Self;

    /// Returns a representation of the constant `2 / sqrt(π)`.
    fn frac_2_sqrt_pi() -> Self;

    /// Returns a representation of the Euler's number `e`.
    fn e() -> Self;

    /// Returns a representation of the constant `log2(e)`.
    fn log2_e() -> Self;

    /// Returns a representation of the constant `log10(e)`.
    fn log10_e() -> Self;

    /// Returns a representation of the constant `ln(2)`.
    fn ln_2() -> Self;

    /// Returns a representation of the constant `ln(10)`.
    fn ln_10() -> Self;

    /// Returns a representation of the constant `sqrt(2)`.
    fn sqrt_2() -> Self;

    /// Returns a representation of the constant `1 / sqrt(2)`.
    fn frac_1_sqrt_2() -> Self;

    /// Returns the `NaN` value for a floating point data type.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// #
    /// let nan = f64::nan();
    /// 
    /// assert!(nan.is_nan());
    /// 
    /// let infinity = f64::INFINITY;
    /// let neg_infinity = f64::NEG_INFINITY;
    /// let e = f64::e();
    /// 
    /// assert!(!infinity.is_nan());
    /// assert!(!neg_infinity.is_nan());
    /// assert!(!e.is_nan());
    /// ```
    fn nan() -> Self;

    /// Returns the positive infinite value for a floating point data type.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarBounded,
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let infinity = f64::infinity();
    /// let max_value = f64::max_value();
    ///
    /// assert!(infinity.is_infinite());
    /// assert!(!infinity.is_finite());
    /// assert!(infinity > max_value);
    /// ```
    fn infinity() -> Self;

    /// Returns the negative infinite value for a floating point data type.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarBounded,
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let neg_infinity = f64::neg_infinity();
    /// let min_value = f64::min_value();
    /// 
    /// assert!(neg_infinity.is_infinite());
    /// assert!(!neg_infinity.is_finite());
    /// assert!(neg_infinity < min_value);
    /// ```
    fn neg_infinity() -> Self;

    /// Returns a representation of `-0.0`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// let neg_infinity = f64::neg_infinity();
    /// let zero = 0_f64;
    /// let neg_zero = f64::neg_zero();
    /// 
    /// assert_eq!(zero, neg_zero);
    /// assert_eq!(7_f64 / neg_infinity, zero);
    /// assert_eq!(neg_zero * 10_f64, zero);
    /// ```
    fn neg_zero() -> Self;

    /// Returns the machine epsilon for a floating point data type.
    /// 
    /// The machine epsilon (or machine precision) is an upper bound on the 
    /// relative approximation error due to rounding in floating point arithmetic.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     SimdScalarFloat,
    /// # };
    /// # use core::f64;
    /// #
    /// assert_eq!(f64::machine_epsilon(), f64::EPSILON);
    /// ```
    fn machine_epsilon() -> Self;
}

impl<T> SimdScalar for T 
where 
    T: Copy
     + Clone 
     + fmt::Debug 
     + fmt::Display
     + Num 
     + NumCast 
     + PartialOrd 
     + ops::AddAssign 
     + ops::SubAssign 
     + ops::MulAssign 
     + ops::DivAssign 
     + ops::RemAssign 
{ 
}

macro_rules! impl_simd_scalar_signed_ord_integer {
    ($($ScalarType:ty),* $(,)*) => {$(
        impl SimdScalarSigned for $ScalarType {
            #[inline]
            fn is_sign_positive(self) -> bool {
                Signed::is_positive(&self)
            }

            #[inline]
            fn is_sign_negative(self) -> bool {
                Signed::is_negative(&self)
            }

            #[inline]
            fn copysign(self, sign: Self) -> Self {
                if sign >= 0 {
                    self.abs()
                } else {
                    -self.abs()
                }
            }

            #[inline]
            fn signum(self) -> Self {
                Signed::signum(&self)
            }

            #[inline]
            fn abs(self) -> Self {
                Signed::abs(&self)
            }
        }

        impl SimdScalarOrd for $ScalarType {
            #[inline]
            fn max(self, other: Self) -> Self {
                core::cmp::Ord::max(self, other)
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                core::cmp::Ord::min(self, other)
            }

            #[inline]
            fn clamp(self, min_value: Self, max_value: Self) -> Self {
                core::cmp::Ord::clamp(self, min_value, max_value)
            }
        }
    )*}
}

impl_simd_scalar_signed_ord_integer!(i8, i16, i32, i64, i128, isize);


macro_rules! impl_simd_scalar_signed_ord_float {
    ($($ScalarType:ty),* $(,)*) => {$(
        impl SimdScalarSigned for $ScalarType {
            #[inline]
            fn is_sign_positive(self) -> bool {
                num_traits::Float::is_sign_positive(self)
            }
            
            #[inline]
            fn is_sign_negative(self) -> bool {
                num_traits::Float::is_sign_negative(self)
            }

            #[inline]
            fn copysign(self, sign: Self) -> Self {
                num_traits::Float::copysign(self, sign)
            }

            #[inline]
            fn signum(self) -> Self {
                num_traits::Float::signum(self)
            }

            #[inline]
            fn abs(self) -> Self {
                num_traits::Float::abs(self)
            }
        }

        impl SimdScalarOrd for $ScalarType {
            #[inline]
            fn max(self, other: Self) -> Self {
                num_traits::Float::max(self, other)
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                num_traits::Float::min(self, other)
            }

            #[inline]
            fn clamp(self, min_value: Self, max_value: Self) -> Self {
                if self < min_value {
                    min_value
                } else if self > max_value {
                    max_value
                } else {
                    self
                }
            }
        }
    )*}
}

impl_simd_scalar_signed_ord_float!(f32, f64);


macro_rules! impl_simd_scalar_bounded {
    ($ScalarType:ty, $min_value:expr, $max_value:expr) => {
        impl SimdScalarBounded for $ScalarType {
            #[inline]
            fn min_value() -> Self {
                $min_value
            }

            #[inline]
            fn max_value() -> Self {
                $max_value
            }
        }
    }
}

impl_simd_scalar_bounded!(i8,    i8::MIN,    i8::MAX);
impl_simd_scalar_bounded!(i16,   i16::MIN,   i16::MAX);
impl_simd_scalar_bounded!(i32,   i32::MIN,   i32::MAX);
impl_simd_scalar_bounded!(i64,   i64::MIN,   i64::MAX);
impl_simd_scalar_bounded!(i128,  i128::MIN,  i128::MAX);
impl_simd_scalar_bounded!(isize, isize::MIN, isize::MAX);

impl_simd_scalar_bounded!(f32, f32::MIN, f32::MAX);
impl_simd_scalar_bounded!(f64, f64::MIN, f64::MAX);


macro_rules! impl_simd_scalar_float {
    ($($ScalarType:ty),* $(,)*) => {$(
        impl SimdScalarFloat for $ScalarType {
            #[inline]
            fn floor(self) -> Self {
                num_traits::Float::floor(self)
            }

            #[inline]
            fn ceil(self) -> Self {
                num_traits::Float::ceil(self)
            }

            #[inline]
            fn round(self) -> Self {
                num_traits::Float::round(self)
            }

            #[inline]
            fn trunc(self) -> Self {
                num_traits::Float::trunc(self)
            }

            #[inline]
            fn fract(self) -> Self {
                num_traits::Float::fract(self)
            }
    
            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                num_traits::Float::mul_add(self, a, b)
            }

            #[inline]
            fn hypot(self, other: Self) -> Self {
                num_traits::Float::hypot(self, other)
            }

            #[inline]
            fn recip(self) -> Self {
                num_traits::Float::recip(self)
            }
    
            #[inline]
            fn atan2(self, other: Self) -> Self {
                num_traits::Float::atan2(self, other)
            }

            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                num_traits::Float::sin_cos(self)
            }

            #[inline]
            fn sin(self) -> Self {
                num_traits::Float::sin(self)
            }

            #[inline]
            fn cos(self) -> Self {
                num_traits::Float::cos(self)
            }

            #[inline]
            fn tan(self) -> Self {
                num_traits::Float::tan(self)
            }

            #[inline]
            fn asin(self) -> Self {
                num_traits::Float::asin(self)
            }

            #[inline]
            fn acos(self) -> Self {
                num_traits::Float::acos(self)
            }

            #[inline]
            fn atan(self) -> Self {
                num_traits::Float::atan(self)
            }
    
            #[inline]
            fn sinh(self) -> Self {
                num_traits::Float::sinh(self)
            }

            #[inline]
            fn cosh(self) -> Self {
                num_traits::Float::cosh(self)
            }

            #[inline]
            fn tanh(self) -> Self {
                num_traits::Float::tanh(self)
            }

            #[inline]
            fn asinh(self) -> Self {
                num_traits::Float::asinh(self)
            }

            #[inline]
            fn acosh(self) -> Self {
                num_traits::Float::acosh(self)
            }

            #[inline]
            fn atanh(self) -> Self {
                num_traits::Float::atanh(self)
            }
    
            #[inline]
            fn log(self, base: Self) -> Self {
                num_traits::Float::log(self, base)
            }

            #[inline]
            fn log2(self) -> Self {
                num_traits::Float::log2(self)
            }

            #[inline]
            fn log10(self) -> Self {
                num_traits::Float::log10(self)
            }

            #[inline]
            fn ln(self) -> Self {
                num_traits::Float::ln(self)
            }

            #[inline]
            fn ln_1p(self) -> Self {
                num_traits::Float::ln_1p(self)
            }
        
            #[inline]
            fn sqrt(self) -> Self {
                num_traits::Float::sqrt(self)
            }

            #[inline]
            fn cbrt(self) -> Self {
                num_traits::Float::cbrt(self)
            }
    
            #[inline]
            fn exp(self) -> Self {
                num_traits::Float::exp(self)
            }

            #[inline]
            fn exp2(self) -> Self {
                num_traits::Float::exp2(self)
            }

            #[inline]
            fn exp_m1(self) -> Self {
                num_traits::Float::exp_m1(self)
            }

            #[inline]
            fn powi(self, n: i32) -> Self {
                num_traits::Float::powi(self, n)
            }

            #[inline]
            fn powf(self, n: Self) -> Self {
                num_traits::Float::powf(self, n)
            }
        
            #[inline]
            fn is_finite(self) -> bool {
                num_traits::Float::is_finite(self)
            }

            #[inline]
            fn is_infinite(self) -> bool {
                num_traits::Float::is_infinite(self)
            }

            #[inline]
            fn is_nan(self) -> bool {
                num_traits::Float::is_nan(self)
            }

            #[inline]
            fn is_normal(self) -> bool {
                num_traits::Float::is_normal(self)
            }

            #[inline]
            fn is_subnormal(self) -> bool {
                num_traits::Float::is_subnormal(self)
            }

            #[inline]
            fn min_positive_value() -> Self {
                num_traits::Float::min_positive_value()
            }

            #[inline]
            fn pi() -> Self {
                num_traits::FloatConst::PI()
            }

            #[inline]
            fn two_pi() -> Self {
                num_traits::FloatConst::TAU()
            }

            #[inline]
            fn frac_pi_2() -> Self {
                num_traits::FloatConst::FRAC_PI_2()
            }

            #[inline]
            fn frac_pi_3() -> Self {
                num_traits::FloatConst::FRAC_PI_3()
            }

            #[inline]
            fn frac_pi_4() -> Self {
                num_traits::FloatConst::FRAC_PI_4()
            }

            #[inline]
            fn frac_pi_6() -> Self {
                num_traits::FloatConst::FRAC_PI_6()
            }

            #[inline]
            fn frac_pi_8() -> Self {
                num_traits::FloatConst::FRAC_PI_8()
            }

            #[inline]
            fn frac_1_pi() -> Self {
                num_traits::FloatConst::FRAC_1_PI()
            }

            #[inline]
            fn frac_2_pi() -> Self {
                num_traits::FloatConst::FRAC_2_PI()
            }

            #[inline]
            fn frac_2_sqrt_pi() -> Self {
                num_traits::FloatConst::FRAC_2_SQRT_PI()
            }

            #[inline]
            fn e() -> Self {
                num_traits::FloatConst::E()
            }

            #[inline]
            fn log2_e() -> Self {
                num_traits::FloatConst::LOG2_E()
            }

            #[inline]
            fn log10_e() -> Self {
                num_traits::FloatConst::LOG10_E()
            }

            #[inline]
            fn ln_2() -> Self {
                num_traits::FloatConst::LN_2()
            }

            #[inline]
            fn ln_10() -> Self {
                num_traits::FloatConst::LN_10()
            }

            #[inline]
            fn sqrt_2() -> Self {
                num_traits::FloatConst::SQRT_2()
            }

            #[inline]
            fn frac_1_sqrt_2() -> Self {
                num_traits::FloatConst::FRAC_1_SQRT_2()
            }

            #[inline]
            fn nan() -> Self {
                num_traits::Float::nan()
            }

            #[inline]
            fn infinity() -> Self {
                num_traits::Float::infinity()
            }

            #[inline]
            fn neg_infinity() -> Self {
                num_traits::Float::neg_infinity()
            }

            #[inline]
            fn neg_zero() -> Self {
                num_traits::Float::neg_zero()
            }

            #[inline]
            fn machine_epsilon() -> Self {
                num_traits::Float::epsilon()
            }
        }
    )*}
}

impl_simd_scalar_float!(f32, f64);

