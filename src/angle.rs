use crate::scalar::{
    Scalar,
    ScalarSigned,
    ScalarFloat,
};
use crate::traits::{
    Zero,
};

use num_traits::{
    Float,
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
pub trait Angle where 
    Self: Copy + Clone,
    Self: PartialEq + PartialOrd,
    Self: Zero,
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
    type Dimensionless: ScalarFloat;

    /// The value of a full rotation around the unit circle for a typed angle.
    fn full_turn() -> Self;

    /// Compute the sine of a typed angle.
    fn sin(self) -> Self::Dimensionless;

    /// Compute the cosine of a typed angle.
    fn cos(self) -> Self::Dimensionless;

    /// Compute the tangent of a typed angle.
    fn tan(self) -> Self::Dimensionless;

    /// Compute the arc sine of a scalar value, returning a typed angle.
    fn asin(ratio: Self::Dimensionless) -> Self;

    /// Compute the arc cosine of a scalar value, returning a typed angle.
    fn acos(ratio: Self::Dimensionless) -> Self;

    /// Compute the arc tangent of a scalar value, returning a typed angle.
    fn atan(ratio: Self::Dimensionless) -> Self;

    /// Compute the four quadrant arc tangent of two arguments, returning a 
    /// typed angle.
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
    fn atan2(y: Self::Dimensionless, x: Self::Dimensionless) -> Self;

    /// Simultaneously compute the sine and cosine of an angle.
    #[inline]
    fn sin_cos(self) -> (Self::Dimensionless, Self::Dimensionless) {
        (Self::sin(self), Self::cos(self))
    }

    /// The value of half of a full turn around the unit circle.
    #[inline]
    fn full_turn_div_2() -> Self {
        let denominator: Self::Dimensionless = num_traits::cast(2).unwrap();
        Self::full_turn() / denominator
    }

    /// The value of a one fourth of a full turn around the unit circle.
    #[inline]
    fn full_turn_div_4() -> Self {
        let denominator: Self::Dimensionless = num_traits::cast(4).unwrap();
        Self::full_turn() / denominator
    }

    /// The value of one sixth of a full turn around the unit circle.
    #[inline]
    fn full_turn_div_6() -> Self {
        let denominator: Self::Dimensionless = num_traits::cast(6).unwrap();
        Self::full_turn() / denominator
    }

    /// The value of one eighth of a full turn around the unit circle.
    #[inline]
    fn full_turn_div_8() -> Self {
        let denominator: Self::Dimensionless = num_traits::cast(8).unwrap();
        Self::full_turn() / denominator
    }

    /// Map an angle to its smallest congruent angle in the range `[0, full_turn)`.
    #[inline]
    fn normalize(self) -> Self {
        let remainder = self % Self::full_turn();
        if remainder < Self::zero() {
            remainder + Self::full_turn()
        } else {
            remainder
        }
    }

    /// Map an angle to its smallest congruent angle in the 
    /// range `[-full_turn / 2, full_turn / 2)`.
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
    #[inline]
    fn opposite(self) -> Self {
        Self::normalize(self + Self::full_turn_div_2())
    }

    /// Compute the interior bisector (the angle that is half-way between the angles) 
    /// of `self` and `other`.
    #[inline]
    fn bisect(self, other: Self) -> Self {
        let one_half = num_traits::cast(0.5_f64).unwrap();
        Self::normalize((other - self) * one_half + self)
    }

    /// Compute the cosecant of a typed angle.
    #[inline]
    fn csc(self) -> Self::Dimensionless {
        Self::sin(self).recip()
    }

    /// Compute the cotangent of a typed angle.
    #[inline]
    fn cot(self) -> Self::Dimensionless {
        Self::tan(self).recip()
    }

    /// Compute the secant of a typed angle.
    #[inline]
    fn sec(self) -> Self::Dimensionless {
        Self::cos(self).recip()
    }
}

/// The angle (arc length) along the unit circle in units of radians.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Hash)]
pub struct Radians<S>(pub S);

impl<S> Radians<S> where S: ScalarFloat {
    /// Returns `true` if the underlying floating point number of the typed
    /// angle is finite.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Angle,
    /// #     Radians,  
    /// # };
    /// # use core::f64;
    /// #
    /// let angle: Radians<f64> = Radians(f64::consts::PI / 4_f64);
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

impl<S> Degrees<S> where S: ScalarFloat {
    /// Returns `true` if the underlying floating point number of the typed
    /// angle is finite.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Angle,
    /// #     Degrees,  
    /// # };
    /// #
    /// let angle: Degrees<f64> = Degrees(45.0);
    /// assert!(angle.is_finite()); 
    /// ```
    #[inline]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

impl<S> From<Degrees<S>> for Radians<S> where S: ScalarFloat {
    #[inline]
    fn from(degrees: Degrees<S>) -> Radians<S> {
        Radians(degrees.0 * num_traits::cast(f64::consts::PI / 180_f64).unwrap())
    }
}

impl<S> From<Radians<S>> for Degrees<S> where S: ScalarFloat {
    #[inline]
    fn from(radians: Radians<S>) -> Degrees<S> {
        Degrees(radians.0 * num_traits::cast(180_f64 / f64::consts::PI).unwrap())
    }
}

impl<S> fmt::Display for Degrees<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{} degrees", self.0)
    }
}

impl<S> fmt::Display for Radians<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{} radians", self.0)
    }
}

impl<S> ops::Add<Degrees<S>> for Degrees<S> where S: Scalar {
    type Output = Degrees<S>;

    #[inline]
    fn add(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<'a, S> ops::Add<&'a Degrees<S>> for Degrees<S> where S: Scalar {
    type Output = Degrees<S>;

    #[inline]
    fn add(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<'a, S> ops::Add<Degrees<S>> for &'a Degrees<S> where S: Scalar{
    type Output = Degrees<S>;

    #[inline]
    fn add(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<'a, 'b, S> ops::Add<&'a Degrees<S>> for &'b Degrees<S> where S: Scalar {
    type Output = Degrees<S>;

    #[inline]
    fn add(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<S> ops::Sub<Degrees<S>> for Degrees<S> where S: Scalar {
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 - other.0)
    } 
}

impl<'a, S> ops::Sub<&'a Degrees<S>> for Degrees<S> where S: Scalar {
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 - other.0)
    } 
}

impl<'a, S> ops::Sub<Degrees<S>> for &'a Degrees<S> where S: Scalar {
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 - other.0)
    } 
}

impl<'a, 'b, S> ops::Sub<&'a Degrees<S>> for &'b Degrees<S> where S: Scalar {
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 - other.0)
    } 
}

impl<S> ops::Mul<S> for Degrees<S> where S: Scalar {
    type Output = Degrees<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Degrees(self.0 * other)
    }
}

impl<'a, S> ops::Mul<S> for &'a Degrees<S> where S: Scalar {
    type Output = Degrees<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Degrees(self.0 * other)
    }
}

impl<S> ops::Div<S> for Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Degrees(self.0 / other)
    }
}

impl<'a, S> ops::Div<S> for &'a Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Degrees(self.0 / other)
    }
}

impl<S> ops::Div<Degrees<S>> for Degrees<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn div(self, other: Degrees<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<'a, S> ops::Div<&'a Degrees<S>> for Degrees<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn div(self, other: &'a Degrees<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<'a, S> ops::Div<Degrees<S>> for &'a Degrees<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn div(self, other: Degrees<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<'a, 'b, S> ops::Div<&'a Degrees<S>> for &'b Degrees<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn div(self, other: &'a Degrees<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<S> ops::Rem<Degrees<S>> for Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn rem(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 % other.0)
    }
}

impl<'a, S> ops::Rem<&'a Degrees<S>> for Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn rem(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 % other.0)
    }
}

impl<'a, S> ops::Rem<Degrees<S>> for &'a Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn rem(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 % other.0)
    }
}

impl<'a, 'b, S> ops::Rem<&'a Degrees<S>> for &'b Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn rem(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 % other.0)
    }
}

impl<S> ops::Neg for Degrees<S> where S: ScalarSigned {
    type Output = Degrees<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Degrees(-self.0)
    }
}

impl<'a, S> ops::Neg for &'a Degrees<S> where S: ScalarSigned {
    type Output = Degrees<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Degrees(-self.0)
    }
}

impl<S> ops::AddAssign<Degrees<S>> for Degrees<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: Degrees<S>) {
        *self = *self + other;
    } 
}

impl<S> ops::SubAssign<Degrees<S>> for Degrees<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: Degrees<S>) {
        *self = *self - other;
    } 
}

impl<S> ops::MulAssign<S> for Degrees<S> where S: Scalar {
    #[inline]
    fn mul_assign(&mut self, other: S) {
        *self = *self * other;
    } 
}

impl<S> ops::DivAssign<S> for Degrees<S> where S: ScalarFloat {
    #[inline]
    fn div_assign(&mut self, other: S) {
        *self = *self / other;
    } 
}

impl<S> ops::RemAssign<Degrees<S>> for Degrees<S> where S: ScalarFloat {
    #[inline]
    fn rem_assign(&mut self, other: Degrees<S>) {
        *self = *self % other;
    } 
}

impl<S> Zero for Degrees<S> where S: Scalar {
    #[inline]
    fn zero() -> Degrees<S> {
        Degrees(S::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<S> approx::AbsDiffEq for Degrees<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Degrees<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.0, &other.0, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Degrees<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.0, &other.0, epsilon, max_ulps)
    }
}

impl<S> ops::Add<Radians<S>> for Radians<S> where S: Scalar {
    type Output = Radians<S>;

    #[inline]
    fn add(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 + other.0)
    } 
}

impl<'a, S> ops::Add<&'a Radians<S>> for Radians<S> where S: Scalar {
    type Output = Radians<S>;

    #[inline]
    fn add(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 + other.0)
    } 
}

impl<'a, S> ops::Add<Radians<S>> for &'a Radians<S> where S: Scalar {
    type Output = Radians<S>;

    #[inline]
    fn add(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 + other.0)
    } 
}

impl<'a, 'b, S> ops::Add<&'a Radians<S>> for &'b Radians<S> where S: Scalar {
    type Output = Radians<S>;

    #[inline]
    fn add(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 + other.0)
    } 
}

impl<S> ops::Sub<Radians<S>> for Radians<S> where S: Scalar {
    type Output = Radians<S>;

    #[inline]
    fn sub(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 - other.0)
    } 
}

impl<'a, S> ops::Sub<&'a Radians<S>> for Radians<S> where S: Scalar {
    type Output = Radians<S>;

    #[inline]
    fn sub(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 - other.0)
    } 
}

impl<'a, S> ops::Sub<Radians<S>> for &'a Radians<S> where S: Scalar {
    type Output = Radians<S>;

    #[inline]
    fn sub(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 - other.0)
    } 
}

impl<'a, 'b, S> ops::Sub<&'a Radians<S>> for &'b Radians<S> where S: Scalar {
    type Output = Radians<S>;

    #[inline]
    fn sub(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 - other.0)
    } 
}

impl<S> ops::Mul<S> for Radians<S> where S: Scalar {
    type Output = Radians<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Radians(self.0 * other)
    }
}

impl<'a, S> ops::Mul<S> for &'a Radians<S> where S: Scalar {
    type Output = Radians<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Radians(self.0 * other)
    }
}

impl<S> ops::Div<S> for Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Radians(self.0 / other)
    }
}

impl<'a, S> ops::Div<S> for &'a Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Radians(self.0 / other)
    }
}

impl<S> ops::Div<Radians<S>> for Radians<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn div(self, other: Radians<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<'a, S> ops::Div<&'a Radians<S>> for Radians<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn div(self, other: &'a Radians<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<'a, S> ops::Div<Radians<S>> for &'a Radians<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn div(self, other: Radians<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<'a, 'b, S> ops::Div<&'a Radians<S>> for &'b Radians<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn div(self, other: &'a Radians<S>) -> Self::Output {
        self.0 / other.0
    }
}

impl<S> ops::Rem<Radians<S>> for Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn rem(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 % other.0)
    }
}

impl<'a, S> ops::Rem<&'a Radians<S>> for Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn rem(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 % other.0)
    }
}

impl<'a, S> ops::Rem<Radians<S>> for &'a Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn rem(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 % other.0)
    }
}

impl<'a, 'b, S> ops::Rem<&'a Radians<S>> for &'b Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn rem(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 % other.0)
    }
}

impl<S> ops::Neg for Radians<S> where S: ScalarSigned {
    type Output = Radians<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Radians(-self.0)
    }
}

impl<'a, S> ops::Neg for &'a Radians<S> where S: ScalarSigned {
    type Output = Radians<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Radians(-self.0)
    }
}

impl<S> ops::AddAssign<Radians<S>> for Radians<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: Radians<S>) {
        self.0 += other.0;
    } 
}

impl<S> ops::SubAssign<Radians<S>> for Radians<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: Radians<S>) {
        self.0 -= other.0;
    } 
}

impl<S> ops::MulAssign<S> for Radians<S> where S: Scalar {
    #[inline]
    fn mul_assign(&mut self, other: S) {
        self.0 *= other;
    } 
}

impl<S> ops::DivAssign<S> for Radians<S> where S: ScalarFloat {
    #[inline]
    fn div_assign(&mut self, other: S) {
        self.0 /= other;
    } 
}

impl<S> ops::RemAssign<Radians<S>> for Radians<S> where S: ScalarFloat {
    #[inline]
    fn rem_assign(&mut self, other: Radians<S>) {
        self.0 = self.0 % other.0;
    } 
}

impl<S> Zero for Radians<S> where S: Scalar {
    #[inline]
    fn zero() -> Radians<S> {
        Radians(S::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<S> approx::AbsDiffEq for Radians<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Radians<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.0, &other.0, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Radians<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.0, &other.0, epsilon, max_ulps)
    }
}

impl<S> Angle for Radians<S> where S: ScalarFloat {
    type Dimensionless = S;

    #[inline]
    fn full_turn() -> Self {
        Radians(num_traits::cast(2_f64 * f64::consts::PI).unwrap())
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
        Radians(Self::Dimensionless::asin(ratio))
    }

    #[inline]
    fn acos(ratio: Self::Dimensionless) -> Self {
        Radians(Self::Dimensionless::acos(ratio))
    }

    #[inline]
    fn atan(ratio: Self::Dimensionless) -> Self {
        Radians(Self::Dimensionless::atan(ratio))
    }

    #[inline]
    fn atan2(a: Self::Dimensionless, b: Self::Dimensionless) -> Self {
        Radians(Self::Dimensionless::atan2(a, b))
    }
}

impl<S> Angle for Degrees<S> where S: ScalarFloat {
    type Dimensionless = S;

    #[inline]
    fn full_turn() -> Self {
        Degrees(num_traits::cast(360).unwrap())
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

