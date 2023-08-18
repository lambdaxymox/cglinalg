use num_traits::{
    Num, 
    NumCast,
    Signed,
};
use core::fmt::{
    Debug,
    Display,
};
use core::ops::{
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    Neg,
};


/// A data type with this trait has the properties of a 
/// set of scalar numbers underlying vector and matrix 
/// data types.
pub trait SimdScalar 
where
    Self: Copy,
    Self: Clone,
    Self: Debug,
    Self: Display,
    Self: Num,
    Self: NumCast,
    Self: PartialOrd, 
    Self: AddAssign,
    Self: SubAssign,
    Self: MulAssign,
    Self: DivAssign,
    Self: RemAssign,
{
}

/// Scalar numbers with a notion of subtraction and have additive 
/// inverses. 
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

pub trait SimdScalarBounded
where
    Self: SimdScalar + SimdScalarOrd
{
    /// Returns the smallest finite value of a number type.
    /// 
    /// # Example
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

/// Scalar numbers that have the properties of finite precision
/// floating point arithmetic.
pub trait SimdScalarFloat:
      SimdScalarSigned + SimdScalarOrd
    + approx::AbsDiffEq<Epsilon = Self>
    + approx::RelativeEq<Epsilon = Self>
    + approx::UlpsEq<Epsilon = Self>
    + Neg<Output = Self> 
{    
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    
    
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn hypot(self, other: Self) -> Self;
    fn recip(self) -> Self;
    
    fn atan2(self, other: Self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
    
    fn sinh_cosh(self) -> (Self, Self) {
        (SimdScalarFloat::sinh(self), SimdScalarFloat::cosh(self))
    }
    
    fn log(self, base: Self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn ln(self) -> Self;
    fn ln_1p(self) -> Self;
    
    /// Calculate the square root of a floating point number.
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

    /// Calculate the sube root of a number.
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
    
    
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn exp_m1(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, n: Self) -> Self;
        
    
    fn is_finite(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_nan(self) -> bool;

    
    fn pi() -> Self;
    fn two_pi() -> Self;
    fn frac_pi_2() -> Self;
    fn frac_pi_3() -> Self;
    fn frac_pi_4() -> Self;
    fn frac_pi_6() -> Self;
    fn frac_pi_8() -> Self;
    fn frac_1_pi() -> Self;
    fn frac_2_pi() -> Self;
    fn frac_2_sqrt_pi() -> Self;
    fn e() -> Self;
    fn log2_e() -> Self;
    fn log10_e() -> Self;
    fn ln_2() -> Self;
    fn ln_10() -> Self;
    fn sqrt_2() -> Self;

}

impl<T> SimdScalar for T 
where 
    T: Copy
     + Clone 
     + Debug 
     + Display
     + Num 
     + NumCast 
     + PartialOrd 
     + AddAssign 
     + SubAssign 
     + MulAssign 
     + DivAssign 
     + RemAssign 
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
        }
    )*}
}

impl_simd_scalar_float!(f32, f64);

