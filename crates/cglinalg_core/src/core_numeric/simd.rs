use num_traits::{
    Num, 
    NumCast,
    Signed,
};
use core::fmt::{
    Debug,
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
    fn is_sign_positive(self) -> bool;
    fn is_sign_negative(self) -> bool;
    fn copysign(self, sign: Self) -> Self;
    fn signum(self) -> Self;
    fn abs(self) -> Self;
}

pub trait SimdScalarOrd
where
    Self: SimdScalar + PartialOrd
{
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn clamp(self, min_value: Self, max_value: Self) -> Self;
}

pub trait SimdScalarBounded
where
    Self: SimdScalar + SimdScalarOrd
{
    fn min_value() -> Self;
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
        (<Self as SimdScalarFloat>::sinh(self), <Self as SimdScalarFloat>::cosh(self))
    }
    
    fn log(self, base: Self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn ln(self) -> Self;
    fn ln_1p(self) -> Self;
        
    fn sqrt(self) -> Self;
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
                <Self as Signed>::is_positive(&self)
            }

            #[inline]
            fn is_sign_negative(self) -> bool {
                <Self as Signed>::is_negative(&self)
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
                <Self as Signed>::signum(&self)
            }

            #[inline]
            fn abs(self) -> Self {
                <Self as Signed>::abs(&self)
            }
        }

        impl SimdScalarOrd for $ScalarType {
            #[inline]
            fn max(self, other: Self) -> Self {
                <Self as core::cmp::Ord>::max(self, other)
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                <Self as core::cmp::Ord>::min(self, other)
            }

            #[inline]
            fn clamp(self, min_value: Self, max_value: Self) -> Self {
                <Self as core::cmp::Ord>::clamp(self, min_value, max_value)
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
                <Self as num_traits::Float>::is_sign_positive(self)
            }
            
            #[inline]
            fn is_sign_negative(self) -> bool {
                <Self as num_traits::Float>::is_sign_negative(self)
            }

            #[inline]
            fn copysign(self, sign: Self) -> Self {
                <Self as num_traits::Float>::copysign(self, sign)
            }

            #[inline]
            fn signum(self) -> Self {
                <Self as num_traits::Float>::signum(self)
            }

            #[inline]
            fn abs(self) -> Self {
                <Self as num_traits::Float>::abs(self)
            }
        }

        impl SimdScalarOrd for $ScalarType {
            #[inline]
            fn max(self, other: Self) -> Self {
                <Self as num_traits::Float>::max(self, other)
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                <Self as num_traits::Float>::min(self, other)
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
                <Self as num_traits::Float>::floor(self)
            }

            #[inline]
            fn ceil(self) -> Self {
                <Self as num_traits::Float>::ceil(self)
            }

            #[inline]
            fn round(self) -> Self {
                <Self as num_traits::Float>::round(self)
            }

            #[inline]
            fn trunc(self) -> Self {
                <Self as num_traits::Float>::trunc(self)
            }

            #[inline]
            fn fract(self) -> Self {
                <Self as num_traits::Float>::fract(self)
            }
    
            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                <Self as num_traits::Float>::mul_add(self, a, b)
            }

            #[inline]
            fn hypot(self, other: Self) -> Self {
                <Self as num_traits::Float>::hypot(self, other)
            }

            #[inline]
            fn recip(self) -> Self {
                <Self as num_traits::Float>::recip(self)
            }
    
            #[inline]
            fn atan2(self, other: Self) -> Self {
                <Self as num_traits::Float>::atan2(self, other)
            }

            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                <Self as num_traits::Float>::sin_cos(self)
            }

            #[inline]
            fn sin(self) -> Self {
                <Self as num_traits::Float>::sin(self)
            }

            #[inline]
            fn cos(self) -> Self {
                <Self as num_traits::Float>::cos(self)
            }

            #[inline]
            fn tan(self) -> Self {
                <Self as num_traits::Float>::tan(self)
            }

            #[inline]
            fn asin(self) -> Self {
                <Self as num_traits::Float>::asin(self)
            }

            #[inline]
            fn acos(self) -> Self {
                <Self as num_traits::Float>::acos(self)
            }

            #[inline]
            fn atan(self) -> Self {
                <Self as num_traits::Float>::atan(self)
            }
    
            #[inline]
            fn sinh(self) -> Self {
                <Self as num_traits::Float>::sinh(self)
            }

            #[inline]
            fn cosh(self) -> Self {
                <Self as num_traits::Float>::cosh(self)
            }

            #[inline]
            fn tanh(self) -> Self {
                <Self as num_traits::Float>::tanh(self)
            }

            #[inline]
            fn asinh(self) -> Self {
                <Self as num_traits::Float>::asinh(self)
            }

            #[inline]
            fn acosh(self) -> Self {
                <Self as num_traits::Float>::acosh(self)
            }

            #[inline]
            fn atanh(self) -> Self {
                <Self as num_traits::Float>::atanh(self)
            }
    
            #[inline]
            fn log(self, base: Self) -> Self {
                <Self as num_traits::Float>::log(self, base)
            }

            #[inline]
            fn log2(self) -> Self {
                <Self as num_traits::Float>::log2(self)
            }

            #[inline]
            fn log10(self) -> Self {
                <Self as num_traits::Float>::log10(self)
            }

            #[inline]
            fn ln(self) -> Self {
                <Self as num_traits::Float>::ln(self)
            }

            #[inline]
            fn ln_1p(self) -> Self {
                <Self as num_traits::Float>::ln_1p(self)
            }
        
            #[inline]
            fn sqrt(self) -> Self {
                <Self as num_traits::Float>::sqrt(self)
            }

            #[inline]
            fn cbrt(self) -> Self {
                <Self as num_traits::Float>::cbrt(self)
            }
    
            #[inline]
            fn exp(self) -> Self {
                <Self as num_traits::Float>::exp(self)
            }

            #[inline]
            fn exp2(self) -> Self {
                <Self as num_traits::Float>::exp2(self)
            }

            #[inline]
            fn exp_m1(self) -> Self {
                <Self as num_traits::Float>::exp_m1(self)
            }

            #[inline]
            fn powi(self, n: i32) -> Self {
                <Self as num_traits::Float>::powi(self, n)
            }

            #[inline]
            fn powf(self, n: Self) -> Self {
                <Self as num_traits::Float>::powf(self, n)
            }
        
            #[inline]
            fn is_finite(self) -> bool {
                <Self as num_traits::Float>::is_finite(self)
            }

            #[inline]
            fn is_infinite(self) -> bool {
                <Self as num_traits::Float>::is_infinite(self)
            }

            #[inline]
            fn is_nan(self) -> bool {
                <Self as num_traits::Float>::is_nan(self)
            }


            #[inline]
            fn pi() -> Self {
                <Self as num_traits::FloatConst>::PI()
            }

            #[inline]
            fn two_pi() -> Self {
                <Self as num_traits::FloatConst>::TAU()
            }

            #[inline]
            fn frac_pi_2() -> Self {
                <Self as num_traits::FloatConst>::FRAC_PI_2()
            }

            #[inline]
            fn frac_pi_3() -> Self {
                <Self as num_traits::FloatConst>::FRAC_PI_3()
            }

            #[inline]
            fn frac_pi_4() -> Self {
                <Self as num_traits::FloatConst>::FRAC_PI_4()
            }

            #[inline]
            fn frac_pi_6() -> Self {
                <Self as num_traits::FloatConst>::FRAC_PI_6()
            }

            #[inline]
            fn frac_pi_8() -> Self {
                <Self as num_traits::FloatConst>::FRAC_PI_8()
            }

            #[inline]
            fn frac_1_pi() -> Self {
                <Self as num_traits::FloatConst>::FRAC_1_PI()
            }

            #[inline]
            fn frac_2_pi() -> Self {
                <Self as num_traits::FloatConst>::FRAC_2_PI()
            }

            #[inline]
            fn frac_2_sqrt_pi() -> Self {
                <Self as num_traits::FloatConst>::FRAC_2_SQRT_PI()
            }

            #[inline]
            fn e() -> Self {
                <Self as num_traits::FloatConst>::E()
            }

            #[inline]
            fn log2_e() -> Self {
                <Self as num_traits::FloatConst>::LOG2_E()
            }

            #[inline]
            fn log10_e() -> Self {
                <Self as num_traits::FloatConst>::LOG10_E()
            }

            #[inline]
            fn ln_2() -> Self {
                <Self as num_traits::FloatConst>::LN_2()
            }

            #[inline]
            fn ln_10() -> Self {
                <Self as num_traits::FloatConst>::LN_10()
            }

            #[inline]
            fn sqrt_2() -> Self {
                <Self as num_traits::FloatConst>::SQRT_2()
            }
        }
    )*}
}

impl_simd_scalar_float!(f32, f64);

