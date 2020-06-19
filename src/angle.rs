use base::ScalarFloat;
use structure::{
    Angle,
    Zero,
};
use std::f64;
use std::fmt;
use std::ops;


#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
#[repr(C)]
pub struct Radians<S>(S);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
#[repr(C)]
pub struct Degrees<S>(S);


impl<S> From<Degrees<S>> for Radians<S> where S: ScalarFloat {
    #[inline]
    fn from(degrees: Degrees<S>) -> Radians<S> {
        Radians(degrees.0 * num_traits::cast(f64::consts::PI / 180_f64).unwrap())
    }
}

impl<S> From<Radians<S>> for Degrees<S> where S: ScalarFloat {
    fn from(radians: Radians<S>) -> Degrees<S> {
        Degrees(radians.0 * num_traits::cast(180_f64 / f64::consts::PI).unwrap())
    }
}

impl<S> fmt::Display for Degrees<S> where S: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} degrees", self.0)
    }
}

impl<S> fmt::Display for Radians<S> where S: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} radians", self.0)
    }
}

impl<S> ops::Add<Degrees<S>> for Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn add(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<'a, S> ops::Add<&'a Degrees<S>> for Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn add(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<'a, S> ops::Add<Degrees<S>> for &'a Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn add(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<'a, 'b, S> ops::Add<&'a Degrees<S>> for &'b Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn add(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<S> ops::Sub<Degrees<S>> for Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 - other.0)
    } 
}

impl<'a, S> ops::Sub<&'a Degrees<S>> for Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 - other.0)
    } 
}

impl<'a, S> ops::Sub<Degrees<S>> for &'a Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 - other.0)
    } 
}

impl<'a, 'b, S> ops::Sub<&'a Degrees<S>> for &'b Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 - other.0)
    } 
}

impl<S> ops::Mul<S> for Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Degrees(self.0 * other)
    }
}

impl<'a, S> ops::Mul<S> for &'a Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Degrees(self.0 * other)
    }
}

impl<S> ops::Mul<Degrees<S>> for Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn mul(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 * other.0)
    }
}

impl<'a, S> ops::Mul<&'a Degrees<S>> for Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn mul(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 * other.0)
    }
}

impl<'a, S> ops::Mul<Degrees<S>> for &'a Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn mul(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 * other.0)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Degrees<S>> for &'b Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn mul(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 * other.0)
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
    type Output = Degrees<S>;

    #[inline]
    fn div(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 / other.0)
    }
}

impl<'a, S> ops::Div<&'a Degrees<S>> for Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn div(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 / other.0)
    }
}

impl<'a, S> ops::Div<Degrees<S>> for &'a Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn div(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 / other.0)
    }
}

impl<'a, 'b, S> ops::Div<&'a Degrees<S>> for &'b Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn div(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 / other.0)
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

impl<S> ops::Neg for Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Degrees(-self.0)
    }
}

impl<'a, S> ops::Neg for &'a Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Degrees(-self.0)
    }
}

impl<S> ops::AddAssign<Degrees<S>> for Degrees<S> where S: ScalarFloat {
    #[inline]
    fn add_assign(&mut self, other: Degrees<S>) {
        *self = *self + other;
    } 
}

impl<S> ops::SubAssign<Degrees<S>> for Degrees<S> where S: ScalarFloat {
    #[inline]
    fn sub_assign(&mut self, other: Degrees<S>) {
        *self = *self - other;
    } 
}

impl<S> ops::MulAssign<S> for Degrees<S> where S: ScalarFloat {
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

impl<S> Zero for Degrees<S> where S: ScalarFloat {
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




impl<S> ops::Add<Radians<S>> for Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn add(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 + other.0)
    } 
}

impl<'a, S> ops::Add<&'a Radians<S>> for Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn add(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 + other.0)
    } 
}

impl<'a, S> ops::Add<Radians<S>> for &'a Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn add(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 + other.0)
    } 
}

impl<'a, 'b, S> ops::Add<&'a Radians<S>> for &'b Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn add(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 + other.0)
    } 
}

impl<S> ops::Sub<Radians<S>> for Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn sub(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 - other.0)
    } 
}

impl<'a, S> ops::Sub<&'a Radians<S>> for Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn sub(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 - other.0)
    } 
}

impl<'a, S> ops::Sub<Radians<S>> for &'a Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn sub(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 - other.0)
    } 
}

impl<'a, 'b, S> ops::Sub<&'a Radians<S>> for &'b Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn sub(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 - other.0)
    } 
}

impl<S> ops::Mul<S> for Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Radians(self.0 * other)
    }
}

impl<'a, S> ops::Mul<S> for &'a Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Radians(self.0 * other)
    }
}

impl<S> ops::Mul<Radians<S>> for Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn mul(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 * other.0)
    }
}

impl<'a, S> ops::Mul<&'a Radians<S>> for Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn mul(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 * other.0)
    }
}

impl<'a, S> ops::Mul<Radians<S>> for &'a Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn mul(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 * other.0)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Radians<S>> for &'b Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn mul(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 * other.0)
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
    type Output = Radians<S>;

    #[inline]
    fn div(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 / other.0)
    }
}

impl<'a, S> ops::Div<&'a Radians<S>> for Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn div(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 / other.0)
    }
}

impl<'a, S> ops::Div<Radians<S>> for &'a Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn div(self, other: Radians<S>) -> Self::Output {
        Radians(self.0 / other.0)
    }
}

impl<'a, 'b, S> ops::Div<&'a Radians<S>> for &'b Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn div(self, other: &'a Radians<S>) -> Self::Output {
        Radians(self.0 / other.0)
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

impl<S> ops::Neg for Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Radians(-self.0)
    }
}

impl<'a, S> ops::Neg for &'a Radians<S> where S: ScalarFloat {
    type Output = Radians<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Radians(-self.0)
    }
}

impl<S> ops::AddAssign<Radians<S>> for Radians<S> where S: ScalarFloat {
    #[inline]
    fn add_assign(&mut self, other: Radians<S>) {
        self.0 += other.0;
    } 
}

impl<S> ops::SubAssign<Radians<S>> for Radians<S> where S: ScalarFloat {
    #[inline]
    fn sub_assign(&mut self, other: Radians<S>) {
        self.0 -= other.0;
    } 
}

impl<S> ops::MulAssign<S> for Radians<S> where S: ScalarFloat {
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

impl<S> Zero for Radians<S> where S: ScalarFloat {
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
    type Scalar = S;

    #[inline]
    fn full_turn() -> Self {
        Radians(num_traits::cast(2_f64 * f64::consts::PI).unwrap())
    }

    #[inline]
    fn sin(self) -> Self::Scalar {
        S::sin(self.0)
    }

    #[inline]
    fn cos(self) -> Self::Scalar {
        S::cos(self.0)
    }

    #[inline]
    fn tan(self) -> Self::Scalar {
        S::tan(self.0)
    }

    #[inline]
    fn asin(ratio: Self::Scalar) -> Self {
        Radians(Self::Scalar::asin(ratio))
    }

    #[inline]
    fn acos(ratio: Self::Scalar) -> Self {
        Radians(Self::Scalar::acos(ratio))
    }

    #[inline]
    fn atan(ratio: Self::Scalar) -> Self {
        Radians(Self::Scalar::atan(ratio))
    }

    #[inline]
    fn atan2(a: Self::Scalar, b: Self::Scalar) -> Self {
        Radians(Self::Scalar::atan2(a, b))
    }
}

impl<S> Angle for Degrees<S> where S: ScalarFloat {
    type Scalar = S;

    #[inline]
    fn full_turn() -> Self {
        Degrees(num_traits::cast(360).unwrap())
    }

    #[inline]
    fn sin(self) -> Self::Scalar {
        Radians::from(self).sin()
    }

    #[inline]
    fn cos(self) -> Self::Scalar {
        Radians::from(self).cos()
    }

    #[inline]
    fn tan(self) -> Self::Scalar {
        Radians::from(self).tan()
    }

    #[inline]
    fn asin(ratio: Self::Scalar) -> Self {
        Radians(ratio.asin()).into()
    }

    #[inline]
    fn acos(ratio: Self::Scalar) -> Self {
        Radians(ratio.acos()).into()
    }

    #[inline]
    fn atan(ratio: Self::Scalar) -> Self {
        Radians(ratio.atan()).into()
    }

    #[inline]
    fn atan2(a: Self::Scalar, b: Self::Scalar) -> Self {
        Radians(Self::Scalar::atan2(a, b)).into()
    }
}

#[cfg(test)]
mod conversion_tests {
    use super::{
        Radians,
        Degrees,
    };
    use std::f64;


    struct Test<A, B> {
        input: A,
        expected: B, 
    }

    fn radians_to_degrees_tests() -> Vec<Test<Radians<f64>, Degrees<f64>>> {
        let pi = f64::consts::PI;
        vec![
            Test { input: Radians(0_f64),              expected: Degrees(0_f64)   },
            Test { input: Radians(pi / 4_f64),         expected: Degrees(45_f64)  },
            Test { input: Radians(pi / 2_f64),         expected: Degrees(90_f64)  },
            Test { input: Radians(3_f64 * pi / 4_f64), expected: Degrees(135_f64) },
            Test { input: Radians(pi),                 expected: Degrees(180_f64) },
            Test { input: Radians(5_f64 * pi / 4_f64), expected: Degrees(225_f64) },
            Test { input: Radians(3_f64 * pi / 2_f64), expected: Degrees(270_f64) },
            Test { input: Radians(7_f64 * pi / 4_f64), expected: Degrees(315_f64) },
            Test { input: Radians(2_f64 * pi),         expected: Degrees(360_f64) },
        ]
    }

    fn degrees_to_radians_tests() -> Vec<Test<Degrees<f64>, Radians<f64>>> {
        let pi = f64::consts::PI;
        vec![
            Test { input: Degrees(0_f64),   expected: Radians(0_f64)              },
            Test { input: Degrees(45_f64),  expected: Radians(pi / 4_f64)         },
            Test { input: Degrees(90_f64),  expected: Radians(pi / 2_f64)         },
            Test { input: Degrees(135_f64), expected: Radians(3_f64 * pi / 4_f64) },
            Test { input: Degrees(180_f64), expected: Radians(pi)                 },
            Test { input: Degrees(225_f64), expected: Radians(5_f64 * pi / 4_f64) },
            Test { input: Degrees(270_f64), expected: Radians(3_f64 * pi / 2_f64) },
            Test { input: Degrees(315_f64), expected: Radians(7_f64 * pi / 4_f64) },
            Test { input: Degrees(360_f64), expected: Radians(2_f64 * pi)         },
        ]
    }

    #[test]
    fn convert_radians_to_degrees() {
        radians_to_degrees_tests().iter().for_each(|test| {
            let result = Degrees::from(test.input);
            let expected = test.expected;

            assert_eq!(result, expected);
        })
    }

    #[test]
    fn convert_degrees_to_radians() {
        degrees_to_radians_tests().iter().for_each(|test| {
            let result = Radians::from(test.input);
            let expected = test.expected;

            assert_eq!(result, expected);
        })
    }
}

#[cfg(test)]
mod degrees_arithmetic_tests {
    use super::{
        Degrees,
    };

    #[test]
    fn test_addition() {
        let angle1 = Degrees(30_f64); 
        let angle2 = Degrees(45_f64);
        let expected = Degrees(75_f64);
        
        let result = angle1 + angle2;
        assert_eq!(result, expected);

        let result = &angle1 + angle2;
        assert_eq!(result, expected);

        let result = angle1 + &angle2;
        assert_eq!(result, expected);

        let result = &angle1 + &angle2;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(45_f64);
        let expected = -Degrees(15_f64);
        
        let result = angle1 - angle2;
        assert_eq!(result, expected);

        let result = &angle1 - angle2;
        assert_eq!(result, expected);

        let result = angle1 - &angle2;
        assert_eq!(result, expected);

        let result = &angle1 - &angle2;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiplication() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(45_f64);
        let expected = Degrees(30_f64 * 45_f64);
        
        let result = angle1 * angle2;
        assert_eq!(result, expected);

        let result = &angle1 * angle2;
        assert_eq!(result, expected);

        let result = angle1 * &angle2;
        assert_eq!(result, expected);

        let result = &angle1 * &angle2;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_division() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(45_f64);
        let expected = Degrees(30_f64 / 45_f64);
        
        let result = angle1 / angle2;
        assert_eq!(result, expected);

        let result = &angle1 / angle2;
        assert_eq!(result, expected);

        let result = angle1 / &angle2;
        assert_eq!(result, expected);

        let result = &angle1 / &angle2;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_negation() {
        let angle = Degrees(30_f64);
        let expected = Degrees(-30_f64);
        
        let result = -angle;
        assert_eq!(result, expected);

        let result = -&angle;
        assert_eq!(result, expected)
    }

    #[test]
    fn test_remainder_less_than_modulus() {
        let angle = Degrees(45_f64);
        let modulus = Degrees(360_f64);
        let expected = angle;

        let result = angle % modulus;
        assert_eq!(result, expected);

        let result = &angle % modulus;
        assert_eq!(result, expected);

        let result = angle % &modulus;
        assert_eq!(result, expected);

        let result = &angle % &modulus;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_remainder_greater_than_modulus() {
        let angle = Degrees(405_f64);
        let modulus = Degrees(360_f64);
        let expected = Degrees(45_f64);

        let result = angle % modulus;
        assert_eq!(result, expected);

        let result = &angle % modulus;
        assert_eq!(result, expected);

        let result = angle % &modulus;
        assert_eq!(result, expected);

        let result = &angle % &modulus;
        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod radians_arithmetic_tests {
    use super::{
        Radians,
    };
    use approx::{
        relative_eq,
    };
    use std::f64;

    const PI: Radians<f64> = Radians(f64::consts::PI);


    #[test]
    fn test_addition() {
        let angle1 = PI / 6_f64; 
        let angle2 = PI / 4_f64;
        let expected = PI * 10_f64 / 24_f64;

        let result = angle1 + angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 + angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = angle1 + &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 + &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_subtraction() {
        let angle1 = PI / 6_f64;
        let angle2 = PI / 4_f64;
        let expected = -PI / 12_f64;

        let result = angle1 - angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 - angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = angle1 - &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 - &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_multiplication() {
        let angle1 = PI / 6_f64;
        let angle2 = PI / 4_f64;
        let expected = Radians(f64::consts::PI * f64::consts::PI / 24_f64);

        let result = angle1 * angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 * angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = angle1 * &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 * &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_division() {
        let angle1 = PI / 6_f64;
        let angle2 = PI / 4_f64;
        let expected = Radians(4_f64 / 6_f64);

        let result = angle1 / angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 / angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = angle1 / &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 / &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_negation() {
        let angle = PI / 6_f64;
        let expected = -PI / 6_f64;
        
        let result = -angle;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = -&angle;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_remainder_less_than_modulus() {
        let angle = PI / 4_f64;
        let modulus = PI * 2_f64;
        let expected = angle;

        let result = angle % modulus;
        assert_eq!(result, expected);

        let result = &angle % modulus;
        assert_eq!(result, expected);

        let result = angle % &modulus;
        assert_eq!(result, expected);

        let result = &angle % &modulus;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_remainder_greater_than_modulus() {
        let angle = PI * 2_f64 + PI / 4_f64;
        let modulus = PI * 2_f64;
        let expected = PI / 4_f64;

        let result = angle % modulus;
        assert_eq!(result, expected);

        let result = &angle % modulus;
        assert_eq!(result, expected);

        let result = angle % &modulus;
        assert_eq!(result, expected);

        let result = &angle % &modulus;
        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod radian_angle_tests {

}

#[cfg(test)]
mod degree_angle_tests {

}
