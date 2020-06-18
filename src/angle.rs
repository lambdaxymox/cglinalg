use base::{
    Scalar,
    ScalarFloat,
};
use std::f64;
use std::ops;


#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
#[repr(C)]
pub struct Radians<S>(S);

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
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
        Degrees(self.0 + other.0)
    } 
}

impl<'a, S> ops::Sub<&'a Degrees<S>> for Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<'a, S> ops::Sub<Degrees<S>> for &'a Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
    } 
}

impl<'a, 'b, S> ops::Sub<&'a Degrees<S>> for &'b Degrees<S> where S: ScalarFloat {
    type Output = Degrees<S>;

    #[inline]
    fn sub(self, other: &'a Degrees<S>) -> Self::Output {
        Degrees(self.0 + other.0)
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

impl<S> num_traits::Zero for Degrees<S> where S: ScalarFloat {
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
