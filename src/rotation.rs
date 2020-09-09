use scalar::{
    Scalar,
    ScalarFloat,
};
use structure::{
    Euclidean,
    One,
};
use matrix::{
    Matrix2,
    Matrix3,
};
use std::fmt;
use std::iter;
use std::ops;


pub trait Rotation<P> where 
    P: Euclidean,
    Self: Sized + Copy + One,
{
    fn look_at(direction: P::Difference, up: P::Difference) -> Self;

    fn between_vectors(v1: P::Difference, v2: P::Difference) -> Self;

    fn rotate_vector(&self, vector: P::Difference) -> P::Difference;

    fn inverse(&self) -> Self;

    fn rotate_point(&self, point: P) -> P;
}

#[derive(Copy, Clone, PartialEq)]
pub struct RotationMatrix2<S> {
    matrix: Matrix2<S>,
}

impl<S> fmt::Debug for RotationMatrix2<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RotationMatrix2 ")?;
        <[S; 4] as fmt::Debug>::fmt(self.matrix.as_ref(), f)
    }
}

impl<S> fmt::Display for RotationMatrix2<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RotationMatrix2 ")?;
        <[S; 4] as fmt::Debug>::fmt(self.matrix.as_ref(), f)
    }
}

impl<S> From<RotationMatrix2<S>> for Matrix2<S> where S: Copy {
    #[inline]
    fn from(rotation: RotationMatrix2<S>) -> Matrix2<S> {
        rotation.matrix
    }
}

impl<S> AsRef<Matrix2<S>> for RotationMatrix2<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix2<S> {
        &self.matrix
    }
}

impl<S> ops::Mul<RotationMatrix2<S>> for RotationMatrix2<S> where S: Scalar {
    type Output = RotationMatrix2<S>;
    
    #[inline]
    fn mul(self, other: RotationMatrix2<S>) -> Self::Output {
        RotationMatrix2 { matrix: self.matrix * other.matrix }
    }
}

impl<S> ops::Mul<RotationMatrix2<S>> for &RotationMatrix2<S> where S: Scalar {
    type Output = RotationMatrix2<S>;
    
    #[inline]
    fn mul(self, other: RotationMatrix2<S>) -> Self::Output {
        RotationMatrix2 { matrix: self.matrix * other.matrix }
    }
}

impl<S> ops::Mul<&RotationMatrix2<S>> for RotationMatrix2<S> where S: Scalar {
    type Output = RotationMatrix2<S>;
    
    #[inline]
    fn mul(self, other: &RotationMatrix2<S>) -> Self::Output {
        RotationMatrix2 { matrix: self.matrix * other.matrix }
    }
}

impl<'a, 'b, S> ops::Mul<&'a RotationMatrix2<S>> for &'b RotationMatrix2<S> where S: Scalar {
    type Output = RotationMatrix2<S>;
    
    #[inline]
    fn mul(self, other: &'a RotationMatrix2<S>) -> Self::Output {
        RotationMatrix2 { matrix: self.matrix * other.matrix }
    }
}

impl<S> One for RotationMatrix2<S> where S: Scalar {
    #[inline]
    fn one() -> RotationMatrix2<S> {
        RotationMatrix2 { matrix: Matrix2::one() }
    }
}

impl<S> iter::Product<RotationMatrix2<S>> for RotationMatrix2<S> where S: Scalar {
    #[inline]
    fn product<I: Iterator<Item = RotationMatrix2<S>>>(iter: I) -> RotationMatrix2<S> {
        iter.fold(RotationMatrix2::one(), ops::Mul::mul)
    }
}

impl<'a, S> iter::Product<&'a RotationMatrix2<S>> for RotationMatrix2<S> where S: 'a + Scalar {
    #[inline]
    fn product<I: Iterator<Item = &'a RotationMatrix2<S>>>(iter: I) -> RotationMatrix2<S> {
        iter.fold(RotationMatrix2::one(), ops::Mul::mul)
    }
}

impl<S> approx::AbsDiffEq for RotationMatrix2<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Matrix2::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S> approx::RelativeEq for RotationMatrix2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix2::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for RotationMatrix2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix2::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

#[derive(Copy, Clone, PartialEq)]
pub struct RotationMatrix3<S> {
    matrix: Matrix3<S>,
}

impl<S> fmt::Debug for RotationMatrix3<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RotationMatrix3 ")?;
        <[S; 9] as fmt::Debug>::fmt(self.matrix.as_ref(), f)
    }
}

impl<S> fmt::Display for RotationMatrix3<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RotationMatrix3 ")?;
        <[S; 9] as fmt::Debug>::fmt(self.matrix.as_ref(), f)
    }
}

impl<S> From<RotationMatrix3<S>> for Matrix3<S> where S: Copy {
    #[inline]
    fn from(rotation: RotationMatrix3<S>) -> Matrix3<S> {
        rotation.matrix
    }
}

impl<S> AsRef<Matrix3<S>> for RotationMatrix3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3<S> {
        &self.matrix
    }
}

impl<S> ops::Mul<RotationMatrix3<S>> for RotationMatrix3<S> where S: Scalar {
    type Output = RotationMatrix3<S>;
    
    #[inline]
    fn mul(self, other: RotationMatrix3<S>) -> Self::Output {
        RotationMatrix3 { matrix: self.matrix * other.matrix }
    }
}

impl<S> ops::Mul<RotationMatrix3<S>> for &RotationMatrix3<S> where S: Scalar {
    type Output = RotationMatrix3<S>;
    
    #[inline]
    fn mul(self, other: RotationMatrix3<S>) -> Self::Output {
        RotationMatrix3 { matrix: self.matrix * other.matrix }
    }
}

impl<S> ops::Mul<&RotationMatrix3<S>> for RotationMatrix3<S> where S: Scalar {
    type Output = RotationMatrix3<S>;
    
    #[inline]
    fn mul(self, other: &RotationMatrix3<S>) -> Self::Output {
        RotationMatrix3 { matrix: self.matrix * other.matrix }
    }
}

impl<'a, 'b, S> ops::Mul<&'a RotationMatrix3<S>> for &'b RotationMatrix3<S> where S: Scalar {
    type Output = RotationMatrix3<S>;
    
    #[inline]
    fn mul(self, other: &'a RotationMatrix3<S>) -> Self::Output {
        RotationMatrix3 { matrix: self.matrix * other.matrix }
    }
}

impl<S> One for RotationMatrix3<S> where S: Scalar {
    #[inline]
    fn one() -> RotationMatrix3<S> {
        RotationMatrix3 { matrix: Matrix3::one() }
    }
}

impl<S> iter::Product<RotationMatrix3<S>> for RotationMatrix3<S> where S: Scalar {
    #[inline]
    fn product<I: Iterator<Item = RotationMatrix3<S>>>(iter: I) -> RotationMatrix3<S> {
        iter.fold(RotationMatrix3::one(), ops::Mul::mul)
    }
}

impl<'a, S> iter::Product<&'a RotationMatrix3<S>> for RotationMatrix3<S> where S: 'a + Scalar {
    #[inline]
    fn product<I: Iterator<Item = &'a RotationMatrix3<S>>>(iter: I) -> RotationMatrix3<S> {
        iter.fold(RotationMatrix3::one(), ops::Mul::mul)
    }
}

impl<S> approx::AbsDiffEq for RotationMatrix3<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Matrix3::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S> approx::RelativeEq for RotationMatrix3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix3::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for RotationMatrix3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix3::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

