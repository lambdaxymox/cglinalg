use crate::traits::{Array, Zero, VectorSpace, MetricSpace, DotProduct, Lerp};
use crate::vector::*;
use std::fmt;
use std::mem;
use std::ops;
use std::cmp;


const EPSILON: f32 = 0.00001; 


///
/// The `Matrix2` type represents 2x2 matrices in column-major order.
///
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Matrix2 {
    /// Column 1 of the matrix.
    c0r0: f32, c0r1: f32,
    /// Column 2 of the matrix.
    c1r0: f32, c1r1: f32,
}

impl Matrix2 {
    pub fn new(c0r0: f32, c0r1: f32, c1r0: f32, c1r1: f32) -> Matrix2 {
        Matrix2 { c0r0: c0r0, c0r1: c0r1, c1r0: c1r0, c1r1: c1r1 }
    }

    pub fn from_cols(c0: Vector2, c1: Vector2) -> Matrix2 {
        Matrix2 { c0r0: c0.x, c0r1: c0.y, c1r0: c1.x, c1r1: c1.y }
    }

    ///
    /// Return the zero matrix.
    ///
    pub fn zero() -> Matrix2 {
        Matrix2::new(0.0, 0.0, 0.0, 0.0)
    }

    ///
    /// Return the identity matrix.
    ///
    pub fn one() -> Matrix2 {
        Matrix2::new(1.0, 0.0, 0.0, 1.0)
    }

    ///
    /// Compute the transpose of a 2x2 matrix.
    ///
    pub fn transpose(&self) -> Matrix2 {
        Matrix2::new(
            self.c0r0, self.c1r0,
            self.c0r1, self.c1r1,
        )
    }
}

impl Array for Matrix2 {
    type Element = f32;

    #[inline]
    fn len() -> usize {
        4
    }

    #[inline]
    fn as_ptr(&self) -> *const f32 {
        &self.c0r0
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut f32 {
        &mut self.c0r0
    }
}

impl fmt::Display for Matrix2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, 
            "\n[{:.2}][{:.2}]\n[{:.2}][{:.2}]", 
            self.c0r0, self.c1r0,
            self.c0r1, self.c1r1,
        )
    }
}

impl AsRef<[f32; 4]> for Matrix2 {
    fn as_ref(&self) -> &[f32; 4] {
        unsafe {
            mem::transmute(self)
        }
    }
}

impl AsRef<[[f32; 2]; 2]> for Matrix2 {
    fn as_ref(&self) -> &[[f32; 2]; 2] {
        unsafe {
            mem::transmute(self)
        }
    }
}

impl AsMut<[f32; 4]> for Matrix2 {
    fn as_mut(&mut self) -> &mut [f32; 4] {
        unsafe {
            mem::transmute(self)
        }
    }
}

impl ops::Add<Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn add(self, other: Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a> ops::Add<&'a Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn add(self, other: &'a Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a> ops::Add<Matrix2> for &'a Matrix2 {
    type Output = Matrix2;

    fn add(self, other: Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a, 'b> ops::Add<&'a Matrix2> for &'b Matrix2 {
    type Output = Matrix2;

    fn add(self, other: &'a Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl ops::Sub<Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn sub(self, other: Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a> ops::Sub<&'a Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn sub(self, other: &'a Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a> ops::Sub<Matrix2> for &'a Matrix2 {
    type Output = Matrix2;

    fn sub(self, other: Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a, 'b> ops::Sub<&'a Matrix2> for &'b Matrix2 {
    type Output = Matrix2;

    fn sub(self, other: &'a Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a> ops::Mul<&'a Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn mul(self, other: &'a Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1;
        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a, 'b> ops::Mul<&'a Matrix2> for &'b Matrix2 {
    type Output = Matrix2;

    fn mul(self, other: &'a Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1;
        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl ops::Mul<Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn mul(self, other: Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1;
        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a> ops::Mul<Matrix2> for &'a Matrix2 {
    type Output = Matrix2;

    fn mul(self, other: Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1;
        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl ops::Mul<f32> for Matrix2 {
    type Output = Matrix2;

    fn mul(self, other: f32) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a> ops::Mul<f32> for &'a Matrix2 {
    type Output = Matrix2;

    fn mul(self, other: f32) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl ops::Div<f32> for Matrix2 {
    type Output = Matrix2;

    fn div(self, other: f32) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a> ops::Div<f32> for &'a Matrix2 {
    type Output = Matrix2;

    fn div(self, other: f32) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

