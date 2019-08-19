use std::fmt;
use std::mem;
use std::ops;
use std::cmp;

use crate::traits::{Array, One, Zero, VectorSpace, MetricSpace, DotProduct, Lerp};
use crate::vector::*;


const EPSILON: f32 = 0.00001;
const M_PI: f32 = 3.14159265358979323846264338327950288;
const ONE_DEG_IN_RAD: f32 = (2.0 * M_PI) / 360.0; // == 0.017444444

///
/// The `Matrix2` type represents 2x2 matrices in column-major order.
///
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Matrix2 {
    /// Column 1 of the matrix.
    pub c0r0: f32, pub c0r1: f32,
    /// Column 2 of the matrix.
    pub c1r0: f32, pub c1r1: f32,
}

impl Matrix2 {
    ///
    /// Construct a new 2x2 matrix from its field elements.
    /// 
    pub fn new(c0r0: f32, c0r1: f32, c1r0: f32, c1r1: f32) -> Matrix2 {
        Matrix2 { c0r0: c0r0, c0r1: c0r1, c1r0: c1r0, c1r1: c1r1 }
    }

    ///
    /// Construct a 2x2 matrix from a pair of two-dimensional vectors.
    /// 
    pub fn from_cols(c0: Vector2, c1: Vector2) -> Matrix2 {
        Matrix2 { c0r0: c0.x, c0r1: c0.y, c1r0: c1.x, c1r1: c1.y }
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

impl Zero for Matrix2 {
    fn zero() -> Matrix2 {
        Matrix2::new(0.0, 0.0, 0.0, 0.0)
    }

    fn is_zero(&self) -> bool {
        self.c0r0 == 0.0 && self.c0r1 == 0.0 &&
        self.c1r0 == 0.0 && self.c1r1 == 0.0
    }
}

impl One for Matrix2 {
    fn one() -> Matrix2 {
        Matrix2::new(1.0, 0.0, 0.0, 1.0)
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
        unsafe { mem::transmute(self) }
    }
}

impl AsRef<[[f32; 2]; 2]> for Matrix2 {
    fn as_ref(&self) -> &[[f32; 2]; 2] {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<[f32; 4]> for Matrix2 {
    fn as_mut(&mut self) -> &mut [f32; 4] {
        unsafe { mem::transmute(self) }
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

impl ops::Add<&Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn add(self, other: &Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl ops::Add<Matrix2> for &Matrix2 {
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

impl ops::Sub<&Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn sub(self, other: &Matrix2) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl ops::Sub<Matrix2> for &Matrix2 {
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

impl ops::Mul<&Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn mul(self, other: &Matrix2) -> Self::Output {
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

impl ops::Mul<Matrix2> for &Matrix2 {
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

impl ops::Mul<f32> for &Matrix2 {
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

impl ops::Div<f32> for &Matrix2 {
    type Output = Matrix2;

    fn div(self, other: f32) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}


///
/// The `Matrix3` type represents 3x3 matrices in column-major order.
///
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Matrix3 {
    /// Column 1 of the matrix.
    pub c0r0: f32, pub c0r1: f32, pub c0r2: f32,
    /// Column 2 of the matrix.
    pub c1r0: f32, pub c1r1: f32, pub c1r2: f32,
    /// Column 3 of the matrix.
    pub c2r0: f32, pub c2r1: f32, pub c2r2: f32,
}

impl Matrix3 {
    pub fn new(
        c0r0: f32, c0r1: f32, c0r2: f32,
        c1r0: f32, c1r1: f32, c1r2: f32,
        c2r0: f32, c2r1: f32, c2r2: f32) -> Matrix3 {

        Matrix3 {
            // Column 1 of the matrix.
            c0r0: c0r0, c0r1: c0r1, c0r2: c0r2,
            // Column 2 of the matrix.
            c1r0: c1r0, c1r1: c1r1, c1r2: c1r2,
            // Column 3 of the matrix.
            c2r0: c2r0, c2r1: c2r1, c2r2: c2r2,
        }
    }

    /// Create a 3x3 matrix from a triple of three-dimensional column vectors.
    pub fn from_cols(c0: Vector3, c1: Vector3, c2: Vector3) -> Matrix3 {
        Matrix3 {
            c0r0: c0.x, c0r1: c0.y, c0r2: c0.z, 
            c1r0: c1.x, c1r1: c1.y, c1r2: c1.z,
            c2r0: c2.x, c2r1: c2.y, c2r2: c2.z,
        }
    }

    /// Generate a 3x3 diagonal matrix of ones.
    pub fn one() -> Matrix3 {
        Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    }

    /// Compute the transpose of a 3x3 matrix.
    pub fn transpose(&self) -> Matrix3 {
        Matrix3::new(
            self.c0r0, self.c1r0, self.c2r0,
            self.c0r1, self.c1r1, self.c2r1,
            self.c0r2, self.c1r2, self.c2r2,
        )
    }
}

impl Array for Matrix3 {
    type Element = f32;

    #[inline]
    fn len() -> usize {
        9
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

impl Zero for Matrix3 {
    fn zero() -> Matrix3 {
        Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    fn is_zero(&self) -> bool {
        self.c0r0 == 0.0 && self.c0r1 == 0.0 && self.c0r2 == 0.0 &&
        self.c1r0 == 0.0 && self.c1r1 == 0.0 && self.c1r2 == 0.0 &&
        self.c2r0 == 0.0 && self.c2r1 == 0.0 && self.c2r2 == 0.0
    }
}

impl One for Matrix3 {
    fn one() -> Matrix3 {
        Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    }
}

impl fmt::Display for Matrix3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, 
            "\n[{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}]", 
            self.c0r0, self.c1r0, self.c2r0,
            self.c0r1, self.c1r1, self.c2r1,
            self.c0r2, self.c1r2, self.c2r2,
        )
    }
}

impl AsRef<[f32; 9]> for Matrix3 {
    fn as_ref(&self) -> &[f32; 9] {
        unsafe { mem::transmute(self) }
    }
}

impl AsRef<[[f32; 3]; 3]> for Matrix3 {
    fn as_ref(&self) -> &[[f32; 3]; 3] {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<[f32; 9]> for Matrix3 {
    fn as_mut(&mut self) -> &mut [f32; 9] {
        unsafe { mem::transmute(self) }
    }
}

impl ops::Add<Matrix3> for Matrix3 {
    type Output = Matrix3;

    fn add(self, other: Matrix3) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;
        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl ops::Add<&Matrix3> for Matrix3 {
    type Output = Matrix3;

    fn add(self, other: &Matrix3) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;
        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl ops::Add<Matrix3> for &Matrix3 {
    type Output = Matrix3;

    fn add(self, other: Matrix3) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;
        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<'a, 'b> ops::Add<&'a Matrix3> for &'b Matrix3 {
    type Output = Matrix3;

    fn add(self, other: &'a Matrix3) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;
        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl ops::Sub<Matrix3> for Matrix3 {
    type Output = Matrix3;

    fn sub(self, other: Matrix3) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;
        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl ops::Sub<&Matrix3> for Matrix3 {
    type Output = Matrix3;

    fn sub(self, other: &Matrix3) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;
        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl ops::Sub<Matrix3> for &Matrix3 {
    type Output = Matrix3;

    fn sub(self, other: Matrix3) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;
        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<'a, 'b> ops::Sub<&'a Matrix3> for &'b Matrix3 {
    type Output = Matrix3;

    fn sub(self, other: &'a Matrix3) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;
        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl ops::Mul<&Matrix3> for Matrix3 {
    type Output = Matrix3;

    fn mul(self, other: &Matrix3) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<'a, 'b> ops::Mul<&'a Matrix3> for &'b Matrix3 {
    type Output = Matrix3;

    fn mul(self, other: &'a Matrix3) -> Matrix3 {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl ops::Mul<Matrix3> for Matrix3 {
    type Output = Matrix3;

    fn mul(self, other: Matrix3) -> Matrix3 {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl ops::Mul<Matrix3> for &Matrix3 {
    type Output = Matrix3;

    fn mul(self, other: Matrix3) -> Matrix3 {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl ops::Mul<f32> for Matrix3 {
    type Output = Matrix3;

    fn mul(self, other: f32) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c0r2 = self.c0r2 * other;

        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;
        let c1r2 = self.c1r2 * other;
        
        let c2r0 = self.c2r0 * other;
        let c2r1 = self.c2r1 * other;
        let c2r2 = self.c2r2 * other;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl ops::Mul<f32> for &Matrix3 {
    type Output = Matrix3;

    fn mul(self, other: f32) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c0r2 = self.c0r2 * other;

        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;
        let c1r2 = self.c1r2 * other;
        
        let c2r0 = self.c2r0 * other;
        let c2r1 = self.c2r1 * other;
        let c2r2 = self.c2r2 * other;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl ops::Div<f32> for Matrix3 {
    type Output = Matrix3;

    fn div(self, other: f32) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c0r2 = self.c0r2 / other;

        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;
        let c1r2 = self.c1r2 / other;
        
        let c2r0 = self.c2r0 / other;
        let c2r1 = self.c2r1 / other;
        let c2r2 = self.c2r2 / other;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl ops::Div<f32> for &Matrix3 {
    type Output = Matrix3;

    fn div(self, other: f32) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c0r2 = self.c0r2 / other;

        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;
        let c1r2 = self.c1r2 / other;
        
        let c2r0 = self.c2r0 / other;
        let c2r1 = self.c2r1 / other;
        let c2r2 = self.c2r2 / other;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}


///
/// The `Matrix4` type represents 4x4 matrices in column-major order.
///
#[derive(Copy, Clone, Debug)]
pub struct Matrix4 {
    /// Column 1 of the matrix.
    pub c0r0: f32, pub c0r1: f32, pub c0r2: f32, pub c0r3: f32,
    /// Column 2 of the matrix.
    pub c1r0: f32, pub c1r1: f32, pub c1r2: f32, pub c1r3: f32,
    /// Column 3 of the matrix.
    pub c2r0: f32, pub c2r1: f32, pub c2r2: f32, pub c2r3: f32,
    /// Column 4 of the matrix.
    pub c3r0: f32, pub c3r1: f32, pub c3r2: f32, pub c3r3: f32,
}

impl Matrix4 {
    pub fn new(
        c0r0: f32, c0r1: f32, c0r2: f32, c0r3: f32,
        c1r0: f32, c1r1: f32, c1r2: f32, c1r3: f32,
        c2r0: f32, c2r1: f32, c2r2: f32, c2r3: f32,
        c3r0: f32, c3r1: f32, c3r2: f32, c3r3: f32) -> Matrix4 {

        Matrix4 {
            c0r0: c0r0, c0r1: c0r1, c0r2: c0r2, c0r3: c0r3,
            c1r0: c1r0, c1r1: c1r1, c1r2: c1r2, c1r3: c1r3,
            c2r0: c2r0, c2r1: c2r1, c2r2: c2r2, c2r3: c2r3,
            c3r0: c3r0, c3r1: c3r1, c3r2: c3r2, c3r3: c3r3,
        }
    }

    ///
    /// Transpose a 4x4 matrix.
    ///
    pub fn transpose(&self) -> Matrix4 {
        Matrix4::new(
            self.c0r0, self.c1r0, self.c2r0, self.c3r0,
            self.c0r1, self.c1r1, self.c2r1, self.c3r1, 
            self.c0r2, self.c1r2, self.c2r2, self.c3r2, 
            self.c0r3, self.c1r3, self.c2r3, self.c3r3
        )
    }

    ///
    /// Create a affine translation matrix.
    ///
    #[inline]
    pub fn from_translation(distance: Vector3) -> Matrix4 {
        Matrix4::new(
            1.0,        0.0,        0.0,        0.0,
            0.0,        1.0,        0.0,        0.0,
            0.0,        0.0,        1.0,        0.0,
            distance.x, distance.y, distance.z, 1.0
        )
    }

    ///
    /// Create a rotation matrix around the x axis by an angle in `degrees` degrees.
    ///
    pub fn from_rotation_x(degrees: f32) -> Matrix4 {
        // Convert to radians.
        let radians = degrees * ONE_DEG_IN_RAD;
        let mut rot_mat = Matrix4::one();
        rot_mat.c1r1 =  f32::cos(radians);
        rot_mat.c2r1 = -f32::sin(radians);
        rot_mat.c1r2 =  f32::sin(radians);
        rot_mat.c2r2 =  f32::cos(radians);
    
        rot_mat
    }

    ///
    /// Create a rotation matrix around the y axis by an angle in `degrees` degrees.
    ///
    pub fn from_rotation_y(degrees: f32) -> Matrix4 {
        // Convert to radians.
        let radians = degrees * ONE_DEG_IN_RAD;
        let mut rot_mat = Matrix4::one();
        rot_mat.c0r0 =  f32::cos(radians);
        rot_mat.c2r0 =  f32::sin(radians);
        rot_mat.c0r2 = -f32::sin(radians);
        rot_mat.c2r2 =  f32::cos(radians);
    
        rot_mat
    }

    ///
    /// Create a rotation matrix around the z axis by an angle in `degrees` degrees.
    ///
    pub fn from_rotation_z(degrees: f32) -> Matrix4 {
        // Convert to radians.
        let radians = degrees * ONE_DEG_IN_RAD;
        let mut rot_mat = Matrix4::one();
        rot_mat.c0r0 =  f32::cos(radians);
        rot_mat.c1r0 = -f32::sin(radians);
        rot_mat.c0r1 =  f32::sin(radians);
        rot_mat.c1r1 =  f32::cos(radians);
    
        rot_mat
    }

    ///
    /// Scale a matrix uniformly.
    ///
    #[inline]
    pub fn from_scale(value: f32) -> Matrix4 {
        Matrix4::from_nonuniform_scale(value, value, value)
    }

    ///
    /// Scale a matrix in a nonuniform fashion.
    ///
    #[inline]
    pub fn from_nonuniform_scale(sx: f32, sy: f32, sz: f32) -> Matrix4 {
        Matrix4::new(
            sx,  0.0, 0.0, 0.0,
            0.0, sy,  0.0, 0.0,
            0.0, 0.0, sz,  0.0,
            0.0, 0.0, 0.0, 1.0
        )
    }

    ///
    /// Computes the determinant of a 4x4 matrix.
    ///
    pub fn determinant(&self) -> f32 {
        self.c3r0 * self.c2r1 * self.c1r2 * self.c0r3 -
        self.c2r0 * self.c3r1 * self.c1r2 * self.c0r3 -
        self.c3r0 * self.c1r1 * self.c2r2 * self.c0r3 +
        self.c1r0 * self.c3r1 * self.c2r2 * self.c0r3 +
        self.c2r0 * self.c1r1 * self.c3r2 * self.c0r3 -
        self.c1r0 * self.c2r1 * self.c3r2 * self.c0r3 -
        self.c3r0 * self.c2r1 * self.c0r2 * self.c1r3 +
        self.c2r0 * self.c3r1 * self.c0r2 * self.c1r3 +
        self.c3r0 * self.c0r1 * self.c2r2 * self.c1r3 -
        self.c0r0 * self.c3r1 * self.c2r2 * self.c1r3 -
        self.c2r0 * self.c0r1 * self.c3r2 * self.c1r3 +
        self.c0r0 * self.c2r1 * self.c3r2 * self.c1r3 +
        self.c3r0 * self.c1r1 * self.c0r2 * self.c2r3 -
        self.c1r0 * self.c3r1 * self.c0r2 * self.c2r3 -
        self.c3r0 * self.c0r1 * self.c1r2 * self.c2r3 +
        self.c0r0 * self.c3r1 * self.c1r2 * self.c2r3 +
        self.c1r0 * self.c0r1 * self.c3r2 * self.c2r3 -
        self.c0r1 * self.c1r1 * self.c3r2 * self.c2r3 -
        self.c2r0 * self.c1r1 * self.c0r2 * self.c3r3 +
        self.c1r0 * self.c2r1 * self.c0r2 * self.c3r3 +
        self.c2r0 * self.c0r1 * self.c1r2 * self.c3r3 -
        self.c0r0 * self.c2r1 * self.c1r2 * self.c3r3 -
        self.c1r0 * self.c0r1 * self.c2r2 * self.c3r3 +
        self.c0r0 * self.c1r1 * self.c2r2 * self.c3r3
    }

    pub fn is_invertible(&self) -> bool {
        self.determinant() != 0.0
    }

    ///
    /// Compute the inverse of a 4x4 matrix.
    ///
    pub fn inverse(&self) -> Matrix4 {
        let det = self.determinant();
        
        // A matrix with zero determinant has no inverse.
        if det == 0.0 {
            eprintln!("WARNING. Matrix has zero determinant. It cannot be inverted.");
            
            return *self;
        }

        let inv_det = 1.0 / det;

        return Matrix4::new(
            inv_det * ( self.c2r1 * self.c3r2 * self.c1r3 - self.c3r1 * self.c2r2 * self.c1r3 +
                                    self.c3r1 * self.c1r2 * self.c2r3 - self.c1r1 * self.c3r2 * self.c2r3 -
                                    self.c2r1 * self.c1r2 * self.c3r3 + self.c1r1 * self.c2r2 * self.c3r3 ),
            inv_det * ( self.c3r1 * self.c2r2 * self.c0r3 - self.c2r1 * self.c3r2 * self.c0r3 -
                                    self.c3r1 * self.c0r2 * self.c2r3 + self.c0r1 * self.c3r2 * self.c2r3 +
                                    self.c2r1 * self.c0r2 * self.c3r3 - self.c0r1 * self.c2r2 * self.c3r3 ),
            inv_det * ( self.c1r1 * self.c3r2 * self.c0r3 - self.c3r1 * self.c1r1 * self.c0r3 +
                                    self.c3r1 * self.c0r2 * self.c1r2 - self.c0r1 * self.c3r2 * self.c1r3 -
                                    self.c1r1 * self.c0r2 * self.c3r3 + self.c0r1 * self.c1r2 * self.c3r3 ),
            inv_det * ( self.c2r1 * self.c1r2 * self.c0r3 - self.c1r1 * self.c2r2 * self.c0r3 -
                                    self.c2r1 * self.c0r2 * self.c1r3 + self.c0r1 * self.c2r2 * self.c1r3 +
                                    self.c1r1 * self.c0r2 * self.c2r3 - self.c0r1 * self.c1r2 * self.c2r3 ),
            inv_det * ( self.c3r0 * self.c2r2 * self.c1r3 - self.c2r0 * self.c3r2 * self.c1r3 -
                                    self.c3r0 * self.c1r2 * self.c2r3 + self.c1r0 * self.c3r2 * self.c2r3 +
                                    self.c2r0 * self.c1r2 * self.c3r3 - self.c1r0 * self.c2r2 * self.c3r3 ),
            inv_det * ( self.c2r0 * self.c3r2 * self.c0r3 - self.c3r0 * self.c2r2 * self.c0r3 +
                                    self.c3r0 * self.c0r2 * self.c2r3 - self.c0r0 * self.c3r2 * self.c2r3 -
                                    self.c2r0 * self.c0r2 * self.c3r3 + self.c0r0 * self.c2r2 * self.c3r3 ),
            inv_det * ( self.c3r0 * self.c1r2 * self.c0r3 - self.c1r0 * self.c3r2 * self.c0r3 -
                                    self.c3r0 * self.c0r2 * self.c1r3 + self.c0r0 * self.c3r2 * self.c1r3 +
                                    self.c1r0 * self.c0r2 * self.c3r3 - self.c0r0 * self.c1r2 * self.c3r3 ),
            inv_det * ( self.c1r0 * self.c2r2 * self.c0r3 - self.c2r0 * self.c1r2 * self.c0r3 +
                                    self.c2r0 * self.c0r2 * self.c1r3 - self.c0r0 * self.c2r2 * self.c1r3 -
                                    self.c1r0 * self.c0r2 * self.c2r3 + self.c0r0 * self.c1r2 * self.c2r3 ),
            inv_det * ( self.c2r0 * self.c3r1 * self.c1r3 - self.c3r0 * self.c2r1 * self.c1r3 +
                                    self.c3r0 * self.c1r1 * self.c2r3 - self.c1r0 * self.c3r1 * self.c2r3 -
                                    self.c2r0 * self.c1r1 * self.c3r3 + self.c1r0 * self.c2r1 * self.c3r3 ),
            inv_det * ( self.c3r0 * self.c2r1 * self.c0r3 - self.c2r0 * self.c3r1 * self.c0r3 -
                                    self.c3r0 * self.c0r1 * self.c2r3 + self.c0r0 * self.c3r1 * self.c2r3 +
                                    self.c2r0 * self.c0r1 * self.c3r3 - self.c0r0 * self.c2r1 * self.c3r3 ),
            inv_det * ( self.c1r0 * self.c3r1 * self.c0r3 - self.c3r0 * self.c1r1 * self.c0r3 +
                                    self.c3r0 * self.c0r1 * self.c1r3 - self.c0r0 * self.c3r1 * self.c1r3 -
                                    self.c1r0 * self.c0r1 * self.c3r3 + self.c0r0 * self.c1r1 * self.c3r3 ),
            inv_det * ( self.c2r0 * self.c1r1 * self.c0r3 - self.c1r0 * self.c2r1 * self.c0r3 -
                                    self.c2r0 * self.c0r1 * self.c1r3 + self.c0r0 * self.c2r1 * self.c1r3 +
                                    self.c1r0 * self.c0r1 * self.c2r3 - self.c0r0 * self.c1r1 * self.c2r3 ),
            inv_det * ( self.c3r0 * self.c2r1 * self.c1r2 - self.c2r0 * self.c3r1 * self.c1r2 -
                                    self.c3r0 * self.c1r1 * self.c2r2 + self.c1r0 * self.c3r1 * self.c2r2 +
                                    self.c2r0 * self.c1r1 * self.c3r2 - self.c1r0 * self.c2r1 * self.c3r2 ),
            inv_det * ( self.c2r0 * self.c3r1 * self.c0r2 - self.c3r0 * self.c2r1 * self.c0r2 +
                                    self.c3r0 * self.c0r1 * self.c2r2 - self.c0r0 * self.c3r1 * self.c2r2 -
                                    self.c2r0 * self.c0r1 * self.c3r2 + self.c0r0 * self.c2r1 * self.c3r2 ),
            inv_det * ( self.c3r0 * self.c1r1 * self.c0r2 - self.c1r0 * self.c3r1 * self.c0r2 -
                                    self.c3r0 * self.c0r1 * self.c1r2 + self.c0r0 * self.c3r1 * self.c1r2 +
                                    self.c1r0 * self.c0r1 * self.c3r2 - self.c0r0 * self.c1r1 * self.c3r2 ),
            inv_det * ( self.c1r0 * self.c2r1 * self.c0r2 - self.c2r0 * self.c1r1 * self.c0r2 +
                                    self.c2r0 * self.c0r1 * self.c1r2 - self.c0r0 * self.c2r1 * self.c1r2 -
                                    self.c1r0 * self.c0r1 * self.c2r2 + self.c0r0 * self.c1r1 * self.c2r2 ) );
    }
}

impl Array for Matrix4 {
    type Element = f32;

    #[inline]
    fn len() -> usize {
        16
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

impl Zero for Matrix4 {
    fn zero() -> Matrix4 {
        Matrix4::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
    }

    fn is_zero(&self) -> bool {
        self.c0r0 == 0.0 && self.c0r1 == 0.0 && self.c0r2 == 0.0 && self.c0r3 == 0.0 &&
        self.c1r0 == 0.0 && self.c1r1 == 0.0 && self.c1r2 == 0.0 && self.c1r3 == 0.0 &&
        self.c2r0 == 0.0 && self.c2r1 == 0.0 && self.c2r2 == 0.0 && self.c2r3 == 0.0 &&
        self.c3r0 == 0.0 && self.c3r1 == 0.0 && self.c3r2 == 0.0 && self.c3r3 == 0.0
    }
}

impl One for Matrix4 {
    fn one() -> Matrix4 {
        Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0
        )
    }
}

impl fmt::Display for Matrix4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, 
            "\n[{:.2}][{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}][{:.2}]", 
            self.c0r0, self.c1r0, self.c2r0, self.c3r0,
            self.c0r1, self.c1r1, self.c2r1, self.c3r1,
            self.c0r2, self.c1r2, self.c2r2, self.c3r2,
            self.c0r3, self.c1r3, self.c2r3, self.c3r3
        )
    }
}

impl AsRef<[f32; 16]> for Matrix4 {
    fn as_ref(&self) -> &[f32; 16] {
        unsafe { mem::transmute(self) }
    }
}

impl AsRef<[[f32; 4]; 4]> for Matrix4 {
    fn as_ref(&self) -> &[[f32; 4]; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<[f32; 16]> for Matrix4 {
    fn as_mut(&mut self) -> &mut [f32; 16] {
        unsafe { mem::transmute(self) }
    }
}

impl PartialEq for Matrix4 {
    fn eq(&self, other: &Matrix4) -> bool {
        f32::abs(self.c0r0 - other.c0r0) < EPSILON &&
        f32::abs(self.c0r1 - other.c0r1) < EPSILON &&
        f32::abs(self.c0r2 - other.c0r2) < EPSILON &&
        f32::abs(self.c0r3 - other.c0r3) < EPSILON &&
        f32::abs(self.c1r0 - other.c1r0) < EPSILON &&
        f32::abs(self.c1r1 - other.c1r1) < EPSILON &&
        f32::abs(self.c1r2 - other.c1r2) < EPSILON &&
        f32::abs(self.c1r3 - other.c1r3) < EPSILON &&
        f32::abs(self.c2r0 - other.c2r0) < EPSILON &&
        f32::abs(self.c2r1 - other.c2r1) < EPSILON &&
        f32::abs(self.c2r2 - other.c2r2) < EPSILON &&
        f32::abs(self.c2r3 - other.c2r3) < EPSILON &&
        f32::abs(self.c3r0 - other.c3r0) < EPSILON &&
        f32::abs(self.c3r1 - other.c3r1) < EPSILON &&
        f32::abs(self.c3r2 - other.c3r2) < EPSILON &&
        f32::abs(self.c3r3 - other.c3r3) < EPSILON
    }
}

impl ops::Add<Matrix4> for Matrix4 {
    type Output = Matrix4;

    fn add(self, other: Matrix4) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;
        let c0r3 = self.c0r3 + other.c0r3;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;
        let c1r3 = self.c1r3 + other.c1r3;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;
        let c2r3 = self.c2r3 + other.c2r3;

        let c3r0 = self.c3r0 + other.c3r0;
        let c3r1 = self.c3r1 + other.c3r1;
        let c3r2 = self.c3r2 + other.c3r2;
        let c3r3 = self.c3r3 + other.c3r3;

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c3r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl ops::Add<&Matrix4> for Matrix4 {
    type Output = Matrix4;

    fn add(self, other: &Matrix4) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;
        let c0r3 = self.c0r3 + other.c0r3;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;
        let c1r3 = self.c1r3 + other.c1r3;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;
        let c2r3 = self.c2r3 + other.c2r3;

        let c3r0 = self.c3r0 + other.c3r0;
        let c3r1 = self.c3r1 + other.c3r1;
        let c3r2 = self.c3r2 + other.c3r2;
        let c3r3 = self.c3r3 + other.c3r3;

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c3r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl ops::Add<Matrix4> for &Matrix4 {
    type Output = Matrix4;

    fn add(self, other: Matrix4) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;
        let c0r3 = self.c0r3 + other.c0r3;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;
        let c1r3 = self.c1r3 + other.c1r3;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;
        let c2r3 = self.c2r3 + other.c2r3;

        let c3r0 = self.c3r0 + other.c3r0;
        let c3r1 = self.c3r1 + other.c3r1;
        let c3r2 = self.c3r2 + other.c3r2;
        let c3r3 = self.c3r3 + other.c3r3;

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c3r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<'a, 'b> ops::Add<&'a Matrix4> for &'b Matrix4 {
    type Output = Matrix4;

    fn add(self, other: &'a Matrix4) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;
        let c0r3 = self.c0r3 + other.c0r3;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;
        let c1r3 = self.c1r3 + other.c1r3;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;
        let c2r3 = self.c2r3 + other.c2r3;

        let c3r0 = self.c3r0 + other.c3r0;
        let c3r1 = self.c3r1 + other.c3r1;
        let c3r2 = self.c3r2 + other.c3r2;
        let c3r3 = self.c3r3 + other.c3r3;

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c3r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl ops::Sub<Matrix4> for Matrix4 {
    type Output = Matrix4;

    fn sub(self, other: Matrix4) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;
        let c0r3 = self.c0r3 - other.c0r3;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;
        let c1r3 = self.c1r3 - other.c1r3;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;
        let c2r3 = self.c2r3 - other.c2r3;

        let c3r0 = self.c3r0 - other.c3r0;
        let c3r1 = self.c3r1 - other.c3r1;
        let c3r2 = self.c3r2 - other.c3r2;
        let c3r3 = self.c3r3 - other.c3r3;

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c3r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl ops::Sub<&Matrix4> for Matrix4 {
    type Output = Matrix4;

    fn sub(self, other: &Matrix4) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;
        let c0r3 = self.c0r3 - other.c0r3;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;
        let c1r3 = self.c1r3 - other.c1r3;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;
        let c2r3 = self.c2r3 - other.c2r3;

        let c3r0 = self.c3r0 - other.c3r0;
        let c3r1 = self.c3r1 - other.c3r1;
        let c3r2 = self.c3r2 - other.c3r2;
        let c3r3 = self.c3r3 - other.c3r3;

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c3r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl ops::Sub<Matrix4> for &Matrix4 {
    type Output = Matrix4;

    fn sub(self, other: Matrix4) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;
        let c0r3 = self.c0r3 - other.c0r3;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;
        let c1r3 = self.c1r3 - other.c1r3;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;
        let c2r3 = self.c2r3 - other.c2r3;

        let c3r0 = self.c3r0 - other.c3r0;
        let c3r1 = self.c3r1 - other.c3r1;
        let c3r2 = self.c3r2 - other.c3r2;
        let c3r3 = self.c3r3 - other.c3r3;

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c3r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<'a, 'b> ops::Sub<&'a Matrix4> for &'b Matrix4 {
    type Output = Matrix4;

    fn sub(self, other: &'a Matrix4) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;
        let c0r3 = self.c0r3 - other.c0r3;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;
        let c1r3 = self.c1r3 - other.c1r3;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;
        let c2r3 = self.c2r3 - other.c2r3;

        let c3r0 = self.c3r0 - other.c3r0;
        let c3r1 = self.c3r1 - other.c3r1;
        let c3r2 = self.c3r2 - other.c3r2;
        let c3r3 = self.c3r3 - other.c3r3;

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c3r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl ops::Mul<Vector4> for Matrix4 {
    type Output = Vector4;

    fn mul(self, other: Vector4) -> Self::Output {
        let x = self.c0r0 * other[0] + self.c1r0 * other[1] + self.c2r0 * other[2] + self.c3r0 * other[3];
        let y = self.c0r1 * other[0] + self.c1r1 * other[1] + self.c2r1 * other[2] + self.c3r1 * other[3];
        let z = self.c0r2 * other[0] + self.c1r2 * other[1] + self.c2r2 * other[2] + self.c3r2 * other[3];
        let w = self.c0r3 * other[0] + self.c1r3 * other[1] + self.c2r3 * other[2] + self.c3r3 * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl ops::Mul<&Matrix4> for Matrix4 {
    type Output = Matrix4;

    fn mul(self, other: &Matrix4) -> Self::Output {
        let mut m = Matrix4::zero();

        m.c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2 + self.c3r0 * other.c0r3;
        m.c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2 + self.c3r1 * other.c0r3;
        m.c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2 + self.c3r2 * other.c0r3;
        m.c0r3 = self.c0r3 * other.c0r0 + self.c1r3 * other.c0r1 + self.c2r3 * other.c0r2 + self.c3r3 * other.c0r3;
        m.c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2 + self.c3r0 * other.c1r3;
        m.c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2 + self.c3r1 * other.c1r3;
        m.c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2 + self.c3r2 * other.c1r3;
        m.c1r3 = self.c0r3 * other.c1r0 + self.c1r3 * other.c1r1 + self.c2r3 * other.c1r2 + self.c3r3 * other.c1r3;
        m.c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2 + self.c3r0 * other.c2r3;
        m.c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2 + self.c3r1 * other.c2r3;
        m.c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2 + self.c3r2 * other.c2r3;
        m.c2r3 = self.c0r3 * other.c2r0 + self.c1r3 * other.c2r1 + self.c2r3 * other.c2r2 + self.c3r3 * other.c2r3;
        m.c3r0 = self.c0r0 * other.c3r0 + self.c1r0 * other.c3r1 + self.c2r0 * other.c3r2 + self.c3r0 * other.c3r3;
        m.c3r1 = self.c0r1 * other.c3r0 + self.c1r1 * other.c3r1 + self.c2r1 * other.c3r2 + self.c3r1 * other.c3r3;
        m.c3r2 = self.c0r2 * other.c3r0 + self.c1r2 * other.c3r1 + self.c2r2 * other.c3r2 + self.c3r2 * other.c3r3;
        m.c3r3 = self.c0r3 * other.c3r0 + self.c1r3 * other.c3r1 + self.c2r3 * other.c3r2 + self.c3r3 * other.c3r3;

        m
    }
}

impl<'a, 'b> ops::Mul<&'a Matrix4> for &'b Matrix4 {
    type Output = Matrix4;

    fn mul(self, other: &'a Matrix4) -> Self::Output {
        let mut m = Matrix4::zero();

        m.c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2 + self.c3r0 * other.c0r3;
        m.c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2 + self.c3r1 * other.c0r3;
        m.c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2 + self.c3r2 * other.c0r3;
        m.c0r3 = self.c0r3 * other.c0r0 + self.c1r3 * other.c0r1 + self.c2r3 * other.c0r2 + self.c3r3 * other.c0r3;
        m.c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2 + self.c3r0 * other.c1r3;
        m.c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2 + self.c3r1 * other.c1r3;
        m.c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2 + self.c3r2 * other.c1r3;
        m.c1r3 = self.c0r3 * other.c1r0 + self.c1r3 * other.c1r1 + self.c2r3 * other.c1r2 + self.c3r3 * other.c1r3;
        m.c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2 + self.c3r0 * other.c2r3;
        m.c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2 + self.c3r1 * other.c2r3;
        m.c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2 + self.c3r2 * other.c2r3;
        m.c2r3 = self.c0r3 * other.c2r0 + self.c1r3 * other.c2r1 + self.c2r3 * other.c2r2 + self.c3r3 * other.c2r3;
        m.c3r0 = self.c0r0 * other.c3r0 + self.c1r0 * other.c3r1 + self.c2r0 * other.c3r2 + self.c3r0 * other.c3r3;
        m.c3r1 = self.c0r1 * other.c3r0 + self.c1r1 * other.c3r1 + self.c2r1 * other.c3r2 + self.c3r1 * other.c3r3;
        m.c3r2 = self.c0r2 * other.c3r0 + self.c1r2 * other.c3r1 + self.c2r2 * other.c3r2 + self.c3r2 * other.c3r3;
        m.c3r3 = self.c0r3 * other.c3r0 + self.c1r3 * other.c3r1 + self.c2r3 * other.c3r2 + self.c3r3 * other.c3r3;

        m
    }
}

impl ops::Mul<Matrix4> for Matrix4 {
    type Output = Matrix4;

    fn mul(self, other: Matrix4) -> Self::Output {
        let mut m = Matrix4::zero();

        m.c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2 + self.c3r0 * other.c0r3;
        m.c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2 + self.c3r1 * other.c0r3;
        m.c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2 + self.c3r2 * other.c0r3;
        m.c0r3 = self.c0r3 * other.c0r0 + self.c1r3 * other.c0r1 + self.c2r3 * other.c0r2 + self.c3r3 * other.c0r3;
        m.c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2 + self.c3r0 * other.c1r3;
        m.c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2 + self.c3r1 * other.c1r3;
        m.c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2 + self.c3r2 * other.c1r3;
        m.c1r3 = self.c0r3 * other.c1r0 + self.c1r3 * other.c1r1 + self.c2r3 * other.c1r2 + self.c3r3 * other.c1r3;
        m.c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2 + self.c3r0 * other.c2r3;
        m.c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2 + self.c3r1 * other.c2r3;
        m.c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2 + self.c3r2 * other.c2r3;
        m.c2r3 = self.c0r3 * other.c2r0 + self.c1r3 * other.c2r1 + self.c2r3 * other.c2r2 + self.c3r3 * other.c2r3;
        m.c3r0 = self.c0r0 * other.c3r0 + self.c1r0 * other.c3r1 + self.c2r0 * other.c3r2 + self.c3r0 * other.c3r3;
        m.c3r1 = self.c0r1 * other.c3r0 + self.c1r1 * other.c3r1 + self.c2r1 * other.c3r2 + self.c3r1 * other.c3r3;
        m.c3r2 = self.c0r2 * other.c3r0 + self.c1r2 * other.c3r1 + self.c2r2 * other.c3r2 + self.c3r2 * other.c3r3;
        m.c3r3 = self.c0r3 * other.c3r0 + self.c1r3 * other.c3r1 + self.c2r3 * other.c3r2 + self.c3r3 * other.c3r3;

        m
    }
}

impl ops::Mul<f32> for Matrix4 {
    type Output = Matrix4;

    fn mul(self, other: f32) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c0r2 = self.c0r2 * other;
        let c0r3 = self.c0r3 * other;

        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;
        let c1r2 = self.c1r2 * other;
        let c1r3 = self.c1r3 * other;
        
        let c2r0 = self.c2r0 * other;
        let c2r1 = self.c2r1 * other;
        let c2r2 = self.c2r2 * other;
        let c2r3 = self.c2r3 * other;

        let c3r0 = self.c3r0 * other;
        let c3r1 = self.c3r1 * other;
        let c3r2 = self.c3r2 * other;
        let c3r3 = self.c3r3 * other;

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl ops::Mul<f32> for &Matrix4 {
    type Output = Matrix4;

    fn mul(self, other: f32) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c0r2 = self.c0r2 * other;
        let c0r3 = self.c0r3 * other;

        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;
        let c1r2 = self.c1r2 * other;
        let c1r3 = self.c1r3 * other;
        
        let c2r0 = self.c2r0 * other;
        let c2r1 = self.c2r1 * other;
        let c2r2 = self.c2r2 * other;
        let c2r3 = self.c2r3 * other;

        let c3r0 = self.c3r0 * other;
        let c3r1 = self.c3r1 * other;
        let c3r2 = self.c3r2 * other;
        let c3r3 = self.c3r3 * other;

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl ops::Div<f32> for Matrix4 {
    type Output = Matrix4;

    fn div(self, other: f32) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c0r2 = self.c0r2 / other;
        let c0r3 = self.c0r3 / other;

        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;
        let c1r2 = self.c1r2 / other;
        let c1r3 = self.c1r3 / other;
        
        let c2r0 = self.c2r0 / other;
        let c2r1 = self.c2r1 / other;
        let c2r2 = self.c2r2 / other;
        let c2r3 = self.c2r3 / other;

        let c3r0 = self.c3r0 / other;
        let c3r1 = self.c3r1 / other;
        let c3r2 = self.c3r2 / other;
        let c3r3 = self.c3r3 / other;

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl ops::Div<f32> for &Matrix4 {
    type Output = Matrix4;

    fn div(self, other: f32) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c0r2 = self.c0r2 / other;
        let c0r3 = self.c0r3 / other;

        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;
        let c1r2 = self.c1r2 / other;
        let c1r3 = self.c1r3 / other;
        
        let c2r0 = self.c2r0 / other;
        let c2r1 = self.c2r1 / other;
        let c2r2 = self.c2r2 / other;
        let c2r3 = self.c2r3 / other;

        let c3r0 = self.c3r0 / other;
        let c3r1 = self.c3r1 / other;
        let c3r2 = self.c3r2 / other;
        let c3r3 = self.c3r3 / other;

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}


#[cfg(test)]
mod matrix2_tests {
    use std::slice::Iter;
    use vector::Vector2;
    use super::Matrix2;
    use traits::{One, Zero};


    struct TestCase {
        c: f32,
        a_mat: Matrix2,
        b_mat: Matrix2,
        expected: Matrix2,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase {
                    c: 802.3435169,
                    a_mat: Matrix2::new(80.0,  23.43,     426.1,   23.5724),
                    b_mat: Matrix2::new(36.84, 427.46894, 7.04217, 61.891390),
                    expected: Matrix2::new(185091.72, 10939.63, 26935.295, 1623.9266),
                },
                TestCase {
                    c: 6.2396,
                    a_mat: Matrix2::one(),
                    b_mat: Matrix2::one(),
                    expected: Matrix2::one(),
                },
                TestCase {
                    c: 6.2396,
                    a_mat: Matrix2::zero(),
                    b_mat: Matrix2::zero(),
                    expected: Matrix2::zero(),
                },
                TestCase {
                    c:  14.5093,
                    a_mat: Matrix2::new(68.32, 0.0, 0.0, 37.397),
                    b_mat: Matrix2::new(57.72, 0.0, 0.0, 9.5433127),
                    expected: Matrix2::new(3943.4304, 0.0, 0.0, 356.89127),
                },
            ]
        }
    }

    #[test]
    fn test_mat_times_identity_equals_mat() {
        for test in test_cases().iter() {
            let a_mat_times_identity = test.a_mat * Matrix2::one();
            let b_mat_times_identity = test.b_mat * Matrix2::one();

            assert_eq!(a_mat_times_identity, test.a_mat);
            assert_eq!(b_mat_times_identity, test.b_mat);
        }
    }

    #[test]
    fn test_mat_times_zero_equals_zero() {
        for test in test_cases().iter() {
            let a_mat_times_zero = test.a_mat * Matrix2::zero();
            let b_mat_times_zero = test.b_mat * Matrix2::zero();

            assert_eq!(a_mat_times_zero, Matrix2::zero());
            assert_eq!(b_mat_times_zero, Matrix2::zero());
        }
    }

    #[test]
    fn test_zero_times_mat_equals_zero() {
        for test in test_cases().iter() {
            let zero_times_a_mat = Matrix2::zero() * test.a_mat;
            let zero_times_b_mat = Matrix2::zero() * test.b_mat;

            assert_eq!(zero_times_a_mat, Matrix2::zero());
            assert_eq!(zero_times_b_mat, Matrix2::zero());
        }
    }

    #[test]
    fn test_mat_times_identity_equals_identity_times_mat() {
        for test in test_cases().iter() {
            let a_mat_times_identity = test.a_mat * Matrix2::one();
            let identity_times_a_mat = Matrix2::one() * test.a_mat;
            let b_mat_times_identity = test.b_mat * Matrix2::one();
            let identity_times_b_mat = Matrix2::one() * test.b_mat;

            assert_eq!(a_mat_times_identity, identity_times_a_mat);
            assert_eq!(b_mat_times_identity, identity_times_b_mat);
        }
    }

    #[test]
    fn test_mat_transpose_transpose_equals_mat() {
        for test in test_cases().iter() {
            let a_mat_tr_tr = test.a_mat.transpose().transpose();
            let b_mat_tr_tr = test.b_mat.transpose().transpose();
            
            assert_eq!(a_mat_tr_tr, test.a_mat);
            assert_eq!(b_mat_tr_tr, test.b_mat);
        }
    }

    #[test]
    fn test_identity_transpose_equals_identity() {
        let identity = Matrix2::one();
        let identity_tr = identity.transpose();
            
        assert_eq!(identity, identity_tr);
    }

    #[test]
    fn test_matrix_multiplication() {
        for test in test_cases().iter() {
            let result = test.a_mat * test.b_mat;
            let expected = test.expected;

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector2::new(1.0, 2.0);
        let c1 = Vector2::new(3.0, 4.0);
        let expected = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        let result = Matrix2::from_cols(c0, c1);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_constant_times_identity_is_constant_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix2::one();
        let expected = Matrix2::new(c, 0.0, 0.0, c);

        assert_eq!(id * c, expected);
    }

    #[test]
    fn test_identity_divide_constant_is_constant_inverse_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix2::one();
        let expected = Matrix2::new(1.0/c, 0.0, 0.0, 1.0/c);

        assert_eq!(id / c, expected);
    }

    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero = Matrix2::zero();
        let matrix = Matrix2::new(36.84, 427.46, 7.47, 61.89);

        assert_eq!(matrix + zero, matrix);
    }

        #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero = Matrix2::zero();
        let matrix = Matrix2::new(36.84, 427.46, 7.47, 61.89);

        assert_eq!(zero + matrix, matrix);
    }
}


#[cfg(test)]
mod matrix3_tests {
    use std::slice::Iter;
    use vector::Vector3;
    use super::Matrix3;
    use traits::{One, Zero};

    struct TestCase {
        c: f32,
        a_mat: Matrix3,
        b_mat: Matrix3,
        expected: Matrix3,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase {
                    c: 802.3435169,
                    a_mat: Matrix3::new(80.0, 426.1, 43.393, 23.43, 23.5724, 1.27, 81.439, 12.19, 43.36),
                    b_mat: Matrix3::new(36.84, 7.04217, 5.74, 427.46894, 61.89139, 96.27, 152.66, 86.333, 26.71),
                    expected: Matrix3::new(3579.6579, 15933.496, 1856.4281, 43487.7660, 184776.9752, 22802.0289, 16410.8178, 67409.1000, 7892.1646),
                },
                TestCase {
                    c: 6.2396,
                    a_mat: Matrix3::one(),
                    b_mat: Matrix3::one(),
                    expected: Matrix3::one(),
                },
                TestCase {
                    c: 6.2396,
                    a_mat: Matrix3::zero(),
                    b_mat: Matrix3::zero(),
                    expected: Matrix3::zero(),
                },
                TestCase {
                    c:  14.5093,
                    a_mat: Matrix3::new(68.32, 0.0, 0.0, 0.0, 37.397, 0.0, 0.0, 0.0, 43.393),
                    b_mat: Matrix3::new(57.72, 0.0, 0.0, 0.0, 9.5433127, 0.0, 0.0, 0.0, 12.19),
                    expected: Matrix3::new(3943.4304, 0.0, 0.0, 0.0, 356.89127, 0.0, 0.0, 0.0, 528.96067),
                },
            ]
        }
    }

    #[test]
    fn test_mat_times_identity_equals_mat() {
        for test in test_cases().iter() {
            let a_mat_times_identity = test.a_mat * Matrix3::one();
            let b_mat_times_identity = test.b_mat * Matrix3::one();

            assert_eq!(a_mat_times_identity, test.a_mat);
            assert_eq!(b_mat_times_identity, test.b_mat);
        }
    }

    #[test]
    fn test_mat_times_zero_equals_zero() {
        for test in test_cases().iter() {
            let a_mat_times_zero = test.a_mat * Matrix3::zero();
            let b_mat_times_zero = test.b_mat * Matrix3::zero();

            assert_eq!(a_mat_times_zero, Matrix3::zero());
            assert_eq!(b_mat_times_zero, Matrix3::zero());
        }
    }

    #[test]
    fn test_zero_times_mat_equals_zero() {
        for test in test_cases().iter() {
            let zero_times_a_mat = Matrix3::zero() * test.a_mat;
            let zero_times_b_mat = Matrix3::zero() * test.b_mat;

            assert_eq!(zero_times_a_mat, Matrix3::zero());
            assert_eq!(zero_times_b_mat, Matrix3::zero());
        }
    }

    #[test]
    fn test_mat_times_identity_equals_identity_times_mat() {
        for test in test_cases().iter() {
            let a_mat_times_identity = test.a_mat * Matrix3::one();
            let identity_times_a_mat = Matrix3::one() * test.a_mat;
            let b_mat_times_identity = test.b_mat * Matrix3::one();
            let identity_times_b_mat = Matrix3::one() * test.b_mat;

            assert_eq!(a_mat_times_identity, identity_times_a_mat);
            assert_eq!(b_mat_times_identity, identity_times_b_mat);
        }
    }

    #[test]
    fn test_mat_transpose_transpose_equals_mat() {
        for test in test_cases().iter() {
            let a_mat_tr_tr = test.a_mat.transpose().transpose();
            let b_mat_tr_tr = test.b_mat.transpose().transpose();
            
            assert_eq!(a_mat_tr_tr, test.a_mat);
            assert_eq!(b_mat_tr_tr, test.b_mat);
        }
    }

    #[test]
    fn test_identity_transpose_equals_identity() {
        let identity = Matrix3::one();
        let identity_tr = identity.transpose();
            
        assert_eq!(identity, identity_tr);
    }

    #[test]
    fn test_matrix_multiplication() {
        for test in test_cases().iter() {
            let result = test.a_mat * test.b_mat;
            let expected = test.expected;

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector3::new(1.0, 2.0, 3.0);
        let c1 = Vector3::new(4.0, 5.0, 6.0);
        let c2 = Vector3::new(7.0, 8.0, 9.0);
        let expected = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let result = Matrix3::from_cols(c0, c1, c2);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_constant_times_identity_is_constant_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix3::one();
        let expected = Matrix3::new(c, 0.0, 0.0, 0.0, c, 0.0, 0.0, 0.0, c);

        assert_eq!(id * c, expected);
    }

    #[test]
    fn test_identity_divide_constant_is_constant_inverse_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix3::one();
        let expected = Matrix3::new(1.0/c, 0.0, 0.0, 0.0, 1.0/c, 0.0, 0.0, 0.0, 1.0/c);

        assert_eq!(id / c, expected);
    }

    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero = Matrix3::zero();
        let matrix = Matrix3::new(80.0, 426.1, 43.393, 23.43, 23.5724, 1.27, 81.439, 12.19, 43.36);

        assert_eq!(matrix + zero, matrix);
    }

        #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero = Matrix3::zero();
        let matrix = Matrix3::new(80.0, 426.1, 43.393, 23.43, 23.5724, 1.27, 81.439, 12.19, 43.36);

        assert_eq!(zero + matrix, matrix);
    }
}

#[cfg(test)]
mod matrix4_tests {
    use std::slice::Iter;
    use vector::{Vector3, Vector4};
    use super::{Matrix4};
    use traits::{One, Zero};


    struct TestCase {
        c: f32,
        a_mat: Matrix4,
        b_mat: Matrix4,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase {
                    c: 802.3435169,
                    a_mat: Matrix4::new(
                        80.0,   23.43,   43.569,  6.741, 
                        426.1,  23.5724, 27.6189, 13.90,
                        4.2219, 258.083, 31.70,   42.17, 
                        70.0,   49.0,    95.0,    89.9138
                    ),
                    b_mat: Matrix4::new(
                        36.84,   427.46894, 8827.1983, 89.5049494, 
                        7.04217, 61.891390, 56.31,     89.0, 
                        72.0,    936.5,     413.80,    50.311160,  
                        37.6985,  311.8,    60.81,     73.8393
                    ),
                },
                TestCase {
                    c: 6.2396,
                    a_mat: Matrix4::one(),
                    b_mat: Matrix4::one(),
                },
                TestCase {
                    c: 6.2396,
                    a_mat: Matrix4::zero(),
                    b_mat: Matrix4::zero(),
                },
                TestCase {
                    c:  14.5093,
                    a_mat: Matrix4::new(
                        68.32, 0.0,    0.0,   0.0,
                        0.0,   37.397, 0.0,   0.0,
                        0.0,   0.0,    9.483, 0.0,
                        0.0,   0.0,    0.0,   887.710
                    ),
                    b_mat: Matrix4::new(
                        57.72, 0.0,       0.0,       0.0, 
                        0.0,   9.5433127, 0.0,       0.0, 
                        0.0,   0.0,       86.731265, 0.0,
                        0.0,   0.0,       0.0,       269.1134546
                    )
                },
            ]
        }
    }

    #[test]
    fn test_mat_times_identity_equals_mat() {
        for test in test_cases().iter() {
            let a_mat_times_identity = test.a_mat * Matrix4::one();
            let b_mat_times_identity = test.b_mat * Matrix4::one();

            assert_eq!(a_mat_times_identity, test.a_mat);
            assert_eq!(b_mat_times_identity, test.b_mat);
        }
    }

    #[test]
    fn test_mat_times_zero_equals_zero() {
        for test in test_cases().iter() {
            let a_mat_times_zero = test.a_mat * Matrix4::zero();
            let b_mat_times_zero = test.b_mat * Matrix4::zero();

            assert_eq!(a_mat_times_zero, Matrix4::zero());
            assert_eq!(b_mat_times_zero, Matrix4::zero());
        }
    }

    #[test]
    fn test_zero_times_mat_equals_zero() {
        for test in test_cases().iter() {
            let zero_times_a_mat = Matrix4::zero() * test.a_mat;
            let zero_times_b_mat = Matrix4::zero() * test.b_mat;

            assert_eq!(zero_times_a_mat, Matrix4::zero());
            assert_eq!(zero_times_b_mat, Matrix4::zero());
        }
    }

    #[test]
    fn test_mat_times_identity_equals_identity_times_mat() {
        for test in test_cases().iter() {
            let a_mat_times_identity = test.a_mat * Matrix4::one();
            let identity_times_a_mat = Matrix4::one() * test.a_mat;
            let b_mat_times_identity = test.b_mat * Matrix4::one();
            let identity_times_b_mat = Matrix4::one() * test.b_mat;

            assert_eq!(a_mat_times_identity, identity_times_a_mat);
            assert_eq!(b_mat_times_identity, identity_times_b_mat);
        }
    }

    #[test]
    fn test_mat_times_mat_inverse_equals_identity() {
        for test in test_cases().iter() {
            let identity = Matrix4::one();
            if test.a_mat.is_invertible() {
                let a_mat_inverse = test.a_mat.inverse();
                assert_eq!(a_mat_inverse * test.a_mat, identity);
            }
            if test.b_mat.is_invertible() {
                let b_mat_inverse = test.b_mat.inverse();
                assert_eq!(b_mat_inverse * test.b_mat, identity);
            }
        }
    }

    #[test]
    fn test_mat_inverse_times_mat_equals_identity() {
        for test in test_cases().iter() {
            let identity = Matrix4::one();
            if test.a_mat.is_invertible() {
                let a_mat_inverse = test.a_mat.inverse();
                assert_eq!(test.a_mat * a_mat_inverse, identity);
            }
            if test.b_mat.is_invertible() {
                let b_mat_inverse = test.b_mat.inverse();
                assert_eq!(test.b_mat * b_mat_inverse, identity);
            }
        }
    }

    #[test]
    fn test_mat_transpose_transpose_equals_mat() {
        for test in test_cases().iter() {
            let a_mat_tr_tr = test.a_mat.transpose().transpose();
            let b_mat_tr_tr = test.b_mat.transpose().transpose();
            
            assert_eq!(a_mat_tr_tr, test.a_mat);
            assert_eq!(b_mat_tr_tr, test.b_mat);
        }
    }

    #[test]
    fn test_identity_transpose_equals_identity() {
        let identity = Matrix4::one();
        let identity_tr = identity.transpose();
            
        assert_eq!(identity, identity_tr);
    }

    #[test]
    fn test_identity_mat4_translates_vector_along_vector() {
        let v = Vector3::from((2.0, 2.0, 2.0));
        let trans_mat = Matrix4::from_translation(v);
        let zero_vec4 = Vector4::from((0.0, 0.0, 0.0, 1.0));
        let zero_vec3 = Vector3::from((0.0, 0.0, 0.0));

        let result = trans_mat * zero_vec4;
        assert_eq!(result, Vector4::from((zero_vec3 + v, 1.0)));
    }

    #[test]
    fn test_constant_times_identity_is_identity_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix4::one();
        let expected = Matrix4::new(
            c, 0.0, 0.0, 0.0, 0.0, c, 0.0, 0.0, 0.0, 0.0, c, 0.0, 0.0, 0.0, 0.0, c
        );

        assert_eq!(id * c, expected);
    }

    #[test]
    fn test_identity_divide_constant_is_constant_inverse_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix4::one();
        let expected = Matrix4::new(
            1.0/c, 0.0, 0.0, 0.0, 0.0, 1.0/c, 0.0, 0.0, 0.0, 0.0, 1.0/c, 0.0, 0.0, 0.0, 0.0, 1.0/c
        );

        assert_eq!(id / c, expected);
    }

        #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero = Matrix4::zero();
        let matrix = Matrix4::new(
            36.84,   427.46894, 8827.1983, 89.5049494, 
            7.04217, 61.891390, 56.31,     89.0, 
            72.0,    936.5,     413.80,    50.311160,  
            37.6985,  311.8,    60.81,     73.8393
        );

        assert_eq!(matrix + zero, matrix);
    }

        #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero = Matrix4::zero();
        let matrix = Matrix4::new(
            36.84,   427.46894, 8827.1983, 89.5049494, 
            7.04217, 61.891390, 56.31,     89.0, 
            72.0,    936.5,     413.80,    50.311160,  
            37.6985,  311.8,    60.81,     73.8393
        );

        assert_eq!(zero + matrix, matrix);
    }
}
