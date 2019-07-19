use traits::{Array, Zero, VectorSpace, MetricSpace, DotProduct, Lerp};
use std::fmt;
use std::mem;
use std::ops;
use std::cmp;


const EPSILON: f32 = 0.00001; 


///
/// The `Matrix2` type represents 2x2 matrices in column-major order.
///
#[derive(Copy, Clone, Debug)]
pub struct Matrix2 {
    m: [f32; 4],
}

impl Matrix2 {
    pub fn new(m11: f32, m12: f32, m21: f32, m22: f32) -> Matrix2 {
        Matrix2 {
            m: [
                m11, m12, // Column 1
                m21, m22, // Column 2
            ]
        }
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
            self.m[0], self.m[2],
            self.m[1], self.m[3],
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
        self.m.as_ptr()
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut f32 {
        self.m.as_mut_ptr()
    }
}

impl fmt::Display for Matrix2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, 
            "\n[{:.2}][{:.2}]\n[{:.2}][{:.2}]", 
            self.m[0], self.m[2],
            self.m[1], self.m[3],
        )
    }
}

impl AsRef<[f32; 4]> for Matrix2 {
    fn as_ref(&self) -> &[f32; 4] {
        &self.m
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
        &mut self.m
    }
}


impl ops::Add<Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn add(self, other: Matrix2) -> Self::Output {
        let m11 = self.m[0] + other.m[0];
        let m21 = self.m[1] + other.m[1];
        let m12 = self.m[2] + other.m[2];
        let m22 = self.m[3] + other.m[3];

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl<'a> ops::Add<&'a Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn add(self, other: &'a Matrix2) -> Self::Output {
        let m11 = self.m[0] + other.m[0];
        let m21 = self.m[1] + other.m[1];
        let m12 = self.m[2] + other.m[2];
        let m22 = self.m[3] + other.m[3];

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl<'a> ops::Add<Matrix2> for &'a Matrix2 {
    type Output = Matrix2;

    fn add(self, other: Matrix2) -> Self::Output {
        let m11 = self.m[0] + other.m[0];
        let m21 = self.m[1] + other.m[1];
        let m12 = self.m[2] + other.m[2];
        let m22 = self.m[3] + other.m[3];

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl<'a, 'b> ops::Add<&'a Matrix2> for &'b Matrix2 {
    type Output = Matrix2;

    fn add(self, other: &'a Matrix2) -> Self::Output {
        let m11 = self.m[0] + other.m[0];
        let m21 = self.m[1] + other.m[1];
        let m12 = self.m[2] + other.m[2];
        let m22 = self.m[3] + other.m[3];

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl ops::Sub<Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn sub(self, other: Matrix2) -> Self::Output {
        let m11 = self.m[0] + other.m[0];
        let m21 = self.m[1] + other.m[1];
        let m12 = self.m[2] + other.m[2];
        let m22 = self.m[3] + other.m[3];

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl<'a> ops::Sub<&'a Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn sub(self, other: &'a Matrix2) -> Self::Output {
        let m11 = self.m[0] - other.m[0];
        let m21 = self.m[1] - other.m[1];
        let m12 = self.m[2] - other.m[2];
        let m22 = self.m[3] - other.m[3];

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl<'a> ops::Sub<Matrix2> for &'a Matrix2 {
    type Output = Matrix2;

    fn sub(self, other: Matrix2) -> Self::Output {
        let m11 = self.m[0] - other.m[0];
        let m21 = self.m[1] - other.m[1];
        let m12 = self.m[2] - other.m[2];
        let m22 = self.m[3] - other.m[3];

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl<'a, 'b> ops::Sub<&'a Matrix2> for &'b Matrix2 {
    type Output = Matrix2;

    fn sub(self, other: &'a Matrix2) -> Self::Output {
        let m11 = self.m[0] - other.m[0];
        let m21 = self.m[1] - other.m[1];
        let m12 = self.m[2] - other.m[2];
        let m22 = self.m[3] - other.m[3];

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl<'a> ops::Mul<&'a Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn mul(self, other: &'a Matrix2) -> Self::Output {
        let m11 = self.m[0] * other.m[0] + self.m[2] * other.m[1];
        let m21 = self.m[1] * other.m[0] + self.m[3] * other.m[1];
        let m12 = self.m[0] * other.m[2] + self.m[2] * other.m[3];
        let m22 = self.m[1] * other.m[2] + self.m[3] * other.m[3];

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl<'a, 'b> ops::Mul<&'a Matrix2> for &'b Matrix2 {
    type Output = Matrix2;

    fn mul(self, other: &'a Matrix2) -> Self::Output {
        let m11 = self.m[0] * other.m[0] + self.m[2] * other.m[1];
        let m21 = self.m[1] * other.m[0] + self.m[3] * other.m[1];
        let m12 = self.m[0] * other.m[2] + self.m[2] * other.m[3];
        let m22 = self.m[1] * other.m[2] + self.m[3] * other.m[3];

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl ops::Mul<Matrix2> for Matrix2 {
    type Output = Matrix2;

    fn mul(self, other: Matrix2) -> Self::Output {
        let m11 = self.m[0] * other.m[0] + self.m[2] * other.m[1];
        let m21 = self.m[1] * other.m[0] + self.m[3] * other.m[1];
        let m12 = self.m[0] * other.m[2] + self.m[2] * other.m[3];
        let m22 = self.m[1] * other.m[2] + self.m[3] * other.m[3];

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl<'a> ops::Mul<Matrix2> for &'a Matrix2 {
    type Output = Matrix2;

    fn mul(self, other: Matrix2) -> Self::Output {
        let m11 = self.m[0] * other.m[0] + self.m[2] * other.m[1];
        let m21 = self.m[1] * other.m[0] + self.m[3] * other.m[1];
        let m12 = self.m[0] * other.m[2] + self.m[2] * other.m[3];
        let m22 = self.m[1] * other.m[2] + self.m[3] * other.m[3];

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl ops::Mul<f32> for Matrix2 {
    type Output = Matrix2;

    fn mul(self, other: f32) -> Self::Output {
        let m11 = self.m[0] * other;
        let m21 = self.m[1] * other;
        let m12 = self.m[2] * other;
        let m22 = self.m[3] * other;

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl<'a> ops::Mul<f32> for &'a Matrix2 {
    type Output = Matrix2;

    fn mul(self, other: f32) -> Self::Output {
        let m11 = self.m[0] * other;
        let m21 = self.m[1] * other;
        let m12 = self.m[2] * other;
        let m22 = self.m[3] * other;

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl ops::Div<f32> for Matrix2 {
    type Output = Matrix2;

    fn div(self, other: f32) -> Self::Output {
        let m11 = self.m[0] / other;
        let m21 = self.m[1] / other;
        let m12 = self.m[2] / other;
        let m22 = self.m[3] / other;

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

impl<'a> ops::Div<f32> for &'a Matrix2 {
    type Output = Matrix2;

    fn div(self, other: f32) -> Self::Output {
        let m11 = self.m[0] / other;
        let m21 = self.m[1] / other;
        let m12 = self.m[2] / other;
        let m22 = self.m[3] / other;

        Matrix2::new(
            m11, m21,
            m12, m22
        )
    }
}

