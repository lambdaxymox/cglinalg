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


///
/// The `Matrix3` type represents 3x3 matrices in column-major order.
///
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Matrix3 {
    m: [f32; 9],
}

impl Matrix3 {
    pub fn new(
        m11: f32, m12: f32, m13: f32, 
        m21: f32, m22: f32, m23: f32, 
        m31: f32, m32: f32, m33: f32) -> Matrix3 {

        Matrix3 {
            m: [
                m11, m12, m13, // Column 1
                m21, m22, m23, // Column 2
                m31, m32, m33  // Column 3
            ]
        }
    }

    /// Create a 3x3 matrix from a triple of three-dimensional column vectors.
    pub fn from_cols(c0: Vector3, c1: Vector3, c2: Vector3) -> Matrix3 {
        Matrix3 {
            m: [
                c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z
            ]
        }
    }

    /// Generate a 3x3 matrix of zeros.
    pub fn zero() -> Matrix3 {
        Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Generate a 3x3 diagonal matrix of ones.
    pub fn one() -> Matrix3 {
        Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    }

    /// Compute the transpose of a 3x3 matrix.
    pub fn transpose(&self) -> Matrix3 {
        Matrix3::new(
            self.m[0], self.m[3], self.m[6],  
            self.m[1], self.m[4], self.m[7],  
            self.m[2], self.m[5], self.m[8]
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
        self.m.as_ptr()
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut f32 {
        self.m.as_mut_ptr()
    }
}

impl fmt::Display for Matrix3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, 
            "\n[{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}]", 
            self.m[0], self.m[3], self.m[6],
            self.m[1], self.m[4], self.m[7],
            self.m[2], self.m[5], self.m[8],
        )
    }
}

impl AsRef<[f32; 9]> for Matrix3 {
    fn as_ref(&self) -> &[f32; 9] {
        &self.m
    }
}

impl AsRef<[[f32; 3]; 3]> for Matrix3 {
    fn as_ref(&self) -> &[[f32; 3]; 3] {
        unsafe {
            mem::transmute(self)
        }
    }
}

impl AsMut<[f32; 9]> for Matrix3 {
    fn as_mut(&mut self) -> &mut [f32; 9] {
        &mut self.m
    }
}

impl<'a> ops::Mul<&'a Matrix3> for Matrix3 {
    type Output = Matrix3;

    fn mul(self, other: &'a Matrix3) -> Self::Output {
        let m11 = self.m[0] * other.m[0] + self.m[3] * other.m[1] + self.m[6] * other.m[2]; // 0
        let m21 = self.m[1] * other.m[0] + self.m[4] * other.m[1] + self.m[7] * other.m[2]; // 1
        let m31 = self.m[2] * other.m[0] + self.m[5] * other.m[1] + self.m[8] * other.m[2]; // 2

        let m12 = self.m[0] * other.m[3] + self.m[3] * other.m[4] + self.m[6] * other.m[5]; // 3
        let m22 = self.m[1] * other.m[3] + self.m[4] * other.m[4] + self.m[7] * other.m[5]; // 4
        let m32 = self.m[2] * other.m[3] + self.m[5] * other.m[4] + self.m[8] * other.m[5]; // 5

        let m13 = self.m[0] * other.m[6] + self.m[3] * other.m[7] + self.m[6] * other.m[8]; // 6
        let m23 = self.m[1] * other.m[6] + self.m[4] * other.m[7] + self.m[7] * other.m[8]; // 7
        let m33 = self.m[2] * other.m[6] + self.m[5] * other.m[7] + self.m[8] * other.m[8]; // 8

        Matrix3::new(
            m11, m21, m31,
            m12, m22, m32,
            m13, m23, m33,
        )
    }
}

impl<'a, 'b> ops::Mul<&'a Matrix3> for &'b Matrix3 {
    type Output = Matrix3;

    fn mul(self, other: &'a Matrix3) -> Matrix3 {
        let m11 = self.m[0] * other.m[0] + self.m[3] * other.m[1] + self.m[6] * other.m[2]; // 0
        let m21 = self.m[1] * other.m[0] + self.m[4] * other.m[1] + self.m[7] * other.m[2]; // 1
        let m31 = self.m[2] * other.m[0] + self.m[5] * other.m[1] + self.m[8] * other.m[2]; // 2

        let m12 = self.m[0] * other.m[3] + self.m[3] * other.m[4] + self.m[6] * other.m[5]; // 3
        let m22 = self.m[1] * other.m[3] + self.m[4] * other.m[4] + self.m[7] * other.m[5]; // 4
        let m32 = self.m[2] * other.m[3] + self.m[5] * other.m[4] + self.m[8] * other.m[5]; // 5

        let m13 = self.m[0] * other.m[6] + self.m[3] * other.m[7] + self.m[6] * other.m[8]; // 6
        let m23 = self.m[1] * other.m[6] + self.m[4] * other.m[7] + self.m[7] * other.m[8]; // 7
        let m33 = self.m[2] * other.m[6] + self.m[5] * other.m[7] + self.m[8] * other.m[8]; // 8

        Matrix3::new(
            m11, m21, m31,
            m12, m22, m32,
            m13, m23, m33,
        )
    }
}

impl ops::Mul<Matrix3> for Matrix3 {
    type Output = Matrix3;

    fn mul(self, other: Matrix3) -> Matrix3 {
        let m11 = self.m[0] * other.m[0] + self.m[3] * other.m[1] + self.m[6] * other.m[2]; // 0
        let m21 = self.m[1] * other.m[0] + self.m[4] * other.m[1] + self.m[7] * other.m[2]; // 1
        let m31 = self.m[2] * other.m[0] + self.m[5] * other.m[1] + self.m[8] * other.m[2]; // 2

        let m12 = self.m[0] * other.m[3] + self.m[3] * other.m[4] + self.m[6] * other.m[5]; // 3
        let m22 = self.m[1] * other.m[3] + self.m[4] * other.m[4] + self.m[7] * other.m[5]; // 4
        let m32 = self.m[2] * other.m[3] + self.m[5] * other.m[4] + self.m[8] * other.m[5]; // 5

        let m13 = self.m[0] * other.m[6] + self.m[3] * other.m[7] + self.m[6] * other.m[8]; // 6
        let m23 = self.m[1] * other.m[6] + self.m[4] * other.m[7] + self.m[7] * other.m[8]; // 7
        let m33 = self.m[2] * other.m[6] + self.m[5] * other.m[7] + self.m[8] * other.m[8]; // 8

        Matrix3::new(
            m11, m21, m31,
            m12, m22, m32,
            m13, m23, m33,
        )
    }
}


#[cfg(test)]
mod matrix2_tests {
    use std::slice::Iter;
    use vector::Vector2;
    use super::Matrix2;

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
}


#[cfg(test)]
mod matrix3_tests {
    use std::slice::Iter;
    use vector::Vector3;
    use super::Matrix3;

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
}

