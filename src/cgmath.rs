use std::fmt;
use std::mem;
use std::ops;
use std::cmp;

use crate::traits::Array;
use crate::vector::*;


// Constants used to convert degrees into radians.
pub const M_PI: f32 = 3.14159265358979323846264338327950288;
pub const TAU: f32 = 2.0 * M_PI;
pub const ONE_DEG_IN_RAD: f32 = (2.0 * M_PI) / 360.0; // == 0.017444444
pub const ONE_RAD_IN_DEG: f32 = 360.0 / (2.0 * M_PI); // == 57.2957795
pub const EPSILON: f32 = 0.00001; 


///
/// Compute the orthographic projection matrix for converting from camera space to
/// normalized device coordinates.
///
#[inline]
pub fn ortho<Spec: Into<Orthographic>>(spec: Spec) -> Matrix4 {
    Matrix4::from(spec.into())
}

///
/// Compute a perspective matrix from a view frustum.
///
/// This is the equivalent of the now deprecated [glFrustum]
/// (http://www.opengl.org/sdk/docs/man2/xhtml/glFrustum.xml) function.
///
#[inline]
pub fn frustum<Spec: Into<Perspective>>(spec: Spec) -> Matrix4 {
    Matrix4::from(spec.into())
}

///
/// Compute the perspective matrix for converting from camera space to 
/// normalized device coordinates. This is the equivalent to the
/// [gluPerspective] (http://www.opengl.org/sdk/docs/man2/xhtml/gluPerspective.xml)
/// function.
///
#[inline]
pub fn perspective<Spec: Into<PerspectiveFov>>(spec: Spec) -> Matrix4 {
    Matrix4::from(spec.into())
}

///
/// Construct a new one-dimensional vector. This follows the style of
/// other GLSL vector constructors even though GLSL itself lacks a
/// `vec1()` function.
///
#[inline]
pub fn vec1<T: Into<Vector1>>(v: T) -> Vector1 {
    v.into()
}

///
/// Construct a new two-dimensional vector in the style of
/// a GLSL `vec2` constructor.
///
#[inline]
pub fn vec2<T: Into<Vector2>>(v: T) -> Vector2 {
    v.into()
}

///
/// Construct a new three-dimensional vector in the style of
/// a GLSL `vec3` constructor.
///
#[inline]
pub fn vec3<T: Into<Vector3>>(v: T) -> Vector3 {
    v.into()
}

///
/// Construct a new four-dimensional vector in the style of
/// a GLSL `vec4` constructor.
///
#[inline]
pub fn vec4<T: Into<Vector4>>(v: T) -> Vector4 {
    v.into()
}

///
/// Create a new quaternion in the style of a GLSL type
/// constructor. This is not a built-in function in GLSL, but it exists
/// for convenience.
///
#[inline]
pub fn quat<T: Into<Quaternion>>(q: T) -> Quaternion {
    q.into()
}

///
/// Create a new 2x2 matrix in the style of a GLSL type
/// constructor.
///
#[inline]
pub fn mat2(
    m11: f32, m12: f32,
    m21: f32, m22: f32) -> Matrix2 {

    Matrix2::new(m11, m12, m21, m22)
}

///
/// Create a new 3x3 matrix in the style of a GLSL type
/// constructor.
///
#[inline]
pub fn mat3(
    m11: f32, m12: f32, m13: f32,
    m21: f32, m22: f32, m23: f32,
    m31: f32, m32: f32, m33: f32) -> Matrix3 {

    Matrix3::new(m11, m12, m13, m21, m22, m23, m31, m32, m33)
}

///
/// Create a new 4x4 matrix in the style of a GLSL type
/// constructor.
///
#[inline]
pub fn mat4(
        m11: f32, m12: f32, m13: f32, m14: f32, 
        m21: f32, m22: f32, m23: f32, m24: f32,
        m31: f32, m32: f32, m33: f32, m34: f32,
        m41: f32, m42: f32, m43: f32, m44: f32) -> Matrix4 {

    Matrix4::new(
        m11, m12, m13, m14, 
        m21, m22, m23, m24, 
        m31, m32, m33, m34, 
        m41, m42, m43, m44
    )
}


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

///
/// The `Matrix3` type represents 3x3 matrices in column-major order.
///
#[derive(Copy, Clone, Debug)]
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

    pub fn zero() -> Matrix3 {
        Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    pub fn one() -> Matrix3 {
        Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    }

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

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Orthographic {
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
}

impl Into<Orthographic> for (f32, f32, f32, f32, f32, f32) {
    #[inline]
    fn into(self) -> Orthographic {
        match self {
            (left, right, bottom, top, near, far) => {
                Orthographic {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<'a> Into<Orthographic> for &'a (f32, f32, f32, f32, f32, f32) {
    #[inline]
    fn into(self) -> Orthographic {
        match *self {
            (left, right, bottom, top, near, far) => {
                Orthographic {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl Into<Orthographic> for [f32; 6] {
    #[inline]
    fn into(self) -> Orthographic {
        match self {
            [left, right, bottom, top, near, far] => {
                Orthographic {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<'a> Into<Orthographic> for &'a [f32; 6] {
    #[inline]
    fn into(self) -> Orthographic {
        match *self {
            [left, right, bottom, top, near, far] => {
                Orthographic {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Perspective {
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
}

impl Into<Perspective> for (f32, f32, f32, f32, f32, f32) {
    #[inline]
    fn into(self) -> Perspective {
        match self {
            (left, right, bottom, top, near, far) => {
                Perspective {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<'a> Into<Perspective> for &'a (f32, f32, f32, f32, f32, f32) {
    #[inline]
    fn into(self) -> Perspective {
        match *self {
            (left, right, bottom, top, near, far) => {
                Perspective {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl Into<Perspective> for [f32; 6] {
    #[inline]
    fn into(self) -> Perspective {
        match self {
            [left, right, bottom, top, near, far] => {
                Perspective {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<'a> Into<Perspective> for &'a [f32; 6] {
    #[inline]
    fn into(self) -> Perspective {
        match *self {
            [left, right, bottom, top, near, far] => {
                Perspective {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerspectiveFov {
    fovy: f32,
    aspect: f32,
    near: f32,
    far: f32,
}

impl Into<PerspectiveFov> for (f32, f32, f32, f32) {
    #[inline]
    fn into(self) -> PerspectiveFov {
        match self {
            (fovy, aspect, near, far) => {
                PerspectiveFov {
                    fovy: fovy,
                    aspect: aspect,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<'a> Into<PerspectiveFov> for &'a (f32, f32, f32, f32) {
    #[inline]
    fn into(self) -> PerspectiveFov {
        match *self {
            (fovy, aspect, near, far) => {
                PerspectiveFov {
                    fovy: fovy,
                    aspect: aspect,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl Into<PerspectiveFov> for [f32; 4] {
    #[inline]
    fn into(self) -> PerspectiveFov {
        match self {
            [fovy, aspect, near, far] => {
                PerspectiveFov {
                    fovy: fovy,
                    aspect: aspect,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<'a> Into<PerspectiveFov> for &'a [f32; 4] {
    #[inline]
    fn into(self) -> PerspectiveFov {
        match *self {
            [fovy, aspect, near, far] => {
                PerspectiveFov {
                    fovy: fovy,
                    aspect: aspect,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

///
/// The `Matrix4` type represents 4x4 matrices in column-major order.
///
#[derive(Copy, Clone, Debug)]
pub struct Matrix4 {
    pub m: [f32; 16],
}

impl Matrix4 {
    pub fn new(
        m11: f32, m12: f32, m13: f32, m14: f32,
        m21: f32, m22: f32, m23: f32, m24: f32,
        m31: f32, m32: f32, m33: f32, m34: f32,
        m41: f32, m42: f32, m43: f32, m44: f32) -> Matrix4 {

        Matrix4 {
            m: [
                m11, m12, m13, m14, // Column 1
                m21, m22, m23, m24, // Column 2
                m31, m32, m33, m34, // Column 3
                m41, m42, m43, m44  // Column 4
            ]
        }
    }

    ///
    /// Return the zero matrix.
    ///
    pub fn zero() -> Matrix4 {
        Matrix4::new(
            0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0
        )
    }

    ///
    /// Return the identity matrix.
    ///
    pub fn one() -> Matrix4 {
        Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0
        )
    }

    ///
    /// Transpose a 4x4 matrix.
    ///
    pub fn transpose(&self) -> Matrix4 {
        Matrix4::new(
            self.m[0], self.m[4], self.m[8],  self.m[12],
            self.m[1], self.m[5], self.m[9],  self.m[13], 
            self.m[2], self.m[6], self.m[10], self.m[14], 
            self.m[3], self.m[7], self.m[11], self.m[15]
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
        rot_mat.m[5]  =  f32::cos(radians);
        rot_mat.m[9]  = -f32::sin(radians);
        rot_mat.m[6]  =  f32::sin(radians);
        rot_mat.m[10] =  f32::cos(radians);
    
        rot_mat
    }

    ///
    /// Create a rotation matrix around the y axis by an angle in `degrees` degrees.
    ///
    pub fn from_rotation_y(degrees: f32) -> Matrix4 {
        // Convert to radians.
        let radians = degrees * ONE_DEG_IN_RAD;
        let mut rot_mat = Matrix4::one();
        rot_mat.m[0]  =  f32::cos(radians);
        rot_mat.m[8]  =  f32::sin(radians);
        rot_mat.m[2]  = -f32::sin(radians);
        rot_mat.m[10] =  f32::cos(radians);
    
        rot_mat
    }

    ///
    /// Create a rotation matrix around the z axis by an angle in `degrees` degrees.
    ///
    pub fn from_rotation_z(degrees: f32) -> Matrix4 {
        // Convert to radians.
        let radians = degrees * ONE_DEG_IN_RAD;
        let mut rot_mat = Matrix4::one();
        rot_mat.m[0] =  f32::cos(radians);
        rot_mat.m[4] = -f32::sin(radians);
        rot_mat.m[1] =  f32::sin(radians);
        rot_mat.m[5] =  f32::cos(radians);
    
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
        self.m[12] * self.m[9]  * self.m[6]  * self.m[3]  -
        self.m[8]  * self.m[13] * self.m[6]  * self.m[3]  -
        self.m[12] * self.m[5]  * self.m[10] * self.m[3]  +
        self.m[4]  * self.m[13] * self.m[10] * self.m[3]  +
        self.m[8]  * self.m[5]  * self.m[14] * self.m[3]  -
        self.m[4]  * self.m[9]  * self.m[14] * self.m[3]  -
        self.m[12] * self.m[9]  * self.m[2]  * self.m[7]  +
        self.m[8]  * self.m[13] * self.m[2]  * self.m[7]  +
        self.m[12] * self.m[1]  * self.m[10] * self.m[7]  -
        self.m[0]  * self.m[13] * self.m[10] * self.m[7]  -
        self.m[8]  * self.m[1]  * self.m[14] * self.m[7]  +
        self.m[0]  * self.m[9]  * self.m[14] * self.m[7]  +
        self.m[12] * self.m[5]  * self.m[2]  * self.m[11] -
        self.m[4]  * self.m[13] * self.m[2]  * self.m[11] -
        self.m[12] * self.m[1]  * self.m[6]  * self.m[11] +
        self.m[0]  * self.m[13] * self.m[6]  * self.m[11] +
        self.m[4]  * self.m[1]  * self.m[14] * self.m[11] -
        self.m[0]  * self.m[5]  * self.m[14] * self.m[11] -
        self.m[8]  * self.m[5]  * self.m[2]  * self.m[15] +
        self.m[4]  * self.m[9]  * self.m[2]  * self.m[15] +
        self.m[8]  * self.m[1]  * self.m[6]  * self.m[15] -
        self.m[0]  * self.m[9]  * self.m[6]  * self.m[15] -
        self.m[4]  * self.m[1]  * self.m[10] * self.m[15] +
        self.m[0]  * self.m[5]  * self.m[10] * self.m[15]
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

        return mat4(
            inv_det * ( self.m[9] * self.m[14] * self.m[7] - self.m[13] * self.m[10] * self.m[7] +
                                    self.m[13] * self.m[6] * self.m[11] - self.m[5] * self.m[14] * self.m[11] -
                                    self.m[9] * self.m[6] * self.m[15] + self.m[5] * self.m[10] * self.m[15] ),
            inv_det * ( self.m[13] * self.m[10] * self.m[3] - self.m[9] * self.m[14] * self.m[3] -
                                    self.m[13] * self.m[2] * self.m[11] + self.m[1] * self.m[14] * self.m[11] +
                                    self.m[9] * self.m[2] * self.m[15] - self.m[1] * self.m[10] * self.m[15] ),
            inv_det * ( self.m[5] * self.m[14] * self.m[3] - self.m[13] * self.m[6] * self.m[3] +
                                    self.m[13] * self.m[2] * self.m[7] - self.m[1] * self.m[14] * self.m[7] -
                                    self.m[5] * self.m[2] * self.m[15] + self.m[1] * self.m[6] * self.m[15] ),
            inv_det * ( self.m[9] * self.m[6] * self.m[3] - self.m[5] * self.m[10] * self.m[3] -
                                    self.m[9] * self.m[2] * self.m[7] + self.m[1] * self.m[10] * self.m[7] +
                                    self.m[5] * self.m[2] * self.m[11] - self.m[1] * self.m[6] * self.m[11] ),
            inv_det * ( self.m[12] * self.m[10] * self.m[7] - self.m[8] * self.m[14] * self.m[7] -
                                    self.m[12] * self.m[6] * self.m[11] + self.m[4] * self.m[14] * self.m[11] +
                                    self.m[8] * self.m[6] * self.m[15] - self.m[4] * self.m[10] * self.m[15] ),
            inv_det * ( self.m[8] * self.m[14] * self.m[3] - self.m[12] * self.m[10] * self.m[3] +
                                    self.m[12] * self.m[2] * self.m[11] - self.m[0] * self.m[14] * self.m[11] -
                                    self.m[8] * self.m[2] * self.m[15] + self.m[0] * self.m[10] * self.m[15] ),
            inv_det * ( self.m[12] * self.m[6] * self.m[3] - self.m[4] * self.m[14] * self.m[3] -
                                    self.m[12] * self.m[2] * self.m[7] + self.m[0] * self.m[14] * self.m[7] +
                                    self.m[4] * self.m[2] * self.m[15] - self.m[0] * self.m[6] * self.m[15] ),
            inv_det * ( self.m[4] * self.m[10] * self.m[3] - self.m[8] * self.m[6] * self.m[3] +
                                    self.m[8] * self.m[2] * self.m[7] - self.m[0] * self.m[10] * self.m[7] -
                                    self.m[4] * self.m[2] * self.m[11] + self.m[0] * self.m[6] * self.m[11] ),
            inv_det * ( self.m[8] * self.m[13] * self.m[7] - self.m[12] * self.m[9] * self.m[7] +
                                    self.m[12] * self.m[5] * self.m[11] - self.m[4] * self.m[13] * self.m[11] -
                                    self.m[8] * self.m[5] * self.m[15] + self.m[4] * self.m[9] * self.m[15] ),
            inv_det * ( self.m[12] * self.m[9] * self.m[3] - self.m[8] * self.m[13] * self.m[3] -
                                    self.m[12] * self.m[1] * self.m[11] + self.m[0] * self.m[13] * self.m[11] +
                                    self.m[8] * self.m[1] * self.m[15] - self.m[0] * self.m[9] * self.m[15] ),
            inv_det * ( self.m[4] * self.m[13] * self.m[3] - self.m[12] * self.m[5] * self.m[3] +
                                    self.m[12] * self.m[1] * self.m[7] - self.m[0] * self.m[13] * self.m[7] -
                                    self.m[4] * self.m[1] * self.m[15] + self.m[0] * self.m[5] * self.m[15] ),
            inv_det * ( self.m[8] * self.m[5] * self.m[3] - self.m[4] * self.m[9] * self.m[3] -
                                    self.m[8] * self.m[1] * self.m[7] + self.m[0] * self.m[9] * self.m[7] +
                                    self.m[4] * self.m[1] * self.m[11] - self.m[0] * self.m[5] * self.m[11] ),
            inv_det * ( self.m[12] * self.m[9] * self.m[6] - self.m[8] * self.m[13] * self.m[6] -
                                    self.m[12] * self.m[5] * self.m[10] + self.m[4] * self.m[13] * self.m[10] +
                                    self.m[8] * self.m[5] * self.m[14] - self.m[4] * self.m[9] * self.m[14] ),
            inv_det * ( self.m[8] * self.m[13] * self.m[2] - self.m[12] * self.m[9] * self.m[2] +
                                    self.m[12] * self.m[1] * self.m[10] - self.m[0] * self.m[13] * self.m[10] -
                                    self.m[8] * self.m[1] * self.m[14] + self.m[0] * self.m[9] * self.m[14] ),
            inv_det * ( self.m[12] * self.m[5] * self.m[2] - self.m[4] * self.m[13] * self.m[2] -
                                    self.m[12] * self.m[1] * self.m[6] + self.m[0] * self.m[13] * self.m[6] +
                                    self.m[4] * self.m[1] * self.m[14] - self.m[0] * self.m[5] * self.m[14] ),
            inv_det * ( self.m[4] * self.m[9] * self.m[2] - self.m[8] * self.m[5] * self.m[2] +
                                    self.m[8] * self.m[1] * self.m[6] - self.m[0] * self.m[9] * self.m[6] -
                                    self.m[4] * self.m[1] * self.m[10] + self.m[0] * self.m[5] * self.m[10] ) );
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
        self.m.as_ptr()
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut f32 {
        self.m.as_mut_ptr()
    }
}

impl fmt::Display for Matrix4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, 
            "\n[{:.2}][{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}][{:.2}]", 
            self.m[0], self.m[4], self.m[8],  self.m[12],
            self.m[1], self.m[5], self.m[9],  self.m[13],
            self.m[2], self.m[6], self.m[10], self.m[14],
            self.m[3], self.m[7], self.m[11], self.m[15]
        )
    }
}

impl AsRef<[f32; 16]> for Matrix4 {
    fn as_ref(&self) -> &[f32; 16] {
        &self.m
    }
}

impl AsRef<[[f32; 4]; 4]> for Matrix4 {
    fn as_ref(&self) -> &[[f32; 4]; 4] {
        unsafe {
            mem::transmute(self)
        }
    }
}

impl AsMut<[f32; 16]> for Matrix4 {
    fn as_mut(&mut self) -> &mut [f32; 16] {
        &mut self.m
    }
}

impl ops::Mul<Vector4> for Matrix4 {
    type Output = Vector4;

    fn mul(self, other: Vector4) -> Self::Output {
        let x = self.m[0] * other[0] + self.m[4] * other[1] + self.m[8]  * other[2] + self.m[12] * other[3];
        let y = self.m[1] * other[0] + self.m[5] * other[1] + self.m[9]  * other[2] + self.m[13] * other[3];
        let z = self.m[2] * other[0] + self.m[6] * other[1] + self.m[10] * other[2] + self.m[14] * other[3];
        let w = self.m[3] * other[0] + self.m[7] * other[1] + self.m[11] * other[2] + self.m[15] * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl<'a> ops::Mul<&'a Matrix4> for Matrix4 {
    type Output = Matrix4;

    fn mul(self, other: &'a Matrix4) -> Matrix4 {
        let mut mm = Matrix4::zero();

        mm.m[0]  = self.m[0] * other.m[0]  + self.m[4] * other.m[1]  + self.m[8]  * other.m[2]  + self.m[12] * other.m[3];
        mm.m[1]  = self.m[1] * other.m[0]  + self.m[5] * other.m[1]  + self.m[9]  * other.m[2]  + self.m[13] * other.m[3];
        mm.m[2]  = self.m[2] * other.m[0]  + self.m[6] * other.m[1]  + self.m[10] * other.m[2]  + self.m[14] * other.m[3];
        mm.m[3]  = self.m[3] * other.m[0]  + self.m[7] * other.m[1]  + self.m[11] * other.m[2]  + self.m[15] * other.m[3];
        mm.m[4]  = self.m[0] * other.m[4]  + self.m[4] * other.m[5]  + self.m[8]  * other.m[6]  + self.m[12] * other.m[7];
        mm.m[5]  = self.m[1] * other.m[4]  + self.m[5] * other.m[5]  + self.m[9]  * other.m[6]  + self.m[13] * other.m[7];
        mm.m[6]  = self.m[2] * other.m[4]  + self.m[6] * other.m[5]  + self.m[10] * other.m[6]  + self.m[14] * other.m[7];
        mm.m[7]  = self.m[3] * other.m[4]  + self.m[7] * other.m[5]  + self.m[11] * other.m[6]  + self.m[15] * other.m[7];
        mm.m[8]  = self.m[0] * other.m[8]  + self.m[4] * other.m[9]  + self.m[8]  * other.m[10] + self.m[12] * other.m[11];
        mm.m[9]  = self.m[1] * other.m[8]  + self.m[5] * other.m[9]  + self.m[9]  * other.m[10] + self.m[13] * other.m[11];
        mm.m[10] = self.m[2] * other.m[8]  + self.m[6] * other.m[9]  + self.m[10] * other.m[10] + self.m[14] * other.m[11];
        mm.m[11] = self.m[3] * other.m[8]  + self.m[7] * other.m[9]  + self.m[11] * other.m[10] + self.m[15] * other.m[11];
        mm.m[12] = self.m[0] * other.m[12] + self.m[4] * other.m[13] + self.m[8]  * other.m[14] + self.m[12] * other.m[15];
        mm.m[13] = self.m[1] * other.m[12] + self.m[5] * other.m[13] + self.m[9]  * other.m[14] + self.m[13] * other.m[15];
        mm.m[14] = self.m[2] * other.m[12] + self.m[6] * other.m[13] + self.m[10] * other.m[14] + self.m[14] * other.m[15];
        mm.m[15] = self.m[3] * other.m[12] + self.m[7] * other.m[13] + self.m[11] * other.m[14] + self.m[15] * other.m[15];

        mm
    }
}

impl<'a, 'b> ops::Mul<&'a Matrix4> for &'b Matrix4 {
    type Output = Matrix4;

    fn mul(self, other: &'a Matrix4) -> Matrix4 {
        let mut mm = Matrix4::zero();

        mm.m[0]  = self.m[0]*other.m[0]  + self.m[4]*other.m[1]  + self.m[8]*other.m[2]   + self.m[12]*other.m[3];
        mm.m[1]  = self.m[1]*other.m[0]  + self.m[5]*other.m[1]  + self.m[9]*other.m[2]   + self.m[13]*other.m[3];
        mm.m[2]  = self.m[2]*other.m[0]  + self.m[6]*other.m[1]  + self.m[10]*other.m[2]  + self.m[14]*other.m[3];
        mm.m[3]  = self.m[3]*other.m[0]  + self.m[7]*other.m[1]  + self.m[11]*other.m[2]  + self.m[15]*other.m[3];
        mm.m[4]  = self.m[0]*other.m[4]  + self.m[4]*other.m[5]  + self.m[8]*other.m[6]   + self.m[12]*other.m[7];
        mm.m[5]  = self.m[1]*other.m[4]  + self.m[5]*other.m[5]  + self.m[9]*other.m[6]   + self.m[13]*other.m[7];
        mm.m[6]  = self.m[2]*other.m[4]  + self.m[6]*other.m[5]  + self.m[10]*other.m[6]  + self.m[14]*other.m[7];
        mm.m[7]  = self.m[3]*other.m[4]  + self.m[7]*other.m[5]  + self.m[11]*other.m[6]  + self.m[15]*other.m[7];
        mm.m[8]  = self.m[0]*other.m[8]  + self.m[4]*other.m[9]  + self.m[8]*other.m[10]  + self.m[12]*other.m[11];
        mm.m[9]  = self.m[1]*other.m[8]  + self.m[5]*other.m[9]  + self.m[9]*other.m[10]  + self.m[13]*other.m[11];
        mm.m[10] = self.m[2]*other.m[8]  + self.m[6]*other.m[9]  + self.m[10]*other.m[10] + self.m[14]*other.m[11];
        mm.m[11] = self.m[3]*other.m[8]  + self.m[7]*other.m[9]  + self.m[11]*other.m[10] + self.m[15]*other.m[11];
        mm.m[12] = self.m[0]*other.m[12] + self.m[4]*other.m[13] + self.m[8]*other.m[14]  + self.m[12]*other.m[15];
        mm.m[13] = self.m[1]*other.m[12] + self.m[5]*other.m[13] + self.m[9]*other.m[14]  + self.m[13]*other.m[15];
        mm.m[14] = self.m[2]*other.m[12] + self.m[6]*other.m[13] + self.m[10]*other.m[14] + self.m[14]*other.m[15];
        mm.m[15] = self.m[3]*other.m[12] + self.m[7]*other.m[13] + self.m[11]*other.m[14] + self.m[15]*other.m[15];

        mm
    }
}

impl ops::Mul<Matrix4> for Matrix4 {
    type Output = Matrix4;

    fn mul(self, other: Matrix4) -> Matrix4 {
        let mut mm = Matrix4::zero();

        mm.m[0]  = self.m[0]*other.m[0]  + self.m[4]*other.m[1]  + self.m[8]*other.m[2]   + self.m[12]*other.m[3];
        mm.m[1]  = self.m[1]*other.m[0]  + self.m[5]*other.m[1]  + self.m[9]*other.m[2]   + self.m[13]*other.m[3];
        mm.m[2]  = self.m[2]*other.m[0]  + self.m[6]*other.m[1]  + self.m[10]*other.m[2]  + self.m[14]*other.m[3];
        mm.m[3]  = self.m[3]*other.m[0]  + self.m[7]*other.m[1]  + self.m[11]*other.m[2]  + self.m[15]*other.m[3];
        mm.m[4]  = self.m[0]*other.m[4]  + self.m[4]*other.m[5]  + self.m[8]*other.m[6]   + self.m[12]*other.m[7];
        mm.m[5]  = self.m[1]*other.m[4]  + self.m[5]*other.m[5]  + self.m[9]*other.m[6]   + self.m[13]*other.m[7];
        mm.m[6]  = self.m[2]*other.m[4]  + self.m[6]*other.m[5]  + self.m[10]*other.m[6]  + self.m[14]*other.m[7];
        mm.m[7]  = self.m[3]*other.m[4]  + self.m[7]*other.m[5]  + self.m[11]*other.m[6]  + self.m[15]*other.m[7];
        mm.m[8]  = self.m[0]*other.m[8]  + self.m[4]*other.m[9]  + self.m[8]*other.m[10]  + self.m[12]*other.m[11];
        mm.m[9]  = self.m[1]*other.m[8]  + self.m[5]*other.m[9]  + self.m[9]*other.m[10]  + self.m[13]*other.m[11];
        mm.m[10] = self.m[2]*other.m[8]  + self.m[6]*other.m[9]  + self.m[10]*other.m[10] + self.m[14]*other.m[11];
        mm.m[11] = self.m[3]*other.m[8]  + self.m[7]*other.m[9]  + self.m[11]*other.m[10] + self.m[15]*other.m[11];
        mm.m[12] = self.m[0]*other.m[12] + self.m[4]*other.m[13] + self.m[8]*other.m[14]  + self.m[12]*other.m[15];
        mm.m[13] = self.m[1]*other.m[12] + self.m[5]*other.m[13] + self.m[9]*other.m[14]  + self.m[13]*other.m[15];
        mm.m[14] = self.m[2]*other.m[12] + self.m[6]*other.m[13] + self.m[10]*other.m[14] + self.m[14]*other.m[15];
        mm.m[15] = self.m[3]*other.m[12] + self.m[7]*other.m[13] + self.m[11]*other.m[14] + self.m[15]*other.m[15];

        mm
    }
}

impl cmp::PartialEq for Matrix4 {
    fn eq(&self, other: &Matrix4) -> bool {
        for i in 0..self.m.len() {
            if f32::abs(self.m[i] - other.m[i]) > EPSILON {
                return false;
            }
        }

        true
    }
}

impl From<PerspectiveFov> for Matrix4 {
    fn from(persp: PerspectiveFov) -> Matrix4 {
        let fov_rad = persp.fovy * ONE_DEG_IN_RAD;
        let range = f32::tan(fov_rad / 2.0) * persp.near;
        let sx = (2.0 * persp.near) / (range * persp.aspect + range * persp.aspect);
        let sy = persp.near / range;
        let sz = (persp.far + persp.near) / (persp.near - persp.far);
        let pz = (2.0 * persp.far * persp.near) / (persp.near - persp.far);
        
        Matrix4::new(
             sx, 0.0, 0.0,  0.0,
            0.0,  sy, 0.0,  0.0,
            0.0, 0.0,  sz, -1.0,
            0.0, 0.0,  pz,  0.0
        )
    }
}

impl From<Perspective> for Matrix4 {
    fn from(persp: Perspective) -> Matrix4 {
        let c0r0 = (2.0 * persp.near) / (persp.right - persp.left);
        let c0r1 = 0.0;
        let c0r2 = 0.0;
        let c0r3 = 0.0;

        let c1r0 = 0.0;
        let c1r1 = (2.0 * persp.near) / (persp.top - persp.bottom);
        let c1r2 = 0.0;
        let c1r3 = 0.0;

        let c2r0 =  (persp.right + persp.left)   / (persp.right - persp.left);
        let c2r1 =  (persp.top   + persp.bottom) / (persp.top   - persp.bottom);
        let c2r2 = -(persp.far   + persp.near)   / (persp.far   - persp.near);
        let c2r3 = -1.0;

        let c3r0 = 0.0;
        let c3r1 = 0.0;
        let c3r2 = -(2.0 * persp.far * persp.near) / (persp.far - persp.near);
        let c3r3 = 0.0;

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3,
        )
    }
}

impl From<Orthographic> for Matrix4 {
    fn from(ortho: Orthographic) -> Matrix4 {
        let sx = 2.0 / (ortho.right - ortho.left);
        let sy = 2.0 / (ortho.top - ortho.bottom);
        let sz = 2.0 / (ortho.far - ortho.near);
        let tx = (ortho.right + ortho.left) / (ortho.right - ortho.left);
        let ty = (ortho.top + ortho.bottom) / (ortho.top - ortho.bottom);
        let tz = (ortho.far + ortho.near) / (ortho.far - ortho.near);

        Matrix4::new(
             sx, 0.0, 0.0, 0.0,
            0.0,  sy, 0.0, 0.0,
            0.0, 0.0,  sz, 0.0,
             -tx, -ty, -tz, 1.0
        )
    }
}

#[derive(Copy, Clone, PartialEq)]
pub struct Quaternion {
    s: f32,
    x: f32,
    y: f32,
    z: f32,
}

impl Quaternion {
    pub fn new(s: f32, x: f32, y: f32, z: f32) -> Quaternion {
        let q = Quaternion { s: s, x: x, y: y, z: z };

        q.normalize()
    }

    pub fn normalize(&self) -> Quaternion {
        let sum = self.s * self.s + self.x * self.x + self.y * self.y + self.z * self.z;
        // NOTE: f32s have min 6 digits of precision.
        let threshold = 0.0001;
        if f32::abs(1.0 - sum) < threshold {
            return *self;
        }

        let norm = f32::sqrt(sum);
        self / norm
    }

    ///
    /// Create a zero quaterion. It is a quaternion such that 
    /// q - q = 0.
    ///
    pub fn zero() -> Quaternion {
        Quaternion { s: 0.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    ///
    /// Create a unit quaternion who is also the multiplicative identity:
    /// q * q^-1 == 1.
    ///
    pub fn one() -> Quaternion {
        Quaternion { s: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    ///
    /// Compute the inner (dot) product of two quaternions.
    ///
    pub fn dot(&self, r: &Quaternion) -> f32 {
        self.s * r.s + self.x * r.x + self.y * r.y + self.z * r.z
    }

    ///
    /// Compute the euclidean norm of a quaternion.
    ///
    pub fn norm(&self) -> f32 {
        f32::sqrt(self.s * self.s + self.x * self.x + self.y * self.y + self.z * self.z)
    }

    ///
    /// Compute the squared Euclidean norm of a quaternion.
    ///
    pub fn norm2(&self) -> f32 {
        self.s * self.s + self.x * self.x + self.y * self.y + self.z * self.z
    }

    ///
    /// Compute a quaternion from it's scalar and vector parts.
    ///
    pub fn from_sv(s: f32, v: Vector3) -> Quaternion {
        Quaternion { s: s, x: v.x, y: v.y, z: v.z }
    }

    ///
    /// Compute a quaternion corresponding to rotating about an axis in radians.
    ///
    pub fn from_axis_rad(radians: f32, axis: Vector3) -> Quaternion {
        Quaternion {
            s: f32::cos(radians / 2.0),
            x: f32::sin(radians / 2.0) * axis.x,
            y: f32::sin(radians / 2.0) * axis.y,
            z: f32::sin(radians / 2.0) * axis.z,
        }
    }

    ///
    /// Computer a quaternion corresponding to rotating about an axis in degrees.
    ///
    pub fn from_axis_deg(degrees: f32, axis: Vector3) -> Quaternion {
        Self::from_axis_rad(ONE_DEG_IN_RAD * degrees, axis)
    }

    ///
    /// Compute the conjugate of a quaternion.
    ///
    pub fn conjugate(&self) -> Quaternion {
        Quaternion { s: self.s, x: -self.x, y: -self.y, z: -self.z }
    }

    ///
    /// Convert a quaternion to its equivalent matrix form using .
    ///
    pub fn to_mut_mat4(&self, m: &mut Matrix4) {
        let s = self.s;
        let x = self.x;
        let y = self.y;
        let z = self.z;
        m.m[0] = 1.0 - 2.0 * y * y - 2.0 * z * z;
        m.m[1] = 2.0 * x * y + 2.0 * s * z;
        m.m[2] = 2.0 * x * z - 2.0 * s * y;
        m.m[3] = 0.0;
        m.m[4] = 2.0 * x * y - 2.0 * s * z;
        m.m[5] = 1.0 - 2.0 * x * x - 2.0 * z * z;
        m.m[6] = 2.0 * y * z + 2.0 * s * x;
        m.m[7] = 0.0;
        m.m[8] = 2.0 * x * z + 2.0 * s * y;
        m.m[9] = 2.0 * y * z - 2.0 * s * x;
        m.m[10] = 1.0 - 2.0 * x * x - 2.0 * y * y;
        m.m[11] = 0.0;
        m.m[12] = 0.0;
        m.m[13] = 0.0;
        m.m[14] = 0.0;
        m.m[15] = 1.0;
    }

    pub fn slerp(q: &mut Quaternion, r: &Quaternion, t: f32) -> Quaternion {
        // angle between q0-q1
        let mut cos_half_theta = q.dot(r);
        // as found here
        // http://stackoverflow.com/questions/2886606/flipping-issue-when-interpolating-rotations-using-quaternions
        // if dot product is negative then one quaternion should be negated, to make
        // it take the short way around, rather than the long way
        // yeah! and furthermore Susan, I had to recalculate the d.p. after this
        if cos_half_theta < 0.0 {
            q.s *= -1.0;
            q.x *= -1.0;
            q.y *= -1.0;
            q.z *= -1.0;

            cos_half_theta = q.dot(r);
        }
        // if qa=qb or qa=-qb then theta = 0 and we can return qa
        if f32::abs(cos_half_theta) >= 1.0 {
            return *q;
        }

        // Calculate temporary values
        let sin_half_theta = f32::sqrt(1.0 - cos_half_theta * cos_half_theta);
        // if theta = 180 degrees then result is not fully defined
        // we could rotate around any axis normal to qa or qb
        let mut result = Quaternion { s: 1.0, x: 0.0, y: 0.0, z: 0.0 };
        if f32::abs(sin_half_theta) < 0.001 {
            result.s = (1.0 - t) * q.s + t * r.s;
            result.x = (1.0 - t) * q.x + t * r.x;
            result.y = (1.0 - t) * q.y + t * r.y;
            result.z = (1.0 - t) * q.z + t * r.z;

            return result;
        }
        let half_theta = f32::acos(cos_half_theta);
        let a = f32::sin((1.0 - t) * half_theta) / sin_half_theta;
        let b = f32::sin(t * half_theta) / sin_half_theta;
        
        result.s = q.s * a + r.s * b;
        result.x = q.x * a + r.x * b;
        result.y = q.y * a + r.y * b;
        result.z = q.z * a + r.z * b;

        result
    }
}

impl AsRef<[f32; 4]> for Quaternion {
    fn as_ref(&self) -> &[f32; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl AsRef<(f32, f32, f32, f32)> for Quaternion {
    fn as_ref(&self) -> &(f32, f32, f32, f32) {
        unsafe { mem::transmute(self) }
    }
}

impl From<Quaternion> for Matrix3 {
    fn from(quat: Quaternion) -> Matrix3 {
        let s = quat.s;
        let x = quat.x;
        let y = quat.y;
        let z = quat.z;
    
        Matrix3::new(
            1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y + 2.0 * s * z,       2.0 * x * z - 2.0 * s * y,
            2.0 * x * y - 2.0 * s * z,       1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z + 2.0 * s * x,
            2.0 * x * z + 2.0 * s * y,       2.0 * y * z - 2.0 * s * x,       1.0 - 2.0 * x * x - 2.0 * y * y,
        )
    }
}

impl From<Quaternion> for Matrix4 {
    fn from(quat: Quaternion) -> Matrix4 {
        let s = quat.s;
        let x = quat.x;
        let y = quat.y;
        let z = quat.z;
    
        Matrix4::new(
            1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y + 2.0 * s * z,       2.0 * x * z - 2.0 * s * y,       0.0, 
            2.0 * x * y - 2.0 * s * z,       1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z + 2.0 * s * x,       0.0, 
            2.0 * x * z + 2.0 * s * y,       2.0 * y * z - 2.0 * s * x,       1.0 - 2.0 * x * x - 2.0 * y * y, 0.0, 
            0.0,                             0.0,                             0.0,                             1.0
        )
    }
}

impl From<[f32; 4]> for Quaternion {
    #[inline]
    fn from(v: [f32; 4]) -> Quaternion {
        Quaternion::new(v[0], v[1], v[2], v[3])
    }
}

impl<'a> From<&'a [f32; 4]> for &'a Quaternion {
    #[inline]
    fn from(v: &'a [f32; 4]) -> &'a Quaternion {
        unsafe { mem::transmute(v) }
    }
}

impl From<(f32, f32, f32, f32)> for Quaternion {
    #[inline]
    fn from(v: (f32, f32, f32, f32)) -> Quaternion {
        Quaternion::new(v.0, v.1, v.2, v.3)
    }
}

impl<'a> From<&'a (f32, f32, f32, f32)> for &'a Quaternion {
    #[inline]
    fn from(v: &'a (f32, f32, f32, f32)) -> &'a Quaternion {
        unsafe { mem::transmute(v) }
    }
}

impl ops::Index<usize> for Quaternion {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[f32; 4] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::Range<usize>> for Quaternion {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        let v: &[f32; 4] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::RangeTo<usize>> for Quaternion {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        let v: &[f32; 4] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::RangeFrom<usize>> for Quaternion {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        let v: &[f32; 4] = self.as_ref();
        &v[index]
    }
}

impl fmt::Debug for Quaternion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector4 ")?;
        writeln!(f, "[{}, [{}, {}, {}]]", self.s, self.x, self.y, self.z)
    }
}

impl fmt::Display for Quaternion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[{:.2}, [{:.2}, {:.2}, {:.2}]]", self.s, self.x, self.y, self.z)
    }
}

impl ops::Neg for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn neg(self) -> Self::Output {
        Quaternion { s: -self.s, x: -self.x, y: -self.y, z: -self.z }
    }
}

impl<'a> ops::Neg for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn neg(self) -> Self::Output {
        Quaternion { s: -self.s, x: -self.x, y: -self.y, z: -self.z }
    }
}

impl ops::Add<Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn add(self, other: Quaternion) -> Self::Output {
        Quaternion {
            s: other.s + self.s,
            x: other.x + self.x,
            y: other.y + self.y,
            z: other.z + self.z,
        }
    }
}

impl<'a> ops::Add<Quaternion> for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn add(self, other: Quaternion) -> Self::Output {
        Quaternion {
            s: other.s + self.s,
            x: other.x + self.x,
            y: other.y + self.y,
            z: other.z + self.z,
        }
    }
}

impl<'a> ops::Add<&'a Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn add(self, other: &'a Quaternion) -> Self::Output {
        Quaternion {
            s: other.s + self.s,
            x: other.x + self.x,
            y: other.y + self.y,
            z: other.z + self.z,
        }
    }
}

impl<'a, 'b> ops::Add<&'a Quaternion> for &'b Quaternion {
    type Output = Quaternion;

    #[inline]
    fn add(self, other: &'a Quaternion) -> Self::Output {
        Quaternion {
            s: other.s + self.s,
            x: other.x + self.x,
            y: other.y + self.y,
            z: other.z + self.z,
        }
    }
}

impl ops::Sub<Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn sub(self, other: Quaternion) -> Self::Output {
        Quaternion {
            s: other.s - self.s,
            x: other.x - self.x,
            y: other.y - self.y,
            z: other.z - self.z,
        }
    }
}

impl<'a> ops::Sub<Quaternion> for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn sub(self, other: Quaternion) -> Self::Output {
        Quaternion {
            s: other.s - self.s,
            x: other.x - self.x,
            y: other.y - self.y,
            z: other.z - self.z,
        }
    }
}

impl<'a> ops::Sub<&'a Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn sub(self, other: &'a Quaternion) -> Self::Output {
        Quaternion {
            s: other.s - self.s,
            x: other.x - self.x,
            y: other.y - self.y,
            z: other.z - self.z,
        }
    }
}

impl<'a, 'b> ops::Sub<&'a Quaternion> for &'b Quaternion {
    type Output = Quaternion;

    #[inline]
    fn sub(self, other: &'a Quaternion) -> Self::Output {
        Quaternion {
            s: other.s - self.s,
            x: other.x - self.x,
            y: other.y - self.y,
            z: other.z - self.z,
        }
    }
}

impl ops::Mul<f32> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: f32) -> Quaternion {
        Quaternion {
            s: self.s * other,
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl<'a> ops::Mul<Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: Quaternion) -> Self::Output {
        Quaternion {
            s: other.s * self.s - other.x * self.x - other.y * self.y - other.z * self.z,
            x: other.s * self.x + other.x * self.s - other.y * self.z + other.z * self.y,
            y: other.s * self.y + other.x * self.z + other.y * self.s - other.z * self.x,
            z: other.s * self.z - other.x * self.y + other.y * self.x + other.z * self.s,
        }
    }
}

impl<'a> ops::Mul<&'a Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: &'a Quaternion) -> Self::Output {
        Quaternion {
            s: other.s * self.s - other.x * self.x - other.y * self.y - other.z * self.z,
            x: other.s * self.x + other.x * self.s - other.y * self.z + other.z * self.y,
            y: other.s * self.y + other.x * self.z + other.y * self.s - other.z * self.x,
            z: other.s * self.z - other.x * self.y + other.y * self.x + other.z * self.s,
        }
    }
}

impl<'a> ops::Mul<Quaternion> for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: Quaternion) -> Self::Output {
        Quaternion {
            s: other.s * self.s - other.x * self.x - other.y * self.y - other.z * self.z,
            x: other.s * self.x + other.x * self.s - other.y * self.z + other.z * self.y,
            y: other.s * self.y + other.x * self.z + other.y * self.s - other.z * self.x,
            z: other.s * self.z - other.x * self.y + other.y * self.x + other.z * self.s,
        }
    }
}

impl<'a, 'b> ops::Mul<&'a Quaternion> for &'b Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: &'a Quaternion) -> Self::Output {
        Quaternion {
            s: other.s * self.s - other.x * self.x - other.y * self.y - other.z * self.z,
            x: other.s * self.x + other.x * self.s - other.y * self.z + other.z * self.y,
            y: other.s * self.y + other.x * self.z + other.y * self.s - other.z * self.x,
            z: other.s * self.z - other.x * self.y + other.y * self.x + other.z * self.s,
        }
    }
}

impl ops::Div<f32> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn div(self, other: f32) -> Quaternion {
        Quaternion {
            s: self.s / other, 
            x: self.x / other, 
            y: self.y / other, 
            z: self.z / other,
        }
    }
}

impl<'a> ops::Div<f32> for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn div(self, other: f32) -> Quaternion {
        Quaternion {
            s: self.s / other, 
            x: self.x / other, 
            y: self.y / other, 
            z: self.z / other,
        }
    }
}


#[cfg(test)]
mod mat4_tests {
    use std::slice::Iter;
    use super::{Matrix4};

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
                    a_mat: super::mat4(
                        80.0,   23.43,   43.569,  6.741, 
                        426.1,  23.5724, 27.6189, 13.90,
                        4.2219, 258.083, 31.70,   42.17, 
                        70.0,   49.0,    95.0,    89.9138
                    ),
                    b_mat: super::mat4(
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
                    a_mat: super::mat4(
                        68.32, 0.0,    0.0,   0.0,
                        0.0,   37.397, 0.0,   0.0,
                        0.0,   0.0,    9.483, 0.0,
                        0.0,   0.0,    0.0,   887.710
                    ),
                    b_mat: super::mat4(
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
        let v = super::vec3((2.0, 2.0, 2.0));
        let trans_mat = Matrix4::from_translation(v);
        let zero_vec4 = super::vec4((0.0, 0.0, 0.0, 1.0));
        let zero_vec3 = super::vec3((0.0, 0.0, 0.0));

        let result = trans_mat * zero_vec4;
        assert_eq!(result, super::vec4((zero_vec3 + v, 1.0)));
    }
}
