use std::fmt;
use std::mem;
use std::ops;
use std::cmp;

use crate::traits::*;
use crate::vector::*;
use crate::matrix::*;
use crate::quaternion::*;
use crate::projection::*;


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
pub fn mat2(c0r0: f32, c0r1: f32, c1r0: f32, c1r1: f32) -> Matrix2 {
    Matrix2::new(c0r0, c0r1, c1r0, c1r1)
}

///
/// Create a new 3x3 matrix in the style of a GLSL type
/// constructor.
///
#[inline]
pub fn mat3(
    c0r0: f32, c0r1: f32, c0r2: f32,
    c1r0: f32, c1r1: f32, c1r2: f32,
    c2r0: f32, c2r1: f32, c2r2: f32) -> Matrix3 {

    Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
}

///
/// Create a new 4x4 matrix in the style of a GLSL type
/// constructor.
///
#[inline]
pub fn mat4(
    c0r0: f32, c0r1: f32, c0r2: f32, c0r3: f32,
    c1r0: f32, c1r1: f32, c1r2: f32, c1r3: f32,
    c2r0: f32, c2r1: f32, c2r2: f32, c2r3: f32,
    c3r0: f32, c3r1: f32, c3r2: f32, c3r3: f32) -> Matrix4 {

    Matrix4::new(
        c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3,
        c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
    )
}

#[inline]
pub fn dot<V: DotProduct>(a: V, b: V) -> f32 {
    V::dot(a, b)
}
