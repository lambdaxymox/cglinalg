use std::fmt;
use std::mem;
use std::ops;
use std::cmp;

use crate::traits::Array;
use crate::vector::*;
use crate::matrix::*;
use crate::quaternion::*;


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


