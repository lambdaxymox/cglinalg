use crate::vector::{
    Vector1,
    Vector2,
    Vector3,
    Vector4,
};
use crate::matrix::{
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
};
use crate::scalar::{
    Scalar,
    ScalarFloat,
};
use crate::traits::{
    DotProduct,
    CrossProduct,
};
use crate::quaternion::{
    Quaternion,
};
use crate::projection::{
    Orthographic,
    Perspective,
    PerspectiveFov,
};


/// Construct a new one-dimensional vector. This follows the style of
/// other GLSL vector constructors even though GLSL itself lacks a
/// `vec1()` function.
pub fn vec1<S: Scalar>(x: S) -> Vector1<S> {
    Vector1::new(x)
}

/// Construct a new two-dimensional vector in the style of
/// a GLSL `vec2` constructor.
pub fn vec2<S: Scalar>(x: S, y: S) -> Vector2<S> {
    Vector2::new(x, y)
}

/// Construct a new three-dimensional vector in the style of
/// a GLSL `vec3` short constructor.
pub fn vec3<S: Scalar>(x: S, y: S, z: S) -> Vector3<S> {
    Vector3::new(x, y, z)
}

/// Construct a new four-dimensional vector in the style of
/// a GLSL `vec4` short constructor.
pub fn vec4<S: Scalar>(x: S, y: S, z: S, w: S) -> Vector4<S> {
    Vector4::new(x, y, z, w)
}

/// Create a new quaternion in the style of a GLSL type
/// constructor. This is not a built-in function in GLSL, but it exists
/// for convenience.
pub fn quat<S: Scalar>(s: S, x: S, y: S, z: S) -> Quaternion<S> {
    Quaternion::new(s, x, y, z)
}

/// Create a new 2x2 matrix in the style of a GLSL `mat2` type constructor.
pub fn mat2<S: Scalar>(m00: S, m01: S, m10: S, m11: S) -> Matrix2x2<S> {
    Matrix2x2::new(m00, m01, m10, m11)
}

/// Create a new 3x3 matrix in the style of a GLSL `mat3` type constructor.
pub fn mat3<S: Scalar>(
    m00: S, m01: S, m02: S, m10: S, m11: S, m12: S, m20: S, m21: S, m22: S) -> Matrix3x3<S> {
    
    Matrix3x3::new(
        m00, m01, m02, 
        m10, m11, m12, 
        m20, m21, m22
    )
}

/// Create a new 4x4 matrix in the style of a GLSL `mat4` type constructor.
pub fn mat4<S: Scalar>(
    m00: S, m01: S, m02: S, m03: S, 
    m10: S, m11: S, m12: S, m13: S,
    m20: S, m21: S, m22: S, m23: S,
    m30: S, m31: S, m32: S, m33: S) -> Matrix4x4<S> {
    
    Matrix4x4::new(
        m00, m01, m02, m03,
        m10, m11, m12, m13,
        m20, m21, m22, m23,
        m30, m31, m32, m33
    )
}

/// Compute the orthographic projection matrix for converting from camera space to
/// normalized device coordinates.
///
/// This function is equivalent to the now deprecated [glOrtho]
/// (https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml) function.
pub fn ortho<S, Spec>(spec: Spec) -> Matrix4x4<S> 
    where 
        S: ScalarFloat,
        Spec: Into<Orthographic<S>>,
{
    Matrix4x4::from(spec.into())
}

/// Compute a perspective matrix from a view frustum.
///
/// This is the equivalent of the now deprecated [glFrustum]
/// (https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml) function.
pub fn frustum<S, Spec>(spec: Spec) -> Matrix4x4<S> 
    where 
        S: ScalarFloat,
        Spec: Into<Perspective<S>>        
{
    Matrix4x4::from(spec.into())
}

/// Compute the perspective matrix for converting from camera space to 
/// normalized device coordinates. 
///
/// This is the equivalent to the [gluPerspective] 
/// (https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml)
/// function.
pub fn perspective<S, Spec>(spec: Spec) -> Matrix4x4<S> 
    where 
        S: ScalarFloat,
        Spec: Into<PerspectiveFov<S>>
{
    Matrix4x4::from(spec.into())
}

/// Compute the dot product between two vectors.
pub fn dot<W, V>(v1: V, v2: W) -> <V as DotProduct<W>>::Output 
    where
        W: Copy + Clone,
        V: DotProduct<W>,
{
    V::dot(v1, v2)
}

/// Compute the cross product of two three-dimensional vectors.
pub fn cross<S: Scalar>(v1: &Vector3<S>, v2: &Vector3<S>) -> Vector3<S> {
    v1.cross(v2)
}
