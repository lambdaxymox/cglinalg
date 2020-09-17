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
pub fn vec1<S, T>(vector: T) -> Vector1<S> 
    where T: Into<Vector1<S>>
{
    vector.into()
}

/// Construct a new two-dimensional vector in the style of
/// a GLSL `vec2` constructor.
pub fn vec2<S, T>(vector: T) -> Vector2<S>
    where T: Into<Vector2<S>>
{
    vector.into()
}

/// Construct a new three-dimensional vector in the style of
/// a GLSL `vec3` short constructor.
pub fn vec3<S, T>(vector: T) -> Vector3<S> 
    where T: Into<Vector3<S>>
{
    vector.into()
}

/// Construct a new four-dimensional vector in the style of
/// a GLSL `vec4` short constructor.
pub fn vec4<S, T>(vector: T) -> Vector4<S> 
    where T: Into<Vector4<S>>
{
    vector.into()
}

/// Create a new quaternion in the style of a GLSL type
/// constructor. This is not a built-in function in GLSL, but it exists
/// for convenience.
pub fn quat<S, T>(quaternion: T) -> Quaternion<S> 
    where T: Into<Quaternion<S>>
{
    quaternion.into()
}

/// Create a new 2x2 matrix in the style of a GLSL `mat2` type constructor.
pub fn mat2<S, T>(matrix: T) -> Matrix2x2<S> 
    where T: Into<Matrix2x2<S>>
{
    matrix.into()
}

/// Create a new 3x3 matrix in the style of a GLSL `mat3` type constructor.
pub fn mat3<S, T>(matrix: T) -> Matrix3x3<S> 
    where T: Into<Matrix3x3<S>>
{
    matrix.into()
}

/// Create a new 4x4 matrix in the style of a GLSL type constructor.
pub fn mat4<S, T>(matrix: T) -> Matrix4x4<S> 
    where T: Into<Matrix4x4<S>>
{
    matrix.into()
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
pub fn cross<V, W, S>(v1: V, v2: W) -> <V as CrossProduct<W>>::Output
    where 
        S: Scalar,
        V: CrossProduct<W>,
{
    v1.cross(v2)
}
