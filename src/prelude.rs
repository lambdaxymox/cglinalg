use crate::scalar::{
    ScalarFloat
};
use crate::structure::*;
use crate::vector::*;
use crate::matrix::*;
use crate::quaternion::*;
use crate::projection::*;
use crate::point::*;


/// Compute the orthographic projection matrix for converting from camera space to
/// normalized device coordinates.
#[inline]
pub fn ortho<S, Spec: Into<Orthographic<S>>>(spec: Spec) -> Matrix4<S> where S: ScalarFloat {
    Matrix4::from(spec.into())
}

/// Compute a perspective matrix from a view frustum.
///
/// This is the equivalent of the now deprecated [glFrustum]
/// (http://www.opengl.org/sdk/docs/man2/xhtml/glFrustum.xml) function.
#[inline]
pub fn frustum<S, Spec: Into<Perspective<S>>>(spec: Spec) -> Matrix4<S> where S: ScalarFloat {
    Matrix4::from(spec.into())
}

/// Compute the perspective matrix for converting from camera space to 
/// normalized device coordinates. This is the equivalent to the
/// [gluPerspective] (http://www.opengl.org/sdk/docs/man2/xhtml/gluPerspective.xml)
/// function.
#[inline]
pub fn perspective<S, Spec: Into<PerspectiveFov<S>>>(spec: Spec) -> Matrix4<S> where S: ScalarFloat {
    Matrix4::from(spec.into())
}

/// Construct a new one-dimensional vector. This follows the style of
/// other GLSL vector constructors even though GLSL itself lacks a
/// `vec1()` function.
#[inline]
pub fn vec1<S, T: Into<Vector1<S>>>(v: T) -> Vector1<S> {
    v.into()
}

/// Construct a new two-dimensional vector in the style of
/// a GLSL `vec2` constructor.
#[inline]
pub fn vec2<S, T: Into<Vector2<S>>>(v: T) -> Vector2<S> {
    v.into()
}

/// Construct a new three-dimensional vector in the style of
/// a GLSL `vec3` constructor.
#[inline]
pub fn vec3<S, T: Into<Vector3<S>>>(v: T) -> Vector3<S> {
    v.into()
}

/// Construct a new four-dimensional vector in the style of
/// a GLSL `vec4` constructor.
#[inline]
pub fn vec4<S, T: Into<Vector4<S>>>(v: T) -> Vector4<S> {
    v.into()
}

/// Create a new quaternion in the style of a GLSL type
/// constructor. This is not a built-in function in GLSL, but it exists
/// for convenience.
#[inline]
pub fn quat<S, T: Into<Quaternion<S>>>(q: T) -> Quaternion<S> {
    q.into()
}

/// Create a new 2x2 matrix in the style of a GLSL type
/// constructor.
#[inline]
pub fn mat2<S, T: Into<Matrix2<S>>>(m: T) -> Matrix2<S> {
    m.into()
}

/// Create a new 3x3 matrix in the style of a GLSL type
/// constructor.
#[inline]
pub fn mat3<S, T: Into<Matrix3<S>>>(m: T) -> Matrix3<S> {
    m.into()
}

/// Create a new 4x4 matrix in the style of a GLSL type
/// constructor.
#[inline]
pub fn mat4<S, T: Into<Matrix4<S>>>(m: T) -> Matrix4<S> {
    m.into()
}

/// Compute the dot product between two vectors.
#[inline]
pub fn dot<W: Copy + Clone, V: DotProduct<W>>(a: V, b: W) -> <V as DotProduct<W>>::Output {
    V::dot(a, b)
}


#[cfg(test)]
mod tests {
    use matrix::{Matrix2, Matrix3, Matrix4};

    #[test]
    fn test_mat2() {
        let expected = Matrix2::new(
            1_f32, 2_f32, 
            3_f32, 4_f32
        );
        let result = super::mat2([
            1_f32, 2_f32, 3_f32, 4_f32
        ]);

        assert_eq!(result, expected);
    }

        #[test]
    fn test_mat3() {
        let expected = Matrix3::new(
            1_f32, 2_f32, 3_f32, 
            4_f32, 5_f32, 6_f32, 
            7_f32, 8_f32, 9_f32
        );
        let result = super::mat3([
            1_f32, 2_f32, 3_f32, 4_f32, 5_f32, 6_f32, 7_f32, 8_f32, 9_f32
        ]);

        assert_eq!(result, expected);
    }

        #[test]
    fn test_mat4() {
        let expected = Matrix4::new(
            1_f32,  2_f32,  3_f32,  4_f32,  5_f32,  6_f32,  7_f32,  8_f32, 
            9_f32, 10_f32, 11_f32, 12_f32, 13_f32, 14_f32, 15_f32, 15_f32
        );
        let result = super::mat4([
            1_f32,  2_f32,  3_f32,  4_f32,  5_f32,  6_f32,  7_f32,  8_f32, 
            9_f32, 10_f32, 11_f32, 12_f32, 13_f32, 14_f32, 15_f32, 15_f32
        ]);

        assert_eq!(result, expected);
    }
}
