use crate::scalar::{
    Scalar,
    ScalarFloat,
};
use crate::matrix::{
    Matrix3x3,
    Matrix4x4,
};
use crate::vector::{
    Vector2,
    Vector3,
};
use crate::point::{
    Point2,
    Point3,
};
use crate::traits::{
    Identity,
    InvertibleSquareMatrix,
};

use core::fmt;


/// A generic two dimensional transformation.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform2<S> {
    /// The underlying matrix that implements the transformation.
    matrix: Matrix3x3<S>,
}

impl<S> Transform2<S> where S: Scalar {
    /// Convert a 3x3 matrix to a two-dimensional transformation. This 
    /// function is for internal use in implementing type conversions.
    #[inline]
    pub(crate) fn to_transform2d<T: Into<Matrix3x3<S>>>(transform: T) -> Transform2<S> {
        // TODO: Make this function const when const fn stabilizes for traits other than
        // Sized. See issue #57563: <https://github.com/rust-lang/rust/issues/57563>
        Transform2 {
            matrix: transform.into(),
        }
    }

    /// Get a reference to the underlying matrix that represents the 
    /// transformation.
    #[inline]
    pub fn matrix(&self) -> &Matrix3x3<S> {
        &self.matrix
    }

    /// The identity transformation for a generic two-dimensional 
    /// transformation.
    #[inline]
    pub fn identity() -> Transform2<S> {
        Transform2 { 
            matrix: Matrix3x3::identity(),
        }
    }
}

impl<S> Transform2<S> where S: ScalarFloat {
    /// Compute the inverse of the transformation if it exists.
    #[inline]
    pub fn inverse(&self) -> Option<Transform2<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform2 {
                matrix: matrix
            })
        } else {
            None
        }
    }

    /// Apply the transformation to a vector.
    #[inline]
    pub fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Apply the inverse of the transformation to a point.
    #[inline]
    pub fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous()).unwrap()
    }

    /// Apply the inverse of the transformation to a vector.
    #[inline]
    pub fn inverse_transform_vector(&self, vector: &Vector2<S>) -> Option<Vector2<S>> {
        self.inverse()
            .map(|matrix_inverse| matrix_inverse.transform_vector(vector))
    }

    /// Apply the inverse of the transformation to a point.
    #[inline]
    pub fn inverse_transform_point(&self, point: &Point2<S>) -> Option<Point2<S>> {
        self.inverse()
            .map(|matrix_inverse| matrix_inverse.transform_point(point))
    }
}

impl<S> AsRef<Matrix3x3<S>> for Transform2<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Transform2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Transform2 [{}]",
            self.matrix
        )
    }
}

impl<S> From<Transform2<S>> for Matrix3x3<S> where S: Copy {
    #[inline]
    fn from(transformation: Transform2<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> From<&Transform2<S>> for Matrix3x3<S> where S: Copy {
    #[inline]
    fn from(transformation: &Transform2<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> approx::AbsDiffEq for Transform2<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Matrix3x3::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S> approx::RelativeEq for Transform2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix3x3::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Transform2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix3x3::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}


/// A generic three-dimensional transformation.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform3<S> {
    /// The underlying matrix implementing the transformation.
    matrix: Matrix4x4<S>,
}

impl<S> Transform3<S> where S: Scalar {
    /// Convert a 4x4 matrix to a three-dimensional transformation. 
    /// This function is for internal use in implementing type conversions.
    #[inline]
    pub(crate) fn to_transform3d<T: Into<Matrix4x4<S>>>(transform: T) -> Transform3<S> {
        // TODO: Make this function const when const fn stabilizes for traits other than
        // Sized. See issue #57563: <https://github.com/rust-lang/rust/issues/57563>.
        Transform3 {
            matrix: transform.into(),
        }
    }

    /// Get a reference to the underlying matrix that represents the 
    /// transformation.
    #[inline]
    pub fn matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// The identity transformation for a generic three-dimensional 
    /// transformation.
    #[inline]
    pub fn identity() -> Transform3<S> {
        Transform3 { 
            matrix: Matrix4x4::identity(),
        }
    }
}

impl<S> Transform3<S> where S: ScalarFloat {
    /// Compute the inverse of the transformation if it exists.
    #[inline]
    pub fn inverse(&self) -> Option<Transform3<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform3 {
                matrix: matrix
            })
        } else {
            None
        }
    }

    /// Apply the transformation to a vector.
    #[inline]
    pub fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Apply the inverse of the transformation to a point.
    #[inline]
    pub fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous()).unwrap()
    }

    /// Apply the inverse of the transformation to a vector.
    #[inline]
    pub fn transform_inverse_vector(&self, vector: &Vector3<S>) -> Option<Vector3<S>> {
        self.inverse()
            .map(|matrix_inverse| matrix_inverse.transform_vector(vector))
    }

    /// Apply the inverse of the transformation to a point.
    #[inline]
    pub fn inverse_transform_point(&self, point: &Point3<S>) -> Option<Point3<S>> {
        self.inverse()
            .map(|matrix_inverse| matrix_inverse.transform_point(point))
    }
}

impl<S> AsRef<Matrix4x4<S>> for Transform3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Transform3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Transform3 [{}]",
            self.matrix
        )
    }
}

impl<S> From<Transform3<S>> for Matrix4x4<S> where S: Copy {
    #[inline]
    fn from(transformation: Transform3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> From<&Transform3<S>> for Matrix4x4<S> where S: Copy {
    #[inline]
    fn from(transformation: &Transform3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> approx::AbsDiffEq for Transform3<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Matrix4x4::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S> approx::RelativeEq for Transform3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix4x4::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Transform3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix4x4::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

