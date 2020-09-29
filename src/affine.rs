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


/// A trait for implementing two-dimensional affine transformations.
pub trait AffineTransformation2<S> where Self: Sized {
    /// The identity transformation for this type.
    fn identity() -> Self;

    /// Compute the inverse of an affine transformation.
    fn inverse(&self) -> Option<Self>;

    /// Apply the affine transformation to a vector.
    fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S>;

    /// Apply the affine transformation to a point.
    fn transform_point(&self, point: &Point2<S>) -> Point2<S>;

    /// Apply the inverse of the affine transformation to a vector.
    #[inline]
    fn inverse_transform_vector(&self, vector: &Vector2<S>) -> Option<Vector2<S>> {
        self.inverse()
            .map(|matrix_inverse| matrix_inverse.transform_vector(vector))
    }

    /// Apply the inverse of the affine transformation to a vector.
    #[inline]
    fn inverse_transform_point(&self, point: &Point2<S>) -> Option<Point2<S>> {
        self.inverse()
            .map(|matrix_inverse| matrix_inverse.transform_point(point))
    }

    /// Convert a specific two-dimensional affine transformation into a generic 
    /// two-dimensional affine transformation.
    fn to_transform2d(&self) -> Transform2<S>;
}

/// A trait for implementing three-dimensional affine transformations.
pub trait AffineTransformation3<P, V, S> where Self: Sized {
    /// The output associated type for points. This allows us to use both 
    /// pointers and values on the inputs.
    type OutPoint;
    /// The output associated type for vectors. This allows us to use both 
    /// pointers and values on the inputs.
    type OutVector;

    /// The identity transformation for this type.
    fn identity() -> Self;

    /// Compute the inverse of an affine transformation.
    fn inverse(&self) -> Option<Self>;

    /// Apply the affine transformation to a vector.
    fn transform_vector(&self, vector: V) -> Self::OutVector;

    /// Apply the affine transformation to a point.
    fn transform_point(&self, point: P) -> Self::OutPoint;

    /// Apply the inverse of the affine transformation to a vector.
    #[inline]
    fn transform_inverse_vector(&self, vector: V) -> Option<Self::OutVector> {
        self.inverse()
            .map(|matrix_inverse| matrix_inverse.transform_vector(vector))
    }

    /// Convert a specific three-dimensional affine transformation into a 
    /// generic three-dimensional affine transformation.
    fn to_transform3d(&self) -> Transform3<S>;
}


/// A generic two dimensional affine transformation.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct Transform2<S> {
    /// The underlying matrix that implements the transformation.
    matrix: Matrix3x3<S>,
}

impl<S> Transform2<S> where S: Scalar {
    /// Convert a 3x3 matrix to a two-dimensional affine transformation. This 
    /// function is for internal use in implementing type conversions for 
    /// affine transformations.
    #[inline]
    pub(crate) fn matrix_to_transform2d(matrix: Matrix3x3<S>) -> Transform2<S> {
        // TODO: Make this function const when const fn stabilizes for traits other than
        // Sized. See issue #57563: <https://github.com/rust-lang/rust/issues/57563>
        Transform2 {
            matrix: matrix,
        }
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

impl<S> AffineTransformation2<S> for Transform2<S> 
    where S: ScalarFloat
{
    #[inline]
    fn identity() -> Transform2<S> {
        Transform2 { 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Transform2<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform2 {
                matrix: matrix
            })
        } else {
            None
        }
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2<S> {
        *self
    }
}


/// A generic three-dimensional affine transformation.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct Transform3<S> {
    /// The underlying matrix implementing the transformation.
    matrix: Matrix4x4<S>,
}

impl<S> Transform3<S> where S: Scalar {
    /// Convert a 4x4 matrix to a three-dimensional affine transformation. 
    /// This function is for internal use in implementing type conversions for 
    /// affine transformations.
    #[inline]
    pub(crate) fn matrix_to_transform3d(matrix: Matrix4x4<S>) -> Transform3<S> {
        // TODO: Make this function const when const fn stabilizes for traits other than
        // Sized. See issue #57563: <https://github.com/rust-lang/rust/issues/57563>.
        Transform3 {
            matrix: matrix,
        }
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
    fn from(transformation: Transform3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> From<&Transform3<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: &Transform3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation3<Point3<S>, Vector3<S>, S> for Transform3<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Transform3<S> {
        Transform3 { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Transform3<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform3 {
                matrix: matrix
            })
        } else {
            None
        }
    }

    #[inline]
    fn transform_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn transform_point(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3<S> {
        *self
    }
}

impl<S> AffineTransformation3<Point3<S>, &Vector3<S>, S> for Transform3<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Transform3<S> {
        Transform3 { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Transform3<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform3 {
                matrix: matrix
            })
        } else {
            None
        }
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn transform_point(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3<S> {
        *self
    }
}

impl<S> AffineTransformation3<&Point3<S>, Vector3<S>, S> for Transform3<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Transform3<S> {
        Transform3 { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Transform3<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform3 {
                matrix: matrix
            })
        } else {
            None
        }
    }

    #[inline]
    fn transform_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3<S> {
        *self
    }
}

impl<'a, 'b, S> AffineTransformation3<&'a Point3<S>, &'b Vector3<S>, S> for Transform3<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Transform3<S> {
        Transform3 { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Transform3<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform3 {
                matrix: matrix
            })
        } else {
            None
        }
    }

    #[inline]
    fn transform_vector(&self, vector: &'b Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn transform_point(&self, point: &'a Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3<S> {
        *self
    }
}


