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
use crate::affine::*;

use core::fmt;


/// The scale transformation in two dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct Scale2<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix3x3<S>,
}

impl<S> Scale2<S> where S: Scalar {
    /// Construct a scale transformation from a vector of scale factors.
    #[inline]
    pub fn from_vector(scale: Vector2<S>) -> Scale2<S> {
        Scale2 {
            matrix: Matrix3x3::from_affine_nonuniform_scale(scale.x, scale.y),
        }
    }

    /// Construct a two-dimensional scale transformation from a nonuniform scale 
    /// across coordinates.
    #[inline]
    pub fn from_nonuniform_scale(scale_x: S, scale_y: S) -> Scale2<S> {
        Scale2 {
            matrix: Matrix3x3::from_affine_nonuniform_scale(scale_x, scale_y),
        }
    }

    /// Construct a two-dimensional scale transformation from a uniform scale 
    /// factor.
    #[inline]
    pub fn from_scale(scale: S) -> Scale2<S> {
        Scale2 {
            matrix: Matrix3x3::from_scale(scale),
        }
    }

    /// Construct a scale transformation that scales each coordinate by the 
    /// reciprocal of the scaling factors of the scale operator `self`.
    #[inline]
    pub fn inverse(&self) -> Option<Scale2<S>> {
        Some(Scale2::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1
        ))
    }

    /// Apply a scale operation to a vector.
    #[inline]
    pub fn scale_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Apply a scale operation to a point.
    #[inline]
    pub fn scale_point(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> AsRef<Matrix3x3<S>> for Scale2<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Scale2<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Scale2<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: Scale2<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> From<&Scale2<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: &Scale2<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation2<Point2<S>, Vector2<S>, S> for Scale2<S> 
    where S: ScalarFloat 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Scale2<S> {
        Scale2::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2<S>> {
        self.inverse()
    }

    #[inline]
    fn transform_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        self.scale_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: Point2<S>) -> Point2<S> {
        self.scale_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2<S> {
        Transform2::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2<Point2<S>, &Vector2<S>, S> for Scale2<S> 
    where S: ScalarFloat 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Scale2<S> {
        Scale2::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2<S>> {
        self.inverse()
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.scale_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: Point2<S>) -> Point2<S> {
        self.scale_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2<S> {
        Transform2::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2<&Point2<S>, Vector2<S>, S> for Scale2<S> 
    where S: ScalarFloat 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Scale2<S> {
        Scale2::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2<S>> {
        self.inverse()
    }

    #[inline]
    fn transform_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        self.scale_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        self.scale_point(*point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2<S> {
        Transform2::matrix_to_transform2d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation2<&'a Point2<S>, &'b Vector2<S>, S> for Scale2<S> 
    where S: ScalarFloat 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Scale2<S> {
        Scale2::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2<S>> {
        self.inverse()
    }

    #[inline]
    fn transform_vector(&self, vector: &'b Vector2<S>) -> Vector2<S> {
        self.scale_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: &'a Point2<S>) -> Point2<S> {
        self.scale_point(*point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2<S> {
        Transform2::matrix_to_transform2d(self.matrix)
    }
}


/// The scale transformation in three dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct Scale3<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix4x4<S>,
}

impl<S> Scale3<S> where S: Scalar {
    /// Construct a scale transformation from a vector of scale factors.
    pub fn from_vector(scale: Vector3<S>) -> Scale3<S> {
        Scale3 {
            matrix: Matrix4x4::from_affine_nonuniform_scale(
                scale.x, 
                scale.y, 
                scale.z
            ),
        }
    }

    /// Construct a three-dimensional scale transformation from a nonuniform scale 
    /// across coordinates.
    #[inline]
    pub fn from_nonuniform_scale(scale_x: S, scale_y: S, scale_z: S) -> Scale3<S> {
        Scale3 {
            matrix: Matrix4x4::from_affine_nonuniform_scale(
                scale_x, 
                scale_y, 
                scale_z
            ),
        }
    }

    /// Construct a three-dimensional scale transformation from a uniform scale 
    /// factor.
    #[inline]
    pub fn from_scale(scale: S) -> Scale3<S> {
        Scale3 {
            matrix: Matrix4x4::from_affine_scale(scale),
        }
    }

    /// Construct a scale transformation that scales each coordinate by the 
    /// reciprocal of the scaling factors of the scale operator `self`.
    #[inline]
    pub fn inverse(&self) -> Option<Scale3<S>> {
        Some(Scale3::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1,
            S::one() / self.matrix.c2r2
        ))
    }

    /// Apply a scale operation to a vector.
    #[inline]
    pub fn scale_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Apply a scale operation to a point.
    #[inline]
    pub fn scale_point(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> AsRef<Matrix4x4<S>> for Scale3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Scale3<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Scale3<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: Scale3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> From<&Scale3<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: &Scale3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation3<Point3<S>, Vector3<S>, S> for Scale3<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Scale3<S> {
        Scale3::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3<S>> {
        self.inverse()
    }

    #[inline]
    fn transform_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        self.scale_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: Point3<S>) -> Point3<S> {
        self.scale_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3<Point3<S>, &Vector3<S>, S> for Scale3<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Scale3<S> {
        Scale3::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3<S>> {
        self.inverse()
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.scale_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: Point3<S>) -> Point3<S> {
        self.scale_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3<&Point3<S>, Vector3<S>, S> for Scale3<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Scale3<S> {
        Scale3::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3<S>> {
        self.inverse()
    }

    #[inline]
    fn transform_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        self.scale_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        self.scale_point(*point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::matrix_to_transform3d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation3<&'a Point3<S>, &'b Vector3<S>, S> for Scale3<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Scale3<S> {
        Scale3::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3<S>> {
        self.inverse()
    }

    #[inline]
    fn transform_vector(&self, vector: &'b Vector3<S>) -> Vector3<S> {
        self.scale_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: &'a Point3<S>) -> Point3<S> {
        self.scale_point(*point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::matrix_to_transform3d(self.matrix)
    }
}
