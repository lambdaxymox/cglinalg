use crate::scalar::{
    Scalar,
    ScalarSigned,
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
};
use crate::transformation::*;

use core::fmt;


/// A type implementing this trait represents a type that acts as a generic 
/// translation. 
///
/// A translation is an operation that creates displacement motions. 
/// In a Euclidean setting, translations preserve differences between two points 
/// and acts as the identity on vectors.
pub trait Translation<P, V> where Self: Sized + Copy {
    /// The type of the output points (locations in space).
    type OutPoint;
    /// The type of the output vectors (displacements in space).
    type OutVector;

    /// Construct a translation between two vectors.
    fn between_vectors(vector1: V, vector2: V) -> Self;

    /// Construct a translation between two points.
    fn between_points(point1: P, point2: P) -> Self;

    /// Construct a rotation that rotates a vector in the opposite direction 
    /// of `self`.
    /// 
    /// Given a rotation operator that rotates a vector about an axis by an 
    /// angle `theta`, construct a rotation that rotates a vector about the 
    /// same axis by an angle `-theta`.
    fn inverse(&self) -> Self;

    /// Apply the rotation operation to a point.
    fn translate_point(&self, point: P) -> Self::OutPoint;

    /// Apply the translation operation to a vector. This should act as the
    /// identity since vectors represent displacements. That is, let `p1` and 
    /// `p2` be points and let `v = p2 - p1` be their difference. If we translate 
    /// each point by a vector `a`, then `(p2 + a) - (p1 + a) = p2 - p1 = v`.
    fn translate_vector(&self, vector: V) -> Self::OutVector;
}

/// A trait for types implementing translation operators in two dimensions.
pub trait Translation2<S> where 
    S: ScalarSigned,
    Self: Translation<Point2<S>, Vector2<S>> + Into<Matrix3x3<S>> + Into<Translation2D<S>>,
{
    /// Construct a translation operator from a vector of displacements.
    fn from_translation(distance: Vector2<S>) -> Self;
}

/// A trait for types implementing translation operators in three dimensions.
pub trait Translation3<S> where 
    S: ScalarSigned,
    Self: Translation<Point3<S>, Vector3<S>> + Into<Matrix4x4<S>> + Into<Translation3D<S>>,
{
    /// Construct a translation operator from a vector of displacements.
    fn from_translation(distance: Vector3<S>) -> Self;
}


/// A translation transformation in two dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Translation2D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix3x3<S>,
}

impl<S> Translation2D<S> where S: Scalar {
    /// Construct a translation operator from a vector of displacements.
    #[inline]
    pub fn from_vector(distance: Vector2<S>) -> Translation2D<S> {
        Translation2D {
            matrix: Matrix3x3::from_affine_translation(distance),
        }
    }
}

impl<S> AsRef<Matrix3x3<S>> for Translation2D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Translation2D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Translation2D<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: Translation2D<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> From<&Translation2D<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: &Translation2D<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> Translation<Point2<S>, Vector2<S>> for Translation2D<S> 
    where 
        S: ScalarSigned 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn between_vectors(vector1: Vector2<S>, vector2: Vector2<S>) -> Self {
        let distance = vector2 - vector1;

        Translation2D::from_vector(distance)
    }

    #[inline]
    fn between_points(point1: Point2<S>, point2: Point2<S>) -> Self {
        let distance = point2 - point1;

        Translation2D::from_vector(distance)
    }

    #[inline]
    fn inverse(&self) -> Self {
        let distance = Vector2::new(-self.matrix.c2r0, -self.matrix.c2r1);

        Translation2D::from_vector(distance)
    }
    
    #[inline]
    fn translate_point(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn translate_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }
}

impl<S> Translation2<S> for Translation2D<S> 
    where 
        S: ScalarSigned 
{    
    fn from_translation(distance: Vector2<S>) -> Translation2D<S> {
        Translation2D {
            matrix: Matrix3x3::from_affine_translation(distance),
        }
    }
}

impl<S> AffineTransformation2D<Point2<S>, Vector2<S>, S> for Translation2D<S> 
    where 
        S: ScalarSigned 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Translation2D<S> {
        Translation2D { 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation2D<S>> {
        Some(<Self as Translation<Point2<S>, Vector2<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        self.translate_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: Point2<S>) -> Point2<S> {
        self.translate_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2D<Point2<S>, &Vector2<S>, S> for Translation2D<S> 
    where 
        S: ScalarSigned
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Translation2D<S> {
        Translation2D { 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation2D<S>> {
        Some(<Self as Translation<Point2<S>, Vector2<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.translate_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: Point2<S>) -> Point2<S> {
        self.translate_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2D<&Point2<S>, Vector2<S>, S> for Translation2D<S> 
    where 
        S: ScalarSigned 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Translation2D<S> {
        Translation2D { 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation2D<S>> {
        Some(<Self as Translation<Point2<S>, Vector2<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        self.translate_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        self.translate_point(*point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation2D<&'a Point2<S>, &'b Vector2<S>, S> for Translation2D<S>
    where 
        S: ScalarSigned 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Translation2D<S> {
        Translation2D { 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation2D<S>> {
        Some(<Self as Translation<Point2<S>, Vector2<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: &'b Vector2<S>) -> Vector2<S> {
        self.translate_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: &'a Point2<S>) -> Point2<S> {
        self.translate_point(*point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}


/// A translation transformation in three dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Translation3D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix4x4<S>,
}

impl<S> Translation3D<S> where S: Scalar {
    /// Construct a translation operator from a vector of displacements.
    pub fn from_vector(distance: Vector3<S>) -> Translation3D<S> {
        Translation3D {
            matrix: Matrix4x4::from_affine_translation(distance),
        }
    }
}

impl<S> AsRef<Matrix4x4<S>> for Translation3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Translation3D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Translation3D<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: Translation3D<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> From<&Translation3D<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: &Translation3D<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> Translation<Point3<S>, Vector3<S>> for Translation3D<S> 
    where 
        S: ScalarSigned 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn between_vectors(vector1: Vector3<S>, vector2: Vector3<S>) -> Self {
        let distance = vector2 - vector1;

        Translation3D::from_vector(distance)
    }

    #[inline]
    fn between_points(point1: Point3<S>, point2: Point3<S>) -> Self {
        let distance = point2 - point1;

        Translation3D::from_vector(distance)
    }

    #[inline]
    fn inverse(&self) -> Self {
        let distance = Vector3::new(
            -self.matrix.c3r0, 
            -self.matrix.c3r1, 
            -self.matrix.c3r2
        );

        Translation3D::from_vector(distance)
    }
    
    #[inline]
    fn translate_point(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn translate_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }
}

impl<S> Translation3<S> for Translation3D<S> 
    where 
        S: ScalarSigned 
{
    fn from_translation(distance: Vector3<S>) -> Translation3D<S> {
        Translation3D {
            matrix: Matrix4x4::from_affine_translation(distance),
        }
    }
}

impl<S> AffineTransformation3D<Point3<S>, Vector3<S>, S> for Translation3D<S> 
    where 
        S: ScalarSigned 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Translation3D<S> {
        Translation3D { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation3D<S>> {
        Some(<Self as Translation<Point3<S>, Vector3<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        self.translate_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: Point3<S>) -> Point3<S> {
        self.translate_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3D<Point3<S>, &Vector3<S>, S> for Translation3D<S> 
    where 
        S: ScalarSigned 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Translation3D<S> {
        Translation3D { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation3D<S>> {
        Some(<Self as Translation<Point3<S>, Vector3<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.translate_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: Point3<S>) -> Point3<S> {
        self.translate_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3D<&Point3<S>, Vector3<S>, S> for Translation3D<S> 
    where S: ScalarSigned 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Translation3D<S> {
        Translation3D { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation3D<S>> {
        Some(<Self as Translation<Point3<S>, Vector3<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        self.translate_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        self.translate_point(*point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation3D<&'a Point3<S>, &'b Vector3<S>, S> for Translation3D<S> 
    where 
        S: ScalarSigned 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Translation3D<S> {
        Translation3D { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation3D<S>> {
        Some(<Self as Translation<Point3<S>, Vector3<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: &'b Vector3<S>) -> Vector3<S> {
        self.translate_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: &'a Point3<S>) -> Point3<S> {
        self.translate_point(*point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

