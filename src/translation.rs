use crate::scalar::{
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
use crate::affine::*;

use core::fmt;


/// A translation transformation in two dimensions.
///
/// A translation is an operation that creates displacement motions. 
/// In a Euclidean setting, translations preserve differences between two points 
/// and acts as the identity on vectors.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct Translation2<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix3x3<S>,
}

impl<S> Translation2<S> where S: ScalarSigned {
    /// Construct a translation operator from a vector of displacements.
    #[inline]
    pub fn from_vector(distance: Vector2<S>) -> Translation2<S> {
        Translation2 {
            matrix: Matrix3x3::from_affine_translation(distance),
        }
    }

    /// Construct a translation operator from a vector of displacements.
    pub fn from_translation(distance: Vector2<S>) -> Translation2<S> {
        Translation2 {
            matrix: Matrix3x3::from_affine_translation(distance),
        }
    }

    /// Construct a translation between two vectors.
    #[inline]
    pub fn between_vectors(vector1: Vector2<S>, vector2: Vector2<S>) -> Self {
        let distance = vector2 - vector1;

        Translation2::from_vector(distance)
    }

    /// Construct a translation between two points.
    #[inline]
    pub fn between_points(point1: Point2<S>, point2: Point2<S>) -> Self {
        let distance = point2 - point1;

        Translation2::from_vector(distance)
    }

    /// Construct a translation that translates a vector or point in the opposite
    /// direction of the translation applied by `self`.
    ///
    /// If `self` is a translation of a vector by a displacement `distance`, then its
    /// inverse will be a translation by a displacement `-distance`.
    #[inline]
    pub fn inverse(&self) -> Self {
        let distance = Vector2::new(-self.matrix.c2r0, -self.matrix.c2r1);

        Translation2::from_vector(distance)
    }
    
    /// Apply the translation operation to a point.
    #[inline]
    pub fn translate_point(&self, point: &Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    /// Apply the translation operation to a vector. 
    ///
    /// This should act as the identity since vectors represent differences 
    /// between points. Let `p1` and `p2` be points and let `v = p2 - p1` 
    /// be their difference. If we translate each point by a vector `a`, 
    /// then `(p2 + a) - (p1 + a) = p2 - p1 = v`.
    #[inline]
    pub fn translate_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }
}

impl<S> AsRef<Matrix3x3<S>> for Translation2<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Translation2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Translation2 [{}]", self.matrix)
    }
}

impl<S> From<Translation2<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: Translation2<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> From<&Translation2<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: &Translation2<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation2<S> for Translation2<S> 
    where S: ScalarSigned 
{
    #[inline]
    fn identity() -> Translation2<S> {
        Translation2 { 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation2<S>> {
        Some(self.inverse())
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.translate_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        self.translate_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2<S> {
        Transform2::matrix_to_transform2d(self.matrix)
    }
}


/// A translation transformation in three dimensions.
///
/// A translation is an operation that creates displacement motions. 
/// In a Euclidean setting, translations preserve differences between two points 
/// and acts as the identity on vectors.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct Translation3<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix4x4<S>,
}

impl<S> Translation3<S> where S: ScalarSigned {
    /// Construct a translation operator from a vector of displacements.
    pub fn from_vector(distance: Vector3<S>) -> Translation3<S> {
        Translation3 {
            matrix: Matrix4x4::from_affine_translation(distance),
        }
    }

    /// Construct a translation operator from a vector of displacements.
    pub fn from_translation(distance: Vector3<S>) -> Translation3<S> {
        Translation3 {
            matrix: Matrix4x4::from_affine_translation(distance),
        }
    }

    /// Construct a translation between two vectors.
    #[inline]
    pub fn between_vectors(vector1: Vector3<S>, vector2: Vector3<S>) -> Self {
        let distance = vector2 - vector1;

        Translation3::from_vector(distance)
    }

    /// Construct a translation between two points.
    #[inline]
    pub fn between_points(point1: Point3<S>, point2: Point3<S>) -> Self {
        let distance = point2 - point1;

        Translation3::from_vector(distance)
    }

    /// Construct a translation that translates a vector or point in the opposite
    /// direction of the translation applied by `self`.
    ///
    /// If `self` is a translation of a vector by a displacement `distance`, then its
    /// inverse will be a translation by a displacement `-distance`.
    #[inline]
    pub fn inverse(&self) -> Self {
        let distance = Vector3::new(
            -self.matrix.c3r0, 
            -self.matrix.c3r1, 
            -self.matrix.c3r2
        );

        Translation3::from_vector(distance)
    }
    
    #[inline]
    pub fn translate_point(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    pub fn translate_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }
}

impl<S> AsRef<Matrix4x4<S>> for Translation3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Translation3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Translation3 [{}]", self.matrix)
    }
}

impl<S> From<Translation3<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: Translation3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> From<&Translation3<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: &Translation3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation3<Point3<S>, Vector3<S>, S> for Translation3<S> 
    where 
        S: ScalarSigned 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Translation3<S> {
        Translation3 { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation3<S>> {
        Some(self.inverse())
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
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3<Point3<S>, &Vector3<S>, S> for Translation3<S> 
    where 
        S: ScalarSigned 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Translation3<S> {
        Translation3 { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation3<S>> {
        Some(self.inverse())
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
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3<&Point3<S>, Vector3<S>, S> for Translation3<S> 
    where S: ScalarSigned 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Translation3<S> {
        Translation3 { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation3<S>> {
        Some(self.inverse())
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
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::matrix_to_transform3d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation3<&'a Point3<S>, &'b Vector3<S>, S> for Translation3<S> 
    where 
        S: ScalarSigned 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Translation3<S> {
        Translation3 { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation3<S>> {
        Some(self.inverse())
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
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::matrix_to_transform3d(self.matrix)
    }
}

