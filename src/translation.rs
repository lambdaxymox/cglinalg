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
    Zero,
};
use crate::affine::*;

use core::fmt;


/// A translation transformation in two dimensions.
///
/// A translation is an operation that creates displacement motions. 
/// In a Euclidean setting, translations preserve differences between two points 
/// and acts as the identity on vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Translation2<S> {
    distance: Vector2<S>,
}

impl<S> Translation2<S> where S: ScalarSigned {
    /// Construct a translation operator from a vector of displacements.
    #[inline]
    pub fn from_vector(vector: &Vector2<S>) -> Translation2<S> {
        Translation2 {
            distance: vector.clone(),
        }
    }

    /// Construct a translation between two vectors.
    #[inline]
    pub fn translation_between_vectors(vector1: &Vector2<S>, vector2: &Vector2<S>) -> Self {
        let distance = vector2 - vector1;

        Translation2::from_vector(&distance)
    }

    /// Construct a translation between two points.
    #[inline]
    pub fn translation_between_points(point1: &Point2<S>, point2: &Point2<S>) -> Self {
        let distance = point2 - point1;

        Translation2::from_vector(&distance)
    }

    /// Construct a translation that translates a vector or point in the opposite
    /// direction of the translation applied by `self`.
    ///
    /// If `self` is a translation of a vector by a displacement `distance`, then its
    /// inverse will be a translation by a displacement `-distance`.
    #[inline]
    pub fn inverse(&self) -> Self {
        Translation2::from_vector(&(-self.distance))
    }
    
    /// Apply the translation operation to a point.
    #[inline]
    pub fn translate_point(&self, point: &Point2<S>) -> Point2<S> {
        point + self.distance
    }

    /// Apply the translation operation to a vector. 
    ///
    /// This should act as the identity since vectors represent differences 
    /// between points. Let `p1` and `p2` be points and let `v = p2 - p1` 
    /// be their difference. If we translate each point by a vector `a`, 
    /// then `(p2 + a) - (p1 + a) = p2 - p1 = v`.
    #[inline]
    pub fn translate_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        vector + self.distance
    }
}

impl<S> AsRef<Vector2<S>> for Translation2<S> {
    #[inline]
    fn as_ref(&self) -> &Vector2<S> {
        &self.distance
    }
}

impl<S> fmt::Display for Translation2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Translation2 [x={}, y={}]", 
            self.distance.x, self.distance.y
        )
    }
}

impl<S> From<Translation2<S>> for Matrix3x3<S> where S: ScalarSigned {
    fn from(transform: Translation2<S>) -> Matrix3x3<S> {
        Matrix3x3::from_affine_translation(&transform.distance)
    }
}

impl<S> From<&Translation2<S>> for Matrix3x3<S> where S: ScalarSigned {
    fn from(transform: &Translation2<S>) -> Matrix3x3<S> {
        Matrix3x3::from_affine_translation(&transform.distance)
    }
}

impl<S> AffineTransformation2<S> for Translation2<S> 
    where S: ScalarSigned 
{
    #[inline]
    fn identity() -> Translation2<S> {
        Translation2 { 
            distance: Vector2::zero(),
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
        Transform2::to_transform2d(self)
    }
}


/// A translation transformation in three dimensions.
///
/// A translation is an operation that creates displacement motions. 
/// In a Euclidean setting, translations preserve differences between two points 
/// and acts as the identity on vectors.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Translation3<S> {
    distance: Vector3<S>,
}

impl<S> Translation3<S> where S: ScalarSigned {
    /// Construct a translation operator from a vector of displacements.
    pub fn from_vector(distance: &Vector3<S>) -> Translation3<S> {
        Translation3 {
            distance: distance.clone(),
        }
    }

    /// Construct a translation between two vectors.
    #[inline]
    pub fn translation_between_vectors(vector1: &Vector3<S>, vector2: &Vector3<S>) -> Self {
        let distance = vector2 - vector1;

        Translation3::from_vector(&distance)
    }

    /// Construct a translation between two points.
    #[inline]
    pub fn translation_between_points(point1: &Point3<S>, point2: &Point3<S>) -> Self {
        let distance = point2 - point1;

        Translation3::from_vector(&distance)
    }

    /// Construct a translation that translates a vector or point in the opposite
    /// direction of the translation applied by `self`.
    ///
    /// If `self` is a translation of a vector by a displacement `distance`, then its
    /// inverse will be a translation by a displacement `-distance`.
    #[inline]
    pub fn inverse(&self) -> Self {
        Translation3::from_vector(&(-self.distance))
    }
    
    #[inline]
    pub fn translate_point(&self, point: &Point3<S>) -> Point3<S> {
        point + self.distance
    }

    #[inline]
    pub fn translate_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        *vector
    }
}

impl<S> AsRef<Vector3<S>> for Translation3<S> {
    #[inline]
    fn as_ref(&self) -> &Vector3<S> {
        &self.distance
    }
}

impl<S> fmt::Display for Translation3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Translation3 [x-{}, y={}, z={}]", 
            self.distance.x, self.distance.y, self.distance.z
        )
    }
}

impl<S> From<Translation3<S>> for Matrix4x4<S> where S: ScalarSigned {
    fn from(transform: Translation3<S>) -> Matrix4x4<S> {
        Matrix4x4::from_affine_translation(&transform.distance)
    }
}

impl<S> From<&Translation3<S>> for Matrix4x4<S> where S: ScalarSigned {
    fn from(transform: &Translation3<S>) -> Matrix4x4<S> {
        Matrix4x4::from_affine_translation(&transform.distance)
    }
}

impl<S> AffineTransformation3<S> for Translation3<S> 
    where S: ScalarSigned 
{
    #[inline]
    fn identity() -> Translation3<S> {
        Translation3 { 
            distance: Vector3::zero(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation3<S>> {
        Some(self.inverse())
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.translate_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        self.translate_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::to_transform3d(self)
    }
}

