use scalar::{
    Scalar,
};
use matrix::{
    Matrix3x3,
    Matrix4x4,
};
use vector::{
    Vector2,
    Vector3,
};
use point::{
    Point2,
    Point3,
};
use traits::{
    Identity,
};
use transform::*;

use std::fmt;


/// A trait defining the identity transformation.
pub trait IdentityTransformation<P, V> where Self: Sized + Copy {
    type OutPoint;
    type OutVector;

    /// Construct a new identity transformation.
    fn identity() -> Self;

    /// Compute the inverse of an identity map. This is 
    /// also just the identity map.
    fn inverse(&self) -> Option<Self>;

    /// Apply the identity transformation to a vector.
    fn identify_vector(&self, vector: V) -> Self::OutVector;

    /// Apply the identity transformation to a point.
    fn identify_point(&self, point: P) -> Self::OutPoint;
}

/// A trait defining the identity transformation in two dimensions.
pub trait IdentityTransformation2<S> where 
    S: Scalar,
    Self: IdentityTransformation<Point2<S>, Vector2<S>> + Into<Matrix3x3<S>> + Into<IdentityTranformation2D<S>>,
{
}

/// A trait defining the identity transformation in three dimensions.
pub trait IdentityTransformation3<S> where 
    S: Scalar,
    Self: IdentityTransformation<Point3<S>, Vector3<S>> + Into<Matrix4x4<S>> + Into<IdentityTranformation3D<S>>,
{
}


/// The identity affine transformation in two dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct IdentityTranformation2D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix3x3<S>,
}

impl<S> AsRef<Matrix3x3<S>> for IdentityTranformation2D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for IdentityTranformation2D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<IdentityTranformation2D<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: IdentityTranformation2D<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> From<&IdentityTranformation2D<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: &IdentityTranformation2D<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> IdentityTransformation<Point2<S>, Vector2<S>> for IdentityTranformation2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> IdentityTranformation2D<S> {
        IdentityTranformation2D {
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        Some(*self)
    }

    #[inline]
    fn identify_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        vector
    }

    #[inline]
    fn identify_point(&self, point: Point2<S>) -> Point2<S> {
        point
    }
}

impl<S> IdentityTransformation2<S> for IdentityTranformation2D<S> where S: Scalar {}

impl<S> AffineTransformation2D<Point2<S>, Vector2<S>, S> for IdentityTranformation2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> IdentityTranformation2D<S> {
        <Self as IdentityTransformation<Point2<S>, Vector2<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<IdentityTranformation2D<S>> {
        <Self as IdentityTransformation<Point2<S>, Vector2<S>>>::inverse(&self)
    }

    #[inline]
    fn apply_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        self.identify_vector(vector)
    }

    #[inline]
    fn apply_point(&self, point: Point2<S>) -> Point2<S> {
        self.identify_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2D<Point2<S>, &Vector2<S>, S> for IdentityTranformation2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> IdentityTranformation2D<S> {
        <Self as IdentityTransformation<Point2<S>, Vector2<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<IdentityTranformation2D<S>> {
        <Self as IdentityTransformation<Point2<S>, Vector2<S>>>::inverse(&self)
    }

    #[inline]
    fn apply_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        *vector
    }

    #[inline]
    fn apply_point(&self, point: Point2<S>) -> Point2<S> {
        point
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2D<&Point2<S>, Vector2<S>, S> for IdentityTranformation2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> IdentityTranformation2D<S> {
        <Self as IdentityTransformation<Point2<S>, Vector2<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<IdentityTranformation2D<S>> {
        <Self as IdentityTransformation<Point2<S>, Vector2<S>>>::inverse(&self)
    }

    #[inline]
    fn apply_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        vector
    }

    #[inline]
    fn apply_point(&self, point: &Point2<S>) -> Point2<S> {
        *point
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation2D<&'a Point2<S>, &'b Vector2<S>, S> for IdentityTranformation2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> IdentityTranformation2D<S> {
        <Self as IdentityTransformation<Point2<S>, Vector2<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<IdentityTranformation2D<S>> {
        <Self as IdentityTransformation<Point2<S>, Vector2<S>>>::inverse(&self)
    }

    #[inline]
    fn apply_vector(&self, vector: &'b Vector2<S>) -> Vector2<S> {
        *vector
    }

    #[inline]
    fn apply_point(&self, point: &'a Point2<S>) -> Point2<S> {
        *point
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}


/// The identity transformation in three dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct IdentityTranformation3D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix4x4<S>,
}

impl<S> AsRef<Matrix4x4<S>> for IdentityTranformation3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for IdentityTranformation3D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<IdentityTranformation3D<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: IdentityTranformation3D<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> From<&IdentityTranformation3D<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: &IdentityTranformation3D<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> IdentityTransformation<Point3<S>, Vector3<S>> for IdentityTranformation3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;
    
    #[inline]
    fn identity() -> IdentityTranformation3D<S> {
        IdentityTranformation3D {
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        Some(*self)
    }

    #[inline]
    fn identify_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        vector
    }

    #[inline]
    fn identify_point(&self, point: Point3<S>) -> Point3<S> {
        point
    }
}

impl<S> IdentityTransformation3<S> for IdentityTranformation3D<S> where S: Scalar {}

impl<S> AffineTransformation3D<Point3<S>, Vector3<S>, S> for IdentityTranformation3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> IdentityTranformation3D<S> {
        <Self as IdentityTransformation<Point3<S>, Vector3<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<IdentityTranformation3D<S>> {
        <Self as IdentityTransformation<Point3<S>, Vector3<S>>>::inverse(&self)
    }

    #[inline]
    fn apply_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        self.identify_vector(vector)
    }

    #[inline]
    fn apply_point(&self, point: Point3<S>) -> Point3<S> {
        self.identify_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3D<Point3<S>, &Vector3<S>, S> for IdentityTranformation3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> IdentityTranformation3D<S> {
        <Self as IdentityTransformation<Point3<S>, Vector3<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<IdentityTranformation3D<S>> {
        <Self as IdentityTransformation<Point3<S>, Vector3<S>>>::inverse(&self)
    }

    #[inline]
    fn apply_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.identify_vector(*vector)
    }

    #[inline]
    fn apply_point(&self, point: Point3<S>) -> Point3<S> {
        self.identify_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3D<&Point3<S>, Vector3<S>, S> for IdentityTranformation3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> IdentityTranformation3D<S> {
        <Self as IdentityTransformation<Point3<S>, Vector3<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<IdentityTranformation3D<S>> {
        <Self as IdentityTransformation<Point3<S>, Vector3<S>>>::inverse(&self)
    }

    #[inline]
    fn apply_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        self.identify_vector(vector)
    }

    #[inline]
    fn apply_point(&self, point: &Point3<S>) -> Point3<S> {
        self.identify_point(*point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation3D<&'a Point3<S>, &'b Vector3<S>, S> for IdentityTranformation3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> IdentityTranformation3D<S> {
        <Self as IdentityTransformation<Point3<S>, Vector3<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<IdentityTranformation3D<S>> {
        <Self as IdentityTransformation<Point3<S>, Vector3<S>>>::inverse(&self)
    }

    #[inline]
    fn apply_vector(&self, vector: &'b Vector3<S>) -> Vector3<S> {
        self.identify_vector(*vector)
    }

    #[inline]
    fn apply_point(&self, point: &'a Point3<S>) -> Point3<S> {
        self.identify_point(*point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

