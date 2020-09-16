use scalar::{
    Scalar,
};
use matrix::{
    Matrix3,
    Matrix4,
};
use vector::{
    Vector2,
    Vector3,
};
use point::{
    Point2,
    Point3,
};
use structure::{
    One,
    Euclidean,
};
use transform::*;

use std::fmt;


/// A trait defining the identity transformation.
pub trait Identity<P, V> where Self: Sized + Copy {
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
pub trait Identity2<S> where 
    S: Scalar,
    Self: Identity<Point2<S>, Vector2<S>> + Into<Matrix3<S>> + Into<Identity2D<S>>,
{
}

/// A trait defining the identity transformation in three dimensions.
pub trait Identity3<S> where 
    S: Scalar,
    Self: Identity<Point3<S>, Vector3<S>> + Into<Matrix4<S>> + Into<Identity3D<S>>,
{
}


/// The identity affine transformation in two dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Identity2D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix3<S>,
}

impl<S> AsRef<Matrix3<S>> for Identity2D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Identity2D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Identity2D<S>> for Matrix3<S> where S: Copy {
    fn from(transformation: Identity2D<S>) -> Matrix3<S> {
        transformation.matrix
    }
}

impl<S> From<&Identity2D<S>> for Matrix3<S> where S: Copy {
    fn from(transformation: &Identity2D<S>) -> Matrix3<S> {
        transformation.matrix
    }
}

impl<S> Identity<Point2<S>, Vector2<S>> for Identity2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Identity2D<S> {
        Identity2D {
            matrix: Matrix3::one(),
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

impl<S> Identity2<S> for Identity2D<S> where S: Scalar {}

impl<S> AffineTransformation2D<Point2<S>, Vector2<S>, S> for Identity2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Identity2D<S> {
        <Self as Identity<Point2<S>, Vector2<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity2D<S>> {
        <Self as Identity<Point2<S>, Vector2<S>>>::inverse(&self)
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

impl<S> AffineTransformation2D<Point2<S>, &Vector2<S>, S> for Identity2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Identity2D<S> {
        <Self as Identity<Point2<S>, Vector2<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity2D<S>> {
        <Self as Identity<Point2<S>, Vector2<S>>>::inverse(&self)
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

impl<S> AffineTransformation2D<&Point2<S>, Vector2<S>, S> for Identity2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Identity2D<S> {
        <Self as Identity<Point2<S>, Vector2<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity2D<S>> {
        <Self as Identity<Point2<S>, Vector2<S>>>::inverse(&self)
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

impl<'a, 'b, S> AffineTransformation2D<&'a Point2<S>, &'b Vector2<S>, S> for Identity2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Identity2D<S> {
        <Self as Identity<Point2<S>, Vector2<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity2D<S>> {
        <Self as Identity<Point2<S>, Vector2<S>>>::inverse(&self)
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
pub struct Identity3D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix4<S>,
}

impl<S> AsRef<Matrix4<S>> for Identity3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Identity3D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Identity3D<S>> for Matrix4<S> where S: Copy {
    fn from(transformation: Identity3D<S>) -> Matrix4<S> {
        transformation.matrix
    }
}

impl<S> From<&Identity3D<S>> for Matrix4<S> where S: Copy {
    fn from(transformation: &Identity3D<S>) -> Matrix4<S> {
        transformation.matrix
    }
}

impl<S> Identity<Point3<S>, Vector3<S>> for Identity3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;
    
    #[inline]
    fn identity() -> Identity3D<S> {
        Identity3D {
            matrix: Matrix4::one(),
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

impl<S> Identity3<S> for Identity3D<S> where S: Scalar {}

impl<S> AffineTransformation3D<Point3<S>, Vector3<S>, S> for Identity3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Identity3D<S> {
        <Self as Identity<Point3<S>, Vector3<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity3D<S>> {
        <Self as Identity<Point3<S>, Vector3<S>>>::inverse(&self)
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

impl<S> AffineTransformation3D<Point3<S>, &Vector3<S>, S> for Identity3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Identity3D<S> {
        <Self as Identity<Point3<S>, Vector3<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity3D<S>> {
        <Self as Identity<Point3<S>, Vector3<S>>>::inverse(&self)
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

impl<S> AffineTransformation3D<&Point3<S>, Vector3<S>, S> for Identity3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Identity3D<S> {
        <Self as Identity<Point3<S>, Vector3<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity3D<S>> {
        <Self as Identity<Point3<S>, Vector3<S>>>::inverse(&self)
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

impl<'a, 'b, S> AffineTransformation3D<&'a Point3<S>, &'b Vector3<S>, S> for Identity3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Identity3D<S> {
        <Self as Identity<Point3<S>, Vector3<S>>>::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity3D<S>> {
        <Self as Identity<Point3<S>, Vector3<S>>>::inverse(&self)
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

