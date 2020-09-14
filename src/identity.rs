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
};
use affine::*;

use std::fmt;


/// The identity affine transformation in two dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Identity2D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix3<S>,
}

impl<S> Identity2D<S> where S: Scalar {
    /// Construct a new identity transformation.
    #[inline]
    pub fn identity() -> Identity2D<S> {
        Identity2D {
            matrix: Matrix3::one(),
        }
    }
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

impl<S> AffineTransformation2D<Point2<S>, Vector2<S>> for Identity2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Identity2D<S> {
        Identity2D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity2D<S>> {
        Some(Identity2D::identity())
    }

    #[inline]
    fn apply_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        vector
    }

    #[inline]
    fn apply_point(&self, point: Point2<S>) -> Point2<S> {
        point
    }
}

impl<S> AffineTransformation2D<Point2<S>, &Vector2<S>> for Identity2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Identity2D<S> {
        Identity2D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity2D<S>> {
        Some(Identity2D::identity())
    }

    #[inline]
    fn apply_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        *vector
    }

    #[inline]
    fn apply_point(&self, point: Point2<S>) -> Point2<S> {
        point
    }
}

impl<S> AffineTransformation2D<&Point2<S>, Vector2<S>> for Identity2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Identity2D<S> {
        Identity2D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity2D<S>> {
        Some(Identity2D::identity())
    }

    #[inline]
    fn apply_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        vector
    }

    #[inline]
    fn apply_point(&self, point: &Point2<S>) -> Point2<S> {
        *point
    }
}

impl<'a, 'b, S> AffineTransformation2D<&'a Point2<S>, &'b Vector2<S>> for Identity2D<S> where S: Scalar {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Identity2D<S> {
        Identity2D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity2D<S>> {
        Some(Identity2D::identity())
    }

    #[inline]
    fn apply_vector(&self, vector: &'b Vector2<S>) -> Vector2<S> {
        *vector
    }

    #[inline]
    fn apply_point(&self, point: &'a Point2<S>) -> Point2<S> {
        *point
    }
}


/// The identity transformation in three dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Identity3D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix4<S>,
}

impl<S> Identity3D<S> where S: Scalar {
    /// Construct a new identity transformation.
    #[inline]
    pub fn identity() -> Identity3D<S> {
        Identity3D {
            matrix: Matrix4::one(),
        }
    }
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

impl<S> AffineTransformation3D<Point3<S>, Vector3<S>> for Identity3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Identity3D<S> {
        Identity3D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity3D<S>> {
        Some(Identity3D::identity())
    }

    #[inline]
    fn apply_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        vector
    }

    #[inline]
    fn apply_point(&self, point: Point3<S>) -> Point3<S> {
        point
    }
}

impl<S> AffineTransformation3D<Point3<S>, &Vector3<S>> for Identity3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Identity3D<S> {
        Identity3D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity3D<S>> {
        Some(Identity3D::identity())
    }

    #[inline]
    fn apply_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        *vector
    }

    #[inline]
    fn apply_point(&self, point: Point3<S>) -> Point3<S> {
        point
    }
}

impl<S> AffineTransformation3D<&Point3<S>, Vector3<S>> for Identity3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Identity3D<S> {
        Identity3D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity3D<S>> {
        Some(Identity3D::identity())
    }

    #[inline]
    fn apply_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        vector
    }

    #[inline]
    fn apply_point(&self, point: &Point3<S>) -> Point3<S> {
        *point
    }
}

impl<'a, 'b, S> AffineTransformation3D<&'a Point3<S>, &'b Vector3<S>> for Identity3D<S> where S: Scalar {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Identity3D<S> {
        Identity3D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity3D<S>> {
        Some(Identity3D::identity())
    }

    #[inline]
    fn apply_vector(&self, vector: &'b Vector3<S>) -> Vector3<S> {
        *vector
    }

    #[inline]
    fn apply_point(&self, point: &'a Point3<S>) -> Point3<S> {
        *point
    }
}
