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
use crate::transformation::*;

use core::fmt;


/// A trait for implementing scaling transformations.
pub trait Scale<P, V> where Self: Sized + Copy {
    /// The type of the output points (locations in space).
    type OutPoint;
    /// The type of the output vectors (displacements in space).
    type OutVector;

    /// Construct a scale transformation that scales each coordinate by the reciprocal 
    /// of the scaling factors of the scale operator `self`.
    fn inverse(&self) -> Option<Self>;

    /// Apply a scale operation to a vector.
    fn scale_vector(&self, vector: V) -> Self::OutVector;

    /// Apply a scale operation to a point.
    fn scale_point(&self, point: P) -> Self::OutPoint;
}

/// A trait defining scaling transformations in two dimensions.
pub trait Scale2<S> where 
    S: ScalarFloat,
    Self: Scale<Point2<S>, Vector2<S>> + Into<Matrix3x3<S>> + Into<Scale2D<S>>,
{
    /// Construct a two-dimensional scale transformation from a nonuniform scale 
    /// across coordinates.
    fn from_nonuniform_scale(scale_x: S, scale_y: S) -> Self;
  
    /// Construct a two-dimensional scale transformation from a uniform scale factor.
    fn from_scale(scale: S) -> Self;
}

/// A trait defining scaling transformations in three dimensions.
pub trait Scale3<S> where 
    S: ScalarFloat,
    Self: Scale<Point3<S>, Vector3<S>> + Into<Matrix4x4<S>> + Into<Scale3D<S>>,
{
    /// Construct a three-dimensional scale transformation from a nonuniform scale 
    /// across coordinates.
    fn from_nonuniform_scale(scale_x: S, scale_y: S, scale_z: S) -> Self;

    /// Construct a three-dimensional scale transformation from a uniform scale factor.
    fn from_scale(scale: S) -> Self;
}


/// The scale transformation in two dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Scale2D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix3x3<S>,
}

impl<S> Scale2D<S> where S: Scalar {
    /// Construct a scale transformation from a vector of scale factors.
    #[inline]
    pub fn from_vector(scale: Vector2<S>) -> Scale2D<S> {
        Scale2D {
            matrix: Matrix3x3::from_affine_nonuniform_scale(scale.x, scale.y),
        }
    }
}

impl<S> AsRef<Matrix3x3<S>> for Scale2D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Scale2D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Scale2D<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: Scale2D<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> From<&Scale2D<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: &Scale2D<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> Scale<Point2<S>, Vector2<S>> for Scale2D<S> where S: ScalarFloat {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn inverse(&self) -> Option<Scale2D<S>> {
        Some(Scale2D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1
        ))
    }

    #[inline]
    fn scale_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn scale_point(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> Scale2<S> for Scale2D<S> where S: ScalarFloat {
    #[inline]
    fn from_nonuniform_scale(scale_x: S, scale_y: S) -> Scale2D<S> {
        Scale2D {
            matrix: Matrix3x3::from_affine_nonuniform_scale(scale_x, scale_y),
        }
    }

    #[inline]
    fn from_scale(scale: S) -> Scale2D<S> {
        Scale2D {
            matrix: Matrix3x3::from_scale(scale),
        }
    }
}

impl<S> AffineTransformation2D<Point2<S>, Vector2<S>, S> for Scale2D<S> where S: ScalarFloat {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Scale2D<S> {
        Scale2D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2D<S>> {
        <Self as Scale<Point2<S>, Vector2<S>>>::inverse(&self)
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
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2D<Point2<S>, &Vector2<S>, S> for Scale2D<S> where S: ScalarFloat {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Scale2D<S> {
        Scale2D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2D<S>> {
        <Self as Scale<Point2<S>, Vector2<S>>>::inverse(&self)
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
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2D<&Point2<S>, Vector2<S>, S> for Scale2D<S> where S: ScalarFloat {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Scale2D<S> {
        Scale2D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2D<S>> {
        <Self as Scale<Point2<S>, Vector2<S>>>::inverse(&self)
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
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation2D<&'a Point2<S>, &'b Vector2<S>, S> for Scale2D<S> where S: ScalarFloat {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Scale2D<S> {
        Scale2D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2D<S>> {
        <Self as Scale<Point2<S>, Vector2<S>>>::inverse(&self)
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
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}


/// The scale transformation in three dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Scale3D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix4x4<S>,
}

impl<S> Scale3D<S> where S: Scalar {
    /// Construct a scale transformation from a vector of scale factors.
    pub fn from_vector(scale: Vector3<S>) -> Scale3D<S> {
        Scale3D {
            matrix: Matrix4x4::from_affine_nonuniform_scale(scale.x, scale.y, scale.z),
        }
    }
}

impl<S> AsRef<Matrix4x4<S>> for Scale3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Scale3D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Scale3D<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: Scale3D<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> From<&Scale3D<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: &Scale3D<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> Scale<Point3<S>, Vector3<S>> for Scale3D<S> where S: ScalarFloat {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn inverse(&self) -> Option<Scale3D<S>> {
        Some(Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1,
            S::one() / self.matrix.c2r2
        ))
    }

    #[inline]
    fn scale_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn scale_point(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> Scale3<S> for Scale3D<S> where S: ScalarFloat {
    #[inline]
    fn from_nonuniform_scale(scale_x: S, scale_y: S, scale_z: S) -> Scale3D<S> {
        Scale3D {
            matrix: Matrix4x4::from_affine_nonuniform_scale(scale_x, scale_y, scale_z),
        }
    }

    #[inline]
    fn from_scale(scale: S) -> Scale3D<S> {
        Scale3D {
            matrix: Matrix4x4::from_affine_scale(scale),
        }
    }
}

impl<S> AffineTransformation3D<Point3<S>, Vector3<S>, S> for Scale3D<S> where S: ScalarFloat {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Scale3D<S> {
        Scale3D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3D<S>> {
        <Self as Scale<Point3<S>, Vector3<S>>>::inverse(&self)
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
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3D<Point3<S>, &Vector3<S>, S> for Scale3D<S> where S: ScalarFloat {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Scale3D<S> {
        Scale3D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3D<S>> {
        <Self as Scale<Point3<S>, Vector3<S>>>::inverse(&self)
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
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3D<&Point3<S>, Vector3<S>, S> for Scale3D<S> where S: ScalarFloat {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Scale3D<S> {
        Scale3D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3D<S>> {
        <Self as Scale<Point3<S>, Vector3<S>>>::inverse(&self)
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
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation3D<&'a Point3<S>, &'b Vector3<S>, S> for Scale3D<S> where S: ScalarFloat {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Scale3D<S> {
        Scale3D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3D<S>> {
        <Self as Scale<Point3<S>, Vector3<S>>>::inverse(&self)
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
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}
