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
use crate::transform::*;

use std::fmt;


/// A trait defining the operations on a shearing transformation.
pub trait Shear<P, V> where Self: Sized + Copy {
    /// The type of the output points (locations in space).
    type OutPoint;
    /// The type of the output vectors (displacements in space).
    type OutVector;

    /// Compute the inverse shearing operation.
    fn inverse(&self) -> Option<Self>;

    /// Apply a shearing transformation to a vector.
    fn shear_vector(&self, vector: V) -> Self::OutVector;

    /// Apply a shearing transformation to a point.
    fn shear_point(&self, point: P) -> Self::OutPoint;
}

/// A trait implementing shearing transformations in two dimensions.
pub trait Shear2<S> where 
    S: ScalarSigned,
    Self: Shear<Point2<S>, Vector2<S>> + Into<Matrix3x3<S>> + Into<Shear2D<S>>,
{
    /// Construct a general shearing transformations in two dimensions. There are 
    /// two possible parameters describing a shearing transformation in two dimensions.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the x-dimension to shearing along the y-dimension.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the contribution 
    /// of the y-dimension to the shearing along the x-dimension.
    fn from_shear(shear_x_with_y: S, shear_y_with_x: S) -> Self;

    /// Construct a shearing transformation along the x-axis, holding the y-axis constant.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the y-dimension to shearing along the x-dimension.
    fn from_shear_x(shear_x_with_y: S) -> Self;

    /// Construct a shearing transformation along the y-axis, holding the x-axis constant.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the x-dimension to shearing along the y-dimension.
    fn from_shear_y(shear_y_with_x: S) -> Self;
}

/// A trait for types implementing translation operators in three dimensions.
pub trait Shear3<S> where 
    S: ScalarSigned,
    Self: Shear<Point3<S>, Vector3<S>> + Into<Matrix4x4<S>> + Into<Shear3D<S>>,
{
    /// Construct a general shearing transformation.
    ///
    /// The parameters `shear_x_with_y` and `shear_x_with_z` denote the multiplicative
    /// factors for the contributions from the y-axis and the z-axis respectively for the
    /// shearing along the x-axis.
    /// 
    /// The parameters `shear_y_with_x` and `shear_y_with_z` denote the multiplicative
    /// factors for the contributions from the x-axis and the z-axis respectively for the
    /// shearing along the y-axis.
    ///
    /// The parameters `shear_z_with_x` and `shear_z_with_y` denote the multiplicative
    /// factors for the contributions from the x-axis and the y-axis respectively for the
    /// shearing along the z-axis.
    fn from_shear(
        shear_x_with_y: S, shear_x_with_z: S, 
        shear_y_with_x: S, shear_y_with_z: S, 
        shear_z_with_x: S, shear_z_with_y: S) -> Self;

    /// Construct a shearing transformation along the x-axis.
    ///
    /// The parameters `shear_x_with_y` and `shear_x_with_z` denote the multiplicative
    /// factors for the contributions from the y-axis and the z-axis respectively for the
    /// shearing along the x-axis.
    fn from_shear_x(shear_x_with_y: S, shear_x_with_z: S) -> Self;

    /// Construct a shearing transformation along the y-axis.
    ///
    /// The parameters `shear_y_with_x` and `shear_y_with_z` denote the multiplicative
    /// factors for the contributions from the x-axis and the z-axis respectively for the
    /// shearing along the y-axis.
    fn from_shear_y(shear_y_with_x: S, shear_y_with_z: S) -> Self;

    /// Construct a shearing transformation along the z-axis.
    ///
    /// The parameters `shear_z_with_x` and `shear_z_with_y` denote the multiplicative
    /// factors for the contributions from the x-axis and the y-axis respectively for the
    /// shearing along the z-axis.
    fn from_shear_z(shear_z_with_x: S, shear_z_with_y: S) -> Self;
}


/// A shearing transformation in two dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Shear2D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix3x3<S>,
}

impl<S> Shear2D<S> where S: Scalar {
    /// Construct a shearing transformation from a vector of shearing factors.
    #[inline]
    pub fn from_vector(shear: Vector2<S>) -> Shear2D<S> {
        Shear2D {
            matrix: Matrix3x3::from_affine_shear(shear.x, shear.y),
        }
    }
}

impl<S> AsRef<Matrix3x3<S>> for Shear2D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Shear2D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Shear2D<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: Shear2D<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> From<&Shear2D<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: &Shear2D<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> Shear<Point2<S>, Vector2<S>> for Shear2D<S> where S: ScalarSigned {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    fn inverse(&self) -> Option<Shear2D<S>> {
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_x_with_y = -self.matrix.c1r0;
        let matrix = Matrix3x3::from_affine_shear(shear_x_with_y, shear_y_with_x);
        
        Some(Shear2D {
            matrix: matrix,
        })
    }

    fn shear_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    fn shear_point(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> Shear2<S> for Shear2D<S> where S: ScalarSigned {
    #[inline]
    fn from_shear_x(shear_x_with_y: S) -> Shear2D<S> {
        Shear2D {
            matrix: Matrix3x3::from_affine_shear_x(shear_x_with_y),
        }
    }

    #[inline]
    fn from_shear_y(shear_y_with_x: S) -> Shear2D<S> {
        Shear2D {
            matrix: Matrix3x3::from_affine_shear_y(shear_y_with_x),
        }
    }

    #[inline]
    fn from_shear(shear_x_with_y: S, shear_y_with_x: S) -> Shear2D<S> {
        Shear2D {
            matrix: Matrix3x3::from_affine_shear(shear_x_with_y, shear_y_with_x),
        }
    }
}

impl<S> AffineTransformation2D<Point2<S>, Vector2<S>, S> for Shear2D<S> where S: ScalarSigned {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Shear2D<S> {
        Shear2D { 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear2D<S>> {
        <Self as Shear<Point2<S>, Vector2<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        self.shear_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: Point2<S>) -> Point2<S> {
        self.shear_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2D<Point2<S>, &Vector2<S>, S> for Shear2D<S> where S: ScalarSigned {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Shear2D<S> {
        Shear2D { 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear2D<S>> {
        <Self as Shear<Point2<S>, Vector2<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.shear_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: Point2<S>) -> Point2<S> {
        self.shear_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2D<&Point2<S>, Vector2<S>, S> for Shear2D<S> where S: ScalarSigned {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Shear2D<S> {
        Shear2D { 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear2D<S>> {
        <Self as Shear<Point2<S>, Vector2<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        self.shear_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        self.shear_point(*point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation2D<&'a Point2<S>, &'b Vector2<S>, S> for Shear2D<S> where S: ScalarSigned {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Shear2D<S> {
        Shear2D { 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear2D<S>> {
        <Self as Shear<Point2<S>, Vector2<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: &'b Vector2<S>) -> Vector2<S> {
        self.shear_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: &'a Point2<S>) -> Point2<S> {
        self.shear_point(*point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}


/// A shearing transformation in three dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Shear3D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix4x4<S>,
}

impl<S> Shear<Point3<S>, Vector3<S>> for Shear3D<S> where S: ScalarSigned {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    fn inverse(&self) -> Option<Shear3D<S>> {
        let shear_x_with_y = -self.matrix.c1r0;
        let shear_x_with_z = -self.matrix.c2r0;
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_y_with_z = -self.matrix.c2r1;
        let shear_z_with_x = -self.matrix.c0r2;
        let shear_z_with_y = -self.matrix.c1r2;
        let matrix = Matrix4x4::from_affine_shear(
            shear_x_with_y, shear_x_with_z, 
            shear_y_with_x, shear_y_with_z, 
            shear_z_with_x, shear_z_with_y
        );
        
        Some(Shear3D {
            matrix: matrix,
        })
    }

    fn shear_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    fn shear_point(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> Shear3<S> for Shear3D<S> where S: ScalarSigned {
    #[inline]
    fn from_shear_x(shear_x_with_y: S, shear_x_with_z: S) -> Shear3D<S> {
        Shear3D {
            matrix: Matrix4x4::from_affine_shear_x(shear_x_with_y, shear_x_with_z),
        }
    }

    #[inline]
    fn from_shear_y(shear_y_with_x: S, shear_y_with_z: S) -> Shear3D<S> {
        Shear3D {
            matrix: Matrix4x4::from_affine_shear_y(shear_y_with_x, shear_y_with_z),
        }
    }

    #[inline]
    fn from_shear_z(shear_z_with_x: S, shear_z_with_y: S) -> Shear3D<S> {
        Shear3D {
            matrix: Matrix4x4::from_affine_shear_z(shear_z_with_x, shear_z_with_y),
        }
    }

    #[inline]
    fn from_shear(
        shear_x_with_y: S, shear_x_with_z: S, 
        shear_y_with_x: S, shear_y_with_z: S, 
        shear_z_with_x: S, shear_z_with_y: S) -> Shear3D<S>
    {
        Shear3D {
            matrix: Matrix4x4::from_affine_shear(
                shear_x_with_y, shear_x_with_z, 
                shear_y_with_x, shear_y_with_z, 
                shear_z_with_x, shear_z_with_y
            )
        }
    }
}

impl<S> AsRef<Matrix4x4<S>> for Shear3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Shear3D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Shear3D<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: Shear3D<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> From<&Shear3D<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: &Shear3D<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation3D<Point3<S>, Vector3<S>, S> for Shear3D<S> where S: ScalarSigned {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Shear3D<S> {
        Shear3D { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear3D<S>> {
        <Self as Shear<Point3<S>, Vector3<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        self.shear_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: Point3<S>) -> Point3<S> {
        self.shear_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3D<Point3<S>, &Vector3<S>, S> for Shear3D<S> where S: ScalarSigned {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Shear3D<S> {
        Shear3D { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear3D<S>> {
        <Self as Shear<Point3<S>, Vector3<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.shear_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: Point3<S>) -> Point3<S> {
        self.shear_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3D<&Point3<S>, Vector3<S>, S> for Shear3D<S> where S: ScalarSigned {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Shear3D<S> {
        Shear3D { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear3D<S>> {
        <Self as Shear<Point3<S>, Vector3<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        self.shear_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        self.shear_point(*point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation3D<&'a Point3<S>, &'b Vector3<S>, S> for Shear3D<S> where S: ScalarSigned {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Shear3D<S> {
        Shear3D { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear3D<S>> {
        <Self as Shear<Point3<S>, Vector3<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: &'b Vector3<S>) -> Vector3<S> {
        self.shear_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: &'a Point3<S>) -> Point3<S> {
        self.shear_point(*point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

