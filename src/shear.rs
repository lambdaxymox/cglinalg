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
use crate::affine::*;

use core::fmt;


/// A shearing transformation in two dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct Shear2D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix3x3<S>,
}

impl<S> Shear2D<S> where S: ScalarSigned {
    /// Construct a shearing transformation from a vector of shearing factors.
    #[inline]
    pub fn from_vector(shear: Vector2<S>) -> Shear2D<S> {
        Shear2D {
            matrix: Matrix3x3::from_affine_shear(shear.x, shear.y),
        }
    }

    /// Compute the inverse shearing operation.
    pub fn inverse(&self) -> Option<Shear2D<S>> {
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_x_with_y = -self.matrix.c1r0;
        let matrix = Matrix3x3::from_affine_shear(shear_x_with_y, shear_y_with_x);
        
        Some(Shear2D {
            matrix: matrix,
        })
    }

    /// Apply a shearing transformation to a vector.
    pub fn shear_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Apply a shearing transformation to a point.
    pub fn shear_point(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    /// Construct a shearing transformation along the _x-axis_, holding the 
    /// _y-axis_ constant.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the _y-axis_ to shearing along the _x-axis_.
    #[inline]
    pub fn from_shear_x(shear_x_with_y: S) -> Shear2D<S> {
        Shear2D {
            matrix: Matrix3x3::from_affine_shear_x(shear_x_with_y),
        }
    }

    /// Construct a shearing transformation along the _y-axis_, holding the 
    /// _x-axis_ constant.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the _x-axis_ to shearing along the _y-axis_.
    #[inline]
    pub fn from_shear_y(shear_y_with_x: S) -> Shear2D<S> {
        Shear2D {
            matrix: Matrix3x3::from_affine_shear_y(shear_y_with_x),
        }
    }

    /// Construct a general shearing transformations in two dimensions. 
    ///
    /// There are two parameters describing a shearing transformation 
    /// in two dimensions.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the _x-axis_ to shearing along the _y-axis_.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the 
    /// contribution of the _y-axis_ to the shearing along the _x-axis_.
    #[inline]
    pub fn from_shear(shear_x_with_y: S, shear_y_with_x: S) -> Shear2D<S> {
        Shear2D {
            matrix: Matrix3x3::from_affine_shear(shear_x_with_y, shear_y_with_x),
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

impl<S> AffineTransformation2D<Point2<S>, Vector2<S>, S> for Shear2D<S> 
    where S: ScalarSigned 
{
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
        self.inverse()
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

impl<S> AffineTransformation2D<Point2<S>, &Vector2<S>, S> for Shear2D<S> 
    where S: ScalarSigned 
{
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
        self.inverse()
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

impl<S> AffineTransformation2D<&Point2<S>, Vector2<S>, S> for Shear2D<S> 
    where S: ScalarSigned 
{
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
        self.inverse()
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

impl<'a, 'b, S> AffineTransformation2D<&'a Point2<S>, &'b Vector2<S>, S> for Shear2D<S> 
    where S: ScalarSigned 
{
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
        self.inverse()
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
#[repr(transparent)]
pub struct Shear3D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix4x4<S>,
}

impl<S> Shear3D<S> where S: ScalarSigned {
    /// Apply a shearing transformation to a vector.
    pub fn inverse(&self) -> Option<Shear3D<S>> {
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

     /// Apply a shearing transformation to a vector.
    pub fn shear_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Apply a shearing transformation to a point.
    pub fn shear_point(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    /// Construct a shearing transformation along the _x-axis_.
    ///
    /// The parameters `shear_x_with_y` and `shear_x_with_z` denote the 
    /// multiplicative factors for the contributions from the _y-axis_ and the 
    /// _z-axis_ respectively for the shearing along the _x-axis_.
    #[inline]
    pub fn from_shear_x(shear_x_with_y: S, shear_x_with_z: S) -> Shear3D<S> {
        Shear3D {
            matrix: Matrix4x4::from_affine_shear_x(shear_x_with_y, shear_x_with_z),
        }
    }

    /// Construct a shearing transformation along the _y-axis_.
    ///
    /// The parameters `shear_y_with_x` and `shear_y_with_z` denote the 
    /// multiplicative factors for the contributions from the _x-axis_ and the 
    /// _z-axis_ respectively for the shearing along the _y-axis_.
    #[inline]
    pub fn from_shear_y(shear_y_with_x: S, shear_y_with_z: S) -> Shear3D<S> {
        Shear3D {
            matrix: Matrix4x4::from_affine_shear_y(shear_y_with_x, shear_y_with_z),
        }
    }

    /// Construct a shearing transformation along the _z-axis_.
    ///
    /// The parameters `shear_z_with_x` and `shear_z_with_y` denote the 
    /// multiplicative factors for the contributions from the _x-axis_ and the 
    /// _y-axis_ respectively for the shearing along the _z-axis_.
    #[inline]
    pub fn from_shear_z(shear_z_with_x: S, shear_z_with_y: S) -> Shear3D<S> {
        Shear3D {
            matrix: Matrix4x4::from_affine_shear_z(shear_z_with_x, shear_z_with_y),
        }
    }

    /// Construct a general shearing transformation.
    ///
    /// The parameters `shear_x_with_y` and `shear_x_with_z` denote the 
    /// multiplicative factors for the contributions from the _y-axis_ and the 
    /// _z-axis_ respectively for the shearing along the _x-axis_.
    /// 
    /// The parameters `shear_y_with_x` and `shear_y_with_z` denote the 
    /// multiplicative factors for the contributions from the _x-axis_ and the 
    /// _z-axis_ respectively for the shearing along the _y-axis_.
    ///
    /// The parameters `shear_z_with_x` and `shear_z_with_y` denote the 
    /// multiplicative factors for the contributions from the _x-axis_ and the 
    /// _y-axis_ respectively for the shearing along the _z-axis_.
    #[inline]
    pub fn from_shear(
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

impl<S> AffineTransformation3D<Point3<S>, Vector3<S>, S> for Shear3D<S>
    where 
        S: ScalarSigned
{
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
        self.inverse()
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

impl<S> AffineTransformation3D<Point3<S>, &Vector3<S>, S> for Shear3D<S> 
    where 
        S: ScalarSigned
{
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
        self.inverse()
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

impl<S> AffineTransformation3D<&Point3<S>, Vector3<S>, S> for Shear3D<S>
    where 
        S: ScalarSigned 
{
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
        self.inverse()
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

impl<'a, 'b, S> AffineTransformation3D<&'a Point3<S>, &'b Vector3<S>, S> for Shear3D<S> 
    where 
        S: ScalarSigned
{
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
        self.inverse()
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

