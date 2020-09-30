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


/// A shearing transformation in two dimensions.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Shear2<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix3x3<S>,
}

impl<S> Shear2<S> where S: ScalarSigned {
    /// Construct a shearing transformation from a vector of shearing factors.
    #[inline]
    pub fn from_vector(shear: Vector2<S>) -> Shear2<S> {
        Shear2 {
            matrix: Matrix3x3::from_affine_shear(shear.x, shear.y),
        }
    }

    /// Construct a shearing transformation along the _x-axis_, holding the 
    /// _y-axis_ constant.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the _y-axis_ to shearing along the _x-axis_.
    #[inline]
    pub fn from_shear_x(shear_x_with_y: S) -> Shear2<S> {
        Shear2 {
            matrix: Matrix3x3::from_affine_shear_x(shear_x_with_y),
        }
    }

    /// Construct a shearing transformation along the _y-axis_, holding the 
    /// _x-axis_ constant.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the _x-axis_ to shearing along the _y-axis_.
    #[inline]
    pub fn from_shear_y(shear_y_with_x: S) -> Shear2<S> {
        Shear2 {
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
    pub fn from_shear(shear_x_with_y: S, shear_y_with_x: S) -> Shear2<S> {
        Shear2 {
            matrix: Matrix3x3::from_affine_shear(shear_x_with_y, shear_y_with_x),
        }
    }

    /// Compute the inverse shearing operation.
    pub fn inverse(&self) -> Shear2<S> {
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_x_with_y = -self.matrix.c1r0;
        let matrix = Matrix3x3::from_affine_shear(shear_x_with_y, shear_y_with_x);
        
        Shear2 {
            matrix: matrix,
        }
    }

    /// Apply a shearing transformation to a vector.
    pub fn shear_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Apply a shearing transformation to a point.
    pub fn shear_point(&self, point: &Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> AsRef<Matrix3x3<S>> for Shear2<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Shear2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Shear2 [{}]", self.matrix)
    }
}

impl<S> From<Shear2<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: Shear2<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> From<&Shear2<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: &Shear2<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation2<S> for Shear2<S> 
    where S: ScalarSigned 
{
    #[inline]
    fn identity() -> Shear2<S> {
        Shear2 { 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear2<S>> {
        Some(self.inverse())
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.shear_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        self.shear_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2<S> {
        Transform2::matrix_to_transform2d(self.matrix)
    }
}


/// A shearing transformation in three dimensions.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Shear3<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix4x4<S>,
}

impl<S> Shear3<S> where S: ScalarSigned {
    /// Construct a shearing transformation along the _x-axis_.
    ///
    /// The parameters `shear_x_with_y` and `shear_x_with_z` denote the 
    /// multiplicative factors for the contributions from the _y-axis_ and the 
    /// _z-axis_ respectively for the shearing along the _x-axis_.
    #[inline]
    pub fn from_shear_x(shear_x_with_y: S, shear_x_with_z: S) -> Shear3<S> {
        Shear3 {
            matrix: Matrix4x4::from_affine_shear_x(shear_x_with_y, shear_x_with_z),
        }
    }

    /// Construct a shearing transformation along the _y-axis_.
    ///
    /// The parameters `shear_y_with_x` and `shear_y_with_z` denote the 
    /// multiplicative factors for the contributions from the _x-axis_ and the 
    /// _z-axis_ respectively for the shearing along the _y-axis_.
    #[inline]
    pub fn from_shear_y(shear_y_with_x: S, shear_y_with_z: S) -> Shear3<S> {
        Shear3 {
            matrix: Matrix4x4::from_affine_shear_y(shear_y_with_x, shear_y_with_z),
        }
    }

    /// Construct a shearing transformation along the _z-axis_.
    ///
    /// The parameters `shear_z_with_x` and `shear_z_with_y` denote the 
    /// multiplicative factors for the contributions from the _x-axis_ and the 
    /// _y-axis_ respectively for the shearing along the _z-axis_.
    #[inline]
    pub fn from_shear_z(shear_z_with_x: S, shear_z_with_y: S) -> Shear3<S> {
        Shear3 {
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
        shear_z_with_x: S, shear_z_with_y: S) -> Shear3<S>
    {
        Shear3 {
            matrix: Matrix4x4::from_affine_shear(
                shear_x_with_y, shear_x_with_z, 
                shear_y_with_x, shear_y_with_z, 
                shear_z_with_x, shear_z_with_y
            )
        }
    }

    /// Apply a shearing transformation to a vector.
    pub fn inverse(&self) -> Shear3<S> {
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
        
        Shear3 {
            matrix: matrix,
        }
    }

    /// Apply a shearing transformation to a vector.
    pub fn shear_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Apply a shearing transformation to a point.
    pub fn shear_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> AsRef<Matrix4x4<S>> for Shear3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Shear3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Shear3 [{}]", self.matrix)
    }
}

impl<S> From<Shear3<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: Shear3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> From<&Shear3<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: &Shear3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation3<S> for Shear3<S>
    where S: ScalarSigned
{
    #[inline]
    fn identity() -> Shear3<S> {
        Shear3 { 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear3<S>> {
        Some(self.inverse())
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.shear_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        self.shear_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::matrix_to_transform3d(self.matrix)
    }
}

