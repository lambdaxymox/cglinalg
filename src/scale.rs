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
use crate::affine::*;

use core::fmt;


/// The scale transformation in two dimensions.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Scale2<S> {
    /// The scale factor for the `x`-component.
    x: S,
    /// The scale factor for the `y`-component.
    y: S,
}

impl<S> Scale2<S> where S: Scalar {
    /// Construct a two-dimensional scale transformation from a nonuniform scale 
    /// across coordinates.
    #[inline]
    pub fn from_nonuniform_scale(scale_x: S, scale_y: S) -> Scale2<S> {
        Scale2 {
            x: scale_x,
            y: scale_y,
        }
    }

    /// Construct a two-dimensional scale transformation from a uniform scale 
    /// factor.
    #[inline]
    pub fn from_scale(scale: S) -> Scale2<S> {
        Scale2 {
            x: scale,
            y: scale,
        }
    }

    /// Construct a scale transformation that scales each coordinate by the 
    /// reciprocal of the scaling factors of the scale operator `self`.
    #[inline]
    pub fn inverse(&self) -> Scale2<S> {
        Scale2::from_nonuniform_scale(
            S::one() / self.x, 
            S::one() / self.y
        )
    }

    /// Apply a scale operation to a vector.
    #[inline]
    pub fn scale_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        Vector2::new(self.x * vector.x, self.y * vector.y)
    }

    /// Apply a scale operation to a point.
    #[inline]
    pub fn scale_point(&self, point: &Point2<S>) -> Point2<S> {
        Point2::new(self.x * point.x, self.y * point.y)
    }
}

impl<S> fmt::Display for Scale2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Scale2 [scale_x={}, scale_y={}]", 
            self.x, self.y
        )
    }
}

impl<S> From<Scale2<S>> for Matrix3x3<S> where S: Scalar {
    fn from(scale: Scale2<S>) -> Matrix3x3<S> {
        Matrix3x3::from_affine_nonuniform_scale(scale.x, scale.y)
    }
}

impl<S> From<&Scale2<S>> for Matrix3x3<S> where S: Scalar {
    fn from(scale: &Scale2<S>) -> Matrix3x3<S> {
        Matrix3x3::from_affine_nonuniform_scale(scale.x, scale.y)
    }
}

impl<S> AffineTransformation2<S> for Scale2<S> 
    where S: ScalarFloat 
{
    #[inline]
    fn identity() -> Scale2<S> {
        Scale2::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2<S>> {
        Some(self.inverse())
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.scale_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        self.scale_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2<S> {
        Transform2::to_transform2d(self)
    }
}


/// The scale transformation in three dimensions.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Scale3<S> {
    /// The scale factor for the `x`-component.
    x: S,
    /// The scale factor for the `y`-component.
    y: S,
    /// The scale factor for the `z`-component.
    z: S,
}

impl<S> Scale3<S> where S: Scalar {
    /// Construct a three-dimensional scale transformation from a nonuniform scale 
    /// across coordinates.
    #[inline]
    pub fn from_nonuniform_scale(scale_x: S, scale_y: S, scale_z: S) -> Scale3<S> {
        Scale3 {
            x: scale_x,
            y: scale_y,
            z: scale_z,
        }
    }

    /// Construct a three-dimensional scale transformation from a uniform scale 
    /// factor.
    #[inline]
    pub fn from_scale(scale: S) -> Scale3<S> {
        Scale3 {
            x: scale,
            y: scale,
            z: scale,
        }
    }

    /// Construct a scale transformation that scales each coordinate by the 
    /// reciprocal of the scaling factors of the scale operator `self`.
    #[inline]
    pub fn inverse(&self) -> Option<Scale3<S>> {
        Some(Scale3::from_nonuniform_scale(
            S::one() / self.x, 
            S::one() / self.y,
            S::one() / self.z
        ))
    }

    /// Apply a scale operation to a vector.
    #[inline]
    pub fn scale_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        Vector3::new(self.x * vector.x, self.y * vector.y, self.z * vector.z)
    }

    /// Apply a scale operation to a point.
    #[inline]
    pub fn scale_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::new(self.x * point.x, self.y * point.y, self.z * point.z)
    }
}

impl<S> fmt::Display for Scale3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Scale3 [scale_x={}, scale_y={}, scale_z={}]", 
            self.x, self.y, self.z
        )
    }
}

impl<S> From<Scale3<S>> for Matrix4x4<S> where S: Scalar {
    fn from(scale: Scale3<S>) -> Matrix4x4<S> {
        Matrix4x4::from_affine_nonuniform_scale(
            scale.x, 
            scale.y, 
            scale.z
        )
    }
}

impl<S> From<&Scale3<S>> for Matrix4x4<S> where S: Scalar {
    fn from(scale: &Scale3<S>) -> Matrix4x4<S> {
        Matrix4x4::from_affine_nonuniform_scale(
            scale.x, 
            scale.y, 
            scale.z
        )
    }
}

impl<S> AffineTransformation3<S> for Scale3<S> 
    where S: ScalarFloat 
{
    #[inline]
    fn identity() -> Scale3<S> {
        Scale3::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3<S>> {
        self.inverse()
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.scale_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        self.scale_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::to_transform3d(self)
    }
}

