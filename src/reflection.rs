use crate::scalar::{
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
use crate::traits::{
    Identity,
    Zero,
};
use crate::unit::{
    Unit,
};
use crate::transform::*;

use core::fmt;


/// A reflection transformation about a plane (line) in two dimensions.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reflection2<S> {
    /// A known point on the line of reflection.
    bias: Vector2<S>,
    /// The normal vector to the plane.
    normal: Vector2<S>,
    /// The matrix representing the affine transformation.
    matrix: Matrix3x3<S>,
}

impl<S> Reflection2<S> where S: ScalarFloat {
    /// Construct a new reflection transformation from the vector normal to the 
    /// plane of reflection.
    pub fn from_normal_bias(normal: &Unit<Vector2<S>>, bias: &Vector2<S>) -> Reflection2<S> {
        Reflection2 {
            bias: *bias,
            normal: normal.into_inner(),
            matrix: Matrix3x3::from_affine_reflection(normal, bias),
        }
    }

    /// Return the bias for calculating the reflections.
    ///
    /// The `bias` is the coordinates of a known point in the plane of 
    /// reflection.
    pub fn bias(&self) -> Vector2<S> {
        self.bias
    }

    /// Return the normal vector to the reflection plane.
    pub fn normal(&self) -> Vector2<S> {
        self.normal
    }

    /// Calculate the inverse reflection transformation.
    pub fn inverse(&self) -> Option<Reflection2<S>> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let normal = self.normal;
        let inverse_det = one / (one - two * normal.x * normal.x - two * normal.y * normal.y);

        let c0r0 = one - two * normal.y * normal.y;
        let c0r1 = two * normal.x * normal.y;
        let c0r2 = zero;
        
        let c1r0 = two * normal.x * normal.y;
        let c1r1 = one - two * normal.x * normal.x - two * normal.y * normal.y;
        let c1r2 = zero;
        
        let c2r0 = zero;
        let c2r1 = zero;
        let c2r2 = one;

        let matrix = Matrix3x3::new(
            c0r0, c0r1, c0r2,                                   
            c1r0, c1r1, c1r2, 
            c2r0, c2r1, c2r2
        );

        Some(Reflection2 {
            bias: self.bias,
            normal: normal, 
            matrix: matrix * inverse_det 
        })
    }

    /// Reflect a vector across a line.
    pub fn reflect_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Reflect a point across a line.
    pub fn reflect_point(&self, point: &Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> AsRef<Matrix3x3<S>> for Reflection2<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Reflection2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Reflection2 [{}]",
            self.matrix
        )
    }
}

impl<S> From<Reflection2<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: Reflection2<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> From<&Reflection2<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: &Reflection2<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation2<S> for Reflection2<S> 
    where S: ScalarFloat 
{
    #[inline]
    fn identity() -> Reflection2<S> {
        Reflection2 { 
            bias: Vector2::zero(),
            normal: Vector2::zero(), 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Reflection2<S>> {
        self.inverse()
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.reflect_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        self.reflect_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2<S> {
        Transform2::to_transform2d(self)
    }
}


/// A reflection transformation about a plane in three dimensions.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reflection3<S> {
    /// a known point on the plane of reflection.
    bias: Vector3<S>,
    /// The normal vector to the plane.
    normal: Vector3<S>,
    /// The matrix representing the affine transformation.
    matrix: Matrix4x4<S>,
}

impl<S> Reflection3<S> where S: ScalarFloat {
    /// Construct a new reflection transformation from the vector normal to the 
    /// plane of reflection.
    pub fn from_normal_bias(normal: &Unit<Vector3<S>>, bias: &Vector3<S>) -> Reflection3<S> {
        Reflection3 {
            bias: *bias,
            normal: normal.into_inner(),
            matrix: Matrix4x4::from_affine_reflection(normal, bias),
        }
    }

    /// Return the bias for calculating the reflections.
    ///
    /// The `bias` is the coordinates of a known point in the plane of 
    /// reflection.
    pub fn bias(&self) -> Vector3<S> {
        self.bias
    }

    /// Return the normal vector to the reflection plane.
    pub fn normal(&self) -> Vector3<S> {
        self.normal
    }

    /// Calculate the inverse reflection transformation.
    pub fn inverse(&self) -> Option<Reflection3<S>> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let normal = self.normal;
        let inverse_det = one / (one - two * normal.x * normal.x - two * normal.y * normal.y - two * normal.z * normal.z);

        let c0r0 = one - two * normal.y * normal.y - normal.z * normal.z;
        let c0r1 = two * normal.x * normal.y;
        let c0r2 = two * normal.x * normal.z;
        let c0r3 = zero;

        let c1r0 = two * normal.x * normal.y;
        let c1r1 = one - two * normal.x * normal.x - two * normal.z * normal.z;
        let c1r2 = two * normal.y * normal.z;
        let c1r3 = zero;

        let c2r0 = two * normal.x * normal.z;
        let c2r1 = two * normal.y * normal.z;
        let c2r2 = one - two * normal.x * normal.x - two * normal.y * normal.y;
        let c2r3 = zero;

        let c3r0 = zero;
        let c3r1 = zero;
        let c3r2 = zero;
        let c3r3 = one;


        let matrix = Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        );

        Some(Reflection3 { 
            bias: self.bias,
            normal: normal, 
            matrix: matrix * inverse_det,
        })
    }

    /// Reflect a vector across a line.
    pub fn reflect_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Reflect a point across a plane.
    pub fn reflect_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> AsRef<Matrix4x4<S>> for Reflection3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Reflection3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Reflection3 [{}]",
            self.matrix
        )
    }
}

impl<S> From<Reflection3<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: Reflection3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> From<&Reflection3<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: &Reflection3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}


impl<S> AffineTransformation3<S> for Reflection3<S> where S: ScalarFloat {
    #[inline]
    fn identity() -> Reflection3<S> {
        Reflection3 { 
            bias: Vector3::zero(),
            normal: Vector3::zero(), 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Reflection3<S>> {
       self.inverse()
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.reflect_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        self.reflect_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::to_transform3d(self.matrix)
    }
}

