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
use crate::traits::{
    Identity,
    Zero,
};
use crate::affine::*;

use core::fmt;


/// A trait defining reflections about a plane.
pub trait Reflection<P, V> where Self: Sized + Copy {
    /// The type of the output points (locations in space).
    type OutPoint;
    /// The type of the output vectors (displacements in space).
    type OutVector;

    /// Return the bias for calculating the reflections.
    ///
    /// The _bias_ is the coorindates of a known point in the plane of 
    /// reflection.
    fn bias(&self) -> Self::OutVector;

    /// Return the normal vector to the reflection plane.
    fn normal(&self) -> Self::OutVector;

    /// Construct a new reflection transformation from the vector normal to the 
    /// plane of reflection.
    fn from_normal_bias(normal: V, bias: V) -> Self;

    /// Calculate the inverse reflection transformation.
    fn inverse(&self) -> Option<Self>;

    /// Reflect a vector across a plane.
    fn reflect_vector(&self, vector: V) -> Self::OutVector;

    /// Reflect a point across a plane.
    fn reflect_point(&self, point: P) -> Self::OutPoint;
}

/// A trait defining reflections about a plane (line) in two dimensions.
pub trait Reflection2<S> where 
    S: ScalarFloat,
    Self: Reflection<Point2<S>, Vector2<S>> + Into<Matrix3x3<S>> + Into<Reflection2D<S>>,
{
}

/// A trait defining reflections about a plane in three dimensions.
pub trait Reflection3<S> where 
    S: ScalarFloat,
    Self: Reflection<Point3<S>, Vector3<S>> + Into<Matrix4x4<S>> + Into<Reflection3D<S>>,
{
}


/// A reflection transformation about a plane (line) in two dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Reflection2D<S> {
    /// A known point on the line of reflection.
    bias: Vector2<S>,
    /// The normal vector to the plane.
    normal: Vector2<S>,
    /// The matrix representing the affine transformation.
    matrix: Matrix3x3<S>,
}

impl<S> AsRef<Matrix3x3<S>> for Reflection2D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Reflection2D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Reflection2D<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: Reflection2D<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> From<&Reflection2D<S>> for Matrix3x3<S> where S: Copy {
    fn from(transformation: &Reflection2D<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> Reflection<Point2<S>, Vector2<S>> for Reflection2D<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    fn bias(&self) -> Vector2<S> {
        self.bias
    }

    fn normal(&self) -> Vector2<S> {
        self.normal
    }

    fn from_normal_bias(normal: Vector2<S>, bias: Vector2<S>) -> Reflection2D<S> {
        Reflection2D {
            bias: bias,
            normal: normal,
            matrix: Matrix3x3::from_affine_reflection(normal, bias),
        }
    }

    fn inverse(&self) -> Option<Reflection2D<S>> {
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

        Some(Reflection2D {
            bias: self.bias,
            normal: normal, 
            matrix: matrix * inverse_det 
        })
    }

    fn reflect_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    fn reflect_point(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> Reflection2<S> for Reflection2D<S> where S: ScalarFloat
{
}

impl<S> AffineTransformation2D<Point2<S>, Vector2<S>, S> for Reflection2D<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Reflection2D<S> {
        Reflection2D { 
            bias: Vector2::zero(),
            normal: Vector2::zero(), 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Reflection2D<S>> {
        <Self as Reflection<Point2<S>, Vector2<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        self.reflect_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: Point2<S>) -> Point2<S> {
        self.reflect_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2D<Point2<S>, &Vector2<S>, S> for Reflection2D<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Reflection2D<S> {
        Reflection2D { 
            bias: Vector2::zero(),
            normal: Vector2::zero(), 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Reflection2D<S>> {
        <Self as Reflection<Point2<S>, Vector2<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.reflect_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: Point2<S>) -> Point2<S> {
        self.reflect_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2D<&Point2<S>, Vector2<S>, S> for Reflection2D<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Reflection2D<S> {
        Reflection2D { 
            bias: Vector2::zero(),
            normal: Vector2::zero(), 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Reflection2D<S>> {
        <Self as Reflection<Point2<S>, Vector2<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        self.reflect_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        self.reflect_point(*point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation2D<&'a Point2<S>, &'b Vector2<S>, S> for Reflection2D<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Reflection2D<S> {
        Reflection2D { 
            bias: Vector2::zero(),
            normal: Vector2::zero(), 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Reflection2D<S>> {
        <Self as Reflection<Point2<S>, Vector2<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: &'b Vector2<S>) -> Vector2<S> {
        self.reflect_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: &'a Point2<S>) -> Point2<S> {
        self.reflect_point(*point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}


/// A reflection transformation about a plane in three dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Reflection3D<S> {
    /// a known point on the plane of reflection.
    bias: Vector3<S>,
    /// The normal vector to the plane.
    normal: Vector3<S>,
    /// The matrix representing the affine transformation.
    matrix: Matrix4x4<S>,
}

impl<S> Reflection3D<S> where S: ScalarFloat {
    /// Construct a new reflection transformation from the vector normal to the
    /// plane of reflection.
    pub fn from_normal_bias(normal: Vector3<S>, bias: Vector3<S>) -> Reflection3D<S> {
        Reflection3D {
            bias: bias,
            normal: normal,
            matrix: Matrix4x4::from_affine_reflection(normal, bias),
        }
    }
}

impl<S> AsRef<Matrix4x4<S>> for Reflection3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Reflection3D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Reflection3D<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: Reflection3D<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> From<&Reflection3D<S>> for Matrix4x4<S> where S: Copy {
    fn from(transformation: &Reflection3D<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> Reflection<Point3<S>, Vector3<S>> for Reflection3D<S> where S: ScalarFloat {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    fn bias(&self) -> Vector3<S> {
        self.bias
    }

    fn normal(&self) -> Vector3<S> {
        self.normal
    }

    fn from_normal_bias(normal: Vector3<S>, bias: Vector3<S>) -> Reflection3D<S> {
        Reflection3D {
            bias: bias,
            normal: normal,
            matrix: Matrix4x4::from_affine_reflection(normal, bias),
        }
    }

    fn inverse(&self) -> Option<Reflection3D<S>> {
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

        Some(Reflection3D { 
            bias: self.bias,
            normal: normal, 
            matrix: matrix * inverse_det,
        })
    }

    fn reflect_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    fn reflect_point(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> Reflection3<S> for Reflection3D<S> where S: ScalarFloat
{
}

impl<S> AffineTransformation3D<Point3<S>, Vector3<S>, S> for Reflection3D<S> where S: ScalarFloat {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Reflection3D<S> {
        Reflection3D { 
            bias: Vector3::zero(),
            normal: Vector3::zero(), 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Reflection3D<S>> {
        <Self as Reflection<Point3<S>, Vector3<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        self.reflect_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: Point3<S>) -> Point3<S> {
        self.reflect_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3D<Point3<S>, &Vector3<S>, S> for Reflection3D<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Reflection3D<S> {
        Reflection3D { 
            bias: Vector3::zero(),
            normal: Vector3::zero(), 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Reflection3D<S>> {
        <Self as Reflection<Point3<S>, Vector3<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.reflect_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: Point3<S>) -> Point3<S> {
        self.reflect_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3D<&Point3<S>, Vector3<S>, S> for Reflection3D<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Reflection3D<S> {
        Reflection3D { 
            bias: Vector3::zero(),
            normal: Vector3::zero(), 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Reflection3D<S>> {
        <Self as Reflection<Point3<S>, Vector3<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        self.reflect_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        self.reflect_point(*point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation3D<&'a Point3<S>, &'b Vector3<S>, S> for Reflection3D<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Reflection3D<S> {
        Reflection3D { 
            bias: Vector3::zero(),
            normal: Vector3::zero(), 
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Reflection3D<S>> {
        <Self as Reflection<Point3<S>, Vector3<S>>>::inverse(&self)
    }

    #[inline]
    fn transform_vector(&self, vector: &'b Vector3<S>) -> Vector3<S> {
        self.reflect_vector(*vector)
    }

    #[inline]
    fn transform_point(&self, point: &'a Point3<S>) -> Point3<S> {
        self.reflect_point(*point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

