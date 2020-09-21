use approx::{
    ulps_eq,
};
use crate::angle::{
    Radians,
};
use crate::scalar::{
    ScalarFloat,
};
use crate::traits::{
    Angle,
    DotProduct,
    CrossProduct,
    Identity,
    InvertibleSquareMatrix,
    Magnitude,
};
use crate::matrix::{
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
};
use crate::point::{
    Point2,
    Point3,
};
use crate::vector::{
    Vector2,
    Vector3,
};
use crate::quaternion::{
    Quaternion,
};
use crate::affine::*;

use core::fmt;


/// A type implementing this trait represents a type that acts as a generic 
/// rotation. A rotation is an operation that creates circular motions and 
/// preserves at least one point. Rotations preserve the length of vectors and 
/// therefore act as a class of rigid body transformations.
pub trait Rotation<P, V> where Self: Sized + Copy {
    /// The type of the output points (locations in space).
    type OutPoint;
    /// The type of the output vectors (displacements in space).
    type OutVector;

    /// Point a vector at the point `direction`.
    fn look_at(direction: V, up: V) -> Self;

    /// Construct a rotation that rotates the shortest angular distance 
    /// between two unit vectors.
    fn between_vectors(v1: V, v2: V) -> Self;

    /// Construct a rotation that rotates a vector in the opposite direction 
    /// of `self`. In particular, given a rotation operator that rotates a vector 
    /// about an axis by an angle `theta`, construct a rotation that rotates a 
    /// vector about the same axis by an angle `-theta`.
    fn inverse(&self) -> Self;

    /// Apply the rotation operation to a vector.
    fn rotate_vector(&self, vector: V) -> Self::OutVector;

    /// Apply the rotation operation to a point.
    fn rotate_point(&self, point: P) -> Self::OutPoint;
}

/// A trait that implements rotation operators in two-dimensions. 
/// 
/// Two-dimensional rotations are different than three-dimensional rotations in 
/// that mathematically we cannot define an axis of rotation in two dimensions. 
/// Instead we have to talk about rotating in the xy-plane by an angle. In 
/// low-dimensional settings, the notion of rotation axis is only well-defined 
/// in three dimensions because dimension three is the only dimension where 
/// every plane is guaranteed to have a normal vector. 
/// 
/// If one wants to talk about rotating a vector in the the xy-plane about a 
/// normal vector, we are implicitly rotating about the z-axis in 
/// three dimensions. Otherwise, to avoid cheating in that fashion we must 
/// abolish coordinate axes and only talk about (hyper)planes, but this 
/// requires the language of geometric algebra to express precisely.
pub trait Rotation2<S> where 
    S: ScalarFloat,
    Self: Rotation<Point2<S>, Vector2<S>> + Into<Matrix3x3<S>> + Into<Rotation2D<S>>,
{
    /// Rotate a two-dimensional vector in the xy-plane by an angle `angle`.
    fn from_angle<A: Into<Radians<S>>>(angle: A) -> Self;
}

/// A trait that implements rotation operators in three dimensions.
pub trait Rotation3<S> where 
    S: ScalarFloat,
    Self: Rotation<Point3<S>, Vector3<S>>,
    Self: Into<Matrix4x4<S>> + Into<Rotation3D<S>> + Into<Quaternion<S>>,
{
    /// Construct a new three-dimensional rotation about an axis `axis` by 
    /// an amount `angle`.
    fn from_axis_angle<A: Into<Radians<S>>>(axis: Vector3<S>, angle: A) -> Self;

    /// Construct a new three-dimensional rotation about the x-axis in the 
    /// yz-plane by an angle `angle`.
    #[inline]
    fn from_angle_x<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_axis_angle(Vector3::unit_x(), angle)
    }

    /// Construct a new three-dimensional rotation about the y-axis in the 
    /// xz-plane by an angle `angle`.
    #[inline]
    fn from_angle_y<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_axis_angle(Vector3::unit_y(), angle)
    }

    /// Construct a new three-dimensional rotation about the z-axis in the 
    /// xy-plane by an angle `angle`.
    #[inline]
    fn from_angle_z<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_axis_angle(Vector3::unit_z(), angle)
    }
}


/// A rotation operator in two dimensions.
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct Rotation2D<S> {
    /// The angle of rotation.
    angle: Radians<S>,
    /// The underlying matrix for the rotation.
    matrix: Matrix3x3<S>,
}

impl<S> fmt::Debug for Rotation2D<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Rotation2D ")?;
        <[S; 9] as fmt::Debug>::fmt(self.matrix.as_ref(), f)
    }
}

impl<S> fmt::Display for Rotation2D<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Rotation2D ")?;
        <[S; 9] as fmt::Debug>::fmt(self.matrix.as_ref(), f)
    }
}

impl<S> From<Rotation2D<S>> for Matrix3x3<S> where S: Copy {
    #[inline]
    fn from(rotation: Rotation2D<S>) -> Matrix3x3<S> {
        rotation.matrix
    }
}

impl<S> AsRef<Matrix3x3<S>> for Rotation2D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> approx::AbsDiffEq for Rotation2D<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Matrix3x3::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S> approx::RelativeEq for Rotation2D<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix3x3::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Rotation2D<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix3x3::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S> Rotation2<S> for Rotation2D<S> where S: ScalarFloat {
    fn from_angle<A: Into<Radians<S>>>(angle: A) -> Rotation2D<S> {
        let radians = angle.into();
        let matrix = Matrix3x3::from(Matrix2x2::from_angle(radians));
        
        Rotation2D {
            angle: radians,
            matrix: matrix,
        }
    }
}

impl<S> Rotation<Point2<S>, Vector2<S>> for Rotation2D<S> 
    where 
        S: ScalarFloat 
{ 
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn look_at(direction: Vector2<S>, up: Vector2<S>) -> Rotation2D<S> {
        let matrix = Matrix3x3::from(Matrix2x2::look_at(direction, up));
        let angle = Radians::acos(matrix.c0r0);
        
        Rotation2D {
            angle: angle,
            matrix: matrix,
        }
    }

    #[inline]
    fn between_vectors(a: Vector2<S>, b: Vector2<S>) -> Rotation2D<S> {
        Rotation2::from_angle(Radians::acos(DotProduct::dot(a, b)))
    }

    #[inline]
    fn rotate_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn inverse(&self) -> Rotation2D<S> {
        Rotation2D {
            angle: -self.angle,
            matrix: self.matrix.inverse().unwrap(),
        }
    }

    #[inline]
    fn rotate_point(&self, point: Point2<S>) -> Point2<S> { 
        Point2::from_vector(self.rotate_vector(point.to_vector()))
    }
}

impl<S> AffineTransformation2D<Point2<S>, Vector2<S>, S> for Rotation2D<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Rotation2D<S> {
        Rotation2D { 
            angle: Radians(S::zero()),
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation2D<S>> {
        Some(<Self as Rotation<Point2<S>, Vector2<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn transform_point(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2D<Point2<S>, &Vector2<S>, S> for Rotation2D<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Rotation2D<S> {
        Rotation2D { 
            angle: Radians(S::zero()),
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation2D<S>> {
        Some(<Self as Rotation<Point2<S>, Vector2<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn transform_point(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2D<&Point2<S>, Vector2<S>, S> for Rotation2D<S> 
    where 
        S: ScalarFloat
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Rotation2D<S> {
        Rotation2D { 
            angle: Radians(S::zero()),
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation2D<S>> {
        Some(<Self as Rotation<Point2<S>, Vector2<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation2D<&'a Point2<S>, &'b Vector2<S>, S> for Rotation2D<S> 
    where 
        S: ScalarFloat
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Rotation2D<S> {
        Rotation2D { 
            angle: Radians(S::zero()),
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation2D<S>> {
        Some(<Self as Rotation<Point2<S>, Vector2<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: &'b Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn transform_point(&self, point: &'a Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2D<S> {
        Transform2D::matrix_to_transform2d(self.matrix)
    }
}


/// A rotation operator in three dimensions.
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct Rotation3D<S> {
    /// The angle of rotation.
    angle: Radians<S>,
    /// The underlying matrix representing the rotation.
    matrix: Matrix4x4<S>,
}

impl<S> Rotation3D<S> where S: ScalarFloat {
    /// Construct a three-dimensional rotation matrix from a quaternion.
    #[inline]
    pub fn from_quaternion(quaternion: &Quaternion<S>) -> Rotation3D<S> {
        let quaternion_normalized = quaternion.normalize();
        let two = S::one() + S::one();
        let angle = Radians::acos(quaternion_normalized.s) * two;

        Rotation3D {
            angle: angle,
            matrix: Matrix4x4::from(quaternion),
        }
    }
}

impl<S> fmt::Debug for Rotation3D<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Rotation3D ")?;
        <[S; 16] as fmt::Debug>::fmt(self.matrix.as_ref(), f)
    }
}

impl<S> fmt::Display for Rotation3D<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Rotation3D ")?;
        <[S; 16] as fmt::Debug>::fmt(self.matrix.as_ref(), f)
    }
}

impl<S> From<Rotation3D<S>> for Matrix4x4<S> where S: Copy {
    #[inline]
    fn from(rotation: Rotation3D<S>) -> Matrix4x4<S> {
        rotation.matrix
    }
}

impl<S> From<Quaternion<S>> for Rotation3D<S> where S: ScalarFloat {
    #[inline]
    fn from(quaternion: Quaternion<S>) -> Rotation3D<S> {
        Rotation3D::from_quaternion(&quaternion)
    }
}

impl<S> From<Rotation3D<S>> for Quaternion<S> where S: ScalarFloat {
    #[inline]
    fn from(rotation: Rotation3D<S>) -> Quaternion<S> {
        let matrix = Matrix3x3::new(
            rotation.matrix.c0r0, rotation.matrix.c0r1, rotation.matrix.c0r2,
            rotation.matrix.c1r0, rotation.matrix.c1r1, rotation.matrix.c1r2,
            rotation.matrix.c2r0, rotation.matrix.c2r1, rotation.matrix.c2r2
        );
        Quaternion::from(&matrix)
    }
}

impl<S> AsRef<Matrix4x4<S>> for Rotation3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> approx::AbsDiffEq for Rotation3D<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Matrix4x4::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S> approx::RelativeEq for Rotation3D<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix4x4::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Rotation3D<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix4x4::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S> Rotation<Point3<S>, Vector3<S>> for Quaternion<S> where S: ScalarFloat {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;
    
    #[inline]
    fn look_at(dir: Vector3<S>, up: Vector3<S>) -> Quaternion<S> {
        Matrix3x3::look_at(dir, up).into()
    }

    #[inline]
    fn between_vectors(v1: Vector3<S>, v2: Vector3<S>) -> Quaternion<S> {
        let k_cos_theta = v1.dot(v2);

        // The vectors point in the same direction.
        if ulps_eq!(k_cos_theta, S::one()) {
            return Quaternion::<S>::identity();
        }

        let k = (v1.magnitude_squared() * v2.magnitude_squared()).sqrt();

        // The vectors point in opposite directions.
        if ulps_eq!(k_cos_theta / k, -S::one()) {
            let mut orthogonal = v1.cross(Vector3::unit_x());
            if ulps_eq!(orthogonal.magnitude_squared(), S::zero()) {
                orthogonal = v1.cross(Vector3::unit_y());
            }
            return Quaternion::from_sv(S::zero(), orthogonal.normalize());
        }

        // The vectors point in any other direction.
        Quaternion::from_sv(k + k_cos_theta, v1.cross(v2)).normalize()
    }

    #[inline]
    fn rotate_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        let rotation_matrix = Matrix3x3::from(self);
        rotation_matrix * vector
    }

    #[inline]
    fn inverse(&self) -> Quaternion<S> {
        self.conjugate() / self.magnitude_squared()
    }

    #[inline]
    fn rotate_point(&self, point: Point3<S>) -> Point3<S> { 
        Point3::from_vector(self.rotate_vector(point.to_vector()))
    }
}

impl<S> Rotation3<S> for Rotation3D<S> where S: ScalarFloat {
    fn from_axis_angle<A: Into<Radians<S>>>(axis: Vector3<S>, angle: A) -> Rotation3D<S> {
        let radians = angle.into();
        
        Rotation3D {
            angle: radians,
            matrix: Matrix4x4::from_affine_axis_angle(axis, radians),
        }
    }
}

impl<S> Rotation3<S> for Quaternion<S> where S: ScalarFloat {
    #[inline]
    fn from_axis_angle<A: Into<Radians<S>>>(axis: Vector3<S>, angle: A) -> Quaternion<S> {
        let one_half = num_traits::cast(0.5_f64).unwrap();
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into() * one_half);
        Quaternion::from_sv(cos_angle, axis * sin_angle)
    }
}

impl<S> Rotation<Point3<S>, Vector3<S>> for Rotation3D<S> where S: ScalarFloat {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn look_at(dir: Vector3<S>, up: Vector3<S>) -> Rotation3D<S> {
        let matrix3 = Matrix3x3::look_at(dir, up);
        let quaternion = Quaternion::from(&matrix3);
        let two = S::one() + S::one();
        let angle = Radians::acos(quaternion.s) * two;
        let matrix = quaternion.into();
    
        Rotation3D {
            angle: angle,
            matrix: matrix,
        }
    }

    #[inline]
    fn between_vectors(v1: Vector3<S>, v2: Vector3<S>) -> Rotation3D<S> {
        let q: Quaternion<S> = Rotation::between_vectors(v1, v2);
        q.into()
    }

    #[inline]
    fn rotate_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn inverse(&self) -> Rotation3D<S> {
        Rotation3D {
            angle: -self.angle,
            matrix: self.matrix.inverse().unwrap(),
        }
    }

    #[inline]
    fn rotate_point(&self, point: Point3<S>) -> Point3<S> { 
        Point3::from_vector(self.rotate_vector(point.to_vector()))
    }
}

impl<S> AffineTransformation3D<Point3<S>, Vector3<S>, S> for Rotation3D<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Rotation3D<S> {
        Rotation3D { 
            angle: Radians(S::zero()),
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation3D<S>> {
        Some(<Self as Rotation<Point3<S>, Vector3<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn transform_point(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3D<Point3<S>, &Vector3<S>, S> for Rotation3D<S>
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Rotation3D<S> {
        Rotation3D { 
            angle: Radians(S::zero()),
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation3D<S>> {
        Some(<Self as Rotation<Point3<S>, Vector3<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn transform_point(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3D<&Point3<S>, Vector3<S>, S> for Rotation3D<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Rotation3D<S> {
        Rotation3D { 
            angle: Radians(S::zero()),
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation3D<S>> {
        Some(<Self as Rotation<Point3<S>, Vector3<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation3D<&'a Point3<S>, &'b Vector3<S>, S> for Rotation3D<S> 
    where 
        S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Rotation3D<S> {
        Rotation3D { 
            angle: Radians(S::zero()),
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation3D<S>> {
        Some(<Self as Rotation<Point3<S>, Vector3<S>>>::inverse(&self))
    }

    #[inline]
    fn transform_vector(&self, vector: &'b Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    #[inline]
    fn transform_point(&self, point: &'a Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3D<S> {
        Transform3D::matrix_to_transform3d(self.matrix)
    }
}

