use crate::angle::{
    Radians,
};
use crate::scalar::{
    ScalarFloat,
};
use crate::traits::{
    Angle,
    DotProduct,
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
use crate::unit::{
    Unit,
};
use crate::affine::*;

use core::fmt;


/// A rotation operator in two dimensions.
/// 
/// Two-dimensional rotations are different than three-dimensional rotations 
/// because mathematically we cannot define an axis of rotation in two 
/// dimensions. Instead we have to talk about rotating in the _xy-plane_ by an 
/// angle. In low-dimensional settings, the notion of rotation axis is 
/// only well-defined in three dimensions because dimension three is the 
/// only dimension where every plane is guaranteed to have a normal vector. 
/// 
/// If one wants to talk about rotating a vector in the the _xy-plane_ about a 
/// normal vector, we are implicitly rotating about the _z-axis_ in 
/// three dimensions.
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct Rotation2<S> {
    /// The angle of rotation.
    angle: Radians<S>,
    /// The underlying rotation matrix.
    matrix: Matrix3x3<S>,
}

impl<S> Rotation2<S> where S: ScalarFloat {
    /// Rotate a two-dimensional vector in the _xy-plane_ by an angle `angle`.
    pub fn from_angle<A: Into<Radians<S>>>(angle: A) -> Rotation2<S> {
        let radians = angle.into();
        let matrix = Matrix3x3::from(Matrix2x2::from_angle(radians));
        
        Rotation2 {
            angle: radians,
            matrix: matrix,
        }
    }

    /// Point a vector at the point `direction`.
    #[inline]
    pub fn look_at(direction: Vector2<S>, up: Vector2<S>) -> Rotation2<S> {
        let matrix = Matrix3x3::from(Matrix2x2::look_at(direction, up));
        let angle = Radians::acos(matrix.c0r0);
        
        Rotation2 {
            angle: angle,
            matrix: matrix,
        }
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between two unit vectors.
    #[inline]
    pub fn rotation_between_vectors(a: &Unit<Vector2<S>>, b: &Unit<Vector2<S>>) -> Rotation2<S> {
        let _a = a.as_ref();
        let _b = b.as_ref();
        Rotation2::from_angle(Radians::acos(DotProduct::dot(_a, _b)))
    }

    /// Construct a rotation that rotates a vector in the opposite direction 
    /// of `self`. 
    ///
    /// Given a rotation operator that rotates a vector about a normal vector 
    /// `axis` by an angle `theta`, construct a rotation that rotates a 
    /// vector about the same axis by an angle `-theta`.
    #[inline]
    pub fn inverse(&self) -> Rotation2<S> {
        Rotation2 {
            angle: -self.angle,
            matrix: self.matrix.inverse().unwrap(),
        }
    }

    /// Apply the rotation operation to a vector.
    #[inline]
    pub fn rotate_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Apply the rotation operation to a point.
    #[inline]
    pub fn rotate_point(&self, point: Point2<S>) -> Point2<S> { 
        Point2::from_vector(self.rotate_vector(point.to_vector()))
    }
}

impl<S> fmt::Debug for Rotation2<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Rotation2 ")?;
        <[S; 9] as fmt::Debug>::fmt(self.matrix.as_ref(), f)
    }
}

impl<S> fmt::Display for Rotation2<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Rotation2 ")?;
        <[S; 9] as fmt::Debug>::fmt(self.matrix.as_ref(), f)
    }
}

impl<S> From<Rotation2<S>> for Matrix3x3<S> where S: Copy {
    #[inline]
    fn from(rotation: Rotation2<S>) -> Matrix3x3<S> {
        rotation.matrix
    }
}

impl<S> AsRef<Matrix3x3<S>> for Rotation2<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> approx::AbsDiffEq for Rotation2<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Rotation2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix3x3::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Rotation2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix3x3::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S> AffineTransformation2<Point2<S>, Vector2<S>, S> for Rotation2<S> 
    where S: ScalarFloat 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Rotation2<S> {
        Rotation2 { 
            angle: Radians(S::zero()),
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation2<S>> {
        Some(self.inverse())
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
    fn to_transform2d(&self) -> Transform2<S> {
        Transform2::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2<Point2<S>, &Vector2<S>, S> for Rotation2<S> 
    where S: ScalarFloat 
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Rotation2<S> {
        Rotation2 { 
            angle: Radians(S::zero()),
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation2<S>> {
        Some(self.inverse())
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
    fn to_transform2d(&self) -> Transform2<S> {
        Transform2::matrix_to_transform2d(self.matrix)
    }
}

impl<S> AffineTransformation2<&Point2<S>, Vector2<S>, S> for Rotation2<S> 
    where S: ScalarFloat
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Rotation2<S> {
        Rotation2 { 
            angle: Radians(S::zero()),
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation2<S>> {
        Some(self.inverse())
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
    fn to_transform2d(&self) -> Transform2<S> {
        Transform2::matrix_to_transform2d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation2<&'a Point2<S>, &'b Vector2<S>, S> for Rotation2<S> 
    where S: ScalarFloat
{
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Rotation2<S> {
        Rotation2 { 
            angle: Radians(S::zero()),
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation2<S>> {
        Some(self.inverse())
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
    fn to_transform2d(&self) -> Transform2<S> {
        Transform2::matrix_to_transform2d(self.matrix)
    }
}


/// A rotation operator in three dimensions.
///
/// A rotation is an operation that creates circular motions and 
/// preserves at least one point. Rotations preserve the length of vectors and 
/// therefore act as a class of rigid body transformations.
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct Rotation3<S> {
    /// The angle of rotation.
    angle: Radians<S>,
    /// The underlying matrix representing the rotation.
    matrix: Matrix4x4<S>,
}

impl<S> Rotation3<S> where S: ScalarFloat {
    /// Construct a three-dimensional rotation matrix from a quaternion.
    #[inline]
    pub fn from_quaternion(quaternion: &Quaternion<S>) -> Rotation3<S> {
        let quaternion_normalized = quaternion.normalize();
        let two = S::one() + S::one();
        let angle = Radians::acos(quaternion_normalized.s) * two;

        Rotation3 {
            angle: angle,
            matrix: Matrix4x4::from(quaternion),
        }
    }

    /// Construct a new three-dimensional rotation about an axis `axis` by 
    /// an angle `angle`.
    pub fn from_axis_angle<A: Into<Radians<S>>>(axis: Unit<Vector3<S>>, angle: A) -> Rotation3<S> {
        let radians = angle.into();
        
        Rotation3 {
            angle: radians,
            matrix: Matrix4x4::from_affine_axis_angle(axis, radians),
        }
    }

    /// Construct a new three-dimensional rotation about the _x-axis_ in the 
    /// _yz-plane_ by an angle `angle`.
    #[inline]
    pub fn from_angle_x<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_axis_angle(Unit::new_unchecked(Vector3::unit_x()), angle)
    }

    /// Construct a new three-dimensional rotation about the _y-axis_ in the 
    /// _xz-plane_ by an angle `angle`.
    #[inline]
    pub fn from_angle_y<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_axis_angle(Unit::new_unchecked(Vector3::unit_y()), angle)
    }

    /// Construct a new three-dimensional rotation about the _z-axis_ in the 
    /// _xy-plane_ by an angle `angle`.
    #[inline]
    pub fn from_angle_z<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_axis_angle(Unit::new_unchecked(Vector3::unit_z()), angle)
    }

    /// Point a vector at the point `direction`.
    #[inline]
    pub fn look_at(dir: Vector3<S>, up: Vector3<S>) -> Rotation3<S> {
        let matrix3 = Matrix3x3::look_at(dir, up);
        let quaternion = Quaternion::from(&matrix3);
        let two = S::one() + S::one();
        let angle = Radians::acos(quaternion.s) * two;
        let matrix = quaternion.into();
    
        Rotation3 {
            angle: angle,
            matrix: matrix,
        }
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between two unit vectors.
    #[inline]
    pub fn rotation_between_vectors(v1: &Unit<Vector3<S>>, v2: &Unit<Vector3<S>>) -> Rotation3<S> {
        let q = Quaternion::rotation_between_vectors(v1, v2);
        q.into()
    }

    /// Construct a rotation that rotates a vector in the opposite direction 
    /// of `self`. 
    ///
    /// Given a rotation operator that rotates a vector about a normal vector 
    /// `axis` by an angle `theta`, construct a rotation that rotates a 
    /// vector about the same axis by an angle `-theta`.
    #[inline]
    pub fn inverse(&self) -> Rotation3<S> {
        Rotation3 {
            angle: -self.angle,
            matrix: self.matrix.inverse().unwrap(),
        }
    }

    /// Apply the rotation operation to a vector.
    #[inline]
    pub fn rotate_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Apply the rotation operation to a point.
    #[inline]
    pub fn rotate_point(&self, point: Point3<S>) -> Point3<S> { 
        Point3::from_vector(self.rotate_vector(point.to_vector()))
    }
}

impl<S> fmt::Debug for Rotation3<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Rotation3 ")?;
        <[S; 16] as fmt::Debug>::fmt(self.matrix.as_ref(), f)
    }
}

impl<S> fmt::Display for Rotation3<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Rotation3 ")?;
        <[S; 16] as fmt::Debug>::fmt(self.matrix.as_ref(), f)
    }
}

impl<S> From<Rotation3<S>> for Matrix4x4<S> where S: Copy {
    #[inline]
    fn from(rotation: Rotation3<S>) -> Matrix4x4<S> {
        rotation.matrix
    }
}

impl<S> From<Quaternion<S>> for Rotation3<S> where S: ScalarFloat {
    #[inline]
    fn from(quaternion: Quaternion<S>) -> Rotation3<S> {
        Rotation3::from_quaternion(&quaternion)
    }
}

impl<S> From<Rotation3<S>> for Quaternion<S> where S: ScalarFloat {
    #[inline]
    fn from(rotation: Rotation3<S>) -> Quaternion<S> {
        let matrix = Matrix3x3::new(
            rotation.matrix.c0r0, rotation.matrix.c0r1, rotation.matrix.c0r2,
            rotation.matrix.c1r0, rotation.matrix.c1r1, rotation.matrix.c1r2,
            rotation.matrix.c2r0, rotation.matrix.c2r1, rotation.matrix.c2r2
        );
        Quaternion::from(&matrix)
    }
}

impl<S> AsRef<Matrix4x4<S>> for Rotation3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> approx::AbsDiffEq for Rotation3<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Rotation3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix4x4::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Rotation3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix4x4::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S> AffineTransformation3<Point3<S>, Vector3<S>, S> for Rotation3<S> 
    where S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Rotation3<S> {
        Rotation3 { 
            angle: Radians(S::zero()),
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation3<S>> {
        Some(self.inverse())
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
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3<Point3<S>, &Vector3<S>, S> for Rotation3<S>
    where S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Rotation3<S> {
        Rotation3 { 
            angle: Radians(S::zero()),
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation3<S>> {
        Some(self.inverse())
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
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::matrix_to_transform3d(self.matrix)
    }
}

impl<S> AffineTransformation3<&Point3<S>, Vector3<S>, S> for Rotation3<S> 
    where S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Rotation3<S> {
        Rotation3 { 
            angle: Radians(S::zero()),
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation3<S>> {
        Some(self.inverse())
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
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::matrix_to_transform3d(self.matrix)
    }
}

impl<'a, 'b, S> AffineTransformation3<&'a Point3<S>, &'b Vector3<S>, S> for Rotation3<S> 
    where S: ScalarFloat 
{
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Rotation3<S> {
        Rotation3 { 
            angle: Radians(S::zero()),
            matrix: Matrix4x4::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation3<S>> {
        Some(self.inverse())
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
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::matrix_to_transform3d(self.matrix)
    }
}

