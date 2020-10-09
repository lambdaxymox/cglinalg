use crate::angle::{
    Angle,
    Radians,
};
use crate::scalar::{
    ScalarFloat,
};
use crate::traits::{
    DotProduct,
    Identity,
    Matrix,
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
use crate::transform::*;

use core::fmt;


/// A rotation operator in two dimensions.
/// 
/// Two-dimensional rotations are different than three-dimensional rotations 
/// because mathematically we cannot define an axis of rotation in two 
/// dimensions. Instead we have to talk about rotating in the **xy-plane** by an 
/// angle. In low-dimensional settings, the notion of rotation axis is 
/// only well-defined in three dimensions because dimension three is the 
/// only dimension where every plane is guaranteed to have a normal vector. 
/// 
/// If one wants to talk about rotating a vector in the the **xy-plane** about a 
/// normal vector, we are implicitly rotating about the **z-axis** in 
/// three dimensions.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rotation2<S> {
    /// The underlying rotation matrix.
    matrix: Matrix2x2<S>,
}

impl<S> Rotation2<S> where S: ScalarFloat {
    /// Get a reference to the underlying matrix that represents the 
    /// rotation.
    #[inline]
    pub fn matrix(&self) -> &Matrix2x2<S> {
        &self.matrix
    }

    /// Get the rotation angle of the rotation transformation.
    #[inline]
    pub fn angle(&self) -> Radians<S> {
        Radians::atan2(self.matrix.c0r1, self.matrix.c0r0)
    }

    /// Rotate a two-dimensional vector in the **xy-plane** by an angle `angle`.
    #[inline]
    pub fn from_angle<A: Into<Radians<S>>>(angle: A) -> Rotation2<S> {  
        Rotation2 {
            matrix: Matrix2x2::from_angle(angle.into()),
        }
    }

    /// Point a vector at the point `direction`.
    #[inline]
    pub fn look_at(direction: &Vector2<S>, up: &Vector2<S>) -> Rotation2<S> {
        Rotation2 {
            matrix: Matrix2x2::look_at(direction, up),
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
            matrix: self.matrix.transpose(),
        }
    }

    /// Apply the rotation operation to a vector.
    #[inline]
    pub fn rotate_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.matrix * vector
    }

    /// Apply the rotation operation to a point.
    #[inline]
    pub fn rotate_point(&self, point: &Point2<S>) -> Point2<S> {
        let vector = Vector2::new(point.x, point.y);
        let result = self.matrix * vector;
        
        Point2::new(result.x, result.y)
    }
}

impl<S> fmt::Display for Rotation2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Rotation2 [{}]", self.matrix)
    }
}

impl<S> From<Rotation2<S>> for Matrix2x2<S> where S: ScalarFloat {
    #[inline]
    fn from(rotation: Rotation2<S>) -> Matrix2x2<S> {
        rotation.matrix
    }
}

impl<S> From<Rotation2<S>> for Matrix3x3<S> where S: ScalarFloat {
    #[inline]
    fn from(rotation: Rotation2<S>) -> Matrix3x3<S> {
        Matrix3x3::from(&rotation.matrix)
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
        Matrix2x2::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S> approx::RelativeEq for Rotation2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix2x2::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Rotation2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix2x2::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S> AffineTransformation2<S> for Rotation2<S> 
    where S: ScalarFloat 
{
    #[inline]
    fn identity() -> Rotation2<S> {
        Rotation2 { 
            matrix: Matrix2x2::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation2<S>> {
        Some(self.inverse())
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.rotate_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        self.rotate_point(point)
    }

    #[inline]
    fn to_transform2d(&self) -> Transform2<S> {
        Transform2::to_transform2d(self.matrix)
    }
}


/// A rotation operator in three dimensions.
///
/// A rotation is an operation that creates circular motions and 
/// preserves at least one point. Rotations preserve the length of vectors and 
/// therefore act as a class of rigid body transformations.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rotation3<S> {
    /// The underlying matrix representing the rotation.
    matrix: Matrix3x3<S>,
}

impl<S> Rotation3<S> where S: ScalarFloat {
    /// Get a reference to the underlying matrix that represents the 
    /// rotation.
    #[inline]
    pub fn matrix(&self) -> &Matrix3x3<S> {
        &self.matrix
    }

    /// Compute the angle of the rotation.
    #[inline]
    pub fn angle(&self) -> Radians<S> {
        let two = num_traits::cast(2_i8).unwrap();
        Radians::acos((
            self.matrix.c0r0 + self.matrix.c1r1 + self.matrix.c2r2 - S::one()) / two
        )
    }

    /// Compute the axis of the rotation, if it exists.
    #[inline]
    pub fn axis(&self) -> Option<Unit<Vector3<S>>> {
        let axis = Vector3::new(
            self.matrix.c1r2 - self.matrix.c2r1,
            self.matrix.c2r0 - self.matrix.c0r2,
            self.matrix.c0r1 - self.matrix.c1r0
        );

        Unit::try_from_value(axis, S::default_epsilon())
    }

    /// Compute the axis and angle of the rotation.
    #[inline]
    pub fn axis_angle(&self) -> Option<(Unit<Vector3<S>>, Radians<S>)> {
        if let Some(axis) = self.axis() {
            Some((axis, self.angle()))
        } else {
            None
        }
    }

    /// Construct a three-dimensional rotation matrix from a quaternion.
    #[inline]
    pub fn from_quaternion(quaternion: &Quaternion<S>) -> Rotation3<S> {
        Rotation3 {
            matrix: quaternion.to_matrix3x3(),
        }
    }

    /// Construct a new three-dimensional rotation about an axis `axis` by 
    /// an angle `angle`.
    pub fn from_axis_angle<A: Into<Radians<S>>>(axis: &Unit<Vector3<S>>, angle: A) -> Rotation3<S> {
        Rotation3 {
            matrix: Matrix3x3::from_axis_angle(axis, angle.into()),
        }
    }

    /// Construct a new three-dimensional rotation about the **x-axis** in the 
    /// **yz-plane** by an angle `angle`.
    #[inline]
    pub fn from_angle_x<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_axis_angle(&Unit::from_value_unchecked(Vector3::unit_x()), angle)
    }

    /// Construct a new three-dimensional rotation about the **y-axis** in the 
    /// **xz-plane** by an angle `angle`.
    #[inline]
    pub fn from_angle_y<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_axis_angle(&Unit::from_value_unchecked(Vector3::unit_y()), angle)
    }

    /// Construct a new three-dimensional rotation about the **z-axis** in the 
    /// **xy-plane** by an angle `angle`.
    #[inline]
    pub fn from_angle_z<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_axis_angle(&Unit::from_value_unchecked(Vector3::unit_z()), angle)
    }

    /// Construct a coordinate transformation that transforms
    /// a coordinate system of an observer located at the position `eye` facing 
    /// the direction `direction` into the coordinate system of an observer located
    /// at the origin facing the **negative z-axis**.
    ///
    /// The function maps the direction `direction` to the **negative z-axis** and 
    /// locates the `eye` position to the origin in the new the coordinate system.
    /// This transformation is a **right-handed** coordinate transformation. It is
    /// conventionally used in computer graphics for camera view transformations.
    #[inline]
    pub fn look_at_rh(direction: &Vector3<S>, up: &Vector3<S>) -> Rotation3<S> {
        Rotation3 {
            matrix: Matrix3x3::look_at_rh(direction, up),
        }
    }

    /// Construct a coordinate transformation that transforms
    /// a coordinate system of an observer located at the position `eye` facing 
    /// the direction `direction` into the coordinate system of an observer located
    /// at the origin facing the **positive z-axis**.
    ///
    /// The function maps the direction `direction` to the **positive z-axis** and 
    /// locates the `eye` position to the origin in the new the coordinate system.
    /// This transformation is a **left-handed** coordinate transformation.
    #[inline]
    pub fn look_at_lh(direction: &Vector3<S>, up: &Vector3<S>) -> Rotation3<S> {
        Rotation3 {
            matrix: Matrix3x3::look_at_lh(direction, up),
        }
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between two vectors.
    #[inline]
    pub fn rotation_between(
        v1: &Vector3<S>, v2: &Vector3<S>) -> Option<Rotation3<S>> {
            
        Quaternion::rotation_between(v1, v2).map(|q| q.into())
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between two unit vectors.
    #[inline]
    pub fn rotation_between_axis(
        v1: &Unit<Vector3<S>>, v2: &Unit<Vector3<S>>) -> Option<Rotation3<S>> {
            
        Quaternion::rotation_between_axis(v1, v2).map(|q| q.into())
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
            matrix: self.matrix.transpose(),
        }
    }

    /// Apply the rotation operation to a vector.
    #[inline]
    pub fn rotate_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.matrix * vector
    }

    /// Apply the rotation operation to a point.
    #[inline]
    pub fn rotate_point(&self, point: &Point3<S>) -> Point3<S> { 
        let vector = Vector3::new(point.x, point.y, point.z);
        let result = self.matrix * vector;

        Point3::new(result.x, result.y, result.z)
    }
}

impl<S> fmt::Display for Rotation3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Rotation3 [{}]", self.matrix)
    }
}

impl<S> From<Rotation3<S>> for Matrix4x4<S> where S: ScalarFloat {
    #[inline]
    fn from(rotation: Rotation3<S>) -> Matrix4x4<S> {
        Matrix4x4::from(&rotation.matrix)
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

impl<S> AsRef<Matrix3x3<S>> for Rotation3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
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
        Matrix3x3::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S> approx::RelativeEq for Rotation3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix3x3::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Rotation3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix3x3::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S> AffineTransformation3<S> for Rotation3<S> 
    where S: ScalarFloat 
{
    #[inline]
    fn identity() -> Rotation3<S> {
        Rotation3 { 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Rotation3<S>> {
        Some(self.inverse())
    }

    #[inline]
    fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.rotate_vector(vector)
    }

    #[inline]
    fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        self.rotate_point(point)
    }

    #[inline]
    fn to_transform3d(&self) -> Transform3<S> {
        Transform3::to_transform3d(self.matrix)
    }
}

