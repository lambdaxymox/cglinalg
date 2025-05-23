use crate::transform::{
    Transform2,
    Transform3,
};
use cglinalg_core::{
    Matrix3x3,
    Matrix4x4,
    Point,
    Point2,
    Point3,
    Unit,
    Vector,
    Vector2,
    Vector3,
};
use cglinalg_numeric::{
    SimdScalarFloat,
    SimdScalarSigned,
};

use core::fmt;
use core::ops;

/// A shearing transformation in two dimensions.
pub type Shear2<S> = Shear<S, 2>;

/// A shearing transformation in three dimensions.
pub type Shear3<S> = Shear<S, 3>;

/// A shearing transformation.
///
/// This is the most general shearing type. The vast majority of applications
/// should use [`Shear2`] or [`Shear3`] instead of this type directly.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Shear<S, const N: usize> {
    shear_factor: S,
    origin: Point<S, N>,
    direction: Vector<S, N>,
    normal: Vector<S, N>,
}

impl<S, const N: usize> Shear<S, N>
where
    S: Copy,
{
    /// Get the shear factor of the shearing transformation.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Shear2;
    /// #
    /// let shear_factor = 4_f64;
    /// let origin = Point2::new(0_f64, 0_f64);
    /// let direction = Unit::from_value(Vector2::unit_y());
    /// let normal = Unit::from_value(Vector2::unit_x());
    /// let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
    ///
    /// assert_eq!(shear.shear_factor(), shear_factor);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Shear3;
    /// # use core::f64;
    /// #
    /// let shear_factor = 8_f64;
    /// let origin = Point3::new(0_f64, 0_f64, 0_f64);
    /// let direction = Unit::from_value(Vector3::new(
    ///     -1_f64 / f64::sqrt(2_f64),
    ///      1_f64 / f64::sqrt(2_f64),
    ///      0_f64,
    /// ));
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
    ///
    /// assert_eq!(shear.shear_factor(), shear_factor);
    /// ```
    #[inline]
    pub const fn shear_factor(&self) -> S {
        self.shear_factor
    }

    /// Get the origin of the affine frame of the shearing transformation.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Shear2;
    /// #
    /// let shear_factor = 4_f64;
    /// let origin = Point2::new(0_f64, 0_f64);
    /// let direction = Unit::from_value(Vector2::unit_y());
    /// let normal = Unit::from_value(Vector2::unit_x());
    /// let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
    ///
    /// assert_eq!(shear.origin(), origin);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Shear3;
    /// # use core::f64;
    /// #
    /// let shear_factor = 8_f64;
    /// let origin = Point3::new(0_f64, 0_f64, 0_f64);
    /// let direction = Unit::from_value(Vector3::new(
    ///     -1_f64 / f64::sqrt(2_f64),
    ///      1_f64 / f64::sqrt(2_f64),
    ///      0_f64,
    /// ));
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
    ///
    /// assert_eq!(shear.origin(), origin);
    /// ```
    #[inline]
    pub const fn origin(&self) -> Point<S, N> {
        self.origin
    }

    /// Get the direction vector of shearing in the shearing plane of the
    /// shearing transformation.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Shear2;
    /// #
    /// let shear_factor = 4_f64;
    /// let origin = Point2::new(0_f64, 0_f64);
    /// let direction = Unit::from_value(Vector2::unit_y());
    /// let normal = Unit::from_value(Vector2::unit_x());
    /// let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
    ///
    /// assert_eq!(shear.direction(), direction.into_inner());
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Shear3;
    /// # use core::f64;
    /// #
    /// let shear_factor = 8_f64;
    /// let origin = Point3::new(0_f64, 0_f64, 0_f64);
    /// let direction = Unit::from_value(Vector3::new(
    ///     -1_f64 / f64::sqrt(2_f64),
    ///      1_f64 / f64::sqrt(2_f64),
    ///      0_f64,
    /// ));
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
    ///
    /// assert_eq!(shear.direction(), direction.into_inner());
    /// ```
    #[inline]
    pub const fn direction(&self) -> Vector<S, N> {
        self.direction
    }

    /// Get the normal vector to the shearing plane of the shearing transformation.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Shear2;
    /// #
    /// let shear_factor = 4_f64;
    /// let origin = Point2::new(0_f64, 0_f64);
    /// let direction = Unit::from_value(Vector2::unit_y());
    /// let normal = Unit::from_value(Vector2::unit_x());
    /// let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
    ///
    /// assert_eq!(shear.normal(), normal.into_inner());
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Shear3;
    /// # use core::f64;
    /// #
    /// let shear_factor = 8_f64;
    /// let origin = Point3::new(0_f64, 0_f64, 0_f64);
    /// let direction = Unit::from_value(Vector3::new(
    ///     -1_f64 / f64::sqrt(2_f64),
    ///      1_f64 / f64::sqrt(2_f64),
    ///      0_f64,
    /// ));
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
    ///
    /// assert_eq!(shear.normal(), normal.into_inner());
    /// ```
    #[inline]
    pub const fn normal(&self) -> Vector<S, N> {
        self.normal
    }
}

impl<S, const N: usize> Shear<S, N>
where
    S: SimdScalarSigned,
{
    /// Apply a shear transformation to a vector.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Shear2;
    /// #
    /// let shear_factor = 4_f64;
    /// let direction = Unit::from_value(Vector2::unit_x());
    /// let normal = Unit::from_value(Vector2::unit_y());
    /// let shear = Shear2::from_shear(shear_factor, &direction, &normal);
    /// let vertices = [
    ///     Vector2::new( 1_f64,  1_f64),
    ///     Vector2::new(-1_f64,  1_f64),
    ///     Vector2::new(-1_f64, -1_f64),
    ///     Vector2::new( 1_f64, -1_f64),
    /// ];
    /// let expected = [
    ///     Vector2::new( 1_f64 + shear_factor,  1_f64),
    ///     Vector2::new(-1_f64 + shear_factor,  1_f64),
    ///     Vector2::new(-1_f64 - shear_factor, -1_f64),
    ///     Vector2::new( 1_f64 - shear_factor, -1_f64),
    /// ];
    /// let result = vertices.map(|v| shear.apply_vector(&v));
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Shear3;
    /// # use core::f64;
    /// #
    /// let shear_factor = 8_f64;
    /// let direction = Unit::from_value(Vector3::new(1_f64 / f64::sqrt(2_f64), 1_f64 / f64::sqrt(2_f64), 0_f64));
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let shear = Shear3::from_shear(shear_factor, &direction, &normal);
    /// let vertices = [
    ///     Vector3::new( 1_f64,  1_f64,  1_f64),
    ///     Vector3::new(-1_f64,  1_f64,  1_f64),
    ///     Vector3::new(-1_f64, -1_f64,  1_f64),
    ///     Vector3::new( 1_f64, -1_f64,  1_f64),
    ///     Vector3::new( 1_f64,  1_f64, -1_f64),
    ///     Vector3::new(-1_f64,  1_f64, -1_f64),
    ///     Vector3::new(-1_f64, -1_f64, -1_f64),
    ///     Vector3::new( 1_f64, -1_f64, -1_f64),
    /// ];
    /// let expected = [
    ///     Vector3::new( 1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Vector3::new(-1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Vector3::new(-1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Vector3::new( 1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Vector3::new( 1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    ///     Vector3::new(-1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    ///     Vector3::new(-1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    ///     Vector3::new( 1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),  
    /// ];
    /// let result = vertices.map(|v| shear.apply_vector(&v));
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn apply_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        let origin = self.origin.to_vector();
        let factor = self.shear_factor * (vector - origin).dot(&self.normal);

        vector + self.direction * factor
    }

    /// Apply a shear transformation to a vector.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Shear2;
    /// #
    /// let shear_factor = 4_f64;
    /// let direction = Unit::from_value(Vector2::unit_x());
    /// let normal = Unit::from_value(Vector2::unit_y());
    /// let shear = Shear2::from_shear(shear_factor, &direction, &normal);
    /// let vertices = [
    ///     Point2::new( 1_f64,  1_f64),
    ///     Point2::new(-1_f64,  1_f64),
    ///     Point2::new(-1_f64, -1_f64),
    ///     Point2::new( 1_f64, -1_f64),
    /// ];
    /// let expected = [
    ///     Point2::new( 1_f64 + shear_factor,  1_f64),
    ///     Point2::new(-1_f64 + shear_factor,  1_f64),
    ///     Point2::new(-1_f64 - shear_factor, -1_f64),
    ///     Point2::new( 1_f64 - shear_factor, -1_f64),
    /// ];
    /// let result = vertices.map(|p| shear.apply_point(&p));
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Shear3;
    /// # use core::f64;
    /// #
    /// let shear_factor = 8_f64;
    /// let direction = Unit::from_value(Vector3::new(1_f64 / f64::sqrt(2_f64), 1_f64 / f64::sqrt(2_f64), 0_f64));
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let shear = Shear3::from_shear(shear_factor, &direction, &normal);
    /// let vertices = [
    ///     Point3::new( 1_f64,  1_f64,  1_f64),
    ///     Point3::new(-1_f64,  1_f64,  1_f64),
    ///     Point3::new(-1_f64, -1_f64,  1_f64),
    ///     Point3::new( 1_f64, -1_f64,  1_f64),
    ///     Point3::new( 1_f64,  1_f64, -1_f64),
    ///     Point3::new(-1_f64,  1_f64, -1_f64),
    ///     Point3::new(-1_f64, -1_f64, -1_f64),
    ///     Point3::new( 1_f64, -1_f64, -1_f64),
    /// ];
    /// let expected = [
    ///     Point3::new( 1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Point3::new(-1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Point3::new(-1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Point3::new( 1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Point3::new( 1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    ///     Point3::new(-1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    ///     Point3::new(-1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    ///     Point3::new( 1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    /// ];
    /// let result = vertices.map(|p| shear.apply_point(&p));
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn apply_point(&self, point: &Point<S, N>) -> Point<S, N> {
        let factor = self.shear_factor * (point - self.origin).dot(&self.normal);

        point + self.direction * factor
    }

    /// Construct an identity shear transformation.
    ///
    /// The identity shear is a shear transformation that does not shear
    /// any coordinates of a vector or a point. The identity shear returned
    /// by this function is not unique. With a shearing factor of zero, any
    /// combination of direction and normal can act as an identity.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::Point2;
    /// # use cglinalg_transform::Shear2;
    /// #
    /// let shear = Shear2::identity();
    /// let vertices = [
    ///     Point2::new( 1_f64,  1_f64),
    ///     Point2::new(-1_f64,  1_f64),
    ///     Point2::new(-1_f64, -1_f64),
    ///     Point2::new( 1_f64, -1_f64),
    /// ];
    /// let expected = vertices;
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::Point3;
    /// # use cglinalg_transform::Shear3;
    /// #
    /// let shear = Shear3::identity();
    /// let vertices = [
    ///     Point3::new( 1_f64,  1_f64,  1_f64),
    ///     Point3::new(-1_f64,  1_f64,  1_f64),
    ///     Point3::new(-1_f64, -1_f64,  1_f64),
    ///     Point3::new( 1_f64, -1_f64,  1_f64),
    ///     Point3::new( 1_f64,  1_f64, -1_f64),
    ///     Point3::new(-1_f64,  1_f64, -1_f64),
    ///     Point3::new(-1_f64, -1_f64, -1_f64),
    ///     Point3::new( 1_f64, -1_f64, -1_f64),
    /// ];
    /// let expected = vertices;
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        let mut direction = Vector::zero();
        direction[0] = S::one();
        let mut normal = Vector::zero();
        normal[N - 1] = S::one();

        Self {
            shear_factor: S::zero(),
            origin: Point::origin(),
            direction,
            normal,
        }
    }
}

impl<S, const N: usize> Shear<S, N>
where
    S: SimdScalarSigned,
{
    /// Calculate an inverse of a shear transformation.
    ///
    /// The shearing transformation as represented by the [`Shear`] data type
    /// does not have a unique inverse representation: the matrix for the
    /// shearing transformation encodes more than one possible inverse representation
    /// for a given shearing transformation. Negating the shear factor, negate the
    /// normal vector, negate the direction, or negating all three of them all yield
    /// the same underlying matrix representing the shearing transformation. As a
    /// consequence, this function returns the simplest inverse representation: negating
    /// the shear factor. The shearing transformation representation returned
    /// by this function uses the same direction and normal vectors as the original
    /// shearing transformation.
    ///
    /// # Discussion
    ///
    /// The general shearing transformation is defined geometrically by a shearing
    /// plane `S`, a point `Q` in `S` which defines the origin for the affine frame
    /// of the shearing transformation, a shearing direction `v` inside `S`, a normal
    /// vector `n` perpendicular to `S`, and a shearing factor `m`. The
    /// **shearing transformation** is defined geometrically by
    /// ```text
    /// H(p) := p + (m * dot(p - Q, n)) * v
    /// ```
    /// where `p` is a point in Euclidean space. The inverse of `H` is given by
    /// ```text
    /// Hinv(p) := p - (m * dot(p - Q, n)) * v
    /// ```
    /// To see this, we do a simple computation.
    /// ```text
    /// Hinv(H(p)) == H(p) - (m * dot(H(p) - Q, n)) * v
    ///            == H(p) - (m * dot(p + m * dot(p - Q, n) * v - Q, n)) * v
    ///            == H(p) - [m * dot(p - Q + m * dot(p - Q, n) * v, n)] * v
    ///            == H(p) - [m * dot(p - Q, n) + (m * m) * dot(p - Q, n) * dot(v, n))] * v
    ///            == H(p) - [m * dot(p - Q, n)] * v + [(m * m) * dot(p - Q) * dot(v, n)] * v
    ///            == H(p) - [m * dot(p - Q, n)] * v
    ///            == p + [m * dot(p - Q, n)] * v - [m * dot(p - Q, n)] * v
    ///            == p
    /// ```
    /// and we see that `Hinv` is indeed the inverse of `H`. The sixth equality follows
    /// from the fact that the set {`v`, `n`, and `v x n`} together with `Q` define an
    /// affine coordinate frame, so that `dot(v, n) == 0`.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Shear2;
    /// # use core::f64;
    /// #
    /// let shear_factor = 4_f64;
    /// let origin = Point2::new(3_f64, 3_f64);
    /// let direction = Unit::from_value(Vector2::new(
    ///     f64::sqrt(9_f64 / 10_f64),
    ///     f64::sqrt(1_f64 / 10_f64),
    /// ));
    /// let normal = Unit::from_value(Vector2::new(
    ///     -f64::sqrt(1_f64 / 10_f64),
    ///      f64::sqrt(9_f64 / 10_f64),
    /// ));
    /// let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
    /// let shear_inv = shear.inverse();
    /// let point = Point2::new(1_f64, 2_f64);
    ///
    /// assert_relative_eq!((shear * shear_inv) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!((shear_inv * shear) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    ///
    /// let other_shear_inv1 = Shear2::from_affine_shear(shear_factor, &origin, &(-direction), &normal);
    /// let other_shear_inv2 = Shear2::from_affine_shear(shear_factor, &origin, &direction, &(-normal));
    /// let other_shear_inv3 = Shear2::from_affine_shear(-shear_factor, &origin, &(-direction), &(-normal));
    ///
    /// assert_relative_eq!((shear * other_shear_inv1) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!((other_shear_inv1 * shear) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!((shear * other_shear_inv2) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!((other_shear_inv2 * shear) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!((shear * other_shear_inv3) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!((other_shear_inv3 * shear) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    ///
    /// // The inverse of the shearing transformation is not unique.
    /// assert_ne!(other_shear_inv1, shear_inv);
    /// assert_ne!(other_shear_inv2, shear_inv);
    /// assert_ne!(other_shear_inv3, shear_inv);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Shear3;
    /// # use core::f64;
    /// #
    /// let shear_factor = 8_f64;
    /// let origin = Point3::new(0_f64, 0_f64, 2_f64);
    /// let direction = Unit::from_value(Vector3::new(
    ///     f64::sqrt(1_f64 / 2_f64),
    ///     f64::sqrt(1_f64 / 2_f64),
    ///     0_f64,
    /// ));
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
    /// let shear_inv = shear.inverse();
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    ///
    /// assert_relative_eq!((shear * shear_inv) * point, point, abs_diff_all <= f64::EPSILON, relative_all <= f64::EPSILON);
    /// assert_relative_eq!((shear_inv * shear) * point, point, abs_diff_all <= f64::EPSILON, relative_all <= f64::EPSILON);
    ///
    /// let other_shear_inv1 = Shear3::from_affine_shear(shear_factor, &origin, &(-direction), &normal);
    /// let other_shear_inv2 = Shear3::from_affine_shear(shear_factor, &origin, &direction, &(-normal));
    /// let other_shear_inv3 = Shear3::from_affine_shear(-shear_factor, &origin, &(-direction), &(-normal));
    ///
    /// assert_relative_eq!((shear * other_shear_inv1) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!((other_shear_inv1 * shear) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!((shear * other_shear_inv2) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!((other_shear_inv2 * shear) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!((shear * other_shear_inv3) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!((other_shear_inv3 * shear) * point, point, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    ///
    /// // The inverse of the shearing transformation is not unique.
    /// assert_ne!(other_shear_inv1, shear_inv);
    /// assert_ne!(other_shear_inv2, shear_inv);
    /// assert_ne!(other_shear_inv3, shear_inv);
    /// ```
    #[inline]
    pub fn inverse(&self) -> Self {
        Self {
            shear_factor: -self.shear_factor,
            origin: self.origin,
            direction: self.direction,
            normal: self.normal,
        }
    }

    /// Apply the inverse of the shear transformation to a vector.
    ///
    /// For further discussion on the geometric character of the inverse of the
    /// shearing transformation, see [`Shear::inverse`].
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Shear2;
    /// #
    /// let shear_factor = 4_f64;
    /// let direction = Unit::from_value(Vector2::unit_x());
    /// let normal = Unit::from_value(Vector2::unit_y());
    /// let shear = Shear2::from_shear(shear_factor, &direction, &normal);
    /// let vertices = [
    ///     Vector2::new( 1_f64,  1_f64),
    ///     Vector2::new(-1_f64,  1_f64),
    ///     Vector2::new(-1_f64, -1_f64),
    ///     Vector2::new( 1_f64, -1_f64),
    /// ];
    /// let expected = [
    ///     Vector2::new( 1_f64 - shear_factor,  1_f64),
    ///     Vector2::new(-1_f64 - shear_factor,  1_f64),
    ///     Vector2::new(-1_f64 + shear_factor, -1_f64),
    ///     Vector2::new( 1_f64 + shear_factor, -1_f64),
    /// ];
    /// let result = vertices.map(|v| shear.inverse_apply_vector(&v));
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Shear3;
    /// # use core::f64;
    /// #
    /// let shear_factor = 8_f64;
    /// let direction = Unit::from_value(Vector3::new(1_f64 / f64::sqrt(2_f64), 1_f64 / f64::sqrt(2_f64), 0_f64));
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let shear = Shear3::from_shear(shear_factor, &direction, &normal);
    /// let vertices = [
    ///     Vector3::new( 1_f64,  1_f64,  1_f64),
    ///     Vector3::new(-1_f64,  1_f64,  1_f64),
    ///     Vector3::new(-1_f64, -1_f64,  1_f64),
    ///     Vector3::new( 1_f64, -1_f64,  1_f64),
    ///     Vector3::new( 1_f64,  1_f64, -1_f64),
    ///     Vector3::new(-1_f64,  1_f64, -1_f64),
    ///     Vector3::new(-1_f64, -1_f64, -1_f64),
    ///     Vector3::new( 1_f64, -1_f64, -1_f64),
    /// ];
    /// let expected = [
    ///     Vector3::new( 1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Vector3::new(-1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Vector3::new(-1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Vector3::new( 1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Vector3::new( 1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    ///     Vector3::new(-1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    ///     Vector3::new(-1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    ///     Vector3::new( 1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    /// ];
    /// let result = vertices.map(|v| shear.inverse_apply_vector(&v));
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn inverse_apply_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        let origin = self.origin.to_vector();
        let factor = self.shear_factor * (vector - origin).dot(&self.normal);

        vector - self.direction * factor
    }

    /// Apply the inverse of the shear transformation to a point.
    ///
    /// For further discussion on the geometric character of the inverse of the
    /// shearing transformation, see [`Shear::inverse`].
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Shear2;
    /// #
    /// let shear_factor = 4_f64;
    /// let direction = Unit::from_value(Vector2::unit_x());
    /// let normal = Unit::from_value(Vector2::unit_y());
    /// let shear = Shear2::from_shear(shear_factor, &direction, &normal);
    /// let vertices = [
    ///     Point2::new( 1_f64,  1_f64),
    ///     Point2::new(-1_f64,  1_f64),
    ///     Point2::new(-1_f64, -1_f64),
    ///     Point2::new( 1_f64, -1_f64),
    /// ];
    /// let expected = [
    ///     Point2::new( 1_f64 - shear_factor,  1_f64),
    ///     Point2::new(-1_f64 - shear_factor,  1_f64),
    ///     Point2::new(-1_f64 + shear_factor, -1_f64),
    ///     Point2::new( 1_f64 + shear_factor, -1_f64),
    /// ];
    /// let result = vertices.map(|p| shear.inverse_apply_point(&p));
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Shear3;
    /// # use core::f64;
    /// #
    /// let shear_factor = 8_f64;
    /// let direction = Unit::from_value(Vector3::new(1_f64 / f64::sqrt(2_f64), 1_f64 / f64::sqrt(2_f64), 0_f64));
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let shear = Shear3::from_shear(shear_factor, &direction, &normal);
    /// let vertices = [
    ///     Point3::new( 1_f64,  1_f64,  1_f64),
    ///     Point3::new(-1_f64,  1_f64,  1_f64),
    ///     Point3::new(-1_f64, -1_f64,  1_f64),
    ///     Point3::new( 1_f64, -1_f64,  1_f64),
    ///     Point3::new( 1_f64,  1_f64, -1_f64),
    ///     Point3::new(-1_f64,  1_f64, -1_f64),
    ///     Point3::new(-1_f64, -1_f64, -1_f64),
    ///     Point3::new( 1_f64, -1_f64, -1_f64),
    /// ];
    /// let expected = [
    ///     Point3::new( 1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Point3::new(-1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Point3::new(-1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Point3::new( 1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 - (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64),
    ///     Point3::new( 1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    ///     Point3::new(-1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor,  1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    ///     Point3::new(-1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),
    ///     Point3::new( 1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64 + (1_f64 / f64::sqrt(2_f64)) * shear_factor, -1_f64),  
    /// ];
    /// let result = vertices.map(|p| shear.inverse_apply_point(&p));
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn inverse_apply_point(&self, point: &Point<S, N>) -> Point<S, N> {
        let factor = self.shear_factor * (point - self.origin).dot(&self.normal);

        point - self.direction * factor
    }
}

impl<S, const N: usize> Shear<S, N>
where
    S: SimdScalarFloat,
{
    /// Construct a general shearing transformation in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`.
    ///
    /// # Parameters
    ///
    /// The shearing transformation constructor has the following parameters
    /// * `shear_factor`: The amount by which a point in a plane parallel to the shearing
    ///    plane gets sheared.
    /// * `direction`: The direction along which the shearing happens.
    /// * `normal`: The normal vector to the shearing plane.
    ///
    /// # Example (Two Dimensions)
    ///
    /// Shearing a rotated square parallel to the line `y == (1 / 2) * x` along the
    /// line `y == (1 / 2) * x`.
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Shear2;
    /// # use core::f64;
    /// #
    /// let shear_factor = 4_f64;
    /// let direction = Unit::from_value(Vector2::new(2_f64, 1_f64));
    /// let normal = Unit::from_value(Vector2::new(-1_f64, 2_f64));
    /// let shear = Shear2::from_shear(shear_factor, &direction, &normal);
    ///
    /// // The square's top and bottom sides run parallel to the line `y == (1 / 2) * x`.
    /// // The square's left and right sides run perpendicular to the line `y == (1 / 2) * x`.
    /// let vertices = [
    ///     Point2::new( 1_f64 / f64::sqrt(5_f64),  3_f64 / f64::sqrt(5_f64)),
    ///     Point2::new(-3_f64 / f64::sqrt(5_f64),  1_f64 / f64::sqrt(5_f64)),
    ///     Point2::new(-1_f64 / f64::sqrt(5_f64), -3_f64 / f64::sqrt(5_f64)),
    ///     Point2::new( 3_f64 / f64::sqrt(5_f64), -1_f64 / f64::sqrt(5_f64)),
    /// ];
    /// let expected = [
    ///     Point2::new(
    ///         (2_f64 / f64::sqrt(5_f64)) * (1_f64 + shear_factor) - 1_f64 / f64::sqrt(5_f64),
    ///         (1_f64 / f64::sqrt(5_f64)) * (1_f64 + shear_factor) + 2_f64 / f64::sqrt(5_f64),
    ///     ),
    ///     Point2::new(
    ///         (2_f64 / f64::sqrt(5_f64)) * (-1_f64 + shear_factor) - 1_f64 / f64::sqrt(5_f64),
    ///         (1_f64 / f64::sqrt(5_f64)) * (-1_f64 + shear_factor) + 2_f64 / f64::sqrt(5_f64),
    ///     ),
    ///     Point2::new(
    ///         (2_f64 / f64::sqrt(5_f64)) * (-1_f64 - shear_factor) + 1_f64 / f64::sqrt(5_f64),
    ///         (1_f64 / f64::sqrt(5_f64)) * (-1_f64 - shear_factor) - 2_f64 / f64::sqrt(5_f64),
    ///     ),
    ///     Point2::new(
    ///         (2_f64 / f64::sqrt(5_f64)) * (1_f64 - shear_factor) + 1_f64 / f64::sqrt(5_f64),
    ///         (1_f64 / f64::sqrt(5_f64)) * (1_f64 - shear_factor) - 2_f64 / f64::sqrt(5_f64),
    ///     ),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    ///
    /// let vertices_in_line = [
    ///     Point2::new( 1_f64 / f64::sqrt(5_f64),  1_f64 / (2_f64 * f64::sqrt(5_f64))),
    ///     Point2::new(-3_f64 / f64::sqrt(5_f64), -3_f64 / (2_f64 * f64::sqrt(5_f64))),
    ///     Point2::new(-1_f64 / f64::sqrt(5_f64), -1_f64 / (2_f64 * f64::sqrt(5_f64))),
    ///     Point2::new( 3_f64 / f64::sqrt(5_f64),  3_f64 / (2_f64 * f64::sqrt(5_f64))),
    ///     Point2::new( 0_f64, 0_f64),
    /// ];
    /// let expected_in_line = vertices_in_line;
    /// let result_in_line = vertices_in_line.map(|p| shear * p);
    ///
    /// assert_relative_eq!(result_in_line, expected_in_line, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Shear3;
    /// #
    /// let shear_factor = 8_f64;
    /// let direction = Unit::from_value(Vector3::unit_x());
    /// let normal = Unit::from_value(-Vector3::unit_y());
    /// let shear = Shear3::from_shear(shear_factor, &direction, &normal);
    ///
    /// let vertices = [
    ///     Point3::new( 1_f64,  1_f64,  1_f64),
    ///     Point3::new(-1_f64,  1_f64,  1_f64),
    ///     Point3::new(-1_f64, -1_f64,  1_f64),
    ///     Point3::new( 1_f64, -1_f64,  1_f64),
    ///     Point3::new( 1_f64,  1_f64, -1_f64),
    ///     Point3::new(-1_f64,  1_f64, -1_f64),
    ///     Point3::new(-1_f64, -1_f64, -1_f64),
    ///     Point3::new( 1_f64, -1_f64, -1_f64),
    /// ];
    /// let expected = [
    ///     Point3::new( 1_f64 - shear_factor,  1_f64,  1_f64),
    ///     Point3::new(-1_f64 - shear_factor,  1_f64,  1_f64),
    ///     Point3::new(-1_f64 + shear_factor, -1_f64,  1_f64),
    ///     Point3::new( 1_f64 + shear_factor, -1_f64,  1_f64),
    ///     Point3::new( 1_f64 - shear_factor,  1_f64, -1_f64),
    ///     Point3::new(-1_f64 - shear_factor,  1_f64, -1_f64),
    ///     Point3::new(-1_f64 + shear_factor, -1_f64, -1_f64),
    ///     Point3::new( 1_f64 + shear_factor, -1_f64, -1_f64),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Point3::new( 1_f64, 0_f64,  1_f64),
    ///     Point3::new(-1_f64, 0_f64,  1_f64),
    ///     Point3::new(-1_f64, 0_f64, -1_f64),
    ///     Point3::new( 1_f64, 0_f64, -1_f64),
    ///     Point3::new( 0_f64, 0_f64,  0_f64),
    /// ];
    /// // Points in the shearing plane don't move.
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|p| shear * p);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[inline]
    pub fn from_shear(shear_factor: S, direction: &Unit<Vector<S, N>>, normal: &Unit<Vector<S, N>>) -> Self {
        Self {
            shear_factor,
            origin: Point::origin(),
            direction: direction.into_inner(),
            normal: normal.into_inner(),
        }
    }

    /// Construct a general shearing transformation in three dimensions with respect to
    /// a plane passing through the origin `origin`, not necessarily `[0, 0, 0]`.
    ///
    /// # Parameters
    ///
    /// The shearing transformation constructor has four parameters
    /// * `origin`: The origin of the affine frame for the shearing transformation.
    /// * `shear_factor`: The amount by which a point in a plane parallel to the shearing
    ///    plane gets sheared.
    /// * `direction`: The direction along which the shearing happens in the shearing plane.
    /// * `normal`: The normal vector to the shearing plane.
    ///
    /// # Examples (Two Dimensions)
    ///
    /// Shearing along the **x-axis** with a non-zero origin on the **x-axis**.
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Shear2;
    /// #
    /// let shear_factor = 15_f64;
    /// let origin = Point2::new(-2_f64, 0_f64);
    /// let direction = Unit::from_value(Vector2::unit_x());
    /// let normal = Unit::from_value(Vector2::unit_y());
    /// let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
    /// let vertices = [
    ///     Point2::new( 1_f64,  1_f64),
    ///     Point2::new(-1_f64,  1_f64),
    ///     Point2::new(-1_f64, -1_f64),
    ///     Point2::new( 1_f64, -1_f64),
    /// ];
    /// let expected = [
    ///     Point2::new( 1_f64 + shear_factor,  1_f64),
    ///     Point2::new(-1_f64 + shear_factor,  1_f64),
    ///     Point2::new(-1_f64 - shear_factor, -1_f64),
    ///     Point2::new( 1_f64 - shear_factor, -1_f64),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    ///
    /// let vertices_in_line = [
    ///     Point2::new( 1_f64, 0_f64),
    ///     Point2::new(-1_f64, 0_f64),
    ///     Point2::new( 0_f64, 0_f64),
    /// ];
    /// let expected_in_line = vertices_in_line;
    /// let result_in_line = vertices_in_line.map(|p| shear * p);
    ///
    /// assert_relative_eq!(result_in_line, expected_in_line, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// Shearing along the line `y == (1 / 2) * x + 1` using the origin `(2, 2)`.
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Shear2;
    /// # use core::f64;
    /// #
    /// let shear_factor = 7_f64;
    /// let origin = Point2::new(2_f64, 2_f64);
    /// let direction = Unit::from_value(Vector2::new(2_f64, 1_f64));
    /// let normal = Unit::from_value(Vector2::new(-1_f64, 2_f64));
    /// let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
    ///
    /// // The square's top and bottom sides run parallel to the line `y == (1 / 2) * x + 1`.
    /// // The square's left and right sides run perpendicular to the line `y == (1 / 2) * x + 1`.
    /// let vertices = [
    ///     Point2::new( 1_f64 / f64::sqrt(5_f64),  3_f64 / f64::sqrt(5_f64) + 1_f64),
    ///     Point2::new(-3_f64 / f64::sqrt(5_f64),  1_f64 / f64::sqrt(5_f64) + 1_f64),
    ///     Point2::new(-1_f64 / f64::sqrt(5_f64), -3_f64 / f64::sqrt(5_f64) + 1_f64),
    ///     Point2::new( 3_f64 / f64::sqrt(5_f64), -1_f64 / f64::sqrt(5_f64) + 1_f64),
    /// ];
    /// let rotated_origin = Vector2::new(f64::sqrt(5_f64), 0_f64);
    /// let expected = [
    ///     Point2::new(
    ///          (1_f64 / f64::sqrt(5_f64)) + (2_f64 / f64::sqrt(5_f64)) * shear_factor,
    ///          (3_f64 / f64::sqrt(5_f64)) + (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
    ///     ),
    ///     Point2::new(
    ///         -(3_f64 / f64::sqrt(5_f64)) + (2_f64 / f64::sqrt(5_f64)) * shear_factor,
    ///          (1_f64 / f64::sqrt(5_f64))  + (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
    ///     ),
    ///     Point2::new(
    ///         -(1_f64 / f64::sqrt(5_f64)) - (2_f64 / f64::sqrt(5_f64)) * shear_factor,
    ///         -(3_f64 / f64::sqrt(5_f64)) - (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
    ///     ),
    ///     Point2::new(
    ///          (3_f64 / f64::sqrt(5_f64)) - (2_f64 / f64::sqrt(5_f64)) * shear_factor,
    ///         -(1_f64 / f64::sqrt(5_f64)) - (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
    ///     ),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    ///
    /// let vertices_in_plane = [
    ///     Point2::new( 1_f64 / f64::sqrt(5_f64),  1_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64),
    ///     Point2::new(-3_f64 / f64::sqrt(5_f64), -3_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64),
    ///     Point2::new(-1_f64 / f64::sqrt(5_f64), -1_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64),
    ///     Point2::new( 3_f64 / f64::sqrt(5_f64),  3_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64),
    ///     Point2::new( 0_f64, 1_f64),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|p| shear * p);
    ///
    /// assert_relative_eq!(result_in_plane, expected_in_plane, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Shear3;
    /// # use core::f64;
    /// #
    /// let shear_factor = 15_f64;
    /// let origin = Point3::origin();
    /// let direction = Unit::from_value(Vector3::new(
    ///     1_f64 / f64::sqrt(2_f64),
    ///     1_f64 / f64::sqrt(2_f64),
    ///     0_f64
    /// ));
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
    /// let vertices = [
    ///     Point3::new( 1_f64,  1_f64,  1_f64),
    ///     Point3::new(-1_f64,  1_f64,  1_f64),
    ///     Point3::new(-1_f64, -1_f64,  1_f64),
    ///     Point3::new( 1_f64, -1_f64,  1_f64),
    ///     Point3::new( 1_f64,  1_f64, -1_f64),
    ///     Point3::new(-1_f64,  1_f64, -1_f64),
    ///     Point3::new(-1_f64, -1_f64, -1_f64),
    ///     Point3::new( 1_f64, -1_f64, -1_f64),
    /// ];
    /// let expected = [
    ///     Point3::new( 1_f64 + shear_factor / f64::sqrt(2_f64),  1_f64 + shear_factor / f64::sqrt(2_f64),  1_f64),
    ///     Point3::new(-1_f64 + shear_factor / f64::sqrt(2_f64),  1_f64 + shear_factor / f64::sqrt(2_f64),  1_f64),
    ///     Point3::new(-1_f64 + shear_factor / f64::sqrt(2_f64), -1_f64 + shear_factor / f64::sqrt(2_f64),  1_f64),
    ///     Point3::new( 1_f64 + shear_factor / f64::sqrt(2_f64), -1_f64 + shear_factor / f64::sqrt(2_f64),  1_f64),
    ///     Point3::new( 1_f64 - shear_factor / f64::sqrt(2_f64),  1_f64 - shear_factor / f64::sqrt(2_f64), -1_f64),
    ///     Point3::new(-1_f64 - shear_factor / f64::sqrt(2_f64),  1_f64 - shear_factor / f64::sqrt(2_f64), -1_f64),
    ///     Point3::new(-1_f64 - shear_factor / f64::sqrt(2_f64), -1_f64 - shear_factor / f64::sqrt(2_f64), -1_f64),
    ///     Point3::new( 1_f64 - shear_factor / f64::sqrt(2_f64), -1_f64 - shear_factor / f64::sqrt(2_f64), -1_f64),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    ///
    /// let vertices_in_plane = [
    ///     Point3::new( 1_f64,  1_f64, 0_f64),
    ///     Point3::new(-1_f64,  1_f64, 0_f64),
    ///     Point3::new(-1_f64, -1_f64, 0_f64),
    ///     Point3::new( 1_f64, -1_f64, 0_f64),
    ///     Point3::new( 0_f64,  0_f64, 0_f64),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|p| shear * p);
    ///
    /// assert_relative_eq!(result_in_plane, expected_in_plane, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn from_affine_shear(shear_factor: S, origin: &Point<S, N>, direction: &Unit<Vector<S, N>>, normal: &Unit<Vector<S, N>>) -> Self {
        Self {
            shear_factor,
            origin: *origin,
            direction: direction.into_inner(),
            normal: normal.into_inner(),
        }
    }
}

impl<S, const N: usize> fmt::Display for Shear<S, N>
where
    S: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Shear{} [shear_factor={}, origin={}, direction={}, normal={}]",
            N, self.shear_factor, self.origin, self.direction, self.normal
        )
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for Shear<S, N>
where
    S: SimdScalarSigned,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Point<S, N>> for Shear<S, N>
where
    S: SimdScalarSigned,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for &Shear<S, N>
where
    S: SimdScalarSigned,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Point<S, N>> for &'b Shear<S, N>
where
    S: SimdScalarSigned,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &'a Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Vector<S, N>> for Shear<S, N>
where
    S: SimdScalarSigned,
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Vector<S, N>> for Shear<S, N>
where
    S: SimdScalarSigned,
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: &Vector<S, N>) -> Self::Output {
        self.apply_vector(other)
    }
}

impl<S, const N: usize> ops::Mul<Vector<S, N>> for &Shear<S, N>
where
    S: SimdScalarSigned,
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Vector<S, N>> for &'b Shear<S, N>
where
    S: SimdScalarSigned,
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: &'a Vector<S, N>) -> Self::Output {
        self.apply_vector(other)
    }
}

impl<S> Shear2<S>
where
    S: SimdScalarSigned,
{
    /// Construct a shearing transformation in two dimensions with respect to
    /// a line passing through the origin `[0, 0]`, using the **x-axis**
    /// as the shearing direction, and the **y-axis** as the normal vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Point2;
    /// # use cglinalg_transform::Shear2;
    /// #
    /// let shear_factor = 4_i32;
    /// let shear = Shear2::from_shear_xy(shear_factor);
    /// let vertices = [
    ///     Point2::new( 1_i32,  1_i32),
    ///     Point2::new(-1_i32,  1_i32),
    ///     Point2::new(-1_i32, -1_i32),
    ///     Point2::new( 1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Point2::new( 1_i32 + shear_factor,  1_i32),
    ///     Point2::new(-1_i32 + shear_factor,  1_i32),
    ///     Point2::new(-1_i32 - shear_factor, -1_i32),
    ///     Point2::new( 1_i32 - shear_factor, -1_i32),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_line = [
    ///     Point2::new( 1_i32, 0_i32),
    ///     Point2::new(-1_i32, 0_i32),
    ///     Point2::new( 0_i32, 0_i32),
    /// ];
    /// let expected_in_line = vertices_in_line;
    /// let result_in_line = vertices_in_line.map(|p| shear * p);
    ///
    /// assert_eq!(result_in_line, expected_in_line);
    /// ```
    #[inline]
    pub fn from_shear_xy(shear_factor: S) -> Self {
        Self {
            shear_factor,
            origin: Point2::origin(),
            direction: Vector2::unit_x(),
            normal: Vector2::unit_y(),
        }
    }

    /// Construct a shearing transformation in two dimensions with respect to
    /// a line passing through the origin `[0, 0]`, using the **y-axis**
    /// as the shearing direction, and the **x-axis** as the normal vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Point2;
    /// # use cglinalg_transform::Shear2;
    /// #
    /// let shear_factor = 4_i32;
    /// let shear = Shear2::from_shear_yx(shear_factor);
    /// let vertices = [
    ///     Point2::new( 1_i32,  1_i32),
    ///     Point2::new(-1_i32,  1_i32),
    ///     Point2::new(-1_i32, -1_i32),
    ///     Point2::new( 1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Point2::new( 1_i32,  1_i32 + shear_factor),
    ///     Point2::new(-1_i32,  1_i32 - shear_factor),
    ///     Point2::new(-1_i32, -1_i32 - shear_factor),
    ///     Point2::new( 1_i32, -1_i32 + shear_factor),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_line = [
    ///     Point2::new(0_i32,  1_i32),
    ///     Point2::new(0_i32, -1_i32),
    ///     Point2::new(0_i32,  0_i32),
    /// ];
    /// let expected_in_line = vertices_in_line;
    /// let result_in_line = vertices_in_line.map(|p| shear * p);
    ///
    /// assert_eq!(result_in_line, expected_in_line);
    /// ```
    #[inline]
    pub fn from_shear_yx(shear_factor: S) -> Self {
        Self {
            shear_factor,
            origin: Point2::origin(),
            direction: Vector2::unit_y(),
            normal: Vector2::unit_x(),
        }
    }
}

impl<S> Shear2<S>
where
    S: SimdScalarFloat,
{
    /// Convert a shear transformation to an affine matrix.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Shear2;
    /// #
    /// let shear_factor = 4_f64;
    /// let origin = Point2::new(0_f64, -3_f64);
    /// let direction = Unit::from_value(Vector2::unit_x());
    /// let normal = Unit::from_value(Vector2::unit_y());
    /// let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
    /// let expected = Matrix3x3::new(
    ///     1_f64,                0_f64, 0_f64,
    ///     shear_factor,         1_f64, 0_f64,
    ///     3_f64 * shear_factor, 0_f64, 1_f64,
    /// );
    /// let result = shear.to_affine_matrix();
    ///
    /// assert_eq!(result, expected);
    ///
    /// assert_relative_eq!(
    ///     result.trace(),
    ///     3_f64,
    ///     abs_diff_all <= f64::EPSILON,
    ///     relative_all <= f64::EPSILON,
    /// );
    /// assert_relative_eq!(
    ///     result.determinant(),
    ///     1_f64,
    ///     abs_diff_all <= f64::EPSILON,
    ///     relative_all <= f64::EPSILON,
    /// );
    /// ```
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix3x3<S> {
        let direction = Unit::from_value(self.direction);
        let normal = Unit::from_value(self.normal);

        Matrix3x3::from_affine_shear(self.shear_factor, &self.origin, &direction, &normal)
    }

    /// Convert a shear transformation into a generic affine transformation.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Shear2,
    /// #     Transform2,
    /// # };
    /// #
    /// let shear_factor = 4_f64;
    /// let origin = Point2::new(0_f64, -3_f64);
    /// let direction = Unit::from_value(Vector2::unit_x());
    /// let normal = Unit::from_value(Vector2::unit_y());
    /// let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
    /// let expected = Transform2::from_matrix_unchecked(Matrix3x3::new(
    ///     1_f64,                0_f64, 0_f64,
    ///     shear_factor,         1_f64, 0_f64,
    ///     3_f64 * shear_factor, 0_f64, 1_f64,
    /// ));
    /// let result = shear.to_transform();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_transform(&self) -> Transform2<S> {
        Transform2::from_matrix_unchecked(self.to_affine_matrix())
    }
}

impl<S> From<Shear2<S>> for Matrix3x3<S>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn from(shear: Shear2<S>) -> Matrix3x3<S> {
        shear.to_affine_matrix()
    }
}

impl<S> From<&Shear2<S>> for Matrix3x3<S>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn from(shear: &Shear2<S>) -> Matrix3x3<S> {
        shear.to_affine_matrix()
    }
}

impl<S> ops::Mul<Shear2<S>> for Shear2<S>
where
    S: SimdScalarFloat,
{
    type Output = Transform2<S>;

    #[inline]
    fn mul(self, other: Shear2<S>) -> Self::Output {
        let lhs = self.to_transform();
        let rhs = other.to_transform();

        lhs * rhs
    }
}

impl<S> ops::Mul<&Shear2<S>> for Shear2<S>
where
    S: SimdScalarFloat,
{
    type Output = Transform2<S>;

    #[inline]
    fn mul(self, other: &Shear2<S>) -> Self::Output {
        let lhs = self.to_transform();
        let rhs = other.to_transform();

        lhs * rhs
    }
}

impl<S> ops::Mul<Shear2<S>> for &Shear2<S>
where
    S: SimdScalarFloat,
{
    type Output = Transform2<S>;

    #[inline]
    fn mul(self, other: Shear2<S>) -> Self::Output {
        let lhs = self.to_transform();
        let rhs = other.to_transform();

        lhs * rhs
    }
}

impl<'a, 'b, S> ops::Mul<&'a Shear2<S>> for &'b Shear2<S>
where
    S: SimdScalarFloat,
{
    type Output = Transform2<S>;

    #[inline]
    fn mul(self, other: &'a Shear2<S>) -> Self::Output {
        let lhs = self.to_transform();
        let rhs = other.to_transform();

        lhs * rhs
    }
}

impl<S> Shear3<S>
where
    S: SimdScalarSigned,
{
    /// Construct a shearing transformation in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **x-axis**
    /// as the shearing direction, and the **y-axis** as the normal vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Point3;
    /// # use cglinalg_transform::Shear3;
    /// #
    /// let shear_factor = 8_i32;
    /// let shear = Shear3::from_shear_xy(shear_factor);
    /// let vertices = [
    ///     Point3::new( 1_i32,  1_i32,  1_i32),
    ///     Point3::new(-1_i32,  1_i32,  1_i32),
    ///     Point3::new(-1_i32, -1_i32,  1_i32),
    ///     Point3::new( 1_i32, -1_i32,  1_i32),
    ///     Point3::new( 1_i32,  1_i32, -1_i32),
    ///     Point3::new(-1_i32,  1_i32, -1_i32),
    ///     Point3::new(-1_i32, -1_i32, -1_i32),
    ///     Point3::new( 1_i32, -1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Point3::new( 1_i32 + shear_factor,  1_i32,  1_i32),
    ///     Point3::new(-1_i32 + shear_factor,  1_i32,  1_i32),
    ///     Point3::new(-1_i32 - shear_factor, -1_i32,  1_i32),
    ///     Point3::new( 1_i32 - shear_factor, -1_i32,  1_i32),
    ///     Point3::new( 1_i32 + shear_factor,  1_i32, -1_i32),
    ///     Point3::new(-1_i32 + shear_factor,  1_i32, -1_i32),
    ///     Point3::new(-1_i32 - shear_factor, -1_i32, -1_i32),
    ///     Point3::new( 1_i32 - shear_factor, -1_i32, -1_i32),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Point3::new( 1_i32, 0_i32,  1_i32),
    ///     Point3::new(-1_i32, 0_i32,  1_i32),
    ///     Point3::new(-1_i32, 0_i32, -1_i32),
    ///     Point3::new( 1_i32, 0_i32, -1_i32),
    ///     Point3::new( 0_i32, 0_i32,  0_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|p| shear * p);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[inline]
    pub fn from_shear_xy(shear_factor: S) -> Self {
        Self {
            shear_factor,
            origin: Point3::origin(),
            direction: Vector3::unit_x(),
            normal: Vector3::unit_y(),
        }
    }

    /// Construct a shearing transformation in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **x-axis**
    /// as the shearing direction, and the **y-axis** as the normal vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Point3;
    /// # use cglinalg_transform::Shear3;
    /// #
    /// let shear_factor = 8_i32;
    /// let shear = Shear3::from_shear_xz(shear_factor);
    /// let vertices = [
    ///     Point3::new( 1_i32,  1_i32,  1_i32),
    ///     Point3::new(-1_i32,  1_i32,  1_i32),
    ///     Point3::new(-1_i32, -1_i32,  1_i32),
    ///     Point3::new( 1_i32, -1_i32,  1_i32),
    ///     Point3::new( 1_i32,  1_i32, -1_i32),
    ///     Point3::new(-1_i32,  1_i32, -1_i32),
    ///     Point3::new(-1_i32, -1_i32, -1_i32),
    ///     Point3::new( 1_i32, -1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Point3::new( 1_i32 + shear_factor,  1_i32,  1_i32),
    ///     Point3::new(-1_i32 + shear_factor,  1_i32,  1_i32),
    ///     Point3::new(-1_i32 + shear_factor, -1_i32,  1_i32),
    ///     Point3::new( 1_i32 + shear_factor, -1_i32,  1_i32),
    ///     Point3::new( 1_i32 - shear_factor,  1_i32, -1_i32),
    ///     Point3::new(-1_i32 - shear_factor,  1_i32, -1_i32),
    ///     Point3::new(-1_i32 - shear_factor, -1_i32, -1_i32),
    ///     Point3::new( 1_i32 - shear_factor, -1_i32, -1_i32),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Point3::new( 1_i32,  1_i32, 0_i32),
    ///     Point3::new(-1_i32,  1_i32, 0_i32),
    ///     Point3::new(-1_i32, -1_i32, 0_i32),
    ///     Point3::new( 1_i32, -1_i32, 0_i32),
    ///     Point3::new( 0_i32,  0_i32, 0_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|p| shear * p);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[inline]
    pub fn from_shear_xz(shear_factor: S) -> Self {
        Self {
            shear_factor,
            origin: Point3::origin(),
            direction: Vector3::unit_x(),
            normal: Vector3::unit_z(),
        }
    }

    /// Construct a shearing transformation in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **y-axis**
    /// as the shearing direction, and the **x-axis** as the normal vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Point3;
    /// # use cglinalg_transform::Shear3;
    /// #
    /// let shear_factor = 8_i32;
    /// let shear = Shear3::from_shear_yx(shear_factor);
    /// let vertices = [
    ///     Point3::new( 1_i32,  1_i32,  1_i32),
    ///     Point3::new(-1_i32,  1_i32,  1_i32),
    ///     Point3::new(-1_i32, -1_i32,  1_i32),
    ///     Point3::new( 1_i32, -1_i32,  1_i32),
    ///     Point3::new( 1_i32,  1_i32, -1_i32),
    ///     Point3::new(-1_i32,  1_i32, -1_i32),
    ///     Point3::new(-1_i32, -1_i32, -1_i32),
    ///     Point3::new( 1_i32, -1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Point3::new( 1_i32,  1_i32 + shear_factor,  1_i32),
    ///     Point3::new(-1_i32,  1_i32 - shear_factor,  1_i32),
    ///     Point3::new(-1_i32, -1_i32 - shear_factor,  1_i32),
    ///     Point3::new( 1_i32, -1_i32 + shear_factor,  1_i32),
    ///     Point3::new( 1_i32,  1_i32 + shear_factor, -1_i32),
    ///     Point3::new(-1_i32,  1_i32 - shear_factor, -1_i32),
    ///     Point3::new(-1_i32, -1_i32 - shear_factor, -1_i32),
    ///     Point3::new( 1_i32, -1_i32 + shear_factor, -1_i32),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Point3::new(0_i32,  1_i32,  1_i32),
    ///     Point3::new(0_i32, -1_i32,  1_i32),
    ///     Point3::new(0_i32, -1_i32, -1_i32),
    ///     Point3::new(0_i32,  1_i32, -1_i32),
    ///     Point3::new(0_i32,  0_i32,  0_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|p| shear * p);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[inline]
    pub fn from_shear_yx(shear_factor: S) -> Self {
        Self {
            shear_factor,
            origin: Point3::origin(),
            direction: Vector3::unit_y(),
            normal: Vector3::unit_x(),
        }
    }

    /// Construct a shearing transformation in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **y-axis**
    /// as the shearing direction, and the **z-axis** as the normal vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Point3;
    /// # use cglinalg_transform::Shear3;
    /// #
    /// let shear_factor = 8_i32;
    /// let shear = Shear3::from_shear_yz(shear_factor);
    /// let vertices = [
    ///     Point3::new( 1_i32,  1_i32,  1_i32),
    ///     Point3::new(-1_i32,  1_i32,  1_i32),
    ///     Point3::new(-1_i32, -1_i32,  1_i32),
    ///     Point3::new( 1_i32, -1_i32,  1_i32),
    ///     Point3::new( 1_i32,  1_i32, -1_i32),
    ///     Point3::new(-1_i32,  1_i32, -1_i32),
    ///     Point3::new(-1_i32, -1_i32, -1_i32),
    ///     Point3::new( 1_i32, -1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Point3::new( 1_i32,  1_i32 + shear_factor,  1_i32),
    ///     Point3::new(-1_i32,  1_i32 + shear_factor,  1_i32),
    ///     Point3::new(-1_i32, -1_i32 + shear_factor,  1_i32),
    ///     Point3::new( 1_i32, -1_i32 + shear_factor,  1_i32),
    ///     Point3::new( 1_i32,  1_i32 - shear_factor, -1_i32),
    ///     Point3::new(-1_i32,  1_i32 - shear_factor, -1_i32),
    ///     Point3::new(-1_i32, -1_i32 - shear_factor, -1_i32),
    ///     Point3::new( 1_i32, -1_i32 - shear_factor, -1_i32),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Point3::new( 1_i32,  1_i32, 0_i32),
    ///     Point3::new(-1_i32,  1_i32, 0_i32),
    ///     Point3::new(-1_i32, -1_i32, 0_i32),
    ///     Point3::new( 1_i32, -1_i32, 0_i32),
    ///     Point3::new( 0_i32,  0_i32, 0_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|p| shear * p);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[inline]
    pub fn from_shear_yz(shear_factor: S) -> Self {
        Self {
            shear_factor,
            origin: Point3::origin(),
            direction: Vector3::unit_y(),
            normal: Vector3::unit_z(),
        }
    }

    /// Construct a shearing transformation in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **z-axis**
    /// as the shearing direction, and the **x-axis** as the normal vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Point3;
    /// # use cglinalg_transform::Shear3;
    /// #
    /// let shear_factor = 8_i32;
    /// let shear = Shear3::from_shear_zx(shear_factor);
    /// let vertices = [
    ///     Point3::new( 1_i32,  1_i32,  1_i32),
    ///     Point3::new(-1_i32,  1_i32,  1_i32),
    ///     Point3::new(-1_i32, -1_i32,  1_i32),
    ///     Point3::new( 1_i32, -1_i32,  1_i32),
    ///     Point3::new( 1_i32,  1_i32, -1_i32),
    ///     Point3::new(-1_i32,  1_i32, -1_i32),
    ///     Point3::new(-1_i32, -1_i32, -1_i32),
    ///     Point3::new( 1_i32, -1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Point3::new( 1_i32,  1_i32,  1_i32 + shear_factor),
    ///     Point3::new(-1_i32,  1_i32,  1_i32 - shear_factor),
    ///     Point3::new(-1_i32, -1_i32,  1_i32 - shear_factor),
    ///     Point3::new( 1_i32, -1_i32,  1_i32 + shear_factor),
    ///     Point3::new( 1_i32,  1_i32, -1_i32 + shear_factor),
    ///     Point3::new(-1_i32,  1_i32, -1_i32 - shear_factor),
    ///     Point3::new(-1_i32, -1_i32, -1_i32 - shear_factor),
    ///     Point3::new( 1_i32, -1_i32, -1_i32 + shear_factor),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Point3::new(0_i32,  1_i32,  1_i32),
    ///     Point3::new(0_i32, -1_i32,  1_i32),
    ///     Point3::new(0_i32, -1_i32, -1_i32),
    ///     Point3::new(0_i32,  1_i32, -1_i32),
    ///     Point3::new(0_i32,  0_i32,  0_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|p| shear * p);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[inline]
    pub fn from_shear_zx(shear_factor: S) -> Self {
        Self {
            shear_factor,
            origin: Point3::origin(),
            direction: Vector3::unit_z(),
            normal: Vector3::unit_x(),
        }
    }

    /// Construct a shearing transformation in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **z-axis**
    /// as the shearing direction, and the **y-axis** as the normal vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Point3;
    /// # use cglinalg_transform::Shear3;
    /// #
    /// let shear_factor = 8_i32;
    /// let shear = Shear3::from_shear_zy(shear_factor);
    /// let vertices = [
    ///     Point3::new( 1_i32,  1_i32,  1_i32),
    ///     Point3::new(-1_i32,  1_i32,  1_i32),
    ///     Point3::new(-1_i32, -1_i32,  1_i32),
    ///     Point3::new( 1_i32, -1_i32,  1_i32),
    ///     Point3::new( 1_i32,  1_i32, -1_i32),
    ///     Point3::new(-1_i32,  1_i32, -1_i32),
    ///     Point3::new(-1_i32, -1_i32, -1_i32),
    ///     Point3::new( 1_i32, -1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Point3::new( 1_i32,  1_i32,  1_i32 + shear_factor),
    ///     Point3::new(-1_i32,  1_i32,  1_i32 + shear_factor),
    ///     Point3::new(-1_i32, -1_i32,  1_i32 - shear_factor),
    ///     Point3::new( 1_i32, -1_i32,  1_i32 - shear_factor),
    ///     Point3::new( 1_i32,  1_i32, -1_i32 + shear_factor),
    ///     Point3::new(-1_i32,  1_i32, -1_i32 + shear_factor),
    ///     Point3::new(-1_i32, -1_i32, -1_i32 - shear_factor),
    ///     Point3::new( 1_i32, -1_i32, -1_i32 - shear_factor),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Point3::new( 1_i32, 0_i32,  1_i32),
    ///     Point3::new(-1_i32, 0_i32,  1_i32),
    ///     Point3::new(-1_i32, 0_i32, -1_i32),
    ///     Point3::new( 1_i32, 0_i32, -1_i32),
    ///     Point3::new( 0_i32, 0_i32,  0_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|p| shear * p);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[inline]
    pub fn from_shear_zy(shear_factor: S) -> Self {
        Self {
            shear_factor,
            origin: Point3::origin(),
            direction: Vector3::unit_z(),
            normal: Vector3::unit_y(),
        }
    }
}

impl<S> Shear3<S>
where
    S: SimdScalarFloat,
{
    /// Convert a shear transformation to an affine matrix.
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Shear3;
    /// #
    /// let shear_factor = 8_f64;
    /// let origin = Point3::new(3_f64, 3_f64, -3_f64);
    /// let direction = Unit::from_value(Vector3::unit_x());
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
    /// let expected = Matrix4x4::new(
    ///     1_f64,                0_f64, 0_f64, 0_f64,
    ///     0_f64,                1_f64, 0_f64, 0_f64,
    ///     shear_factor,         0_f64, 1_f64, 0_f64,
    ///     3_f64 * shear_factor, 0_f64, 0_f64, 1_f64,
    /// );
    /// let result = shear.to_affine_matrix();
    ///
    /// assert_eq!(result, expected);
    ///
    /// assert_relative_eq!(result.trace(), 4_f64, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result.determinant(), 1_f64, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix4x4<S> {
        let direction = Unit::from_value(self.direction);
        let normal = Unit::from_value(self.normal);

        Matrix4x4::from_affine_shear(self.shear_factor, &self.origin, &direction, &normal)
    }

    /// Convert a shear transformation into a generic affine transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Shear3,
    /// #     Transform3,
    /// # };
    /// #
    /// let shear_factor = 8_f64;
    /// let origin = Point3::new(3_f64, 3_f64, -3_f64);
    /// let direction = Unit::from_value(Vector3::unit_x());
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
    /// let expected = Transform3::from_matrix_unchecked(Matrix4x4::new(
    ///     1_f64,                0_f64, 0_f64, 0_f64,
    ///     0_f64,                1_f64, 0_f64, 0_f64,
    ///     shear_factor,         0_f64, 1_f64, 0_f64,
    ///     3_f64 * shear_factor, 0_f64, 0_f64, 1_f64,
    /// ));
    /// let result = shear.to_transform();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_transform(&self) -> Transform3<S> {
        Transform3::from_matrix_unchecked(self.to_affine_matrix())
    }
}

impl<S> From<Shear3<S>> for Matrix4x4<S>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn from(shear: Shear3<S>) -> Matrix4x4<S> {
        shear.to_affine_matrix()
    }
}

impl<S> From<&Shear3<S>> for Matrix4x4<S>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn from(shear: &Shear3<S>) -> Matrix4x4<S> {
        shear.to_affine_matrix()
    }
}

impl<S> ops::Mul<Shear3<S>> for Shear3<S>
where
    S: SimdScalarFloat,
{
    type Output = Transform3<S>;

    #[inline]
    fn mul(self, other: Shear3<S>) -> Self::Output {
        let lhs = self.to_transform();
        let rhs = other.to_transform();

        lhs * rhs
    }
}

impl<S> ops::Mul<&Shear3<S>> for Shear3<S>
where
    S: SimdScalarFloat,
{
    type Output = Transform3<S>;

    #[inline]
    fn mul(self, other: &Shear3<S>) -> Self::Output {
        let lhs = self.to_transform();
        let rhs = other.to_transform();

        lhs * rhs
    }
}

impl<S> ops::Mul<Shear3<S>> for &Shear3<S>
where
    S: SimdScalarFloat,
{
    type Output = Transform3<S>;

    #[inline]
    fn mul(self, other: Shear3<S>) -> Self::Output {
        let lhs = self.to_transform();
        let rhs = other.to_transform();

        lhs * rhs
    }
}

impl<'a, 'b, S> ops::Mul<&'a Shear3<S>> for &'b Shear3<S>
where
    S: SimdScalarFloat,
{
    type Output = Transform3<S>;

    #[inline]
    fn mul(self, other: &'a Shear3<S>) -> Self::Output {
        let lhs = self.to_transform();
        let rhs = other.to_transform();

        lhs * rhs
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ShearTol<S, const N: usize> {
    shear_factor: S,
    origin: Vector<S, N>,
    direction: Vector<S, N>,
    normal: Vector<S, N>,
}

impl<S, const N: usize> ShearTol<S, N> {
    #[inline]
    pub const fn from_parts(shear_factor: S, origin: Vector<S, N>, direction: Vector<S, N>, normal: Vector<S, N>) -> Self {
        Self {
            shear_factor,
            origin,
            direction,
            normal,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ShearDiff<S, const N: usize> {
    shear_factor: S,
    origin: Vector<S, N>,
    direction: Vector<S, N>,
    normal: Vector<S, N>,
}

impl<S, const N: usize> ShearDiff<S, N> {
    #[inline]
    const fn from_parts(shear_factor: S, origin: Vector<S, N>, direction: Vector<S, N>, normal: Vector<S, N>) -> Self {
        Self {
            shear_factor,
            origin,
            direction,
            normal,
        }
    }
}

impl<S, const N: usize> approx_cmp::AbsDiffEq for Shear<S, N>
where
    S: SimdScalarFloat,
{
    type Tolerance = ShearTol<<S as approx_cmp::AbsDiffEq>::Tolerance, N>;

    #[inline]
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> bool {
        let lhs_shear_factor = &self.shear_factor();
        let rhs_shear_factor = &other.shear_factor();
        let lhs_origin = &self.origin();
        let rhs_origin = &other.origin();
        let lhs_direction = &self.direction();
        let rhs_direction = &other.direction();
        let lhs_normal = &self.normal();
        let rhs_normal = &other.normal();

        approx_cmp::AbsDiffEq::abs_diff_eq(lhs_shear_factor, rhs_shear_factor, &max_abs_diff.shear_factor)
            && approx_cmp::AbsDiffEq::abs_diff_eq(lhs_origin, rhs_origin, &max_abs_diff.origin)
            && approx_cmp::AbsDiffEq::abs_diff_eq(lhs_direction, rhs_direction, &max_abs_diff.direction)
            && approx_cmp::AbsDiffEq::abs_diff_eq(lhs_normal, rhs_normal, &max_abs_diff.normal)
    }
}

impl<S, const N: usize> approx_cmp::AbsDiffAllEq for Shear<S, N>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::AbsDiffAllEq>::AllTolerance;

    #[inline]
    fn abs_diff_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> bool {
        let lhs_shear_factor = &self.shear_factor();
        let rhs_shear_factor = &other.shear_factor();
        let lhs_origin = &self.origin();
        let rhs_origin = &other.origin();
        let lhs_direction = &self.direction();
        let rhs_direction = &other.direction();
        let lhs_normal = &self.normal();
        let rhs_normal = &other.normal();

        approx_cmp::AbsDiffAllEq::abs_diff_all_eq(lhs_shear_factor, rhs_shear_factor, max_abs_diff)
            && approx_cmp::AbsDiffAllEq::abs_diff_all_eq(lhs_origin, rhs_origin, max_abs_diff)
            && approx_cmp::AbsDiffAllEq::abs_diff_all_eq(lhs_direction, rhs_direction, max_abs_diff)
            && approx_cmp::AbsDiffAllEq::abs_diff_all_eq(lhs_normal, rhs_normal, max_abs_diff)
    }
}

impl<S, const N: usize> approx_cmp::AssertAbsDiffEq for Shear<S, N>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = ShearDiff<<S as approx_cmp::AssertAbsDiffEq>::DebugAbsDiff, N>;
    type DebugTolerance = ShearTol<<S as approx_cmp::AssertAbsDiffEq>::DebugTolerance, N>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertAbsDiffEq::debug_abs_diff(lhs, rhs)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertAbsDiffEq::debug_abs_diff(lhs, rhs)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertAbsDiffEq::debug_abs_diff(lhs, rhs)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertAbsDiffEq::debug_abs_diff(lhs, rhs)
        };

        ShearDiff::from_parts(shear_factor, origin, direction, normal)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertAbsDiffEq::debug_abs_diff_tolerance(lhs, rhs, &max_abs_diff.shear_factor)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertAbsDiffEq::debug_abs_diff_tolerance(lhs, rhs, &max_abs_diff.origin)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertAbsDiffEq::debug_abs_diff_tolerance(lhs, rhs, &max_abs_diff.direction)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertAbsDiffEq::debug_abs_diff_tolerance(lhs, rhs, &max_abs_diff.normal)
        };

        ShearTol::from_parts(shear_factor, origin, direction, normal)
    }
}

impl<S, const N: usize> approx_cmp::AssertAbsDiffAllEq for Shear<S, N>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = ShearTol<<S as approx_cmp::AssertAbsDiffAllEq>::AllDebugTolerance, N>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertAbsDiffAllEq::debug_abs_diff_all_tolerance(lhs, rhs, max_abs_diff)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertAbsDiffAllEq::debug_abs_diff_all_tolerance(lhs, rhs, max_abs_diff)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertAbsDiffAllEq::debug_abs_diff_all_tolerance(lhs, rhs, max_abs_diff)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertAbsDiffAllEq::debug_abs_diff_all_tolerance(lhs, rhs, max_abs_diff)
        };

        ShearTol::from_parts(shear_factor, origin, direction, normal)
    }
}

impl<S, const N: usize> approx_cmp::RelativeEq for Shear<S, N>
where
    S: SimdScalarFloat,
{
    type Tolerance = ShearTol<<S as approx_cmp::RelativeEq>::Tolerance, N>;

    #[inline]
    fn relative_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_relative: &Self::Tolerance) -> bool {
        let lhs_shear_factor = &self.shear_factor();
        let rhs_shear_factor = &other.shear_factor();
        let lhs_origin = &self.origin();
        let rhs_origin = &other.origin();
        let lhs_direction = &self.direction();
        let rhs_direction = &other.direction();
        let lhs_normal = &self.normal();
        let rhs_normal = &other.normal();

        approx_cmp::RelativeEq::relative_eq(
            lhs_shear_factor,
            rhs_shear_factor,
            &max_abs_diff.shear_factor,
            &max_relative.shear_factor,
        ) && approx_cmp::RelativeEq::relative_eq(lhs_origin, rhs_origin, &max_abs_diff.origin, &max_relative.origin)
            && approx_cmp::RelativeEq::relative_eq(lhs_direction, rhs_direction, &max_abs_diff.direction, &max_relative.direction)
            && approx_cmp::RelativeEq::relative_eq(lhs_normal, rhs_normal, &max_abs_diff.normal, &max_relative.normal)
    }
}

impl<S, const N: usize> approx_cmp::RelativeAllEq for Shear<S, N>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::RelativeAllEq>::AllTolerance;

    #[inline]
    fn relative_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_relative: &Self::AllTolerance) -> bool {
        let lhs_shear_factor = &self.shear_factor();
        let rhs_shear_factor = &other.shear_factor();
        let lhs_origin = &self.origin();
        let rhs_origin = &other.origin();
        let lhs_direction = &self.direction();
        let rhs_direction = &other.direction();
        let lhs_normal = &self.normal();
        let rhs_normal = &other.normal();

        approx_cmp::RelativeAllEq::relative_all_eq(lhs_shear_factor, rhs_shear_factor, max_abs_diff, max_relative)
            && approx_cmp::RelativeAllEq::relative_all_eq(lhs_origin, rhs_origin, max_abs_diff, max_relative)
            && approx_cmp::RelativeAllEq::relative_all_eq(lhs_direction, rhs_direction, max_abs_diff, max_relative)
            && approx_cmp::RelativeAllEq::relative_all_eq(lhs_normal, rhs_normal, max_abs_diff, max_relative)
    }
}

impl<S, const N: usize> approx_cmp::AssertRelativeEq for Shear<S, N>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = ShearDiff<<S as approx_cmp::AssertRelativeEq>::DebugAbsDiff, N>;
    type DebugTolerance = ShearTol<<S as approx_cmp::AssertRelativeEq>::DebugTolerance, N>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertRelativeEq::debug_abs_diff(lhs, rhs)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertRelativeEq::debug_abs_diff(lhs, rhs)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertRelativeEq::debug_abs_diff(lhs, rhs)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertRelativeEq::debug_abs_diff(lhs, rhs)
        };

        ShearDiff::from_parts(shear_factor, origin, direction, normal)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertRelativeEq::debug_abs_diff_tolerance(lhs, rhs, &max_abs_diff.shear_factor)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertRelativeEq::debug_abs_diff_tolerance(lhs, rhs, &max_abs_diff.origin)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertRelativeEq::debug_abs_diff_tolerance(lhs, rhs, &max_abs_diff.direction)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertRelativeEq::debug_abs_diff_tolerance(lhs, rhs, &max_abs_diff.normal)
        };

        ShearTol::from_parts(shear_factor, origin, direction, normal)
    }

    #[inline]
    fn debug_relative_tolerance(&self, other: &Self, max_relative: &Self::Tolerance) -> Self::DebugTolerance {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertRelativeEq::debug_relative_tolerance(lhs, rhs, &max_relative.shear_factor)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertRelativeEq::debug_relative_tolerance(lhs, rhs, &max_relative.origin)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertRelativeEq::debug_relative_tolerance(lhs, rhs, &max_relative.direction)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertRelativeEq::debug_relative_tolerance(lhs, rhs, &max_relative.normal)
        };

        ShearTol::from_parts(shear_factor, origin, direction, normal)
    }
}

impl<S, const N: usize> approx_cmp::AssertRelativeAllEq for Shear<S, N>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = ShearTol<<S as approx_cmp::AssertRelativeAllEq>::AllDebugTolerance, N>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertRelativeAllEq::debug_abs_diff_all_tolerance(lhs, rhs, max_abs_diff)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertRelativeAllEq::debug_abs_diff_all_tolerance(lhs, rhs, max_abs_diff)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertRelativeAllEq::debug_abs_diff_all_tolerance(lhs, rhs, max_abs_diff)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertRelativeAllEq::debug_abs_diff_all_tolerance(lhs, rhs, max_abs_diff)
        };

        ShearTol::from_parts(shear_factor, origin, direction, normal)
    }

    #[inline]
    fn debug_relative_all_tolerance(&self, other: &Self, max_relative: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertRelativeAllEq::debug_relative_all_tolerance(lhs, rhs, max_relative)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertRelativeAllEq::debug_relative_all_tolerance(lhs, rhs, max_relative)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertRelativeAllEq::debug_relative_all_tolerance(lhs, rhs, max_relative)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertRelativeAllEq::debug_relative_all_tolerance(lhs, rhs, max_relative)
        };

        ShearTol::from_parts(shear_factor, origin, direction, normal)
    }
}

impl<S, const N: usize> approx_cmp::UlpsEq for Shear<S, N>
where
    S: SimdScalarFloat,
{
    type Tolerance = ShearTol<<S as approx_cmp::UlpsEq>::Tolerance, N>;
    type UlpsTolerance = ShearTol<<S as approx_cmp::UlpsEq>::UlpsTolerance, N>;

    fn ulps_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_ulps: &Self::UlpsTolerance) -> bool {
        let lhs_shear_factor = &self.shear_factor();
        let rhs_shear_factor = &other.shear_factor();
        let lhs_origin = &self.origin();
        let rhs_origin = &other.origin();
        let lhs_direction = &self.direction();
        let rhs_direction = &other.direction();
        let lhs_normal = &self.normal();
        let rhs_normal = &other.normal();

        approx_cmp::UlpsEq::ulps_eq(
            lhs_shear_factor,
            rhs_shear_factor,
            &max_abs_diff.shear_factor,
            &max_ulps.shear_factor,
        ) && approx_cmp::UlpsEq::ulps_eq(lhs_origin, rhs_origin, &max_abs_diff.origin, &max_ulps.origin)
            && approx_cmp::UlpsEq::ulps_eq(lhs_direction, rhs_direction, &max_abs_diff.direction, &max_ulps.direction)
            && approx_cmp::UlpsEq::ulps_eq(lhs_normal, rhs_normal, &max_abs_diff.normal, &max_ulps.normal)
    }
}

impl<S, const N: usize> approx_cmp::UlpsAllEq for Shear<S, N>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::UlpsAllEq>::AllTolerance;
    type AllUlpsTolerance = <S as approx_cmp::UlpsAllEq>::AllUlpsTolerance;

    #[inline]
    fn ulps_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_ulps: &Self::AllUlpsTolerance) -> bool {
        let lhs_shear_factor = &self.shear_factor();
        let rhs_shear_factor = &other.shear_factor();
        let lhs_origin = &self.origin();
        let rhs_origin = &other.origin();
        let lhs_direction = &self.direction();
        let rhs_direction = &other.direction();
        let lhs_normal = &self.normal();
        let rhs_normal = &other.normal();

        approx_cmp::UlpsAllEq::ulps_all_eq(lhs_shear_factor, rhs_shear_factor, max_abs_diff, max_ulps)
            && approx_cmp::UlpsAllEq::ulps_all_eq(lhs_origin, rhs_origin, max_abs_diff, max_ulps)
            && approx_cmp::UlpsAllEq::ulps_all_eq(lhs_direction, rhs_direction, max_abs_diff, max_ulps)
            && approx_cmp::UlpsAllEq::ulps_all_eq(lhs_normal, rhs_normal, max_abs_diff, max_ulps)
    }
}

impl<S, const N: usize> approx_cmp::AssertUlpsEq for Shear<S, N>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = ShearDiff<<S as approx_cmp::AssertUlpsEq>::DebugAbsDiff, N>;
    type DebugUlpsDiff = ShearDiff<<S as approx_cmp::AssertUlpsEq>::DebugUlpsDiff, N>;
    type DebugTolerance = ShearTol<<S as approx_cmp::AssertUlpsEq>::DebugTolerance, N>;
    type DebugUlpsTolerance = ShearTol<<S as approx_cmp::AssertUlpsEq>::DebugUlpsTolerance, N>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertUlpsEq::debug_abs_diff(lhs, rhs)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertUlpsEq::debug_abs_diff(lhs, rhs)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertUlpsEq::debug_abs_diff(lhs, rhs)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertUlpsEq::debug_abs_diff(lhs, rhs)
        };

        ShearDiff::from_parts(shear_factor, origin, direction, normal)
    }

    #[inline]
    fn debug_ulps_diff(&self, other: &Self) -> Self::DebugUlpsDiff {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertUlpsEq::debug_ulps_diff(lhs, rhs)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertUlpsEq::debug_ulps_diff(lhs, rhs)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertUlpsEq::debug_ulps_diff(lhs, rhs)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertUlpsEq::debug_ulps_diff(lhs, rhs)
        };

        ShearDiff::from_parts(shear_factor, origin, direction, normal)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertUlpsEq::debug_abs_diff_tolerance(lhs, rhs, &max_abs_diff.shear_factor)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertUlpsEq::debug_abs_diff_tolerance(lhs, rhs, &max_abs_diff.origin)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertUlpsEq::debug_abs_diff_tolerance(lhs, rhs, &max_abs_diff.direction)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertUlpsEq::debug_abs_diff_tolerance(lhs, rhs, &max_abs_diff.normal)
        };

        ShearTol::from_parts(shear_factor, origin, direction, normal)
    }

    #[inline]
    fn debug_ulps_tolerance(&self, other: &Self, max_ulps: &Self::UlpsTolerance) -> Self::DebugUlpsTolerance {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertUlpsEq::debug_ulps_tolerance(lhs, rhs, &max_ulps.shear_factor)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertUlpsEq::debug_ulps_tolerance(lhs, rhs, &max_ulps.origin)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertUlpsEq::debug_ulps_tolerance(lhs, rhs, &max_ulps.direction)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertUlpsEq::debug_ulps_tolerance(lhs, rhs, &max_ulps.normal)
        };

        ShearTol::from_parts(shear_factor, origin, direction, normal)
    }
}

impl<S, const N: usize> approx_cmp::AssertUlpsAllEq for Shear<S, N>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = ShearTol<<S as approx_cmp::AssertUlpsAllEq>::AllDebugTolerance, N>;
    type AllDebugUlpsTolerance = ShearTol<<S as approx_cmp::AssertUlpsAllEq>::AllDebugUlpsTolerance, N>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertUlpsAllEq::debug_abs_diff_all_tolerance(lhs, rhs, max_abs_diff)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertUlpsAllEq::debug_abs_diff_all_tolerance(lhs, rhs, max_abs_diff)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertUlpsAllEq::debug_abs_diff_all_tolerance(lhs, rhs, max_abs_diff)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertUlpsAllEq::debug_abs_diff_all_tolerance(lhs, rhs, max_abs_diff)
        };

        ShearTol::from_parts(shear_factor, origin, direction, normal)
    }

    #[inline]
    fn debug_ulps_all_tolerance(&self, other: &Self, max_ulps: &Self::AllUlpsTolerance) -> Self::AllDebugUlpsTolerance {
        let shear_factor = {
            let lhs = &self.shear_factor();
            let rhs = &other.shear_factor();
            approx_cmp::AssertUlpsAllEq::debug_ulps_all_tolerance(lhs, rhs, max_ulps)
        };
        let origin = {
            let lhs = &self.origin();
            let rhs = &other.origin();
            approx_cmp::AssertUlpsAllEq::debug_ulps_all_tolerance(lhs, rhs, max_ulps)
        };
        let direction = {
            let lhs = &self.direction();
            let rhs = &other.direction();
            approx_cmp::AssertUlpsAllEq::debug_ulps_all_tolerance(lhs, rhs, max_ulps)
        };
        let normal = {
            let lhs = &self.normal();
            let rhs = &other.normal();
            approx_cmp::AssertUlpsAllEq::debug_ulps_all_tolerance(lhs, rhs, max_ulps)
        };

        ShearTol::from_parts(shear_factor, origin, direction, normal)
    }
}
