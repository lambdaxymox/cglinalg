use cglinalg_core::{
    SimdScalarFloat,
    Matrix3x3,
    Matrix4x4,
    Point2,
    Point3,
    Vector2,
    Vector3,
    Radians,
    Angle,
    Unit,
};
use crate::rotation::{
    Rotation2,
    Rotation3,
};
use crate::translation::{
    Translation2,
    Translation3,
};
use crate::transform::{
    Transform2,
    Transform3,
};

use core::fmt;
use core::ops;


/// A two-dimensional isometry (i.e. rigid body transformation) is a 
/// transformation whose motion does not distort the shape of an object. 
///
/// In terms of transformations, an isometry is a combination of 
/// a rotation and a translation. Rigid body transformations preserve the lengths
/// of vectors, hence the name isometry. In terms of transforming points and 
/// vectors, an isometry applies the rotation, followed by the translation.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Isometry2<S> {
    /// The rotation component of the isometry.
    pub(crate) rotation: Rotation2<S>,
    /// The translation component of the isometry.
    pub(crate) translation: Translation2<S>,
}

impl<S> Isometry2<S> 
where 
    S: SimdScalarFloat 
{
    /// Construct a new isometry directly from a translation and a rotation.
    #[inline]
    pub const fn from_parts(translation: &Translation2<S>, rotation: &Rotation2<S>) -> Self {
        Self {
            rotation: *rotation,
            translation: *translation,
        }
    }

    /// Construct a new isometry from a translation.
    #[inline]
    pub fn from_translation(translation: &Translation2<S>) -> Self {
        Self::from_parts(translation, &Rotation2::identity())
    }

    /// Construct a new isometry from a rotation.
    #[inline]
    pub fn from_rotation(rotation: &Rotation2<S>) -> Self {
        Self::from_parts(&Translation2::identity(), rotation)
    }

    /// Construct a new isometry from a rotation angle and a displacement vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Angle,
    /// #     Degrees, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// #     Translation2,
    /// #     Rotation2,
    /// # };
    /// #
    /// let angle = Degrees(72_f64);
    /// let distance = Vector2::new(1_f64, 2_f64);
    /// let translation = Translation2::from_vector(&distance);
    /// let rotation = Rotation2::from_angle(angle);
    /// let expected = Isometry2::from_parts(&translation, &rotation);
    /// let result = Isometry2::from_angle_translation(angle, &distance);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_angle_translation<A: Into<Radians<S>>>(angle: A, distance: &Vector2<S>) -> Self
    {
        Self {
            rotation: Rotation2::from_angle(angle),
            translation: Translation2::from_vector(distance),
        }
    }

    /// Construct an isometry that is a pure rotation by an angle `angle`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Degrees,
    /// #     Point2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let isometry = Isometry2::from_angle(Degrees(45_f64));
    /// let point = Point2::new(f64::sqrt(8_f64), f64::sqrt(8_f64));
    /// let expected = Point2::new(0_f64, 4_f64);
    /// let result = isometry.transform_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn from_angle<A: Into<Radians<S>>>(angle: A) -> Self {
        let translation = Translation2::identity();
        let rotation = Rotation2::from_angle(angle);
        
        Self::from_parts(&translation, &rotation)
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between two unit vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
    /// #     Unit, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// # };
    /// #
    /// let point = Point2::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64);
    /// let vector1 = Unit::from_value(Vector2::unit_y());
    /// let vector2 = Unit::from_value(Vector2::unit_x());
    /// let isometry = Isometry2::rotation_between_axis(&vector1, &vector2);
    /// let expected = Point2::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
    /// let result = isometry.transform_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn rotation_between_axis(a: &Unit<Vector2<S>>, b: &Unit<Vector2<S>>) -> Self {
        let unit_a = a.as_ref();
        let unit_b = b.as_ref();
        let cos_angle = unit_a.dot(unit_b);
        let sin_angle = unit_a.x * unit_b.y - unit_a.y * unit_b.x;

        Self::from_angle(Radians::atan2(sin_angle, cos_angle))
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// # };
    /// #
    /// let point = Point2::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64);
    /// let vector1 = 3_f64 * Vector2::unit_y();
    /// let vector2 = 6_f64 * Vector2::unit_x();
    /// let isometry = Isometry2::rotation_between(&vector1, &vector2);
    /// let expected = Point2::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
    /// let result = isometry.transform_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn rotation_between(a: &Vector2<S>, b: &Vector2<S>) -> Self {
        if let (Some(unit_a), Some(unit_b)) = (
            Unit::try_from_value(*a, S::zero()), 
            Unit::try_from_value(*b, S::zero()))
        {
            Self::rotation_between_axis(&unit_a, &unit_b)
        } else {
            Self::identity()
        }
    }

    /// Convert an isometry to a generic transformation.
    #[inline]
    pub fn to_transform(&self) -> Transform2<S> {
        let matrix = self.to_affine_matrix();
        Transform2::from_specialized(matrix)
    }

    /// Convert an isometry to an equivalent affine transformation matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector2,
    /// #     Degrees, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,    
    /// # };
    /// #
    /// let angle = Degrees(90_f64);
    /// let distance = Vector2::new(2_f64, 3_f64);
    /// let isometry = Isometry2::from_angle_translation(angle, &distance);
    /// let expected = Matrix3x3::new(
    ///      0_f64, 1_f64, 0_f64,
    ///     -1_f64, 0_f64, 0_f64,
    ///      2_f64, 3_f64, 1_f64
    /// );
    /// let result = isometry.to_affine_matrix();
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix3x3<S> {
        let zero = S::zero();
        let one = S::one();
        let rotation_matrix = self.rotation.matrix();
        let translation = self.translation.as_ref();

        Matrix3x3::new(
            rotation_matrix.c0r0, rotation_matrix.c0r1, zero,
            rotation_matrix.c1r0, rotation_matrix.c1r1, zero,
            translation[0],       translation[1],       one
        )
    }
    
    /// Get the rotation part of the isometry.
    #[inline]
    pub const fn rotation(&self) -> &Rotation2<S> {
        &self.rotation
    }

    /// Get the translation part of the isometry.
    #[inline]
    pub const fn translation(&self) -> &Translation2<S> {
        &self.translation
    }

    /// Construct the inverse isometry of an isometry.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Degrees,
    /// #     Point2,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let angle = Degrees(90_f64);
    /// let distance = Vector2::new(2_f64, 3_f64);
    /// let isometry = Isometry2::from_angle_translation(angle, &distance);
    /// let isometry_inv = isometry.inverse();
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = point;
    /// let result = isometry_inv * (isometry * point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse(&self) -> Self {
        let rotation = self.rotation.inverse();
        let distance = self.translation.as_ref();
        let vector = rotation.rotate_vector(&(-distance));
        let translation = Translation2::from_vector(&vector);
        
        Self::from_parts(&translation, &rotation)
    }

    /// Mutably invert an isometry in place.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
    /// #     Degrees, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// # };
    /// #
    /// let angle = Degrees(90_f64);
    /// let distance = Vector2::new(2_f64, 3_f64);
    /// let isometry = Isometry2::from_angle_translation(angle, &distance);
    /// let mut isometry_mut = Isometry2::from_angle_translation(angle, &distance);
    /// isometry_mut.inverse_mut();
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = point;
    /// let transformed_point = isometry.transform_point(&point);
    /// let result = isometry_mut.transform_point(&transformed_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.rotation.inverse_mut();
        self.translation.inverse_mut();
        self.translation.vector = self.rotation.rotate_vector(&self.translation.vector);
    }

    /// Transform a point with the isometry.
    ///
    /// The isometry applies the rotation followed by the translation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
    /// #     Radians,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let distance = Vector2::new(1_f64, 2_f64);
    /// let isometry = Isometry2::from_angle_translation(angle, &distance);
    /// let point = Point2::new(f64::sqrt(2_f64), f64::sqrt(2_f64));
    /// let expected = Point2::new(1_f64, 4_f64);
    /// let result = isometry.transform_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        let rotated_point = self.rotation.rotate_point(point);

        self.translation.translate_point(&rotated_point)
    }

    /// Apply a rotation followed by a translation.
    ///
    /// The isometry applies the rotation and not the translation to vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Radians,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let distance = Vector2::new(1_f64, 2_f64);
    /// let isometry = Isometry2::from_angle_translation(angle, &distance);
    /// let vector = Vector2::new(f64::sqrt(2_f64), f64::sqrt(2_f64));
    /// let expected = Vector2::new(0_f64, 2_f64);
    /// let result = isometry.transform_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    #[inline]
    pub fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        let rotated_vector = self.rotation.rotate_vector(vector);
        
        self.translation.translate_vector(&rotated_vector)
    }

    /// Transform a point with the inverse isometry.
    ///
    /// The inverse isometry applies the inverse translation followed by the
    /// inverse rotation. This is the reverse of the isometry.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
    /// #     Radians,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let distance = Vector2::new(1_f64, 2_f64);
    /// let isometry = Isometry2::from_angle_translation(angle, &distance);
    /// let point = Point2::new(f64::sqrt(2_f64), f64::sqrt(2_f64));
    /// let transformed_point = isometry.transform_point(&point);
    /// let expected = point;
    /// let result = isometry.inverse_transform_point(&transformed_point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_transform_point(&self, point: &Point2<S>) -> Point2<S> {
        self.rotation.inverse_rotate_point(&(point - self.translation.as_ref()))
    }
    
    /// Transform a vector with the inverse isometry.
    ///
    /// The inverse isometry applies the inverse rotation to the vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Radians,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let distance = Vector2::new(1_f64, 2_f64);
    /// let isometry = Isometry2::from_angle_translation(angle, &distance);
    /// let vector = Vector2::new(f64::sqrt(2_f64), f64::sqrt(2_f64));
    /// let transformed_vector = isometry.transform_vector(&vector);
    /// let expected = vector;
    /// let result = isometry.inverse_transform_vector(&transformed_vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.rotation.inverse_rotate_vector(vector)
    }

    /// Construct the identity isometry.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
    /// #     Radians,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// # };
    /// # use core::f64;
    /// # 
    /// let isometry = Isometry2::identity();
    /// let point = Point2::new(1_f64, 2_f64);
    /// 
    /// assert_eq!(isometry * point, point);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self {
            rotation: Rotation2::identity(),
            translation: Translation2::identity()
        }
    }
}

impl<S> fmt::Display for Isometry2<S> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Isometry2 [rotation={}, translation={}]", 
            self.rotation, self.translation
        )
    }
}

impl<S> approx::AbsDiffEq for Isometry2<S> 
where 
    S: SimdScalarFloat 
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Rotation2::abs_diff_eq(
            &self.rotation, 
            &other.rotation, 
            epsilon
        ) 
        && Translation2::abs_diff_eq(
            &self.translation, 
            &other.translation, 
            epsilon
        )
    }
}

impl<S> approx::RelativeEq for Isometry2<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Rotation2::relative_eq(
            &self.rotation, 
            &other.rotation, 
            epsilon, 
            max_relative
        ) 
        && Translation2::relative_eq(
            &self.translation, 
            &other.translation, 
            epsilon, 
            max_relative
        )
    }
}

impl<S> approx::UlpsEq for Isometry2<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Rotation2::ulps_eq(
            &self.rotation, 
            &other.rotation, 
            epsilon, 
            max_ulps
        ) 
        && Translation2::ulps_eq(
            &self.translation, 
            &other.translation, 
            epsilon, 
            max_ulps
        )
    }
}

impl<S> ops::Mul<Point2<S>> for Isometry2<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<S> ops::Mul<&Point2<S>> for Isometry2<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: &Point2<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Point2<S>> for &Isometry2<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point2<S>> for &'b Isometry2<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: &'a Point2<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Isometry2<S>> for Isometry2<S> 
where 
    S: SimdScalarFloat
{
    type Output = Isometry2<S>;

    #[inline]
    fn mul(self, other: Isometry2<S>) -> Self::Output {
        let shift = self.rotation.rotate_vector(&other.translation.vector);

        Isometry2::from_parts(
            &Translation2::from_vector(&(self.translation.vector + shift)),
            &(self.rotation * other.rotation)
        )
    }
}

impl<S> ops::Mul<&Isometry2<S>> for Isometry2<S> 
where 
    S: SimdScalarFloat
{
    type Output = Isometry2<S>;

    #[inline]
    fn mul(self, other: &Isometry2<S>) -> Self::Output {
        let shift = self.rotation.rotate_vector(&other.translation.vector);

        Isometry2::from_parts(
            &Translation2::from_vector(&(self.translation.vector + shift)),
            &(self.rotation * other.rotation)
        )
    }
}

impl<S> ops::Mul<Isometry2<S>> for &Isometry2<S> 
where 
    S: SimdScalarFloat
{
    type Output = Isometry2<S>;

    #[inline]
    fn mul(self, other: Isometry2<S>) -> Self::Output {
        let shift = self.rotation.rotate_vector(&other.translation.vector);

        Isometry2::from_parts(
            &Translation2::from_vector(&(self.translation.vector + shift)),
            &(self.rotation * other.rotation)
        )
    }
}

impl<'a, 'b, S> ops::Mul<&'a Isometry2<S>> for &'b Isometry2<S> 
where 
    S: SimdScalarFloat
{
    type Output = Isometry2<S>;

    #[inline]
    fn mul(self, other: &'a Isometry2<S>) -> Self::Output {
        let shift = self.rotation.rotate_vector(&other.translation.vector);

        Isometry2::from_parts(
            &Translation2::from_vector(&(self.translation.vector + shift)),
            &(self.rotation * other.rotation)
        )
    }
}


/// A three-dimensional isometry (i.e. rigid body transformation) is a 
/// transformation whose motion does not distort the shape of an object. 
///
/// In terms of transformations, an isometry is a combination of 
/// a rotation and a translation. Rigid body transformations preserve the lengths
/// of vectors, hence the name isometry. In terms of transforming points and 
/// vectors, an isometry applies the rotation, followed by the translation.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Isometry3<S> {
    /// The rotation component of an isometry.
    pub(crate) rotation: Rotation3<S>,
    /// The translation component of an isometry.
    pub(crate) translation: Translation3<S>,
}

impl<S> Isometry3<S> 
where 
    S: SimdScalarFloat
{
    /// Construct a new isometry directly from a translation and a rotation.
    #[inline]
    pub const fn from_parts(translation: &Translation3<S>, rotation: &Rotation3<S>) -> Self {
        Self {
            rotation: *rotation,
            translation: *translation,
        }
    }

    /// Construct a new isometry from a rotation axis, rotation angle, and a 
    /// displacement vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Degrees,
    /// #     Unit, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// #     Translation3,
    /// #     Rotation3,
    /// # };
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle = Degrees(72_f64);
    /// let distance = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let translation = Translation3::from_vector(&distance);
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let expected = Isometry3::from_parts(&translation, &rotation);
    /// let result = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_axis_angle_translation<A: Into<Radians<S>>>(
        axis: &Unit<Vector3<S>>, angle: A, distance: &Vector3<S>) -> Self
    {
        Self {
            rotation: Rotation3::from_axis_angle(axis, angle),
            translation: Translation3::from_vector(distance),
        }
    }

    /// Construct a new isometry from a translation.
    #[inline]
    pub fn from_translation(translation: &Translation3<S>) -> Self {
        Self::from_parts(translation, &Rotation3::identity())
    }

    /// Construct a new isometry from a rotation.
    #[inline]
    pub fn from_rotation(rotation: &Rotation3<S>) -> Self {
        Self::from_parts(&Translation3::identity(), rotation)
    }

    /// Construct a new isometry from a rotation axis and a rotation angle.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Degrees,
    /// #     Unit, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// #     Rotation3,
    /// # };
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle = Degrees(72_f64);
    /// let distance = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let expected = Isometry3::from_rotation(&rotation);
    /// let result = Isometry3::from_axis_angle(&axis, angle);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_axis_angle<A: Into<Radians<S>>>(axis: &Unit<Vector3<S>>, angle: A) -> Self {
        let translation = Translation3::identity();
        let rotation = Rotation3::from_axis_angle(axis, angle);
        
        Self::from_parts(&translation, &rotation)
    }

    /// Construct an isometry from a rotation angle in the **yz-plane** about 
    /// the **x-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// #     Rotation3,
    /// # };
    /// #
    /// let angle = Degrees(72_f64);
    /// let rotation = Rotation3::from_angle_x(angle);
    /// let expected = Isometry3::from_rotation(&rotation);
    /// let result = Isometry3::from_angle_x(angle);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_angle_x<A: Into<Radians<S>>>(angle: A) -> Self {
        let translation = Translation3::identity();
        let rotation = Rotation3::from_angle_x(angle);
        
        Self::from_parts(&translation, &rotation)
    }

    /// Construct an isometry from a rotation angle in the **zx-plane** about 
    /// the **y-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// #     Rotation3,
    /// # };
    /// #
    /// let angle = Degrees(72_f64);
    /// let rotation = Rotation3::from_angle_y(angle);
    /// let expected = Isometry3::from_rotation(&rotation);
    /// let result = Isometry3::from_angle_y(angle);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_angle_y<A: Into<Radians<S>>>(angle: A) -> Self {
        let translation = Translation3::identity();
        let rotation = Rotation3::from_angle_y(angle);
        
        Self::from_parts(&translation, &rotation)
    }

    /// Construct an isometry from a rotation angle in the **xy-plane** about 
    /// the **z-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// #     Rotation3,
    /// # };
    /// #
    /// let angle = Degrees(72_f64);
    /// let rotation = Rotation3::from_angle_z(angle);
    /// let expected = Isometry3::from_rotation(&rotation);
    /// let result = Isometry3::from_angle_z(angle);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_angle_z<A: Into<Radians<S>>>(angle: A) -> Self {
        let translation = Translation3::identity();
        let rotation = Rotation3::from_angle_z(angle);
        
        Self::from_parts(&translation, &rotation)
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between two vectors.
    ///
    /// The rotation uses the unit directional vectors of the input vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let vector = 3_f64 * Vector3::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64, 0_f64);
    /// let vector1 = 5_f64 * Vector3::unit_y();
    /// let vector2 = 12_f64 * Vector3::unit_x();
    /// let isometry = Isometry3::rotation_between(&vector1, &vector2).unwrap();
    /// let expected = 3_f64 * Vector3::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64, 0_f64);
    /// let result = isometry.transform_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn rotation_between(
        v1: &Vector3<S>, v2: &Vector3<S>) -> Option<Self> 
    {        
        if let (Some(unit_v1), Some(unit_v2)) = (
            Unit::try_from_value(*v1, S::default_epsilon()), 
            Unit::try_from_value(*v2, S::default_epsilon())
        ) {
            Self::rotation_between_axis(&unit_v1, &unit_v2)
        } else {
            None
        }
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between two unit vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let vector = 3_f64 * Vector3::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64, 0_f64);
    /// let unit1 = Unit::from_value(5_f64 * Vector3::unit_y());
    /// let unit2 = Unit::from_value(12_f64 * Vector3::unit_x());
    /// let isometry = Isometry3::rotation_between_axis(&unit1, &unit2).unwrap();
    /// let expected = 3_f64 * Vector3::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64, 0_f64);
    /// let result = isometry.transform_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn rotation_between_axis(
        v1: &Unit<Vector3<S>>, v2: &Unit<Vector3<S>>) -> Option<Self> 
    {    
        Rotation3::rotation_between_axis(v1, v2).map(|rotation| {
            let translation = Translation3::identity();
            Self::from_parts(&translation, &rotation)
        })
    }

    /// Construct a coordinate transformation that maps the coordinate system 
    /// of an observer located at the position `eye` facing the direction 
    /// `direction` into a coordinate system of an observer located at the 
    /// origin facing the **positive z-axis**. The resulting coordinate 
    /// transformation is a **left-handed** coordinate transformation.
    ///
    /// The resulting isometry maps the direction `direction` to the 
    /// **positive z-axis** and locates position `eye` to the the origin.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(1_f64, -1_f64, 1_f64);
    /// let direction = target - eye;
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let isometry = Isometry3::look_to_lh(&eye, &direction, &up);
    /// let origin: Point3<f64> = Point3::origin();
    /// let unit_z = Vector3::unit_z();
    ///
    /// assert_relative_eq!(
    ///     isometry.transform_point(&eye), 
    ///     origin, 
    ///     epsilon = 1e-10,
    /// );
    /// assert_relative_eq!(
    ///     isometry.transform_vector(&direction).normalize(), 
    ///     unit_z, 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_to_lh(eye: &Point3<S>, direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        let rotation = Rotation3::look_to_lh(direction, up);
        let vector = rotation * (-eye) - Point3::origin();
        let translation = Translation3::from_vector(&vector);

        Self::from_parts(&translation, &rotation)
    }

    /// Construct a coordinate transformation that maps the coordinate system 
    /// of an observer located at the position `eye` facing the direction 
    /// `direction` into a coordinate system of an observer located at the 
    /// origin facing the **negative z-axis** The resulting coordinate 
    /// transformation is a **right-handed** coordinate transformation.
    ///
    /// The resulting isometry maps the direction `direction` to the 
    /// **negative z-axis** and locates the position `eye` to the origin.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(1_f64, -1_f64, 1_f64);
    /// let direction = target - eye;
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let isometry = Isometry3::look_to_rh(&eye, &direction, &up);
    /// let origin = Point3::origin();
    /// let minus_unit_z = -Vector3::unit_z();
    ///
    /// assert_relative_eq!(
    ///     isometry.transform_point(&eye), 
    ///     origin, 
    ///     epsilon = 1e-10,
    /// );
    /// assert_relative_eq!(
    ///     isometry.transform_vector(&direction).normalize(), 
    ///     minus_unit_z, 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_to_rh(eye: &Point3<S>, direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        let rotation = Rotation3::look_to_rh(direction, up);
        let vector = rotation * (-eye) - Point3::origin();
        let translation = Translation3::from_vector(&vector);

        Self::from_parts(&translation, &rotation)
    }

    /// Construct an coordinate transformation that transforms the 
    /// coordinate system of an observer located at the position `eye` facing 
    /// the direction of the target `target` into the coordinate system of an 
    /// observer located at the origin facing the **positive z-axis**.
    ///
    /// The isometry maps the direction along the ray between the eye position 
    /// `eye` and position of the target `target` to the **positive z-axis** and 
    /// locates the `eye` position to the origin in the new the coordinate system. 
    /// This transformation is a **left-handed** coordinate transformation. 
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # 
    /// let target = Point3::new(0_f64, 6_f64, 0_f64);
    /// let up: Vector3<f64> = Vector3::unit_x();
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let isometry = Isometry3::look_at_lh(&eye, &target, &up);
    /// let direction = target - eye;
    /// let unit_z = Vector3::unit_z();
    /// let origin = Point3::origin();
    ///
    /// assert_relative_eq!(
    ///     isometry.transform_vector(&direction).normalize(), 
    ///     unit_z, 
    ///     epsilon = 1e-10,
    /// );
    /// assert_relative_eq!(
    ///     isometry.transform_point(&eye), 
    ///     origin, 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_at_lh(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        let rotation = Rotation3::look_at_lh(eye, target, up);
        let vector = rotation * (-eye) - Point3::origin();
        let translation = Translation3::from_vector(&vector);

        Self::from_parts(&translation, &rotation)
    }

    /// Construct an coordinate transformation that transforms the
    /// coordinate system of an observer located at the position `eye` facing 
    /// the direction of the target `target` into the coordinate system of an 
    /// observer located at the origin facing the **negative z-axis**.
    ///
    /// The function maps the direction along the ray between the eye position 
    /// `eye` and position of the target `target` to the **negative z-axis** and 
    /// locates the `eye` position to the origin in the new the coordinate system. 
    /// This transformation is a **right-handed** coordinate transformation. 
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # 
    /// let target = Point3::new(0_f64, 6_f64, 0_f64);
    /// let up: Vector3<f64> = Vector3::unit_x();
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let isometry = Isometry3::look_at_rh(&eye, &target, &up);
    /// let direction = target - eye;
    /// let minus_unit_z = -Vector3::unit_z();
    /// let origin = Point3::origin();
    ///
    /// assert_relative_eq!(
    ///     isometry.transform_vector(&direction).normalize(), 
    ///     minus_unit_z, 
    ///     epsilon = 1e-10,
    /// );
    /// assert_relative_eq!(
    ///     isometry.transform_point(&eye), 
    ///     origin, 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_at_rh(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Isometry3<S>{
        let rotation = Rotation3::look_at_rh(eye, target, up);
        let vector = rotation * (-eye) - Point3::origin();
        let translation = Translation3::from_vector(&vector);
    
        Self::from_parts(&translation, &rotation)  
    }

    /// Construct a coordinate transformation that maps the coordinate system 
    /// of an observer located at the origin facing the **positive z-axis** into a 
    /// coordinate system of an observer located at the position `eye` facing the 
    /// direction `direction`. The resulting coordinate transformation is a 
    /// **left-handed** coordinate transformation.
    ///
    /// The resulting isometry maps the **positive z-axis** to the direction 
    /// `direction` and locates the origin of the coordinate system to the `eye` 
    /// position.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(1_f64, -1_f64, 1_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let direction = target - eye;
    /// let isometry = Isometry3::look_to_lh_inv(&eye, &direction, &up);
    /// let origin: Point3<f64> = Point3::origin();
    /// let unit_z = Vector3::unit_z();
    ///
    /// assert_relative_eq!(
    ///     isometry.transform_vector(&unit_z).normalize(), 
    ///     direction.normalize(), 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_to_lh_inv(eye: &Point3<S>, direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        let rotation = Rotation3::look_to_lh_inv(direction, up);
        let vector = rotation * (-eye) - Point3::origin();
        let translation = Translation3::from_vector(&vector);

        Self::from_parts(&translation, &rotation)
    }

    /// Construct a coordinate transformation that maps the coordinate system 
    /// of an observer located at the origin facing the **negative z-axis** into a 
    /// coordinate system of an observer located at the position `eye` facing the 
    /// direction `direction`. The resulting coordinate transformation is a 
    /// **right-handed** coordinate transformation.
    ///
    /// The resulting isometry maps the **negative z-axis** to the direction 
    /// `direction` and locates the origin of the coordinate system to the `eye` 
    /// position.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(1_f64, -1_f64, 1_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let direction = target - eye;
    /// let isometry = Isometry3::look_to_rh_inv(&eye, &direction, &up);
    /// let minus_unit_z = -Vector3::unit_z();
    ///
    /// assert_relative_eq!(
    ///     isometry.transform_vector(&minus_unit_z), 
    ///     direction.normalize(), 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_to_rh_inv(eye: &Point3<S>, direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        let rotation = Rotation3::look_to_rh_inv(direction, up);
        let vector = rotation * (-eye) - Point3::origin();
        let translation = Translation3::from_vector(&vector);

        Self::from_parts(&translation, &rotation)
    }

    /// Construct a coordinate transformation that maps the coordinate system 
    /// of an observer located at the origin facing the **positive z-axis** into a 
    /// coordinate system of an observer located at the position `eye` facing the 
    /// direction `direction`. The resulting coordinate transformation is a 
    /// **left-handed** coordinate transformation.
    ///
    /// The resulting isometry maps the **positive z-axis** to the direction 
    /// `direction` and locates the origin of the coordinate system to the `eye` 
    /// position.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(1_f64, -1_f64, 1_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let isometry = Isometry3::look_at_lh_inv(&eye, &target, &up);
    /// let direction = target - eye;
    /// let unit_z = Vector3::unit_z();
    ///
    /// assert_relative_eq!(
    ///     isometry.transform_vector(&unit_z), 
    ///     direction.normalize(), 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_at_lh_inv(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        let rotation = Rotation3::look_at_lh_inv(eye, target, up);
        let vector = rotation * (-eye) - Point3::origin();
        let translation = Translation3::from_vector(&vector);

        Self::from_parts(&translation, &rotation)
    }

    /// Construct a coordinate transformation that maps the coordinate system 
    /// of an observer located at the origin facing the **negative z-axis** into a 
    /// coordinate system of an observer located at the position `eye` facing the 
    /// direction `direction`. The resulting coordinate transformation is a 
    /// **right-handed** coordinate transformation.
    ///
    /// The resulting isometry maps the **negative z-axis** to the direction 
    /// `direction` and locates the origin of the coordinate system to the `eye` 
    /// position.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(1_f64, -1_f64, 1_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let isometry = Isometry3::look_at_rh_inv(&eye, &target, &up);
    /// let minus_unit_z = -Vector3::unit_z();
    /// let direction = target - eye;
    ///
    /// assert_relative_eq!(
    ///     isometry.transform_vector(&minus_unit_z), 
    ///     direction.normalize(), 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_at_rh_inv(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        let rotation = Rotation3::look_at_rh_inv(eye, target, up);
        let vector = rotation * (-eye) - Point3::origin();
        let translation = Translation3::from_vector(&vector);

        Self::from_parts(&translation, &rotation)
    }

    /// Convert an isometry into a generic transformation.
    #[inline]
    pub fn to_transform(&self) -> Transform3<S> {
        let matrix = self.to_affine_matrix();
        Transform3::from_specialized(matrix)
    }

    /// Convert an isometry to an equivalent affine transformation matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Degrees,
    /// #     Unit, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,    
    /// # };
    /// #
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let angle = Degrees(90_f64);
    /// let distance = Vector3::new(2_f64, 3_f64, 4_f64);
    /// let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    /// let expected = Matrix4x4::new(
    ///      0_f64, 1_f64, 0_f64, 0_f64,
    ///     -1_f64, 0_f64, 0_f64, 0_f64,
    ///      0_f64, 0_f64, 1_f64, 0_f64,
    ///      2_f64, 3_f64, 4_f64, 1_f64
    /// );
    /// let result = isometry.to_affine_matrix();
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix4x4<S> {
        let zero = S::zero();
        let one = S::one();
        let rotation_matrix = self.rotation.matrix();
        let translation = self.translation.as_ref();

        Matrix4x4::new(
            rotation_matrix.c0r0, rotation_matrix.c0r1, rotation_matrix.c0r2, zero,
            rotation_matrix.c1r0, rotation_matrix.c1r1, rotation_matrix.c1r2, zero,
            rotation_matrix.c2r0, rotation_matrix.c2r1, rotation_matrix.c2r2, zero,
            translation[0],       translation[1],       translation[2],       one
        )
    }
    
    /// Get the rotation component of the isometry.
    #[inline]
    pub const fn rotation(&self) -> &Rotation3<S> {
        &self.rotation
    }

    /// Get the translation part of the isometry.
    #[inline]
    pub const fn translation(&self) -> &Translation3<S> {
        &self.translation
    }

    /// Construct the inverse isometry of an isometry.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Degrees,
    /// #     Point3,
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle = Degrees(90_f64);
    /// let distance = Vector3::new(2_f64, 3_f64, 4_f64);
    /// let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    /// let isometry_inv = isometry.inverse();
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = point;
    /// let result = isometry_inv * (isometry * point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse(&self) -> Self {
        let rotation = self.rotation.inverse();
        let distance = self.translation.as_ref();
        let vector = rotation.rotate_vector(&(-distance));
        let translation = Translation3::from_vector(&vector);
        
        Self::from_parts(&translation, &rotation)
    }

    /// Mutably invert an isometry in place.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3,
    /// #     Degrees,
    /// #     Unit, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle = Degrees(90_f64);
    /// let distance = Vector3::new(2_f64, 3_f64, 4_f64);
    /// let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    /// let mut isometry_mut = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    /// isometry_mut.inverse_mut();
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = point;
    /// let transformed_point = isometry.transform_point(&point);
    /// let result = isometry_mut.transform_point(&transformed_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.rotation.inverse_mut();
        self.translation.inverse_mut();
        self.translation.vector = self.rotation.rotate_vector(&self.translation.vector);
    }

    /// Transform a point with the isometry.
    ///
    /// The isometry applies the rotation followed by the translation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3,
    /// #     Radians,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let distance = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    /// let point = Point3::new(f64::sqrt(2_f64), f64::sqrt(2_f64), f64::sqrt(2_f64));
    /// let expected = Point3::new(1_f64, 4_f64, f64::sqrt(2_f64) + 3_f64);
    /// let result = isometry.transform_point(&point);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        let rotated_point = self.rotation.rotate_point(point);

        self.translation.translate_point(&rotated_point)
    }

    /// Transform a vector with the isometry.
    ///
    /// The isometry applies the rotation to the vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Radians,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let distance = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    /// let vector = Vector3::new(f64::sqrt(2_f64), f64::sqrt(2_f64), f64::sqrt(2_f64));
    /// let expected = Vector3::new(0_f64, 2_f64, f64::sqrt(2_f64));
    /// let result = isometry.transform_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.rotation.rotate_vector(vector)
    }

    /// Transform a point with the inverse of an isometry.
    ///
    /// The inverse isometry applies the inverse translation followed by the
    /// rotation. This is the reverse of the isometry.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3,
    /// #     Radians,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let distance = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    /// let point = Point3::new(f64::sqrt(2_f64), f64::sqrt(2_f64), f64::sqrt(2_f64));
    /// let expected = point;
    /// let transformed_point = isometry.transform_point(&point);
    /// let result = isometry.inverse_transform_point(&transformed_point);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_transform_point(&self, point: &Point3<S>) -> Point3<S> {
        self.rotation.inverse_rotate_point(&(point - self.translation.as_ref()))
    }
    
    /// Transform a vector with the inverse of an isometry.
    ///
    /// The inverse isometry applies the inverse rotation to vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Radians,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let distance = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    /// let vector = Vector3::new(f64::sqrt(2_f64), f64::sqrt(2_f64), f64::sqrt(2_f64));
    /// let expected = vector;
    /// let transformed_vector = isometry.transform_vector(&vector);
    /// let result = isometry.inverse_transform_vector(&transformed_vector);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.rotation.inverse_rotate_vector(vector)
    }

    /// Construct the identity isometry.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3,
    /// #     Radians,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// # };
    /// # use core::f64;
    /// # 
    /// let isometry = Isometry3::identity();
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// 
    /// assert_eq!(isometry * point, point);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self {
            rotation: Rotation3::identity(),
            translation: Translation3::identity()
        }
    }
}

impl<S> fmt::Display for Isometry3<S> 
where 
    S: fmt::Display
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Isometry3 [rotation={}, translation={}]", 
            self.rotation, self.translation
        )
    }
}

impl<S> approx::AbsDiffEq for Isometry3<S> 
where 
    S: SimdScalarFloat
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Rotation3::abs_diff_eq(
            &self.rotation, 
            &other.rotation, 
            epsilon
        ) 
        && Translation3::abs_diff_eq(
            &self.translation, 
            &other.translation, 
            epsilon
        )
    }
}

impl<S> approx::RelativeEq for Isometry3<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Rotation3::relative_eq(
            &self.rotation, 
            &other.rotation, 
            epsilon, 
            max_relative
        ) 
        && Translation3::relative_eq(
            &self.translation, 
            &other.translation, 
            epsilon, 
            max_relative
        )
    }
}

impl<S> approx::UlpsEq for Isometry3<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Rotation3::ulps_eq(
            &self.rotation, 
            &other.rotation, 
            epsilon, 
            max_ulps
        ) 
        && Translation3::ulps_eq(
            &self.translation, 
            &other.translation, 
            epsilon, 
            max_ulps
        )
    }
}

impl<S> ops::Mul<Point3<S>> for Isometry3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for Isometry3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &Isometry3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b Isometry3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Isometry3<S>> for Isometry3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Isometry3<S>;

    #[inline]
    fn mul(self, other: Isometry3<S>) -> Self::Output {
        let shift = self.rotation.rotate_vector(&other.translation.vector);

        Isometry3::from_parts(
            &Translation3::from_vector(&(self.translation.vector + shift)),
            &(self.rotation * other.rotation)
        )
    }
}

impl<S> ops::Mul<&Isometry3<S>> for Isometry3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Isometry3<S>;

    #[inline]
    fn mul(self, other: &Isometry3<S>) -> Self::Output {
        let shift = self.rotation.rotate_vector(&other.translation.vector);

        Isometry3::from_parts(
            &Translation3::from_vector(&(self.translation.vector + shift)),
            &(self.rotation * other.rotation)
        )
    }
}

impl<S> ops::Mul<Isometry3<S>> for &Isometry3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Isometry3<S>;

    #[inline]
    fn mul(self, other: Isometry3<S>) -> Self::Output {
        let shift = self.rotation.rotate_vector(&other.translation.vector);

        Isometry3::from_parts(
            &Translation3::from_vector(&(self.translation.vector + shift)),
            &(self.rotation * other.rotation)
        )
    }
}

impl<'a, 'b, S> ops::Mul<&'a Isometry3<S>> for &'b Isometry3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Isometry3<S>;

    #[inline]
    fn mul(self, other: &'a Isometry3<S>) -> Self::Output {
        let shift = self.rotation.rotate_vector(&other.translation.vector);

        Isometry3::from_parts(
            &Translation3::from_vector(&(self.translation.vector + shift)),
            &(self.rotation * other.rotation)
        )
    }
}

