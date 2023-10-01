use cglinalg_numeric::{
    SimdScalarFloat,
};
use cglinalg_trigonometry::{
    Angle,
    Radians,
};
use cglinalg_core::{
    Const,
    ShapeConstraint,
    DimAdd,
    DimMul,
    DimLt,
    Matrix,
    Point,
    Point3,
    Vector,
    Vector2,
    Vector3,
    Unit,
};
use crate::rotation::{
    Rotation,
    Rotation2,
    Rotation3,
};
use crate::translation::{
    Translation,
    Translation2,
    Translation3,
};
use crate::transform::{
    Transform,
};

use core::fmt;
use core::ops;


/// An isometry in two dimensions.
pub type Isometry2<S> = Isometry<S, 2>;

/// An isometry in three dimensions.
pub type Isometry3<S> = Isometry<S, 3>;


/// An isometry (i.e. rigid body transformation) is a transformation whose motion 
/// does not distort the shape of an object. 
///
/// In terms of transformations, an isometry is a combination of 
/// a rotation and a translation. Rigid body transformations preserve the lengths
/// of vectors, hence the name isometry. In terms of transforming points and 
/// vectors, an isometry applies the rotation, followed by the translation.
/// 
/// This is the most general isometry type. The vast majority of applications 
/// should use [`Isometry2`] or [`Isometry3`] instead of this type directly.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Isometry<S, const N: usize> {
    /// The rotation component of an isometry.
    pub(crate) rotation: Rotation<S, N>,
    /// The translation component of an isometry.
    pub(crate) translation: Translation<S, N>,
}

impl<S, const N: usize> Isometry<S, N> 
where 
    S: SimdScalarFloat
{
    /// Construct a new isometry directly from a translation and a rotation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// #     Translation2,
    /// #     Rotation2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let rotation = Rotation2::from_angle(angle);
    /// let translation = Translation2::new(1_f64, 2_f64);
    /// let isometry = Isometry2::from_parts(&translation, &rotation);
    /// let point = Point2::new(1_f64, 0_f64);
    /// let expected = Point2::new(
    ///     1_f64 / f64::sqrt(2_f64) + 1_f64, 
    ///     1_f64 / f64::sqrt(2_f64) + 2_f64
    /// );
    /// let result = isometry.apply_point(&point);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// #     Translation3,
    /// #     Rotation3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let rotation = Rotation3::from_angle_z(angle);
    /// let translation = Translation3::new(1_f64, 2_f64, 0_f64);
    /// let isometry = Isometry3::from_parts(&translation, &rotation);
    /// let point = Point3::new(1_f64, 0_f64, 3_f64);
    /// let expected = Point3::new(
    ///     1_f64 / f64::sqrt(2_f64) + 1_f64, 
    ///     1_f64 / f64::sqrt(2_f64) + 2_f64,
    ///     3_f64
    /// );
    /// let result = isometry.apply_point(&point);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn from_parts(translation: &Translation<S, N>, rotation: &Rotation<S, N>) -> Self {
        Self {
            rotation: *rotation,
            translation: *translation,
        }
    }

    /// Construct a new isometry from a translation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// #     Translation2, 
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2,
    /// # };
    /// #
    /// let translation = Translation2::new(1_f64, 2_f64);
    /// let isometry = Isometry2::from_translation(&translation);
    /// let point = Point2::origin();
    /// let expected = Point2::new(1_f64, 2_f64);
    /// let result = isometry.apply_point(&point);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// #     Translation3, 
    /// # };
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// #
    /// let translation = Translation3::new(1_f64, 2_f64, 3_f64);
    /// let isometry = Isometry3::from_translation(&translation);
    /// let point = Point3::origin();
    /// let expected = Point3::new(1_f64, 2_f64, 3_f64);
    /// let result = isometry.apply_point(&point);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_translation(translation: &Translation<S, N>) -> Self {
        Self::from_parts(translation, &Rotation::identity())
    }

    /// Construct a new isometry from a rotation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// #     Rotation2,    
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let rotation = Rotation2::from_angle(angle);
    /// let point = Point2::new(1_f64, 0_f64);
    /// let expected = Point2::new(0_f64, 1_f64);
    /// let result = rotation.apply_point(&point);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-15);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// #     Rotation3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let rotation = Rotation3::from_angle_z(angle);
    /// let point = Point3::new(1_f64, 0_f64, 0_f64);
    /// let expected = Point3::new(0_f64, 1_f64, 0_f64);
    /// let result = rotation.apply_point(&point);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-15);
    /// ```
    #[inline]
    pub fn from_rotation(rotation: &Rotation<S, N>) -> Self {
        Self::from_parts(&Translation::identity(), rotation)
    }
}

impl<S, const N: usize> Isometry<S, N> 
where 
    S: SimdScalarFloat
{
    /// Get the rotation component of an isometry.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// #     Translation2,
    /// #     Rotation2,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let rotation = Rotation2::from_angle(angle);
    /// let translation = Translation2::new(1_f64, 2_f64);
    /// let isometry = Isometry2::from_parts(&translation, &rotation);
    /// let expected = &rotation;
    /// let result = isometry.rotation();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// #     Translation3,
    /// #     Rotation3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let rotation = Rotation3::from_angle_z(angle);
    /// let translation = Translation3::new(1_f64, 2_f64, 0_f64);
    /// let isometry = Isometry3::from_parts(&translation, &rotation);
    /// let expected = &rotation;
    /// let result = isometry.rotation();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn rotation(&self) -> &Rotation<S, N> {
        &self.rotation
    }

    /// Get the translation part of an isometry.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// #     Translation2,
    /// #     Rotation2,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let rotation = Rotation2::from_angle(angle);
    /// let translation = Translation2::new(1_f64, 2_f64);
    /// let isometry = Isometry2::from_parts(&translation, &rotation);
    /// let expected = &rotation;
    /// let result = isometry.rotation();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// #     Translation3,
    /// #     Rotation3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let rotation = Rotation3::from_angle_z(angle);
    /// let translation = Translation3::new(1_f64, 2_f64, 0_f64);
    /// let isometry = Isometry3::from_parts(&translation, &rotation);
    /// let expected = &rotation;
    /// let result = isometry.rotation();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn translation(&self) -> &Translation<S, N> {
        &self.translation
    }

    /// Construct the inverse isometry of an isometry.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
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
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
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
        let vector = rotation.apply_vector(&(-distance));
        let translation = Translation::from_vector(&vector);
        
        Self::from_parts(&translation, &rotation)
    }

    /// Mutably invert an isometry in place.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2, 
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
    /// let transformed_point = isometry.apply_point(&point);
    /// let result = isometry_mut.apply_point(&transformed_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3,
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
    /// let transformed_point = isometry.apply_point(&point);
    /// let result = isometry_mut.apply_point(&transformed_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.rotation.inverse_mut();
        self.translation.inverse_mut();
        self.translation.vector = self.rotation.apply_vector(&self.translation.vector);
    }

    /// Transform a point with the isometry.
    ///
    /// The isometry applies the rotation followed by the translation.
    ///
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
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
    /// let result = isometry.apply_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
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
    /// # use core::f64;
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let distance = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    /// let point = Point3::new(f64::sqrt(2_f64), f64::sqrt(2_f64), f64::sqrt(2_f64));
    /// let expected = Point3::new(1_f64, 4_f64, f64::sqrt(2_f64) + 3_f64);
    /// let result = isometry.apply_point(&point);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn apply_point(&self, point: &Point<S, N>) -> Point<S, N> {
        let rotated_point = self.rotation.apply_point(point);

        self.translation.apply_point(&rotated_point)
    }

    /// Transform a vector with the isometry.
    ///
    /// The isometry applies the rotation to the vector.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
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
    /// let result = isometry.apply_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
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
    /// # use core::f64;
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let distance = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    /// let vector = Vector3::new(f64::sqrt(2_f64), f64::sqrt(2_f64), f64::sqrt(2_f64));
    /// let expected = Vector3::new(0_f64, 2_f64, f64::sqrt(2_f64));
    /// let result = isometry.apply_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn apply_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        self.rotation.apply_vector(vector)
    }

    /// Transform a point with the inverse of an isometry.
    ///
    /// The inverse isometry applies the inverse translation followed by the
    /// rotation. This is the reverse of the isometry.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
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
    /// let transformed_point = isometry.apply_point(&point);
    /// let expected = point;
    /// let result = isometry.inverse_apply_point(&transformed_point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
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
    /// # use core::f64;
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let distance = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    /// let point = Point3::new(f64::sqrt(2_f64), f64::sqrt(2_f64), f64::sqrt(2_f64));
    /// let expected = point;
    /// let transformed_point = isometry.apply_point(&point);
    /// let result = isometry.inverse_apply_point(&transformed_point);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_apply_point(&self, point: &Point<S, N>) -> Point<S, N> {
        self.rotation.inverse_apply_point(&(point - self.translation.as_ref()))
    }
    
    /// Transform a vector with the inverse of an isometry.
    ///
    /// The inverse isometry applies the inverse rotation to vectors.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
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
    /// let transformed_vector = isometry.apply_vector(&vector);
    /// let expected = vector;
    /// let result = isometry.inverse_apply_vector(&transformed_vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
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
    /// # use core::f64;
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let distance = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    /// let vector = Vector3::new(f64::sqrt(2_f64), f64::sqrt(2_f64), f64::sqrt(2_f64));
    /// let expected = vector;
    /// let transformed_vector = isometry.apply_vector(&vector);
    /// let result = isometry.inverse_apply_vector(&transformed_vector);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_apply_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        self.rotation.inverse_apply_vector(vector)
    }

    /// Construct the identity isometry.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
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
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3,
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
            rotation: Rotation::identity(),
            translation: Translation::identity()
        }
    }
}

impl<S, const N: usize, const NPLUS1: usize> Isometry<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimLt<Const<N>, Const<NPLUS1>>
{
    /// Convert an isometry to an affine matrix.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
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
    /// let expected = Matrix3x3::new(
    ///      0_f64, 1_f64, 0_f64,
    ///     -1_f64, 0_f64, 0_f64,
    ///      2_f64, 3_f64, 1_f64
    /// );
    /// let result = isometry.to_affine_matrix();
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
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
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix<S, NPLUS1, NPLUS1> {
        let translation = self.translation.as_ref();
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Matrix::from(self.rotation.matrix());
        for i in 0..N {
            result[N][i] = translation[i];
        }

        result
    }

    /// Convert an isometry into a generic transformation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Isometry2,
    /// #     Rotation2,
    /// #     Translation2,
    /// #     Transform2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_3);
    /// let translation = Translation2::new(2_f64, 3_f64);
    /// let rotation = Rotation2::from_angle(angle);
    /// let isometry = Isometry2::from_parts(&translation, &rotation);
    /// let expected = Transform2::from_matrix_unchecked(Matrix3x3::new(
    ///      1_f64 / 2_f64,            f64::sqrt(3_f64) / 2_f64, 0_f64,
    ///     -f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64,            0_f64,
    ///      2_f64,                    3_f64,                    1_f64
    /// ));
    /// let result = isometry.to_transform();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-15);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Isometry3,
    /// #     Rotation3,
    /// #     Translation3,
    /// #     Transform3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_3);
    /// let translation = Translation3::new(2_f64, 3_f64, 4_f64);
    /// let rotation = Rotation3::from_angle_z(angle);
    /// let isometry = Isometry3::from_parts(&translation, &rotation);
    /// let expected = Transform3::from_matrix_unchecked(Matrix4x4::new(
    ///     1_f64 / 2_f64,            f64::sqrt(3_f64) / 2_f64, 0_f64, 0_f64,
    ///    -f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64,            0_f64, 0_f64,
    ///     0_f64,                    0_f64,                    1_f64, 0_f64,
    ///     2_f64,                    3_f64,                    4_f64, 1_f64
    /// ));
    /// let result = isometry.to_transform();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-15);
    /// ```
    #[inline]
    pub fn to_transform(&self) -> Transform<S, N, NPLUS1> {
        let matrix = self.to_affine_matrix();
        
        Transform::from_matrix_unchecked(matrix)
    }
}

impl<S, const N: usize> fmt::Display for Isometry<S, N> 
where 
    S: fmt::Display
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Isometry{} [rotation={}, translation={}]", 
            N, self.rotation, self.translation
        )
    }
}

impl<S, const N: usize, const NPLUS1: usize> From<Isometry<S, N>> for Matrix<S, NPLUS1, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimLt<Const<N>, Const<NPLUS1>>
{
    #[inline]
    fn from(isometry: Isometry<S, N>) -> Matrix<S, NPLUS1, NPLUS1> {
        isometry.to_affine_matrix()
    }
}

impl<S, const N: usize, const NPLUS1: usize> From<&Isometry<S, N>> for Matrix<S, NPLUS1, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimLt<Const<N>, Const<NPLUS1>>
{
    #[inline]
    fn from(isometry: &Isometry<S, N>) -> Matrix<S, NPLUS1, NPLUS1> {
        isometry.to_affine_matrix()
    }
}

impl<S, const N: usize> approx::AbsDiffEq for Isometry<S, N> 
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
        Rotation::abs_diff_eq(
            &self.rotation, 
            &other.rotation, 
            epsilon
        ) 
        && Translation::abs_diff_eq(
            &self.translation, 
            &other.translation, 
            epsilon
        )
    }
}

impl<S, const N: usize> approx::RelativeEq for Isometry<S, N> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        Rotation::relative_eq(
            &self.rotation, 
            &other.rotation, 
            epsilon, 
            max_relative
        ) 
        && Translation::relative_eq(
            &self.translation, 
            &other.translation, 
            epsilon, 
            max_relative
        )
    }
}

impl<S, const N: usize> approx::UlpsEq for Isometry<S, N> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        Rotation::ulps_eq(
            &self.rotation, 
            &other.rotation, 
            epsilon, 
            max_ulps
        ) 
        && Translation::ulps_eq(
            &self.translation, 
            &other.translation, 
            epsilon, 
            max_ulps
        )
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for Isometry<S, N> 
where 
    S: SimdScalarFloat
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Point<S, N>> for Isometry<S, N> 
where 
    S: SimdScalarFloat
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for &Isometry<S, N> 
where 
    S: SimdScalarFloat
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Point<S, N>> for &'b Isometry<S, N> 
where 
    S: SimdScalarFloat
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &'a Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Vector<S, N>> for Isometry<S, N> 
where 
    S: SimdScalarFloat
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Vector<S, N>> for Isometry<S, N> 
where 
    S: SimdScalarFloat
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: &Vector<S, N>) -> Self::Output {
        self.apply_vector(other)
    }
}

impl<S, const N: usize> ops::Mul<Vector<S, N>> for &Isometry<S, N> 
where 
    S: SimdScalarFloat
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Vector<S, N>> for &'b Isometry<S, N> 
where 
    S: SimdScalarFloat
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: &'a Vector<S, N>) -> Self::Output {
        self.apply_vector(other)
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<Isometry<S, N>> for Isometry<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: Isometry<S, N>) -> Self::Output {
        let shift = self.rotation.apply_vector(&other.translation.vector);

        Isometry::from_parts(
            &Translation::from_vector(&(self.translation.vector + shift)),
            &(self.rotation * other.rotation)
        )
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<&Isometry<S, N>> for Isometry<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: &Isometry<S, N>) -> Self::Output {
        let shift = self.rotation.apply_vector(&other.translation.vector);

        Isometry::from_parts(
            &Translation::from_vector(&(self.translation.vector + shift)),
            &(self.rotation * other.rotation)
        )
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<Isometry<S, N>> for &Isometry<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: Isometry<S, N>) -> Self::Output {
        let shift = self.rotation.apply_vector(&other.translation.vector);

        Isometry::from_parts(
            &Translation::from_vector(&(self.translation.vector + shift)),
            &(self.rotation * other.rotation)
        )
    }
}

impl<'a, 'b, S, const N: usize, const NN: usize> ops::Mul<&'a Isometry<S, N>> for &'b Isometry<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: &'a Isometry<S, N>) -> Self::Output {
        let shift = self.rotation.apply_vector(&other.translation.vector);

        Isometry::from_parts(
            &Translation::from_vector(&(self.translation.vector + shift)),
            &(self.rotation * other.rotation)
        )
    }
}


impl<S> Isometry2<S> 
where 
    S: SimdScalarFloat 
{
    /// Construct a new isometry from a rotation angle and a displacement vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2, 
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
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
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
    /// let result = isometry.apply_point(&point);
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
    /// let result = isometry.apply_point(&point);
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
    /// let result = isometry.apply_point(&point);
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
}

impl<S> Isometry3<S> 
where 
    S: SimdScalarFloat
{
    /// Construct a new isometry from a rotation axis, rotation angle, and a 
    /// displacement vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
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

    /// Construct a new isometry from a rotation axis and a rotation angle.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
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
    /// # use cglinalg_trigonometry::{
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
    /// # use cglinalg_trigonometry::{
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
    /// # use cglinalg_trigonometry::{
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
    /// let result = isometry.apply_vector(&vector);
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
    /// let result = isometry.apply_vector(&vector);
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
    ///     isometry.apply_point(&eye), 
    ///     origin, 
    ///     epsilon = 1e-10,
    /// );
    /// assert_relative_eq!(
    ///     isometry.apply_vector(&direction).normalize(), 
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
    ///     isometry.apply_point(&eye), 
    ///     origin, 
    ///     epsilon = 1e-10,
    /// );
    /// assert_relative_eq!(
    ///     isometry.apply_vector(&direction).normalize(), 
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
    ///     isometry.apply_vector(&direction).normalize(), 
    ///     unit_z, 
    ///     epsilon = 1e-10,
    /// );
    /// assert_relative_eq!(
    ///     isometry.apply_point(&eye), 
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
    ///     isometry.apply_vector(&direction).normalize(), 
    ///     minus_unit_z, 
    ///     epsilon = 1e-10,
    /// );
    /// assert_relative_eq!(
    ///     isometry.apply_point(&eye), 
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
    ///     isometry.apply_vector(&unit_z).normalize(), 
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
    ///     isometry.apply_vector(&minus_unit_z), 
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
    ///     isometry.apply_vector(&unit_z), 
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
    ///     isometry.apply_vector(&minus_unit_z), 
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
}

