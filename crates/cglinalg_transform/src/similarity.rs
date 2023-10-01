use cglinalg_numeric::{
    SimdScalarFloat,
};
use cglinalg_trigonometry::{
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
    Vector3,
    Unit,
    Normed,
};
use crate::rotation::{
    Rotation,
};
use crate::translation::{
    Translation,
};
use crate::transform::{
    Transform,
};
use crate::isometry::{
    Isometry,
    Isometry2,
    Isometry3,
};

use core::fmt;
use core::ops;


/// A similarity transformation in two dimensions.
pub type Similarity2<S> = Similarity<S, 2>;

/// A similarity transformation in three dimensions.
pub type Similarity3<S> = Similarity<S, 3>;


/// A similarity transformation is a transformation consisting of a scaling,
/// a rotation, and a translation. The similarity transformation applies the
/// scaling, followed by the rotation, and finally the translation.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Similarity<S, const N: usize> {
    /// The underlying rigid body transformation of the similarity transformation.
    isometry: Isometry<S, N>,
    /// The uniform scale factor of the similarity transformation.
    scale: S,
}

impl<S, const N: usize> Similarity<S, N> 
where 
    S: SimdScalarFloat 
{
    /// Construct a similarity transformation directly from the scale, rotation,
    /// and translation parts.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// #     Isometry2,
    /// #     Rotation2,
    /// #     Translation2,
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
    /// let scale = 3_f64;
    /// let angle = Radians(-f64::consts::FRAC_PI_4);
    /// let rotation = Rotation2::from_angle(angle);
    /// let translation = Translation2::new(1_f64, 2_f64);
    /// let similarity = Similarity2::from_parts(&translation, &rotation, scale);
    /// let point = Point2::new(1_f64, 0_f64);
    /// let expected = Point2::new(
    ///     scale * (1_f64 / f64::sqrt(2_f64)) + 1_f64,
    ///     scale * (-1_f64 / f64::sqrt(2_f64)) + 2_f64
    /// );
    /// let result = similarity.apply_point(&point);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Similarity3,
    /// #     Isometry3,
    /// #     Rotation3,
    /// #     Translation3,
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
    /// let scale = 3_f64;
    /// let angle = Radians(-f64::consts::FRAC_PI_4);
    /// let rotation = Rotation3::from_angle_z(angle);
    /// let translation = Translation3::new(1_f64, 2_f64, 3_f64);
    /// let similarity = Similarity3::from_parts(&translation, &rotation, scale);
    /// let point = Point3::new(1_f64, 0_f64, 3_f64);
    /// let expected = Point3::new(
    ///     scale * (1_f64 / f64::sqrt(2_f64)) + 1_f64,
    ///     scale * (-1_f64 / f64::sqrt(2_f64)) + 2_f64,
    ///     scale * 3_f64 + 3_f64
    /// );
    /// let result = similarity.apply_point(&point);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub const fn from_parts(translation: &Translation<S, N>, rotation: &Rotation<S, N>, scale: S) -> Self {
        let isometry = Isometry::from_parts(translation, rotation);
        
        Self { isometry, scale }
    }

    /// Construct a similarity transformation from a rotation only.
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
    /// #     Similarity2,
    /// #     Rotation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// # 
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let rotation = Rotation2::from_angle(angle);
    /// let similarity = Similarity2::from_rotation(&rotation);
    /// let vector = Vector2::new(2_f64, 0_f64);
    /// let expected = Vector2::new(f64::sqrt(2_f64), f64::sqrt(2_f64));
    /// let result = similarity.apply_vector(&vector);
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
    /// #     Similarity3,
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// # 
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let similarity = Similarity3::from_rotation(&rotation);
    /// let vector = Vector3::new(2_f64, 0_f64, 5_f64);
    /// let expected = Vector3::new(f64::sqrt(2_f64), f64::sqrt(2_f64), 5_f64);
    /// let result = similarity.apply_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn from_rotation(rotation: &Rotation<S, N>) -> Self {
        let isometry = Isometry::from_rotation(rotation);

        Self {
            isometry,
            scale: S::one(),
        }
    }

    /// Construct a similarity transformation from a scale factor only.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// # };
    /// #
    /// let scale = 10_f64;
    /// let similarity = Similarity2::from_scale(scale);
    /// let vector = Vector2::new(1_f64, 2_f64);
    /// let expected = Vector2::new(10_f64, 20_f64);
    /// let result = similarity.apply_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3, 
    /// # }; 
    /// # use cglinalg_transform::{
    /// #     Similarity3,
    /// # };
    /// #
    /// let scale = 15_f64;
    /// let similarity = Similarity3::from_scale(scale);
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Vector3::new(15_f64, 30_f64, 45_f64);
    /// let result = similarity.apply_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_scale(scale: S) -> Self {
        let isometry = Isometry::identity();

        Self { isometry, scale }
    }

    /// Construct a similarity transformation from a translation only.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// #     Translation2,
    /// # };
    /// #
    /// let translation = Translation2::new(1_f64, 2_f64);
    /// let similarity = Similarity2::from_translation(&translation);
    /// let point = Point2::new(5_f64, 5_f64);
    /// let expected = Point2::new(6_f64, 7_f64);
    /// let result = similarity.apply_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity3,
    /// #     Translation3,
    /// # };
    /// # 
    /// let distance = Vector3::new(5_f64, 5_f64, 5_f64);
    /// let translation = Translation3::from_vector(&distance);
    /// let similarity = Similarity3::from_translation(&translation);
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    ///
    /// assert_eq!(similarity * point, point + distance);
    /// ```
    #[inline]
    pub fn from_translation(translation: &Translation<S, N>) -> Self {
        let isometry = Isometry::from_translation(translation);

        Self {
            isometry,
            scale: S::one(),
        }
    }

    /// Construct a similarity transformation from an isometry.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Point2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// #     Isometry2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_3);
    /// let distance = Vector2::new(5_f64, 5_f64);
    /// let isometry = Isometry2::from_angle_translation(angle, &distance);
    /// let similarity = Similarity2::from_isometry(&isometry);
    /// let point = Point2::new(2_f64, 0_f64);
    /// let expected = Point2::new(6_f64, f64::sqrt(3_f64) + 5_f64);
    /// let result = similarity.apply_point(&point);
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
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity3,
    /// #     Isometry3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_3);
    /// let distance = Vector3::new(5_f64, 5_f64, 0_f64);
    /// let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    /// let similarity = Similarity3::from_isometry(&isometry);
    /// let point = Point3::new(2_f64, 0_f64, 13_f64);
    /// let expected = Point3::new(6_f64, f64::sqrt(3_f64) + 5_f64, 13_f64);
    /// let result = similarity.apply_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn from_isometry(isometry: &Isometry<S, N>) -> Self {
        Self {
            isometry: *isometry,
            scale: S::one(),
        }
    }

    /// Get the uniform scale factor of the similarity transformation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// #     Isometry2,
    /// #     Rotation2,
    /// #     Translation2,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 3_f64;
    /// let angle = Radians(-f64::consts::FRAC_PI_4);
    /// let rotation = Rotation2::from_angle(angle);
    /// let translation = Translation2::new(1_f64, 2_f64);
    /// let similarity = Similarity2::from_parts(&translation, &rotation, scale);
    /// let expected = scale;
    /// let result = similarity.scale();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Similarity3,
    /// #     Isometry3,
    /// #     Rotation3,
    /// #     Translation3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 3_f64;
    /// let angle = Radians(-f64::consts::FRAC_PI_4);
    /// let rotation = Rotation3::from_angle_z(angle);
    /// let translation = Translation3::new(1_f64, 2_f64, 3_f64);
    /// let similarity = Similarity3::from_parts(&translation, &rotation, scale);
    /// let expected = scale;
    /// let result = similarity.scale();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn scale(&self) -> S {
        self.scale
    }

    /// Get the rotation part of the similarity transformation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// #     Isometry2,
    /// #     Rotation2,
    /// #     Translation2,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 3_f64;
    /// let angle = Radians(-f64::consts::FRAC_PI_4);
    /// let rotation = Rotation2::from_angle(angle);
    /// let translation = Translation2::new(1_f64, 2_f64);
    /// let similarity = Similarity2::from_parts(&translation, &rotation, scale);
    /// let expected = &rotation;
    /// let result = similarity.rotation();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Similarity3,
    /// #     Isometry3,
    /// #     Rotation3,
    /// #     Translation3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 3_f64;
    /// let angle = Radians(-f64::consts::FRAC_PI_4);
    /// let rotation = Rotation3::from_angle_z(angle);
    /// let translation = Translation3::new(1_f64, 2_f64, 3_f64);
    /// let similarity = Similarity3::from_parts(&translation, &rotation, scale);
    /// let expected = &rotation;
    /// let result = similarity.rotation();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn rotation(&self) -> &Rotation<S, N> {
        self.isometry.rotation()
    }

    /// Get the translation part of the similarity transformation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// #     Isometry2,
    /// #     Rotation2,
    /// #     Translation2,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 3_f64;
    /// let angle = Radians(-f64::consts::FRAC_PI_4);
    /// let rotation = Rotation2::from_angle(angle);
    /// let translation = Translation2::new(1_f64, 2_f64);
    /// let similarity = Similarity2::from_parts(&translation, &rotation, scale);
    /// let expected = &translation;
    /// let result = similarity.translation();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Similarity3,
    /// #     Isometry3,
    /// #     Rotation3,
    /// #     Translation3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 3_f64;
    /// let angle = Radians(-f64::consts::FRAC_PI_4);
    /// let rotation = Rotation3::from_angle_z(angle);
    /// let translation = Translation3::new(1_f64, 2_f64, 3_f64);
    /// let similarity = Similarity3::from_parts(&translation, &rotation, scale);
    /// let expected = &translation;
    /// let result = similarity.translation();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn translation(&self) -> &Translation<S, N> {
        self.isometry.translation()
    }

    /// Construct an identity transformation.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// # };
    /// #
    /// let similarity = Similarity2::identity();
    /// let point = Point2::new(1_f64, 2_f64);
    ///
    /// assert_eq!(similarity * point, point);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity3,
    /// # };
    /// #
    /// let similarity = Similarity3::identity();
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    ///
    /// assert_eq!(similarity * point, point);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self {
            isometry: Isometry::identity(),
            scale: S::one(),
        }
    }
}

impl<S, const N: usize> Similarity<S, N> 
where 
    S: SimdScalarFloat 
{
    /// Calculate the inverse of the similarity transformation.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Point2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let scale = 5_f64;
    /// let angle = Degrees(72_f64);
    /// let distance = Vector2::new(1_f64, 2_f64);
    /// let translation = Translation2::from_vector(&distance);
    /// let rotation = Rotation2::from_angle(angle);
    /// let similarity = Similarity2::from_parts(&translation, &rotation, scale);
    /// let similarity_inv = similarity.inverse();
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = point;
    /// let transformed_point = similarity.apply_point(&point);
    /// let result = similarity_inv.apply_point(&transformed_point);
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
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let scale = 5_f64;
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let angle = Degrees(72_f64);
    /// let distance = Vector3::new(6_f64, 7_f64, 8_f64);
    /// let translation = Translation3::from_vector(&distance);
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let similarity = Similarity3::from_parts(&translation, &rotation, scale);
    /// let similarity_inv = similarity.inverse();
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = point;
    /// let transformed_point = similarity.apply_point(&point);
    /// let result = similarity_inv.apply_point(&transformed_point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse(&self) -> Self {
        let mut similarity_inv = *self;
        similarity_inv.inverse_mut();

        similarity_inv
    }

    /// Calculate the inverse of the similarity transformation.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Point2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let scale = 5_f64;
    /// let angle = Degrees(72_f64);
    /// let distance = Vector2::new(1_f64, 2_f64);
    /// let translation = Translation2::from_vector(&distance);
    /// let rotation = Rotation2::from_angle(angle);
    /// let similarity = Similarity2::from_parts(&translation, &rotation, scale);
    /// let mut similarity_mut = similarity;
    /// similarity_mut.inverse_mut();
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = point;
    /// let transformed_point = similarity.apply_point(&point);
    /// let result = similarity_mut.apply_point(&transformed_point);
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
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let scale = 5_f64;
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let angle = Degrees(72_f64);
    /// let distance = Vector3::new(6_f64, 7_f64, 8_f64);
    /// let translation = Translation3::from_vector(&distance);
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let similarity = Similarity3::from_parts(&translation, &rotation, scale);
    /// let mut similarity_mut = similarity;
    /// similarity_mut.inverse_mut();
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = point;
    /// let transformed_point = similarity.apply_point(&point);
    /// let result = similarity_mut.apply_point(&transformed_point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.scale = S::one() / self.scale;
        self.isometry.inverse_mut();
        self.isometry.translation.vector *= self.scale;
    }

    /// Apply the inverse of a similarity transformation to a point.
    ///
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Point2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 12_f64;
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let distance = Vector2::new(2_f64, 2_f64);
    /// let translation = Translation2::from_vector(&distance);
    /// let rotation = Rotation2::from_angle(angle);
    /// let similarity = Similarity2::from_parts(&translation, &rotation, scale);
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = point;
    /// let transformed_point = similarity.apply_point(&point);
    /// let result = similarity.inverse_apply_point(&transformed_point);
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
    /// #     Point3,
    /// #     Unit, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 12_f64;
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let distance = Vector3::new(2_f64, 2_f64, 2_f64);
    /// let translation = Translation3::from_vector(&distance);
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let similarity = Similarity3::from_parts(&translation, &rotation, scale);
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = point;
    /// let transformed_point = similarity.apply_point(&point);
    /// let result = similarity.inverse_apply_point(&transformed_point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_apply_point(&self, point: &Point<S, N>) -> Point<S, N> {
        self.isometry.inverse_apply_point(point) / self.scale
    }
    
    /// Apply the inverse of a similarity transformation to a vector.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Unit, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 12_f64;
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let distance = Vector2::new(1_f64, 1_f64);
    /// let translation = Translation2::from_vector(&distance);
    /// let rotation = Rotation2::from_angle(angle);
    /// let similarity = Similarity2::from_parts(&translation, &rotation, scale);
    /// let vector = Vector2::unit_x();
    /// let expected = vector;
    /// let transformed_vector = similarity.apply_vector(&vector);
    /// let result = similarity.inverse_apply_vector(&transformed_vector);
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
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 12_f64;
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let distance = Vector3::new(1_f64, 1_f64, 1_f64);
    /// let translation = Translation3::from_vector(&distance);
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let similarity = Similarity3::from_parts(&translation, &rotation, scale);
    /// let vector = Vector3::unit_x();
    /// let expected = vector;
    /// let transformed_vector = similarity.apply_vector(&vector);
    /// let result = similarity.inverse_apply_vector(&transformed_vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_apply_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        self.isometry.inverse_apply_vector(vector) / self.scale
    }

    /// Apply a similarity transformation to a point.
    ///
    /// The transformation applies the scaling, followed by the rotation,
    /// and finally the translation.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Point2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 12_f64;
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let distance = Vector2::new(2_f64, 2_f64);
    /// let translation = Translation2::from_vector(&distance);
    /// let rotation = Rotation2::from_angle(angle);
    /// let similarity = Similarity2::from_parts(&translation, &rotation, scale);
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = Point2::new(-22_f64, 14_f64);
    /// let result = similarity.apply_point(&point);
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
    /// #     Point3,
    /// #     Unit, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 12_f64;
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let distance = Vector3::new(2_f64, 2_f64, 2_f64);
    /// let translation = Translation3::from_vector(&distance);
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let similarity = Similarity3::from_parts(&translation, &rotation, scale);
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Point3::new(-22_f64, 14_f64, 38_f64);
    /// let result = similarity.apply_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn apply_point(&self, point: &Point<S, N>) -> Point<S, N> {
        let scaled_point = point * self.scale;
        
        self.isometry.apply_point(&scaled_point)
    }

    /// Apply a similarity transformation to a vector.
    ///
    /// The transformation applies the scaling, followed by the rotation,
    /// and finally the translation.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Unit, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 12_f64;
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let distance = Vector2::new(1_f64, 1_f64);
    /// let translation = Translation2::from_vector(&distance);
    /// let rotation = Rotation2::from_angle(angle);
    /// let similarity = Similarity2::from_parts(&translation, &rotation, scale);
    /// let vector = Vector2::unit_x();
    /// let expected = scale * Vector2::unit_y();
    /// let result = similarity.apply_vector(&vector);
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
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let scale = 12_f64;
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let distance = Vector3::new(1_f64, 1_f64, 1_f64);
    /// let translation = Translation3::from_vector(&distance);
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let similarity = Similarity3::from_parts(&translation, &rotation, scale);
    /// let vector = Vector3::unit_x();
    /// let expected = scale * Vector3::unit_y();
    /// let result = similarity.apply_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn apply_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        let scaled_vector = vector * self.scale;
        
        self.isometry.apply_vector(&scaled_vector)
    }

}

impl<S, const N: usize, const NPLUS1: usize> Similarity<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimLt<Const<N>, Const<NPLUS1>>
{
    /// Convert a similarity transformation to an affine matrix.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let scale = 2_f64;
    /// let angle = Degrees(72_f64);
    /// let rotation = Rotation2::from_angle(angle);
    /// let translation = Translation2::new(2_f64, 3_f64);
    /// let similarity = Similarity2::from_parts(&translation, &rotation, scale);
    /// let expected = Matrix3x3::new(
    ///      scale * angle.cos(), scale * angle.sin(), 0_f64,
    ///     -scale * angle.sin(), scale * angle.cos(), 0_f64,
    ///      2_f64,               3_f64,               1_f64
    /// );
    /// let result = similarity.to_affine_matrix();
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-15);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let scale = 2_f64;
    /// let axis = Unit::from_value(Vector3::new(1_f64, 1_f64, 0_f64));
    /// let angle = Degrees(60_f64);
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let translation = Translation3::new(2_f64, 3_f64, 4_f64);
    /// let similarity = Similarity3::from_parts(&translation, &rotation, scale);
    /// let sq_3_8 = f64::sqrt(3_f64 / 8_f64);
    /// let expected = Matrix4x4::new(
    ///      scale * 3_f64 / 4_f64, scale * 1_f64 / 4_f64, scale * -sq_3_8,       0_f64,
    ///      scale * 1_f64 / 4_f64, scale * 3_f64 / 4_f64, scale *  sq_3_8,       0_f64,
    ///      scale * sq_3_8,        scale * -sq_3_8,       scale * 1_f64 / 2_f64, 0_f64,
    ///      2_f64,                 3_f64,                 4_f64,                 1_f64
    /// );
    /// let result = similarity.to_affine_matrix();
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-15);
    /// ```
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix<S, NPLUS1, NPLUS1> {
        let translation = self.isometry.translation().as_ref();
        let scale = self.scale;
        let mut rotation = self.isometry.rotation().matrix().clone();
        rotation.scale_mut(scale);
        
        let mut result = Matrix::from(rotation);
        for i in 0..N {
            result[N][i] = translation[i];
        }

        result
    }

    /// Convert a similarity transformation to a generic transformation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Similarity2,
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
    /// let scale = 12_f64;
    /// let angle = Radians(f64::consts::FRAC_PI_3);
    /// let translation = Translation2::new(2_f64, 3_f64);
    /// let rotation = Rotation2::from_angle(angle);
    /// let isometry = Isometry2::from_parts(&translation, &rotation);
    /// let similarity = Similarity2::from_parts(&translation, &rotation, scale);
    /// let expected = Transform2::from_matrix_unchecked(Matrix3x3::new(
    ///     scale * (1_f64 / 2_f64),             scale * (f64::sqrt(3_f64) / 2_f64), 0_f64,
    ///     scale * (-f64::sqrt(3_f64) / 2_f64), scale * (1_f64 / 2_f64),            0_f64,
    ///     2_f64,                               3_f64,                              1_f64
    /// ));
    /// let result = similarity.to_transform();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-14);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Similarity3,
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
    /// let scale = 12_f64;
    /// let angle = Radians(f64::consts::FRAC_PI_3);
    /// let translation = Translation3::new(2_f64, 3_f64, 4_f64);
    /// let rotation = Rotation3::from_angle_z(angle);
    /// let isometry = Isometry3::from_parts(&translation, &rotation);
    /// let similarity = Similarity3::from_parts(&translation, &rotation, scale);
    /// let expected = Transform3::from_matrix_unchecked(Matrix4x4::new(
    ///     scale * (1_f64 / 2_f64),             scale * (f64::sqrt(3_f64) / 2_f64), 0_f64,         0_f64,
    ///     scale * (-f64::sqrt(3_f64) / 2_f64), scale * (1_f64 / 2_f64),            0_f64,         0_f64,
    ///     0_f64,                               0_f64,                              scale * 1_f64, 0_f64,
    ///     2_f64,                               3_f64,                              4_f64,         1_f64
    /// ));
    /// let result = similarity.to_transform();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-14);
    /// ```
    #[inline]
    pub fn to_transform(&self) -> Transform<S, N, NPLUS1> {
        let matrix = self.to_affine_matrix();

        Transform::from_matrix_unchecked(matrix)
    }
}

impl<S, const N: usize> fmt::Display for Similarity<S, N> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Similarity{} [scale={}, rotation={}, translation={}]",
            N, self.scale, self.isometry.rotation, self.isometry.translation.vector
        )
    }
}

impl<S, const N: usize, const NPLUS1: usize> From<Similarity<S, N>> for Matrix<S, NPLUS1, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimLt<Const<N>, Const<NPLUS1>>
{
    #[inline]
    fn from(isometry: Similarity<S, N>) -> Matrix<S, NPLUS1, NPLUS1> {
        isometry.to_affine_matrix()
    }
}

impl<S, const N: usize, const NPLUS1: usize> From<&Similarity<S, N>> for Matrix<S, NPLUS1, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimLt<Const<N>, Const<NPLUS1>>
{
    #[inline]
    fn from(isometry: &Similarity<S, N>) -> Matrix<S, NPLUS1, NPLUS1> {
        isometry.to_affine_matrix()
    }
}

impl<S, const N: usize> approx::AbsDiffEq for Similarity<S, N> 
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
        Isometry::abs_diff_eq(&self.isometry, &other.isometry, epsilon) 
            && S::abs_diff_eq(&self.scale, &other.scale, epsilon)
    }
}

impl<S, const N: usize> approx::RelativeEq for Similarity<S, N> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        Isometry::relative_eq(&self.isometry, &other.isometry, epsilon, max_relative) 
            && S::relative_eq(&self.scale, &other.scale, epsilon, max_relative)
    }
}

impl<S, const N: usize> approx::UlpsEq for Similarity<S, N> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        Isometry::ulps_eq(&self.isometry, &other.isometry, epsilon, max_ulps) 
            && S::ulps_eq(&self.scale, &other.scale, epsilon, max_ulps)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for Similarity<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Point<S, N>> for Similarity<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for &Similarity<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Point<S, N>> for &'b Similarity<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &'a Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Vector<S, N>> for Similarity<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Vector<S, N>> for Similarity<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: &Vector<S, N>) -> Self::Output {
        self.apply_vector(other)
    }
}

impl<S, const N: usize> ops::Mul<Vector<S, N>> for &Similarity<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Vector<S, N>> for &'b Similarity<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: &'a Vector<S, N>) -> Self::Output {
        self.apply_vector(other)
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<Isometry<S, N>> for Similarity<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Similarity<S, N>;

    #[inline]
    fn mul(self, other: Isometry<S, N>) -> Self::Output {
        let shift = self.isometry.rotation.apply_vector(&other.translation.vector) * self.scale();
        let translation = Translation::from_vector(&(self.isometry.translation.vector + shift));
        let rotation = self.isometry.rotation * other.rotation;

        Similarity::from_parts(&translation, &rotation, self.scale())
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<&Isometry<S, N>> for Similarity<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Similarity<S, N>;

    #[inline]
    fn mul(self, other: &Isometry<S, N>) -> Self::Output {
        let shift = self.isometry.rotation.apply_vector(&other.translation.vector) * self.scale();
        let translation = Translation::from_vector(&(self.isometry.translation.vector + shift));
        let rotation = self.isometry.rotation * other.rotation;

        Similarity::from_parts(&translation, &rotation, self.scale())
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<Isometry<S, N>> for &Similarity<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Similarity<S, N>;

    #[inline]
    fn mul(self, other: Isometry<S, N>) -> Self::Output {
        let shift = self.isometry.rotation.apply_vector(&other.translation.vector) * self.scale();
        let translation = Translation::from_vector(&(self.isometry.translation.vector + shift));
        let rotation = self.isometry.rotation * other.rotation;

        Similarity::from_parts(&translation, &rotation, self.scale())
    }
}

impl<'a, 'b, S, const N: usize, const NN: usize> ops::Mul<&'a Isometry<S, N>> for &'b Similarity<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Similarity<S, N>;

    #[inline]
    fn mul(self, other: &'a Isometry<S, N>) -> Self::Output {
        let shift = self.isometry.rotation.apply_vector(&other.translation.vector) * self.scale();
        let translation = Translation::from_vector(&(self.isometry.translation.vector + shift));
        let rotation = self.isometry.rotation * other.rotation;

        Similarity::from_parts(&translation, &rotation, self.scale())
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<Similarity<S, N>> for Similarity<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Similarity<S, N>;

    #[inline]
    fn mul(self, other: Similarity<S, N>) -> Self::Output {
        let mut result = self * other.isometry;
        result.scale *= other.scale();

        result
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<&Similarity<S, N>> for Similarity<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Similarity<S, N>;

    #[inline]
    fn mul(self, other: &Similarity<S, N>) -> Self::Output {
        let mut result = self * other.isometry;
        result.scale *= other.scale();

        result
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<Similarity<S, N>> for &Similarity<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Similarity<S, N>;

    #[inline]
    fn mul(self, other: Similarity<S, N>) -> Self::Output {
        let mut result = self * other.isometry;
        result.scale *= other.scale();

        result
    }
}

impl<'a, 'b, S, const N: usize, const NN: usize> ops::Mul<&'a Similarity<S, N>> for &'b Similarity<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Similarity<S, N>;

    #[inline]
    fn mul(self, other: &'a Similarity<S, N>) -> Self::Output {
        let mut result = self * other.isometry;
        result.scale *= other.scale();

        result
    }
}


impl<S> Similarity2<S> 
where 
    S: SimdScalarFloat 
{
    /// Construct a two-dimensional similarity transformation from a rotation
    /// angle.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Similarity2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let angle = Degrees(90_f64);
    /// let similarity = Similarity2::from_angle(angle);
    /// let unit_x = Vector2::unit_x();
    /// let unit_y = Vector2::unit_y();
    /// let expected = unit_y;
    /// let result = similarity.apply_vector(&unit_x);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn from_angle<A: Into<Radians<S>>>(angle: A) -> Self {
        Self {
            isometry: Isometry2::from_angle(angle),
            scale: S::one()
        }
    }
}

impl<S> Similarity3<S> 
where 
    S: SimdScalarFloat 
{
    /// Construct a similarity transformation from the axis and angle
    /// of a rotation.
    ///
    /// # Example
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
    /// #     Similarity3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let similarity = Similarity3::from_axis_angle(&axis, angle);
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Vector3::new(-1_f64 / f64::sqrt(2_f64), 3_f64 / f64::sqrt(2_f64), 3_f64);
    /// let result = similarity.apply_vector(&vector);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn from_axis_angle<A: Into<Radians<S>>>(
        axis: &Unit<Vector3<S>>, angle: A) -> Self {
        
        Self {
            isometry: Isometry3::from_axis_angle(axis, angle),
            scale: S::one()
        }
    }

    /// Construct a similarity transformation that maps the coordinate system 
    /// of an observer located at the position `eye` facing the direction 
    /// `direction` into a coordinate system of an observer located at the 
    /// origin facing the **positive z-axis**. The resulting coordinate 
    /// transformation is a **left-handed** coordinate transformation.
    ///
    /// The similarity transformation maps the direction `direction` to the 
    /// **positive z-axis** to the direction, and locates the position `eye` to
    /// the origin.
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
    /// #     Similarity3,
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
    /// let isometry = Similarity3::look_to_lh(&eye, &direction, &up);
    /// let origin = Point3::origin();
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
        let isometry = Isometry3::look_to_lh(eye, direction, up);
    
        Self::from_isometry(&isometry)
    }

    /// Construct a similarity transformation that maps the coordinate system 
    /// of an observer located at the position `eye` facing the direction 
    /// `direction` into a coordinate system of an observer located at the 
    /// origin facing the **negative z-axis**. The resulting coordinate 
    /// transformation is a **right-handed** coordinate transformation.
    ///
    /// The similarity transformation maps the direction `direction` to the 
    /// **negative z-axis** to the direction, and locates the position `eye` to
    /// the origin.
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
    /// #     Similarity3,
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
    /// let isometry = Similarity3::look_to_rh(&eye, &direction, &up);
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
        let isometry = Isometry3::look_to_rh(eye, direction, up);
    
        Self::from_isometry(&isometry)
    }

    /// Construct an similarity transformation that transforms
    /// a coordinate system of an observer located at the position `eye` facing 
    /// the direction of the target `target` into the coordinate system of an 
    /// observer located at the origin facing the **positive z-axis**.
    ///
    /// The similarity transformation maps the direction along the ray between 
    /// the eye position `eye` and the position of the target `target` to 
    /// the **positive z-axis** and locates the `eye` position at the origin 
    /// in the new the coordinate system. This transformation is a 
    /// **left-handed** coordinate transformation.
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
    /// #     Similarity3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # 
    /// let target = Point3::new(0_f64, 6_f64, 0_f64);
    /// let up: Vector3<f64> = Vector3::unit_x();
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let similarity = Similarity3::look_at_lh(&eye, &target, &up);
    /// let direction = target - eye;
    /// let unit_z = Vector3::unit_z();
    /// let origin = Point3::origin();
    ///
    /// assert_relative_eq!(
    ///     similarity.apply_vector(&direction).normalize(),
    ///     unit_z, 
    ///     epsilon = 1e-10,
    /// );
    /// assert_relative_eq!(
    ///     similarity.apply_point(&eye), 
    ///     origin,
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_at_lh(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {      
        let isometry = Isometry3::look_at_lh(eye, target, up);
    
        Self::from_isometry(&isometry)
    }

    /// Construct an similarity transformation that transforms
    /// a coordinate system of an observer located at the position `eye` facing 
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
    /// #     Similarity3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # 
    /// let target = Point3::new(0_f64, 6_f64, 0_f64);
    /// let up: Vector3<f64> = Vector3::unit_x();
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let similarity = Similarity3::look_at_rh(&eye, &target, &up);
    /// let direction = target - eye;
    /// let minus_unit_z = -Vector3::unit_z();
    /// let origin = Point3::origin();
    ///
    /// assert_relative_eq!(
    ///     similarity.apply_vector(&direction).normalize(), 
    ///     minus_unit_z, 
    ///     epsilon = 1e-10,
    /// );
    /// assert_relative_eq!(
    ///     similarity.apply_point(&eye),
    ///     origin,
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_at_rh(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        let isometry = Isometry3::look_at_rh(eye, target, up);
    
        Self::from_isometry(&isometry)
    }

    /// Construct a similarity transformation that maps the coordinate system 
    /// of an observer located at the origin facing the **positive z-axis** into a 
    /// coordinate system of an observer located at the position `eye` facing the 
    /// direction `direction`. The resulting coordinate transformation is a 
    /// **left-handed** coordinate transformation.
    ///
    /// The similarity transformation maps the direction `direction` to the 
    /// **positive z-axis** to the direction, and locates the position `eye` to
    /// the origin.
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
    /// #     Similarity3,
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
    /// let isometry = Similarity3::look_to_lh_inv(&eye, &direction, &up);
    /// let unit_z = Vector3::unit_z();
    ///
    /// assert_relative_eq!(
    ///     isometry.apply_vector(&unit_z), 
    ///     direction.normalize(),
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_to_lh_inv(eye: &Point3<S>, direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        let isometry = Isometry3::look_to_lh_inv(eye, direction, up);
    
        Self::from_isometry(&isometry)
    }

    /// Construct a similarity transformation that maps the coordinate system 
    /// of an observer located at the origin facing the **negative z-axis** into a 
    /// coordinate system of an observer located at the position `eye` facing the 
    /// direction `direction`. The resulting coordinate transformation is a 
    /// **right-handed** coordinate transformation.
    ///
    /// The similarity transformation maps the **negative z-axis** to the direction 
    /// of `target - eye`, and locates the origin of the coordinate system to 
    /// the `eye` position.
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
    /// #     Similarity3,
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
    /// let isometry = Similarity3::look_to_rh_inv(&eye, &direction, &up);
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
        let isometry = Isometry3::look_to_rh_inv(eye, direction, up);
    
        Self::from_isometry(&isometry)
    }

    /// Construct a similarity transformation that maps the coordinate system 
    /// of an observer located at the origin facing the **positive z-axis** into a 
    /// coordinate system of an observer located at the position `eye` facing the 
    /// direction `direction`. The resulting coordinate transformation is a 
    /// **left-handed** coordinate transformation.
    ///
    /// The similarity transformation maps the direction `direction` to the 
    /// **positive z-axis** to the direction, and locates the position `eye` to
    /// the origin.
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
    /// #     Similarity3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(1_f64, -1_f64, 1_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let isometry = Similarity3::look_at_lh_inv(&eye, &target, &up);
    /// let unit_z = Vector3::unit_z();
    /// let direction = target - eye;
    ///
    /// assert_relative_eq!(
    ///     isometry.apply_vector(&unit_z), 
    ///     direction.normalize(),
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_at_lh_inv(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        let isometry = Isometry3::look_at_lh_inv(eye, target, up);
    
        Self::from_isometry(&isometry)
    }

    /// Construct a similarity transformation that maps the coordinate system 
    /// of an observer located at the origin facing the **negative z-axis** into a 
    /// coordinate system of an observer located at the position `eye` facing the 
    /// direction `direction`. The resulting coordinate transformation is a 
    /// **right-handed** coordinate transformation.
    ///
    /// The similarity transformation maps the **negative z-axis** to the direction 
    /// of `target - eye`, and locates the origin of the coordinate system to 
    /// the `eye` position.
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
    /// #     Similarity3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(1_f64, -1_f64, 1_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let isometry = Similarity3::look_at_rh_inv(&eye, &target, &up);
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
        let isometry = Isometry3::look_at_rh_inv(eye, target, up);
    
        Self::from_isometry(&isometry)
    }
}

