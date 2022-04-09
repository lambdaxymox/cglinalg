use crate::rotation::{
    Rotation2,
    Rotation3,
};
use crate::translation::{
    Translation2,
    Translation3,
};
use crate::common::{
    ScalarFloat,
    Unit,
};
use crate::matrix::{
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
use crate::angle::{
    Radians,
};
use crate::transform::{
    Transform2,
    Transform3,
};
use crate::isometry::{
    Isometry2,
    Isometry3,
};

use core::fmt;
use core::ops;


/// A similarity transformation is a transformation consisting of a scaling,
/// a rotation, and a translation. The similarity transformation applies the
/// scaling, followed by the rotation, and finally the translation.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Similarity2<S> {
    /// The underlying rigid body transformation.
    isometry: Isometry2<S>,
    /// The uniform scaling factor.
    scale: S,
}

impl<S> Similarity2<S> 
where 
    S: ScalarFloat 
{
    /// Construct a similarity transformation directly from the scale, rotation,
    /// and translation parts.
    #[inline]
    pub fn from_parts(translation: &Translation2<S>, rotation: &Rotation2<S>, scale: S) -> Self {
        let isometry = Isometry2::from_parts(translation, rotation);
        
        Self { isometry, scale }
    }

    /// Construct a similarity transformation from a rotation only.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Vector2,
    /// #     Radians, 
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
    /// let result = similarity.transform_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn from_rotation(rotation: &Rotation2<S>) -> Self {
        let isometry = Isometry2::from_rotation(rotation);

        Self {
            isometry,
            scale: S::one(),
        }
    }

    /// Construct a similarity transformation from a scale only.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity2,
    /// #     Vector2,
    /// # };
    /// #
    /// let scale = 10_f64;
    /// let similarity = Similarity2::from_scale(scale);
    /// let vector = Vector2::new(1_f64, 2_f64);
    /// let expected = Vector2::new(10_f64, 20_f64);
    /// let result = similarity.transform_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_scale(scale: S) -> Self {
        let isometry = Isometry2::identity();

        Self { isometry, scale }
    }

    /// Construct a similarity transformation from a translation only.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity2,
    /// #     Translation2,
    /// #     Point2,
    /// # };
    /// #
    /// let translation = Translation2::new(1_f64, 2_f64);
    /// let similarity = Similarity2::from_translation(&translation);
    /// let point = Point2::new(5_f64, 5_f64);
    /// let expected = Point2::new(6_f64, 7_f64);
    /// let result = similarity.transform_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_translation(translation: &Translation2<S>) -> Self {
        let isometry = Isometry2::from_translation(translation);

        Self {
            isometry,
            scale: S::one(),
        }
    }

    /// Construct a similarity transformation from an isometry.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity2,
    /// #     Isometry2,
    /// #     Vector2,
    /// #     Point2,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_3);
    /// let distance = Vector2::new(5_f64, 5_f64);
    /// let isometry = Isometry2::from_angle_translation(angle, &distance);
    /// let similarity = Similarity2::from_isometry(isometry);
    /// let point = Point2::new(2_f64, 0_f64);
    /// let expected = Point2::new(6_f64, f64::sqrt(3_f64) + 5_f64);
    /// let result = similarity.transform_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn from_isometry(isometry: Isometry2<S>) -> Self {
        Self {
            isometry,
            scale: S::one(),
        }
    }

    /// Construct a two-dimensional similarity transformation from a rotation
    /// angle.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity2,
    /// #     Vector2,
    /// #     Degrees, 
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
    /// let result = similarity.transform_vector(&unit_x);
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

    /// Convert a similarity transformation to an affine matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// #     Matrix3x3,
    /// #     Vector2,
    /// #     Angle,
    /// #     Degrees,
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
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix3x3<S> {
        let distance = self.isometry.translation().as_ref();
        let scale = self.scale;
        let rotation = self.isometry.rotation().matrix();

        Matrix3x3::new(
            scale * rotation.c0r0, scale * rotation.c0r1, S::zero(),
            scale * rotation.c1r0, scale * rotation.c1r1, S::zero(),
            distance.x,            distance.y,            S::one()
        )
    }
    
    /// Get the uniform scale factor of the similarity transformation.
    #[inline]
    pub fn scale(&self) -> S {
        self.scale
    }

    /// Get the rotation part of the similarity transformation.
    #[inline]
    pub fn rotation(&self) -> &Rotation2<S> {
        self.isometry.rotation()
    }

    /// Get the translation part of the similarity transformation.
    #[inline]
    pub fn translation(&self) -> &Translation2<S> {
        self.isometry.translation()
    }

    /// Construct an identity transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity2,
    /// #     Point2, 
    /// # };
    /// #
    /// let similarity = Similarity2::identity();
    /// let point = Point2::new(1_f64, 2_f64);
    ///
    /// assert_eq!(similarity * point, point);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self {
            isometry: Isometry2::identity(),
            scale: S::one(),
        }
    }

    /// Convert a similarity transformation to a generic transformation.
    #[inline]
    pub fn to_transform(&self) -> Transform2<S> {
        let matrix = self.to_affine_matrix();
        Transform2::from_specialized(matrix)
    }

    /// Calculate the inverse of the similarity transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// #     Degrees,
    /// #     Vector2,
    /// #     Point2,
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
    /// let transformed_point = similarity.transform_point(&point);
    /// let result = similarity_inv.transform_point(&transformed_point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse(&self) -> Self {
        let mut similarity_inv = *self;
        similarity_inv.inverse_mut();

        similarity_inv
    }

    /// Mutably invert a similarity transformation in place..
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// #     Degrees,
    /// #     Vector2,
    /// #     Point2,
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
    /// let transformed_point = similarity.transform_point(&point);
    /// let result = similarity_mut.transform_point(&transformed_point);
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// #     Vector2,
    /// #     Point2,
    /// #     Radians,
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
    /// let transformed_point = similarity.transform_point(&point);
    /// let result = similarity.inverse_transform_point(&transformed_point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_transform_point(&self, point: &Point2<S>) -> Point2<S> {
        self.isometry.inverse_transform_point(point) / self.scale
    }
    
    /// Apply the inverse of a similarity transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// #     Vector2,
    /// #     Radians,
    /// #     Unit, 
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
    /// let transformed_vector = similarity.transform_vector(&vector);
    /// let result = similarity.inverse_transform_vector(&transformed_vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.isometry.inverse_transform_vector(vector) / self.scale
    }

    /// Apply the similarity transformation to a point.
    ///
    /// The transformation applies the scaling, followed by the rotation,
    /// and finally the translation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// #     Vector2,
    /// #     Point2,
    /// #     Radians,
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
    /// let result = similarity.transform_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        let scaled_point = point * self.scale;
        
        self.isometry.transform_point(&scaled_point)
    }

    /// Apply the similarity transformation to a vector.
    ///
    /// The transformation applies the scaling, followed by the rotation,
    /// and finally the translation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity2,
    /// #     Rotation2,
    /// #     Translation2,
    /// #     Vector2,
    /// #     Radians,
    /// #     Unit, 
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
    /// let result = similarity.transform_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        let scaled_vector = vector * self.scale;
        
        self.isometry.transform_vector(&scaled_vector)
    }

}

impl<S> approx::AbsDiffEq for Similarity2<S> 
where 
    S: ScalarFloat 
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Isometry2::abs_diff_eq(&self.isometry, &other.isometry, epsilon) 
            && S::abs_diff_eq(&self.scale, &other.scale, epsilon)
    }
}

impl<S> approx::RelativeEq for Similarity2<S> 
where 
    S: ScalarFloat 
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Isometry2::relative_eq(&self.isometry, &other.isometry, epsilon, max_relative) 
            && S::relative_eq(&self.scale, &other.scale, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Similarity2<S> 
where 
    S: ScalarFloat 
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Isometry2::ulps_eq(&self.isometry, &other.isometry, epsilon, max_ulps) 
            && S::ulps_eq(&self.scale, &other.scale, epsilon, max_ulps)
    }
}

impl<S> fmt::Display for Similarity2<S> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Similarity2 [scale={}, rotation={}, translation={}]",
            self.scale, self.isometry.rotation, self.isometry.translation.vector
        )
    }
}

impl<S> ops::Mul<Point2<S>> for Similarity2<S> 
where 
    S: ScalarFloat 
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<S> ops::Mul<&Point2<S>> for Similarity2<S> 
where 
    S: ScalarFloat 
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: &Point2<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Point2<S>> for &Similarity2<S> 
where 
    S: ScalarFloat 
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point2<S>> for &'b Similarity2<S> 
where 
    S: ScalarFloat 
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: &'a Point2<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Isometry2<S>> for Similarity2<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity2<S>;

    #[inline]
    fn mul(self, other: Isometry2<S>) -> Self::Output {
        let shift = self.isometry.rotation.rotate_vector(&other.translation.vector) * self.scale();
        let translation = Translation2::from_vector(&(self.isometry.translation.vector + shift));
        let rotation = self.isometry.rotation * other.rotation;

        Similarity2::from_parts(&translation, &rotation, self.scale())
    }
}

impl<S> ops::Mul<&Isometry2<S>> for Similarity2<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity2<S>;

    #[inline]
    fn mul(self, other: &Isometry2<S>) -> Self::Output {
        let shift = self.isometry.rotation.rotate_vector(&other.translation.vector) * self.scale();
        let translation = Translation2::from_vector(&(self.isometry.translation.vector + shift));
        let rotation = self.isometry.rotation * other.rotation;

        Similarity2::from_parts(&translation, &rotation, self.scale())
    }
}

impl<S> ops::Mul<Isometry2<S>> for &Similarity2<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity2<S>;

    #[inline]
    fn mul(self, other: Isometry2<S>) -> Self::Output {
        let shift = self.isometry.rotation.rotate_vector(&other.translation.vector) * self.scale();
        let translation = Translation2::from_vector(&(self.isometry.translation.vector + shift));
        let rotation = self.isometry.rotation * other.rotation;

        Similarity2::from_parts(&translation, &rotation, self.scale())
    }
}

impl<'a, 'b, S> ops::Mul<&'a Isometry2<S>> for &'b Similarity2<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity2<S>;

    #[inline]
    fn mul(self, other: &'a Isometry2<S>) -> Self::Output {
        let shift = self.isometry.rotation.rotate_vector(&other.translation.vector) * self.scale();
        let translation = Translation2::from_vector(&(self.isometry.translation.vector + shift));
        let rotation = self.isometry.rotation * other.rotation;

        Similarity2::from_parts(&translation, &rotation, self.scale())
    }
}

impl<S> ops::Mul<Similarity2<S>> for Similarity2<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity2<S>;

    #[inline]
    fn mul(self, other: Similarity2<S>) -> Self::Output {
        let mut result = self * other.isometry;
        result.scale *= other.scale();

        result
    }
}

impl<S> ops::Mul<&Similarity2<S>> for Similarity2<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity2<S>;

    #[inline]
    fn mul(self, other: &Similarity2<S>) -> Self::Output {
        let mut result = self * other.isometry;
        result.scale *= other.scale();

        result
    }
}

impl<S> ops::Mul<Similarity2<S>> for &Similarity2<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity2<S>;

    #[inline]
    fn mul(self, other: Similarity2<S>) -> Self::Output {
        let mut result = self * other.isometry;
        result.scale *= other.scale();

        result
    }
}

impl<'a, 'b, S> ops::Mul<&'a Similarity2<S>> for &'b Similarity2<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity2<S>;

    #[inline]
    fn mul(self, other: &'a Similarity2<S>) -> Self::Output {
        let mut result = self * other.isometry;
        result.scale *= other.scale();

        result
    }
}



/// A similarity transformation is a transformation consisting of a scaling,
/// a rotation, and a translation. The similarity transformation applies the
/// scaling, followed by the rotation, and finally the translation.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Similarity3<S> {
    /// The underlying rigid body transformation of the similarity transformation.
    isometry: Isometry3<S>,
    /// The uniform scale factor of the similarity transformation.
    scale: S,
}

impl<S> Similarity3<S> 
where 
    S: ScalarFloat 
{
    /// Construct a similarity transformation directly from the scale, rotation,
    /// and translation parts.
    #[inline]
    pub fn from_parts(translation: &Translation3<S>, rotation: &Rotation3<S>, scale: S) -> Self {
        let isometry = Isometry3::from_parts(translation, rotation);
        
        Self { isometry, scale }
    }

    /// Construct a similarity transformation from a rotation only.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Vector3,
    /// #     Radians,
    /// #     Unit, 
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
    /// let result = similarity.transform_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn from_rotation(rotation: &Rotation3<S>) -> Self {
        let isometry = Isometry3::from_rotation(rotation);

        Self {
            isometry,
            scale: S::one(),
        }
    }

    /// Construct a similarity transformation from a scale factor only.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Vector3, 
    /// # }; 
    /// #
    /// let scale = 15_f64;
    /// let similarity = Similarity3::from_scale(scale);
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Vector3::new(15_f64, 30_f64, 45_f64);
    /// let result = similarity.transform_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_scale(scale: S) -> Self {
        let isometry = Isometry3::identity();

        Self { isometry, scale }
    }

    /// Construct a similarity transformation from a translation only.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Translation3,
    /// #     Vector3,
    /// #     Point3, 
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
    pub fn from_translation(translation: &Translation3<S>) -> Self {
        let isometry = Isometry3::from_translation(translation);

        Self {
            isometry,
            scale: S::one(),
        }
    }

    /// Construct a similarity transformation from an isometry.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Isometry3,
    /// #     Vector3,
    /// #     Point3,
    /// #     Radians,
    /// #     Unit,
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
    /// let result = similarity.transform_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn from_isometry(isometry: &Isometry3<S>) -> Self {
        Self {
            isometry: *isometry,
            scale: S::one(),
        }
    }

    /// Construct a similarity transformation from the axis and angle
    /// of a rotation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Vector3,
    /// #     Unit,
    /// #     Radians, 
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
    /// let result = similarity.transform_vector(&vector);
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
    /// of an observer located at the origin facing the **z-axis** into a coordinate 
    /// system of an observer located at the position `eye` facing the direction 
    /// `direction`.
    ///
    /// The similarity transformation maps the **z-axis** to the direction 
    /// of `target - eye`, and locates the origin of the coordinate system to 
    /// the `eye` position.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Magnitude,
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(1_f64, -1_f64, 1_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let isometry = Similarity3::face_towards(&eye, &target, &up);
    /// let unit_z = Vector3::unit_z();
    /// let direction = (target - eye).normalize();
    ///
    /// assert_eq!(isometry.transform_vector(&unit_z), direction);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn face_towards(
        eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self
    {
        let isometry = Isometry3::face_towards(eye, target, up);
    
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
    /// **left-handed** coordinate transformation. It is conventionally used in 
    /// computer graphics for camera view transformations.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,
    /// #     Similarity3,
    /// #     Point3,
    /// #     Magnitude,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # 
    /// let target = Point3::new(0_f64, 6_f64, 0_f64);
    /// let up: Vector3<f64> = Vector3::unit_x();
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let similarity = Similarity3::look_at_lh(&eye, &target, &up);
    /// let result = similarity.transform_vector(&(target - eye).normalize());
    /// let expected = Vector3::unit_z();
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// assert_eq!(similarity.transform_point(&eye), Point3::origin());
    /// ```
    #[inline]
    pub fn look_at_lh(
        eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self
    {      
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
    /// It is conventionally used in computer graphics for camera view 
    /// transformations.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,
    /// #     Similarity3,
    /// #     Point3,
    /// #     Magnitude,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # 
    /// let target = Point3::new(0_f64, 6_f64, 0_f64);
    /// let up: Vector3<f64> = Vector3::unit_x();
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let similarity = Similarity3::look_at_rh(&eye, &target, &up);
    /// let result = similarity.transform_vector(&(target - eye).normalize());
    /// let expected = -Vector3::unit_z();
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// assert_eq!(similarity.transform_point(&eye), Point3::origin());
    /// ```
    #[inline]
    pub fn look_at_rh(
        eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
              
        let isometry = Isometry3::look_at_rh(eye, target, up);
    
        Self::from_isometry(&isometry)
    }

    /// Convert a similarity transformation to an affine matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Angle,
    /// #     Degrees,
    /// #     Unit,
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix4x4<S> {
        let distance = self.isometry.translation().as_ref();
        let scale = self.scale;
        let rotation = self.isometry.rotation().matrix();

        Matrix4x4::new(
            scale * rotation.c0r0, scale * rotation.c0r1, scale * rotation.c0r2, S::zero(),
            scale * rotation.c1r0, scale * rotation.c1r1, scale * rotation.c1r2, S::zero(),
            scale * rotation.c2r0, scale * rotation.c2r1, scale * rotation.c2r2, S::zero(),
            distance.x,            distance.y,            distance.z,            S::one()
        )
    }
    
    /// Get the uniform scale factor of the similarity transformation.
    #[inline]
    pub fn scale(&self) -> S {
        self.scale
    }

    /// Get the rotation part of the similarity transformation.
    #[inline]
    pub fn rotation(&self) -> &Rotation3<S> {
        self.isometry.rotation()
    }

    /// Get the translation part of the similarity transformation.
    #[inline]
    pub fn translation(&self) -> &Translation3<S> {
        self.isometry.translation()
    }

    /// Construct an identity transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Point3, 
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
            isometry: Isometry3::identity(),
            scale: S::one(),
        }
    }

    /// Convert a similarity transformation to a generic transformation.
    #[inline]
    pub fn to_transform(&self) -> Transform3<S> {
        let matrix = self.to_affine_matrix();
        Transform3::from_specialized(matrix)
    }

    /// Calculate the inverse of the similarity transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// #     Degrees,
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
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
    /// let transformed_point = similarity.transform_point(&point);
    /// let result = similarity_inv.transform_point(&transformed_point);
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// #     Degrees,
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
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
    /// let transformed_point = similarity.transform_point(&point);
    /// let result = similarity_mut.transform_point(&transformed_point);
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// #     Vector3,
    /// #     Point3,
    /// #     Radians,
    /// #     Unit, 
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
    /// let transformed_point = similarity.transform_point(&point);
    /// let result = similarity.inverse_transform_point(&transformed_point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_transform_point(&self, point: &Point3<S>) -> Point3<S> {
        self.isometry.inverse_transform_point(point) / self.scale
    }
    
    /// Apply the inverse of a similarity transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// #     Vector3,
    /// #     Radians,
    /// #     Unit, 
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
    /// let transformed_vector = similarity.transform_vector(&vector);
    /// let result = similarity.inverse_transform_vector(&transformed_vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.isometry.inverse_transform_vector(vector) / self.scale
    }

    /// Apply a similarity transformation to a point.
    ///
    /// The transformation applies the scaling, followed by the rotation,
    /// and finally the translation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// #     Vector3,
    /// #     Point3,
    /// #     Radians,
    /// #     Unit, 
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
    /// let result = similarity.transform_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        let scaled_point = point * self.scale;
        
        self.isometry.transform_point(&scaled_point)
    }

    /// Apply a similarity transformation to a vector.
    ///
    /// The transformation applies the scaling, followed by the rotation,
    /// and finally the translation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Similarity3,
    /// #     Rotation3,
    /// #     Translation3,
    /// #     Vector3,
    /// #     Radians,
    /// #     Unit, 
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
    /// let result = similarity.transform_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let scaled_vector = vector * self.scale;
        
        self.isometry.transform_vector(&scaled_vector)
    }

}

impl<S> fmt::Display for Similarity3<S> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Similarity3 [scale={}, rotation={}, translation={}]",
            self.scale, self.isometry.rotation, self.isometry.translation.vector
        )
    }
}

impl<S> approx::AbsDiffEq for Similarity3<S> 
where 
    S: ScalarFloat 
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Isometry3::abs_diff_eq(&self.isometry, &other.isometry, epsilon) 
            && S::abs_diff_eq(&self.scale, &other.scale, epsilon)
    }
}

impl<S> approx::RelativeEq for Similarity3<S> 
where 
    S: ScalarFloat 
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Isometry3::relative_eq(&self.isometry, &other.isometry, epsilon, max_relative) 
            && S::relative_eq(&self.scale, &other.scale, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Similarity3<S> 
where 
    S: ScalarFloat 
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Isometry3::ulps_eq(&self.isometry, &other.isometry, epsilon, max_ulps) 
            && S::ulps_eq(&self.scale, &other.scale, epsilon, max_ulps)
    }
}

impl<S> ops::Mul<Point3<S>> for Similarity3<S> 
where 
    S: ScalarFloat 
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for Similarity3<S> 
where 
    S: ScalarFloat 
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &Similarity3<S> 
where 
    S: ScalarFloat 
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b Similarity3<S> 
where 
    S: ScalarFloat 
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Isometry3<S>> for Similarity3<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity3<S>;

    #[inline]
    fn mul(self, other: Isometry3<S>) -> Self::Output {
        let shift = self.isometry.rotation.rotate_vector(&other.translation.vector) * self.scale();
        let translation = Translation3::from_vector(&(self.isometry.translation.vector + shift));
        let rotation = self.isometry.rotation * other.rotation;

        Similarity3::from_parts(&translation, &rotation, self.scale())
    }
}

impl<S> ops::Mul<&Isometry3<S>> for Similarity3<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity3<S>;

    #[inline]
    fn mul(self, other: &Isometry3<S>) -> Self::Output {
        let shift = self.isometry.rotation.rotate_vector(&other.translation.vector) * self.scale();
        let translation = Translation3::from_vector(&(self.isometry.translation.vector + shift));
        let rotation = self.isometry.rotation * other.rotation;

        Similarity3::from_parts(&translation, &rotation, self.scale())
    }
}

impl<S> ops::Mul<Isometry3<S>> for &Similarity3<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity3<S>;

    #[inline]
    fn mul(self, other: Isometry3<S>) -> Self::Output {
        let shift = self.isometry.rotation.rotate_vector(&other.translation.vector) * self.scale();
        let translation = Translation3::from_vector(&(self.isometry.translation.vector + shift));
        let rotation = self.isometry.rotation * other.rotation;

        Similarity3::from_parts(&translation, &rotation, self.scale())
    }
}

impl<'a, 'b, S> ops::Mul<&'a Isometry3<S>> for &'b Similarity3<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity3<S>;

    #[inline]
    fn mul(self, other: &'a Isometry3<S>) -> Self::Output {
        let shift = self.isometry.rotation.rotate_vector(&other.translation.vector) * self.scale();
        let translation = Translation3::from_vector(&(self.isometry.translation.vector + shift));
        let rotation = self.isometry.rotation * other.rotation;

        Similarity3::from_parts(&translation, &rotation, self.scale())
    }
}

impl<S> ops::Mul<Similarity3<S>> for Similarity3<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity3<S>;

    #[inline]
    fn mul(self, other: Similarity3<S>) -> Self::Output {
        let mut result = self * other.isometry;
        result.scale *= other.scale();

        result
    }
}

impl<S> ops::Mul<&Similarity3<S>> for Similarity3<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity3<S>;

    #[inline]
    fn mul(self, other: &Similarity3<S>) -> Self::Output {
        let mut result = self * other.isometry;
        result.scale *= other.scale();

        result
    }
}

impl<S> ops::Mul<Similarity3<S>> for &Similarity3<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity3<S>;

    #[inline]
    fn mul(self, other: Similarity3<S>) -> Self::Output {
        let mut result = self * other.isometry;
        result.scale *= other.scale();

        result
    }
}

impl<'a, 'b, S> ops::Mul<&'a Similarity3<S>> for &'b Similarity3<S> 
where 
    S: ScalarFloat 
{
    type Output = Similarity3<S>;

    #[inline]
    fn mul(self, other: &'a Similarity3<S>) -> Self::Output {
        let mut result = self * other.isometry;
        result.scale *= other.scale();

        result
    }
}

