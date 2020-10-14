use crate::rotation::{
    Rotation2,
    Rotation3,
};
use crate::translation::{
    Translation2,
    Translation3,
};
use crate::scalar::{
    ScalarFloat,
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
    Angle,
};
use crate::transform::{
    Transform2,
    Transform3,
};
use crate::unit::{
    Unit,
};
use crate::traits::{
    DotProduct,
};

use core::fmt;
use core::ops;


#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Isometry2<S> {
    rotation: Rotation2<S>,
    translation: Translation2<S>,
}

impl<S> Isometry2<S> where S: ScalarFloat {
    #[inline]
    pub fn from_parts(translation: Translation2<S>, rotation: Rotation2<S>) -> Isometry2<S> {
        Isometry2 {
            rotation: rotation,
            translation: translation,
        }
    }

    #[inline]
    pub fn from_translation(translation: Translation2<S>) -> Isometry2<S> {
        Self::from_parts(translation, Rotation2::identity())
    }

    #[inline]
    pub fn from_rotation(rotation: Rotation2<S>) -> Isometry2<S> {
        Self::from_parts(Translation2::identity(), rotation)
    }

    #[inline]
    pub fn from_angle_translation<A: Into<Radians<S>>>(angle: A, distance: &Vector2<S>) -> Isometry2<S>
    {
        Isometry2 {
            rotation: Rotation2::from_angle(angle),
            translation: Translation2::from_vector(distance),
        }
    }

    #[inline]
    pub fn from_angle<A: Into<Radians<S>>>(angle: A) -> Isometry2<S> {
        let translation = Translation2::identity();
        let rotation = Rotation2::from_angle(angle);
        
        Self::from_parts(translation, rotation)
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between two unit vectors.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Isometry2,
    /// #     Point2,
    /// #     Vector2,
    /// #     Unit, 
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
    pub fn rotation_between_axis(a: &Unit<Vector2<S>>, b: &Unit<Vector2<S>>) -> Isometry2<S> {
        let unit_a = a.as_ref();
        let unit_b = b.as_ref();
        let cos_angle = unit_a.dot(unit_b);
        let sin_angle = unit_a.x * unit_b.y - unit_a.y * unit_b.x;

        Isometry2::from_angle(Radians::atan2(sin_angle, cos_angle))
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between vectors.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Isometry2,
    /// #     Point2,
    /// #     Vector2, 
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
    pub fn rotation_between(a: &Vector2<S>, b: &Vector2<S>) -> Isometry2<S> {
        if let (Some(unit_a), Some(unit_b)) = (
            Unit::try_from_value(*a, S::zero()), 
            Unit::try_from_value(*b, S::zero()))
        {
            Self::rotation_between_axis(&unit_a, &unit_b)
        } else {
            Self::identity()
        }
    }

    #[inline]
    pub fn to_transform2d(&self) -> Transform2<S> {
        let matrix = self.to_affine_matrix();
        Transform2::from_specialized(matrix)
    }

    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix3x3<S> {
        let zero = S::zero();
        let one = S::one();
        let rotation_matrix = self.rotation.matrix();
        let translation = self.translation.as_ref();

        Matrix3x3::new(
            rotation_matrix.c0r0, rotation_matrix.c0r1, zero,
            rotation_matrix.c1r0, rotation_matrix.c1r1, zero,
            translation[0], translation[1], one
        )
    }
    
    #[inline]
    pub fn rotation(&self) -> &Rotation2<S> {
        &self.rotation
    }

    #[inline]
    pub fn translation(&self) -> &Translation2<S> {
        &self.translation
    }

    #[inline]
    pub fn inverse(&self) -> Isometry2<S> {
        Isometry2 {
            rotation: self.rotation.inverse(),
            translation: self.translation.inverse(),
        }
    }

    /// Apply a rotation followed by a translation.
    #[inline]
    pub fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        let rotated_point = self.rotation.rotate_point(&point);

        self.translation.translate_point(&rotated_point)
    }

    /// Apply a rotation followed by a translation.
    #[inline]
    pub fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        let rotated_vector = self.rotation.rotate_vector(vector);
        
        self.translation.translate_vector(&rotated_vector)
    }

    #[inline]
    pub fn inverse_transform_point(&self, point: &Point2<S>) -> Point2<S> {
        let rotated_point = self.rotation.inverse_rotate_point(point);

        self.translation.inverse_translate_point(&rotated_point)
    }
    
    #[inline]
    pub fn inverse_transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        let rotated_vector = self.rotation.inverse_rotate_vector(vector);

        self.translation.inverse_translate_vector(&rotated_vector)
    }

    #[inline]
    pub fn identity() -> Isometry2<S> {
        Isometry2 {
            rotation: Rotation2::identity(),
            translation: Translation2::identity()
        }
    }
}

impl<S> fmt::Display for Isometry2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Isometry2 [rotation={}, translation={}]", 
            self.rotation, self.translation
        )
    }
}

impl<S> ops::Mul<Point2<S>> for Isometry2<S> where S: ScalarFloat {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<S> ops::Mul<&Point2<S>> for Isometry2<S> where S: ScalarFloat {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: &Point2<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Point2<S>> for &Isometry2<S> where S: ScalarFloat {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point2<S>> for &'b Isometry2<S> where S: ScalarFloat {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: &'a Point2<S>) -> Self::Output {
        self.transform_point(other)
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Isometry3<S> {
    rotation: Rotation3<S>,
    translation: Translation3<S>,
}

impl<S> Isometry3<S> where S: ScalarFloat {
    #[inline]
    pub fn from_parts(translation: Translation3<S>, rotation: Rotation3<S>) -> Isometry3<S> {
        Isometry3 {
            rotation: rotation,
            translation: translation,
        }
    }

    #[inline]
    pub fn from_axis_angle_translation<A: Into<Radians<S>>>(
        axis: &Unit<Vector3<S>>, angle: A, distance: &Vector3<S>) -> Isometry3<S>
    {
        Isometry3 {
            rotation: Rotation3::from_axis_angle(axis, angle),
            translation: Translation3::from_vector(distance),
        }
    }

    #[inline]
    pub fn from_translation(translation: Translation3<S>) -> Isometry3<S> {
        Self::from_parts(translation, Rotation3::identity())
    }

    #[inline]
    pub fn from_rotation(rotation: Rotation3<S>) -> Isometry3<S> {
        Self::from_parts(Translation3::identity(), rotation)
    }

    #[inline]
    pub fn from_axis_angle<A: Into<Radians<S>>>(axis: &Unit<Vector3<S>>, angle: A) -> Isometry3<S> {
        let translation = Translation3::identity();
        let rotation = Rotation3::from_axis_angle(axis, angle);
        
        Self::from_parts(translation, rotation)
    }

    #[inline]
    pub fn from_angle_x<A: Into<Radians<S>>>(angle: A) -> Isometry3<S> {
        let translation = Translation3::identity();
        let rotation = Rotation3::from_angle_x(angle);
        
        Self::from_parts(translation, rotation)
    }

    #[inline]
    pub fn from_angle_y<A: Into<Radians<S>>>(angle: A) -> Isometry3<S> {
        let translation = Translation3::identity();
        let rotation = Rotation3::from_angle_y(angle);
        
        Self::from_parts(translation, rotation)
    }

    #[inline]
    pub fn from_angle_z<A: Into<Radians<S>>>(angle: A) -> Isometry3<S> {
        let translation = Translation3::identity();
        let rotation = Rotation3::from_angle_z(angle);
        
        Self::from_parts(translation, rotation)
    }

    #[inline]
    pub fn to_transform3d(&self) -> Transform3<S> {
        let matrix = self.to_affine_matrix();
        Transform3::from_specialized(matrix)
    }

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
            translation[0], translation[1], translation[2], one
        )
    }
    
    #[inline]
    pub fn rotation(&self) -> &Rotation3<S> {
        &self.rotation
    }

    #[inline]
    pub fn translation(&self) -> &Translation3<S> {
        &self.translation
    }

    #[inline]
    pub fn inverse(&self) -> Isometry3<S> {
        Isometry3 {
            rotation: self.rotation.inverse(),
            translation: self.translation.inverse(),
        }
    }

    /// Apply a rotation followed by a translation.
    #[inline]
    pub fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        let rotated_point = self.rotation.rotate_point(&point);

        self.translation.translate_point(&rotated_point)
    }

    /// Apply a rotation followed by a translation.
    #[inline]
    pub fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let rotated_vector = self.rotation.rotate_vector(vector);
        
        self.translation.translate_vector(&rotated_vector)
    }

    #[inline]
    pub fn inverse_transform_point(&self, point: &Point3<S>) -> Point3<S> {
        let rotated_point = self.rotation.inverse_rotate_point(point);

        self.translation.inverse_translate_point(&rotated_point)
    }
    
    #[inline]
    pub fn inverse_transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let rotated_vector = self.rotation.inverse_rotate_vector(vector);

        self.translation.inverse_translate_vector(&rotated_vector)
    }

    #[inline]
    pub fn identity() -> Isometry3<S> {
        Isometry3 {
            rotation: Rotation3::identity(),
            translation: Translation3::identity()
        }
    }
}

impl<S> fmt::Display for Isometry3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Isometry3 [rotation={}, translation={}]", 
            self.rotation, self.translation
        )
    }
}

impl<S> ops::Mul<Point3<S>> for Isometry3<S> where S: ScalarFloat {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for Isometry3<S> where S: ScalarFloat {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &Isometry3<S> where S: ScalarFloat {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b Isometry3<S> where S: ScalarFloat {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.transform_point(other)
    }
}

