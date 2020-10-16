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
use crate::isometry::{
    Isometry2,
    Isometry3,
};
use crate::unit::{
    Unit,
};
use crate::traits::{
    DotProduct,
};

use approx;

use core::fmt;
use core::ops;


#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Similarity2<S> {
    isometry: Isometry2<S>,
    scale: S,
}

impl<S> Similarity2<S> where S: ScalarFloat {
    #[inline]
    pub fn from_parts(translation: Translation2<S>, rotation: Rotation2<S>, scale: S) -> Similarity2<S> {
        let isometry = Isometry2::from_parts(translation, rotation);
        
        Similarity2 {
            isometry: isometry,
            scale: scale,
        }
    }

    #[inline]
    pub fn from_rotation(rotation: Rotation2<S>) -> Similarity2<S> {
        let isometry = Isometry2::from_rotation(rotation);

        Similarity2 {
            isometry: isometry,
            scale: S::one(),
        }
    }

    #[inline]
    pub fn from_scale(scale: S) -> Similarity2<S> {
        let isometry = Isometry2::identity();

        Similarity2 {
            isometry: isometry,
            scale: scale,
        }
    }

    #[inline]
    pub fn from_translation(translation: Translation2<S>) -> Similarity2<S> {
        let isometry = Isometry2::from_translation(translation);

        Similarity2 {
            isometry: isometry,
            scale: S::one(),
        }
    }

    #[inline]
    pub fn from_isometry(isometry: Isometry2<S>) -> Similarity2<S> {
        Similarity2 {
            isometry: isometry,
            scale: S::one(),
        }
    }

    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix3x3<S> {
        let distance = self.isometry.translation().as_ref();
        let scale = self.scale;
        let rotation = self.isometry.rotation().matrix();

        Matrix3x3::new(
            scale * rotation.c0r0, scale * rotation.c0r1, S::zero(),
            scale * rotation.c1r0, scale * rotation.c1r1, S::zero(),
            distance.x,      distance.y,      S::one()
        )
    }
    
    #[inline]
    pub fn scale(&self) -> S {
        self.scale
    }

    #[inline]
    pub fn rotation(&self) -> &Rotation2<S> {
        self.isometry.rotation()
    }

    #[inline]
    pub fn translation(&self) -> &Translation2<S> {
        self.isometry.translation()
    }

    #[inline]
    pub fn identity() -> Similarity2<S> {
        Similarity2 {
            isometry: Isometry2::identity(),
            scale: S::one(),
        }
    }

    /// Convert a similarity transformation to a generic transformation.
    #[inline]
    pub fn to_transform2d(&self) -> Transform2<S> {
        let matrix = self.to_affine_matrix();
        Transform2::from_specialized(matrix)
    }

    #[inline]
    pub fn inverse(&self) -> Similarity2<S> {
        let mut similarity_inv = self.clone();
        similarity_inv.inverse_mut();

        similarity_inv
    }

    #[inline]
    pub fn inverse_mut(&mut self) {
        self.scale = S::one() / self.scale;
        self.isometry.inverse_mut();
        self.isometry.translation.vector *= self.scale;
    }

    #[inline]
    pub fn inverse_transform_point(&self, point: &Point2<S>) -> Point2<S> {
        self.isometry.inverse_transform_point(point) / self.scale
    }
    
    #[inline]
    pub fn inverse_transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.isometry.inverse_transform_vector(vector) / self.scale
    }

    #[inline]
    pub fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        let scaled_point = point * self.scale;
        
        self.isometry.transform_point(&scaled_point)
    }

    #[inline]
    pub fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        let scaled_vector = vector * self.scale;
        
        self.isometry.transform_vector(&scaled_vector)
    }

}

impl<S> approx::AbsDiffEq for Similarity2<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Similarity2<S> where S: ScalarFloat {
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

impl<S> approx::UlpsEq for Similarity2<S> where S: ScalarFloat {
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

impl<S> fmt::Display for Similarity2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Similarity2 [scale={}, rotation={}, translation={}]",
            self.scale, self.isometry.rotation, self.isometry.translation.vector
        )
    }
}

impl<S> ops::Mul<Point2<S>> for Similarity2<S> where S: ScalarFloat {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<S> ops::Mul<&Point2<S>> for Similarity2<S> where S: ScalarFloat {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: &Point2<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Point2<S>> for &Similarity2<S> where S: ScalarFloat {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point2<S>> for &'b Similarity2<S> where S: ScalarFloat {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: &'a Point2<S>) -> Self::Output {
        self.transform_point(other)
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Similarity3<S> {
    isometry: Isometry3<S>,
    scale: S,
}

impl<S> Similarity3<S> where S: ScalarFloat {
    #[inline]
    pub fn from_parts(translation: Translation3<S>, rotation: Rotation3<S>, scale: S) -> Similarity3<S> {
        let isometry = Isometry3::from_parts(translation, rotation);
        
        Similarity3 {
            isometry: isometry,
            scale: scale,
        }
    }

    #[inline]
    pub fn from_rotation(rotation: Rotation3<S>) -> Similarity3<S> {
        let isometry = Isometry3::from_rotation(rotation);

        Similarity3 {
            isometry: isometry,
            scale: S::one(),
        }
    }

    #[inline]
    pub fn from_scale(scale: S) -> Similarity3<S> {
        let isometry = Isometry3::identity();

        Similarity3 {
            isometry: isometry,
            scale: scale,
        }
    }

    #[inline]
    pub fn from_translation(translation: Translation3<S>) -> Similarity3<S> {
        let isometry = Isometry3::from_translation(translation);

        Similarity3 {
            isometry: isometry,
            scale: S::one(),
        }
    }

    #[inline]
    pub fn from_isometry(isometry: Isometry3<S>) -> Similarity3<S> {
        Similarity3 {
            isometry: isometry,
            scale: S::one(),
        }
    }

    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix4x4<S> {
        let distance = self.isometry.translation().as_ref();
        let scale = self.scale;
        let rotation = self.isometry.rotation().matrix();

        Matrix4x4::new(
            scale * rotation.c0r0, scale * rotation.c0r1, scale * rotation.c0r2, S::zero(),
            scale * rotation.c1r0, scale * rotation.c1r1, scale * rotation.c1r2, S::zero(),
            scale * rotation.c2r0, scale * rotation.c1r2, scale * rotation.c2r2, S::zero(),
            distance.x,            distance.y,            distance.z,            S::one()
        )
    }
    
    #[inline]
    pub fn scale(&self) -> S {
        self.scale
    }

    #[inline]
    pub fn rotation(&self) -> &Rotation3<S> {
        self.isometry.rotation()
    }

    #[inline]
    pub fn translation(&self) -> &Translation3<S> {
        self.isometry.translation()
    }

    #[inline]
    pub fn identity() -> Similarity3<S> {
        Similarity3 {
            isometry: Isometry3::identity(),
            scale: S::one(),
        }
    }

    /// Convert a similarity transformation to a generic transformation.
    #[inline]
    pub fn to_transform2d(&self) -> Transform3<S> {
        let matrix = self.to_affine_matrix();
        Transform3::from_specialized(matrix)
    }

    #[inline]
    pub fn inverse(&self) -> Similarity3<S> {
        let mut similarity_inv = self.clone();
        similarity_inv.inverse_mut();

        similarity_inv
    }

    #[inline]
    pub fn inverse_mut(&mut self) {
        self.scale = S::one() / self.scale;
        self.isometry.inverse_mut();
        self.isometry.translation.vector *= self.scale;
    }

    #[inline]
    pub fn inverse_transform_point(&self, point: &Point3<S>) -> Point3<S> {
        self.isometry.inverse_transform_point(point) / self.scale
    }
    
    #[inline]
    pub fn inverse_transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.isometry.inverse_transform_vector(vector) / self.scale
    }

    #[inline]
    pub fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        let scaled_point = point * self.scale;
        
        self.isometry.transform_point(&scaled_point)
    }

    #[inline]
    pub fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let scaled_vector = vector * self.scale;
        
        self.isometry.transform_vector(&scaled_vector)
    }

}

impl<S> fmt::Display for Similarity3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Similarity3 [scale={}, rotation={}, translation={}]",
            self.scale, self.isometry.rotation, self.isometry.translation.vector
        )
    }
}

impl<S> approx::AbsDiffEq for Similarity3<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Similarity3<S> where S: ScalarFloat {
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

impl<S> approx::UlpsEq for Similarity3<S> where S: ScalarFloat {
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

impl<S> ops::Mul<Point3<S>> for Similarity3<S> where S: ScalarFloat {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for Similarity3<S> where S: ScalarFloat {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &Similarity3<S> where S: ScalarFloat {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b Similarity3<S> where S: ScalarFloat {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.transform_point(other)
    }
}

