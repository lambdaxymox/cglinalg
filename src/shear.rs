use crate::scalar::{
    ScalarSigned,
    ScalarFloat,
};
use crate::matrix::{
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
};
use crate::vector::{
    Vector2,
    Vector3,
};
use crate::point::{
    Point2,
    Point3,
};
use crate::traits::{
    Identity,
};
use crate::transform::*;

use core::fmt;
use core::ops;


/// A shearing transformation in two dimensions.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Shear2<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix2x2<S>,
}

impl<S> Shear2<S> where S: ScalarSigned {
    /// Construct a shearing transformation along the **x-axis**, holding the 
    /// **y-axis** constant.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the **y-axis** to shearing along the **x-axis**.
    #[inline]
    pub fn from_shear_x(shear_x_with_y: S) -> Shear2<S> {
        Shear2 {
            matrix: Matrix2x2::from_shear_x(shear_x_with_y),
        }
    }

    /// Construct a shearing transformation along the **y-axis**, holding the 
    /// **x-axis** constant.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the **x-axis** to shearing along the **y-axis**.
    #[inline]
    pub fn from_shear_y(shear_y_with_x: S) -> Shear2<S> {
        Shear2 {
            matrix: Matrix2x2::from_shear_y(shear_y_with_x),
        }
    }

    /// Construct a general shearing transformations in two dimensions. 
    ///
    /// There are two parameters describing a shearing transformation 
    /// in two dimensions.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the **x-axis** to shearing along the **y-axis**.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the 
    /// contribution of the **y-axis** to the shearing along the **x-axis**.
    #[inline]
    pub fn from_shear(shear_x_with_y: S, shear_y_with_x: S) -> Shear2<S> {
        Shear2 {
            matrix: Matrix2x2::from_shear(shear_x_with_y, shear_y_with_x),
        }
    }

    /// Compute the inverse shearing operation.
    #[inline]
    pub fn inverse(&self) -> Shear2<S> {
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_x_with_y = -self.matrix.c1r0;
        let matrix = Matrix2x2::from_shear(shear_x_with_y, shear_y_with_x);
        
        Shear2 {
            matrix: matrix,
        }
    }

    /// Apply a shearing transformation to a vector.
    #[inline]
    pub fn shear_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.matrix * vector
    }

    /// Apply a shearing transformation to a point.
    #[inline]
    pub fn shear_point(&self, point: &Point2<S>) -> Point2<S> {
        let vector = Vector2::new(point.x, point.y);
        let result = self.matrix * vector;

        Point2::new(result.x, result.y)
    }

    #[inline]
    pub fn inverse_shear_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        let inverse = self.inverse();

        inverse.matrix * vector
    }

    #[inline]
    pub fn inverse_shear_point(&self, point: &Point2<S>) -> Point2<S> {
        let inverse = self.inverse();
        let vector = Vector2::new(point.x, point.y);
        let result = inverse.matrix * vector;

        Point2::new(result.x, result.y)
    }

    #[inline]
    pub fn identity() -> Shear2<S> {
        Shear2 { 
            matrix: Matrix2x2::identity(),
        }
    }

    #[inline]
    pub fn to_transform2d(&self) -> Transform2<S> {
        Transform2::from_specialized(self)
    }
}

impl<S> AsRef<Matrix2x2<S>> for Shear2<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix2x2<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Shear2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Shear2 [{}]", self.matrix)
    }
}

impl<S> From<Shear2<S>> for Matrix2x2<S> where S: ScalarSigned {
    #[inline]
    fn from(shear: Shear2<S>) -> Matrix2x2<S> {
        shear.matrix
    }
}

impl<S> From<&Shear2<S>> for Matrix2x2<S> where S: ScalarSigned {
    #[inline]
    fn from(shear: &Shear2<S>) -> Matrix2x2<S> {
        shear.matrix
    }
}

impl<S> From<Shear2<S>> for Matrix3x3<S> where S: ScalarSigned {
    #[inline]
    fn from(shear: Shear2<S>) -> Matrix3x3<S> {
        Matrix3x3::from(&shear.matrix)
    }
}

impl<S> From<&Shear2<S>> for Matrix3x3<S> where S: ScalarSigned {
    #[inline]
    fn from(shear: &Shear2<S>) -> Matrix3x3<S> {
        Matrix3x3::from(&shear.matrix)
    }
}

impl<S> approx::AbsDiffEq for Shear2<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Shear2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix2x2::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Shear2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix2x2::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S> ops::Mul<Point2<S>> for Shear2<S> where S: ScalarFloat {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.shear_point(&other)
    }
}

impl<S> ops::Mul<&Point2<S>> for Shear2<S> where S: ScalarFloat {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: &Point2<S>) -> Self::Output {
        self.shear_point(other)
    }
}

impl<S> ops::Mul<Point2<S>> for &Shear2<S> where S: ScalarFloat {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.shear_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point2<S>> for &'b Shear2<S> where S: ScalarFloat {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: &'a Point2<S>) -> Self::Output {
        self.shear_point(other)
    }
}


/// A shearing transformation in three dimensions.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Shear3<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix3x3<S>,
}

impl<S> Shear3<S> where S: ScalarSigned {
    /// Construct a shearing transformation along the **x-axis**.
    ///
    /// The parameters `shear_x_with_y` and `shear_x_with_z` denote the 
    /// multiplicative factors for the contributions from the **y-axis** and the 
    /// **z-axis** respectively for the shearing along the **x-axis**.
    #[inline]
    pub fn from_shear_x(shear_x_with_y: S, shear_x_with_z: S) -> Shear3<S> {
        Shear3 {
            matrix: Matrix3x3::from_shear_x(shear_x_with_y, shear_x_with_z),
        }
    }

    /// Construct a shearing transformation along the **y-axis**.
    ///
    /// The parameters `shear_y_with_x` and `shear_y_with_z` denote the 
    /// multiplicative factors for the contributions from the **x-axis** and the 
    /// **z-axis** respectively for the shearing along the **y-axis**.
    #[inline]
    pub fn from_shear_y(shear_y_with_x: S, shear_y_with_z: S) -> Shear3<S> {
        Shear3 {
            matrix: Matrix3x3::from_shear_y(shear_y_with_x, shear_y_with_z),
        }
    }

    /// Construct a shearing transformation along the **z-axis**.
    ///
    /// The parameters `shear_z_with_x` and `shear_z_with_y` denote the 
    /// multiplicative factors for the contributions from the **x-axis** and the 
    /// **y-axis** respectively for the shearing along the **z-axis**.
    #[inline]
    pub fn from_shear_z(shear_z_with_x: S, shear_z_with_y: S) -> Shear3<S> {
        Shear3 {
            matrix: Matrix3x3::from_shear_z(shear_z_with_x, shear_z_with_y),
        }
    }

    /// Construct a general shearing transformation.
    ///
    /// The parameters `shear_x_with_y` and `shear_x_with_z` denote the 
    /// multiplicative factors for the contributions from the **y-axis** and the 
    /// **z-axis** respectively for the shearing along the **x-axis**.
    /// 
    /// The parameters `shear_y_with_x` and `shear_y_with_z` denote the 
    /// multiplicative factors for the contributions from the **x-axis** and the 
    /// **z-axis** respectively for the shearing along the **y-axis**.
    ///
    /// The parameters `shear_z_with_x` and `shear_z_with_y` denote the 
    /// multiplicative factors for the contributions from the **x-axis** and the 
    /// **y-axis** respectively for the shearing along the **z-axis**.
    #[inline]
    pub fn from_shear(
        shear_x_with_y: S, shear_x_with_z: S, 
        shear_y_with_x: S, shear_y_with_z: S, 
        shear_z_with_x: S, shear_z_with_y: S) -> Shear3<S>
    {
        Shear3 {
            matrix: Matrix3x3::from_shear(
                shear_x_with_y, shear_x_with_z, 
                shear_y_with_x, shear_y_with_z, 
                shear_z_with_x, shear_z_with_y
            )
        }
    }

    /// Apply a shearing transformation to a vector.
    #[inline]
    pub fn inverse(&self) -> Shear3<S> {
        let shear_x_with_y = -self.matrix.c1r0;
        let shear_x_with_z = -self.matrix.c2r0;
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_y_with_z = -self.matrix.c2r1;
        let shear_z_with_x = -self.matrix.c0r2;
        let shear_z_with_y = -self.matrix.c1r2;
        let matrix = Matrix3x3::from_shear(
            shear_x_with_y, shear_x_with_z, 
            shear_y_with_x, shear_y_with_z, 
            shear_z_with_x, shear_z_with_y
        );
        
        Shear3 {
            matrix: matrix,
        }
    }

    /// Apply a shearing transformation to a vector.
    #[inline]
    pub fn shear_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.matrix * vector
    }

    /// Apply a shearing transformation to a point.
    #[inline]
    pub fn shear_point(&self, point: &Point3<S>) -> Point3<S> {
        let vector = Vector3::new(point.x, point.y, point.z);
        let result = self.matrix * vector;

        Point3::new(result.x, result.y, result.z)
    }

    #[inline]
    pub fn inverse_shear_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let inverse = self.inverse();

        inverse.matrix * vector
    }

    #[inline]
    pub fn inverse_shear_point(&self, point: &Point3<S>) -> Point3<S> {
        let inverse = self.inverse();
        let vector = Vector3::new(point.x, point.y, point.z);
        let result = inverse.matrix * vector;

        Point3::new(result.x, result.y, result.z)
    }

    #[inline]
    pub fn identity() -> Shear3<S> {
        Shear3 { 
            matrix: Matrix3x3::identity(),
        }
    }

    #[inline]
    pub fn to_transform3d(&self) -> Transform3<S> {
        Transform3::from_specialized(self.matrix)
    }
}

impl<S> AsRef<Matrix3x3<S>> for Shear3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Shear3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Shear3 [{}]", self.matrix)
    }
}

impl<S> From<Shear3<S>> for Matrix3x3<S> where S: ScalarSigned {
    fn from(shear: Shear3<S>) -> Matrix3x3<S> {
        shear.matrix
    }
}

impl<S> From<&Shear3<S>> for Matrix3x3<S> where S: ScalarSigned {
    fn from(shear: &Shear3<S>) -> Matrix3x3<S> {
        shear.matrix
    }
}

impl<S> From<Shear3<S>> for Matrix4x4<S> where S: ScalarSigned {
    fn from(shear: Shear3<S>) -> Matrix4x4<S> {
        Matrix4x4::from(&shear.matrix)
    }
}

impl<S> From<&Shear3<S>> for Matrix4x4<S> where S: ScalarSigned {
    fn from(shear: &Shear3<S>) -> Matrix4x4<S> {
        Matrix4x4::from(&shear.matrix)
    }
}

impl<S> approx::AbsDiffEq for Shear3<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Shear3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix3x3::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Shear3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix3x3::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S> ops::Mul<Point3<S>> for Shear3<S> where S: ScalarFloat {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.shear_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for Shear3<S> where S: ScalarFloat {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.shear_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &Shear3<S> where S: ScalarFloat {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.shear_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b Shear3<S> where S: ScalarFloat {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.shear_point(other)
    }
}

