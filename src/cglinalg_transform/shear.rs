use crate::cglinalg_core::{
    SimdScalarSigned,
    SimdScalarFloat,
};
use crate::cglinalg_core::{
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
    Vector2,
    Vector3,
    Point2,
    Point3,
};
use super::transform::{
    Transform2,
    Transform3,
};

use core::fmt;
use core::ops;


/// A shear transformation in two dimensions.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Shear2<S> {
    matrix: Matrix2x2<S>,
}

impl<S> Shear2<S> 
where 
    S: SimdScalarSigned 
{
    /// Construct a shearing transformation along the **x-axis**, holding the 
    /// **y-axis** constant.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the **y-axis** to shearing along the **x-axis**.
    ///
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg::{
    /// #     Shear2,
    /// # };
    /// #
    /// let shear_x_with_y = 3_f64;
    /// let shear = Shear2::from_shear_x(shear_x_with_y);
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = Point2::new(1_f64 + shear_x_with_y * point.y, 2_f64);
    /// let result = shear * point;
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_shear_x(shear_x_with_y: S) -> Self {
        Self {
            matrix: Matrix2x2::from_shear_x(shear_x_with_y),
        }
    }

    /// Construct a shearing transformation along the **y-axis**, holding the 
    /// **x-axis** constant.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the **x-axis** to shearing along the **y-axis**.
    ///
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg::{
    /// #     Shear2,
    /// # };
    /// #
    /// let shear_y_with_x = 3_f64;
    /// let shear = Shear2::from_shear_y(shear_y_with_x);
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = Point2::new(1_f64, 2_f64 + shear_y_with_x * point.x);
    /// let result = shear * point;
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_shear_y(shear_y_with_x: S) -> Self {
        Self {
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
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg::{
    /// #     Shear2,
    /// # };
    /// #
    /// let shear_x_with_y = 5_f64;
    /// let shear_y_with_x = 9_f64;
    /// let shear = Shear2::from_shear(shear_x_with_y, shear_y_with_x);
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = Point2::new(
    ///     1_f64 + shear_x_with_y * point.y, 
    ///     2_f64 + shear_y_with_x * point.x
    /// );
    /// let result = shear * point;
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_shear(shear_x_with_y: S, shear_y_with_x: S) -> Self {
        Self {
            matrix: Matrix2x2::from_shear(shear_x_with_y, shear_y_with_x),
        }
    }

    /// Apply a shearing transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Vector2, 
    /// # };
    /// # use cglinalg::{
    /// #     Shear2,
    /// # };
    /// #
    /// let shear_x_with_y = 5_f64;
    /// let shear_y_with_x = 9_f64;
    /// let shear = Shear2::from_shear(shear_x_with_y, shear_y_with_x);
    /// let vector = Vector2::new(1_f64, 2_f64);
    /// let expected = Vector2::new(
    ///     1_f64 + shear_x_with_y * vector.y, 
    ///     2_f64 + shear_y_with_x * vector.x
    /// );
    /// let result = shear.shear_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn shear_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        self.matrix * vector
    }

    /// Apply a shearing transformation to a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg::{
    /// #     Shear2,
    /// # };
    /// #
    /// let shear_x_with_y = 5_f64;
    /// let shear_y_with_x = 9_f64;
    /// let shear = Shear2::from_shear(shear_x_with_y, shear_y_with_x);
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = Point2::new(
    ///     1_f64 + shear_x_with_y * point.y, 
    ///     2_f64 + shear_y_with_x * point.x
    /// );
    /// let result = shear.shear_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn shear_point(&self, point: &Point2<S>) -> Point2<S> {
        let vector = Vector2::new(point.x, point.y);
        let result = self.matrix * vector;

        Point2::new(result.x, result.y)
    }

    /// Construct the identity shear transformation.
    ///
    /// The identity shear is a shear transformation that does not shear
    /// any coordinates of a vector or a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Vector2,    
    /// # };
    /// # use cglinalg::{
    /// #     Shear2,
    /// # };
    /// #
    /// let shear = Shear2::identity();
    /// let vector = Vector2::new(1_f64, 2_f64);
    /// let expected = vector;
    /// let result = shear.shear_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self { 
            matrix: Matrix2x2::identity(),
        }
    }

    /// Convert a shear transformation into a generic transformation.
    #[inline]
    pub fn to_transform(&self) -> Transform2<S> {
        Transform2::from_specialized(self)
    }
}

impl<S> Shear2<S> 
where 
    S: SimdScalarFloat 
{
    /// Compute the inverse of the shear transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg::{
    /// #     Shear2,
    /// # };
    /// #
    /// let shear_x_with_y = 3_f64;
    /// let shear_y_with_x = 8_f64;
    /// let shear = Shear2::from_shear(shear_x_with_y, shear_y_with_x);
    /// let shear_inv = shear.inverse();
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = point;
    /// let sheared_point = shear.shear_point(&point);
    /// let result = shear_inv.shear_point(&sheared_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn inverse(&self) -> Self {
        let shear_y_with_x = self.matrix.c0r1;
        let shear_x_with_y = self.matrix.c1r0;
        let det_inverse = S::one() / (shear_x_with_y * shear_y_with_x - S::one());
        let matrix = Matrix2x2::new(
            -S::one() * det_inverse,        shear_y_with_x * det_inverse,
             shear_x_with_y * det_inverse, -S::one() * det_inverse
        );
            
        Self { matrix }
    }
    
    /// Apply the inverse of the shear transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Vector2, 
    /// # };
    /// # use cglinalg::{
    /// #     Shear2,
    /// # };
    /// #
    /// let shear_x_with_y = 3_f64;
    /// let shear_y_with_x = 8_f64;
    /// let shear = Shear2::from_shear(shear_x_with_y, shear_y_with_x);
    /// let vector = Vector2::new(1_f64, 2_f64);
    /// let expected = vector;
    /// let sheared_vector = shear.shear_vector(&vector);
    /// let result = shear.inverse_shear_vector(&sheared_vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_shear_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        let inverse = self.inverse();
    
        inverse.matrix * vector
    }
    
    /// Apply the inverse of the shear transformation to a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg::{
    /// #     Shear2,
    /// # };
    /// #
    /// let shear_x_with_y = 3_f64;
    /// let shear_y_with_x = 8_f64;
    /// let shear = Shear2::from_shear(shear_x_with_y, shear_y_with_x);
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = point;
    /// let sheared_point = shear.shear_point(&point);
    /// let result = shear.inverse_shear_point(&sheared_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_shear_point(&self, point: &Point2<S>) -> Point2<S> {
        let inverse = self.inverse();
        let vector = Vector2::new(point.x, point.y);
        let result = inverse.matrix * vector;
    
        Point2::new(result.x, result.y)
    }
}

impl<S> AsRef<Matrix2x2<S>> for Shear2<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix2x2<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Shear2<S> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Shear2 [{}]", self.matrix)
    }
}

impl<S> From<Shear2<S>> for Matrix2x2<S> 
where 
    S: SimdScalarSigned 
{
    #[inline]
    fn from(shear: Shear2<S>) -> Matrix2x2<S> {
        shear.matrix
    }
}

impl<S> From<&Shear2<S>> for Matrix2x2<S> 
where 
    S: SimdScalarSigned 
{
    #[inline]
    fn from(shear: &Shear2<S>) -> Matrix2x2<S> {
        shear.matrix
    }
}

impl<S> From<Shear2<S>> for Matrix3x3<S> 
where 
    S: SimdScalarSigned 
{
    #[inline]
    fn from(shear: Shear2<S>) -> Matrix3x3<S> {
        Matrix3x3::from(&shear.matrix)
    }
}

impl<S> From<&Shear2<S>> for Matrix3x3<S> 
where 
    S: SimdScalarSigned 
{
    #[inline]
    fn from(shear: &Shear2<S>) -> Matrix3x3<S> {
        Matrix3x3::from(&shear.matrix)
    }
}

impl<S> approx::AbsDiffEq for Shear2<S> 
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
        Matrix2x2::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S> approx::RelativeEq for Shear2<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix2x2::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Shear2<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix2x2::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S> ops::Mul<Point2<S>> for Shear2<S> 
where 
    S: SimdScalarFloat 
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.shear_point(&other)
    }
}

impl<S> ops::Mul<&Point2<S>> for Shear2<S> 
where 
    S: SimdScalarFloat 
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: &Point2<S>) -> Self::Output {
        self.shear_point(other)
    }
}

impl<S> ops::Mul<Point2<S>> for &Shear2<S> 
where 
    S: SimdScalarFloat 
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.shear_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point2<S>> for &'b Shear2<S> 
where 
    S: SimdScalarFloat 
{
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
    matrix: Matrix3x3<S>,
}

impl<S> Shear3<S> 
where 
    S: SimdScalarSigned 
{
    /// Construct a shearing transformation along the **x-axis**.
    ///
    /// The parameters `shear_x_with_y` and `shear_x_with_z` denote the 
    /// multiplicative factors for the contributions from the **y-axis** and the 
    /// **z-axis** respectively for the shearing along the **x-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Point3, 
    /// # }; 
    /// # use cglinalg::{
    /// #     Shear3,
    /// # };
    /// #
    /// let shear_x_with_y = 3_f64;
    /// let shear_x_with_z = 7_f64;
    /// let shear = Shear3::from_shear_x(shear_x_with_y, shear_x_with_z);
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Point3::new(
    ///     1_f64 + shear_x_with_y * point.y + shear_x_with_z * point.z,
    ///     2_f64,
    ///     3_f64 
    /// );
    /// let result = shear * point;
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_shear_x(shear_x_with_y: S, shear_x_with_z: S) -> Self {
        Self {
            matrix: Matrix3x3::from_shear_x(shear_x_with_y, shear_x_with_z),
        }
    }

    /// Construct a shearing transformation along the **y-axis**.
    ///
    /// The parameters `shear_y_with_x` and `shear_y_with_z` denote the 
    /// multiplicative factors for the contributions from the **x-axis** and the 
    /// **z-axis** respectively for the shearing along the **y-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Point3, 
    /// # }; 
    /// # use cglinalg::{
    /// #     Shear3,
    /// # };
    /// #
    /// let shear_y_with_x = 3_f64;
    /// let shear_y_with_z = 7_f64;
    /// let shear = Shear3::from_shear_y(shear_y_with_x, shear_y_with_z);
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Point3::new(
    ///     1_f64,
    ///     2_f64 + shear_y_with_x * point.x + shear_y_with_z * point.z,
    ///     3_f64 
    /// );
    /// let result = shear * point;
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_shear_y(shear_y_with_x: S, shear_y_with_z: S) -> Self {
        Self {
            matrix: Matrix3x3::from_shear_y(shear_y_with_x, shear_y_with_z),
        }
    }

    /// Construct a shearing transformation along the **z-axis**.
    ///
    /// The parameters `shear_z_with_x` and `shear_z_with_y` denote the 
    /// multiplicative factors for the contributions from the **x-axis** and the 
    /// **y-axis** respectively for the shearing along the **z-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Point3, 
    /// # }; 
    /// # use cglinalg::{
    /// #     Shear3,
    /// # };
    /// #
    /// let shear_z_with_x = 3_f64;
    /// let shear_z_with_y = 7_f64;
    /// let shear = Shear3::from_shear_z(shear_z_with_x, shear_z_with_y);
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Point3::new(
    ///     1_f64,
    ///     2_f64,
    ///     3_f64 + shear_z_with_x * point.x + shear_z_with_y * point.y
    /// );
    /// let result = shear * point;
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_shear_z(shear_z_with_x: S, shear_z_with_y: S) -> Self {
        Self {
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
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Point3, 
    /// # }; 
    /// # use cglinalg::{
    /// #     Shear3,
    /// # };
    /// #
    /// let shear_x_with_y = 1_f64;
    /// let shear_x_with_z = 2_f64;
    /// let shear_y_with_x = 10_f64;
    /// let shear_y_with_z = 20_f64;
    /// let shear_z_with_x = 100_f64;
    /// let shear_z_with_y = 200_f64;
    /// let shear = Shear3::from_shear(
    ///     shear_x_with_y,
    ///     shear_x_with_z,
    ///     shear_y_with_x,
    ///     shear_y_with_z,
    ///     shear_z_with_x, 
    ///     shear_z_with_y
    /// );
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Point3::new(
    ///     1_f64 + shear_x_with_y * point.y + shear_x_with_z * point.z,
    ///     2_f64 + shear_y_with_x * point.x + shear_y_with_z * point.z,
    ///     3_f64 + shear_z_with_x * point.x + shear_z_with_y * point.y
    /// );
    /// let result = shear * point;
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear(
        shear_x_with_y: S, shear_x_with_z: S, 
        shear_y_with_x: S, shear_y_with_z: S, 
        shear_z_with_x: S, shear_z_with_y: S) -> Self
    {
        Self {
            matrix: Matrix3x3::from_shear(
                shear_x_with_y, shear_x_with_z, 
                shear_y_with_x, shear_y_with_z, 
                shear_z_with_x, shear_z_with_y
            )
        }
    }

    /// Apply a shear transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Vector3, 
    /// # }; 
    /// # use cglinalg::{
    /// #     Shear3,
    /// # };
    /// #
    /// let shear_x_with_y = 1_f64;
    /// let shear_x_with_z = 2_f64;
    /// let shear_y_with_x = 10_f64;
    /// let shear_y_with_z = 20_f64;
    /// let shear_z_with_x = 100_f64;
    /// let shear_z_with_y = 200_f64;
    /// let shear = Shear3::from_shear(
    ///     shear_x_with_y,
    ///     shear_x_with_z,
    ///     shear_y_with_x,
    ///     shear_y_with_z,
    ///     shear_z_with_x, 
    ///     shear_z_with_y
    /// );
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Vector3::new(
    ///     1_f64 + shear_x_with_y * vector.y + shear_x_with_z * vector.z,
    ///     2_f64 + shear_y_with_x * vector.x + shear_y_with_z * vector.z,
    ///     3_f64 + shear_z_with_x * vector.x + shear_z_with_y * vector.y
    /// );
    /// let result = shear.shear_vector(&vector);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn shear_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        self.matrix * vector
    }

    /// Apply a shear transformation to a point.
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Point3, 
    /// # }; 
    /// # use cglinalg::{
    /// #     Shear3,
    /// # };
    /// #
    /// let shear_x_with_y = 1_f64;
    /// let shear_x_with_z = 2_f64;
    /// let shear_y_with_x = 10_f64;
    /// let shear_y_with_z = 20_f64;
    /// let shear_z_with_x = 100_f64;
    /// let shear_z_with_y = 200_f64;
    /// let shear = Shear3::from_shear(
    ///     shear_x_with_y,
    ///     shear_x_with_z,
    ///     shear_y_with_x,
    ///     shear_y_with_z,
    ///     shear_z_with_x, 
    ///     shear_z_with_y
    /// );
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Point3::new(
    ///     1_f64 + shear_x_with_y * point.y + shear_x_with_z * point.z,
    ///     2_f64 + shear_y_with_x * point.x + shear_y_with_z * point.z,
    ///     3_f64 + shear_z_with_x * point.x + shear_z_with_y * point.y
    /// );
    /// let result = shear.shear_point(&point);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn shear_point(&self, point: &Point3<S>) -> Point3<S> {
        let vector = Vector3::new(point.x, point.y, point.z);
        let result = self.matrix * vector;

        Point3::new(result.x, result.y, result.z)
    }

    /// Construct the identity shear transformation.
    ///
    /// The identity shear is a shear transformation that does not shear
    /// any coordinates of a vector or a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Vector3,    
    /// # };
    /// # use cglinalg::{
    /// #     Shear3,
    /// # };
    /// #
    /// let shear = Shear3::identity();
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = vector;
    /// let result = shear.shear_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self { 
            matrix: Matrix3x3::identity(),
        }
    }

    /// Convert a shear transformation into a generic transformation.
    #[inline]
    pub fn to_transform(&self) -> Transform3<S> {
        Transform3::from_specialized(self.matrix)
    }
}

impl<S> Shear3<S> 
where 
    S: SimdScalarFloat 
{
    /// Calculate the inverse of a shear transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Point3,
    /// # };
    /// # use cglinalg::{
    /// #     Shear3,
    /// # };
    /// #
    /// let shear_x_with_y = 5_f64;
    /// let shear_x_with_z = 10_f64;
    /// let shear = Shear3::from_shear_x(shear_x_with_y, shear_x_with_z);
    /// let shear_inv = shear.inverse();
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = point;
    /// let sheared_point = shear.shear_point(&point);
    /// let result = shear_inv.shear_point(&sheared_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn inverse(&self) -> Self {
        let shear_x_with_y = self.matrix.c1r0;
        let shear_x_with_z = self.matrix.c2r0;
        let shear_y_with_x = self.matrix.c0r1;
        let shear_y_with_z = self.matrix.c2r1;
        let shear_z_with_x = self.matrix.c0r2;
        let shear_z_with_y = self.matrix.c1r2;
        let det_inverse = S::one() / self.matrix.determinant();
        let c0r0 = det_inverse * (S::one() - shear_y_with_z * shear_z_with_y);
        let c0r1 = det_inverse * (shear_y_with_z * shear_z_with_x - shear_y_with_x);
        let c0r2 = det_inverse * (shear_y_with_x * shear_z_with_y - shear_z_with_x);
        let c1r0 = det_inverse * (shear_x_with_z * shear_z_with_y - shear_x_with_y);
        let c1r1 = det_inverse * (S::one() - shear_x_with_z * shear_z_with_x);
        let c1r2 = det_inverse * (shear_x_with_y * shear_z_with_x - shear_z_with_y);
        let c2r0 = det_inverse * (shear_x_with_y * shear_y_with_z - shear_x_with_z);
        let c2r1 = det_inverse * (shear_x_with_z * shear_y_with_x - shear_y_with_z);
        let c2r2 = det_inverse * (S::one() - shear_x_with_y * shear_y_with_x);
        let matrix = Matrix3x3::new(
            c0r0, c0r1, c0r2, 
            c1r0, c1r1, c1r2,
            c2r0, c2r1, c2r2
        );
        
        Self { matrix }
    }

    /// Apply the inverse of the shear transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// # use cglinalg::{
    /// #     Shear3, 
    /// # };
    /// #
    /// let shear_x_with_y = 5_f64;
    /// let shear_x_with_z = 10_f64;
    /// let shear = Shear3::from_shear_x(shear_x_with_y, shear_x_with_z);
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = vector;
    /// let sheared_vector = shear.shear_vector(&vector);
    /// let result = shear.inverse_shear_vector(&sheared_vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_shear_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let inverse = self.inverse();

        inverse.matrix * vector
    }

    /// Apply the inverse of the shear transformation to a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::cglinalg_core::{
    /// #     Point3,
    /// # };
    /// # use cglinalg::{
    /// #     Shear3,
    /// # };
    /// #
    /// let shear_x_with_y = 5_f64;
    /// let shear_x_with_z = 10_f64;
    /// let shear = Shear3::from_shear_x(shear_x_with_y, shear_x_with_z);
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = point;
    /// let sheared_point = shear.shear_point(&point);
    /// let result = shear.inverse_shear_point(&sheared_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_shear_point(&self, point: &Point3<S>) -> Point3<S> {
        let inverse = self.inverse();
        let vector = Vector3::new(point.x, point.y, point.z);
        let result = inverse.matrix * vector;

        Point3::new(result.x, result.y, result.z)
    }
}

impl<S> AsRef<Matrix3x3<S>> for Shear3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Shear3<S> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Shear3 [{}]", self.matrix)
    }
}

impl<S> From<Shear3<S>> for Matrix3x3<S> 
where 
    S: SimdScalarSigned 
{
    fn from(shear: Shear3<S>) -> Matrix3x3<S> {
        shear.matrix
    }
}

impl<S> From<&Shear3<S>> for Matrix3x3<S> 
where 
    S: SimdScalarSigned 
{
    #[inline]
    fn from(shear: &Shear3<S>) -> Matrix3x3<S> {
        shear.matrix
    }
}

impl<S> From<Shear3<S>> for Matrix4x4<S> 
where 
    S: SimdScalarSigned 
{
    #[inline]
    fn from(shear: Shear3<S>) -> Matrix4x4<S> {
        Matrix4x4::from(&shear.matrix)
    }
}

impl<S> From<&Shear3<S>> for Matrix4x4<S> 
where 
    S: SimdScalarSigned 
{
    #[inline]
    fn from(shear: &Shear3<S>) -> Matrix4x4<S> {
        Matrix4x4::from(&shear.matrix)
    }
}

impl<S> approx::AbsDiffEq for Shear3<S> 
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
        Matrix3x3::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S> approx::RelativeEq for Shear3<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix3x3::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Shear3<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix3x3::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S> ops::Mul<Point3<S>> for Shear3<S> 
where 
    S: SimdScalarFloat 
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.shear_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for Shear3<S> 
where 
    S: SimdScalarFloat 
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.shear_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &Shear3<S> 
where 
    S: SimdScalarFloat 
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.shear_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b Shear3<S> 
where 
    S: SimdScalarFloat 
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.shear_point(other)
    }
}

