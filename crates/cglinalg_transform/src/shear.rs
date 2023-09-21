use cglinalg_numeric::{
    SimdScalarSigned,
    SimdScalarFloat,
};
use cglinalg_core::{
    Const,
    ShapeConstraint,
    DimAdd,
    DimLt,
    Matrix,
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
    Vector,
    Vector2,
    Vector3,
    Point,
    Point2,
    Point3,
};
use crate::transform::{
    Transform,
};

use core::fmt;
use core::ops;


/// A shearing matrix in two dimensions.
pub type Shear2<S> = Shear<S, 2>;

/// A shearing matrix in three dimensions.
pub type Shear3<S> = Shear<S, 3>;


/// A shearing matrix.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Shear<S, const N: usize> {
    matrix: Matrix<S, N, N>,
}

impl<S, const N: usize> Shear<S, N> 
where 
    S: SimdScalarSigned 
{
    /// Apply a shear transformation to a vector.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2, 
    /// # };
    /// # use cglinalg_transform::{
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
    /// 
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3, 
    /// # }; 
    /// # use cglinalg_transform::{
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
    pub fn shear_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        self.matrix * vector
    }

    /// Apply a shear transformation to a point.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg_transform::{
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
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3, 
    /// # }; 
    /// # use cglinalg_transform::{
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
    pub fn shear_point(&self, point: &Point<S, N>) -> Point<S, N> {
        let vector = point.to_vector();
        let result = self.matrix * vector;

        Point::from_vector(&result)
    }

    /// Construct the identity shear transformation.
    ///
    /// The identity shear is a shear transformation that does not shear
    /// any coordinates of a vector or a point.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,    
    /// # };
    /// # use cglinalg_transform::{
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
    /// 
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,    
    /// # };
    /// # use cglinalg_transform::{
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
            matrix: Matrix::identity(),
        }
    }
}

impl<S, const N: usize, const NPLUS1: usize> Shear<S, N> 
where 
    S: SimdScalarSigned,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimLt<Const<N>, Const<NPLUS1>>
{
    /// Convert a shear transformation into a generic transformation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear2,
    /// #     Transform2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// # };
    /// #
    /// let shear = Shear2::from_shear(-5_f64, 7_f64);
    /// let expected = Transform2::from_matrix_unchecked(Matrix3x3::new(
    ///      1_f64, 7_f64, 0_f64,
    ///     -5_f64, 1_f64, 0_f64,
    ///      0_f64, 0_f64, 1_f64
    /// ));
    /// let result = shear.to_transform();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear3,
    /// #     Transform3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// # };
    /// #
    /// let shear = Shear3::from_shear(
    ///     5_f64,  7_f64,
    ///     11_f64, 13_f64,
    ///     17_f64, 19_f64
    /// );
    /// let expected = Transform3::from_matrix_unchecked(Matrix4x4::new(
    ///     1_f64, 11_f64, 17_f64, 0_f64,
    ///     5_f64, 1_f64,  19_f64, 0_f64,
    ///     7_f64, 13_f64, 1_f64,  0_f64,
    ///     0_f64, 0_f64,  0_f64,  1_f64
    /// ));
    /// let result = shear.to_transform();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_transform(&self) -> Transform<S, N, NPLUS1> {
        Transform::from_specialized(self.matrix)
    }
}

impl<S, const N: usize> AsRef<Matrix<S, N, N>> for Shear<S, N> {
    #[inline]
    fn as_ref(&self) -> &Matrix<S, N, N> {
        &self.matrix
    }
}

impl<S, const N: usize> fmt::Display for Shear<S, N> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Shear{} [{}]", N, self.matrix)
    }
}

impl<S, const N: usize> From<Shear<S, N>> for Matrix<S, N, N> 
where 
    S: SimdScalarSigned 
{
    fn from(shear: Shear<S, N>) -> Matrix<S, N, N> {
        shear.matrix
    }
}

impl<S, const N: usize> From<&Shear<S, N>> for Matrix<S, N, N> 
where 
    S: SimdScalarSigned 
{
    #[inline]
    fn from(shear: &Shear<S, N>) -> Matrix<S, N, N> {
        shear.matrix
    }
}
/*
impl<S, const N: usize, const NPLUS1: usize> From<Shear<S, N>> for Matrix<S, NPLUS1, NPLUS1>
where 
    S: SimdScalarSigned,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    #[inline]
    fn from(shear: Shear<S, N>) -> Matrix<S, NPLUS1, NPLUS1> {
        Matrix::from(&shear.matrix)
    }
}
*/
/*
impl<S, const N: usize, const NPLUS1: usize> From<&Shear<S, N>> for Matrix<S, NPLUS1, NPLUS1> 
where 
    S: SimdScalarSigned,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    #[inline]
    fn from(shear: &Shear3<S>) -> Matrix4x4<S> {
        Matrix4x4::from(&shear.matrix)
    }
}
*/

impl<S, const N: usize> approx::AbsDiffEq for Shear<S, N> 
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
        Matrix::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S, const N: usize> approx::RelativeEq for Shear<S, N> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S, const N: usize> approx::UlpsEq for Shear<S, N> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for Shear<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.shear_point(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Point<S, N>> for Shear<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &Point<S, N>) -> Self::Output {
        self.shear_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for &Shear<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.shear_point(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Point<S, N>> for &'b Shear<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &'a Point<S, N>) -> Self::Output {
        self.shear_point(other)
    }
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
    /// # use cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg_transform::{
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
    /// # use cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg_transform::{
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
    /// # use cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg_transform::{
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
    /// # use cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg_transform::{
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
    /// # use cglinalg_core::{
    /// #     Vector2, 
    /// # };
    /// # use cglinalg_transform::{
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
    /// # use cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg_transform::{
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
    /// # use cglinalg_core::{
    /// #     Point3, 
    /// # }; 
    /// # use cglinalg_transform::{
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
    /// # use cglinalg_core::{
    /// #     Point3, 
    /// # }; 
    /// # use cglinalg_transform::{
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
    /// # use cglinalg_core::{
    /// #     Point3, 
    /// # }; 
    /// # use cglinalg_transform::{
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
    /// # use cglinalg_core::{
    /// #     Point3, 
    /// # }; 
    /// # use cglinalg_transform::{
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
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// # use cglinalg_transform::{
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
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
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
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// # use cglinalg_transform::{
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

