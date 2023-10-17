use cglinalg_numeric::{
    SimdScalar,
    SimdScalarSigned,
    SimdScalarFloat,
};
use cglinalg_core::{
    Matrix,
    Matrix3x3,
    Matrix4x4,
    Vector,
    Vector2,
    Vector3,
    Point,
    Point2,
    Point3,
    Unit,
};
use crate::transform::{
    Transform2,
    Transform3,
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
    shear_factor: S,
    origin: Point<S, N>,
    direction: Vector<S, N>,
    normal: Vector<S, N>,
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
    /// let result = shear.apply_vector(&vector);
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
    /// let result = shear.apply_vector(&vector);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn apply_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        let origin = self.origin.to_vector();
        let factor = self.shear_factor * (vector - origin).dot(&self.normal);

        vector + self.direction * factor
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
    /// let result = shear.apply_point(&point);
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
    /// let result = shear.apply_point(&point);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn apply_point(&self, point: &Point<S, N>) -> Point<S, N> {
        let factor = self.shear_factor * (point - self.origin).dot(&self.normal);

        point + self.direction * factor
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
    /// let result = shear.apply_vector(&vector);
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
    /// let result = shear.apply_vector(&vector);
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
    S: SimdScalar
{
    /// Convert a shear transformation to a matrix.
    /// 
    /// The resulting matrix is not an affine. For an affine matrix,
    /// use [`Shear::to_affine_matrix`].
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// # };
    /// #
    /// let shear = Shear2::from_shear(5_f64, 7_f64);
    /// let expected = Matrix2x2::new(
    ///     1_f64, 7_f64,
    ///     5_f64, 1_f64
    /// );
    /// let result = shear.to_matrix();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// # };
    /// #
    /// let shear = Shear3::from_shear(
    ///     2_f64, 3_f64,
    ///     4_f64, 5_f64,
    ///     6_f64, 7_f64
    /// );
    /// let expected = Matrix3x3::new(
    ///     1_f64, 4_f64, 6_f64,
    ///     2_f64, 1_f64, 7_f64,
    ///     3_f64, 5_f64, 1_f64
    /// );
    /// let result = shear.to_matrix();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_matrix(&self) -> Matrix<S, N, N> {
        todo!()
    }
}

impl<S, const N: usize> fmt::Display for Shear<S, N> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Shear{} [shear_factor={}, origin={}, direction={}, normal={}]", 
            N,
            self.shear_factor, self.origin, self.direction, self.normal
        )
    }
}

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
        S::abs_diff_eq(&self.shear_factor, &other.shear_factor, epsilon)
            && Point::abs_diff_eq(&self.origin, &other.origin, epsilon)
            && Vector::abs_diff_eq(&self.direction, &other.direction, epsilon)
            && Vector::abs_diff_eq(&self.normal, &other.normal, epsilon)
    }
}

impl<S, const N: usize> approx::RelativeEq for Shear<S, N> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        S::relative_eq(&self.shear_factor, &other.shear_factor, epsilon, max_relative)
            && Point::relative_eq(&self.origin, &other.origin, epsilon, max_relative)
            && Vector::relative_eq(&self.direction, &other.direction, epsilon, max_relative)
            && Vector::relative_eq(&self.normal, &other.normal, epsilon, max_relative)
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
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.shear_factor, &other.shear_factor, epsilon, max_ulps)
            && Point::ulps_eq(&self.origin, &other.origin, epsilon, max_ulps)
            && Vector::ulps_eq(&self.direction, &other.direction, epsilon, max_ulps)
            && Vector::ulps_eq(&self.normal, &other.normal, epsilon, max_ulps)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for Shear<S, N> 
where 
    S: SimdScalarSigned
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Point<S, N>> for Shear<S, N> 
where 
    S: SimdScalarSigned
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for &Shear<S, N> 
where 
    S: SimdScalarSigned
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Point<S, N>> for &'b Shear<S, N> 
where 
    S: SimdScalarSigned
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &'a Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Vector<S, N>> for Shear<S, N> 
where 
    S: SimdScalarSigned
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Vector<S, N>> for Shear<S, N> 
where 
    S: SimdScalarSigned
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: &Vector<S, N>) -> Self::Output {
        self.apply_vector(other)
    }
}

impl<S, const N: usize> ops::Mul<Vector<S, N>> for &Shear<S, N> 
where 
    S: SimdScalarSigned
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Vector<S, N>> for &'b Shear<S, N> 
where 
    S: SimdScalarSigned
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: &'a Vector<S, N>) -> Self::Output {
        self.apply_vector(other)
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
    /// # use cglinalg_transform::{
    /// #     Shear2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2,
    /// # };
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

    /// Construct a shearing transformation along the **y-axis**, holding the 
    /// **x-axis** constant.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the **x-axis** to shearing along the **y-axis**.
    ///
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2,
    /// # };
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
    S: SimdScalarFloat
{
    /// Construct a general shearing matrix in two dimensions with respect to 
    /// a line passing through the origin `[0, 0]`.
    /// 
    /// This version of the shearing transformation is a linear transformation because 
    /// the origin of the coordinate frame for applying the shearing transformation 
    /// is `[0, 0]` so there is no translation term to account for.
    /// 
    /// # Parameters
    /// 
    /// The shearing matrix constructor has the following parameters
    /// * `shear_factor`: The amount by which a point in a line parallel to the shearing 
    ///    line gets sheared.
    /// * `direction`: The direction along which the shearing happens.
    /// * `normal`: The normal vector to the shearing line.
    ///
    /// # Example 
    /// 
    /// Shearing a rotated square parallel to the line `y == (1 / 2) * x` along the 
    /// line `y == (1 / 2) * x`.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
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
    /// assert_relative_eq!(result[0], expected[0], epsilon = 1e-10);
    /// assert_relative_eq!(result[1], expected[1], epsilon = 1e-10);
    /// assert_relative_eq!(result[2], expected[2], epsilon = 1e-10);
    /// assert_relative_eq!(result[3], expected[3], epsilon = 1e-10);
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
    /// assert_relative_eq!(result_in_line[0], expected_in_line[0], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_line[1], expected_in_line[1], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_line[2], expected_in_line[2], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_line[3], expected_in_line[3], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_line[4], expected_in_line[4], epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn from_shear(shear_factor: S, direction: &Unit<Vector2<S>>, normal: &Unit<Vector2<S>>) -> Self {
        Self {
            shear_factor,
            origin: Point2::origin(),
            direction: direction.into_inner(),
            normal: normal.into_inner(),
        }
    }

    /// Construct a general affine shearing matrix in two dimensions with respect to 
    /// a line passing through the origin `origin`, not necessarily `[0, 0]`.
    /// 
    /// # Parameters
    /// 
    /// The affine shearing matrix constructor has four parameters
    /// * `origin`: The origin of the affine frame for the shearing transformation.
    /// * `shear_factor`: The amount by which a point in a plane parallel to the shearing 
    ///    line gets sheared.
    /// * `direction`: The direction along which the shearing happens in the shearing line.
    /// * `normal`: The normal vector to the shearing line.
    /// 
    /// # Examples
    /// 
    /// Shearing along the **x-axis** with a non-zero origin on the **x-axis**.
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
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
    /// assert_relative_eq!(result[0], expected[0], epsilon = 1e-10);
    /// assert_relative_eq!(result[1], expected[1], epsilon = 1e-10);
    /// assert_relative_eq!(result[2], expected[2], epsilon = 1e-10);
    /// assert_relative_eq!(result[3], expected[3], epsilon = 1e-10);
    /// 
    /// let vertices_in_line = [
    ///     Point2::new( 1_f64, 0_f64),
    ///     Point2::new(-1_f64, 0_f64),
    ///     Point2::new( 0_f64, 0_f64),
    /// ];
    /// let expected_in_line = vertices_in_line;
    /// let result_in_line = vertices_in_line.map(|p| shear * p);
    /// 
    /// assert_relative_eq!(result_in_line[0], expected_in_line[0], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_line[1], expected_in_line[1], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_line[2], expected_in_line[2], epsilon = 1e-10);
    /// ```
    /// 
    /// Shearing along the line `y == (1 / 2) * x + 1` using the origin `(2, 2)`.
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
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
    ///          (3_f64 / f64::sqrt(5_f64)) + (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64
    ///     ),
    ///     Point2::new(
    ///         -(3_f64 / f64::sqrt(5_f64)) + (2_f64 / f64::sqrt(5_f64)) * shear_factor,
    ///          (1_f64 / f64::sqrt(5_f64))  + (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64
    ///     ),
    ///     Point2::new(
    ///         -(1_f64 / f64::sqrt(5_f64)) - (2_f64 / f64::sqrt(5_f64)) * shear_factor,
    ///         -(3_f64 / f64::sqrt(5_f64)) - (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64
    ///     ),
    ///     Point2::new(
    ///          (3_f64 / f64::sqrt(5_f64)) - (2_f64 / f64::sqrt(5_f64)) * shear_factor,
    ///         -(1_f64 / f64::sqrt(5_f64)) - (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64
    ///     ),
    /// ];
    /// let result = vertices.map(|p| shear * p);
    /// 
    /// assert_relative_eq!(result[0], expected[0], epsilon = 1e-10);
    /// assert_relative_eq!(result[1], expected[1], epsilon = 1e-10);
    /// assert_relative_eq!(result[2], expected[2], epsilon = 1e-10);
    /// assert_relative_eq!(result[3], expected[3], epsilon = 1e-10);
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
    /// assert_relative_eq!(result_in_plane[0], expected_in_plane[0], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_plane[1], expected_in_plane[1], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_plane[2], expected_in_plane[2], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_plane[3], expected_in_plane[3], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_plane[4], expected_in_plane[4], epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn from_affine_shear(
        shear_factor: S, 
        origin: &Point2<S>, 
        direction: &Unit<Vector2<S>>, 
        normal: &Unit<Vector2<S>>
    ) -> Self
    {
        Self {
            shear_factor, 
            origin: *origin, 
            direction: direction.into_inner(), 
            normal: normal.into_inner(),
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
    /// let sheared_point = shear.apply_point(&point);
    /// let result = shear_inv.apply_point(&sheared_point);
    ///
    /// assert_eq!(result, expected);
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
    /// let sheared_vector = shear.apply_vector(&vector);
    /// let result = shear.inverse_apply_vector(&sheared_vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_apply_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        let origin = self.origin.to_vector();
        let factor = self.shear_factor * (vector - origin).dot(&self.normal);

        vector - self.direction * factor
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
    /// let sheared_point = shear.apply_point(&point);
    /// let result = shear.inverse_apply_point(&sheared_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_apply_point(&self, point: &Point2<S>) -> Point2<S> {
        let factor = self.shear_factor * (point - self.origin).dot(&self.normal);

        point - self.direction * factor
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
    /// # use cglinalg_transform::{
    /// #     Shear2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// # };
    /// #
    /// let shear = Shear2::from_shear(-5_f64, 7_f64);
    /// let expected = Matrix3x3::new(
    ///      1_f64, 7_f64, 0_f64,
    ///     -5_f64, 1_f64, 0_f64,
    ///      0_f64, 0_f64, 1_f64
    /// );
    /// let result = shear.to_affine_matrix();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix3x3<S> {
        let direction = Unit::from_value(self.direction);
        let normal = Unit::from_value(self.normal);

        Matrix3x3::from_affine_shear(self.shear_factor, &self.origin, &direction, &normal)
    }

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
    #[inline]
    pub fn to_transform(&self) -> Transform2<S> {
        Transform2::from_matrix_unchecked(self.to_affine_matrix())
    }
}

impl<S> From<Shear2<S>> for Matrix3x3<S>
where 
    S: SimdScalarFloat
{
    #[inline]
    fn from(shear: Shear2<S>) -> Matrix3x3<S> {
        shear.to_affine_matrix()
    }
}

impl<S> From<&Shear2<S>> for Matrix3x3<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn from(shear: &Shear2<S>) -> Matrix3x3<S> {
        shear.to_affine_matrix()
    }
}


impl<S> Shear3<S> 
where 
    S: SimdScalarSigned 
{
    /// Construct a shearing matrix in three dimensions with respect to 
    /// a plane passing through the origin `[0, 0, 0]`, using the **x-axis**
    /// as the shearing direction, and the **y-axis** as the normal vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point3, 
    /// # };
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
    /// let result_in_plane = vertices.map(|p| shear * p);
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

    /// Construct a shearing matrix in three dimensions with respect to 
    /// a plane passing through the origin `[0, 0, 0]`, using the **x-axis**
    /// as the shearing direction, and the **y-axis** as the normal vector.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
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

    /// Construct a shearing matrix in three dimensions with respect to 
    /// a plane passing through the origin `[0, 0, 0]`, using the **y-axis**
    /// as the shearing direction, and the **x-axis** as the normal vector.
    /// 
    /// This version of the shearing transformation is a linear transformation because 
    /// the origin of the coordinate frame for applying the shearing transformation 
    /// is `[0, 0, 0]` so there is no translation term.
    /// 
    /// For a more in depth exposition on the geometrical underpinnings of the shearing
    /// transformation in general, see [`Matrix3x3::from_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point3, 
    /// # };
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

    /// Construct a shearing matrix in three dimensions with respect to 
    /// a plane passing through the origin `[0, 0, 0]`, using the **y-axis**
    /// as the shearing direction, and the **z-axis** as the normal vector.
    ///
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
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

    /// Construct a shearing matrix in three dimensions with respect to 
    /// a plane passing through the origin `[0, 0, 0]`, using the **z-axis**
    /// as the shearing direction, and the **x-axis** as the normal vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point3, 
    /// # };
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

    /// Construct a shearing matrix in three dimensions with respect to 
    /// a plane passing through the origin `[0, 0, 0]`, using the **z-axis**
    /// as the shearing direction, and the **y-axis** as the normal vector.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
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
    S: SimdScalarFloat
{
    /// Construct a general shearing matrix in three dimensions with respect to 
    /// a plane passing through the origin `[0, 0, 0]`.
    /// 
    /// # Parameters
    /// 
    /// The shearing matrix constructor has the following parameters
    /// * `shear_factor`: The amount by which a point in a plane parallel to the shearing 
    ///    plane gets sheared.
    /// * `direction`: The direction along which the shearing happens.
    /// * `normal`: The normal vector to the shearing plane.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3,
    /// #     Unit,
    /// # };
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
    pub fn from_shear(shear_factor: S, direction: &Unit<Vector3<S>>, normal: &Unit<Vector3<S>>) -> Self {
        Self {
            shear_factor,
            origin: Point3::origin(),
            direction: direction.into_inner(),
            normal: normal.into_inner(),
        }
    }

    /// Construct a general affine shearing matrix in three dimensions with respect to 
    /// a plane passing through the origin `origin`, not necessarily `[0, 0, 0]`.
    /// 
    /// # Parameters
    /// 
    /// The affine shearing matrix constructor has four parameters
    /// * `origin`: The origin of the affine frame for the shearing transformation.
    /// * `shear_factor`: The amount by which a point in a plane parallel to the shearing 
    ///    line gets sheared.
    /// * `direction`: The direction along which the shearing happens in the shearing line.
    /// * `normal`: The normal vector to the shearing line.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_transform::{
    /// #     Shear3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
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
    /// assert_relative_eq!(result[0], expected[0], epsilon = 1e-10);
    /// assert_relative_eq!(result[1], expected[1], epsilon = 1e-10);
    /// assert_relative_eq!(result[2], expected[2], epsilon = 1e-10);
    /// assert_relative_eq!(result[3], expected[3], epsilon = 1e-10);
    /// assert_relative_eq!(result[4], expected[4], epsilon = 1e-10);
    /// assert_relative_eq!(result[5], expected[5], epsilon = 1e-10);
    /// assert_relative_eq!(result[6], expected[6], epsilon = 1e-10);
    /// assert_relative_eq!(result[7], expected[7], epsilon = 1e-10);
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
    /// assert_relative_eq!(result_in_plane[0], expected_in_plane[0], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_plane[1], expected_in_plane[1], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_plane[2], expected_in_plane[2], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_plane[3], expected_in_plane[3], epsilon = 1e-10);
    /// assert_relative_eq!(result_in_plane[4], expected_in_plane[4], epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn from_affine_shear(
        shear_factor: S, 
        origin: &Point3<S>, 
        direction: &Unit<Vector3<S>>, 
        normal: &Unit<Vector3<S>>
    ) -> Self 
    {
        Self {
            shear_factor, 
            origin: *origin, 
            direction: direction.into_inner(), 
            normal: normal.into_inner(),
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
    /// let sheared_point = shear.apply_point(&point);
    /// let result = shear_inv.apply_point(&sheared_point);
    ///
    /// assert_eq!(result, expected);
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
    /// let sheared_vector = shear.apply_vector(&vector);
    /// let result = shear.inverse_apply_vector(&sheared_vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_apply_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let origin = self.origin.to_vector();
        let factor = self.shear_factor * (vector - origin).dot(&self.normal);

        vector - self.direction * factor
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
    /// let sheared_point = shear.apply_point(&point);
    /// let result = shear.inverse_apply_point(&sheared_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_apply_point(&self, point: &Point3<S>) -> Point3<S> {
        let factor = self.shear_factor * (point - self.origin).dot(&self.normal);

        point - self.direction * factor
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
    /// # use cglinalg_transform::{
    /// #     Shear3,
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
    /// let expected = Matrix4x4::new(
    ///     1_f64, 11_f64, 17_f64, 0_f64,
    ///     5_f64, 1_f64,  19_f64, 0_f64,
    ///     7_f64, 13_f64, 1_f64,  0_f64,
    ///     0_f64, 0_f64,  0_f64,  1_f64
    /// );
    /// let result = shear.to_affine_matrix();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix4x4<S> {
        let direction = Unit::from_value(self.direction);
        let normal = Unit::from_value(self.normal);

        Matrix4x4::from_affine_shear(self.shear_factor, &self.origin, &direction, &normal)
    }

    /// Convert a shear transformation into a generic transformation.
    /// 
    /// # Example
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
    pub fn to_transform(&self) -> Transform3<S> {
        Transform3::from_matrix_unchecked(self.to_affine_matrix())
    }
}

impl<S> From<Shear3<S>> for Matrix4x4<S>
where 
    S: SimdScalarFloat
{
    #[inline]
    fn from(shear: Shear3<S>) -> Matrix4x4<S> {
        shear.to_affine_matrix()
    }
}

impl<S> From<&Shear3<S>> for Matrix4x4<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn from(shear: &Shear3<S>) -> Matrix4x4<S> {
        shear.to_affine_matrix()
    }
}

