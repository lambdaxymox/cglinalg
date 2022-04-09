use crate::common::{
    Scalar,
    ScalarFloat,
};
use crate::matrix::{
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

use core::fmt;
use core::ops;


/// A generic two dimensional transformation.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform2<S> {
    /// The underlying matrix that implements the transformation.
    matrix: Matrix3x3<S>,
}

impl<S> Transform2<S> 
where 
    S: Scalar 
{
    /// Convert a 3x3 matrix to a two-dimensional transformation. This 
    /// function is for internal use in implementing type conversions.
    #[inline]
    pub(crate) fn from_specialized<T: Into<Matrix3x3<S>>>(transform: T) -> Self {
        // TODO: Make this function const when const fn stabilizes for traits other than
        // Sized. See issue #57563: <https://github.com/rust-lang/rust/issues/57563>
        Self {
            matrix: transform.into(),
        }
    }

    /// Construct a new transformation from a given homogeneous matrix. The input
    /// matrix is not checked that it is a valid homogeneous matrix.
    #[inline]
    pub fn from_matrix_unchecked(matrix: Matrix3x3<S>) -> Self {
        Self {
            matrix: matrix,
        }
    }

    /// Get a reference to the underlying matrix that represents the 
    /// transformation.
    #[inline]
    pub fn matrix(&self) -> &Matrix3x3<S> {
        &self.matrix
    }

    /// Get a mutable reference to the underlying matrix that represents the 
    /// transformation.
    #[inline]
    pub fn matrix_mut(&mut self) -> &mut Matrix3x3<S> {
        &mut self.matrix
    }

    /// The identity transformation for a generic two-dimensional 
    /// transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Transform2,
    /// #     Point2,
    /// # };
    /// #
    /// let transform = Transform2::identity();
    /// let point = Point2::new(1_f64, 2_f64);
    ///
    /// assert_eq!(transform * point, point);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self { 
            matrix: Matrix3x3::identity(),
        }
    }
}

impl<S> Transform2<S> 
where 
    S: ScalarFloat 
{
    /// Compute the inverse of the transformation if it exists.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Transform2,
    /// #     Point2,
    /// #     Matrix3x3,
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     0_f64, -1_f64, 0_f64,
    ///     1_f64,  0_f64, 0_f64,
    ///     2_f64,  3_f64, 1_f64  
    /// );
    /// let transform = Transform2::from_matrix_unchecked(matrix);
    /// let transform_inv = transform.inverse().unwrap();
    ///
    /// assert_eq!(transform * transform_inv, Transform2::identity());
    /// ```
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        self.matrix.inverse().map(|matrix| Self { matrix })
    }

    /// Apply the transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Transform2,
    /// #     Vector2,
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     6_f64, 7_f64, 0_f64,
    ///     8_f64, 9_f64, 0_f64,
    ///     2_f64, 3_f64, 1_f64,   
    /// );
    /// let transform = Transform2::from_matrix_unchecked(matrix);
    /// let vector = Vector2::new(1_f64, 2_f64);
    /// let expected = Vector2::new(22_f64, 25_f64);
    /// let result = transform.transform_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn transform_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    /// Apply the transformation to a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Transform2,
    /// #     Point2,
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     6_f64, 7_f64, 0_f64,
    ///     8_f64, 9_f64, 0_f64,
    ///     2_f64, 3_f64, 1_f64,   
    /// );
    /// let transform = Transform2::from_matrix_unchecked(matrix);
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = Point2::new(24_f64, 28_f64);
    /// let result = transform.transform_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn transform_point(&self, point: &Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(&(self.matrix * point.to_homogeneous())).unwrap()
    }

    /// Apply the inverse of the transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Transform2,
    /// #     Vector2,
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     6_f64, 7_f64, 0_f64,
    ///     8_f64, 9_f64, 0_f64,
    ///     2_f64, 3_f64, 1_f64,   
    /// );
    /// let transform = Transform2::from_matrix_unchecked(matrix);
    /// let vector = Vector2::new(1_f64, 2_f64);
    /// let expected = Some(vector);
    /// let transformed_vector = transform.transform_vector(&vector);
    /// let result = transform.inverse_transform_vector(&transformed_vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_transform_vector(&self, vector: &Vector2<S>) -> Option<Vector2<S>> {
        self.inverse()
            .map(|matrix_inverse| matrix_inverse.transform_vector(vector))
    }

    /// Apply the inverse of the transformation to a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Transform2,
    /// #     Point2,
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     6_f64, 7_f64, 0_f64,
    ///     8_f64, 9_f64, 0_f64,
    ///     2_f64, 3_f64, 1_f64,   
    /// );
    /// let transform = Transform2::from_matrix_unchecked(matrix);
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = Some(point);
    /// let transformed_point = transform.transform_point(&point);
    /// let result = transform.inverse_transform_point(&transformed_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_transform_point(&self, point: &Point2<S>) -> Option<Point2<S>> {
        self.inverse()
            .map(|matrix_inverse| matrix_inverse.transform_point(point))
    }
}

impl<S> AsRef<Matrix3x3<S>> for Transform2<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3x3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Transform2<S> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Transform2 [{}]",
            self.matrix
        )
    }
}

impl<S> From<Transform2<S>> for Matrix3x3<S> 
where 
    S: Copy 
{
    #[inline]
    fn from(transformation: Transform2<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> From<&Transform2<S>> for Matrix3x3<S> 
where 
    S: Copy 
{
    #[inline]
    fn from(transformation: &Transform2<S>) -> Matrix3x3<S> {
        transformation.matrix
    }
}

impl<S> approx::AbsDiffEq for Transform2<S> 
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
        Matrix3x3::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S> approx::RelativeEq for Transform2<S> 
where 
    S: ScalarFloat 
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

impl<S> approx::UlpsEq for Transform2<S> 
where 
    S: ScalarFloat 
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

impl<S> ops::Mul<Point2<S>> for Transform2<S> 
where 
    S: ScalarFloat 
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<S> ops::Mul<&Point2<S>> for Transform2<S> 
where 
    S: ScalarFloat 
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: &Point2<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Point2<S>> for &Transform2<S> 
where 
    S: ScalarFloat 
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: Point2<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point2<S>> for &'b Transform2<S> 
where 
    S: ScalarFloat 
{
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: &'a Point2<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Transform2<S>> for Transform2<S> 
where 
    S: ScalarFloat 
{
    type Output = Transform2<S>;

    #[inline]
    fn mul(self, other: Transform2<S>) -> Self::Output {
        Transform2 {
            matrix: self.matrix * other.matrix
        }
    }
}

impl<S> ops::Mul<&Transform2<S>> for Transform2<S> 
where 
    S: ScalarFloat 
{
    type Output = Transform2<S>;

    #[inline]
    fn mul(self, other: &Transform2<S>) -> Self::Output {
        Transform2 {
            matrix: self.matrix * other.matrix
        }
    }
}

impl<S> ops::Mul<Transform2<S>> for &Transform2<S> 
where 
    S: ScalarFloat 
{
    type Output = Transform2<S>;

    #[inline]
    fn mul(self, other: Transform2<S>) -> Self::Output {
        Transform2 {
            matrix: self.matrix * other.matrix
        }
    }
}

impl<'a, 'b, S> ops::Mul<&'a Transform2<S>> for &'b Transform2<S> 
where 
    S: ScalarFloat 
{
    type Output = Transform2<S>;

    #[inline]
    fn mul(self, other: &'a Transform2<S>) -> Self::Output {
        Transform2 {
            matrix: self.matrix * other.matrix
        }
    }
}


/// A generic three-dimensional transformation.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform3<S> {
    /// The underlying matrix implementing the transformation.
    matrix: Matrix4x4<S>,
}

impl<S> Transform3<S> 
where 
    S: Scalar 
{
    /// Convert a 4x4 matrix to a three-dimensional transformation. 
    /// This function is for internal use in implementing type conversions.
    #[inline]
    pub(crate) fn from_specialized<T: Into<Matrix4x4<S>>>(transform: T) -> Self {
        // TODO: Make this function const when const fn stabilizes for traits other than
        // Sized. See issue #57563: <https://github.com/rust-lang/rust/issues/57563>.
        Self {
            matrix: transform.into(),
        }
    }

    /// Construct a new transformation from a given homogeneous matrix. The input
    /// matrix is not checked that it is a valid homogeneous matrix.
    #[inline]
    pub fn from_matrix_unchecked(matrix: Matrix4x4<S>) -> Self {
        Self {
            matrix: matrix,
        }
    }

    /// Get a reference to the underlying matrix that represents the 
    /// transformation.
    #[inline]
    pub fn matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Get a mutable reference to the underlying matrix that represents the 
    /// transformation.
    #[inline]
    pub fn matrix_mut(&mut self) -> &mut Matrix4x4<S> {
        &mut self.matrix
    }

    /// The identity transformation for a generic three-dimensional 
    /// transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Transform3,
    /// #     Point3,
    /// # };
    /// #
    /// let transform = Transform3::identity();
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    ///
    /// assert_eq!(transform * point, point);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self { 
            matrix: Matrix4x4::identity(),
        }
    }
}

impl<S> Transform3<S> 
where 
    S: ScalarFloat 
{
    /// Compute the inverse of the transformation if it exists.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Transform3,
    /// #     Point3,
    /// #     Matrix4x4,
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     0_f64, -1_f64, 0_f64, 0_f64,
    ///     1_f64,  0_f64, 0_f64, 0_f64,
    ///     0_f64,  0_f64, 1_f64, 0_f64,
    ///     2_f64,  3_f64, 4_f64, 1_f64  
    /// );
    /// let transform = Transform3::from_matrix_unchecked(matrix);
    /// let transform_inv = transform.inverse().unwrap();
    ///
    /// assert_eq!(transform * transform_inv, Transform3::identity());
    /// ```
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        self.matrix.inverse().map(|matrix| Self { matrix })
    }

    /// Apply the transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Transform3,
    /// #     Vector3,
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     5_f64,  6_f64,  5_f64, 0_f64,
    ///     2_f64,  5_f64,  8_f64, 0_f64,
    ///     12_f64, 15_f64, 9_f64, 0_f64, 
    ///     6_f64,  6_f64,  6_f64, 1_f64
    /// );
    /// let transform = Transform3::from_matrix_unchecked(matrix);
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Vector3::new(45_f64, 61_f64, 48_f64);
    /// let result = transform.transform_vector(&vector);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    /// Apply the inverse of the transformation to a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Transform3,
    /// #     Point3,
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     5_f64,  6_f64,  5_f64, 0_f64,
    ///     2_f64,  5_f64,  8_f64, 0_f64,
    ///     12_f64, 15_f64, 9_f64, 0_f64, 
    ///     6_f64,  6_f64,  6_f64, 1_f64
    /// );
    /// let transform = Transform3::from_matrix_unchecked(matrix);
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Point3::new(51_f64, 67_f64, 54_f64);
    /// let result = transform.transform_point(&point);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(&(self.matrix * point.to_homogeneous())).unwrap()
    }

    /// Apply the inverse of the transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Transform3,
    /// #     Vector3,
    /// #     Matrix4x4, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     5_f64,  6_f64,  5_f64, 0_f64,
    ///     2_f64,  5_f64,  8_f64, 0_f64,
    ///     12_f64, 15_f64, 9_f64, 0_f64, 
    ///     6_f64,  6_f64,  6_f64, 1_f64
    /// );
    /// let transform = Transform3::from_matrix_unchecked(matrix);
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = vector;
    /// let transformed_vector = transform.transform_vector(&vector);
    /// let result = transform.inverse_transform_vector(&transformed_vector).unwrap();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_transform_vector(&self, vector: &Vector3<S>) -> Option<Vector3<S>> {
        self.inverse()
            .map(|matrix_inverse| matrix_inverse.transform_vector(vector))
    }

    /// Apply the inverse of the transformation to a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Transform3,
    /// #     Point3,
    /// #     Matrix4x4, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     5_f64,  6_f64,  5_f64, 0_f64,
    ///     2_f64,  5_f64,  8_f64, 0_f64,
    ///     12_f64, 15_f64, 9_f64, 0_f64, 
    ///     6_f64,  6_f64,  6_f64, 1_f64
    /// );
    /// let transform = Transform3::from_matrix_unchecked(matrix);
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = point;
    /// let transformed_point = transform.transform_point(&point);
    /// let result = transform.inverse_transform_point(&transformed_point).unwrap();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_transform_point(&self, point: &Point3<S>) -> Option<Point3<S>> {
        self.inverse()
            .map(|matrix_inverse| matrix_inverse.transform_point(point))
    }
}

impl<S> AsRef<Matrix4x4<S>> for Transform3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Transform3<S> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Transform3 [{}]",
            self.matrix
        )
    }
}

impl<S> From<Transform3<S>> for Matrix4x4<S> 
where 
    S: Copy 
{
    #[inline]
    fn from(transformation: Transform3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> From<&Transform3<S>> for Matrix4x4<S> 
where 
    S: Copy 
{
    #[inline]
    fn from(transformation: &Transform3<S>) -> Matrix4x4<S> {
        transformation.matrix
    }
}

impl<S> approx::AbsDiffEq for Transform3<S> 
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
        Matrix4x4::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S> approx::RelativeEq for Transform3<S> 
where 
    S: ScalarFloat 
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix4x4::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Transform3<S> 
where 
    S: ScalarFloat 
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix4x4::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S> ops::Mul<Point3<S>> for Transform3<S> 
where 
    S: ScalarFloat 
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for Transform3<S> 
where 
    S: ScalarFloat 
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &Transform3<S> 
where 
    S: ScalarFloat 
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b Transform3<S> 
where 
    S: ScalarFloat 
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S> ops::Mul<Transform3<S>> for Transform3<S> 
where 
    S: ScalarFloat 
{
    type Output = Transform3<S>;

    #[inline]
    fn mul(self, other: Transform3<S>) -> Self::Output {
        Transform3 {
            matrix: self.matrix * other.matrix
        }
    }
}

impl<S> ops::Mul<&Transform3<S>> for Transform3<S> 
where 
    S: ScalarFloat 
{
    type Output = Transform3<S>;

    #[inline]
    fn mul(self, other: &Transform3<S>) -> Self::Output {
        Transform3 {
            matrix: self.matrix * other.matrix
        }
    }
}

impl<S> ops::Mul<Transform3<S>> for &Transform3<S> 
where 
    S: ScalarFloat 
{
    type Output = Transform3<S>;

    #[inline]
    fn mul(self, other: Transform3<S>) -> Self::Output {
        Transform3 {
            matrix: self.matrix * other.matrix
        }
    }
}

impl<'a, 'b, S> ops::Mul<&'a Transform3<S>> for &'b Transform3<S> 
where 
    S: ScalarFloat 
{
    type Output = Transform3<S>;

    #[inline]
    fn mul(self, other: &'a Transform3<S>) -> Self::Output {
        Transform3 {
            matrix: self.matrix * other.matrix
        }
    }
}

