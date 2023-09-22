use cglinalg_numeric::{
    SimdScalar,
    SimdScalarFloat,
};
use cglinalg_core::{
    ShapeConstraint,
    CanExtend,
    CanContract,
    CanMultiply,
    Const,
    DimAdd,
    DimMul,
    Matrix,
    Vector,
    Vector2,
    Vector3,
    Point,
    Point2,
    Point3,
};

use core::fmt;
use core::ops;


/// A generic two-dimensional transformation in homogeneous coordinates.
pub type Transform2<S> = Transform<S, 2, 3>;

/// A generic three-dimensional transformation in homogeneous coordinates.
pub type Transform3<S> = Transform<S, 3, 4>;


/// A generic transformation in homogeneous coordinates.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform<S, const N: usize, const NPLUS1: usize> 
where
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    matrix: Matrix<S, NPLUS1, NPLUS1>,
}

impl<S, const N: usize, const NPLUS1: usize> Transform<S, N, NPLUS1> 
where 
    S: SimdScalar,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    /*
    /// Convert a matrix to a transformation. 
    /// This function is for internal use in implementing type conversions.
    #[inline]
    pub(crate) fn from_specialized<T: Into<Matrix<S, NPLUS1, NPLUS1>>>(transform: T) -> Self {
        // TODO: Make this function const when const fn stabilizes for traits other than
        // Sized. See issue #57563: <https://github.com/rust-lang/rust/issues/57563>.
        Self {
            matrix: transform.into(),
        }
    }
    */

    /// Construct a new transformation from a given homogeneous matrix. The 
    /// function does not check that the input matrix is a valid homogeneous 
    /// matrix.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Transform2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_6);
    /// let matrix = Matrix3x3::from_affine_angle(angle);
    /// let transform = Transform2::from_matrix_unchecked(matrix);
    /// let vector = Vector2::new(1_f64, 1_f64);
    /// let expected = matrix * vector.to_homogeneous();
    /// let result = transform.transform_vector(&vector).to_homogeneous();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Transform3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// # 
    /// let angle = Radians(f64::consts::FRAC_PI_6);
    /// let matrix = Matrix4x4::from_affine_angle_z(angle);
    /// let transform = Transform3::from_matrix_unchecked(matrix);
    /// let vector = Vector3::new(1_f64, 1_f64, 1_f64);
    /// let expected = matrix * vector.to_homogeneous();
    /// let result = transform.transform_vector(&vector).to_homogeneous();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn from_matrix_unchecked(matrix: Matrix<S, NPLUS1, NPLUS1>) -> Self {
        Self { matrix }
    }

    /// Get a reference to the underlying matrix that represents the transformation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Transform2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_6);
    /// let matrix = Matrix3x3::from_affine_angle(angle);
    /// let transform = Transform2::from_matrix_unchecked(matrix);
    /// 
    /// assert_eq!(transform.matrix(), &matrix);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Transform3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// # 
    /// let angle = Radians(f64::consts::FRAC_PI_6);
    /// let matrix = Matrix4x4::from_affine_angle_z(angle);
    /// let transform = Transform3::from_matrix_unchecked(matrix);
    /// 
    /// assert_eq!(transform.matrix(), &matrix);
    /// ```
    #[inline]
    pub const fn matrix(&self) -> &Matrix<S, NPLUS1, NPLUS1> {
        &self.matrix
    }

    /// Get a mutable reference to the underlying matrix that represents the 
    /// transformation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Transform2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let mut transform = Transform2::identity();
    /// {
    ///     let matrix = transform.matrix_mut();
    ///     matrix[0][0] = 2_f64;
    ///     matrix[1][1] = 3_f64;
    /// }
    /// let expected = Matrix3x3::from_diagonal(&Vector3::new(2_f64, 3_f64, 1_f64));
    /// let result = transform.matrix();
    /// 
    /// assert_eq!(result, &expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Transform3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// #
    /// let mut transform = Transform3::identity();
    /// {
    ///     let matrix = transform.matrix_mut();
    ///     matrix[0][0] = 2_f64;
    ///     matrix[1][1] = 3_f64;
    ///     matrix[2][2] = 4_f64;
    /// }
    /// let expected = Matrix4x4::from_diagonal(&Vector4::new(2_f64, 3_f64, 4_f64, 1_f64));
    /// let result = transform.matrix();
    /// 
    /// assert_eq!(result, &expected);
    /// ```
    #[inline]
    pub fn matrix_mut(&mut self) -> &mut Matrix<S, NPLUS1, NPLUS1> {
        &mut self.matrix
    }

    /// The identity transformation for a generic transformation in homogeneous 
    /// coordinates.
    ///
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Transform2,
    /// # };
    /// #
    /// let transform = Transform2::identity();
    /// let point = Point2::new(1_f64, 2_f64);
    ///
    /// assert_eq!(transform * point, point);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Transform3,
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
            matrix: Matrix::identity(),
        }
    }

    /// Convert a transform to its equivalent matrix representation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Transform2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_6);
    /// let matrix = Matrix3x3::from_affine_angle(angle);
    /// let transform = Transform2::from_matrix_unchecked(matrix);
    /// 
    /// assert_eq!(transform.to_matrix(), matrix);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_transform::{
    /// #     Transform3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use core::f64;
    /// # 
    /// let angle = Radians(f64::consts::FRAC_PI_6);
    /// let matrix = Matrix4x4::from_affine_angle_z(angle);
    /// let transform = Transform3::from_matrix_unchecked(matrix);
    /// 
    /// assert_eq!(transform.to_matrix(), matrix);
    /// ```
    #[inline]
    pub const fn to_matrix(&self) -> Matrix<S, NPLUS1, NPLUS1> {
        self.matrix
    }
}

impl<S, const N: usize, const NPLUS1: usize> Transform<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: CanExtend<Const<N>, Const<NPLUS1>> + CanContract<Const<NPLUS1>, Const<N>>
{
    /// Apply the transformation to a vector.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Matrix3x3, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Transform2,
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
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Matrix4x4, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Transform3,
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
    pub fn transform_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    /// Apply the inverse of the transformation to a point.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Matrix3x3, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Transform2,
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
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Matrix4x4, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Transform3,
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
    pub fn transform_point(&self, point: &Point<S, N>) -> Point<S, N> {
        Point::from_homogeneous(&(self.matrix * point.to_homogeneous())).unwrap()
    }
}

impl<S, const N: usize, const NPLUS1: usize> AsRef<Matrix<S, NPLUS1, NPLUS1>> for Transform<S, N, NPLUS1> 
where
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    #[inline]
    fn as_ref(&self) -> &Matrix<S, NPLUS1, NPLUS1> {
        &self.matrix
    }
}

impl<S, const N: usize, const NPLUS1: usize> fmt::Display for Transform<S, N, NPLUS1> 
where 
    S: fmt::Display,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Transform{} [{}]", N, self.matrix)
    }
}

impl<S, const N: usize, const NPLUS1: usize> From<Transform<S, N, NPLUS1>> for Matrix<S, NPLUS1, NPLUS1> 
where 
    S: SimdScalar,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    #[inline]
    fn from(transform: Transform<S, N, NPLUS1>) -> Matrix<S, NPLUS1, NPLUS1> {
        transform.to_matrix()
    }
}

impl<S, const N: usize, const NPLUS1: usize> From<&Transform<S, N, NPLUS1>> for Matrix<S, NPLUS1, NPLUS1> 
where 
    S: SimdScalar,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    #[inline]
    fn from(transform: &Transform<S, N, NPLUS1>) -> Matrix<S, NPLUS1, NPLUS1> {
        transform.to_matrix()
    }
}

impl<S, const N: usize, const NPLUS1: usize> approx::AbsDiffEq for Transform<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
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

impl<S, const N: usize, const NPLUS1: usize> approx::RelativeEq for Transform<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
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

impl<S, const N: usize, const NPLUS1: usize> approx::UlpsEq for Transform<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
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

impl<S, const N: usize, const NPLUS1: usize> ops::Mul<Point<S, N>> for Transform<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: CanContract<Const<NPLUS1>, Const<N>>,
    ShapeConstraint: CanExtend<Const<N>, Const<NPLUS1>>,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<S, const N: usize, const NPLUS1: usize> ops::Mul<&Point<S, N>> for Transform<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: CanContract<Const<NPLUS1>, Const<N>>,
    ShapeConstraint: CanExtend<Const<N>, Const<NPLUS1>>,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &Point<S, N>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S, const N: usize, const NPLUS1: usize> ops::Mul<Point<S, N>> for &Transform<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: CanContract<Const<NPLUS1>, Const<N>>,
    ShapeConstraint: CanExtend<Const<N>, Const<NPLUS1>>,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.transform_point(&other)
    }
}

impl<'a, 'b, S, const N: usize, const NPLUS1: usize> ops::Mul<&'a Point<S, N>> for &'b Transform<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: CanContract<Const<NPLUS1>, Const<N>>,
    ShapeConstraint: CanExtend<Const<N>, Const<NPLUS1>>,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &'a Point<S, N>) -> Self::Output {
        self.transform_point(other)
    }
}

impl<S, const N: usize, const NPLUS1: usize, const NP1NP1: usize> ops::Mul<Transform<S, N, NPLUS1>> for Transform<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: CanMultiply<Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>
{
    type Output = Transform<S, N, NPLUS1>;

    #[inline]
    fn mul(self, other: Transform<S, N, NPLUS1>) -> Self::Output {
        Transform {
            matrix: self.matrix * other.matrix
        }
    }
}

impl<S, const N: usize, const NPLUS1: usize, const NP1NP1: usize> ops::Mul<&Transform<S, N, NPLUS1>> for Transform<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: CanMultiply<Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>
{
    type Output = Transform<S, N, NPLUS1>;

    #[inline]
    fn mul(self, other: &Transform<S, N, NPLUS1>) -> Self::Output {
        Transform {
            matrix: self.matrix * other.matrix
        }
    }
}

impl<S, const N: usize, const NPLUS1: usize, const NP1NP1: usize> ops::Mul<Transform<S, N, NPLUS1>> for &Transform<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: CanMultiply<Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>
{
    type Output = Transform<S, N, NPLUS1>;

    #[inline]
    fn mul(self, other: Transform<S, N, NPLUS1>) -> Self::Output {
        Transform {
            matrix: self.matrix * other.matrix
        }
    }
}

impl<'a, 'b, S, const N: usize, const NPLUS1: usize, const NP1NP1: usize> ops::Mul<&'a Transform<S, N, NPLUS1>> for &'b Transform<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: CanMultiply<Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>
{
    type Output = Transform<S, N, NPLUS1>;

    #[inline]
    fn mul(self, other: &'a Transform<S, N, NPLUS1>) -> Self::Output {
        Transform {
            matrix: self.matrix * other.matrix
        }
    }
}

impl<S> Transform2<S> 
where 
    S: SimdScalarFloat 
{
    /// Compute the inverse of the transformation if it exists.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Matrix3x3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Transform2,
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

    /// Apply the inverse of the transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Matrix3x3, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Transform2,
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
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Matrix3x3, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Transform2,
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

impl<S> Transform3<S>
where
    S: SimdScalarFloat
{
    /// Compute the inverse of the transformation if it exists.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Matrix4x4,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Transform3,
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

    /// Apply the inverse of the transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Matrix4x4, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Transform3,
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
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Matrix4x4, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Transform3,
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

