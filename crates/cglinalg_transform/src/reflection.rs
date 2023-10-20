use cglinalg_numeric::{
    SimdScalarFloat,
};
use cglinalg_core::{
    Const,
    ShapeConstraint,
    DimAdd,
    DimSub,
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
    Transform,
};

use core::fmt;
use core::ops;


/// A reflection matrix in two dimensions.
pub type Reflection2<S> = Reflection<S, 2, 3>;

/// A reflection matrix in three dimensions.
pub type Reflection3<S> = Reflection<S, 3, 4>;


/// A reflection transformation about a plane.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reflection<S, const N: usize, const NPLUS1: usize> 
where
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    bias: Point<S, N>,
    normal: Vector<S, N>,
    matrix: Matrix<S, NPLUS1, NPLUS1>,
}

impl<S, const N: usize, const NPLUS1: usize> Reflection<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    /*
    /// Construct a new reflection transformation from the vector normal to the 
    /// plane of reflection.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// A reflection about the **y-axis** using the origin as the bias.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Point2,
    /// #     Unit,
    /// # };
    /// #
    /// // Normal to the plane of reflection.
    /// let normal = Unit::from_value(Vector2::unit_x());
    /// let bias = Point2::origin();
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let vector = Vector2::new(1_f64, 2_f64);
    /// let expected = Vector2::new(-1_f64, 2_f64);
    /// let result = reflection.apply_vector(&vector);
    /// 
    /// assert_eq!(result, expected);
    /// 
    /// // In two dimensions, we can just as well use the opposite normal.
    /// let opposite_normal = Unit::from_value(-Vector2::unit_x());
    /// let opposite_reflection = Reflection2::from_normal_bias(&opposite_normal, &bias);
    /// let opposite_result = opposite_reflection.apply_vector(&vector);
    /// 
    /// assert_eq!(opposite_result, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// A reflection about the plane `x + y == -z`.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector3::new(1_f64, 1_f64, 1_f64));
    /// let bias = Point3::origin();
    /// let reflection = Reflection3::from_normal_bias(&normal, &bias);
    /// let vector = Vector3::new(-5_f64, 7_f64, -3_f64);
    /// let expected = Vector3::new(-13_f64 / 3_f64, 23_f64 / 3_f64, -7_f64 / 3_f64);
    /// let result = reflection.apply_vector(&vector);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-15);
    /// ```
    #[inline]
    pub fn from_normal_bias(normal: &Unit<Vector<S, N>>, bias: &Vector<S, N>) -> Self {
        Self {
            bias: *bias,
            normal: normal.into_inner(),
            matrix: Matrix::from_affine_reflection(normal, bias),
        }
    }
    */

    /// Return the bias for calculating the reflections.
    ///
    /// The `bias` is the coordinates of a known point in the plane of 
    /// reflection.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// A reflection about the **y-axis** using the origin as the bias.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Point2,
    /// #     Unit,
    /// # };
    /// #
    /// // Normal to the plane of reflection.
    /// let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_x());
    /// let bias = Point2::origin();
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// 
    /// assert_eq!(reflection.bias(), bias);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// A reflection about the plane `x + y == -z`.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector3::new(1_f64, 1_f64, 1_f64));
    /// let bias = Point3::origin();
    /// let reflection = Reflection3::from_normal_bias(&normal, &bias);
    /// 
    /// assert_eq!(reflection.bias(), bias);
    /// ```
    #[inline]
    pub const fn bias(&self) -> Point<S, N> {
        self.bias
    }

    /// Return the normal vector to the reflection plane.
    /// 
    /// There is an ambiguity in the choice of normal to a line in
    /// two dimensions. One can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection transformation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// A reflection about the **y-axis** using the origin as the bias.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Point2,
    /// #     Unit,
    /// # };
    /// #
    /// // Normal to the plane of reflection.
    /// let normal = Vector2::unit_x();
    /// let unit_normal: Unit<Vector2<f64>> = Unit::from_value(normal);
    /// let bias = Point2::origin();
    /// let reflection = Reflection2::from_normal_bias(&unit_normal, &bias);
    /// 
    /// assert_eq!(reflection.normal(), normal);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// A reflection about the plane `x + y == -z`.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// #
    /// let normal = Vector3::new(1_f64, 1_f64, 1_f64);
    /// let unit_normal = Unit::from_value(normal);
    /// let bias = Point3::origin();
    /// let reflection = Reflection3::from_normal_bias(&unit_normal, &bias);
    /// let expected = normal / normal.norm();
    /// let result = reflection.normal();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn normal(&self) -> Vector<S, N> {
        self.normal
    }

    /// Return the underlying matrix of the reflection transformation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// A reflection about the **y-axis** using the origin as the bias.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector2,
    /// #     Point2,
    /// #     Unit,
    /// # };
    /// #
    /// // Normal to the plane of reflection.
    /// let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_x());
    /// let bias = Point2::origin();
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let expected = Matrix3x3::new(
    ///     -1_f64, 0_f64, 0_f64,
    ///      0_f64, 1_f64, 0_f64,
    ///      0_f64, 0_f64, 1_f64
    /// );
    /// let result = reflection.matrix();
    /// 
    /// assert_eq!(result, &expected);
    /// 
    /// // In two dimensions, we can just as well use the opposite normal.
    /// let opposite_normal = Unit::from_value(-Vector2::unit_x());
    /// let opposite_reflection = Reflection2::from_normal_bias(&opposite_normal, &bias);
    /// 
    /// assert_eq!(opposite_reflection.matrix(), reflection.matrix());
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// A reflection about the plane `x + y == -z`.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector3::new(1_f64, 1_f64, 1_f64));
    /// let bias = Point3::origin();
    /// let reflection = Reflection3::from_normal_bias(&normal, &bias);
    /// let expected = Matrix4x4::new(
    ///      1_f64 / 3_f64, -2_f64 / 3_f64, -2_f64 / 3_f64, 0_f64,
    ///     -2_f64 / 3_f64,  1_f64 / 3_f64, -2_f64 / 3_f64, 0_f64,
    ///     -2_f64 / 3_f64, -2_f64 / 3_f64,  1_f64 / 3_f64, 0_f64,
    ///      0_f64,          0_f64,          0_f64,         1_f64
    /// );
    /// let result = reflection.matrix();
    /// 
    /// assert_relative_eq!(result, &expected, epsilon = 1e-15);
    /// ```
    #[inline]
    pub const fn matrix(&self) -> &Matrix<S, NPLUS1, NPLUS1> {
        &self.matrix
    }
}

impl<S, const N: usize, const NPLUS1: usize> Reflection<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>
{
    /// Reflect a vector across the plane described by the reflection 
    /// transformation.
    /// 
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Point2,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Reflection2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
    ///     -1_f64 / 2_f64, 
    ///      1_f64
    /// ));
    /// let bias = Point2::new(0_f64, 1_f64);
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let vector = Vector2::new(1_f64, 1_f64);
    /// let expected = Vector2::new(7_f64 / 5_f64, 1_f64 / 5_f64);
    /// let result = reflection.apply_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Reflection3,
    /// # };
    /// #
    /// let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let bias = Point3::origin();
    /// let reflection = Reflection3::from_normal_bias(&normal, &bias);
    /// let vector = Vector3::new(1_f64, 1_f64, 1_f64);
    /// let expected = Vector3::new(1_f64, 1_f64, -1_f64);
    /// let result = reflection.apply_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn apply_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        let one = S::one();
        let two = one + one;
        let factor = vector.dot(&self.normal) * two;

        vector - self.normal * factor
    }

    /// Reflect a point across the plane described by the reflection 
    /// transformation.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Reflection2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
    ///     -1_f64 / 2_f64, 
    ///      1_f64
    /// ));
    /// let bias = Point2::new(0_f64, 1_f64);
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let point = Point2::new(1_f64, 1_f64);
    /// let expected = Point2::new(3_f64 / 5_f64, 9_f64 / 5_f64);
    /// let result = reflection.apply_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Reflection3,
    /// # };
    /// #
    /// let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let bias = Point3::origin();
    /// let reflection = Reflection3::from_normal_bias(&normal, &bias);
    /// let point = Point3::new(1_f64, 1_f64, 1_f64);
    /// let expected = Point3::new(1_f64, 1_f64, -1_f64);
    /// let result = reflection.apply_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn apply_point(&self, point: &Point<S, N>) -> Point<S, N> {
        let one = S::one();
        let two = one + one;
        let factor = (point - self.bias).dot(&self.normal) * two;

        point - self.normal * factor
    }
}

impl<S, const N: usize, const NPLUS1: usize> Reflection<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    /// Compute the inversse of a reflection.
    /// 
    /// The inverse of a reflection transformation is the reflection transformation
    /// itself. That is, given a reflection `r`
    /// ```text
    /// inverse(r) == r
    /// ```
    /// 
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Reflection2,
    /// # };
    /// #
    /// let reflection = Reflection2::identity();
    /// let vector = Vector2::new(1_f64, 2_f64);
    /// let reflection_inv = reflection.inverse();
    /// let expected = vector;
    /// let result = reflection_inv * (reflection * vector);
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
    /// #     Reflection3,
    /// # };
    /// #
    /// let reflection = Reflection3::identity();
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let reflection_inv = reflection.inverse();
    /// let expected = vector;
    /// let result = reflection_inv * (reflection * vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse(&self) -> Self {
        *self
    }

    /// Compute the identity reflection. 
    ///
    /// The identity reflection is a reflection that does not move a point 
    /// or vector.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Reflection2,
    /// # };
    /// #
    /// let reflection = Reflection2::identity();
    /// let vector = Vector2::new(1_f64, 2_f64);
    /// let expected = vector;
    /// let result = reflection.apply_vector(&vector);
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
    /// #     Reflection3,
    /// # };
    /// #
    /// let reflection = Reflection3::identity();
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = vector;
    /// let result = reflection.apply_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self { 
            bias: Point::origin(),
            normal: Vector::zero(),
            matrix: Matrix::identity(),
        }
    }

    /// Convert a reflection to an affine matrix.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// A reflection about the plane `y == 2 * x`.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector2,
    /// #     Point2,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector2::new(-2_f64, 1_f64));
    /// let bias = Point2::origin();
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let expected = Matrix3x3::new(
    ///     -3_f64 / 5_f64, 4_f64 / 5_f64, 0_f64,
    ///      4_f64 / 5_f64, 3_f64 / 5_f64, 0_f64,
    ///      0_f64,         0_f64,         1_f64
    /// );
    /// let result = reflection.to_affine_matrix();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-15);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// A reflection about the plane `x + y == -z`.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector3::new(1_f64, 1_f64, 1_f64));
    /// let bias = Point3::origin();
    /// let reflection = Reflection3::from_normal_bias(&normal, &bias);
    /// let expected = Matrix4x4::new(
    ///      1_f64 / 3_f64, -2_f64 / 3_f64, -2_f64 / 3_f64, 0_f64,
    ///     -2_f64 / 3_f64,  1_f64 / 3_f64, -2_f64 / 3_f64, 0_f64,
    ///     -2_f64 / 3_f64, -2_f64 / 3_f64,  1_f64 / 3_f64, 0_f64,
    ///      0_f64,          0_f64,          0_f64,         1_f64
    /// );
    /// let result = reflection.to_affine_matrix();
    /// 
    /// assert_relative_eq!(result, &expected, epsilon = 1e-15);
    /// ```
    #[inline]
    pub const fn to_affine_matrix(&self) -> Matrix<S, NPLUS1, NPLUS1> {
        self.matrix
    }

    /// Convert a reflection to a generic transformation.
    /// 
    /// # Example (Two Dimensions)
    /// 
    /// A reflection about the plane `y == 2 * x`.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection2,
    /// #     Transform2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector2,
    /// #     Point2,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector2::new(-2_f64, 1_f64));
    /// let bias = Point2::origin();
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let expected = Transform2::from_matrix_unchecked(Matrix3x3::new(
    ///     -3_f64 / 5_f64, 4_f64 / 5_f64, 0_f64,
    ///      4_f64 / 5_f64, 3_f64 / 5_f64, 0_f64,
    ///      0_f64,         0_f64,         1_f64
    /// ));
    /// let result = reflection.to_transform();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-15);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    /// 
    /// A reflection about the plane `x + y == -z`.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection3,
    /// #     Transform3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector3::new(1_f64, 1_f64, 1_f64));
    /// let bias = Point3::origin();
    /// let reflection = Reflection3::from_normal_bias(&normal, &bias);
    /// let expected = Transform3::from_matrix_unchecked(Matrix4x4::new(
    ///      1_f64 / 3_f64, -2_f64 / 3_f64, -2_f64 / 3_f64, 0_f64,
    ///     -2_f64 / 3_f64,  1_f64 / 3_f64, -2_f64 / 3_f64, 0_f64,
    ///     -2_f64 / 3_f64, -2_f64 / 3_f64,  1_f64 / 3_f64, 0_f64,
    ///      0_f64,          0_f64,          0_f64,         1_f64
    /// ));
    /// let result = reflection.to_transform();
    /// 
    /// assert_relative_eq!(result, &expected, epsilon = 1e-15);
    /// ```
    #[inline]
    pub fn to_transform(&self) -> Transform<S, N, NPLUS1> {
        Transform::from_matrix_unchecked(self.to_affine_matrix())
    }
}

impl<S, const N: usize, const NPLUS1: usize> AsRef<Matrix<S, NPLUS1, NPLUS1>> for Reflection<S, N, NPLUS1>
where
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    #[inline]
    fn as_ref(&self) -> &Matrix<S, NPLUS1, NPLUS1> {
        &self.matrix
    }
}

impl<S, const N: usize, const NPLUS1: usize> fmt::Display for Reflection<S, N, NPLUS1> 
where
    S: fmt::Display,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Reflection{} [{}]", N, self.matrix)
    }
}

impl<S, const N: usize, const NPLUS1: usize> From<Reflection<S, N, NPLUS1>> for Matrix<S, NPLUS1, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    #[inline]
    fn from(transformation: Reflection<S, N, NPLUS1>) -> Matrix<S, NPLUS1, NPLUS1> {
        transformation.to_affine_matrix()
    }
}

impl<S, const N: usize, const NPLUS1: usize> From<&Reflection<S, N, NPLUS1>> for Matrix<S, NPLUS1, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    #[inline]
    fn from(transformation: &Reflection<S, N, NPLUS1>) -> Matrix<S, NPLUS1, NPLUS1> {
        transformation.to_affine_matrix()
    }
}

impl<S, const N: usize, const NPLUS1: usize> approx::AbsDiffEq for Reflection<S, N, NPLUS1> 
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
        Point::abs_diff_eq(&self.bias, &other.bias, epsilon)
            && Vector::abs_diff_eq(&self.normal, &other.normal, epsilon)
            && Matrix::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S, const N: usize, const NPLUS1: usize> approx::RelativeEq for Reflection<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
       Point::relative_eq(&self.bias, &other.bias, epsilon, max_relative)
           && Vector::relative_eq(&self.normal, &other.normal, epsilon, max_relative)
           && Matrix::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S, const N: usize, const NPLUS1: usize> approx::UlpsEq for Reflection<S, N, NPLUS1> 
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
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        Point::ulps_eq(&self.bias, &other.bias, epsilon, max_ulps)
            && Vector::ulps_eq(&self.normal, &other.normal, epsilon, max_ulps)
            && Matrix::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S, const N: usize, const NPLUS1: usize> ops::Mul<Point<S, N>> for Reflection<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<S, const N: usize, const NPLUS1: usize> ops::Mul<&Point<S, N>> for Reflection<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize, const NPLUS1: usize> ops::Mul<Point<S, N>> for &Reflection<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<'a, 'b, S, const N: usize, const NPLUS1: usize> ops::Mul<&'a Point<S, N>> for &'b Reflection<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &'a Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize, const NPLUS1: usize> ops::Mul<Vector<S, N>> for Reflection<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<S, const N: usize, const NPLUS1: usize> ops::Mul<&Vector<S, N>> for Reflection<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: &Vector<S, N>) -> Self::Output {
        self.apply_vector(other)
    }
}

impl<S, const N: usize, const NPLUS1: usize> ops::Mul<Vector<S, N>> for &Reflection<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<'a, 'b, S, const N: usize, const NPLUS1: usize> ops::Mul<&'a Vector<S, N>> for &'b Reflection<S, N, NPLUS1> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: &'a Vector<S, N>) -> Self::Output {
        self.apply_vector(other)
    }
}


impl<S> Reflection2<S> 
where 
    S: SimdScalarFloat 
{
    /// Construct a new reflection transformation from the vector normal to the 
    /// plane of reflection.
    /// 
    /// # Example
    /// 
    /// A reflection about the **y-axis** using the origin as the bias.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Point2,
    /// #     Unit,
    /// # };
    /// #
    /// // Normal to the plane of reflection.
    /// let normal = Unit::from_value(Vector2::unit_x());
    /// let bias = Point2::origin();
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let vector = Vector2::new(1_f64, 2_f64);
    /// let expected = Vector2::new(-1_f64, 2_f64);
    /// let result = reflection.apply_vector(&vector);
    /// 
    /// assert_eq!(result, expected);
    /// 
    /// // In two dimensions, we can just as well use the opposite normal.
    /// let opposite_normal = Unit::from_value(-Vector2::unit_x());
    /// let opposite_reflection = Reflection2::from_normal_bias(&opposite_normal, &bias);
    /// let opposite_result = opposite_reflection.apply_vector(&vector);
    /// 
    /// assert_eq!(opposite_result, expected);
    /// ```
    #[inline]
    pub fn from_normal_bias(normal: &Unit<Vector2<S>>, bias: &Point2<S>) -> Self {
        Self {
            bias: *bias,
            normal: normal.into_inner(),
            matrix: Matrix3x3::from_affine_reflection(normal, bias),
        }
    }
}

impl<S> Reflection3<S> 
where 
    S: SimdScalarFloat 
{
    /// Construct a new reflection transformation from the vector normal to the 
    /// plane of reflection.
    /// 
    /// # Example
    /// 
    /// A reflection about the plane `x + y == -z`.
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector3::new(1_f64, 1_f64, 1_f64));
    /// let bias = Point3::origin();
    /// let reflection = Reflection3::from_normal_bias(&normal, &bias);
    /// let vector = Vector3::new(-5_f64, 7_f64, -3_f64);
    /// let expected = Vector3::new(-13_f64 / 3_f64, 23_f64 / 3_f64, -7_f64 / 3_f64);
    /// let result = reflection.apply_vector(&vector);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-15);
    /// ```
    #[inline]
    pub fn from_normal_bias(normal: &Unit<Vector3<S>>, bias: &Point3<S>) -> Self {
        Self {
            bias: *bias,
            normal: normal.into_inner(),
            matrix: Matrix4x4::from_affine_reflection(normal, bias),
        }
    }
}

