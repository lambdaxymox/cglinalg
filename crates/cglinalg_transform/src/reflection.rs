use cglinalg_numeric::{
    SimdScalarFloat,
};
use cglinalg_core::{
    Matrix3x3,
    Matrix4x4,
    Vector,
    Point,
    Unit,
};
use crate::transform::{
    Transform2,
    Transform3,
    Transform,
};

use core::fmt;
use core::ops;


/// A reflection transformation in two dimensions.
pub type Reflection2<S> = Reflection<S, 2>;

/// A reflection transformation in three dimensions.
pub type Reflection3<S> = Reflection<S, 3>;


/// A reflection transformation about a mirror plane.
/// 
/// The normal vector `normal` is a vector perpendicular to the plane of 
/// reflection, the plane in which points are reflected across. This means 
/// that points are moved in a direction parallel to `normal`.
/// 
/// This is the most general reflection type. The vast majority of applications 
/// should use [`Reflection2`] or [`Reflection3`] instead of this type directly.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reflection<S, const N: usize> {
    normal: Vector<S, N>,
    bias: Point<S, N>,
}

impl<S, const N: usize> Reflection<S, N> 
where 
    S: SimdScalarFloat
{
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
    pub fn from_normal_bias(normal: &Unit<Vector<S, N>>, bias: &Point<S, N>) -> Self {
        Self {
            normal: normal.into_inner(),
            bias: *bias,
        }
    }

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
}

impl<S, const N: usize> Reflection<S, N> 
where 
    S: SimdScalarFloat
{
    /// Reflect a vector across the plane described by the reflection 
    /// transformation.
    /// 
    /// # Discussion
    /// 
    /// The reflection of a point is defined as follows. Let `M` be the plane of 
    /// reflection, also known as the **mirror plane**. Let `n` be a vector normal
    /// to the mirror plane `M`. Since `n` is normal to `M`, reflected points are
    /// reflected in a direction parallel to `n`, i.e. perpendicular to the mirror 
    /// plane `M`. To reflect points correctly, we need a known point `Q` in the plane
    /// of reflection. 
    /// 
    /// For a vector `v`, we can choose vectors `v_per` and `v_par` such that 
    /// `v == v_per + v_par`, `v_per` is perpendicular to the `n` and `v_par` is 
    /// parallel to `n`. Stated different, `v_per` is parallel to the mirror plane `M` 
    /// and `v_par` is perpendicular to the mirror plane `M`. The reflection `Ref` acts 
    /// on `v_per` and `v_par` as follows
    /// ```text
    /// Ref(v_per) :=  v_per
    /// Ref(v_par) := -v_par
    /// ```
    /// by definition. This means that the reflection on vectors is defined by
    /// ```text
    /// Ref(v) := Ref(v_per + v_par)
    ///        := Ref(v_per) + Ref(v_par)
    ///        := Ref(v_per) - v_par
    ///        == v_per - v_par
    ///        == v - v_par - v_par
    ///        == v - 2 * v_par
    ///        == v - (2 * dot(v, n)) * n
    /// ```
    /// And reflection on points is defined by
    /// ```text
    /// Ref(P) := Ref(Q + (P - Q))
    ///        := Q + Ref(P - Q)
    ///        == Q + [(P - Q) - 2 * dot(P - Q, n) * n]
    ///        == P - 2 * dot(P - Q, n) * n
    ///        == I * P - (2 * dot(P, n)) * n + (2 * dot(Q, n)) * n
    ///        == [I - 2 * outer(n, n)] * P + (2 * dot(Q, n)) * n
    /// ```
    /// and the corresponding affine matrix has the form
    /// ```text
    /// M := | I - 2 * outer(n, n)   2 * dot(Q, n) * n |
    ///      | 0^T                   1                 |
    /// ```
    /// geometrically. In the standard basis in two-dimensional Euclidean space, we
    /// have
    /// ```text
    ///      |  1 - 2 * n.x * n.x   -2 * n.x * n.y       2 * dot(Q, n) * n.x |
    /// M == | -2 * n.y * n.x        1 - 2 * n.y * n.y   2 * dot(Q, n) * n.y |
    ///      |  0                    0                   1                   |
    /// ```
    /// and in three-dimensional Euclidean space we have
    /// ```text
    ///      |  1 - 2 * n.x * n.x   -2 * n.x * n.y       -2 * n.x * n.z        2 * dot(Q, n) * n.x |
    /// M == | -2 * n.y * n.x        1 - 2 * n.y * n.y   -2 * n.y * n.z        2 * dot(Q, n) * n.y |
    ///      | -2 * n.z * n.x       -2 * n.z * n.y        1 - 2 * n.z * n.z    2 * dot(Q, n) * n.z |
    ///      |  0                    0                    0                    1                   |
    /// ```
    /// which correspond exactly the how the respective matrices are implemented.
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
    /// # Discussion
    /// 
    /// The reflection of a point is defined as follows. Let `M` be the plane of 
    /// reflection, also known as the **mirror plane**. Let `n` be a vector normal
    /// to the mirror plane `M`. Since `n` is normal to `M`, reflected points are
    /// reflected in a direction parallel to `n`, i.e. perpendicular to the mirror 
    /// plane `M`. To reflect points correctly, we need a known point `Q` in the plane
    /// of reflection. 
    /// 
    /// For a vector `v`, we can choose vectors `v_per` and `v_par` such that 
    /// `v == v_per + v_par`, `v_per` is perpendicular to the `n` and `v_par` is 
    /// parallel to `n`. Stated different, `v_per` is parallel to the mirror plane `M` 
    /// and `v_par` is perpendicular to the mirror plane `M`. The reflection `Ref` acts 
    /// on `v_per` and `v_par` as follows
    /// ```text
    /// Ref(v_per) :=  v_per
    /// Ref(v_par) := -v_par
    /// ```
    /// by definition. This means that the reflection on vectors is defined by
    /// ```text
    /// Ref(v) := Ref(v_per + v_par)
    ///        := Ref(v_per) + Ref(v_par)
    ///        := Ref(v_per) - v_par
    ///        == v_per - v_par
    ///        == v - v_par - v_par
    ///        == v - 2 * v_par
    ///        == v - (2 * dot(v, n)) * n
    /// ```
    /// And reflection on points is defined by
    /// ```text
    /// Ref(P) := Ref(Q + (P - Q))
    ///        := Q + Ref(P - Q)
    ///        == Q + [(P - Q) - 2 * dot(P - Q, n) * n]
    ///        == P - 2 * dot(P - Q, n) * n
    ///        == I * P - (2 * dot(P, n)) * n + (2 * dot(Q, n)) * n
    ///        == [I - 2 * outer(n, n)] * P + (2 * dot(Q, n)) * n
    /// ```
    /// and the corresponding affine matrix has the form
    /// ```text
    /// M := | I - 2 * outer(n, n)   2 * dot(Q, n) * n |
    ///      | 0^T                   1                 |
    /// ```
    /// geometrically. In the standard basis in two-dimensional Euclidean space, we
    /// have
    /// ```text
    ///      |  1 - 2 * n.x * n.x   -2 * n.x * n.y       2 * dot(Q, n) * n.x |
    /// M == | -2 * n.y * n.x        1 - 2 * n.y * n.y   2 * dot(Q, n) * n.y |
    ///      |  0                    0                   1                   |
    /// ```
    /// and in three-dimensional Euclidean space we have
    /// ```text
    ///      |  1 - 2 * n.x * n.x   -2 * n.x * n.y       -2 * n.x * n.z        2 * dot(Q, n) * n.x |
    /// M == | -2 * n.y * n.x        1 - 2 * n.y * n.y   -2 * n.y * n.z        2 * dot(Q, n) * n.y |
    ///      | -2 * n.z * n.x       -2 * n.z * n.y        1 - 2 * n.z * n.z    2 * dot(Q, n) * n.z |
    ///      |  0                    0                    0                    1                   |
    /// ```
    /// which correspond exactly the how the respective matrices are implemented.
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
    /// let result = reflection.inverse_apply_vector(&vector);
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
    /// let result = reflection.inverse_apply_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_apply_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        self.apply_vector(vector)
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
    /// let result = reflection.inverse_apply_point(&point);
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
    /// let result = reflection.inverse_apply_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_apply_point(&self, point: &Point<S, N>) -> Point<S, N> {
        self.apply_point(point)
    }

    /// Compute the inverse of a reflection.
    /// 
    /// The inverse of a reflection transformation is the reflection transformation
    /// itself. That is, given a reflection `r`
    /// ```text
    /// inverse(r) == r
    /// ```
    /// Every reflection transformation satisfies
    /// ```text
    /// r * r == id
    /// ```
    /// where `id` is the identity transformation. In other words, given a 
    /// point `p`
    /// ```text
    /// r * (r * p) == (r * r) * p == p
    /// ```
    /// and given a vector `v`
    /// ```text
    /// r * (r * v) == (r * r) * v == v
    /// ```
    /// 
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection2,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
    /// #     Unit,
    /// # };
    /// #
    /// let bias = Point2::new(-2_f64, 3_f64);
    /// let normal = Unit::from_value(Vector2::unit_x());
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let point = Point2::new(1_f64, 2_f64);
    /// let reflection_inv = reflection.inverse();
    /// let expected = point;
    /// let result1 = reflection_inv * (reflection * point);
    /// let result2 = reflection * (reflection * point);
    ///
    /// assert_eq!(result1, expected);
    /// assert_eq!(result2, expected);
    /// ```
    /// 
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_transform::{
    /// #     Reflection3,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3, 
    /// #     Unit,
    /// # };
    /// #
    /// let bias = Point3::new(-2_f64, 3_f64, -4_f64);
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let reflection = Reflection3::from_normal_bias(&normal, &bias);
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let reflection_inv = reflection.inverse();
    /// let expected = point;
    /// let result1 = reflection_inv * (reflection * point);
    /// let result2 = reflection * (reflection * point);
    ///
    /// assert_eq!(result1, expected);
    /// assert_eq!(result2, expected);
    /// ```
    #[inline]
    pub const fn inverse(&self) -> Self {
        *self
    }
}

impl<S, const N: usize> fmt::Display for Reflection<S, N>
where
    S: fmt::Display
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Reflection{} [normal = {}, bias = {}]", N, self.normal, self.bias)
    }
}

impl<S, const N: usize> approx::AbsDiffEq for Reflection<S, N>
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
        Vector::abs_diff_eq(&self.normal, &other.normal, epsilon)
            && Point::abs_diff_eq(&self.bias, &other.bias, epsilon)
    }
}

impl<S, const N: usize> approx::RelativeEq for Reflection<S, N>
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        Vector::relative_eq(&self.normal, &other.normal, epsilon, max_relative)
           && Point::relative_eq(&self.bias, &other.bias, epsilon, max_relative)
    }
}

impl<S, const N: usize> approx::UlpsEq for Reflection<S, N>
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        Vector::ulps_eq(&self.normal, &other.normal, epsilon, max_ulps)
            && Point::ulps_eq(&self.bias, &other.bias, epsilon, max_ulps)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for Reflection<S, N>
where 
    S: SimdScalarFloat
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Point<S, N>> for Reflection<S, N>
where 
    S: SimdScalarFloat
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for &Reflection<S, N>
where 
    S: SimdScalarFloat
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Point<S, N>> for &'b Reflection<S, N>
where 
    S: SimdScalarFloat
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &'a Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Vector<S, N>> for Reflection<S, N>
where 
    S: SimdScalarFloat
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Vector<S, N>> for Reflection<S, N>
where 
    S: SimdScalarFloat
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: &Vector<S, N>) -> Self::Output {
        self.apply_vector(other)
    }
}

impl<S, const N: usize> ops::Mul<Vector<S, N>> for &Reflection<S, N> 
where 
    S: SimdScalarFloat
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Vector<S, N>> for &'b Reflection<S, N>
where 
    S: SimdScalarFloat
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
    /// Convert a reflection to an affine matrix.
    /// 
    /// # Example
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
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix3x3<S> {
        let normal = Unit::from_value_unchecked(self.normal);

        Matrix3x3::from_affine_reflection(&normal, &self.bias)
    }

    /// Convert a reflection to a generic transformation.
    /// 
    /// # Example
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
    #[inline]
    pub fn to_transform(&self) -> Transform2<S> {
        Transform::from_matrix_unchecked(self.to_affine_matrix())
    }
}

impl<S> From<Reflection2<S>> for Matrix3x3<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn from(transformation: Reflection2<S>) -> Matrix3x3<S> {
        transformation.to_affine_matrix()
    }
}

impl<S> From<&Reflection2<S>> for Matrix3x3<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn from(transformation: &Reflection2<S>) -> Matrix3x3<S> {
        transformation.to_affine_matrix()
    }
}


impl<S> Reflection3<S> 
where 
    S: SimdScalarFloat 
{
    /// Convert a reflection to an affine matrix.
    /// 
    /// # Example
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
    pub fn to_affine_matrix(&self) -> Matrix4x4<S> {
        let normal = Unit::from_value_unchecked(self.normal);

        Matrix4x4::from_affine_reflection(&normal, &self.bias)
    }

    /// Convert a reflection to a generic transformation.
    /// 
    /// # Example
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
    pub fn to_transform(&self) -> Transform3<S> {
        Transform::from_matrix_unchecked(self.to_affine_matrix())
    }
}

impl<S> From<Reflection3<S>> for Matrix4x4<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn from(transformation: Reflection3<S>) -> Matrix4x4<S> {
        transformation.to_affine_matrix()
    }
}

impl<S> From<&Reflection3<S>> for Matrix4x4<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn from(transformation: &Reflection3<S>) -> Matrix4x4<S> {
        transformation.to_affine_matrix()
    }
}

