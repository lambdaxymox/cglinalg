use crate::transform::{
    Transform,
    Transform2,
    Transform3,
};
use cglinalg_core::{
    Matrix3x3,
    Matrix4x4,
    Point,
    Unit,
    Vector,
};
use cglinalg_numeric::SimdScalarFloat;

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
    S: SimdScalarFloat,
{
    /// Construct a new reflection transformation from the vector normal to the
    /// plane of reflection.
    ///
    /// # Example (Two Dimensions)
    ///
    /// A reflection about the **y-axis** using the origin as the bias.
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Reflection2;
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
    /// // We can just as well use the opposite normal.
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
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Reflection3;
    /// #
    /// let normal = Unit::from_value(Vector3::new(1_f64, 1_f64, 1_f64));
    /// let bias = Point3::origin();
    /// let reflection = Reflection3::from_normal_bias(&normal, &bias);
    /// let vector = Vector3::new(-5_f64, 7_f64, -3_f64);
    /// let expected = Vector3::new(-13_f64 / 3_f64, 23_f64 / 3_f64, -7_f64 / 3_f64);
    /// let result = reflection.apply_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
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
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Reflection2;
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
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Reflection3;
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
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Reflection2;
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
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Reflection3;
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
    S: SimdScalarFloat,
{
    /// Reflect a vector across the plane described by the reflection
    /// transformation.
    ///
    /// # Discussion
    ///
    /// The **reflection** of a point is defined as follows. Let `M` be the plane of
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
    /// and reflection on points is defined by
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
    /// M := [ I - 2 * outer(n, n)   2 * dot(Q, n) * n ]
    ///      [ 0^T                   1                 ]
    /// ```
    /// geometrically. In the standard basis in two-dimensional Euclidean space, we
    /// have
    /// ```text
    ///      [  1 - 2 * n.x * n.x   -2 * n.x * n.y       2 * dot(Q, n) * n.x ]
    /// M == [ -2 * n.y * n.x        1 - 2 * n.y * n.y   2 * dot(Q, n) * n.y ]
    ///      [  0                    0                   1                   ]
    /// ```
    /// and in three-dimensional Euclidean space we have
    /// ```text
    ///      [  1 - 2 * n.x * n.x   -2 * n.x * n.y       -2 * n.x * n.z        2 * dot(Q, n) * n.x ]
    /// M == [ -2 * n.y * n.x        1 - 2 * n.y * n.y   -2 * n.y * n.z        2 * dot(Q, n) * n.y ]
    ///      [ -2 * n.z * n.x       -2 * n.z * n.y        1 - 2 * n.z * n.z    2 * dot(Q, n) * n.z ]
    ///      [  0                    0                    0                    1                   ]
    /// ```
    /// which correspond exactly to how the respective matrices are implemented.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Reflection2;
    /// #
    /// let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
    ///     -1_f64 / 2_f64,
    ///      1_f64,
    /// ));
    /// let bias = Point2::new(0_f64, 1_f64);
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let vector = Vector2::new(1_f64, 1_f64);
    /// let expected = Vector2::new(7_f64 / 5_f64, 1_f64 / 5_f64);
    /// let result = reflection.apply_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Reflection3;
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
    /// The **reflection** of a point is defined as follows. Let `M` be the plane of
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
    /// and reflection on points is defined by
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
    /// M := [ I - 2 * outer(n, n)   2 * dot(Q, n) * n ]
    ///      [ 0^T                   1                 ]
    /// ```
    /// geometrically. In the standard basis in two-dimensional Euclidean space, we
    /// have
    /// ```text
    ///      [  1 - 2 * n.x * n.x   -2 * n.x * n.y       2 * dot(Q, n) * n.x ]
    /// M == [ -2 * n.y * n.x        1 - 2 * n.y * n.y   2 * dot(Q, n) * n.y ]
    ///      [  0                    0                   1                   ]
    /// ```
    /// and in three-dimensional Euclidean space we have
    /// ```text
    ///      [  1 - 2 * n.x * n.x   -2 * n.x * n.y       -2 * n.x * n.z        2 * dot(Q, n) * n.x ]
    /// M == [ -2 * n.y * n.x        1 - 2 * n.y * n.y   -2 * n.y * n.z        2 * dot(Q, n) * n.y ]
    ///      [ -2 * n.z * n.x       -2 * n.z * n.y        1 - 2 * n.z * n.z    2 * dot(Q, n) * n.z ]
    ///      [  0                    0                    0                    1                   ]
    /// ```
    /// which correspond exactly to how the respective matrices are implemented.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Reflection2;
    /// #
    /// let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
    ///     -1_f64 / 2_f64,
    ///      1_f64,
    /// ));
    /// let bias = Point2::new(0_f64, 1_f64);
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let point = Point2::new(1_f64, 1_f64);
    /// let expected = Point2::new(3_f64 / 5_f64, 9_f64 / 5_f64);
    /// let result = reflection.apply_point(&point);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Reflection3;
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
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Reflection2;
    /// #
    /// let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
    ///     -1_f64 / 2_f64,
    ///      1_f64,
    /// ));
    /// let bias = Point2::new(0_f64, 1_f64);
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let vector = Vector2::new(1_f64, 1_f64);
    /// let expected = Vector2::new(7_f64 / 5_f64, 1_f64 / 5_f64);
    /// let result = reflection.inverse_apply_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Reflection3;
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
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Reflection2;
    /// #
    /// let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
    ///     -1_f64 / 2_f64,
    ///      1_f64,
    /// ));
    /// let bias = Point2::new(0_f64, 1_f64);
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let point = Point2::new(1_f64, 1_f64);
    /// let expected = Point2::new(3_f64 / 5_f64, 9_f64 / 5_f64);
    /// let result = reflection.inverse_apply_point(&point);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Reflection3;
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
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Reflection2;
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
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Reflection3;
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
    S: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Reflection{} [normal = {}, bias = {}]", N, self.normal, self.bias)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Point<S, N>> for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for &Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.apply_point(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Point<S, N>> for &'b Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &'a Point<S, N>) -> Self::Output {
        self.apply_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Vector<S, N>> for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Vector<S, N>> for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: &Vector<S, N>) -> Self::Output {
        self.apply_vector(other)
    }
}

impl<S, const N: usize> ops::Mul<Vector<S, N>> for &Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: Vector<S, N>) -> Self::Output {
        self.apply_vector(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Vector<S, N>> for &'b Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type Output = Vector<S, N>;

    #[inline]
    fn mul(self, other: &'a Vector<S, N>) -> Self::Output {
        self.apply_vector(other)
    }
}

impl<S> Reflection2<S>
where
    S: SimdScalarFloat,
{
    /// Convert a reflection to an affine matrix.
    ///
    /// # Example
    ///
    /// A reflection about the plane `y == 2 * x`.
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::Reflection2;
    /// #
    /// let normal = Unit::from_value(Vector2::new(-2_f64, 1_f64));
    /// let bias = Point2::origin();
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let expected = Matrix3x3::new(
    ///     -3_f64 / 5_f64, 4_f64 / 5_f64, 0_f64,
    ///      4_f64 / 5_f64, 3_f64 / 5_f64, 0_f64,
    ///      0_f64,         0_f64,         1_f64,
    /// );
    /// let result = reflection.to_affine_matrix();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
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
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Reflection2,
    /// #     Transform2,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector2::new(-2_f64, 1_f64));
    /// let bias = Point2::origin();
    /// let reflection = Reflection2::from_normal_bias(&normal, &bias);
    /// let expected = Transform2::from_matrix_unchecked(Matrix3x3::new(
    ///     -3_f64 / 5_f64, 4_f64 / 5_f64, 0_f64,
    ///      4_f64 / 5_f64, 3_f64 / 5_f64, 0_f64,
    ///      0_f64,         0_f64,         1_f64,
    /// ));
    /// let result = reflection.to_transform();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn to_transform(&self) -> Transform2<S> {
        Transform::from_matrix_unchecked(self.to_affine_matrix())
    }
}

impl<S> From<Reflection2<S>> for Matrix3x3<S>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn from(transformation: Reflection2<S>) -> Matrix3x3<S> {
        transformation.to_affine_matrix()
    }
}

impl<S> From<&Reflection2<S>> for Matrix3x3<S>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn from(transformation: &Reflection2<S>) -> Matrix3x3<S> {
        transformation.to_affine_matrix()
    }
}

impl<S> Reflection3<S>
where
    S: SimdScalarFloat,
{
    /// Convert a reflection to an affine matrix.
    ///
    /// # Example
    ///
    /// A reflection about the plane `x + y == -z`.
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Reflection3;
    /// #
    /// let normal = Unit::from_value(Vector3::new(1_f64, 1_f64, 1_f64));
    /// let bias = Point3::origin();
    /// let reflection = Reflection3::from_normal_bias(&normal, &bias);
    /// let expected = Matrix4x4::new(
    ///      1_f64 / 3_f64, -2_f64 / 3_f64, -2_f64 / 3_f64, 0_f64,
    ///     -2_f64 / 3_f64,  1_f64 / 3_f64, -2_f64 / 3_f64, 0_f64,
    ///     -2_f64 / 3_f64, -2_f64 / 3_f64,  1_f64 / 3_f64, 0_f64,
    ///      0_f64,          0_f64,          0_f64,         1_f64,
    /// );
    /// let result = reflection.to_affine_matrix();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
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
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Reflection3,
    /// #     Transform3,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector3::new(1_f64, 1_f64, 1_f64));
    /// let bias = Point3::origin();
    /// let reflection = Reflection3::from_normal_bias(&normal, &bias);
    /// let expected = Transform3::from_matrix_unchecked(Matrix4x4::new(
    ///      1_f64 / 3_f64, -2_f64 / 3_f64, -2_f64 / 3_f64, 0_f64,
    ///     -2_f64 / 3_f64,  1_f64 / 3_f64, -2_f64 / 3_f64, 0_f64,
    ///     -2_f64 / 3_f64, -2_f64 / 3_f64,  1_f64 / 3_f64, 0_f64,
    ///      0_f64,          0_f64,          0_f64,         1_f64,
    /// ));
    /// let result = reflection.to_transform();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn to_transform(&self) -> Transform3<S> {
        Transform::from_matrix_unchecked(self.to_affine_matrix())
    }
}

impl<S> From<Reflection3<S>> for Matrix4x4<S>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn from(transformation: Reflection3<S>) -> Matrix4x4<S> {
        transformation.to_affine_matrix()
    }
}

impl<S> From<&Reflection3<S>> for Matrix4x4<S>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn from(transformation: &Reflection3<S>) -> Matrix4x4<S> {
        transformation.to_affine_matrix()
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReflectionTol<S, const N: usize> {
    normal: Vector<S, N>,
    bias: Vector<S, N>,
}

impl<S, const N: usize> ReflectionTol<S, N> {
    #[inline]
    pub const fn from_parts(normal: Vector<S, N>, bias: Vector<S, N>) -> Self {
        Self { normal, bias }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReflectionDiff<S, const N: usize> {
    normal: Vector<S, N>,
    bias: Vector<S, N>,
}

impl<S, const N: usize> ReflectionDiff<S, N> {
    #[inline]
    const fn from_parts(normal: Vector<S, N>, bias: Vector<S, N>) -> Self {
        Self { normal, bias }
    }
}

impl<S, const N: usize> approx_cmp::AbsDiffEq for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type Tolerance = ReflectionTol<<S as approx_cmp::AbsDiffEq>::Tolerance, N>;

    #[inline]
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> bool {
        approx_cmp::AbsDiffEq::abs_diff_eq(&self.normal, &other.normal, &max_abs_diff.normal)
            && approx_cmp::AbsDiffEq::abs_diff_eq(&self.bias, &other.bias, &max_abs_diff.bias)
    }
}

impl<S, const N: usize> approx_cmp::AbsDiffAllEq for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::AbsDiffAllEq>::AllTolerance;

    #[inline]
    fn abs_diff_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> bool {
        approx_cmp::AbsDiffAllEq::abs_diff_all_eq(&self.normal, &other.normal, max_abs_diff)
            && approx_cmp::AbsDiffAllEq::abs_diff_all_eq(&self.bias, &other.bias, max_abs_diff)
    }
}

impl<S, const N: usize> approx_cmp::AssertAbsDiffEq for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = ReflectionDiff<<S as approx_cmp::AssertAbsDiffEq>::DebugAbsDiff, N>;
    type DebugTolerance = ReflectionTol<<S as approx_cmp::AssertAbsDiffEq>::DebugTolerance, N>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let normal = approx_cmp::AssertAbsDiffEq::debug_abs_diff(&self.normal, &other.normal);
        let bias = approx_cmp::AssertAbsDiffEq::debug_abs_diff(&self.bias, &other.bias);

        ReflectionDiff::from_parts(normal, bias)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let normal = approx_cmp::AssertAbsDiffEq::debug_abs_diff_tolerance(&self.normal, &other.normal, &max_abs_diff.normal);
        let bias = approx_cmp::AssertAbsDiffEq::debug_abs_diff_tolerance(&self.bias, &other.bias, &max_abs_diff.bias);

        ReflectionTol::from_parts(normal, bias)
    }
}

impl<S, const N: usize> approx_cmp::AssertAbsDiffAllEq for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = ReflectionTol<<S as approx_cmp::AssertAbsDiffAllEq>::AllDebugTolerance, N>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let normal = approx_cmp::AssertAbsDiffAllEq::debug_abs_diff_all_tolerance(&self.normal, &other.normal, max_abs_diff);
        let bias = approx_cmp::AssertAbsDiffAllEq::debug_abs_diff_all_tolerance(&self.bias, &other.bias, max_abs_diff);

        ReflectionTol::from_parts(normal, bias)
    }
}

impl<S, const N: usize> approx_cmp::RelativeEq for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type Tolerance = ReflectionTol<<S as approx_cmp::RelativeEq>::Tolerance, N>;

    #[inline]
    fn relative_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_relative: &Self::Tolerance) -> bool {
        approx_cmp::RelativeEq::relative_eq(&self.normal, &other.normal, &max_abs_diff.normal, &max_relative.normal)
            && approx_cmp::RelativeEq::relative_eq(&self.bias, &other.bias, &max_abs_diff.bias, &max_relative.bias)
    }
}

impl<S, const N: usize> approx_cmp::RelativeAllEq for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::RelativeAllEq>::AllTolerance;

    #[inline]
    fn relative_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_relative: &Self::AllTolerance) -> bool {
        approx_cmp::RelativeAllEq::relative_all_eq(&self.normal, &other.normal, max_abs_diff, max_relative)
            && approx_cmp::RelativeAllEq::relative_all_eq(&self.bias, &other.bias, max_abs_diff, max_relative)
    }
}

impl<S, const N: usize> approx_cmp::AssertRelativeEq for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = ReflectionDiff<<S as approx_cmp::AssertRelativeEq>::DebugAbsDiff, N>;
    type DebugTolerance = ReflectionTol<<S as approx_cmp::AssertRelativeEq>::DebugTolerance, N>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let normal = approx_cmp::AssertRelativeEq::debug_abs_diff(&self.normal, &other.normal);
        let bias = approx_cmp::AssertRelativeEq::debug_abs_diff(&self.bias, &other.bias);

        ReflectionDiff::from_parts(normal, bias)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let normal = approx_cmp::AssertRelativeEq::debug_abs_diff_tolerance(&self.normal, &other.normal, &max_abs_diff.normal);
        let bias = approx_cmp::AssertRelativeEq::debug_abs_diff_tolerance(&self.bias, &other.bias, &max_abs_diff.bias);

        ReflectionTol::from_parts(normal, bias)
    }

    #[inline]
    fn debug_relative_tolerance(&self, other: &Self, max_relative: &Self::Tolerance) -> Self::DebugTolerance {
        let normal = approx_cmp::AssertRelativeEq::debug_relative_tolerance(&self.normal, &other.normal, &max_relative.normal);
        let bias = approx_cmp::AssertRelativeEq::debug_relative_tolerance(&self.bias, &other.bias, &max_relative.bias);

        ReflectionTol::from_parts(normal, bias)
    }
}

impl<S, const N: usize> approx_cmp::AssertRelativeAllEq for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = ReflectionTol<<S as approx_cmp::AssertRelativeAllEq>::AllDebugTolerance, N>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let normal = approx_cmp::AssertRelativeAllEq::debug_abs_diff_all_tolerance(&self.normal, &other.normal, max_abs_diff);
        let bias = approx_cmp::AssertRelativeAllEq::debug_abs_diff_all_tolerance(&self.bias, &other.bias, max_abs_diff);

        ReflectionTol::from_parts(normal, bias)
    }

    #[inline]
    fn debug_relative_all_tolerance(&self, other: &Self, max_relative: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let normal = approx_cmp::AssertRelativeAllEq::debug_relative_all_tolerance(&self.normal, &other.normal, max_relative);
        let bias = approx_cmp::AssertRelativeAllEq::debug_relative_all_tolerance(&self.bias, &other.bias, max_relative);

        ReflectionTol::from_parts(normal, bias)
    }
}

impl<S, const N: usize> approx_cmp::UlpsEq for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type Tolerance = ReflectionTol<<S as approx_cmp::UlpsEq>::Tolerance, N>;
    type UlpsTolerance = ReflectionTol<<S as approx_cmp::UlpsEq>::UlpsTolerance, N>;

    fn ulps_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_ulps: &Self::UlpsTolerance) -> bool {
        approx_cmp::UlpsEq::ulps_eq(&self.normal, &other.normal, &max_abs_diff.normal, &max_ulps.normal)
            && approx_cmp::UlpsEq::ulps_eq(&self.bias, &other.bias, &max_abs_diff.bias, &max_ulps.bias)
    }
}

impl<S, const N: usize> approx_cmp::UlpsAllEq for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::UlpsAllEq>::AllTolerance;
    type AllUlpsTolerance = <S as approx_cmp::UlpsAllEq>::AllUlpsTolerance;

    #[inline]
    fn ulps_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_ulps: &Self::AllUlpsTolerance) -> bool {
        approx_cmp::UlpsAllEq::ulps_all_eq(&self.normal, &other.normal, max_abs_diff, max_ulps)
            && approx_cmp::UlpsAllEq::ulps_all_eq(&self.bias, &other.bias, max_abs_diff, max_ulps)
    }
}

impl<S, const N: usize> approx_cmp::AssertUlpsEq for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = ReflectionDiff<<S as approx_cmp::AssertUlpsEq>::DebugAbsDiff, N>;
    type DebugUlpsDiff = ReflectionDiff<<S as approx_cmp::AssertUlpsEq>::DebugUlpsDiff, N>;
    type DebugTolerance = ReflectionTol<<S as approx_cmp::AssertUlpsEq>::DebugTolerance, N>;
    type DebugUlpsTolerance = ReflectionTol<<S as approx_cmp::AssertUlpsEq>::DebugUlpsTolerance, N>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let normal = approx_cmp::AssertUlpsEq::debug_abs_diff(&self.normal, &other.normal);
        let bias = approx_cmp::AssertUlpsEq::debug_abs_diff(&self.bias, &other.bias);

        ReflectionDiff::from_parts(normal, bias)
    }

    #[inline]
    fn debug_ulps_diff(&self, other: &Self) -> Self::DebugUlpsDiff {
        let normal = approx_cmp::AssertUlpsEq::debug_ulps_diff(&self.normal, &other.normal);
        let bias = approx_cmp::AssertUlpsEq::debug_ulps_diff(&self.bias, &other.bias);

        ReflectionDiff::from_parts(normal, bias)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let normal = approx_cmp::AssertUlpsEq::debug_abs_diff_tolerance(&self.normal, &other.normal, &max_abs_diff.normal);
        let bias = approx_cmp::AssertUlpsEq::debug_abs_diff_tolerance(&self.bias, &other.bias, &max_abs_diff.bias);

        ReflectionTol::from_parts(normal, bias)
    }

    #[inline]
    fn debug_ulps_tolerance(&self, other: &Self, max_ulps: &Self::UlpsTolerance) -> Self::DebugUlpsTolerance {
        let normal = approx_cmp::AssertUlpsEq::debug_ulps_tolerance(&self.normal, &other.normal, &max_ulps.normal);
        let bias = approx_cmp::AssertUlpsEq::debug_ulps_tolerance(&self.bias, &other.bias, &max_ulps.bias);

        ReflectionTol::from_parts(normal, bias)
    }
}

impl<S, const N: usize> approx_cmp::AssertUlpsAllEq for Reflection<S, N>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = ReflectionTol<<S as approx_cmp::AssertUlpsAllEq>::AllDebugTolerance, N>;
    type AllDebugUlpsTolerance = ReflectionTol<<S as approx_cmp::AssertUlpsAllEq>::AllDebugUlpsTolerance, N>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let normal = approx_cmp::AssertUlpsAllEq::debug_abs_diff_all_tolerance(&self.normal, &other.normal, max_abs_diff);
        let bias = approx_cmp::AssertUlpsAllEq::debug_abs_diff_all_tolerance(&self.bias, &other.bias, max_abs_diff);

        ReflectionTol::from_parts(normal, bias)
    }

    #[inline]
    fn debug_ulps_all_tolerance(&self, other: &Self, max_ulps: &Self::AllUlpsTolerance) -> Self::AllDebugUlpsTolerance {
        let normal = approx_cmp::AssertUlpsAllEq::debug_ulps_all_tolerance(&self.normal, &other.normal, max_ulps);
        let bias = approx_cmp::AssertUlpsAllEq::debug_ulps_all_tolerance(&self.bias, &other.bias, max_ulps);

        ReflectionTol::from_parts(normal, bias)
    }
}
