use cglinalg_numeric::{
    SimdScalar,
    SimdScalarFloat,
};
use cglinalg_core::{
    Const,
    ShapeConstraint,
    DimAdd,
    DimSub,
    Matrix,
    Vector,
    Vector2,
    Vector3,
    Point,
};
use crate::transform::{
    Transform,
};

use core::fmt;
use core::ops;


/// A two-dimensional scaling transformation.
pub type Scale2<S> = Scale<S, 2>;

/// A three-dimensional scaling transformation.
pub type Scale3<S> = Scale<S, 3>;


/// The scale transformation which supports nonuniform scaling.
///
/// A scale transformation is a linear map that scales each component of a 
/// vector by a specified amount. Let `s` be a vector of numbers. Let `S` be 
/// a scale transformation that  scales a vector `v` by an amount `s[i]` on 
/// component `i` of `v`. The scale transformation `S` acts on a vector `v` 
/// as follows
/// ```text
/// forall i in 0..N. (Sv)[i] := si * v[i]
/// ```
/// where `N` is the dimensionality of the vector `v`. In particular, in
/// Euclidean space, the scale transformation `S` acts as a diagonal matrix 
/// where
/// ```text
/// forall i in 0..N. S[i][i] := s[i]
/// ```
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Scale<S, const N: usize> {
    vector: Vector<S, N>,
}

impl<S, const N: usize> Scale<S, N> 
where 
    S: SimdScalar 
{
    /// Construct a three-dimensional scale transformation from a nonuniform scale 
    /// across coordinates.
    #[inline]
    pub const fn from_nonuniform_scale(vector: Vector<S, N>) -> Self {
        Self { vector, }
    }

    /// Construct a three-dimensional scale transformation from a uniform scale 
    /// factor.
    #[inline]
    pub const fn from_scale(scale: S) -> Self {
        Self {
            vector: Vector::from_fill(scale),
        }
    }

    /// Apply a scale transformation to a vector.
    /// 
    /// # Examples
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Scale2,
    /// # };
    /// #
    /// let scale_x = 2_f64;
    /// let scale_y = 3_f64;
    /// let scale_vector = Vector2::new(scale_x, scale_y);
    /// let scale = Scale2::from_nonuniform_scale(scale_vector);
    /// let vector = Vector2::new(1_f64, 1_f64);
    /// let expected = Vector2::new(2_f64, 3_f64);
    /// let result = scale.scale_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Scale3,
    /// # };
    /// #
    /// let scale_x = 2_f64;
    /// let scale_y = 3_f64;
    /// let scale_z = 4_f64;
    /// let scale_vector = Vector3::new(scale_x, scale_y, scale_z);
    /// let scale = Scale3::from_nonuniform_scale(scale_vector);
    /// let vector = Vector3::new(1_f64, 1_f64, 1_f64);
    /// let expected = Vector3::new(2_f64, 3_f64, 4_f64);
    /// let result = scale.scale_vector(&vector);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn scale_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Vector::zero();
        for i in 0..N {
            result[i] = self.vector[i] * vector[i];
        }

        result
    }

    /// Apply a scale transformation operation to a point.
    ///
    /// # Examples
    ///
    /// An example in two dimensions. 
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Scale2,
    /// # };
    /// #
    /// let scale_x = 2_f64;
    /// let scale_y = 3_f64;
    /// let scale_vector = Vector2::new(scale_x, scale_y);
    /// let scale = Scale2::from_nonuniform_scale(scale_vector);
    /// let point = Point2::new(1_f64, 1_f64);
    /// let expected = Point2::new(2_f64, 3_f64);
    /// let result = scale.scale_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Scale3,
    /// # };
    /// #
    /// let scale_x = 2_f64;
    /// let scale_y = 3_f64;
    /// let scale_z = 4_f64;
    /// let scale_vector = Vector3::new(scale_x, scale_y, scale_z);
    /// let scale = Scale3::from_nonuniform_scale(scale_vector);
    /// let point = Point3::new(1_f64, 1_f64, 1_f64);
    /// let expected = Point3::new(2_f64, 3_f64, 4_f64);
    /// let result = scale.scale_point(&point);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn scale_point(&self, point: &Point<S, N>) -> Point<S, N> {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Point::origin();
        for i in 0..N {
            result[i] = self.vector[i] * point[i];
        }

        result
    }

    /// Construct the identity scaling transformation. 
    ///
    /// The identity is the scale transform with a scale factor of `1` for 
    /// each component.
    ///
    /// # Examples
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,  
    /// # };
    /// # use cglinalg_transform::{
    /// #     Scale2,
    /// # };
    /// #
    /// let scale = Scale2::identity();
    /// let point = Point2::new(1_f64, 2_f64);
    /// 
    /// assert_eq!(scale * point, point);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,  
    /// # };
    /// # use cglinalg_transform::{
    /// #     Scale3,
    /// # };
    /// #
    /// let scale = Scale3::identity();
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// 
    /// assert_eq!(scale * point, point);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self::from_scale(S::one())
    }

    /// Convert a scaling transformation into a vector with the scaling factors
    /// in each component.
    #[inline]
    pub const fn to_vector(&self) -> Vector<S, N> {
        self.vector
    }
}

impl<S, const N: usize, const NPLUS1: usize> Scale<S, N>
where
    S: SimdScalar,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>
{
    /// Convert a scale transformation into a generic transformation.
    #[inline]
    pub fn to_transform(&self) -> Transform<S, N, NPLUS1> {
        Transform::from_specialized(self)
    }
}

impl<S, const N: usize> Scale<S, N> 
where 
    S: SimdScalarFloat 
{
    /// Construct a scale transformation that scales each coordinate by the 
    /// reciprocal of the scaling factors of the scale operator `self`.
    /// 
    /// # Example
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Scale2, 
    /// # };
    /// #
    /// let scale_x = 2_f64;
    /// let scale_y = 3_f64;
    /// let scale_vector = Vector2::new(scale_x, scale_y);
    /// let scale = Scale2::from_nonuniform_scale(scale_vector);
    /// let expected = Scale2::from_nonuniform_scale(Vector2::new(
    ///     1_f64 / scale_x, 
    ///     1_f64 / scale_y
    /// ));
    /// let result = scale.inverse();
    ///
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #      Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Scale3, 
    /// # };
    /// #
    /// let scale_x = 2_f64;
    /// let scale_y = 3_f64;
    /// let scale_z = 4_f64;
    /// let scale_vector = Vector3::new(scale_x, scale_y, scale_z);
    /// let scale = Scale3::from_nonuniform_scale(scale_vector);
    /// let expected = Scale3::from_nonuniform_scale(Vector3::new(
    ///     1_f64 / scale_x,
    ///     1_f64 / scale_y,
    ///     1_f64 / scale_z,
    /// ));
    /// let result = scale.inverse();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse(&self) -> Self {
        let mut vector = Vector::default();
        for i in 0..N {
            vector[i] = S::one() / self.vector[i];
        }

        Self { vector }
    }

    /// Apply the inverse transformation of the scale transformation to a vector.
    ///
    /// Construct a scale transformation that scales each coordinate by the 
    /// reciprocal of the scaling factors of the scale operator `self`.
    /// 
    /// # Example
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Scale2,
    /// # };
    /// #
    /// let scale_x = 2_f64;
    /// let scale_y = 3_f64;
    /// let scale_vector = Vector2::new(scale_x, scale_y);
    /// let scale = Scale2::from_nonuniform_scale(scale_vector);
    /// let vector = Vector2::new(2_f64, 3_f64);
    /// let expected = Vector2::new(1_f64, 1_f64);
    /// let result = scale.inverse_scale_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Scale3,
    /// # };
    /// #
    /// let scale_x = 2_f64;
    /// let scale_y = 3_f64;
    /// let scale_z = 4_f64;
    /// let scale_vector = Vector3::new(scale_x, scale_y, scale_z);
    /// let scale = Scale3::from_nonuniform_scale(scale_vector);
    /// let vector = Vector3::new(2_f64, 3_f64, 4_f64);
    /// let expected = Vector3::new(1_f64, 1_f64, 1_f64);
    /// let result = scale.inverse_scale_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_scale_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Vector::default();
        for i in 0..N {
            result[i] = vector[i] / self.vector[i];
        }

        result
    }

    /// Apply the inverse transformation of the scale transformation to a point.
    ///
    /// Construct a scale transformation that scales each coordinate by the 
    /// reciprocal of the scaling factors of the scale operator `self`.
    /// 
    /// # Example
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Scale2,
    /// # };
    /// #
    /// let scale_x = 2_f64;
    /// let scale_y = 3_f64;
    /// let scale_vector = Vector2::new(scale_x, scale_y);
    /// let scale = Scale2::from_nonuniform_scale(scale_vector);
    /// let point = Point2::new(2_f64, 3_f64);
    /// let expected = Point2::new(1_f64, 1_f64);
    /// let result = scale.inverse_scale_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Scale3,
    /// # };
    /// #
    /// let scale_x = 2_f64;
    /// let scale_y = 3_f64;
    /// let scale_z = 4_f64;
    /// let scale_vector = Vector3::new(scale_x, scale_y, scale_z);
    /// let scale = Scale3::from_nonuniform_scale(scale_vector);
    /// let point = Point3::new(2_f64, 3_f64, 4_f64);
    /// let expected = Point3::new(1_f64, 1_f64, 1_f64);
    /// let result = scale.inverse_scale_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_scale_point(&self, point: &Point<S, N>) -> Point<S, N> {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Point::default();
        for i in 0..N {
            result[i] = point[i] / self.vector[i];
        }

        result
    }
}

impl<S, const N: usize> fmt::Display for Scale<S, N> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Scale{} [{}]", N, self.vector)
    }
}

impl<S, const N: usize, const NPLUS1: usize> From<Scale<S, N>> for Matrix<S, NPLUS1, NPLUS1> 
where 
    S: SimdScalar,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>
{
    #[inline]
    fn from(scale: Scale<S, N>) -> Matrix<S, NPLUS1, NPLUS1> {
        Matrix::from_affine_nonuniform_scale(&scale.vector)
    }
}

impl<S, const N: usize, const NPLUS1: usize> From<&Scale<S, N>> for Matrix<S, NPLUS1, NPLUS1> 
where 
    S: SimdScalar,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>
{
    #[inline]
    fn from(scale: &Scale<S, N>) -> Matrix<S, NPLUS1, NPLUS1> {
        Matrix::from_affine_nonuniform_scale(&scale.vector)
    }
}

impl<S, const N: usize> approx::AbsDiffEq for Scale<S, N> 
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
        Vector::abs_diff_eq(&self.vector, &other.vector, epsilon)
    }
}

impl<S, const N: usize> approx::RelativeEq for Scale<S, N> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Vector::relative_eq(&self.vector, &other.vector, epsilon, max_relative)
    }
}

impl<S, const N: usize> approx::UlpsEq for Scale<S, N> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Vector::ulps_eq(&self.vector, &other.vector, epsilon, max_ulps)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for Scale<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.scale_point(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Point<S, N>> for Scale<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &Point<S, N>) -> Self::Output {
        self.scale_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for &Scale<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.scale_point(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Point<S, N>> for &'b Scale<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &'a Point<S, N>) -> Self::Output {
        self.scale_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Scale<S, N>> for Scale<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Scale<S, N>;

    #[inline]
    fn mul(self, other: Scale<S, N>) -> Self::Output {
        let vector = self.vector.component_mul(&other.vector);
        
        Scale::from_nonuniform_scale(vector)
    }
}

impl<S, const N: usize> ops::Mul<&Scale<S, N>> for Scale<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Scale<S, N>;

    #[inline]
    fn mul(self, other: &Scale<S, N>) -> Self::Output {
        let vector = self.vector.component_mul(&other.vector);
        
        Scale::from_nonuniform_scale(vector)
    }
}

impl<S, const N: usize> ops::Mul<Scale<S, N>> for &Scale<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Scale<S, N>;

    #[inline]
    fn mul(self, other: Scale<S, N>) -> Self::Output {
        let vector = self.vector.component_mul(&other.vector);
        
        Scale::from_nonuniform_scale(vector)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Scale<S, N>> for &'b Scale<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Scale<S, N>;

    #[inline]
    fn mul(self, other: &'a Scale<S, N>) -> Self::Output {
        let vector = self.vector.component_mul(&other.vector);
        
        Scale::from_nonuniform_scale(vector)
    }
}


impl<S> Scale2<S>
where
    S: SimdScalar
{
    /// Construct a scale transformation from the components of the scale transformation.
    #[inline]
    pub const fn new(scale_x: S, scale_y: S) -> Self {
        Self { 
            vector: Vector2::new(scale_x, scale_y),
        }
    }
}

impl<S> Scale3<S>
where
    S: SimdScalar
{
    /// Construct a scale transformation from the components of the scale transformation.
    #[inline]
    pub const fn new(scale_x: S, scale_y: S, scale_z: S) -> Self {
        Self { 
            vector: Vector3::new(scale_x, scale_y, scale_z),
        }
    }
}

