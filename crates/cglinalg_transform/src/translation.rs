use cglinalg_numeric::{
    SimdScalarSigned,
    SimdScalarFloat,
};
use cglinalg_core::{
    Const,
    ShapeConstraint,
    DimAdd,
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


/// A two-dimensional translation in homogeneous coordinates.
pub type Translation2<S> = Translation<S, 2>;

/// A three-dimensional translation in homogeneous coordinates.
pub type Translation3<S> = Translation<S, 3>;


/// A translation in homogeneous coordinates.
///
/// A translation is an operation that creates displacement motions. 
/// In a Euclidean setting, translations preserve differences between two points 
/// and acts as the identity on vectors. That is, a translation `T` that acts as
/// follows.
/// Let `p` be a **point** in Euclidean space. Then
/// ```text
/// T(p) := p + t
/// ```
/// where `t` is the vector corresponding to the translation `T` displacing the 
/// point `p` by an amount `t`. Let `v` be a **vector** in Euclidean space. Then
/// ```text
/// T(v) == v
/// ```
/// Because `v` is a vector and not a point, `v` describes the **difference** 
/// between two points, rather than an arbitary position in Euclidean space. Indeed, 
/// let `E^3` be three-dimensional Euclidean space with origin `O`. Let `p` and `q` be 
/// points in `E^N`, where `N` is the number of dimensions, and `v := p - q` be the 
/// difference between them. Then in homogeneous coordinates, and by the linearity of `T`
/// ```text
/// T(v) := T(p - q) 
///      == T((p - O) - (q - O)) 
///      == T(p - O) - T(q - O) 
///      == ((p - O) + t) - ((q - O) + t) 
///      == (p + (t - O)) - (q + (t - O))
///      == (p + t) - (q + t)
///      == (p - q) + (t - t) 
///      == p - q 
///      == v
/// ```
/// as desired.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Translation<S, const N: usize> {
    pub(crate) vector: Vector<S, N>,
}

impl<S, const N: usize> Translation<S, N> 
where 
    S: SimdScalarSigned 
{
    /// Construct a translation operator from a vector of displacements.
    #[inline]
    pub const fn from_vector(vector: &Vector<S, N>) -> Self {
        Self {
            vector: *vector,
        }
    }

    /// Construct a translation between two vectors.
    ///
    /// # Examples
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Point2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation2,
    /// # };
    /// #
    /// let vector1 = Vector2::new(1_f64, 2_f64);
    /// let vector2 = Vector2::new(3_f64, 4_f64);
    /// let translation = Translation2::between_vectors(&vector1, &vector2);
    /// let point = Point2::new(0_f64, 0_f64);
    /// let expected = Point2::new(2_f64, 2_f64);
    /// let result = translation.translate_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation3,
    /// # };
    /// #
    /// let vector1 = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let vector2 = Vector3::new(7_f64, 9_f64, 11_f64);
    /// let translation = Translation3::between_vectors(&vector1, &vector2);
    /// let point = Point3::new(0_f64, 0_f64, 0_f64);
    /// let expected = Point3::new(6_f64, 7_f64, 8_f64);
    /// let result = translation.translate_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn between_vectors(vector1: &Vector<S, N>, vector2: &Vector<S, N>) -> Self {
        let distance = vector2 - vector1;

        Self::from_vector(&distance)
    }

    /// Construct a translation between two points.
    ///
    /// # Examples
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation2,
    /// # };
    /// #
    /// let point1 = Point2::new(1_f64, 2_f64);
    /// let point2 = Point2::new(3_f64, 4_f64);
    /// let translation = Translation2::between_points(&point1, &point2);
    /// let point = Point2::new(0_f64, 0_f64);
    /// let expected = Point2::new(2_f64, 2_f64);
    /// let result = translation.translate_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation3,
    /// # };
    /// #
    /// let point1 = Point3::new(1_f64, 2_f64, 3_f64);
    /// let point2 = Point3::new(7_f64, 9_f64, 11_f64);
    /// let translation = Translation3::between_points(&point1, &point2);
    /// let point = Point3::new(0_f64, 0_f64, 0_f64);
    /// let expected = Point3::new(6_f64, 7_f64, 8_f64);
    /// let result = translation.translate_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn between_points(point1: &Point<S, N>, point2: &Point<S, N>) -> Self {
        let distance = point2 - point1;

        Self::from_vector(&distance)
    }

    /// Construct a translation that translates a vector or point in the opposite
    /// direction of the translation applied by `self`.
    ///
    /// If `self` is a translation of a vector by a displacement `distance`, then its
    /// inverse will be a translation by a displacement `-distance`.
    ///
    /// # Examples
    /// 
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation2,
    /// # };
    /// #
    /// let distance = Vector2::new(1_f64, 2_f64);
    /// let translation = Translation2::from_vector(&distance);
    /// let expected = Translation2::from_vector(&(-distance));
    /// let result = translation.inverse();
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
    /// #     Translation3,
    /// # };
    /// #
    /// let distance = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let translation = Translation3::from_vector(&distance);
    /// let expected = Translation3::from_vector(&(-distance));
    /// let result = translation.inverse();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse(&self) -> Self {
        Self::from_vector(&(-self.vector))
    }

    /// Mutably invert a translation in place.
    ///
    /// # Examples
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation2,
    /// # };
    /// #
    /// let mut result = Translation2::new(1_f64, 2_f64);
    /// let expected = Translation2::new(-1_f64, -2_f64);
    /// result.inverse_mut();
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
    /// #     Translation3,
    /// # };
    /// #
    /// let mut result = Translation3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Translation3::new(-1_f64, -2_f64, -3_f64);
    /// result.inverse_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.vector.neg_mut();
    }
    
    /// Apply the translation transformation to a point.
    ///
    /// # Examples
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Point2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation2,
    /// # };
    /// #
    /// let distance = Vector2::new(4_f64, 8_f64);
    /// let translation = Translation2::from_vector(&distance);
    /// let point = Point2::new(0_f64, 0_f64);
    /// let expected = Point2::new(4_f64, 8_f64);
    /// let result = translation.translate_point(&point);
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
    /// #     Translation3,
    /// # };
    /// #
    /// let distance = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let translation = Translation3::from_vector(&distance);
    /// let point = Point3::new(0_f64, 0_f64, 0_f64);
    /// let expected = Point3::new(1_f64, 2_f64, 3_f64);
    /// let result = translation.translate_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn translate_point(&self, point: &Point<S, N>) -> Point<S, N> {
        point + self.vector
    }

    /// Apply the translation transformation to a vector.
    ///
    /// # Examples
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation2,
    /// # };
    /// #
    /// let distance = Vector2::new(4_f64, 8_f64);
    /// let translation = Translation2::from_vector(&distance);
    /// let vector = Vector2::new(0_f64, 0_f64);
    /// let expected = vector;
    /// let result = translation.translate_vector(&vector);
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
    /// #     Translation3,
    /// # };
    /// #
    /// let distance = Vector3::new(100_f64, 200_f64, 300_f64);
    /// let translation = Translation3::from_vector(&distance);
    /// let vector = Vector3::new(0_f64, 0_f64, 0_f64);
    /// let expected = vector;
    /// let result = translation.translate_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn translate_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        *vector
    }

    /// Apply the inverse of the translation to a point.
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
    /// #     Translation2,
    /// # };
    /// #
    /// let distance = Vector2::new(13_f64, 30_f64);
    /// let translation = Translation2::from_vector(&distance);
    /// let translation_inv = translation.inverse();
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = translation_inv.translate_point(&point);
    /// let result = translation.inverse_translate_point(&point);
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
    /// #     Translation3,
    /// # };
    /// #
    /// let distance = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let translation = Translation3::from_vector(&distance);
    /// let point = Point3::new(0_f64, 0_f64, 0_f64);
    /// let expected = point;
    /// let translated_point = translation.translate_point(&point);
    /// let result = translation.inverse_translate_point(&translated_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse_translate_point(&self, point: &Point<S, N>) -> Point<S, N> {
        point - self.vector
    }

    /// Apply the inverse of the translation to a vector.
    ///
    /// # Examples
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation2,
    /// # };
    /// #
    /// let distance = Vector2::new(13_f64, 30_f64);
    /// let translation = Translation2::from_vector(&distance);
    /// let vector = Vector2::new(1_f64, 2_f64);
    ///
    /// assert_eq!(translation.inverse_translate_vector(&vector), vector);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation3,
    /// # };
    /// #
    /// let distance = Vector3::new(100_f64, 200_f64, 300_f64);
    /// let translation = Translation3::from_vector(&distance);
    /// let vector = Vector3::new(0_f64, 0_f64, 0_f64);
    /// let expected = vector;
    /// let result = translation.inverse_translate_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn inverse_translate_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        *vector
    }

    /// The identity transformation for translations, which displaces
    /// a vector or point zero distance.
    ///
    /// # Examples
    /// 
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation2,
    /// # };
    /// #
    /// let translation = Translation2::identity();
    /// let point = Point2::new(1_f64, 2_f64);
    /// 
    /// assert_eq!(translation.translate_point(&point), point);
    /// ```
    ///
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation3,
    /// # };
    /// #
    /// let translation = Translation3::identity();
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// 
    /// assert_eq!(translation.translate_point(&point), point);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self { 
            vector: Vector::zero(),
        }
    }

    /// Convert a translation into its equivalent shift vector.
    /// 
    /// # Examples
    /// 
    /// An example in two dimensions. 
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation2,
    /// # };
    /// #
    /// let translation = Translation2::new(1_f64, 2_f64);
    /// let shift = translation.to_vector();
    /// let point = Point2::origin();
    /// let expected = translation.translate_point(&point);
    /// let result = point + shift;
    /// 
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Translation3,
    /// # };
    /// #
    /// let translation = Translation3::new(1_f64, 2_f64, 3_f64);
    /// let shift = translation.to_vector();
    /// let point = Point3::origin();
    /// let expected = translation.translate_point(&point);
    /// let result = point + shift;
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn to_vector(&self) -> Vector<S, N> {
        self.vector
    }
}

impl<S, const N: usize, const NPLUS1: usize> Translation<S, N>
where
    S: SimdScalarSigned,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    /// Convert a translation to a generic transformation.
    #[inline]
    pub fn to_transform(&self) -> Transform<S, N, NPLUS1> {
        Transform::from_specialized(self)
    }
}


impl<S, const N: usize> AsRef<Vector<S, N>> for Translation<S, N> {
    #[inline]
    fn as_ref(&self) -> &Vector<S, N> {
        &self.vector
    }
}

impl<S, const N: usize> fmt::Display for Translation<S, N> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Translation{} {}",
            N, self.vector
        )
    }
}

impl<S, const N: usize, const NPLUS1: usize> From<Translation<S, N>> for Matrix<S, NPLUS1, NPLUS1> 
where 
    S: SimdScalarSigned,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    #[inline]
    fn from(transform: Translation<S, N>) -> Matrix<S, NPLUS1, NPLUS1> {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Matrix::identity();
        for r in 0..N {
            result[N][r] = transform.vector[r];
        }

        result
    }
}

impl<S, const N: usize, const NPLUS1: usize> From<&Translation<S, N>> for Matrix<S, NPLUS1, NPLUS1> 
where 
    S: SimdScalarSigned,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>
{
    #[inline]
    fn from(transform: &Translation<S, N>) -> Matrix<S, NPLUS1, NPLUS1> {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Matrix::identity();
        for r in 0..N {
            result[N][r] = transform.vector[r];
        }

        result
    }
}

impl<S, const N: usize> approx::AbsDiffEq for Translation<S, N> 
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

impl<S, const N: usize> approx::RelativeEq for Translation<S, N> 
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

impl<S, const N: usize> approx::UlpsEq for Translation<S, N> 
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

impl<S, const N: usize> ops::Mul<Point<S, N>> for Translation<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.translate_point(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Point<S, N>> for Translation<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &Point<S, N>) -> Self::Output {
        self.translate_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for &Translation<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.translate_point(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Point<S, N>> for &'b Translation<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &'a Point<S, N>) -> Self::Output {
        self.translate_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Translation<S, N>> for Translation<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Translation<S, N>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn mul(self, other: Translation<S, N>) -> Self::Output {
        Translation::from_vector(&(self.vector + other.vector))
    }
}

impl<S, const N: usize> ops::Mul<&Translation<S, N>> for Translation<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Translation<S, N>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn mul(self, other: &Translation<S, N>) -> Self::Output {
        Translation::from_vector(&(self.vector + other.vector))
    }
}

impl<S, const N: usize> ops::Mul<Translation<S, N>> for &Translation<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Translation<S, N>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn mul(self, other: Translation<S, N>) -> Self::Output {
        Translation::from_vector(&(self.vector + other.vector))
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Translation<S, N>> for &'b Translation<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Translation<S, N>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn mul(self, other: &'a Translation<S, N>) -> Self::Output {
        Translation::from_vector(&(self.vector + other.vector))
    }
}

impl<S> Translation2<S> 
where 
    S: SimdScalarSigned 
{
    /// Construct a translation from the components of the translation.
    #[inline]
    pub const fn new(x: S, y: S) -> Self {
        Self {
            vector: Vector2::new(x, y)
        }
    }
}

impl<S> Translation3<S> 
where 
    S: SimdScalarSigned 
{
    /// Construct a translation from the components of the translation.
    #[inline]
    pub const fn new(x: S, y: S, z: S) -> Self {
        Self {
            vector: Vector3::new(x, y, z)
        }
    }
}

