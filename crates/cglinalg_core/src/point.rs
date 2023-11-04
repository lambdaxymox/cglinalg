use crate::constraint::{
    CanContract,
    CanExtend,
    Const,
    DimAdd,
    DimSub,
    ShapeConstraint,
};
use crate::normed::Normed;
use crate::unit::Unit;
use crate::vector::{
    Vector,
    Vector1,
    Vector2,
    Vector3,
};
use crate::{
    impl_coords,
    impl_coords_deref,
};
use cglinalg_numeric::{
    SimdCast,
    SimdScalar,
    SimdScalarFloat,
    SimdScalarSigned,
};

use core::fmt;
use core::ops;


/// A point is a location in a one-dimensional Euclidean space.
pub type Point1<S> = Point<S, 1>;

/// A point is a location in a two-dimensional Euclidean space.
pub type Point2<S> = Point<S, 2>;

/// A point is a location in a three-dimensional Euclidean space.
pub type Point3<S> = Point<S, 3>;


/// A point is a location in a one-dimensional Euclidean space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Point<S, const N: usize> {
    coords: Vector<S, N>,
}

impl<S, const N: usize> Point<S, N> {
    /// Returns the length of the underlying array storing the point components.
    #[inline]
    pub const fn len(&self) -> usize {
        N
    }

    /// Tests whether the number of elements in the point is zero.
    ///
    /// Returns `true` when the point is zero-dimensional. Returns `false`
    /// otherwise.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The shape of the underlying array storing the point components.
    ///
    /// The shape is the equivalent number of columns and rows of the
    /// array as though it represents a matrix. The order of the descriptions
    /// of the shape of the array is **(rows, columns)**.
    #[inline]
    pub const fn shape(&self) -> (usize, usize) {
        (N, 1)
    }

    /// Get a pointer to the underlying array.
    #[inline]
    pub const fn as_ptr(&self) -> *const S {
        self.coords.as_ptr()
    }

    /// Get a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        self.coords.as_mut_ptr()
    }

    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        AsRef::<[S; N]>::as_ref(self)
    }
}

impl<S, const N: usize> Point<S, N>
where
    S: SimdCast + Copy,
{
    /// Cast a point from one type of scalars to another type of scalars.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,   
    /// # };
    /// #
    /// let point: Point3<i32> = Point3::new(1_i32, 2_i32, 3_i32);
    /// let expected: Option<Point3<f64>> = Some(Point3::new(1_f64, 2_f64, 3_f64));
    /// let result = point.try_cast::<f64>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn try_cast<T>(&self) -> Option<Point<T, N>>
    where
        T: SimdCast,
    {
        self.coords.try_cast::<T>().map(|coords| Point { coords })
    }
}

impl<S, const N: usize> Point<S, N>
where
    S: Copy,
{
    /// Construct a new point from a fill value.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// #
    /// let fill_value = 3_i32;
    /// let expected = Point3::new(3_i32, 3_i32, 3_i32);
    /// let result = Point3::from_fill(fill_value);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn from_fill(value: S) -> Self {
        Self {
            coords: Vector::from_fill(value),
        }
    }

    /// Map an operation on that acts on the coordinates of a point, returning
    /// a point whose coordinates are of the new scalar type.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,  
    /// # };
    /// #
    /// let vector: Point3<i32> = Point3::new(1_i32, 2_i32, 3_i32);
    /// let expected: Point3<f64> = Point3::new(2_f64, 3_f64, 4_f64);
    /// let result: Point3<f64> = vector.map(|comp| (comp + 1) as f64);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn map<T, F>(&self, op: F) -> Point<T, N>
    where
        F: FnMut(S) -> T,
    {
        Point { coords: self.coords.map(op) }
    }
}

impl<S, const N: usize> Point<S, N>
where
    S: SimdScalar,
{
    /// The preferred origin of the Euclidean vector space.
    ///
    /// In theory, an Euclidean space does not have a clearly defined origin. In
    /// practice, it is useful to have a reference point in which to express the others
    /// as translations of it. By default, we define the origin as `[0, 0, 0]`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// #
    /// let origin = Point3::origin();
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Vector3::new(1_f64, 2_f64, 3_f64);
    /// // We can express `point` as a vector representing a translation from the origin.
    /// let result = point - origin;
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn origin() -> Self {
        Self { coords: Vector::zero() }
    }

    /// Convert a vector to a point.
    ///
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// #
    /// let vector = Vector3::new(1_i32, 2_i32, 3_i32);
    /// let expected = Point3::new(1_i32, 2_i32, 3_i32);
    /// let result = Point3::from_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn from_vector(vector: &Vector<S, N>) -> Self {
        Self { coords: *vector }
    }

    /// Convert a point to a vector.
    ///
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// #
    /// let point = Point3::new(1_i32, 2_i32, 3_i32);
    /// let expected = Vector3::new(1_i32, 2_i32, 3_i32);
    /// let result = point.to_vector();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn to_vector(&self) -> Vector<S, N> {
        self.coords
    }

    /// Compute the dot product (inner product) of two points.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// #
    /// let point1 = Point3::new(1_f64, 2_f64, 3_f64);
    /// let point2 = Point3::new(4_f64, 5_f64, 6_f64);
    ///
    /// assert_eq!(point1.dot(&point2), 32_f64);
    /// ```
    #[inline]
    pub fn dot(&self, other: &Self) -> S {
        self.coords.dot(&other.coords)
    }

    /// Calculate the squared norm of a [`Point`] with respect to the **L2** (Euclidean) norm.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// #
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = 14_f64;
    /// let result = point.norm_squared();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn norm_squared(&self) -> S {
        self.coords.norm_squared()
    }

    /// Calculate the squared metric distance between two [`Point`]s with respect
    /// to the metric induced by the **L2** norm.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// #
    /// let point1 = Point3::origin();
    /// let point2 = Point3::new(1_f64, 1_f64, 1_f64);
    /// let expected = 3_f64;
    /// let result = point1.metric_distance_squared(&point2);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn metric_distance_squared(&self, other: &Self) -> S {
        self.coords.metric_distance_squared(&other.coords)
    }

    /// Calculate the squared norm of a [`Point`] with respect to the **L2** (Euclidean) norm.
    ///
    /// This is a synonym for [`Point::norm_squared`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// #
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = 14_f64;
    /// let result = point.magnitude_squared();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn magnitude_squared(&self) -> S {
        self.norm_squared()
    }
}

impl<S, const N: usize> Point<S, N>
where
    S: SimdScalarFloat,
{
    /// Calculate the norm of a [`Point`] with respect to the **L2** (Euclidean) norm.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// #
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = f64::sqrt(14_f64);
    /// let result = point.norm();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn norm(&self) -> S {
        self.coords.norm()
    }

    /// Calculate the metric distance between two [`Point`]s with respect
    /// to the metric induced by the **L2** norm.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// #
    /// let point1 = Point3::origin();
    /// let point2 = Point3::new(1_f64, 1_f64, 1_f64);
    /// let expected = f64::sqrt(3_f64);
    /// let result = point1.metric_distance(&point2);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn metric_distance(&self, other: &Self) -> S {
        self.coords.metric_distance(&other.coords)
    }

    /// Calculate the norm of a [`Point`] with respect to the **L2** (Euclidean) norm.
    ///
    /// This is a synonym for [`Point::norm`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// #
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = f64::sqrt(14_f64);
    /// let result = point.norm();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn magnitude(&self) -> S {
        self.norm()
    }

    /// Calculate the norm of a [`Point`] with respect to the **L2** (Euclidean) norm.
    ///
    /// This is a synonym for [`Point::norm`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// #
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    /// let expected = f64::sqrt(14_f64);
    /// let result = point.norm();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn l2_norm(&self) -> S {
        self.norm()
    }
}

impl<S, const N: usize, const NPLUS1: usize> Point<S, N>
where
    S: SimdScalar,
    ShapeConstraint: CanExtend<Const<N>, Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>,
{
    /// Expand a point to a point one dimension higher using
    /// the supplied last element `last_element`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Point3,    
    /// # };
    /// #
    /// let point = Point2::new(1_i32, 2_i32);
    /// let expected = Point3::new(1_i32, 2_i32, 3_i32);
    /// let result = point.extend(3_i32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn extend(&self, last_element: S) -> Point<S, NPLUS1> {
        // SAFETY: The output point has length `N + 1` with `last_element` in the
        // component `N` of the output vector.
        let mut result = Point::default();
        for i in 0..N {
            result.coords[i] = self.coords[i];
        }

        result.coords[N] = last_element;

        result
    }

    /// Convert a point to a vector in homogeneous coordinates.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector3,
    /// # };
    /// #
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = Vector3::new(1_f64, 2_f64, 1_f64);
    /// let result = point.to_homogeneous();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_homogeneous(&self) -> Vector<S, NPLUS1> {
        self.coords.extend(S::one())
    }
}

impl<S, const N: usize, const NMINUS1: usize> Point<S, N>
where
    S: SimdScalar,
    ShapeConstraint: CanContract<Const<N>, Const<NMINUS1>>,
    ShapeConstraint: DimSub<Const<N>, Const<1>, Output = Const<NMINUS1>>,
    ShapeConstraint: DimAdd<Const<NMINUS1>, Const<1>, Output = Const<N>>,
{
    /// Contract a point to a point one dimension smaller the last component.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point1,
    /// #     Point2,
    /// # };
    /// #
    /// let point = Point2::new(1_i32, 2_i32);
    /// let expected = Point1::new(1_i32);
    /// let result = point.contract();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn contract(&self) -> Point<S, NMINUS1> {
        // SAFETY: The output vector has length `N - 1`.
        let mut result = Point::default();
        for i in 0..(N - 1) {
            result.coords[i] = self.coords[i];
        }

        result
    }
}

impl<S, const N: usize, const NPLUS1: usize> Point<S, N>
where
    S: SimdScalar,
    ShapeConstraint: CanContract<Const<NPLUS1>, Const<N>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
{
    /// Convert a homogeneous vector into a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector3,
    /// # };
    /// #
    /// let vector = Vector3::new(3_f64, 6_f64, 3_f64);
    /// let expected = Some(Point2::new(1_f64, 2_f64));
    /// let result = Point2::from_homogeneous(&vector);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vector_z_zero = Vector3::new(3_f64, 6_f64, 0_f64);
    /// let result = Point2::from_homogeneous(&vector_z_zero);
    ///
    /// assert!(result.is_none());
    /// ```
    #[inline]
    pub fn from_homogeneous(vector: &Vector<S, NPLUS1>) -> Option<Point<S, N>> {
        // SAFETY: `vector` has length `N + 1`.
        if !vector[N].is_zero() {
            let mut result = Point::default();
            for i in 0..N {
                result[i] = vector[i] / vector[N];
            }

            Some(result)
        } else {
            None
        }
    }
}

impl<S, const N: usize> fmt::Display for Point<S, N>
where
    S: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Point{} [", N).unwrap();
        for i in 0..(N - 1) {
            write!(formatter, "{}, ", self.coords[i]).unwrap();
        }
        write!(formatter, "{}]", self.coords[N - 1])
    }
}

impl<S, const N: usize> Default for Point<S, N>
where
    S: SimdScalar,
{
    fn default() -> Self {
        Self::origin()
    }
}

impl<S, const N: usize> AsRef<[S; N]> for Point<S, N> {
    #[inline]
    fn as_ref(&self) -> &[S; N] {
        self.coords.as_ref()
    }
}

impl<S, const N: usize> AsMut<[S; N]> for Point<S, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [S; N] {
        self.coords.as_mut()
    }
}

impl<S, const N: usize> From<[S; N]> for Point<S, N>
where
    S: Copy,
{
    #[inline]
    fn from(data: [S; N]) -> Self {
        Self { coords: data.into() }
    }
}

impl<S, const N: usize> From<&[S; N]> for Point<S, N>
where
    S: Copy,
{
    #[inline]
    fn from(data: &[S; N]) -> Self {
        Self { coords: data.into() }
    }
}

impl<'a, S, const N: usize> From<&'a [S; N]> for &'a Point<S, N>
where
    S: Copy,
{
    #[inline]
    fn from(data: &'a [S; N]) -> &'a Point<S, N> {
        unsafe { &*(data as *const [S; N] as *const Point<S, N>) }
    }
}

macro_rules! impl_point_index_ops {
    ($IndexType:ty, $Output:ty) => {
        impl<S, const N: usize> ops::Index<$IndexType> for Point<S, N> {
            type Output = $Output;

            #[inline]
            fn index(&self, index: $IndexType) -> &Self::Output {
                let v: &[S; N] = self.as_ref();
                &v[index]
            }
        }

        impl<S, const N: usize> ops::IndexMut<$IndexType> for Point<S, N> {
            #[inline]
            fn index_mut(&mut self, index: $IndexType) -> &mut Self::Output {
                let v: &mut [S; N] = self.as_mut();
                &mut v[index]
            }
        }
    };
}

impl_point_index_ops!(usize, S);
impl_point_index_ops!(ops::Range<usize>, [S]);
impl_point_index_ops!(ops::RangeTo<usize>, [S]);
impl_point_index_ops!(ops::RangeFrom<usize>, [S]);
impl_point_index_ops!(ops::RangeFull, [S]);

impl<S, const N: usize> ops::Add<Vector<S, N>> for Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn add(self, other: Vector<S, N>) -> Self::Output {
        Self::Output { coords: self.coords + other }
    }
}

impl<S, const N: usize> ops::Add<&Vector<S, N>> for Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn add(self, other: &Vector<S, N>) -> Self::Output {
        Self::Output { coords: self.coords + other }
    }
}

impl<S, const N: usize> ops::Add<Vector<S, N>> for &Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn add(self, other: Vector<S, N>) -> Self::Output {
        Self::Output { coords: self.coords + other }
    }
}

impl<'a, 'b, S, const N: usize> ops::Add<&'b Vector<S, N>> for &'a Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn add(self, other: &'b Vector<S, N>) -> Self::Output {
        Self::Output { coords: self.coords + other }
    }
}

impl<S, const N: usize> ops::Sub<Vector<S, N>> for Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn sub(self, other: Vector<S, N>) -> Self::Output {
        Self::Output { coords: self.coords - other }
    }
}

impl<S, const N: usize> ops::Sub<&Vector<S, N>> for Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn sub(self, other: &Vector<S, N>) -> Self::Output {
        Self::Output { coords: self.coords - other }
    }
}

impl<S, const N: usize> ops::Sub<Vector<S, N>> for &Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn sub(self, other: Vector<S, N>) -> Self::Output {
        Self::Output { coords: self.coords - other }
    }
}

impl<'a, 'b, S, const N: usize> ops::Sub<&'b Vector<S, N>> for &'a Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn sub(self, other: &'b Vector<S, N>) -> Self::Output {
        Self::Output { coords: self.coords - other }
    }
}

impl<S, const N: usize> ops::Sub<Point<S, N>> for Point<S, N>
where
    S: SimdScalar,
{
    type Output = Vector<S, N>;

    #[inline]
    fn sub(self, other: Point<S, N>) -> Self::Output {
        self.coords - other.coords
    }
}

impl<S, const N: usize> ops::Sub<&Point<S, N>> for Point<S, N>
where
    S: SimdScalar,
{
    type Output = Vector<S, N>;

    #[inline]
    fn sub(self, other: &Point<S, N>) -> Self::Output {
        self.coords - other.coords
    }
}

impl<S, const N: usize> ops::Sub<Point<S, N>> for &Point<S, N>
where
    S: SimdScalar,
{
    type Output = Vector<S, N>;

    #[inline]
    fn sub(self, other: Point<S, N>) -> Self::Output {
        self.coords - other.coords
    }
}

impl<'a, 'b, S, const N: usize> ops::Sub<&'b Point<S, N>> for &'a Point<S, N>
where
    S: SimdScalar,
{
    type Output = Vector<S, N>;

    #[inline]
    fn sub(self, other: &'b Point<S, N>) -> Self::Output {
        self.coords - other.coords
    }
}

impl<S, const N: usize> ops::Mul<S> for Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Self::Output { coords: self.coords * other }
    }
}

impl<S, const N: usize> ops::Mul<S> for &Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Self::Output { coords: self.coords * other }
    }
}

impl<S, const N: usize> ops::Div<S> for Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Self::Output { coords: self.coords / other }
    }
}

impl<S, const N: usize> ops::Div<S> for &Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Self::Output { coords: self.coords / other }
    }
}

impl<S, const N: usize> ops::Rem<S> for Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        Self::Output { coords: self.coords % other }
    }
}

impl<S, const N: usize> ops::Rem<S> for &Point<S, N>
where
    S: SimdScalar,
{
    type Output = Point<S, N>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        Self::Output { coords: self.coords % other }
    }
}


macro_rules! impl_scalar_point_mul_ops {
    ($($Lhs:ty),* $(,)*) => {$(
        impl<const N: usize> ops::Mul<Point<$Lhs, N>> for $Lhs {
            type Output = Point<$Lhs, N>;

            #[inline]
            fn mul(self, other: Point<$Lhs, N>) -> Self::Output {
                Self::Output {
                    coords: self * other.coords,
                }
            }
        }

        impl<'a, const N: usize> ops::Mul<Point<$Lhs, N>> for &'a $Lhs {
            type Output = Point<$Lhs, N>;

            #[inline]
            fn mul(self, other: Point<$Lhs, N>) -> Self::Output {
                Self::Output {
                    coords: self * other.coords,
                }
            }
        }
    )*}
}

impl_scalar_point_mul_ops!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);


impl<S, const N: usize> ops::Neg for Point<S, N>
where
    S: SimdScalarSigned,
{
    type Output = Point<S, N>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output { coords: -self.coords }
    }
}

impl<S, const N: usize> ops::Neg for &Point<S, N>
where
    S: SimdScalarSigned,
{
    type Output = Point<S, N>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output { coords: -self.coords }
    }
}

impl<S, const N: usize> ops::AddAssign<Vector<S, N>> for Point<S, N>
where
    S: SimdScalar,
{
    #[inline]
    fn add_assign(&mut self, other: Vector<S, N>) {
        self.coords += other;
    }
}

impl<S, const N: usize> ops::AddAssign<&Vector<S, N>> for Point<S, N>
where
    S: SimdScalar,
{
    #[inline]
    fn add_assign(&mut self, other: &Vector<S, N>) {
        self.coords += other;
    }
}

impl<S, const N: usize> ops::SubAssign<Vector<S, N>> for Point<S, N>
where
    S: SimdScalar,
{
    #[inline]
    fn sub_assign(&mut self, other: Vector<S, N>) {
        self.coords -= other;
    }
}

impl<S, const N: usize> ops::SubAssign<&Vector<S, N>> for Point<S, N>
where
    S: SimdScalar,
{
    #[inline]
    fn sub_assign(&mut self, other: &Vector<S, N>) {
        self.coords -= other;
    }
}

impl<S, const N: usize> ops::MulAssign<S> for Point<S, N>
where
    S: SimdScalar,
{
    #[inline]
    fn mul_assign(&mut self, other: S) {
        self.coords *= other;
    }
}

impl<S, const N: usize> ops::DivAssign<S> for Point<S, N>
where
    S: SimdScalar,
{
    #[inline]
    fn div_assign(&mut self, other: S) {
        self.coords /= other;
    }
}

impl<S, const N: usize> ops::RemAssign<S> for Point<S, N>
where
    S: SimdScalar,
{
    #[inline]
    fn rem_assign(&mut self, other: S) {
        self.coords %= other;
    }
}

impl<S, const N: usize> approx::AbsDiffEq for Point<S, N>
where
    S: SimdScalarFloat,
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Vector::abs_diff_eq(&self.coords, &other.coords, epsilon)
    }
}

impl<S, const N: usize> approx::RelativeEq for Point<S, N>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        Vector::relative_eq(&self.coords, &other.coords, epsilon, max_relative)
    }
}

impl<S, const N: usize> approx::UlpsEq for Point<S, N>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        Vector::ulps_eq(&self.coords, &other.coords, epsilon, max_ulps)
    }
}

impl<S, const N: usize> Normed for Point<S, N>
where
    S: SimdScalarFloat,
{
    type Output = S;

    #[inline]
    fn norm_squared(&self) -> Self::Output {
        self.coords.norm_squared()
    }

    #[inline]
    fn norm(&self) -> Self::Output {
        self.coords.norm()
    }

    #[inline]
    fn scale(&self, norm: Self::Output) -> Self {
        let scaled_coords = self.coords.scale(norm);

        Self::from_vector(&scaled_coords)
    }

    #[inline]
    fn scale_mut(&mut self, norm: Self::Output) {
        self.coords.scale_mut(norm);
    }

    #[inline]
    fn unscale(&self, norm: Self::Output) -> Self {
        let unscaled_coords = self.coords.unscale(norm);

        Self::from_vector(&unscaled_coords)
    }

    #[inline]
    fn unscale_mut(&mut self, norm: Self::Output) {
        self.coords.unscale_mut(norm);
    }

    #[inline]
    fn normalize(&self) -> Self {
        let normalized_coords = self.coords.normalize();

        Self::from_vector(&normalized_coords)
    }

    #[inline]
    fn normalize_mut(&mut self) -> Self::Output {
        self.coords.normalize_mut()
    }

    #[inline]
    fn try_normalize(&self, threshold: Self::Output) -> Option<Self> {
        let norm = self.coords.norm();
        if norm <= threshold {
            None
        } else {
            Some(self.normalize())
        }
    }

    #[inline]
    fn try_normalize_mut(&mut self, threshold: Self::Output) -> Option<Self::Output> {
        self.coords.try_normalize_mut(threshold)
    }

    #[inline]
    fn distance_squared(&self, other: &Point<S, N>) -> Self::Output {
        self.coords.metric_distance_squared(&other.coords)
    }

    #[inline]
    fn distance(&self, other: &Self) -> Self::Output {
        self.coords.metric_distance(&other.coords)
    }
}

impl<S, const N: usize> ops::Neg for Unit<Point<S, N>>
where
    S: SimdScalarFloat,
{
    type Output = Unit<Point<S, N>>;

    #[inline]
    fn neg(self) -> Self::Output {
        Unit::from_value_unchecked(-self.into_inner())
    }
}

impl<S, const N: usize> ops::Neg for &Unit<Point<S, N>>
where
    S: SimdScalarFloat,
{
    type Output = Unit<Point<S, N>>;

    #[inline]
    fn neg(self) -> Self::Output {
        Unit::from_value_unchecked(-self.into_inner())
    }
}


impl<S> Point1<S> {
    /// Construct a new point in Euclidean space.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point1,
    /// # };
    /// #
    /// let x = 1_i32;
    /// let point = Point1::new(x);
    ///
    /// assert_eq!(point[0], x);
    /// ```
    #[inline]
    pub const fn new(x: S) -> Self {
        Self { coords: Vector1::new(x) }
    }
}

impl<S> Point2<S> {
    /// Construct a new point in Euclidean space.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// # };
    /// #
    /// let x = 1_i32;
    /// let y = 2_i32;
    /// let point = Point2::new(x, y);
    ///
    /// assert_eq!(point[0], x);
    /// assert_eq!(point[1], y);
    /// ```
    #[inline]
    pub const fn new(x: S, y: S) -> Self {
        Self { coords: Vector2::new(x, y) }
    }
}

impl<S> Point3<S> {
    /// Construct a new point in Euclidean space.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// #
    /// let x = 1_i32;
    /// let y = 2_i32;
    /// let z = 3_i32;
    /// let point = Point3::new(x, y, z);
    ///
    /// assert_eq!(point[0], x);
    /// assert_eq!(point[1], y);
    /// assert_eq!(point[2], z);
    /// ```
    #[inline]
    pub const fn new(x: S, y: S, z: S) -> Self {
        Self { coords: Vector3::new(x, y, z) }
    }
}

impl<S> From<S> for Point1<S>
where
    S: Copy,
{
    #[inline]
    fn from(v: S) -> Self {
        Self::new(v)
    }
}

impl<S> From<(S,)> for Point1<S>
where
    S: Copy,
{
    #[inline]
    fn from(v: (S,)) -> Self {
        Self::new(v.0)
    }
}

impl<S> From<&(S,)> for Point1<S>
where
    S: Copy,
{
    #[inline]
    fn from(v: &(S,)) -> Self {
        Self::new(v.0)
    }
}

impl<'a, S> From<&'a (S,)> for &'a Point1<S>
where
    S: Copy,
{
    #[inline]
    fn from(v: &'a (S,)) -> &'a Point1<S> {
        unsafe { &*(v as *const (S,) as *const Point1<S>) }
    }
}

impl<S> From<(S, S)> for Point2<S>
where
    S: Copy,
{
    #[inline]
    fn from(v: (S, S)) -> Self {
        Self::new(v.0, v.1)
    }
}

impl<S> From<&(S, S)> for Point2<S>
where
    S: Copy,
{
    #[inline]
    fn from(v: &(S, S)) -> Self {
        Self::new(v.0, v.1)
    }
}

impl<'a, S> From<&'a (S, S)> for &'a Point2<S>
where
    S: Copy,
{
    #[inline]
    fn from(v: &'a (S, S)) -> &'a Point2<S> {
        unsafe { &*(v as *const (S, S) as *const Point2<S>) }
    }
}

impl<S> From<(S, S, S)> for Point3<S>
where
    S: Copy,
{
    #[inline]
    fn from(v: (S, S, S)) -> Self {
        Self::new(v.0, v.1, v.2)
    }
}

impl<S> From<&(S, S, S)> for Point3<S>
where
    S: Copy,
{
    #[inline]
    fn from(v: &(S, S, S)) -> Self {
        Self::new(v.0, v.1, v.2)
    }
}

impl<'a, S> From<&'a (S, S, S)> for &'a Point3<S>
where
    S: Copy,
{
    #[inline]
    fn from(v: &'a (S, S, S)) -> &'a Point3<S> {
        unsafe { &*(v as *const (S, S, S) as *const Point3<S>) }
    }
}

impl_coords!(PointCoordsX, { x });
impl_coords_deref!(Point1, PointCoordsX);

impl_coords!(PointCoordsXY, { x, y });
impl_coords_deref!(Point2, PointCoordsXY);

impl_coords!(PointCoordsXYZ, { x, y, z });
impl_coords_deref!(Point3, PointCoordsXYZ);


macro_rules! impl_as_ref_ops {
    ($PointType:ty, $RefType:ty) => {
        impl<S> AsRef<$RefType> for $PointType {
            #[inline]
            fn as_ref(&self) -> &$RefType {
                self.coords.as_ref()
            }
        }

        impl<S> AsMut<$RefType> for $PointType {
            #[inline]
            fn as_mut(&mut self) -> &mut $RefType {
                self.coords.as_mut()
            }
        }
    };
}

impl_as_ref_ops!(Point1<S>, S);
impl_as_ref_ops!(Point1<S>, (S,));
impl_as_ref_ops!(Point2<S>, (S, S));
impl_as_ref_ops!(Point3<S>, (S, S, S));


macro_rules! impl_swizzle {
    ($name:ident() => $PointN:ident => $Output:ident { $($i:expr),+ }) => {
        impl<S> $PointN<S> where S: Copy {
            /// Construct a new point from the components of the input point.
            #[inline]
            pub fn $name(&self) -> $Output<S> {
                $Output::new(
                    $(self.coords[$i]),*
                )
            }
        }
    }
}

impl_swizzle!(x() => Point1 => Point1 { 0 });

impl_swizzle!(x() => Point2 => Point1 { 0 });
impl_swizzle!(y() => Point2 => Point1 { 1 });

impl_swizzle!(xx() => Point2 => Point2 { 0, 0 });
impl_swizzle!(xy() => Point2 => Point2 { 0, 1 });
impl_swizzle!(yx() => Point2 => Point2 { 1, 0 });
impl_swizzle!(yy() => Point2 => Point2 { 1, 1 });

impl_swizzle!(x() => Point3 => Point1 { 0 });
impl_swizzle!(y() => Point3 => Point1 { 1 });
impl_swizzle!(z() => Point3 => Point1 { 2 });

impl_swizzle!(xx() => Point3 => Point2 { 0, 0 });
impl_swizzle!(xy() => Point3 => Point2 { 0, 1 });
impl_swizzle!(xz() => Point3 => Point2 { 0, 2 });
impl_swizzle!(yx() => Point3 => Point2 { 1, 0 });
impl_swizzle!(yy() => Point3 => Point2 { 1, 1 });
impl_swizzle!(yz() => Point3 => Point2 { 1, 2 });
impl_swizzle!(zx() => Point3 => Point2 { 2, 0 });
impl_swizzle!(zy() => Point3 => Point2 { 2, 1 });
impl_swizzle!(zz() => Point3 => Point2 { 2, 2 });

impl_swizzle!(xxx() => Point3 => Point3 { 0, 0, 0 });
impl_swizzle!(xxy() => Point3 => Point3 { 0, 0, 1 });
impl_swizzle!(xxz() => Point3 => Point3 { 0, 0, 2 });
impl_swizzle!(xyx() => Point3 => Point3 { 0, 1, 0 });
impl_swizzle!(xyy() => Point3 => Point3 { 0, 1, 1 });
impl_swizzle!(xyz() => Point3 => Point3 { 0, 1, 2 });
impl_swizzle!(xzx() => Point3 => Point3 { 0, 2, 0 });
impl_swizzle!(xzy() => Point3 => Point3 { 0, 2, 1 });
impl_swizzle!(xzz() => Point3 => Point3 { 0, 2, 2 });
impl_swizzle!(yxx() => Point3 => Point3 { 1, 0, 0 });
impl_swizzle!(yxy() => Point3 => Point3 { 1, 0, 1 });
impl_swizzle!(yxz() => Point3 => Point3 { 1, 0, 2 });
impl_swizzle!(yyx() => Point3 => Point3 { 1, 1, 0 });
impl_swizzle!(yyy() => Point3 => Point3 { 1, 1, 1 });
impl_swizzle!(yyz() => Point3 => Point3 { 1, 1, 2 });
impl_swizzle!(yzx() => Point3 => Point3 { 1, 2, 0 });
impl_swizzle!(yzy() => Point3 => Point3 { 1, 2, 1 });
impl_swizzle!(yzz() => Point3 => Point3 { 1, 2, 2 });
impl_swizzle!(zxx() => Point3 => Point3 { 2, 0, 0 });
impl_swizzle!(zxy() => Point3 => Point3 { 2, 0, 1 });
impl_swizzle!(zxz() => Point3 => Point3 { 2, 0, 2 });
impl_swizzle!(zyx() => Point3 => Point3 { 2, 1, 0 });
impl_swizzle!(zyy() => Point3 => Point3 { 2, 1, 1 });
impl_swizzle!(zyz() => Point3 => Point3 { 2, 1, 2 });
impl_swizzle!(zzx() => Point3 => Point3 { 2, 2, 0 });
impl_swizzle!(zzy() => Point3 => Point3 { 2, 2, 1 });
impl_swizzle!(zzz() => Point3 => Point3 { 2, 2, 2 });
