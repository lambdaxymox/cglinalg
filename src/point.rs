use crate::common::{
    Magnitude,
    SimdScalar,
    SimdScalarSigned,
    SimdScalarFloat,
};
use crate::vector::{
    Vector,
    Vector1,
    Vector2,
    Vector3,
    Vector4,
};
use crate::{
    impl_coords,
    impl_coords_deref,
};
use num_traits::{
    NumCast,
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
    /// Returns the length of the the underlying array storing the point components.
    #[inline]
    pub const fn len(&self) -> usize {
        N
    }

    /// Tests whether the number of elements in the point is zero.
    /// 
    /// Returns `false` when the point is zero-dimensional. Returns `true` 
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
        <Self as AsRef<[S; N]>>::as_ref(self)
    }
}

impl<S, const N: usize> Point<S, N> 
where 
    S: NumCast + Copy
{
    /// Cast a point from one type of scalars to another type of scalars.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3,   
    /// # };
    /// #
    /// let point: Point3<u32> = Point3::new(1_u32, 2_u32, 3_u32);
    /// let expected: Option<Point3<i32>> = Some(Point3::new(1_i32, 2_i32, 3_i32));
    /// let result = point.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Point<T, N>> {
        self.coords.cast::<T>().map(|coords| Point { coords })
    }
}

impl<S, const N: usize> Point<S, N>
where
    S: Copy
{
    /// Construct a new point from a fill value.
    /// 
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3, 
    /// # };
    /// #
    /// let fill_value = 3_u32;
    /// let expected = Point3::new(3_u32, 3_u32, 3_u32);
    /// let result = Point3::from_fill(fill_value);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_fill(value: S) -> Self {
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
    /// # use cglinalg::{
    /// #     Point3,  
    /// # };
    /// #
    /// let vector: Point3<u32> = Point3::new(1_u32, 2_u32, 3_u32);
    /// let expected: Point3<i32> = Point3::new(2_i32, 3_i32, 4_i32);
    /// let result: Point3<i32> = vector.map(|comp| (comp + 1) as i32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn map<T, F>(&self, op: F) -> Point<T, N> 
    where 
        F: FnMut(S) -> T
    {
        Point {
            coords: self.coords.map(op),
        }
    }
}

impl<S, const N: usize> AsRef<[S; N]> for Point<S, N> {
    #[inline]
    fn as_ref(&self) -> &[S; N] {
        unsafe {
            &*(self as *const Point<S, N> as *const [S; N])
        }
    }
}

impl<S, const N: usize> AsMut<[S; N]> for Point<S, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [S; N] {
        unsafe {
            &mut *(self as *mut Point<S, N> as *mut [S; N])
        }
    }
}

impl<S, const N: usize> Point<S, N>
where
    S: SimdScalar
{
    /// Compute the origin of the Euclidean vector space.
    #[inline]
    pub fn origin() -> Self {
        Self {
            coords: Vector::zero(),
        }
    }

    /// Convert a vector to a point. 
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// #
    /// let vector = Vector3::new(1_u32, 2_u32, 3_u32);
    /// let expected = Point3::new(1_u32, 2_u32, 3_u32);
    /// let result = Point3::from_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_vector(vector: &Vector<S, N>) -> Self {
        Self {
            coords: *vector,
        }
    }

    /// Convert a point to a vector.
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// #
    /// let point = Point3::new(1_u32, 2_u32, 3_u32);
    /// let expected = Vector3::new(1_u32, 2_u32, 3_u32);
    /// let result = point.to_vector();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_vector(&self) -> Vector<S, N> {
        self.coords
    }

    /// Compute the dot product (inner product) of two points.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
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
}

impl<S, const N: usize> fmt::Display for Point<S, N> 
where 
    S: fmt::Display 
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
    S: SimdScalar
{
    fn default() -> Self {
        Self::origin()
    }
}

impl<S, const N: usize> From<[S; N]> for Point<S, N> 
where 
    S: Copy
{
    #[inline]
    fn from(data: [S; N]) -> Self {
        Self { 
            coords: data.into(),
        }
    }
}

impl<S, const N: usize> From<&[S; N]> for Point<S, N> 
where 
    S: Copy
{
    #[inline]
    fn from(data: &[S; N]) -> Self {
        Self {
            coords: data.into(),
        }
    }
}

impl<'a, S, const N: usize> From<&'a [S; N]> for &'a Point<S, N> 
where 
    S: Copy
{
    #[inline]
    fn from(data: &'a [S; N]) -> &'a Point<S, N> {
        unsafe { 
            &*(data as *const [S; N] as *const Point<S, N>)
        }
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
    }
}

impl_point_index_ops!(usize, S);
impl_point_index_ops!(ops::Range<usize>, [S]);
impl_point_index_ops!(ops::RangeTo<usize>, [S]);
impl_point_index_ops!(ops::RangeFrom<usize>, [S]);
impl_point_index_ops!(ops::RangeFull, [S]);

impl<S, const N: usize> ops::Add<Vector<S, N>> for Point<S, N> 
where 
    S: SimdScalar 
{
    type Output = Point<S, N>;

    #[inline]
    fn add(self, other: Vector<S, N>) -> Self::Output {
        Self::Output {
            coords: self.coords + other,
        }
    }
}

impl<S, const N: usize> ops::Add<&Vector<S, N>> for Point<S, N> 
where 
    S: SimdScalar 
{
    type Output = Point<S, N>;

    #[inline]
    fn add(self, other: &Vector<S, N>) -> Self::Output {
        Self::Output {
            coords: self.coords + other,
        }
    }
}

impl<S, const N: usize> ops::Add<Vector<S, N>> for &Point<S, N> 
where 
    S: SimdScalar 
{
    type Output = Point<S, N>;

    #[inline]
    fn add(self, other: Vector<S, N>) -> Self::Output {
        Self::Output {
            coords: self.coords + other,
        }
    }
}

impl<'a, 'b, S, const N: usize> ops::Add<&'a Vector<S, N>> for &'b Point<S, N> 
where 
    S: SimdScalar 
{
    type Output = Point<S, N>;

    #[inline]
    fn add(self, other: &'a Vector<S, N>) -> Self::Output {
        Self::Output {
            coords: self.coords + other,
        }
    }
}

impl<S, const N: usize> ops::Sub<Vector<S, N>> for Point<S, N> 
where 
    S: SimdScalar 
{
    type Output = Point<S, N>;

    #[inline]
    fn sub(self, other: Vector<S, N>) -> Self::Output {
        Self::Output {
            coords: self.coords - other,
        }
    }
}

impl<S, const N: usize> ops::Sub<&Vector<S, N>> for Point<S, N> 
where 
    S: SimdScalar 
{
    type Output = Point<S, N>;

    #[inline]
    fn sub(self, other: &Vector<S, N>) -> Self::Output {
        Self::Output {
            coords: self.coords - other,
        }
    }
}

impl<S, const N: usize> ops::Sub<Vector<S, N>> for &Point<S, N> 
where 
    S: SimdScalar 
{
    type Output = Point<S, N>;

    #[inline]
    fn sub(self, other: Vector<S, N>) -> Self::Output {
        Self::Output {
            coords: self.coords - other,
        }
    }
}

impl<'a, 'b, S, const N: usize> ops::Sub<&'a Vector<S, N>> for &'b Point<S, N>
where 
    S: SimdScalar
{
    type Output = Point<S, N>;

    #[inline]
    fn sub(self, other: &'a Vector<S, N>) -> Self::Output {
        Self::Output {
            coords: self.coords - other,
        }
    }
}

impl<S, const N: usize> ops::Sub<Point<S, N>> for Point<S, N> 
where 
    S: SimdScalar
{
    type Output = Vector<S, N>;

    #[inline]
    fn sub(self, other: Point<S, N>) -> Self::Output {
        self.coords - other.coords
    }
}

impl<S, const N: usize> ops::Sub<&Point<S, N>> for Point<S, N> 
where 
    S: SimdScalar 
{
    type Output = Vector<S, N>;

    #[inline]
    fn sub(self, other: &Point<S, N>) -> Self::Output {
        self.coords - other.coords
    }
}

impl<S, const N: usize> ops::Sub<Point<S, N>> for &Point<S, N> 
where 
    S: SimdScalar 
{
    type Output = Vector<S, N>;

    #[inline]
    fn sub(self, other: Point<S, N>) -> Self::Output {
        self.coords - other.coords
    }
}

impl<'a, 'b, S, const N: usize> ops::Sub<&'a Point<S, N>> for &'b Point<S, N> 
where 
    S: SimdScalar
{
    type Output = Vector<S, N>;

    #[inline]
    fn sub(self, other: &'a Point<S, N>) -> Self::Output {
        self.coords - other.coords
    }
}

impl<S, const N: usize> ops::Mul<S> for Point<S, N> 
where 
    S: SimdScalar 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Self::Output { 
            coords: self.coords * other,
        }
    }
}

impl<S, const N: usize> ops::Mul<S> for &Point<S, N> 
where 
    S: SimdScalar
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Self::Output {
            coords: self.coords * other,
        }
    }
}

impl<S, const N: usize> ops::Div<S> for Point<S, N> 
where 
    S: SimdScalar 
{
    type Output = Point<S, N>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Self::Output { 
            coords: self.coords / other,
        }
    }
}

impl<S, const N: usize> ops::Div<S> for &Point<S, N> 
where 
    S: SimdScalar
{
    type Output = Point<S, N>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Self::Output {
            coords: self.coords / other,
        }
    }
}

impl<S, const N: usize> ops::Rem<S> for Point<S, N> 
where 
    S: SimdScalar 
{
    type Output = Point<S, N>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        Self::Output { 
            coords: self.coords % other,
        }
    }
}

impl<S, const N: usize> ops::Rem<S> for &Point<S, N> 
where 
    S: SimdScalar
{
    type Output = Point<S, N>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        Self::Output {
            coords: self.coords % other,
        }
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
    S: SimdScalarSigned
{
    type Output = Point<S, N>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output { coords: -self.coords }
    }
}

impl<S, const N: usize> ops::Neg for &Point<S, N>
where 
    S: SimdScalarSigned
{
    type Output = Point<S, N>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output { coords: -self.coords }
    }
}

impl<S, const N: usize> ops::AddAssign<Vector<S, N>> for Point<S, N> 
where
    S: SimdScalar
{
    #[inline]
    fn add_assign(&mut self, other: Vector<S, N>) {
        self.coords += other;
    }
}

impl<S, const N: usize> ops::AddAssign<&Vector<S, N>> for Point<S, N> 
where 
    S: SimdScalar
{
    #[inline]
    fn add_assign(&mut self, other: &Vector<S, N>) {
        self.coords += other;
    }
}

impl<S, const N: usize> ops::SubAssign<Vector<S, N>> for Point<S, N> 
where 
    S: SimdScalar
{
    #[inline]
    fn sub_assign(&mut self, other: Vector<S, N>) {
        self.coords -= other;
    }
}

impl<S, const N: usize> ops::SubAssign<&Vector<S, N>> for Point<S, N> 
where 
    S: SimdScalar
{
    #[inline]
    fn sub_assign(&mut self, other: &Vector<S, N>) {
        self.coords -= other;
    }
}

impl<S, const N: usize> ops::MulAssign<S> for Point<S, N> 
where
    S: SimdScalar
{
    #[inline]
    fn mul_assign(&mut self, other: S) {
        self.coords *= other;
    }
}

impl<S, const N: usize> ops::DivAssign<S> for Point<S, N> 
where 
    S: SimdScalar
{
    #[inline]
    fn div_assign(&mut self, other: S) {
        self.coords /= other;
    }
}

impl<S, const N: usize> ops::RemAssign<S> for Point<S, N> 
where 
    S: SimdScalar
{
    #[inline]
    fn rem_assign(&mut self, other: S) {
        self.coords %= other;
    }
}

impl<S, const N: usize> Magnitude for Point<S, N> 
where 
    S: SimdScalarFloat
{
    type Output = S;

    #[inline]
    fn magnitude_squared(&self) -> Self::Output {
        self.dot(self)
    }

    #[inline]
    fn magnitude(&self) -> Self::Output {
        self.magnitude_squared().sqrt()
    }
    
    #[inline]
    fn normalize(&self) -> Self {
        self / self.magnitude()
    }
    
    #[inline]
    fn normalize_to(&self, magnitude: Self::Output) -> Self {
        self * (magnitude / self.magnitude())
    }

    #[inline]
    fn try_normalize(&self, threshold: Self::Output) -> Option<Self> {
        let magnitude = self.magnitude();
        if magnitude <= threshold {
            None
        } else {
            Some(self.normalize())
        }
    }

    #[inline]
    fn distance_squared(&self, other: &Point<S, N>) -> Self::Output {
        (self - other).magnitude_squared()
    }

    #[inline]
    fn distance(&self, other: &Self) -> Self::Output {
        self.distance_squared(other).sqrt()
    }
}

impl<S, const N: usize> approx::AbsDiffEq for Point<S, N> 
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
        Vector::abs_diff_eq(&self.coords, &other.coords, epsilon)
    }
}

impl<S, const N: usize> approx::RelativeEq for Point<S, N> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Vector::relative_eq(&self.coords, &other.coords, epsilon, max_relative)
    }
}

impl<S, const N: usize> approx::UlpsEq for Point<S, N> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Vector::ulps_eq(&self.coords, &other.coords, epsilon, max_ulps)
    }
}


impl<S> Point1<S> {
    /// Construct a new point in one-dimensional Euclidean space.
    #[inline]
    pub const fn new(x: S) -> Self {
        Point1 { 
            coords: Vector1::new(x), 
        }
    }
}

impl<S> Point1<S> 
where 
    S: Copy
{
    /// Construct a new two-dimensional point from a one-dimensional point by
    /// supplying the y-coordinate.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Point1,
    /// #     Point2, 
    /// # };
    /// #
    /// let point = Point1::new(1_u32);
    /// let expected = Point2::new(1_u32, 2_u32);
    /// let result = point.extend(2_u32);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn extend(&self, y: S) -> Point2<S> {
        Point2::new(self.coords[0], y)
    }
}

impl<S> Point2<S> {
    /// Construct a new two-dimensional point.
    #[inline]
    pub const fn new(x: S, y: S) -> Self {
        Self { 
            coords: Vector2::new(x, y) 
        }
    }
}

impl<S> Point2<S> 
where 
    S: Copy
{
    /// Expand a two-dimensional point to a three-dimensional point using
    /// the supplied z-value.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2,
    /// #     Point3,    
    /// # };
    /// #
    /// let point = Point2::new(1_u32, 2_u32);
    /// let expected = Point3::new(1_u32, 2_u32, 3_u32);
    /// let result = point.extend(3_u32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn extend(&self, z: S) -> Point3<S> {
        Point3::new(self.coords[0], self.coords[1], z)
    }

    /// Contract a two-dimensional point to a one-dimensional point by
    /// removing its **y-component**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point1,
    /// #     Point2, 
    /// # };
    /// #
    /// let point = Point2::new(1_u32, 2_u32);
    /// let expected = Point1::new(1_u32);
    /// let result = point.contract();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn contract(&self) -> Point1<S> {
        Point1::new(self.coords[0])
    }
}

impl<S> Point2<S> 
where 
    S: SimdScalar
{
    /// Convert a homogeneous vector into a point.
    ///
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
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
    pub fn from_homogeneous(vector: &Vector3<S>) -> Option<Self> {
        if !vector.z.is_zero() {
            Some(Point2::new(vector.x / vector.z, vector.y / vector.z))
        } else {
            None
        }
    }

    /// Convert a point to a vector in homogeneous coordinates.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
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
    pub fn to_homogeneous(&self) -> Vector3<S> {
        self.coords.extend(S::one())
    }
}

impl<S> Point3<S> {
    /// Construct a new point in three-dimensional Euclidean space.
    #[inline]
    pub const fn new(x: S, y: S, z: S) -> Self {
        Self { 
            coords: Vector3::new(x, y, z),
        }
    }
}

impl<S> Point3<S> 
where 
    S: Copy
{
    /// Contract a three-dimensional point, removing its **z-component**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2,
    /// #     Point3, 
    /// # };
    /// #
    /// let point = Point3::new(1_u32, 2_u32, 3_u32);
    /// let expected = Point2::new(1_u32, 2_u32);
    /// let result = point.contract();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn contract(&self) -> Point2<S> {
        Point2::new(self.coords[0], self.coords[1])
    }
}

impl<S> Point3<S> 
where 
    S: SimdScalar
{
    /// Convert a vector in homogeneous coordinates into a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3,
    /// #     Vector4, 
    /// # };
    /// #
    /// let vector = Vector4::new(5_f64, 10_f64, 15_f64, 5_f64);
    /// let expected = Some(Point3::new(1_f64, 2_f64, 3_f64));
    /// let result = Point3::from_homogeneous(&vector);
    ///
    /// assert!(result.is_some());
    /// assert_eq!(result, expected);
    ///
    /// let vector_w_zero = Vector4::new(5_f64, 10_f64, 15_f64, 0_f64);
    /// let result = Point3::from_homogeneous(&vector_w_zero);
    ///
    /// assert!(result.is_none());
    /// ```
    #[inline]
    pub fn from_homogeneous(vector: &Vector4<S>) -> Option<Self> {
        if !vector.w.is_zero() {
            Some(Self::new(
                vector.x / vector.w, 
                vector.y / vector.w, 
                vector.z / vector.w
            ))
        } else {
            None
        }
    }

    /// Convert a point to a vector in homogeneous coordinates.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3,
    /// #     Vector4, 
    /// # };
    /// #
    /// let point = Point3::new(1_u32, 2_u32, 3_u32);
    /// let expected = Vector4::new(1_u32, 2_u32, 3_u32, 1_u32);
    /// let result = point.to_homogeneous();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_homogeneous(&self) -> Vector4<S> {
        self.coords.extend(S::one())
    }
}

impl<S> From<S> for Point1<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: S) -> Self {
        Self::new(v)
    }
}

impl<S> From<(S,)> for Point1<S>
where
    S: Copy
{
    #[inline]
    fn from(v: (S,)) -> Self {
        Self::new(v.0)
    }
}

impl<S> From<&(S,)> for Point1<S>
where
    S: Copy
{
    #[inline]
    fn from(v: &(S,)) -> Self {
        Self::new(v.0)
    }
}

impl<'a, S> From<&'a (S,)> for &'a Point1<S>
where
    S: Copy
{
    #[inline]
    fn from(v: &'a (S,)) -> &'a Point1<S> {
        unsafe {
            &*(v as *const (S,) as *const Point1<S>)
        }
    }
}

impl<S> From<(S, S)> for Point2<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: (S, S)) -> Self {
        Self::new(v.0, v.1)
    }
}

impl<S> From<&(S, S)> for Point2<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: &(S, S)) -> Self {
        Self::new(v.0, v.1)
    }
}

impl<'a, S> From<&'a (S, S)> for &'a Point2<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: &'a (S, S)) -> &'a Point2<S> {
        unsafe { 
            &*(v as *const (S, S) as *const Point2<S>)
        }
    }
}

impl<S> From<(S, S, S)> for Point3<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: (S, S, S)) -> Self {
        Self::new(v.0, v.1, v.2)
    }
}

impl<S> From<&(S, S, S)> for Point3<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: &(S, S, S)) -> Self {
        Self::new(v.0, v.1, v.2)
    }
}

impl<'a, S> From<&'a (S, S, S)> for &'a Point3<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: &'a (S, S, S)) -> &'a Point3<S> {
        unsafe { 
            &*(v as *const (S, S, S) as *const Point3<S>)
        }
    }
}

impl_coords!(X, { x });
impl_coords_deref!(Point1, X);

impl_coords!(XY, { x, y });
impl_coords_deref!(Point2, XY);

impl_coords!(XYZ, { x, y, z });
impl_coords_deref!(Point3, XYZ);


macro_rules! impl_as_ref_ops {
    ($PointType:ty, $RefType:ty) => {
        impl<S> AsRef<$RefType> for $PointType {
            #[inline]
            fn as_ref(&self) -> &$RefType {
                unsafe {
                    &*(self as *const $PointType as *const $RefType)
                }
            }
        }

        impl<S> AsMut<$RefType> for $PointType {
            #[inline]
            fn as_mut(&mut self) -> &mut $RefType {
                unsafe {
                    &mut *(self as *mut $PointType as *mut $RefType)
                }
            }
        }
    }
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

