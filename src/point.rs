use crate::scalar::{
    Scalar,
    ScalarSigned,
    ScalarFloat,
};
use crate::magnitude::{
    Magnitude,
};
use crate::vector::{
    Vector1,
    Vector2,
    Vector3,
    Vector4,
};

use num_traits::{
    NumCast,
};

use core::fmt;
use core::ops;
use core::ops::*;


macro_rules! impl_scalar_vector_mul_ops {
    ($Lhs:ty, $Rhs:ty, $Output:ty, { $($field:ident),* }) => {
        impl ops::Mul<$Rhs> for $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                <$Output>::new( $(self * other.$field),*)
            }
        }

        impl<'a> ops::Mul<$Rhs> for &'a $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                <$Output>::new( $(self * other.$field),*)
            }
        }
    }
}

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

macro_rules! impl_point_index_ops {
    ($T:ty, $n:expr, $IndexType:ty, $Output:ty) => {
        impl<S> ops::Index<$IndexType> for $T {
            type Output = $Output;

            #[inline]
            fn index(&self, index: $IndexType) -> &Self::Output {
                let v: &[S; $n] = self.as_ref();
                &v[index]
            }
        }

        impl<S> ops::IndexMut<$IndexType> for $T {
            #[inline]
            fn index_mut(&mut self, index: $IndexType) -> &mut Self::Output {
                let v: &mut [S; $n] = self.as_mut();
                &mut v[index]
            }
        }
    }
}

macro_rules! impl_point_unary_ops {
    ($OpType:ident, $op:ident, $T:ty, $Output:ty, { $($field:ident),* }) => {
        impl<S> $OpType for $T where S: ScalarSigned {
            type Output = $Output;

            #[inline]
            fn $op(self) -> Self::Output {
                Self::Output::new( 
                    $( self.$field.$op() ),* 
                )
            }
        }

        impl<S> $OpType for &$T where S: ScalarSigned {
            type Output = $Output;

            #[inline]
            fn $op(self) -> Self::Output {
                Self::Output::new( 
                    $( self.$field.$op() ),* 
                )
            }
        }
    }
}

macro_rules! impl_point_binary_assign_ops {
    ($PointType:ty, $VectorType:ty, { $($field:ident),* }) => {
        impl<S> ops::AddAssign<$VectorType> for $PointType where S: Scalar {
            #[inline]
            fn add_assign(&mut self, other: $VectorType) {
                $(self.$field += other.$field);*
            }
        }

        impl<S> ops::AddAssign<&$VectorType> for $PointType where S: Scalar {
            #[inline]
            fn add_assign(&mut self, other: &$VectorType) {
                $(self.$field += other.$field);*
            }
        }

        impl<S> ops::SubAssign<$VectorType> for $PointType where S: Scalar {
            #[inline]
            fn sub_assign(&mut self, other: $VectorType) {
                $(self.$field -= other.$field);*
            }
        }

        impl<S> ops::SubAssign<&$VectorType> for $PointType where S: Scalar {
            #[inline]
            fn sub_assign(&mut self, other: &$VectorType) {
                $(self.$field -= other.$field);*
            }
        }

        impl<S> ops::MulAssign<S> for $PointType where S: Scalar {
            #[inline]
            fn mul_assign(&mut self, other: S) {
                $(self.$field *= other);*
            }
        }
        
        impl<S> ops::DivAssign<S> for $PointType where S: Scalar {
            #[inline]
            fn div_assign(&mut self, other: S) {
                $(self.$field /= other);*
            }
        }
        
        impl<S> ops::RemAssign<S> for $PointType where S: Scalar {
            #[inline]
            fn rem_assign(&mut self, other: S) {
                $(self.$field %= other);*
            }
        }
    }
}

/// A point is a location in a one-dimensional Euclidean space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Point1<S> {
    /// The horizontal coordinate.
    pub x: S,
}

impl<S> Point1<S> {
    /// Construct a new point in one-dimensional Euclidean space.
    #[inline]
    pub const fn new(x: S) -> Point1<S> {
        Point1 { 
            x: x 
        }
    }

    /// Map an operation on that acts on the coordinates of a point, returning 
    /// a point of the new underlying type.
    #[inline]
    pub fn map<T, F>(self, mut op: F) -> Point1<T> 
        where F: FnMut(S) -> T 
    {
        Point1 { 
            x: op(self.x) 
        }
    }
}

impl<S> Point1<S> where S: Copy {
    /// Construct a new two-dimensional point from a one-dimensional point by
    /// supplying the y-coordinate.
    /// 
    /// ## Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Point1,
    /// #     Point2, 
    /// # };
    /// #
    /// let point = Point1::new(1_u32);
    /// let expected = Point2::new(1_u32, 2_u32);
    /// let result = point.expand(2_u32);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn expand(self, y: S) -> Point2<S> {
        Point2::new(self.x, y)
    }

    /// Construct a new point from a fill value.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point1,
    /// # };
    /// 
    /// let fill_value = 1_u32;
    /// let expected = Point1::new(1_u32);
    /// let result = Point1::from_fill(fill_value);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_fill(value: S) -> Point1<S> {
        Point1::new(value)
    }

    /// The length of the the underlying array.
    #[inline]
    pub fn len() -> usize {
        1
    }

    /// The shape of the underlying array.
    #[inline]
    pub fn shape() -> (usize, usize) {
        (1, 1)
    }

    /// Generate a pointer to the underlying array.
    #[inline]
    pub fn as_ptr(&self) -> *const S {
        &self.x
    }

    // Generate a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        &mut self.x
    }

    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        <Self as AsRef<[S; 1]>>::as_ref(self)
    }
}

impl<S> Point1<S> where S: NumCast + Copy {
    /// Cast a point of one type of scalars to a point of another type of scalars.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point1,   
    /// # };
    /// #
    /// let point: Point1<u32> = Point1::new(1_u32);
    /// let expected: Option<Point1<i32>> = Some(Point1::new(1_i32));
    /// let result = point.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Point1<T>> {
        let x = match num_traits::cast(self.x) {
            Some(value) => value,
            None => return None,
        };

        Some(Point1::new(x))
    }
}

impl<S> Point1<S> where S: Scalar {
    /// Compute the origin of the Euclidean vector space.
    #[inline]
    pub fn origin() -> Point1<S> {
        Point1::new(S::zero())
    }

    /// Convert a vector to a point. 
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point1, 
    /// #     Vector1,
    /// # };
    /// #
    /// let vector = Vector1::new(1_u32);
    /// let expected = Point1::new(1_u32);
    /// let result = Point1::from_vector(vector);
    ///
    /// assert_eq!(result, expected);
    /// ``` 
    #[inline]
    pub fn from_vector(v: Vector1<S>) -> Point1<S> {
        Point1::new(v.x)
    }
    
    /// Convert a point to a vector.
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point1, 
    /// #     Vector1,
    /// # };
    /// #
    /// let point = Point1::new(1_u32);
    /// let expected = Vector1::new(1_u32);
    /// let result = point.to_vector();
    ///
    /// assert_eq!(result, expected);
    /// ``` 
    #[inline]
    pub fn to_vector(self) -> Vector1<S> {
        Vector1::new(self.x)
    }

    /// Compute the dot product (inner product) of two points.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point1, 
    /// # };
    /// #
    /// let point1 = Point1::new(1_f64);
    /// let point2 = Point1::new(2_f64);
    /// 
    /// assert_eq!(point1.dot(&point2), 2_f64);
    /// ```
    #[inline]
    pub fn dot(self, other: &Point1<S>) -> S {
        self.x * other.x
    }
}

impl<S> fmt::Display for Point1<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Point1 [{}]", self.x)
    }
}

impl<S> From<S> for Point1<S> where S: Scalar {
    #[inline]
    fn from(v: S) -> Point1<S> {
        Point1 { x: v }
    }
}

impl<S> From<[S; 1]> for Point1<S> where S: Scalar {
    #[inline]
    fn from(v: [S; 1]) -> Point1<S> {
        Point1 { x: v[0] }
    }
}

impl<S> From<&[S; 1]> for Point1<S> where S: Scalar {
    #[inline]
    fn from(v: &[S; 1]) -> Point1<S> {
        Point1 { x: v[0] }
    }
}

impl<'a, S> From<&'a [S; 1]> for &'a Point1<S> where S: Scalar {
    #[inline]
    fn from(v: &'a [S; 1]) -> &'a Point1<S> {
        unsafe { 
            &*(v as *const [S; 1] as *const Point1<S>)
        }
    }
}

impl_as_ref_ops!(Point1<S>, S);
impl_as_ref_ops!(Point1<S>, (S,));
impl_as_ref_ops!(Point1<S>, [S; 1]);

impl_point_index_ops!(Point1<S>, 1, usize, S);
impl_point_index_ops!(Point1<S>, 1, Range<usize>, [S]);
impl_point_index_ops!(Point1<S>, 1, RangeTo<usize>, [S]);
impl_point_index_ops!(Point1<S>, 1, RangeFrom<usize>, [S]);
impl_point_index_ops!(Point1<S>, 1, RangeFull, [S]);

impl<S> ops::Add<Vector1<S>> for Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn add(self, other: Vector1<S>) -> Self::Output {
        Point1 {
            x: self.x + other.x,
        }
    }
}

impl<S> ops::Add<Vector1<S>> for &Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn add(self, other: Vector1<S>) -> Self::Output {
        Point1 {
            x: self.x + other.x,
        }
    }
}

impl<S> ops::Add<&Vector1<S>> for Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn add(self, other: &Vector1<S>) -> Self::Output {
        Point1 {
            x: self.x + other.x,
        }
    }
}

impl<'a, 'b, S> ops::Add<&'b Vector1<S>> for &'a Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn add(self, other: &'b Vector1<S>) -> Self::Output {
        Point1 {
            x: self.x + other.x,
        }
    }
}

impl<S> ops::Sub<Point1<S>> for &Point1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn sub(self, other: Point1<S>) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl<S> ops::Sub<Point1<S>> for Point1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn sub(self, other: Point1<S>) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl<S> ops::Sub<&Point1<S>> for Point1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn sub(self, other: &Point1<S>) -> Self::Output {
        Vector1 {
            x: self.x - other.x,          
        }
    }
}

impl<'a, 'b, S> ops::Sub<&'b Point1<S>> for &'a Point1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn sub(self, other: &'b Point1<S>) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl<S> ops::Sub<Vector1<S>> for &Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn sub(self, other: Vector1<S>) -> Self::Output {
        Point1 {
            x: self.x - other.x,
        }
    }
}

impl<S> ops::Sub<Vector1<S>> for Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn sub(self, other: Vector1<S>) -> Self::Output {
        Point1 {
            x: self.x - other.x,
        }
    }
}

impl<S> ops::Sub<&Vector1<S>> for Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn sub(self, other: &Vector1<S>) -> Self::Output {
        Point1 {
            x: self.x - other.x,          
        }
    }
}

impl<'a, 'b, S> ops::Sub<&'b Vector1<S>> for &'a Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn sub(self, other: &'b Vector1<S>) -> Self::Output {
        Point1 {
            x: self.x - other.x,
        }
    }
}

impl<S> ops::Mul<S> for Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Point1 {
            x: self.x * other,
        }
    }
}

impl<S> ops::Mul<S> for &Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Point1 {
            x: self.x * other,
        }
    }
}

impl<S> ops::Div<S> for Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Point1 {
            x: self.x / other,
        }
    }
}

impl<S> ops::Div<S> for &Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Point1 {
            x: self.x / other,
        }
    }
}

impl<S> ops::Rem<S> for Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        
        Point1::new(x)
    }
}

impl<S> ops::Rem<S> for &Point1<S> where S: Scalar {
    type Output = Point1<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        
        Point1::new(x)
    }
}

impl_scalar_vector_mul_ops!(u8,    Point1<u8>,    Point1<u8>,    { x });
impl_scalar_vector_mul_ops!(u16,   Point1<u16>,   Point1<u16>,   { x });
impl_scalar_vector_mul_ops!(u32,   Point1<u32>,   Point1<u32>,   { x });
impl_scalar_vector_mul_ops!(u64,   Point1<u64>,   Point1<u64>,   { x });
impl_scalar_vector_mul_ops!(u128,  Point1<u128>,  Point1<u128>,  { x });
impl_scalar_vector_mul_ops!(usize, Point1<usize>, Point1<usize>, { x });
impl_scalar_vector_mul_ops!(i8,    Point1<i8>,    Point1<i8>,    { x });
impl_scalar_vector_mul_ops!(i16,   Point1<i16>,   Point1<i16>,   { x });
impl_scalar_vector_mul_ops!(i32,   Point1<i32>,   Point1<i32>,   { x });
impl_scalar_vector_mul_ops!(i64,   Point1<i64>,   Point1<i64>,   { x });
impl_scalar_vector_mul_ops!(i128,  Point1<i128>,  Point1<i128>,  { x });
impl_scalar_vector_mul_ops!(isize, Point1<isize>, Point1<isize>, { x });
impl_scalar_vector_mul_ops!(f32,   Point1<f32>,   Point1<f32>,   { x });
impl_scalar_vector_mul_ops!(f64,   Point1<f64>,   Point1<f64>,   { x });

impl_point_unary_ops!(Neg, neg, Point1<S>, Point1<S>, { x });

impl_point_binary_assign_ops!(Point1<S>, Vector1<S>, { x });

impl<S> approx::AbsDiffEq for Point1<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        S::abs_diff_eq(&self.x, &other.x, epsilon)
    }
}

impl<S> approx::RelativeEq for Point1<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.x, &other.x, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Point1<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.x, &other.x, epsilon, max_ulps)
    }
}

impl<S> Magnitude for Point1<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn magnitude_squared(&self) -> Self::Output {
        self.dot(self)
    }

    #[inline]
    fn magnitude(&self) -> Self::Output {
        Self::Output::sqrt(self.magnitude_squared())
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
    fn distance_squared(&self, other: &Point1<S>) -> Self::Output {
        (self - other).magnitude_squared()
    }
}


/// A point is a location in a two-dimensional Euclidean space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Point2<S> {
   /// The horizontal coordinate.
   pub x: S,
   /// The vertical coordinate.
   pub y: S,
}

impl<S> Point2<S> {
    /// Construct a new two-dimensional point.
    #[inline]
    pub const fn new(x: S, y: S) -> Point2<S> {
        Point2 { 
            x: x, 
            y: y 
        }
    }

    /// Map an operation on that acts on the coordinates of a point, returning 
    /// a point whose coordinates are of the new scalar type.
    #[inline]
    pub fn map<T, F>(self, mut op: F) -> Point2<T> 
        where F: FnMut(S) -> T 
    {
        Point2 {
            x: op(self.x),
            y: op(self.y),
        }
    }
}

impl<S> Point2<S> where S: Copy {
    /// Expand a two-dimensional point to a three-dimensional point using
    /// the supplied z-value.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #    Point2,
    /// #    Point3,    
    /// # };
    /// #
    /// let point = Point2::new(1_u32, 2_u32);
    /// let expected = Point3::new(1_u32, 2_u32, 3_u32);
    /// let result = point.expand(3_u32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn expand(self, z: S) -> Point3<S> {
        Point3::new(self.x, self.y, z)
    }

    /// Contract a two-dimensional point to a one-dimensional point by
    /// removing its `y`-component.
    ///
    /// ## Example
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
    pub fn contract(self) -> Point1<S> {
        Point1::new(self.x)
    }

    /// Construct a new point from a fill value.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2, 
    /// # };
    /// #
    /// let fill_value = 2_u32;
    /// let expected = Point2::new(2_u32, 2_u32);
    /// let result = Point2::from_fill(fill_value);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_fill(value: S) -> Point2<S> {
        Point2::new(value, value)
    }

    /// The length of the the underlying array.
    #[inline]
    pub fn len() -> usize {
        2
    }

    /// The shape of the underlying array.
    #[inline]
    pub fn shape() -> (usize, usize) {
        (2, 1)
    }

    /// Generate a pointer to the underlying array.
    #[inline]
    pub fn as_ptr(&self) -> *const S {
        &self.x
    }

    /// Generate a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        &mut self.x
    }

    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        <Self as AsRef<[S; 2]>>::as_ref(self)
    }
}

impl<S> Point2<S> where S: NumCast + Copy {
    /// Cast a point of one type of scalars to a point of another type of scalars.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2,   
    /// # };
    /// #
    /// let point: Point2<u32> = Point2::new(1_u32, 2_u32);
    /// let expected: Option<Point2<i32>> = Some(Point2::new(1_i32, 2_i32));
    /// let result = point.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Point2<T>> {
        let x = match num_traits::cast(self.x) {
            Some(value) => value,
            None => return None,
        };
        let y = match num_traits::cast(self.y) {
            Some(value) => value,
            None => return None,
        };

        Some(Point2::new(x, y))
    }
}

impl<S> Point2<S> where S: Scalar {
    /// Convert a homogeneous vector into a point.
    ///
    /// ## Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Point2, 
    /// #     Vector3,
    /// # };
    /// #
    /// let vector = Vector3::new(3_f64, 6_f64, 3_f64);
    /// let expected = Some(Point2::new(1_f64, 2_f64));
    /// let result = Point2::from_homogeneous(vector);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vector_z_zero = Vector3::new(3_f64, 6_f64, 0_f64);
    /// let result = Point2::from_homogeneous(vector_z_zero);
    ///
    /// assert!(result.is_none());
    /// ```
    #[inline]
    pub fn from_homogeneous(vector: Vector3<S>) -> Option<Point2<S>> {
        if !vector.z.is_zero() {
            Some(Point2::new(vector.x / vector.z, vector.y / vector.z))
        } else {
            None
        }
    }

    /// Convert a point to a vector in homogeneous coordinates.
    ///
    /// ## Example
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
    pub fn to_homogeneous(self) -> Vector3<S> {
        Vector3::new(self.x, self.y, S::one())
    }

    /// Compute the origin of the Euclidean vector space.
    #[inline]
    pub fn origin() -> Point2<S> {
        Point2::new(S::zero(), S::zero())
    }

    /// Convert a vector to a point. 
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2,
    /// #     Vector2, 
    /// # };
    /// #
    /// let vector = Vector2::new(1_u32, 2_u32);
    /// let expected = Point2::new(1_u32, 2_u32);
    /// let result = Point2::from_vector(vector);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_vector(vector: Vector2<S>) -> Point2<S> {
        Point2::new(vector.x, vector.y)
    }

    /// Convert a point to a vector.
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2,
    /// #     Vector2, 
    /// # };
    /// #
    /// let point = Point2::new(1_u32, 2_u32);
    /// let expected = Vector2::new(1_u32, 2_u32);
    /// let result = point.to_vector();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_vector(self) -> Vector2<S> {
        Vector2::new(self.x, self.y)
    }

    /// Compute the dot product (inner product) of two points.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2, 
    /// # };
    /// #
    /// let point1 = Point2::new(1_f64, 2_f64);
    /// let point2 = Point2::new(3_f64, 4_f64);
    /// 
    /// assert_eq!(point1.dot(&point2), 11_f64);
    /// ```
    #[inline]
    pub fn dot(self, other: &Point2<S>) -> S {
        self.x * other.x + self.y * other.y
    }
}

impl<S> fmt::Display for Point2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Point2 [{}, {}]", self.x, self.y)
    }
}

impl<S> From<(S, S)> for Point2<S> where S: Scalar {
    #[inline]
    fn from((x, y): (S, S)) -> Point2<S> {
        Point2 { x: x, y: y }
    }
}

impl<S> From<[S; 2]> for Point2<S> where S: Scalar {
    #[inline]
    fn from(v: [S; 2]) -> Point2<S> {
        Point2 { x: v[0], y: v[1] }
    }
}

impl<S> From<&[S; 2]> for Point2<S> where S: Scalar {
    #[inline]
    fn from(v: &[S; 2]) -> Point2<S> {
        Point2 { x: v[0], y: v[1] }
    }
}

impl<'a, S> From<&'a [S; 2]> for &'a Point2<S> where S: Scalar {
    #[inline]
    fn from(v: &'a [S; 2]) -> &'a Point2<S> {
        unsafe { 
            &*(v as *const [S; 2] as *const Point2<S>)
        }
    }
}

impl_as_ref_ops!(Point2<S>, (S, S));
impl_as_ref_ops!(Point2<S>, [S; 2]);

impl_point_index_ops!(Point2<S>, 2, usize, S);
impl_point_index_ops!(Point2<S>, 2, Range<usize>, [S]);
impl_point_index_ops!(Point2<S>, 2, RangeTo<usize>, [S]);
impl_point_index_ops!(Point2<S>, 2, RangeFrom<usize>, [S]);
impl_point_index_ops!(Point2<S>, 2, RangeFull, [S]);

impl<S> ops::Add<Vector2<S>> for Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn add(self, other: Vector2<S>) -> Self::Output {
        Point2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<S> ops::Add<Vector2<S>> for &Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn add(self, other: Vector2<S>) -> Self::Output {
        Point2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<S> ops::Add<&Vector2<S>> for Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn add(self, other: &Vector2<S>) -> Self::Output {
        Point2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<'a, 'b, S> ops::Add<&'b Vector2<S>> for &'a Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn add(self, other: &'b Vector2<S>) -> Self::Output {
        Point2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<S> ops::Sub<Point2<S>> for &Point2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn sub(self, other: Point2<S>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S> ops::Sub<Point2<S>> for Point2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn sub(self, other: Point2<S>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S> ops::Sub<&Point2<S>> for Point2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn sub(self, other: &Point2<S>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,             
        }
    }
}

impl<'a, 'b, S> ops::Sub<&'b Point2<S>> for &'a Point2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn sub(self, other: &'b Point2<S>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S> ops::Sub<Vector2<S>> for &Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn sub(self, other: Vector2<S>) -> Self::Output {
        Point2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S> ops::Sub<Vector2<S>> for Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn sub(self, other: Vector2<S>) -> Self::Output {
        Point2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S> ops::Sub<&Vector2<S>> for Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn sub(self, other: &Vector2<S>) -> Self::Output {
        Point2 {
            x: self.x - other.x,
            y: self.y - other.y,             
        }
    }
}

impl<'a, 'b, S> ops::Sub<&'b Vector2<S>> for &'a Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn sub(self, other: &'b Vector2<S>) -> Self::Output {
        Point2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S> ops::Mul<S> for Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Point2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl<S> ops::Mul<S> for &Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Point2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl<S> ops::Div<S> for Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Point2 {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

impl<S> ops::Div<S> for &Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Point2 {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

impl<S> ops::Rem<S> for Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        
        Point2::new(x, y)
    }
}

impl<S> ops::Rem<S> for &Point2<S> where S: Scalar {
    type Output = Point2<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        
        Point2::new(x, y)
    }
}

impl_scalar_vector_mul_ops!(u8,    Point2<u8>,    Point2<u8>,    { x, y });
impl_scalar_vector_mul_ops!(u16,   Point2<u16>,   Point2<u16>,   { x, y });
impl_scalar_vector_mul_ops!(u32,   Point2<u32>,   Point2<u32>,   { x, y });
impl_scalar_vector_mul_ops!(u64,   Point2<u64>,   Point2<u64>,   { x, y });
impl_scalar_vector_mul_ops!(u128,  Point2<u128>,  Point2<u128>,  { x, y });
impl_scalar_vector_mul_ops!(usize, Point2<usize>, Point2<usize>, { x, y });
impl_scalar_vector_mul_ops!(i8,    Point2<i8>,    Point2<i8>,    { x, y });
impl_scalar_vector_mul_ops!(i16,   Point2<i16>,   Point2<i16>,   { x, y });
impl_scalar_vector_mul_ops!(i32,   Point2<i32>,   Point2<i32>,   { x, y });
impl_scalar_vector_mul_ops!(i64,   Point2<i64>,   Point2<i64>,   { x, y });
impl_scalar_vector_mul_ops!(i128,  Point2<i128>,  Point2<i128>,  { x, y });
impl_scalar_vector_mul_ops!(isize, Point2<isize>, Point2<isize>, { x, y });
impl_scalar_vector_mul_ops!(f32,   Point2<f32>,   Point2<f32>,   { x, y });
impl_scalar_vector_mul_ops!(f64,   Point2<f64>,   Point2<f64>,   { x, y });


impl_point_unary_ops!(Neg, neg, Point2<S>, Point2<S>, { x, y });

impl_point_binary_assign_ops!(Point2<S>, Vector2<S>, { x, y });

impl<S> approx::AbsDiffEq for Point2<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        S::abs_diff_eq(&self.x, &other.x, epsilon) && 
        S::abs_diff_eq(&self.y, &other.y, epsilon)
    }
}

impl<S> approx::RelativeEq for Point2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.x, &other.x, epsilon, max_relative) &&
        S::relative_eq(&self.y, &other.y, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Point2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.x, &other.x, epsilon, max_ulps) &&
        S::ulps_eq(&self.y, &other.y, epsilon, max_ulps)
    }
}

impl<S> Magnitude for Point2<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn magnitude_squared(&self) -> Self::Output {
        self.dot(self)
    }

    #[inline]
    fn magnitude(&self) -> Self::Output {
        Self::Output::sqrt(self.magnitude_squared())
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
    fn distance_squared(&self, other: &Point2<S>) -> Self::Output {
        (self - other).magnitude_squared()
    }
}


/// A representation of three-dimensional points in a Euclidean space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Point3<S> {
    /// The horizontal coordinate.
    pub x: S,
    /// The vertical coordinate.
    pub y: S,
    /// The depth coordinate.
    pub z: S,
}

impl<S> Point3<S> {
    /// Construct a new point in three-dimensional Euclidean space.
    #[inline]
    pub const fn new(x: S, y: S, z: S) -> Point3<S> {
        Point3 { 
            x: x, 
            y: y, 
            z: z 
        }
    }

    /// Map an operation on that acts on the coordinates of a point, returning 
    /// a point whose coordinates are of the new scalar type.
    #[inline]
    pub fn map<T, F>(self, mut op: F) -> Point3<T> where F: FnMut(S) -> T {
        Point3 {
            x: op(self.x),
            y: op(self.y),
            z: op(self.z),
        }
    }
}

impl<S> Point3<S> where S: Copy {
    /// Contract a three-dimensional point, removing its `z`-component.
    ///
    /// ## Example
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
    pub fn contract(self) -> Point2<S> {
        Point2::new(self.x, self.y)
    }

    /// Construct a new point from a fill value.
    /// 
    /// ## Example
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
    pub fn from_fill(value: S) -> Point3<S> {
        Point3::new(value, value, value)
    }

    /// The length of the the underlying array.
    #[inline]
    pub fn len() -> usize {
        3
    }

    /// The shape of the underlying array.
    #[inline]
    pub fn shape() -> (usize, usize) {
        (3, 1)
    }

    /// Generate a pointer to the underlying array.
    #[inline]
    pub fn as_ptr(&self) -> *const S {
        &self.x
    }

    /// Generate a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        &mut self.x
    }

    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        <Self as AsRef<[S; 3]>>::as_ref(self)
    }
}

impl<S> Point3<S> where S: NumCast + Copy {
    /// Cast a point from one type of scalars to another type of scalars.
    ///
    /// ## Example
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
    pub fn cast<T: NumCast>(&self) -> Option<Point3<T>> {
        let x = match num_traits::cast(self.x) {
            Some(value) => value,
            None => return None,
        };
        let y = match num_traits::cast(self.y) {
            Some(value) => value,
            None => return None,
        };
        let z = match num_traits::cast(self.z) {
            Some(value) => value,
            None => return None,
        };

        Some(Point3::new(x, y, z))
    }
}

impl<S> Point3<S> where S: Scalar {
    /// Convert a vector in homogeneous coordinates into a point.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3,
    /// #     Vector4, 
    /// # };
    /// #
    /// let vector = Vector4::new(5_f64, 10_f64, 15_f64, 5_f64);
    /// let expected = Some(Point3::new(1_f64, 2_f64, 3_f64));
    /// let result = Point3::from_homogeneous(vector);
    ///
    /// assert!(result.is_some());
    /// assert_eq!(result, expected);
    ///
    /// let vector_w_zero = Vector4::new(5_f64, 10_f64, 15_f64, 0_f64);
    /// let result = Point3::from_homogeneous(vector_w_zero);
    ///
    /// assert!(result.is_none());
    /// ```
    #[inline]
    pub fn from_homogeneous(vector: Vector4<S>) -> Option<Point3<S>> {
        if !vector.w.is_zero() {
            Some(Point3::new(
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
    /// ## Example
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
    pub fn to_homogeneous(self) -> Vector4<S> {
        Vector4::new(self.x, self.y, self.z, S::one())
    }

    /// Compute the origin of the Euclidean vector space.
    #[inline]
    pub fn origin() -> Point3<S> {
        Point3::new(S::zero(), S::zero(), S::zero())
    }

    /// Convert a vector to a point. 
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// #
    /// let vector = Vector3::new(1_u32, 2_u32, 3_u32);
    /// let expected = Point3::new(1_u32, 2_u32, 3_u32);
    /// let result = Point3::from_vector(vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_vector(v: Vector3<S>) -> Point3<S> {
        Point3::new(v.x, v.y, v.z)
    }

    /// Convert a point to a vector.
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// ## Example
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
    pub fn to_vector(self) -> Vector3<S> {
        Vector3::new(self.x, self.y, self.z)
    }

    /// Compute the dot product (inner product) of two points.
    ///
    /// ## Example
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
    pub fn dot(self, other: &Point3<S>) -> S {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl<S> fmt::Display for Point3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Point3 [{}, {}, {}]", self.x, self.y, self.z)
    }
}

impl<S> From<(S, S, S)> for Point3<S> where S: Scalar {
    #[inline]
    fn from((x, y, z): (S, S, S)) -> Point3<S> {
        Point3::new(x, y, z)
    }
}

impl<S> From<(Point2<S>, S)> for Point3<S> where S: Scalar {
    #[inline]
    fn from((v, z): (Point2<S>, S)) -> Point3<S> {
        Point3::new(v.x, v.y, z)
    }
}

impl<S> From<(&Point2<S>, S)> for Point3<S> where S: Scalar {
    #[inline]
    fn from((v, z): (&Point2<S>, S)) -> Point3<S> {
        Point3::new(v.x, v.y, z)
    }
}

impl<S> From<[S; 3]> for Point3<S> where S: Scalar {
    #[inline]
    fn from(v: [S; 3]) -> Point3<S> {
        Point3::new(v[0], v[1], v[2])
    }
}

impl<'a, S> From<&'a [S; 3]> for &'a Point3<S> where S: Scalar {
    #[inline]
    fn from(v: &'a [S; 3]) -> &'a Point3<S> {
        unsafe { 
            &*(v as *const [S; 3] as *const Point3<S>)
        }
    }
}

impl<'a, S> From<&'a (S, S, S)> for &'a Point3<S> where S: Scalar {
    #[inline]
    fn from(v: &'a (S, S, S)) -> &'a Point3<S> {
        unsafe { 
            &*(v as *const (S, S, S) as *const Point3<S>)
        }
    }
}

impl_as_ref_ops!(Point3<S>, (S, S, S));
impl_as_ref_ops!(Point3<S>, [S; 3]);

impl_point_index_ops!(Point3<S>, 3, usize, S);
impl_point_index_ops!(Point3<S>, 3, Range<usize>, [S]);
impl_point_index_ops!(Point3<S>, 3, RangeTo<usize>, [S]);
impl_point_index_ops!(Point3<S>, 3, RangeFrom<usize>, [S]);
impl_point_index_ops!(Point3<S>, 3, RangeFull, [S]);

impl<S> ops::Add<Vector3<S>> for Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn add(self, other: Vector3<S>) -> Self::Output {
        Point3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<S> ops::Add<Vector3<S>> for &Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn add(self, other: Vector3<S>) -> Self::Output {
        Point3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<S> ops::Add<&Vector3<S>> for Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn add(self, other: &Vector3<S>) -> Self::Output {
        Point3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,               
        }
    }
}

impl<'a, 'b, S> ops::Add<&'b Vector3<S>> for &'a Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn add(self, other: &'b Vector3<S>) -> Self::Output {
        Point3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<S> ops::Sub<Point3<S>> for &Point3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn sub(self, other: Point3<S>) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<S> ops::Sub<Point3<S>> for Point3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn sub(self, other: Point3<S>) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<S> ops::Sub<&Point3<S>> for Point3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn sub(self, other: &Point3<S>) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,               
        }
    }
}

impl<'a, 'b, S> ops::Sub<&'b Point3<S>> for &'a Point3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn sub(self, other: &'b Point3<S>) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<S> ops::Sub<Vector3<S>> for &Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn sub(self, other: Vector3<S>) -> Self::Output {
        Point3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<S> ops::Sub<Vector3<S>> for Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn sub(self, other: Vector3<S>) -> Self::Output {
        Point3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<S> ops::Sub<&Vector3<S>> for Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn sub(self, other: &Vector3<S>) -> Self::Output {
        Point3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,               
        }
    }
}

impl<'a, 'b, S> ops::Sub<&'b Vector3<S>> for &'a Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn sub(self, other: &'b Vector3<S>) -> Self::Output {
        Point3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<S> ops::Mul<S> for Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Point3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl<S> ops::Mul<S> for &Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Point3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl<S> ops::Div<S> for Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Point3 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

impl<S> ops::Div<S> for &Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Point3 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

impl<S> ops::Rem<S> for Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        
        Point3::new(x, y, z)
    }
}

impl<S> ops::Rem<S> for &Point3<S> where S: Scalar {
    type Output = Point3<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        
        Point3::new(x, y, z)
    }
}

impl_scalar_vector_mul_ops!(u8,    Point3<u8>,    Point3<u8>,    { x, y, z });
impl_scalar_vector_mul_ops!(u16,   Point3<u16>,   Point3<u16>,   { x, y, z });
impl_scalar_vector_mul_ops!(u32,   Point3<u32>,   Point3<u32>,   { x, y, z });
impl_scalar_vector_mul_ops!(u64,   Point3<u64>,   Point3<u64>,   { x, y, z });
impl_scalar_vector_mul_ops!(u128,  Point3<u128>,  Point3<u128>,  { x, y, z });
impl_scalar_vector_mul_ops!(usize, Point3<usize>, Point3<usize>, { x, y, z });
impl_scalar_vector_mul_ops!(i8,    Point3<i8>,    Point3<i8>,    { x, y, z });
impl_scalar_vector_mul_ops!(i16,   Point3<i16>,   Point3<i16>,   { x, y, z });
impl_scalar_vector_mul_ops!(i32,   Point3<i32>,   Point3<i32>,   { x, y, z });
impl_scalar_vector_mul_ops!(i64,   Point3<i64>,   Point3<i64>,   { x, y, z });
impl_scalar_vector_mul_ops!(i128,  Point3<i128>,  Point3<i128>,  { x, y, z });
impl_scalar_vector_mul_ops!(isize, Point3<isize>, Point3<isize>, { x, y, z });
impl_scalar_vector_mul_ops!(f32,   Point3<f32>,   Point3<f32>,   { x, y, z });
impl_scalar_vector_mul_ops!(f64,   Point3<f64>,   Point3<f64>,   { x, y, z });

impl_point_unary_ops!(Neg, neg, Point3<S>, Point3<S>, { x, y, z });

impl_point_binary_assign_ops!(Point3<S>, Vector3<S>, { x, y, z });

impl<S> approx::AbsDiffEq for Point3<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        S::abs_diff_eq(&self.x, &other.x, epsilon) && 
        S::abs_diff_eq(&self.y, &other.y, epsilon) &&
        S::abs_diff_eq(&self.z, &other.z, epsilon)
    }
}

impl<S> approx::RelativeEq for Point3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.x, &other.x, epsilon, max_relative) &&
        S::relative_eq(&self.y, &other.y, epsilon, max_relative) &&
        S::relative_eq(&self.z, &other.z, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Point3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.x, &other.x, epsilon, max_ulps) &&
        S::ulps_eq(&self.y, &other.y, epsilon, max_ulps) &&
        S::ulps_eq(&self.z, &other.z, epsilon, max_ulps)
    }
}

impl<S> Magnitude for Point3<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn magnitude_squared(&self) -> Self::Output {
        self.dot(self)
    }

    #[inline]
    fn magnitude(&self) -> Self::Output {
        Self::Output::sqrt(self.magnitude_squared())
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
    fn distance_squared(&self, other: &Point3<S>) -> Self::Output {
        (self - other).magnitude_squared()
    }
}

