use crate::scalar::{
    Scalar,
    ScalarSigned,
    ScalarFloat,   
};
use crate::magnitude::{
    Magnitude,
};
use num_traits::{
    NumCast,
};

use core::fmt;
use core::iter;
use core::ops;


macro_rules! impl_mul_operator {
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


/// A representation of one-dimensional vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Vector1<S> {
    pub x: S,
}

impl<S> Vector1<S> {
    /// Construct a new vector.
    #[inline]
    pub const fn new(x: S) -> Vector1<S> {
        Vector1 { 
            x: x 
        }
    }

    /// Map an operation on the elements of a vector, returning a vector of the 
    /// new underlying type.
    #[inline]
    pub fn map<T, F>(self, mut op: F) -> Vector1<T> where F: FnMut(S) -> T {
        Vector1 { 
            x: op(self.x) 
        }
    }
}

impl<S> Vector1<S> where S: NumCast + Copy {
    /// Cast a vector from one type of scalars to another type of scalars.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1,   
    /// # };
    /// #
    /// let vector: Vector1<u32> = Vector1::new(1_u32);
    /// let expected: Option<Vector1<i32>> = Some(Vector1::new(1_i32));
    /// let result = vector.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Vector1<T>> {
        let x = match num_traits::cast(self.x) {
            Some(value) => value,
            None => return None,
        };

        Some(Vector1::new(x))
    }
}

impl<S> Vector1<S> where S: Copy {
    /// Extend a one-dimensional vector into a two-dimensional vector using 
    /// the supplied value for the `y`-component.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1,
    /// #     Vector2,   
    /// # };
    /// #
    /// let vector = Vector1::new(1_f64);
    /// let expected = Vector2::new(1_f64, 2_f64);
    /// let result = vector.expand(2_f64);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn expand(self, y: S) -> Vector2<S> {
        Vector2::new(self.x, y)
    }

    /// Construct a vector from a fill value.
    ///
    /// Every component of the resulting vector will have the same value
    /// supplied by the `value` argument.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1, 
    /// # };
    /// #
    /// let fill_value = 3_f64;
    /// let result = Vector1::from_fill(fill_value);
    /// let expected = Vector1::new(3_f64);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_fill(value: S) -> Vector1<S> {
        Vector1::new(value)
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

    /// Generate a mutable pointer to the underlying array.
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

impl<S> Vector1<S> where S: Scalar {
    /// Returns the **x-axis** unit vector, a unit vector with the `x`-component
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_x() -> Vector1<S> {
        Vector1::new(S::one())
    }

    /// Compute the zero vector.
    ///
    /// The zero vector is the vector in which all of its elements are zero.
    #[inline]
    pub fn zero() -> Vector1<S> {
        Vector1::new(S::zero())
    }
    
    /// Determine whether a vector is the zero vector.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.x.is_zero()
    }

    /// Compute the coordinates of a vector in projective space.
    ///
    /// The function appends a `0` to the vector.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1,
    /// #     Vector2, 
    /// # };
    /// #
    /// let vector = Vector1::new(1_i32);
    /// let expected = Vector2::new(1_i32, 0_i32);
    /// let result = vector.to_homogeneous();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_homogeneous(&self) -> Vector2<S> {
        self.expand(S::zero())
    }
    
    /// Compute the Euclidean dot product (inner product) of two vectors.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1, 
    /// # };
    /// #
    /// let vector1 = Vector1::new(1_f64);
    /// let vector2 = Vector1::new(2_f64);
    /// 
    /// assert_eq!(vector1.dot(&vector2), 2_f64);
    /// ```
    #[inline]
    pub fn dot(self, other: &Vector1<S>) -> S {
        self.x * other.x
    }
}

impl<S> Vector1<S> where S: ScalarSigned {
    /// Compute the negation of a vector mutably in place.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1, 
    /// # };
    /// #
    /// let mut result = Vector1::new(1_i32);
    /// let expected = -result;
    /// result.neg_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn neg_mut(&mut self) {
        self.x = -self.x;
    }
}

impl<S> Vector1<S> where S: ScalarFloat {
    /// Linearly interpolate between the two vectors `self` and `other`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1,
    /// # };
    /// #
    /// let v0 = Vector1::new(0_f64);
    /// let v1 = Vector1::new(10_f64);
    /// let amount = 0.6;
    /// let result = v0.lerp(&v1, amount);
    /// let expected = Vector1::new(6_f64);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Vector1<S>, amount: S) -> Vector1<S> {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of this vector are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A vector is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values.
    ///
    /// ## Example (Finite Vector)
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1,  
    /// # };
    /// #
    /// let v: Vector1<f64> = Vector1::new(2_f64);
    ///
    /// assert!(v.is_finite()); 
    /// ```
    ///
    /// ## Example (Infinite Vector)
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1,  
    /// # };
    /// #
    /// let w: Vector1<f64> = Vector1::new(f64::INFINITY);
    ///
    /// assert!(!w.is_finite()); 
    /// ```
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite()
    }

    /// Compute the projection of the vector `self` onto the vector
    /// `other`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1, 
    /// #     Magnitude,
    /// # };
    /// # 
    /// let vector = Vector1::new(1_f64);
    /// let unit_x = Vector1::unit_x();
    /// let projected_x = vector.project(&unit_x);
    ///
    /// assert_eq!(projected_x, vector.x * unit_x);
    /// ```
    #[inline]
    pub fn project(&self, other: &Vector1<S>) -> Vector1<S> {
        other * (self.dot(other) / other.magnitude_squared())
    }
}

impl<S> AsRef<[S; 1]> for Vector1<S> {
    #[inline]
    fn as_ref(&self) -> &[S; 1] {
        unsafe { 
            &*(self as *const Vector1<S> as *const [S; 1])
        }
    }
}

impl<S> AsRef<S> for Vector1<S> {
    #[inline]
    fn as_ref(&self) -> &S {
        unsafe { 
            &*(self as *const Vector1<S> as *const S)
        }
    }
}

impl<S> AsRef<(S,)> for Vector1<S> {
    #[inline]
    fn as_ref(&self) -> &(S,) {
        unsafe { 
            &*(self as *const Vector1<S> as *const (S,))
        }
    }
}

impl<S> AsMut<[S; 1]> for Vector1<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [S; 1] {
        unsafe { 
            &mut *(self as *mut Vector1<S> as *mut [S; 1])
        }
    }
}

impl<S> AsMut<S> for Vector1<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut S {
        unsafe { 
            &mut *(self as *mut Vector1<S> as *mut S)
        }
    }
}

impl<S> AsMut<(S,)> for Vector1<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut (S,) {
        unsafe { 
            &mut *(self as *mut Vector1<S> as *mut (S,))
        }
    }
}

impl<S> ops::Index<usize> for Vector1<S> {
    type Output = S;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[S; 1] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::Range<usize>> for Vector1<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        let v: &[S; 1] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeTo<usize>> for Vector1<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        let v: &[S; 1] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeFrom<usize>> for Vector1<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        let v: &[S; 1] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeFull> for Vector1<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeFull) -> &Self::Output {
        let v: &[S; 1] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::IndexMut<usize> for Vector1<S> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut S {
        let v: &mut [S; 1] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::Range<usize>> for Vector1<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut [S] {
        let v: &mut [S; 1] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeTo<usize>> for Vector1<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut [S] {
        let v: &mut [S; 1] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFrom<usize>> for Vector1<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut [S] {
        let v: &mut [S; 1] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFull> for Vector1<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFull) -> &mut [S] {
        let v: &mut [S; 1] = self.as_mut();
        &mut v[index]
    }
}

impl<S> fmt::Display for Vector1<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Vector1 [{}]", self.x)
    }
}

impl<S> From<S> for Vector1<S> where S: Scalar {
    #[inline]
    fn from(v: S) -> Vector1<S> {
        Vector1 { 
            x: v,
        }
    }
}

impl<S> From<[S; 1]> for Vector1<S> where S: Scalar {
    #[inline]
    fn from(v: [S; 1]) -> Vector1<S> {
        Vector1 { 
            x: v[0],
        }
    }
}

impl<S> From<&[S; 1]> for Vector1<S> where S: Scalar {
    #[inline]
    fn from(v: &[S; 1]) -> Vector1<S> {
        Vector1 { 
            x: v[0],
        }
    }
}

impl<'a, S> From<&'a [S; 1]> for &'a Vector1<S> where S: Scalar {
    #[inline]
    fn from(v: &'a [S; 1]) -> &'a Vector1<S> {
        unsafe { 
            &*(v as *const [S; 1] as *const Vector1<S>)
        }
    }
}

impl<S> ops::Neg for Vector1<S> where S: ScalarSigned {
    type Output = Vector1<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector1 { 
            x: -self.x,
        }
    }
}

impl<S> ops::Neg for &Vector1<S> where S: ScalarSigned {
    type Output = Vector1<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector1 { 
            x: -self.x,
        }
    }
}


impl<S> ops::Add<Vector1<S>> for &Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn add(self, other: Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x + other.x,
        }
    }
}

impl<S> ops::Add<Vector1<S>> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn add(self, other: Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x + other.x,
        }
    }
}

impl<S> ops::Add<&Vector1<S>> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn add(self, other: &Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x + other.x,
        }
    }
}

impl<'a, 'b, S> ops::Add<&'b Vector1<S>> for &'a Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn add(self, other: &'b Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x + other.x,
        }
    }
}

impl<S> ops::Sub<Vector1<S>> for &Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn sub(self, other: Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl<S> ops::Sub<Vector1<S>> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn sub(self, other: Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl<S> ops::Sub<&Vector1<S>> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn sub(self, other: &Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl<'a, 'b, S> ops::Sub<&'b Vector1<S>> for &'a Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn sub(self, other: &'b Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl<S> ops::AddAssign<Vector1<S>> for Vector1<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: Vector1<S>) {
        self.x = self.x + other.x;
    }
}

impl<S> ops::AddAssign<&Vector1<S>> for Vector1<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: &Vector1<S>) {
        self.x = self.x + other.x;
    }
}

impl<S> ops::SubAssign<Vector1<S>> for Vector1<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: Vector1<S>) {
        self.x = self.x - other.x;
    }
}

impl<S> ops::SubAssign<&Vector1<S>> for Vector1<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: &Vector1<S>) {
        self.x = self.x - other.x;
    }
}

impl<S> ops::Mul<S> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Vector1 {
            x: self.x * other,
        }
    }
}

impl<S> ops::Mul<S> for &Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Vector1 {
            x: self.x * other,
        }
    }
}

impl<S> ops::MulAssign<S> for Vector1<S> where S: Scalar {
    #[inline]
    fn mul_assign(&mut self, other: S) {
        self.x *= other;
    }
}

impl<S> ops::Div<S> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Vector1 {
            x: self.x / other,
        }
    }
}

impl<S> ops::Div<S> for &Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Vector1 {
            x: self.x / other,
        }
    }
}

impl<S> ops::DivAssign<S> for Vector1<S> where S: Scalar {
    #[inline]
    fn div_assign(&mut self, other: S) {
        self.x = self.x / other;
    }
}

impl<S> ops::Rem<S> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        
        Vector1::new(x)
    }
}

impl<S> ops::Rem<S> for &Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        
        Vector1::new(x)
    }
}

impl<S> ops::RemAssign<S> for Vector1<S> where S: Scalar {
    #[inline]
    fn rem_assign(&mut self, other: S) {
        self.x %= other;
    }
}

impl<S> Magnitude for Vector1<S> where S: ScalarFloat {
    type Output = S;
    
    #[inline]
    fn magnitude_squared(&self) -> Self::Output {
        self.dot(self)
    }

    #[inline]
    fn magnitude(&self) -> Self::Output {
        S::sqrt(self.magnitude_squared())
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
    fn distance_squared(&self, other: &Vector1<S>) -> Self::Output {
        (other.x - self.x) * (other.x - self.x)
    }
}

impl<S> approx::AbsDiffEq for Vector1<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Vector1<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.x, &other.x, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Vector1<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.x, &other.x, epsilon, max_ulps)
    }
}

impl<S: Scalar> iter::Sum<Vector1<S>> for Vector1<S> {
    #[inline]
    fn sum<I: Iterator<Item=Vector1<S>>>(iter: I) -> Vector1<S> {
        iter.fold(Vector1::zero(), ops::Add::add)
    }
}

impl<'a, S: 'a + Scalar> iter::Sum<&'a Vector1<S>> for Vector1<S> {
    #[inline]
    fn sum<I: Iterator<Item=&'a Vector1<S>>>(iter: I) -> Vector1<S> {
        iter.fold(Vector1::zero(), ops::Add::add)
    }
}

impl_mul_operator!(u8,    Vector1<u8>,    Vector1<u8>,    { x });
impl_mul_operator!(u16,   Vector1<u16>,   Vector1<u16>,   { x });
impl_mul_operator!(u32,   Vector1<u32>,   Vector1<u32>,   { x });
impl_mul_operator!(u64,   Vector1<u64>,   Vector1<u64>,   { x });
impl_mul_operator!(u128,  Vector1<u128>,  Vector1<u128>,  { x });
impl_mul_operator!(usize, Vector1<usize>, Vector1<usize>, { x });
impl_mul_operator!(i8,    Vector1<i8>,    Vector1<i8>,    { x });
impl_mul_operator!(i16,   Vector1<i16>,   Vector1<i16>,   { x });
impl_mul_operator!(i32,   Vector1<i32>,   Vector1<i32>,   { x });
impl_mul_operator!(i64,   Vector1<i64>,   Vector1<i64>,   { x });
impl_mul_operator!(i128,  Vector1<i128>,  Vector1<i128>,  { x });
impl_mul_operator!(isize, Vector1<isize>, Vector1<isize>, { x });
impl_mul_operator!(f32,   Vector1<f32>,   Vector1<f32>,   { x });
impl_mul_operator!(f64,   Vector1<f64>,   Vector1<f64>,   { x });


/// A representation of two-dimensional vectors in a Euclidean space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Vector2<S> {
    /// The horizontal component.
    pub x: S,
    /// The vertical component.
    pub y: S,
}

impl<S> Vector2<S> {
    /// Construct a new vector.
    #[inline]
    pub const fn new(x: S, y: S) -> Vector2<S> {
        Vector2 { 
            x: x, 
            y: y 
        }
    }

    /// Map an operation on the elements of a vector, returning a vector of the 
    /// new underlying type.
    #[inline]
    pub fn map<T, F>(self, mut op: F) -> Vector2<T> where F: FnMut(S) -> T {
        Vector2 {
            x: op(self.x),
            y: op(self.y),
        }
    }
}

impl<S> Vector2<S> where S: NumCast + Copy {
    /// Cast a vector from one type of scalars to another type of scalars.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector2,   
    /// # };
    /// #
    /// let vector: Vector2<u32> = Vector2::new(1_u32, 2_u32);
    /// let expected: Option<Vector2<i32>> = Some(Vector2::new(1_i32, 2_i32));
    /// let result = vector.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Vector2<T>> {
        let x = match num_traits::cast(self.x) {
            Some(value) => value,
            None => return None,
        };
        let y = match num_traits::cast(self.y) {
            Some(value) => value,
            None => return None,
        };

        Some(Vector2::new(x, y))
    }
}

impl<S> Vector2<S> where S: Copy {
    /// Extend a two-dimensional vector into a three-dimensional vector using the 
    /// supplied `z`-component.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector2,
    /// #     Vector3,   
    /// # };
    /// #
    /// let v = Vector2::new(1_f64, 2_f64);
    /// let expected = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let result = v.expand(3_f64);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn expand(self, z: S) -> Vector3<S> {
        Vector3::new(self.x, self.y, z)
    }

    /// Contract a two-dimensional vector to a one-dimensional vector by removing
    /// the `y`-component.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1,
    /// #     Vector2,   
    /// # };
    /// #
    /// let v = Vector2::new(1_f64, 2_f64);
    /// let expected = Vector1::new(1_f64);
    /// let result = v.contract();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn contract(self) -> Vector1<S> {
        Vector1::new(self.x)
    }

    /// Construct a vector from a fill value.
    ///
    /// Every component of the resulting vector will have the same value
    /// supplied by the `value` argument.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector2, 
    /// # };
    /// #
    /// let fill_value = 3_f64;
    /// let result = Vector2::from_fill(fill_value);
    /// let expected = Vector2::new(3_f64, 3_f64);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_fill(value: S) -> Vector2<S> {
        Vector2::new(value, value)
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

impl<S> Vector2<S> where S: Scalar {
    /// Returns the **x-axis** unit vector, a unit vector with the `x`-component
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_x() -> Vector2<S> {
        Vector2::new(S::one(), S::zero())
    }

    /// Returns the **y-axis** unit vector, a unit vector with the `y`-component
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_y() -> Vector2<S> {
        Vector2::new(S::zero(), S::one())
    }

    /// Compute the zero vector.
    ///
    /// The zero vector is the vector in which all of its elements are zero.
    #[inline]
    pub fn zero() -> Vector2<S> {
        Vector2::new(S::zero(), S::zero())
    }
    
    /// Determine whether a vector is the zero vector.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }

    /// Compute the coordinates of a vector in projective space.
    ///
    /// The function appends a `0` to the vector.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector2,
    /// #     Vector3, 
    /// # };
    /// #
    /// let vector = Vector2::new(1_i32, 2_i32);
    /// let expected = Vector3::new(1_i32, 2_i32, 0_i32);
    /// let result = vector.to_homogeneous();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_homogeneous(&self) -> Vector3<S> {
        self.expand(S::zero())
    }

    /// Compute the coordinates of a projective vector in Euclidean space.
    ///
    /// The function removes a `0` from the end of the vector, otherwise it
    /// returns `None`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1,
    /// #     Vector2, 
    /// # };
    /// #
    /// let vector = Vector2::new(1_i32, 0_i32);
    /// let expected = Some(Vector1::new(1_i32));
    /// let result = vector.from_homogeneous();
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vector = Vector2::new(1_i32, 1_i32);
    /// let expected: Option<Vector1<i32>> = None;
    /// let result = vector.from_homogeneous();
    ///
    /// assert!(result.is_none());
    /// ```
    #[inline]
    pub fn from_homogeneous(&self) -> Option<Vector1<S>> {
        if self.y.is_zero() {
            Some(self.contract())
        } else {
            None
        }
    }

    /// Compute the Euclidean dot product (inner product) of two vectors.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector2, 
    /// # };
    /// #
    /// let vector1 = Vector2::new(1_f64, 2_f64);
    /// let vector2 = Vector2::new(3_f64, 4_f64);
    /// 
    /// assert_eq!(vector1.dot(&vector2), 11_f64);
    /// ```
    #[inline]
    pub fn dot(self, other: &Vector2<S>) -> S {
        self.x * other.x + self.y * other.y
    }
}

impl<S> Vector2<S> where S: ScalarSigned {
    /// Compute the negation of a vector mutably in place.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector2, 
    /// # };
    /// #
    /// let mut result = Vector2::new(1_i32, 2_i32);
    /// let expected = -result;
    /// result.neg_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn neg_mut(&mut self) {
        self.x = -self.x;
        self.y = -self.y;
    }
}

impl<S> Vector2<S> where S: ScalarFloat {
    /// Linearly interpolate between the two vectors `self` and `other`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector2,  
    /// # };
    /// #
    /// let v0 = Vector2::new(0_f64, 0_f64);
    /// let v1 = Vector2::new(10_f64, 20_f64);
    /// let amount = 0.7;
    /// let expected = Vector2::new(7_f64, 14_f64);
    /// let result = v0.lerp(&v1, amount);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Vector2<S>, amount: S) -> Vector2<S> {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of a vector are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A vector is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values.
    ///
    /// ## Example (Finite Vector)
    ///
    /// ```
    /// # use cglinalg::{
    /// #    Vector2,
    /// # };
    /// #
    /// let v = Vector2::new(1_f64, 2_f64);
    ///
    /// assert!(v.is_finite());
    /// ```
    ///
    /// ## Example (Not A Finite Vector)
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector2,
    /// # };
    /// #
    /// let w1 = Vector2::new(f64::INFINITY, f64::NAN);
    /// let w2 = Vector2::new(f64::INFINITY, 2_f64);
    ///
    /// assert!(!w1.is_finite());
    /// assert!(!w2.is_finite());
    /// ```
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Compute the projection of the vector `self` onto the vector
    /// `other`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector2, 
    /// #     Magnitude,
    /// # };
    /// # 
    /// let vector = 3_f64 * Vector2::new(1_f64 / 2_f64, f64::sqrt(3_f64) / 2_f64);
    /// let unit_x = Vector2::unit_x();
    /// let unit_y = Vector2::unit_y();
    /// let projected_x = vector.project(&unit_x);
    /// let projected_y = vector.project(&unit_y);
    ///
    /// assert_eq!(projected_x, vector.x * unit_x);
    /// assert_eq!(projected_y, vector.y * unit_y);
    /// ```
    #[inline]
    pub fn project(&self, other: &Vector2<S>) -> Vector2<S> {
        other * (self.dot(other) / other.magnitude_squared())
    }
}

impl<S> AsRef<[S; 2]> for Vector2<S> {
    #[inline]
    fn as_ref(&self) -> &[S; 2] {
        unsafe { 
            &*(self as *const Vector2<S> as *const [S; 2])
        }
    }
}

impl<S> AsRef<(S, S)> for Vector2<S> {
    #[inline]
    fn as_ref(&self) -> &(S, S) {
        unsafe { 
            &*(self as *const Vector2<S> as *const (S, S))
        }
    }
}

impl<S> AsMut<[S; 2]> for Vector2<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [S; 2] {
        unsafe { 
            &mut *(self as *mut Vector2<S> as *mut [S; 2])
        }
    }
}

impl<S> AsMut<(S, S)> for Vector2<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut (S, S) {
        unsafe { 
            &mut *(self as *mut Vector2<S> as *mut (S, S))
        }
    }
}

impl<S> ops::Index<usize> for Vector2<S> {
    type Output = S;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[S; 2] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::Range<usize>> for Vector2<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        let v: &[S; 2] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeTo<usize>> for Vector2<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        let v: &[S; 2] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeFrom<usize>> for Vector2<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        let v: &[S; 2] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeFull> for Vector2<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeFull) -> &Self::Output {
        let v: &[S; 2] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::IndexMut<usize> for Vector2<S> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut S {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::Range<usize>> for Vector2<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut [S] {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeTo<usize>> for Vector2<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut [S] {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFrom<usize>> for Vector2<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut [S] {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFull> for Vector2<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFull) -> &mut [S] {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> fmt::Display for Vector2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Vector2 [{}, {}]", self.x, self.y)
    }
}

impl<S> From<(S, S)> for Vector2<S> where S: Scalar {
    #[inline]
    fn from((x, y): (S, S)) -> Vector2<S> {
        Vector2 { 
            x: x, 
            y: y,
        }
    }
}

impl<S> From<[S; 2]> for Vector2<S> where S: Scalar {
    #[inline]
    fn from(v: [S; 2]) -> Vector2<S> {
        Vector2 { 
            x: v[0], 
            y: v[1],
        }
    }
}

impl<S> From<&[S; 2]> for Vector2<S> where S: Scalar {
    #[inline]
    fn from(v: &[S; 2]) -> Vector2<S> {
        Vector2 {
            x: v[0], 
            y: v[1],
        }
    }
}

impl<'a, S> From<&'a [S; 2]> for &'a Vector2<S> where S: Scalar {
    #[inline]
    fn from(v: &'a [S; 2]) -> &'a Vector2<S> {
        unsafe { 
            &*(v as *const [S; 2] as *const Vector2<S>)
        }
    }
}

impl<S> ops::Neg for Vector2<S> where S: ScalarSigned {
    type Output = Vector2<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector2 { 
            x: -self.x, 
            y: -self.y,
        }
    }
}

impl<S> ops::Neg for &Vector2<S> where S: ScalarSigned {
    type Output = Vector2<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector2 { 
            x: -self.x, 
            y: -self.y,
        }
    }
}

impl<S> ops::Add<Vector2<S>> for &Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn add(self, other: Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<S> ops::Add<Vector2<S>> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn add(self, other: Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<S> ops::Add<&Vector2<S>> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn add(self, other: &Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<'a, 'b, S> ops::Add<&'b Vector2<S>> for &'a Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn add(self, other: &'b Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<S> ops::Sub<Vector2<S>> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn sub(self, other: Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S> ops::Sub<Vector2<S>> for &Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn sub(self, other: Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S> ops::Sub<&Vector2<S>> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn sub(self, other: &Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<'a, 'b, S> ops::Sub<&'b Vector2<S>> for &'a Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn sub(self, other: &'b Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S> ops::AddAssign<Vector2<S>> for Vector2<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: Vector2<S>) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
    }
}

impl<S> ops::AddAssign<&Vector2<S>> for Vector2<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: &Vector2<S>) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
    }
}

impl<S> ops::SubAssign<Vector2<S>> for Vector2<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: Vector2<S>) {
        self.x = self.x - other.x;
        self.y = self.y - other.y;
    }
}

impl<S> ops::SubAssign<&Vector2<S>> for Vector2<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: &Vector2<S>) {
        self.x = self.x - other.x;
        self.y = self.y - other.y;
    }
}

impl<S> ops::Mul<S> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Vector2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl<S> ops::Mul<S> for &Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Vector2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl_mul_operator!(u8,    Vector2<u8>,    Vector2<u8>,    { x, y });
impl_mul_operator!(u16,   Vector2<u16>,   Vector2<u16>,   { x, y });
impl_mul_operator!(u32,   Vector2<u32>,   Vector2<u32>,   { x, y });
impl_mul_operator!(u64,   Vector2<u64>,   Vector2<u64>,   { x, y });
impl_mul_operator!(u128,  Vector2<u128>,  Vector2<u128>,  { x, y });
impl_mul_operator!(usize, Vector2<usize>, Vector2<usize>, { x, y });
impl_mul_operator!(i8,    Vector2<i8>,    Vector2<i8>,    { x, y });
impl_mul_operator!(i16,   Vector2<i16>,   Vector2<i16>,   { x, y });
impl_mul_operator!(i32,   Vector2<i32>,   Vector2<i32>,   { x, y });
impl_mul_operator!(i64,   Vector2<i64>,   Vector2<i64>,   { x, y });
impl_mul_operator!(i128,  Vector2<i128>,  Vector2<i128>,  { x, y });
impl_mul_operator!(isize, Vector2<isize>, Vector2<isize>, { x, y });
impl_mul_operator!(f32,   Vector2<f32>,   Vector2<f32>,   { x, y });
impl_mul_operator!(f64,   Vector2<f64>,   Vector2<f64>,   { x, y });

impl<S> ops::MulAssign<S> for Vector2<S> where S: Scalar {
    #[inline]
    fn mul_assign(&mut self, other: S) {
        self.x *= other;
        self.y *= other;
    }
}

impl<S> ops::Div<S> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Vector2 {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

impl<S> ops::Div<S> for &Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Vector2 {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

impl<S> ops::DivAssign<S> for Vector2<S> where S: Scalar {
    #[inline]
    fn div_assign(&mut self, other: S) {
        self.x = self.x / other;
        self.y = self.y / other;
    }
}

impl<S> ops::Rem<S> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        
        Vector2::new(x, y)
    }
}

impl<S> ops::Rem<S> for &Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        
        Vector2::new(x, y)
    }
}

impl<S> ops::RemAssign<S> for Vector2<S> where S: Scalar {
    #[inline]
    fn rem_assign(&mut self, other: S) {
        self.x %= other;
        self.y %= other;
    }
}

impl<S> Magnitude for Vector2<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn magnitude(&self) -> Self::Output {
        Self::Output::sqrt(self.magnitude_squared())
    }

    #[inline]
    fn magnitude_squared(&self) -> Self::Output {
        self.dot(&self)
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
    fn distance_squared(&self, to: &Vector2<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
    
        dx_squared + dy_squared
    }
}

impl<S> approx::AbsDiffEq for Vector2<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Vector2<S> where S: ScalarFloat {
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

impl<S> approx::UlpsEq for Vector2<S> where S: ScalarFloat {
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

impl<S: Scalar> iter::Sum<Vector2<S>> for Vector2<S> {
    #[inline]
    fn sum<I: Iterator<Item=Vector2<S>>>(iter: I) -> Vector2<S> {
        iter.fold(Vector2::zero(), ops::Add::add)
    }
}

impl<'a, S: 'a + Scalar> iter::Sum<&'a Vector2<S>> for Vector2<S> {
    #[inline]
    fn sum<I: Iterator<Item=&'a Vector2<S>>>(iter: I) -> Vector2<S> {
        iter.fold(Vector2::zero(), ops::Add::add)
    }
}


/// A representation of three-dimensional vectors in a Euclidean space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Vector3<S> {
    pub x: S,
    pub y: S,
    pub z: S,
}

impl<S> Vector3<S> {
    /// Construct a new vector.
    #[inline]
    pub const fn new(x: S, y: S, z: S) -> Vector3<S> {
        Vector3 { x: x, y: y, z: z }
    }

    /// Map an operation on the elements of a vector, returning a vector of the 
    /// new underlying type.
    #[inline]
    pub fn map<T, F>(self, mut op: F) -> Vector3<T> where F: FnMut(S) -> T {
        Vector3 {
            x: op(self.x),
            y: op(self.y),
            z: op(self.z),
        }
    }
}

impl<S> Vector3<S> where S: NumCast + Copy {
    /// Cast a vector from one type of scalars to another type of scalars.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,   
    /// # };
    /// #
    /// let vector: Vector3<u32> = Vector3::new(1_u32, 2_u32, 3_u32);
    /// let expected: Option<Vector3<i32>> = Some(Vector3::new(1_i32, 2_i32, 3_i32));
    /// let result = vector.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Vector3<T>> {
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

        Some(Vector3::new(x, y, z))
    }
}

impl<S> Vector3<S> where S: Copy {
    /// Extend a three-dimensional vector into a four-dimensional vector using the 
    /// supplied `w`-component.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,
    /// #     Vector4, 
    /// # };
    /// #
    /// let v = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
    /// let result = v.expand(4_f64);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn expand(self, w: S) -> Vector4<S> {
        Vector4::new(self.x, self.y, self.z, w)
    }

    /// Contract a three-dimensional vector to a two-dimensional vector
    /// by removing the `z`-component.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector2,
    /// #     Vector3, 
    /// # };
    /// #
    /// let v = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Vector2::new(1_f64, 2_f64);
    /// let result = v.contract();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn contract(self) -> Vector2<S> {
        Vector2::new(self.x, self.y)
    }

    /// Construct a vector from a fill value.
    ///
    /// Every component of the resulting vector will have the same value
    /// supplied by the `value` argument.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,   
    /// # };
    /// #
    /// let fill_value = 3_f64;
    /// let result = Vector3::from_fill(fill_value);
    /// let expected = Vector3::new(3_f64, 3_f64, 3_f64);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_fill(value: S) -> Vector3<S> {
        Vector3::new(value, value, value)
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

impl<S> Vector3<S> where S: Scalar {
    /// Returns the **x-axis** unit vector, a unit vector with the `x`-component
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_x() -> Vector3<S> {
        Vector3 { 
            x: S::one(), 
            y: S::zero(), 
            z: S::zero(),
        }
    }

    /// Returns the **y-axis** unit vector, a unit vector with the `y`-component
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_y() -> Vector3<S> {
        Vector3 { 
            x: S::zero(), 
            y: S::one(), 
            z: S::zero(),
        }
    }
    
    /// Returns the **z-axis** unit vector, a unit vector with the `z`-component
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_z() -> Vector3<S> {
        Vector3 { 
            x: S::zero(),
            y: S::zero(), 
            z: S::one(),
        }
    }

    /// Compute the zero vector.
    ///
    /// The zero vector is the vector in which all of its elements are zero.
    #[inline]
    pub fn zero() -> Vector3<S> {
        Vector3::new(S::zero(), S::zero(), S::zero())
    }
    
    /// Determine whether a vector is the zero vector.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.x.is_zero() && self.y.is_zero() && self.z.is_zero()
    }

    /// Compute the coordinates of a vector in projective space.
    ///
    /// The function appends a `0` to the vector.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,
    /// #     Vector4, 
    /// # };
    /// #
    /// let vector = Vector3::new(1_i32, 2_i32, 3_i32);
    /// let expected = Vector4::new(1_i32, 2_i32, 3_i32, 0_i32);
    /// let result = vector.to_homogeneous();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_homogeneous(&self) -> Vector4<S> {
        self.expand(S::zero())
    }

    /// Compute the coordinates of a projective vector in Euclidean space.
    ///
    /// The function removes a `0` from the end of the vector, otherwise it
    /// returns `None`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector2,
    /// #     Vector3, 
    /// # };
    /// #
    /// let vector = Vector3::new(1_i32, 2_i32, 0_i32);
    /// let expected = Some(Vector2::new(1_i32, 2_i32));
    /// let result = vector.from_homogeneous();
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vector = Vector3::new(1_i32, 2_i32, 1_i32);
    /// let expected: Option<Vector2<i32>> = None;
    /// let result = vector.from_homogeneous();
    ///
    /// assert!(result.is_none());
    /// ```
    #[inline]
    pub fn from_homogeneous(&self) -> Option<Vector2<S>> {
        if self.z.is_zero() {
            Some(self.contract())
        } else {
            None
        }
    }

    /// Compute the Euclidean dot product (inner product) of two vectors.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3, 
    /// # };
    /// #
    /// let vector1 = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let vector2 = Vector3::new(4_f64, 5_f64, 6_f64);
    /// 
    /// assert_eq!(vector1.dot(&vector2), 32_f64);
    /// ```
    #[inline]
    pub fn dot(self, other: &Vector3<S>) -> S {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl<S> Vector3<S> where S: ScalarSigned {
    /// Compute the negation of a vector mutably in place.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3, 
    /// # };
    /// #
    /// let mut result = Vector3::new(1_i32, 2_i32, 3_i32);
    /// let expected = -result;
    /// result.neg_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn neg_mut(&mut self) {
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
    }

    /// Compute the cross product of two three-dimensional vectors. 
    ///
    /// For the vector dimensions used in computer graphics 
    /// (up to four dimensions), the cross product is well-defined only in 
    /// three dimensions. The cross product is a form of vector 
    /// multiplication that computes a vector normal to the plane swept out by 
    /// the two vectors. The magnitude of this vector is the area of the 
    /// parallelogram swept out by thw two vectors. 
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,  
    /// # };
    /// #
    /// let vector1 = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let vector2 = Vector3::new(4_f64, 5_f64, 6_f64);
    /// let cross = vector1.cross(&vector2);
    ///
    /// assert_eq!(cross, Vector3::new(-3_f64, 6_f64, -3_f64));
    /// // The cross product is perpendicular to the two input vectors.
    /// assert_eq!(cross.dot(&vector1), 0_f64);
    /// assert_eq!(cross.dot(&vector2), 0_f64);
    ///
    /// // The cross product of a vector with itself is zero.
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    ///
    /// assert_eq!(vector.cross(&vector), Vector3::zero());
    /// ```
    #[inline]
    pub fn cross(&self, other: &Vector3<S>) -> Self {
        let x = self.y * other.z - self.z * other.y;
        let y = self.z * other.x - self.x * other.z;
        let z = self.x * other.y - self.y * other.x;
    
        Vector3::new(x, y, z)
    }
}

impl<S> Vector3<S> where S: ScalarFloat {
    /// Linearly interpolate between the two vectors `self` and `other`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,  
    /// # };
    /// #
    /// let v0 = Vector3::new(0_f64, 0_f64, 0_f64);
    /// let v1 = Vector3::new(10_f64, 20_f64, 30_f64);
    /// let amount = 0.7;
    /// let expected = Vector3::new(7_f64, 14_f64, 21_f64);
    /// let result = v0.lerp(&v1, amount);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Vector3<S>, amount: S) -> Vector3<S> {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of a vector are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A vector is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values.
    ///
    /// ## Example (Finite Vector)
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,
    /// # };
    /// #
    /// let v = Vector3::new(1_f64, 2_f64, 3_f64);
    ///
    /// assert!(v.is_finite()); 
    /// ```
    ///
    /// ## Example (Not A Finite Vector)
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,
    /// # };
    /// #
    /// let w = Vector3::new(1_f64, f64::NAN, f64::NEG_INFINITY);
    ///
    /// assert!(!w.is_finite()); 
    /// ```
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Compute the projection of the vector `self` onto the vector
    /// `other`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3, 
    /// #     Magnitude,
    /// # };
    /// # 
    /// let vector = Vector3::new(1_f64 / 2_f64, f64::sqrt(3_f64) / 2_f64, 2_f64);
    /// let unit_x = Vector3::unit_x();
    /// let unit_y = Vector3::unit_y();
    /// let unit_z = Vector3::unit_z();
    /// let projected_x = vector.project(&unit_x);
    /// let projected_y = vector.project(&unit_y);
    /// let projected_z = vector.project(&unit_z);
    ///
    /// assert_eq!(projected_x, vector.x * unit_x);
    /// assert_eq!(projected_y, vector.y * unit_y);
    /// assert_eq!(projected_z, vector.z * unit_z);
    /// ```
    #[inline]
    pub fn project(&self, other: &Vector3<S>) -> Vector3<S> {
        other * (self.dot(other) / other.magnitude_squared())
    }
}

impl<S> AsRef<[S; 3]> for Vector3<S> {
    #[inline]
    fn as_ref(&self) -> &[S; 3] {
        unsafe { 
            &*(self as *const Vector3<S> as *const [S; 3])
        }
    }
}

impl<S> AsRef<(S, S, S)> for Vector3<S> {
    #[inline]
    fn as_ref(&self) -> &(S, S, S) {
        unsafe { 
            &*(self as *const Vector3<S> as *const (S, S, S))
        }
    }
}

impl<S> AsMut<[S; 3]> for Vector3<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [S; 3] {
        unsafe { 
            &mut *(self as *mut Vector3<S> as *mut [S; 3])
        }
    }
}

impl<S> AsMut<(S, S, S)> for Vector3<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut (S, S, S) {
        unsafe { 
            &mut *(self as *mut Vector3<S> as *mut (S, S, S))
        }
    }
}

impl<S> ops::Index<usize> for Vector3<S> {
    type Output = S;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[S; 3] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::Range<usize>> for Vector3<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        let v: &[S; 3] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeTo<usize>> for Vector3<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        let v: &[S; 3] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeFrom<usize>> for Vector3<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        let v: &[S; 3] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeFull> for Vector3<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeFull) -> &Self::Output {
        let v: &[S; 3] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::IndexMut<usize> for Vector3<S> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut S {
        let v: &mut [S; 3] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::Range<usize>> for Vector3<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut [S] {
        let v: &mut [S; 3] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeTo<usize>> for Vector3<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut [S] {
        let v: &mut [S; 3] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFrom<usize>> for Vector3<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut [S] {
        let v: &mut [S; 3] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFull> for Vector3<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFull) -> &mut [S] {
        let v: &mut [S; 3] = self.as_mut();
        &mut v[index]
    }
}

impl<S> fmt::Display for Vector3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Vector3 [{}, {}, {}]", self.x, self.y, self.z)
    }
}

impl<S> From<(S, S, S)> for Vector3<S> where S: Scalar {
    #[inline]
    fn from((x, y, z): (S, S, S)) -> Vector3<S> {
        Vector3::new(x, y, z)
    }
}

impl<S> From<(Vector2<S>, S)> for Vector3<S> where S: Scalar {
    #[inline]
    fn from((v, z): (Vector2<S>, S)) -> Vector3<S> {
        Vector3::new(v.x, v.y, z)
    }
}

impl<S> From<(&Vector2<S>, S)> for Vector3<S> where S: Scalar {
    #[inline]
    fn from((v, z): (&Vector2<S>, S)) -> Vector3<S> {
        Vector3::new(v.x, v.y, z)
    }
}

impl<S> From<[S; 3]> for Vector3<S> where S: Scalar {
    #[inline]
    fn from(v: [S; 3]) -> Vector3<S> {
        Vector3::new(v[0], v[1], v[2])
    }
}

impl<S> From<Vector4<S>> for Vector3<S> where S: Scalar {
    #[inline]
    fn from(v: Vector4<S>) -> Vector3<S> {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl<S> From<&Vector4<S>> for Vector3<S> where S: Scalar {
    #[inline]
    fn from(v: &Vector4<S>) -> Vector3<S> {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl<'a, S> From<&'a [S; 3]> for &'a Vector3<S> where S: Scalar {
    #[inline]
    fn from(v: &'a [S; 3]) -> &'a Vector3<S> {
        unsafe { 
            &*(v as *const [S; 3] as *const Vector3<S>)
        }
    }
}

impl<'a, S> From<&'a (S, S, S)> for &'a Vector3<S> where S: Scalar {
    #[inline]
    fn from(v: &'a (S, S, S)) -> &'a Vector3<S> {
        unsafe { 
            &*(v as *const (S, S, S) as *const Vector3<S>)
        }
    }
}

impl<S> ops::Neg for Vector3<S> where S: ScalarSigned {
    type Output = Vector3<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector3 { 
            x: -self.x,
            y: -self.y, 
            z: -self.z,
        }
    }
}

impl<S> ops::Neg for &Vector3<S> where S: ScalarSigned {
    type Output = Vector3<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector3 { 
            x: -self.x, 
            y: -self.y, 
            z: -self.z,
        }
    }
}

impl<S> ops::Add<Vector3<S>> for &Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn add(self, other: Vector3<S>) -> Self::Output {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<S> ops::Add<Vector3<S>> for Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn add(self, other: Vector3<S>) -> Self::Output {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<S> ops::Add<&Vector3<S>> for Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn add(self, other: &Vector3<S>) -> Self::Output {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,               
        }
    }
}

impl<'a, 'b, S> ops::Add<&'b Vector3<S>> for &'a Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn add(self, other: &'b Vector3<S>) -> Self::Output {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<S> ops::Sub<Vector3<S>> for &Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn sub(self, other: Vector3<S>) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<S> ops::Sub<Vector3<S>> for Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn sub(self, other: Vector3<S>) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<S> ops::Sub<&Vector3<S>> for Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn sub(self, other: &Vector3<S>) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,               
        }
    }
}

impl<'a, 'b, S> ops::Sub<&'b Vector3<S>> for &'a Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn sub(self, other: &'b Vector3<S>) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}


impl<S> ops::AddAssign<Vector3<S>> for Vector3<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: Vector3<S>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<S> ops::AddAssign<&Vector3<S>> for Vector3<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: &Vector3<S>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<S> ops::SubAssign<Vector3<S>> for Vector3<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: Vector3<S>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<S> ops::SubAssign<&Vector3<S>> for Vector3<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: &Vector3<S>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<S> ops::Mul<S> for Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Vector3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl<S> ops::Mul<S> for &Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Vector3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl_mul_operator!(u8,    Vector3<u8>,    Vector3<u8>,    { x, y, z });
impl_mul_operator!(u16,   Vector3<u16>,   Vector3<u16>,   { x, y, z });
impl_mul_operator!(u32,   Vector3<u32>,   Vector3<u32>,   { x, y, z });
impl_mul_operator!(u64,   Vector3<u64>,   Vector3<u64>,   { x, y, z });
impl_mul_operator!(u128,  Vector3<u128>,  Vector3<u128>,  { x, y, z });
impl_mul_operator!(usize, Vector3<usize>, Vector3<usize>, { x, y, z });
impl_mul_operator!(i8,    Vector3<i8>,    Vector3<i8>,    { x, y, z });
impl_mul_operator!(i16,   Vector3<i16>,   Vector3<i16>,   { x, y, z });
impl_mul_operator!(i32,   Vector3<i32>,   Vector3<i32>,   { x, y, z });
impl_mul_operator!(i64,   Vector3<i64>,   Vector3<i64>,   { x, y, z });
impl_mul_operator!(i128,  Vector3<i128>,  Vector3<i128>,  { x, y, z });
impl_mul_operator!(isize, Vector3<isize>, Vector3<isize>, { x, y, z });
impl_mul_operator!(f32,   Vector3<f32>,   Vector3<f32>,   { x, y, z });
impl_mul_operator!(f64,   Vector3<f64>,   Vector3<f64>,   { x, y, z });

impl<S> ops::MulAssign<S> for Vector3<S> where S: Scalar {
    #[inline]
    fn mul_assign(&mut self, other: S) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
    }
}

impl<S> ops::Div<S> for Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Vector3 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

impl<S> ops::Div<S> for &Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Vector3 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

impl<S> ops::DivAssign<S> for Vector3<S> where S: Scalar {
    #[inline]
    fn div_assign(&mut self, other: S) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
    }
}

impl<S> ops::Rem<S> for Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        
        Vector3::new(x, y, z)
    }
}

impl<S> ops::Rem<S> for &Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        
        Vector3::new(x, y, z)
    }
}

impl<S> ops::RemAssign<S> for Vector3<S> where S: Scalar {
    #[inline]
    fn rem_assign(&mut self, other: S) {
        self.x %= other;
        self.y %= other;
        self.z %= other;
    }
}

impl<S> Magnitude for Vector3<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn magnitude(&self) -> Self::Output {
        Self::Output::sqrt(self.magnitude_squared())
    }

    #[inline]
    fn magnitude_squared(&self) -> Self::Output {
        self.dot(self)
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
    fn distance_squared(&self, to: &Vector3<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
        let dz_squared = (to.z - self.z) * (to.z - self.z);
    
        dx_squared + dy_squared + dz_squared
    }
}

impl<S> approx::AbsDiffEq for Vector3<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Vector3<S> where S: ScalarFloat {
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

impl<S> approx::UlpsEq for Vector3<S> where S: ScalarFloat {
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

impl<S: Scalar> iter::Sum<Vector3<S>> for Vector3<S> {
    #[inline]
    fn sum<I: Iterator<Item=Vector3<S>>>(iter: I) -> Vector3<S> {
        iter.fold(Vector3::zero(), ops::Add::add)
    }
}

impl<'a, S: 'a + Scalar> iter::Sum<&'a Vector3<S>> for Vector3<S> {
    #[inline]
    fn sum<I: Iterator<Item=&'a Vector3<S>>>(iter: I) -> Vector3<S> {
        iter.fold(Vector3::zero(), ops::Add::add)
    }
}


/// A representation of four-dimensional vectors in a Euclidean space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Vector4<S> {
    pub x: S,
    pub y: S,
    pub z: S,
    pub w: S,
}

impl<S> Vector4<S> {
    /// Construct a new four-dimensional vector.
    #[inline]
    pub const fn new(x: S, y: S, z: S, w: S) -> Vector4<S> {
        Vector4 { 
            x: x, 
            y: y, 
            z: z, 
            w: w,
        }
    }

    /// Map an operation on the elements of a vector, returning a vector of the 
    /// new underlying type.
    #[inline]
    pub fn map<T, F>(self, mut op: F) -> Vector4<T> where F: FnMut(S) -> T {
        Vector4 {
            x: op(self.x),
            y: op(self.y),
            z: op(self.z),
            w: op(self.w),
        }
    }
}

impl<S> Vector4<S> where S: Copy {
    /// Contract a four-dimensional vector to a three-dimensional vector
    /// by removing the `w`-component.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,
    /// #     Vector4, 
    /// # };
    /// #
    /// let v = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
    /// let expected = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let result = v.contract();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn contract(self) -> Vector3<S> {
        Vector3::new(self.x, self.y, self.z)
    }

    /// Construct a vector from a fill value.
    ///
    /// Every component of the resulting vector will have the same value
    /// supplied by the `value` argument.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4,   
    /// # };
    /// #
    /// let fill_value = 3_f64;
    /// let result = Vector4::from_fill(fill_value);
    /// let expected = Vector4::new(3_f64, 3_f64, 3_f64, 3_f64);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_fill(value: S) -> Vector4<S> {
        Vector4::new(value, value, value, value)
    }

    /// The length of the the underlying array.
    #[inline]
    pub fn len() -> usize {
        4
    }

    /// The shape of the underlying array.
    #[inline]
    pub fn shape() -> (usize, usize) {
        (4, 1)
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
        <Self as AsRef<[S; 4]>>::as_ref(self)
    }
}

impl<S> Vector4<S> where S: NumCast + Copy {
    /// Cast a vector from one type of scalars to another type of scalars.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4,   
    /// # };
    /// #
    /// let vector: Vector4<u32> = Vector4::new(1_u32, 2_u32, 3_u32, 4_u32);
    /// let expected: Option<Vector4<i32>> = Some(Vector4::new(1_i32, 2_i32, 3_i32, 4_i32));
    /// let result = vector.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Vector4<T>> {
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
        let w = match num_traits::cast(self.w) {
            Some(value) => value,
            None => return None,
        };

        Some(Vector4::new(x, y, z, w))
    }
}

impl<S> Vector4<S> where S: Scalar {
    /// Returns the **x-axis** unit vector, a unit vector with the `x`-component
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_x() -> Vector4<S> {
        Vector4::new(S::one(), S::zero(), S::zero(), S::zero())
    }

    /// Returns the **y-axis** unit vector, a unit vector with the `y`-component
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_y() -> Vector4<S> {
        Vector4::new(S::zero(), S::one(), S::zero(), S::zero())
    }
    
    /// Returns the **z-axis** unit vector, a unit vector with the `z`-component
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_z() -> Vector4<S> {
        Vector4::new(S::zero(), S::zero(), S::one(), S::zero())
    }

    /// Returns the **w-axis** unit vector, a unit vector with the `w`-component
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_w() -> Vector4<S> {
        Vector4::new(S::zero(), S::zero(), S::zero(), S::one())
    }

    /// Compute the zero vector.
    ///
    /// The zero vector is the vector in which all of its elements are zero.
    #[inline]
    pub fn zero() -> Vector4<S> {
        Vector4::new(S::zero(), S::zero(), S::zero(), S::zero())
    }
    
    /// Determine whether a vector is the zero vector.
    #[inline]
    pub fn is_zero(&self) -> bool {
            self.x.is_zero() && self.y.is_zero() && self.z.is_zero() && self.w.is_zero()
    }

    /// Compute the coordinates of a projective vector in Euclidean space.
    ///
    /// The function removes a `0` from the end of the vector, otherwise it
    /// returns `None`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,
    /// #     Vector4, 
    /// # };
    /// #
    /// let vector = Vector4::new(1_i32, 2_i32, 3_i32, 0_i32);
    /// let expected = Some(Vector3::new(1_i32, 2_i32, 3_i32));
    /// let result = vector.from_homogeneous();
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vector = Vector4::new(1_i32, 2_i32, 3_i32, 1_i32);
    /// let expected: Option<Vector3<i32>> = None;
    /// let result = vector.from_homogeneous();
    ///
    /// assert!(result.is_none());
    /// ```
    #[inline]
    pub fn from_homogeneous(&self) -> Option<Vector3<S>> {
        if self.w.is_zero() {
            Some(self.contract())
        } else {
            None
        }
    }
    
    /// Compute the Euclidean dot product (inner product) of two vectors.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4, 
    /// # };
    /// #
    /// let vector1 = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
    /// let vector2 = Vector4::new(5_f64, 6_f64, 7_f64, 8_f64);
    /// 
    /// assert_eq!(vector1.dot(&vector2), 70_f64);
    /// ```
    #[inline]
    pub fn dot(self, other: &Vector4<S>) -> S {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl<S> Vector4<S> where S: ScalarSigned {
    /// Compute the negation of a vector mutably in place.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4, 
    /// # };
    /// #
    /// let mut result = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
    /// let expected = -result;
    /// result.neg_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn neg_mut(&mut self) {
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
        self.w = -self.w;
    }
}

impl<S> Vector4<S> where S: ScalarFloat {
    /// Linearly interpolate between the two vectors `self` and `other`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4,  
    /// # };
    /// #
    /// let v0 = Vector4::new(0_f64, 0_f64, 0_f64, 0_f64);
    /// let v1 = Vector4::new(10_f64, 20_f64, 30_f64, 40_f64);
    /// let amount = 0.7;
    /// let expected = Vector4::new(7_f64, 14_f64, 21_f64, 28_f64);
    /// let result = v0.lerp(&v1, amount);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Vector4<S>, amount: S) -> Vector4<S> {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of a vector are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A vector is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values.
    ///
    /// ## Example (Finite Vector)
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4,
    /// # };
    /// #
    /// let v = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
    ///
    /// assert!(v.is_finite()); 
    /// ```
    ///
    /// ## Example (Not A Finite Vector)
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Vector4,
    /// # };
    /// #
    /// let w = Vector4::new(1_f64, f64::NAN, f64::NEG_INFINITY, 4_f64);
    ///
    /// assert!(!w.is_finite()); 
    /// ```
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }

    /// Compute the projection of the vector `self` onto the vector
    /// `other`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4, 
    /// #     Magnitude,
    /// # };
    /// # 
    /// let vector = Vector4::new(1_f64 / 2_f64, f64::sqrt(3_f64) / 2_f64, 2_f64, 1_f64);
    /// let unit_x = Vector4::unit_x();
    /// let unit_y = Vector4::unit_y();
    /// let unit_z = Vector4::unit_z();
    /// let unit_w = Vector4::unit_w();
    /// let projected_x = vector.project(&unit_x);
    /// let projected_y = vector.project(&unit_y);
    /// let projected_z = vector.project(&unit_z);
    /// let projected_w = vector.project(&unit_w);
    ///
    /// assert_eq!(projected_x, vector.x * unit_x);
    /// assert_eq!(projected_y, vector.y * unit_y);
    /// assert_eq!(projected_z, vector.z * unit_z);
    /// assert_eq!(projected_w, vector.w * unit_w);
    /// ```
    #[inline]
    pub fn project(&self, other: &Vector4<S>) -> Vector4<S> {
        other * (self.dot(other) / other.magnitude_squared())
    }
}

impl<S> AsRef<[S; 4]> for Vector4<S> {
    #[inline]
    fn as_ref(&self) -> &[S; 4] {
        unsafe { 
            &*(self as *const Vector4<S> as *const [S; 4])
        }
    }
}

impl<S> AsRef<(S, S, S, S)> for Vector4<S> {
    #[inline]
    fn as_ref(&self) -> &(S, S, S, S) {
        unsafe { 
            &*(self as *const Vector4<S> as *const (S, S, S, S))
        }
    }
}

impl<S> AsMut<[S; 4]> for Vector4<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [S; 4] {
        unsafe { 
            &mut *(self as *mut Vector4<S> as *mut [S; 4])
        }
    }
}

impl<S> AsMut<(S, S, S, S)> for Vector4<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut (S, S, S, S) {
        unsafe { 
            &mut *(self as *mut Vector4<S> as *mut (S, S, S, S)) 
        }
    }
}

impl<S> ops::Index<usize> for Vector4<S> {
    type Output = S;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[S; 4] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::Range<usize>> for Vector4<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        let v: &[S; 4] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeTo<usize>> for Vector4<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        let v: &[S; 4] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeFrom<usize>> for Vector4<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        let v: &[S; 4] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeFull> for Vector4<S> {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeFull) -> &Self::Output {
        let v: &[S; 4] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::IndexMut<usize> for Vector4<S> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut S {
        let v: &mut [S; 4] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::Range<usize>> for Vector4<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut [S] {
        let v: &mut [S; 4] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeTo<usize>> for Vector4<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut [S] {
        let v: &mut [S; 4] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFrom<usize>> for Vector4<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut [S] {
        let v: &mut [S; 4] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFull> for Vector4<S> {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFull) -> &mut [S] {
        let v: &mut [S; 4] = self.as_mut();
        &mut v[index]
    }
}

impl<S> From<(S, S, S, S)> for Vector4<S> where S: Scalar {
    #[inline]
    fn from((x, y, z, w): (S, S, S, S)) -> Vector4<S> {
        Vector4::new(x, y, z, w)
    }
}

impl<S> From<(Vector2<S>, S, S)> for Vector4<S> where S: Scalar {
    #[inline]
    fn from((v, z, w): (Vector2<S>, S, S)) -> Vector4<S> {
        Vector4::new(v.x, v.y, z, w)
    }
}

impl<S> From<(&Vector2<S>, S, S)> for Vector4<S> where S: Scalar {
    #[inline]
    fn from((v, z, w): (&Vector2<S>, S, S)) -> Vector4<S> {
        Vector4::new(v.x, v.y, z, w)
    }
}

impl<S> From<(Vector3<S>, S)> for Vector4<S> where S: Scalar {
    #[inline]
    fn from((v, w): (Vector3<S>, S)) -> Vector4<S> {
        Vector4::new(v.x, v.y, v.z, w)
    }
}

impl<S> From<(&Vector3<S>, S)> for Vector4<S> where S: Scalar {
    #[inline]
    fn from((v, w): (&Vector3<S>, S)) -> Vector4<S> {
        Vector4::new(v.x, v.y, v.z, w)
    }
}

impl<S> From<[S; 4]> for Vector4<S> where S: Scalar {
    #[inline]
    fn from(v: [S; 4]) -> Vector4<S> {
        Vector4::new(v[0], v[1], v[2], v[3])
    }
}

impl<S> From<&[S; 4]> for Vector4<S> where S: Scalar {
    #[inline]
    fn from(v: &[S; 4]) -> Vector4<S> {
        Vector4::new(v[0], v[1], v[2], v[3])
    }
}

impl<S> From<&(S, S, S, S)> for Vector4<S> where S: Scalar {
    #[inline]
    fn from(v: &(S, S, S, S)) -> Vector4<S> {
        Vector4::new(v.0, v.1, v.2, v.3)
    }
}

impl<'a, S> From<&'a [S; 4]> for &'a Vector4<S> where S: Scalar {
    #[inline]
    fn from(v: &'a [S; 4]) -> &'a Vector4<S> {
        unsafe { 
            &*(v as *const [S; 4] as *const Vector4<S>)    
        }
    }
}

impl<'a, S> From<&'a (S, S, S, S)> for &'a Vector4<S> where S: Scalar {
    #[inline]
    fn from(v: &'a (S, S, S, S)) -> &'a Vector4<S> {
        unsafe { 
            &*(v as *const (S, S, S, S) as *const Vector4<S>)
        }
    }
}

impl<S> fmt::Display for Vector4<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Vector4 [{}, {}, {}, {}]", self.x, self.y, self.z, self.w)
    }
}

impl<S> ops::Neg for Vector4<S> where S: ScalarSigned {
    type Output = Vector4<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector4 { 
            x: -self.x,
            y: -self.y,
            z: -self.z, 
            w: -self.w,
        }
    }
}

impl<S> ops::Neg for &Vector4<S> where S: ScalarSigned {
    type Output = Vector4<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector4 { 
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

impl<S> ops::Add<Vector4<S>> for &Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn add(self, other: Vector4<S>) -> Self::Output {
        Vector4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl<S> ops::Add<Vector4<S>> for Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn add(self, other: Vector4<S>) -> Self::Output {
        Vector4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl<S> ops::Add<&Vector4<S>> for Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn add(self, other: &Vector4<S>) -> Self::Output {
        Vector4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,   
            w: self.w + other.w,            
        }
    }
}

impl<'a, 'b, S> ops::Add<&'a Vector4<S>> for &'b Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn add(self, other: &'a Vector4<S>) -> Self::Output {
        Vector4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl<S> ops::Sub<Vector4<S>> for &Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn sub(self, other: Vector4<S>) -> Self::Output {
        Vector4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl<S> ops::Sub<Vector4<S>> for Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn sub(self, other: Vector4<S>) -> Self::Output {
        Vector4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl<S> ops::Sub<&Vector4<S>> for Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn sub(self, other: &Vector4<S>) -> Self::Output {
        Vector4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl<'a, 'b, S> ops::Sub<&'b Vector4<S>> for &'a Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn sub(self, other: &'b Vector4<S>) -> Self::Output {
        Vector4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl<S> ops::AddAssign<Vector4<S>> for Vector4<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: Vector4<S>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl<S> ops::AddAssign<&Vector4<S>> for Vector4<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: &Vector4<S>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl<S> ops::SubAssign<Vector4<S>> for Vector4<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: Vector4<S>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl<S> ops::SubAssign<&Vector4<S>> for Vector4<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: &Vector4<S>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl<S> ops::Mul<S> for Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Vector4 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
            w: self.w * other,
        }
    }
}

impl<S> ops::Mul<S> for &Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Vector4 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
            w: self.w * other,
        }
    }
}

impl_mul_operator!(u8,    Vector4<u8>,    Vector4<u8>,    { x, y, z, w });
impl_mul_operator!(u16,   Vector4<u16>,   Vector4<u16>,   { x, y, z, w });
impl_mul_operator!(u32,   Vector4<u32>,   Vector4<u32>,   { x, y, z, w });
impl_mul_operator!(u64,   Vector4<u64>,   Vector4<u64>,   { x, y, z, w });
impl_mul_operator!(u128,  Vector4<u128>,  Vector4<u128>,  { x, y, z, w });
impl_mul_operator!(usize, Vector4<usize>, Vector4<usize>, { x, y, z, w });
impl_mul_operator!(i8,    Vector4<i8>,    Vector4<i8>,    { x, y, z, w });
impl_mul_operator!(i16,   Vector4<i16>,   Vector4<i16>,   { x, y, z, w });
impl_mul_operator!(i32,   Vector4<i32>,   Vector4<i32>,   { x, y, z, w });
impl_mul_operator!(i64,   Vector4<i64>,   Vector4<i64>,   { x, y, z, w });
impl_mul_operator!(i128,  Vector4<i128>,  Vector4<i128>,  { x, y, z, w });
impl_mul_operator!(isize, Vector4<isize>, Vector4<isize>, { x, y, z, w });
impl_mul_operator!(f32,   Vector4<f32>,   Vector4<f32>,   { x, y, z, w });
impl_mul_operator!(f64,   Vector4<f64>,   Vector4<f64>,   { x, y, z, w });

impl<S> ops::MulAssign<S> for Vector4<S> where S: Scalar {
    #[inline]
    fn mul_assign(&mut self, other: S) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
        self.w *= other;
    }
}

impl<S> ops::Div<S> for Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Vector4 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
            w: self.w / other,
        }
    }
}

impl<S> ops::Div<S> for &Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Vector4 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
            w: self.w / other,
        }
    }
}

impl<S> ops::DivAssign<S> for Vector4<S> where S: Scalar {
    #[inline]
    fn div_assign(&mut self, other: S) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
        self.w /= other;
    }
}

impl<S> ops::Rem<S> for Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        let w = self.w % other;

        Vector4::new(x, y, z, w)
    }
}

impl<S> ops::Rem<S> for &Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        let w = self.w % other;
        
        Vector4::new(x, y, z, w)
    }
}

impl<S> ops::RemAssign<S> for Vector4<S> where S: Scalar {
    #[inline]
    fn rem_assign(&mut self, other: S) {
        self.x %= other;
        self.y %= other;
        self.z %= other;
        self.w %= other;
    }
}

impl<S> Magnitude for Vector4<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn magnitude(&self) -> Self::Output {
        Self::Output::sqrt(self.magnitude_squared())
    }

    #[inline]
    fn magnitude_squared(&self) -> Self::Output {
        self.dot(self)
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
    fn distance_squared(&self, to: &Vector4<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
        let dz_squared = (to.z - self.z) * (to.z - self.z);
        let dw_squared = (to.w - self.w) * (to.w - self.w);

        dx_squared + dy_squared + dz_squared + dw_squared
    }
}

impl<S> approx::AbsDiffEq for Vector4<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        S::abs_diff_eq(&self.x, &other.x, epsilon) && 
        S::abs_diff_eq(&self.y, &other.y, epsilon) &&
        S::abs_diff_eq(&self.z, &other.z, epsilon) &&
        S::abs_diff_eq(&self.w, &other.w, epsilon)
    }
}

impl<S> approx::RelativeEq for Vector4<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.x, &other.x, epsilon, max_relative) &&
        S::relative_eq(&self.y, &other.y, epsilon, max_relative) &&
        S::relative_eq(&self.z, &other.z, epsilon, max_relative) &&
        S::relative_eq(&self.w, &other.w, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Vector4<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.x, &other.x, epsilon, max_ulps) &&
        S::ulps_eq(&self.y, &other.y, epsilon, max_ulps) &&
        S::ulps_eq(&self.z, &other.z, epsilon, max_ulps) &&
        S::ulps_eq(&self.w, &other.w, epsilon, max_ulps)
    }
}

impl<S: Scalar> iter::Sum<Vector4<S>> for Vector4<S> {
    #[inline]
    fn sum<I: Iterator<Item=Vector4<S>>>(iter: I) -> Vector4<S> {
        iter.fold(Vector4::zero(), ops::Add::add)
    }
}

impl<'a, S: 'a + Scalar> iter::Sum<&'a Vector4<S>> for Vector4<S> {
    #[inline]
    fn sum<I: Iterator<Item=&'a Vector4<S>>>(iter: I) -> Vector4<S> {
        iter.fold(Vector4::zero(), ops::Add::add)
    }
}

