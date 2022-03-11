use crate::common::{
    Scalar,
    ScalarSigned,
    ScalarFloat,   
};
use crate::common::{
    Magnitude,
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


pub type Vector1<S> = Vector<S, 1>;
pub type Vector2<S> = Vector<S, 2>;
pub type Vector3<S> = Vector<S, 3>;
pub type Vector4<S> = Vector<S, 4>;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]

pub struct Vector<S, const N: usize> {
    data: [S; N],
}

impl<S, const N: usize> Vector<S, N> {
    /// The length of the the underlying array storing the vector components.
    #[inline]
    pub const fn len(&self) -> usize {
        N
    }

    /// The shape of the underlying array storing the vector components.
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
        &self.data[0]
    }

    /// Get a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        &mut self.data[0]
    }

    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        <Self as AsRef<[S; N]>>::as_ref(self)
    }
}

impl<S, const N: usize> AsRef<[S; N]> for Vector<S, N> {
    #[inline]
    fn as_ref(&self) -> &[S; N] {
        unsafe {
            &*(self as *const Vector<S, N> as *const [S; N])
        }
    }
}

impl<S, const N: usize> AsMut<[S; N]> for Vector<S, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [S; N] {
        unsafe {
            &mut *(self as *mut Vector<S, N> as *mut [S; N])
        }
    }
}

impl<S, const N: usize> AsRef<[[S; N]; 1]> for Vector<S, N> {
    #[inline]
    fn as_ref(&self) -> &[[S; N]; 1] {
        unsafe {
            &*(self as *const Vector<S, N> as *const [[S; N]; 1])
        }
    }
}

impl<S, const N: usize> AsMut<[[S; N]; 1]> for Vector<S, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [[S; N]; 1] {
        unsafe {
            &mut *(self as *mut Vector<S, N> as *mut [[S; N]; 1])
        }
    }
}

impl<'a, S, const N: usize> From<&'a [S; N]> for &'a Vector<S, N> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &'a [S; N]) -> &'a Vector<S, N> {
        unsafe { 
            &*(v as *const [S; N] as *const Vector<S, N>)
        }
    }
}


impl<S> Vector1<S> {
    /// Construct a new vector.
    #[inline]
    pub const fn new(x: S) -> Self {
        Self { 
            data: [x], 
        }
    }
}

impl<S> Vector1<S> 
where 
    S: NumCast + Copy
{
    /// Cast a vector from one type of scalars to another type of scalars.
    ///
    /// # Example
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
        let x = match num_traits::cast(self.data[0]) {
            Some(value) => value,
            None => return None,
        };

        Some(Vector1::new(x))
    }
}

impl<S> Vector1<S> 
where 
    S: Copy
{
    /// Extend a one-dimensional vector into a two-dimensional vector using 
    /// the supplied value for the **y-component**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1,
    /// #     Vector2,   
    /// # };
    /// #
    /// let vector = Vector1::new(1_f64);
    /// let expected = Vector2::new(1_f64, 2_f64);
    /// let result = vector.extend(2_f64);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn extend(&self, y: S) -> Vector2<S> {
        Vector2::new(self.data[0], y)
    }

    /// Construct a vector from a fill value.
    ///
    /// Every component of the resulting vector will have the same value
    /// supplied by the `value` argument.
    ///
    /// # Example
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
    pub fn from_fill(value: S) -> Self {
        Self::new(value)
    }

    /// Map an operation on that acts on the coordinates of a vector, returning 
    /// a vector whose coordinates are of the new scalar type.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector1,  
    /// # };
    /// #
    /// let vector: Vector1<u32> = Vector1::new(1_u32);
    /// let expected: Vector1<i32> = Vector1::new(2_i32);
    /// let result: Vector1<i32> = vector.map(|comp| (comp + 1) as i32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn map<T, F>(&self, mut op: F) -> Vector1<T> 
    where 
        F: FnMut(S) -> T
    {
        Vector1::new(op(self.data[0]))
    }
}

impl<S> Vector1<S> 
where 
    S: Scalar
{
    /// Returns the **x-axis** unit vector, a unit vector with the **x-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_x() -> Self {
        Vector1::new(S::one())
    }

    /// Compute the zero vector.
    ///
    /// The zero vector is the vector in which all of its elements are zero.
    #[inline]
    pub fn zero() -> Self {
        Vector1::new(S::zero())
    }
    
    /// Determine whether a vector is the zero vector.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.data[0].is_zero()
    }

    /// Compute the coordinates of a vector in projective space.
    ///
    /// The function appends a `0` to the vector.
    ///
    /// # Example
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
        self.extend(S::zero())
    }
    
    /// Compute the Euclidean dot product (inner product) of two vectors.
    ///
    /// # Example
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
    pub fn dot(&self, other: &Self) -> S {
        self.data[0] * other.data[0]
    }
}

impl<S> Vector1<S> 
where 
    S: ScalarSigned
{
    /// Compute the negation of a vector mutably in place.
    ///
    /// # Example
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
        self.data[0] = -self.data[0];
    }
}

impl<S> Vector1<S> 
where 
    S: ScalarFloat
{
    /// Linearly interpolate between the two vectors `self` and `other`.
    ///
    /// # Example
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
    pub fn lerp(&self, other: &Self, amount: S) -> Self {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of this vector are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A vector is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values.
    ///
    /// # Example (Finite Vector)
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
    /// # Example (Infinite Vector)
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
        self.data[0].is_finite()
    }

    /// Compute the projection of the vector `self` onto the vector
    /// `other`.
    ///
    /// # Example
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
    pub fn project(&self, other: &Self) -> Self {
        other * (self.dot(other) / other.magnitude_squared())
    }
}

impl<S> fmt::Display for Vector1<S> 
where 
    S: fmt::Display
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Vector1 [{}]", 
            self.data[0]
        )
    }
}

impl<S> Default for Vector1<S>
where
    S: Scalar
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<S> From<S> for Vector1<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: S) -> Self {
        Self::new(v)
    }
}

impl<S> From<(S,)> for Vector1<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: (S,)) -> Self {
        Self::new(v.0)
    }
}

impl<S> From<&(S,)> for Vector1<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &(S,)) -> Self  {
        Self::new(v.0)
    }
}

impl<S> From<[S; 1]> for Vector1<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: [S; 1]) -> Self {
        Self::new(v[0])
    }
}

impl<S> From<&[S; 1]> for Vector1<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &[S; 1]) -> Self {
        Self::new(v[0])
    }
}

impl<'a, S> From<&'a (S,)> for &'a Vector1<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &'a (S,)) -> &'a Vector1<S> {
        unsafe { 
            &*(v as *const (S,) as *const Vector1<S>)
        }
    }
}
/*
impl<'a, S> From<&'a [S; 1]> for &'a Vector1<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &'a [S; 1]) -> &'a Vector1<S> {
        unsafe { 
            &*(v as *const [S; 1] as *const Vector1<S>)
        }
    }
}
*/

impl<S> Vector2<S> {
    /// Construct a new vector.
    #[inline]
    pub const fn new(x: S, y: S) -> Self {
        Self { 
            data: [x, y],
        }
    }
}

impl<S> Vector2<S> 
where 
    S: NumCast + Copy
{
    /// Cast a vector from one type of scalars to another type of scalars.
    ///
    /// # Example
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
        let x = match num_traits::cast(self.data[0]) {
            Some(value) => value,
            None => return None,
        };
        let y = match num_traits::cast(self.data[1]) {
            Some(value) => value,
            None => return None,
        };

        Some(Vector2::new(x, y))
    }
}

impl<S> Vector2<S> 
where 
    S: Copy 
{
    /// Extend a two-dimensional vector into a three-dimensional vector using the 
    /// supplied **z-component**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector2,
    /// #     Vector3,   
    /// # };
    /// #
    /// let v = Vector2::new(1_f64, 2_f64);
    /// let expected = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let result = v.extend(3_f64);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn extend(&self, z: S) -> Vector3<S> {
        Vector3::new(self.data[0], self.data[1], z)
    }

    /// Contract a two-dimensional vector to a one-dimensional vector by removing
    /// the **y-component**.
    ///
    /// # Example
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
    pub fn contract(&self) -> Vector1<S> {
        Vector1::new(self.data[0])
    }

    /// Construct a vector from a fill value.
    ///
    /// Every component of the resulting vector will have the same value
    /// supplied by the `value` argument.
    ///
    /// # Example
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
    pub fn from_fill(value: S) -> Self {
        Self::new(value, value)
    }

    /// Map an operation on that acts on the coordinates of a vector, returning 
    /// a vector whose coordinates are of the new scalar type.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector2,  
    /// # };
    /// #
    /// let vector: Vector2<u32> = Vector2::new(1_u32, 2_u32);
    /// let expected: Vector2<i32> = Vector2::new(2_i32, 3_i32);
    /// let result: Vector2<i32> = vector.map(|comp| (comp + 1) as i32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn map<T, F>(&self, mut op: F) -> Vector2<T> 
    where 
        F: FnMut(S) -> T
    {
        Vector2::new(op(self.data[0]), op(self.data[1]))
    }
}

impl<S> Vector2<S> 
where 
    S: Scalar 
{
    /// Returns the **x-axis** unit vector, a unit vector with the **x-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_x() -> Self {
        Self::new(S::one(), S::zero())
    }

    /// Returns the **y-axis** unit vector, a unit vector with the **y-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_y() -> Self {
        Self::new(S::zero(), S::one())
    }

    /// Compute the zero vector.
    ///
    /// The zero vector is the vector in which all of its elements are zero.
    #[inline]
    pub fn zero() -> Self {
        Self::new(S::zero(), S::zero())
    }
    
    /// Determine whether a vector is the zero vector.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.data[0].is_zero() && self.data[1].is_zero()
    }

    /// Compute the coordinates of a vector in projective space.
    ///
    /// The function appends a `0` to the vector.
    ///
    /// # Example
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
        self.extend(S::zero())
    }

    /// Compute the coordinates of a projective vector in Euclidean space.
    ///
    /// The function removes a `0` from the end of the vector, otherwise it
    /// returns `None`.
    ///
    /// # Example
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
        if self.data[1].is_zero() {
            Some(self.contract())
        } else {
            None
        }
    }

    /// Compute the Euclidean dot product (inner product) of two vectors.
    ///
    /// # Example
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
    pub fn dot(&self, other: &Self) -> S {
        self.data[0] * other.data[0] + self.data[1] * other.data[1]
    }
}

impl<S> Vector2<S> 
where 
    S: ScalarSigned 
{
    /// Compute the negation of a vector mutably in place.
    ///
    /// # Example
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
        self.data[0] = -self.data[0];
        self.data[1] = -self.data[1];
    }
}

impl<S> Vector2<S> 
where 
    S: ScalarFloat
{
    /// Linearly interpolate between the two vectors `self` and `other`.
    ///
    /// # Example
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
    pub fn lerp(&self, other: &Self, amount: S) -> Self {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of a vector are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A vector is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values.
    ///
    /// # Example (Finite Vector)
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
    /// # Example (Not A Finite Vector)
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
        self.data[0].is_finite() && self.data[1].is_finite()
    }

    /// Compute the projection of the vector `self` onto the vector
    /// `other`.
    ///
    /// # Example
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
    pub fn project(&self, other: &Self) -> Self {
        other * (self.dot(other) / other.magnitude_squared())
    }
}

impl<S> fmt::Display for Vector2<S> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Vector2 [{}, {}]", 
            self.data[0], self.data[1]
        )
    }
}

impl<S> Default for Vector2<S>
where
    S: Scalar
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<S> From<(S, S)> for Vector2<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: (S, S)) -> Self {
        Self::new(v.0, v.1)
    }
}

impl<S> From<[S; 2]> for Vector2<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: [S; 2]) -> Self {
        Self::new(v[0], v[1])
    }
}

impl<S> From<&(S, S)> for Vector2<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &(S, S)) -> Self {
        Self::new(v.0, v.1)
    }
}

impl<S> From<&[S; 2]> for Vector2<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &[S; 2]) -> Self {
        Self::new(v[0], v[1])
    }
}

impl<'a, S> From<&'a (S, S)> for &'a Vector2<S> 
where
    S: Scalar
{
    #[inline]
    fn from(v: &'a (S, S)) -> &'a Vector2<S> {
        unsafe {
            &*(v as *const (S, S) as *const Vector2<S>)
        }
    }
}
/*
impl<'a, S> From<&'a [S; 2]> for &'a Vector2<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &'a [S; 2]) -> &'a Vector2<S> {
        unsafe { 
            &*(v as *const [S; 2] as *const Vector2<S>)
        }
    }
}
*/
impl<S> Vector3<S> {
    /// Construct a new vector.
    #[inline]
    pub const fn new(x: S, y: S, z: S) -> Self {
        Self { 
            data: [x, y, z],
        }
    }
}

impl<S> Vector3<S> 
where 
    S: NumCast + Copy
{
    /// Cast a vector from one type of scalars to another type of scalars.
    ///
    /// # Example
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
        let x = match num_traits::cast(self.data[0]) {
            Some(value) => value,
            None => return None,
        };
        let y = match num_traits::cast(self.data[1]) {
            Some(value) => value,
            None => return None,
        };
        let z = match num_traits::cast(self.data[2]) {
            Some(value) => value,
            None => return None,
        };

        Some(Vector3::new(x, y, z))
    }
}

impl<S> Vector3<S> 
where 
    S: Copy
{
    /// Extend a three-dimensional vector into a four-dimensional vector using the 
    /// supplied **w-component**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,
    /// #     Vector4, 
    /// # };
    /// #
    /// let v = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
    /// let result = v.extend(4_f64);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn extend(&self, w: S) -> Vector4<S> {
        Vector4::new(self.data[0], self.data[1], self.data[2], w)
    }

    /// Contract a three-dimensional vector to a two-dimensional vector
    /// by removing the **z-component**.
    ///
    /// # Example
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
    pub fn contract(&self) -> Vector2<S> {
        Vector2::new(self.data[0], self.data[1])
    }

    /// Construct a vector from a fill value.
    ///
    /// Every component of the resulting vector will have the same value
    /// supplied by the `value` argument.
    ///
    /// # Example
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
    pub fn from_fill(value: S) -> Self {
        Self::new(value, value, value)
    }

    /// Map an operation on that acts on the coordinates of a vector, returning 
    /// a vector whose coordinates are of the new scalar type.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector3,  
    /// # };
    /// #
    /// let vector: Vector3<u32> = Vector3::new(1_u32, 2_u32, 3_u32);
    /// let expected: Vector3<i32> = Vector3::new(2_i32, 3_i32, 4_i32);
    /// let result: Vector3<i32> = vector.map(|comp| (comp + 1) as i32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn map<T, F>(&self, mut op: F) -> Vector3<T> 
    where 
        F: FnMut(S) -> T
    {
        Vector3::new(op(self.data[0]), op(self.data[1]), op(self.data[2]))
    }
}

impl<S> Vector3<S> 
where 
    S: Scalar
{
    /// Returns the **x-axis** unit vector, a unit vector with the **x-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_x() -> Self {
        Self::new(S::one(), S::zero(), S::zero())
    }

    /// Returns the **y-axis** unit vector, a unit vector with the **y-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_y() -> Self {
        Self::new(S::zero(), S::one(), S::zero())
    }
    
    /// Returns the **z-axis** unit vector, a unit vector with the **z-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_z() -> Self {
        Self::new(S::zero(), S::zero(), S::one())
    }

    /// Compute the zero vector.
    ///
    /// The zero vector is the vector in which all of its elements are zero.
    #[inline]
    pub fn zero() -> Self {
        Self::new(S::zero(), S::zero(), S::zero())
    }
    
    /// Determine whether a vector is the zero vector.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.data[0].is_zero() && 
        self.data[1].is_zero() && 
        self.data[2].is_zero()
    }

    /// Compute the coordinates of a vector in projective space.
    ///
    /// The function appends a `0` to the vector.
    ///
    /// # Example
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
        self.extend(S::zero())
    }

    /// Compute the coordinates of a projective vector in Euclidean space.
    ///
    /// The function removes a `0` from the end of the vector, otherwise it
    /// returns `None`.
    ///
    /// # Example
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
        if self.data[2].is_zero() {
            Some(self.contract())
        } else {
            None
        }
    }

    /// Compute the Euclidean dot product (inner product) of two vectors.
    ///
    /// # Example
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
    pub fn dot(&self, other: &Self) -> S {
        self.data[0] * other.data[0] + 
        self.data[1] * other.data[1] + 
        self.data[2] * other.data[2]
    }
}

impl<S> Vector3<S> 
where 
    S: ScalarSigned
{
    /// Compute the negation of a vector mutably in place.
    ///
    /// # Example
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
        self.data[0] = -self.data[0];
        self.data[1] = -self.data[1];
        self.data[2] = -self.data[2];
    }

    /// Compute the cross product of two three-dimensional vectors. 
    ///
    /// For the vector dimensions used in computer graphics 
    /// (up to four dimensions), the cross product is well-defined only in 
    /// three dimensions. The cross product is a form of vector 
    /// multiplication that computes a vector normal to the plane swept out by 
    /// the two vectors. The magnitude of this vector is the area of the 
    /// parallelogram swept out by the two vectors. 
    ///
    /// # Example
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
    pub fn cross(&self, other: &Self) -> Self {
        let x = self.data[1] * other.data[2] - self.data[2] * other.data[1];
        let y = self.data[2] * other.data[0] - self.data[0] * other.data[2];
        let z = self.data[0] * other.data[1] - self.data[1] * other.data[0];
    
        Vector3::new(x, y, z)
    }
}

impl<S> Vector3<S> 
where 
    S: ScalarFloat
{
    /// Linearly interpolate between the two vectors `self` and `other`.
    ///
    /// # Example
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
    pub fn lerp(&self, other: &Self, amount: S) -> Self {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of a vector are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A vector is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values.
    ///
    /// # Example (Finite Vector)
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
    /// # Example (Not A Finite Vector)
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
        self.data[0].is_finite() && 
        self.data[1].is_finite() && 
        self.data[2].is_finite()
    }

    /// Compute the projection of the vector `self` onto the vector
    /// `other`.
    ///
    /// # Example
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
    pub fn project(&self, other: &Self) -> Self {
        other * (self.dot(other) / other.magnitude_squared())
    }
}

impl<S> fmt::Display for Vector3<S> 
where 
    S: fmt::Display
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Vector3 [{}, {}, {}]", 
            self.data[0], self.data[1], self.data[2]
        )
    }
}

impl<S> Default for Vector3<S>
where
    S: Scalar
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<S> From<(S, S, S)> for Vector3<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: (S, S, S)) -> Self {
        Self::new(v.0, v.1, v.2)
    }
}

impl<S> From<[S; 3]> for Vector3<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: [S; 3]) -> Self {
        Self::new(v[0], v[1], v[2])
    }
}

impl<S> From<&(S, S, S)> for Vector3<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &(S, S, S)) -> Self {
        Self::new(v.0, v.1, v.2)
    }
}

impl<S> From<&[S; 3]> for Vector3<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &[S; 3]) -> Self {
        Self::new(v[0], v[1], v[2])
    }
}

impl<'a, S> From<&'a (S, S, S)> for &'a Vector3<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &'a (S, S, S)) -> &'a Vector3<S> {
        unsafe { 
            &*(v as *const (S, S, S) as *const Vector3<S>)
        }
    }
}
/*
impl<'a, S> From<&'a [S; 3]> for &'a Vector3<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &'a [S; 3]) -> &'a Vector3<S> {
        unsafe { 
            &*(v as *const [S; 3] as *const Vector3<S>)
        }
    }
}
*/

impl<S> From<Vector4<S>> for Vector3<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: Vector4<S>) -> Self {
        Self::new(v.data[0], v.data[1], v.data[2])
    }
}

impl<S> From<&Vector4<S>> for Vector3<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &Vector4<S>) -> Self {
        Self::new(v.data[0], v.data[1], v.data[2])
    }
}


impl<S> Vector4<S> {
    /// Construct a new four-dimensional vector.
    #[inline]
    pub const fn new(x: S, y: S, z: S, w: S) -> Self {
        Self { 
            data: [x, y, z, w],
        }
    }
}

impl<S> Vector4<S> 
where 
    S: NumCast + Copy
{
    /// Cast a vector from one type of scalars to another type of scalars.
    ///
    /// # Example
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
        let x = match num_traits::cast(self.data[0]) {
            Some(value) => value,
            None => return None,
        };
        let y = match num_traits::cast(self.data[1]) {
            Some(value) => value,
            None => return None,
        };
        let z = match num_traits::cast(self.data[2]) {
            Some(value) => value,
            None => return None,
        };
        let w = match num_traits::cast(self.data[3]) {
            Some(value) => value,
            None => return None,
        };

        Some(Vector4::new(x, y, z, w))
    }
}

impl<S> Vector4<S> 
where 
    S: Copy
{
    /// Contract a four-dimensional vector to a three-dimensional vector
    /// by removing the **w-component**.
    ///
    /// # Example
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
    pub fn contract(&self) -> Vector3<S> {
        Vector3::new(self.data[0], self.data[1], self.data[2])
    }

    /// Construct a vector from a fill value.
    ///
    /// Every component of the resulting vector will have the same value
    /// supplied by the `value` argument.
    ///
    /// # Example
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
    pub fn from_fill(value: S) -> Self {
        Self::new(value, value, value, value)
    }

    /// Map an operation on that acts on the coordinates of a vector, returning 
    /// a vector whose coordinates are of the new scalar type.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Vector4,  
    /// # };
    /// #
    /// let vector: Vector4<u32> = Vector4::new(1_u32, 2_u32, 3_u32, 4_u32);
    /// let expected: Vector4<i32> = Vector4::new(2_i32, 3_i32, 4_i32, 5_i32);
    /// let result: Vector4<i32> = vector.map(|comp| (comp + 1) as i32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn map<T, F>(&self, mut op: F) -> Vector4<T> 
    where 
        F: FnMut(S) -> T
    {
        Vector4::new(
            op(self.data[0]),
            op(self.data[1]),
            op(self.data[2]),
            op(self.data[3]),
        )
    }
}

impl<S> Vector4<S> 
where 
    S: Scalar
{
    /// Returns the **x-axis** unit vector, a unit vector with the **x-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_x() -> Self {
        Self::new(S::one(), S::zero(), S::zero(), S::zero())
    }

    /// Returns the **y-axis** unit vector, a unit vector with the **y-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_y() -> Self {
        Self::new(S::zero(), S::one(), S::zero(), S::zero())
    }
    
    /// Returns the **z-axis** unit vector, a unit vector with the **z-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_z() -> Self {
        Self::new(S::zero(), S::zero(), S::one(), S::zero())
    }

    /// Returns the **w-axis** unit vector, a unit vector with the **w-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_w() -> Self {
        Self::new(S::zero(), S::zero(), S::zero(), S::one())
    }

    /// Compute the zero vector.
    ///
    /// The zero vector is the vector in which all of its elements are zero.
    #[inline]
    pub fn zero() -> Self {
        Self::new(S::zero(), S::zero(), S::zero(), S::zero())
    }
    
    /// Determine whether a vector is the zero vector.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.data[0].is_zero() && 
        self.data[1].is_zero() && 
        self.data[2].is_zero() && 
        self.data[3].is_zero()
    }

    /// Compute the coordinates of a projective vector in Euclidean space.
    ///
    /// The function removes a `0` from the end of the vector, otherwise it
    /// returns `None`.
    ///
    /// # Example
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
        if self.data[3].is_zero() {
            Some(self.contract())
        } else {
            None
        }
    }
    
    /// Compute the Euclidean dot product (inner product) of two vectors.
    ///
    /// # Example
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
    pub fn dot(&self, other: &Self) -> S {
        self.data[0] * other.data[0] + 
        self.data[1] * other.data[1] + 
        self.data[2] * other.data[2] + 
        self.data[3] * other.data[3]
    }
}

impl<S> Vector4<S> 
where 
    S: ScalarSigned
{
    /// Compute the negation of a vector mutably in place.
    ///
    /// # Example
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
        self.data[0] = -self.data[0];
        self.data[1] = -self.data[1];
        self.data[2] = -self.data[2];
        self.data[3] = -self.data[3];
    }
}

impl<S> Vector4<S> 
where 
    S: ScalarFloat
{
    /// Linearly interpolate between the two vectors `self` and `other`.
    ///
    /// # Example
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
    pub fn lerp(&self, other: &Self, amount: S) -> Self {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of a vector are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A vector is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values.
    ///
    /// # Example (Finite Vector)
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
    /// # Example (Not A Finite Vector)
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
        self.data[0].is_finite() && 
        self.data[1].is_finite() && 
        self.data[2].is_finite() && 
        self.data[3].is_finite()
    }

    /// Compute the projection of the vector `self` onto the vector
    /// `other`.
    ///
    /// # Example
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
    pub fn project(&self, other: &Self) -> Self {
        other * (self.dot(other) / other.magnitude_squared())
    }
}

impl<S> fmt::Display for Vector4<S> 
where 
    S: fmt::Display
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Vector4 [{}, {}, {}, {}]", 
            self.data[0], self.data[1], self.data[2], self.data[3]
        )
    }
}

impl<S> Default for Vector4<S>
where
    S: Scalar
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<S> From<(S, S, S, S)> for Vector4<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: (S, S, S, S)) -> Self {
        Self::new(v.0, v.1, v.2, v.3)
    }
}

impl<S> From<[S; 4]> for Vector4<S> 
where 
    S: Scalar 
{
    #[inline]
    fn from(v: [S; 4]) -> Self {
        Self::new(v[0], v[1], v[2], v[3])
    }
}

impl<S> From<&(S, S, S, S)> for Vector4<S> 
where 
    S: Scalar 
{
    #[inline]
    fn from(v: &(S, S, S, S)) -> Self {
        Self::new(v.0, v.1, v.2, v.3)
    }
}

impl<S> From<&[S; 4]> for Vector4<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &[S; 4]) -> Self {
        Self::new(v[0], v[1], v[2], v[3])
    }
}

impl<'a, S> From<&'a (S, S, S, S)> for &'a Vector4<S> 
where 
    S: Scalar 
{
    #[inline]
    fn from(v: &'a (S, S, S, S)) -> &'a Vector4<S> {
        unsafe { 
            &*(v as *const (S, S, S, S) as *const Vector4<S>)
        }
    }
}
/*
impl<'a, S> From<&'a [S; 4]> for &'a Vector4<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &'a [S; 4]) -> &'a Vector4<S> {
        unsafe { 
            &*(v as *const [S; 4] as *const Vector4<S>)    
        }
    }
}
*/
macro_rules! impl_scalar_vector_mul_ops {
    ($Lhs:ty => $Rhs:ty => $Output:ty, { $($index:expr),* }) => {
        impl ops::Mul<$Rhs> for $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                Self::Output::new( $(self * other.data[$index]),* )
            }
        }

        impl<'a> ops::Mul<$Rhs> for &'a $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                Self::Output::new( $(self * other.data[$index]),* )
            }
        }
    }
}

impl_scalar_vector_mul_ops!(u8    => Vector1<u8>    => Vector1<u8>,    { 0 });
impl_scalar_vector_mul_ops!(u16   => Vector1<u16>   => Vector1<u16>,   { 0 });
impl_scalar_vector_mul_ops!(u32   => Vector1<u32>   => Vector1<u32>,   { 0 });
impl_scalar_vector_mul_ops!(u64   => Vector1<u64>   => Vector1<u64>,   { 0 });
impl_scalar_vector_mul_ops!(u128  => Vector1<u128>  => Vector1<u128>,  { 0 });
impl_scalar_vector_mul_ops!(usize => Vector1<usize> => Vector1<usize>, { 0 });
impl_scalar_vector_mul_ops!(i8    => Vector1<i8>    => Vector1<i8>,    { 0 });
impl_scalar_vector_mul_ops!(i16   => Vector1<i16>   => Vector1<i16>,   { 0 });
impl_scalar_vector_mul_ops!(i32   => Vector1<i32>   => Vector1<i32>,   { 0 });
impl_scalar_vector_mul_ops!(i64   => Vector1<i64>   => Vector1<i64>,   { 0 });
impl_scalar_vector_mul_ops!(i128  => Vector1<i128>  => Vector1<i128>,  { 0 });
impl_scalar_vector_mul_ops!(isize => Vector1<isize> => Vector1<isize>, { 0 });
impl_scalar_vector_mul_ops!(f32   => Vector1<f32>   => Vector1<f32>,   { 0 });
impl_scalar_vector_mul_ops!(f64   => Vector1<f64>   => Vector1<f64>,   { 0 });

impl_scalar_vector_mul_ops!(u8    => Vector2<u8>    => Vector2<u8>,    { 0, 1 });
impl_scalar_vector_mul_ops!(u16   => Vector2<u16>   => Vector2<u16>,   { 0, 1 });
impl_scalar_vector_mul_ops!(u32   => Vector2<u32>   => Vector2<u32>,   { 0, 1 });
impl_scalar_vector_mul_ops!(u64   => Vector2<u64>   => Vector2<u64>,   { 0, 1 });
impl_scalar_vector_mul_ops!(u128  => Vector2<u128>  => Vector2<u128>,  { 0, 1 });
impl_scalar_vector_mul_ops!(usize => Vector2<usize> => Vector2<usize>, { 0, 1 });
impl_scalar_vector_mul_ops!(i8    => Vector2<i8>    => Vector2<i8>,    { 0, 1 });
impl_scalar_vector_mul_ops!(i16   => Vector2<i16>   => Vector2<i16>,   { 0, 1 });
impl_scalar_vector_mul_ops!(i32   => Vector2<i32>   => Vector2<i32>,   { 0, 1 });
impl_scalar_vector_mul_ops!(i64   => Vector2<i64>   => Vector2<i64>,   { 0, 1 });
impl_scalar_vector_mul_ops!(i128  => Vector2<i128>  => Vector2<i128>,  { 0, 1 });
impl_scalar_vector_mul_ops!(isize => Vector2<isize> => Vector2<isize>, { 0, 1 });
impl_scalar_vector_mul_ops!(f32   => Vector2<f32>   => Vector2<f32>,   { 0, 1 });
impl_scalar_vector_mul_ops!(f64   => Vector2<f64>   => Vector2<f64>,   { 0, 1 });

impl_scalar_vector_mul_ops!(u8    => Vector3<u8>    => Vector3<u8>,    { 0, 1, 2 });
impl_scalar_vector_mul_ops!(u16   => Vector3<u16>   => Vector3<u16>,   { 0, 1, 2 });
impl_scalar_vector_mul_ops!(u32   => Vector3<u32>   => Vector3<u32>,   { 0, 1, 2 });
impl_scalar_vector_mul_ops!(u64   => Vector3<u64>   => Vector3<u64>,   { 0, 1, 2 });
impl_scalar_vector_mul_ops!(u128  => Vector3<u128>  => Vector3<u128>,  { 0, 1, 2 });
impl_scalar_vector_mul_ops!(usize => Vector3<usize> => Vector3<usize>, { 0, 1, 2 });
impl_scalar_vector_mul_ops!(i8    => Vector3<i8>    => Vector3<i8>,    { 0, 1, 2 });
impl_scalar_vector_mul_ops!(i16   => Vector3<i16>   => Vector3<i16>,   { 0, 1, 2 });
impl_scalar_vector_mul_ops!(i32   => Vector3<i32>   => Vector3<i32>,   { 0, 1, 2 });
impl_scalar_vector_mul_ops!(i64   => Vector3<i64>   => Vector3<i64>,   { 0, 1, 2 });
impl_scalar_vector_mul_ops!(i128  => Vector3<i128>  => Vector3<i128>,  { 0, 1, 2 });
impl_scalar_vector_mul_ops!(isize => Vector3<isize> => Vector3<isize>, { 0, 1, 2 });
impl_scalar_vector_mul_ops!(f32   => Vector3<f32>   => Vector3<f32>,   { 0, 1, 2 });
impl_scalar_vector_mul_ops!(f64   => Vector3<f64>   => Vector3<f64>,   { 0, 1, 2 });

impl_scalar_vector_mul_ops!(u8    => Vector4<u8>    => Vector4<u8>,    { 0, 1, 2, 3 });
impl_scalar_vector_mul_ops!(u16   => Vector4<u16>   => Vector4<u16>,   { 0, 1, 2, 3 });
impl_scalar_vector_mul_ops!(u32   => Vector4<u32>   => Vector4<u32>,   { 0, 1, 2, 3 });
impl_scalar_vector_mul_ops!(u64   => Vector4<u64>   => Vector4<u64>,   { 0, 1, 2, 3 });
impl_scalar_vector_mul_ops!(u128  => Vector4<u128>  => Vector4<u128>,  { 0, 1, 2, 3 });
impl_scalar_vector_mul_ops!(usize => Vector4<usize> => Vector4<usize>, { 0, 1, 2, 3 });
impl_scalar_vector_mul_ops!(i8    => Vector4<i8>    => Vector4<i8>,    { 0, 1, 2, 3 });
impl_scalar_vector_mul_ops!(i16   => Vector4<i16>   => Vector4<i16>,   { 0, 1, 2, 3 });
impl_scalar_vector_mul_ops!(i32   => Vector4<i32>   => Vector4<i32>,   { 0, 1, 2, 3 });
impl_scalar_vector_mul_ops!(i64   => Vector4<i64>   => Vector4<i64>,   { 0, 1, 2, 3 });
impl_scalar_vector_mul_ops!(i128  => Vector4<i128>  => Vector4<i128>,  { 0, 1, 2, 3 });
impl_scalar_vector_mul_ops!(isize => Vector4<isize> => Vector4<isize>, { 0, 1, 2, 3 });
impl_scalar_vector_mul_ops!(f32   => Vector4<f32>   => Vector4<f32>,   { 0, 1, 2, 3 });
impl_scalar_vector_mul_ops!(f64   => Vector4<f64>   => Vector4<f64>,   { 0, 1, 2, 3 });


macro_rules! impl_vector_scalar_binary_ops {
    ($OpType:ident, $op:ident, $T:ty, $Output:ty, { $($index:expr),* }) => {
        impl<S> ops::$OpType<S> for $T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: S) -> Self::Output {
                Self::Output::new( 
                    $( self.data[$index].$op(other) ),* 
                )
            }
        }

        impl<S> ops::$OpType<S> for &$T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: S) -> Self::Output {
                Self::Output::new( 
                    $( self.data[$index].$op(other) ),* 
                )
            }
        }
    }
}

impl_vector_scalar_binary_ops!(Mul, mul, Vector1<S>, Vector1<S>, { 0 });
impl_vector_scalar_binary_ops!(Div, div, Vector1<S>, Vector1<S>, { 0 });
impl_vector_scalar_binary_ops!(Rem, rem, Vector1<S>, Vector1<S>, { 0 });
impl_vector_scalar_binary_ops!(Mul, mul, Vector2<S>, Vector2<S>, { 0, 1 });
impl_vector_scalar_binary_ops!(Div, div, Vector2<S>, Vector2<S>, { 0, 1 });
impl_vector_scalar_binary_ops!(Rem, rem, Vector2<S>, Vector2<S>, { 0, 1 });
impl_vector_scalar_binary_ops!(Mul, mul, Vector3<S>, Vector3<S>, { 0, 1, 2 });
impl_vector_scalar_binary_ops!(Div, div, Vector3<S>, Vector3<S>, { 0, 1, 2 });
impl_vector_scalar_binary_ops!(Rem, rem, Vector3<S>, Vector3<S>, { 0, 1, 2 });
impl_vector_scalar_binary_ops!(Mul, mul, Vector4<S>, Vector4<S>, { 0, 1, 2, 3 });
impl_vector_scalar_binary_ops!(Div, div, Vector4<S>, Vector4<S>, { 0, 1, 2, 3 });
impl_vector_scalar_binary_ops!(Rem, rem, Vector4<S>, Vector4<S>, { 0, 1, 2, 3 });


macro_rules! impl_vector_vector_binary_ops {
    ($OpType:ident, $op:ident, $T:ty, $Output:ty, { $($index:expr),* }) => {
        impl<S> ops::$OpType<$T> for $T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: $T) -> Self::Output {
                Self::Output::new( 
                    $( self.data[$index].$op(other.data[$index]) ),* 
                )
            }
        }

        impl<S> ops::$OpType<&$T> for $T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: &$T) -> Self::Output {
                Self::Output::new( 
                    $( self.data[$index].$op(other.data[$index]) ),* 
                )
            }
        }

        impl<S> ops::$OpType<$T> for &$T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: $T) -> Self::Output {
                Self::Output::new( 
                    $( self.data[$index].$op(other.data[$index]) ),* 
                )
            }
        }

        impl<'a, 'b, S> ops::$OpType<&'a $T> for &'b $T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: &'a $T) -> Self::Output {
                Self::Output::new( 
                    $( self.data[$index].$op(other.data[$index]) ),* 
                )
            }
        }
    }
}

impl_vector_vector_binary_ops!(Add, add, Vector1<S>, Vector1<S>, { 0 });
impl_vector_vector_binary_ops!(Sub, sub, Vector1<S>, Vector1<S>, { 0 });
impl_vector_vector_binary_ops!(Add, add, Vector2<S>, Vector2<S>, { 0, 1 });
impl_vector_vector_binary_ops!(Sub, sub, Vector2<S>, Vector2<S>, { 0, 1 });
impl_vector_vector_binary_ops!(Add, add, Vector3<S>, Vector3<S>, { 0, 1, 2 });
impl_vector_vector_binary_ops!(Sub, sub, Vector3<S>, Vector3<S>, { 0, 1, 2 });
impl_vector_vector_binary_ops!(Add, add, Vector4<S>, Vector4<S>, { 0, 1, 2, 3 });
impl_vector_vector_binary_ops!(Sub, sub, Vector4<S>, Vector4<S>, { 0, 1, 2, 3 });


macro_rules! impl_vector_binary_assign_ops {
    ($T:ty, { $($index:expr),* }) => {
        impl<S> ops::AddAssign<$T> for $T where S: Scalar {
            #[inline]
            fn add_assign(&mut self, other: $T) {
                $(self.data[$index] += other.data[$index]);*
            }
        }

        impl<S> ops::AddAssign<&$T> for $T where S: Scalar {
            #[inline]
            fn add_assign(&mut self, other: &$T) {
                $(self.data[$index] += other.data[$index]);*
            }
        }

        impl<S> ops::SubAssign<$T> for $T where S: Scalar {
            #[inline]
            fn sub_assign(&mut self, other: $T) {
                $(self.data[$index] -= other.data[$index]);*
            }
        }

        impl<S> ops::SubAssign<&$T> for $T where S: Scalar {
            #[inline]
            fn sub_assign(&mut self, other: &$T) {
                $(self.data[$index] -= other.data[$index]);*
            }
        }

        impl<S> ops::MulAssign<S> for $T where S: Scalar {
            #[inline]
            fn mul_assign(&mut self, other: S) {
                $(self.data[$index] *= other);*
            }
        }
        
        impl<S> ops::DivAssign<S> for $T where S: Scalar {
            #[inline]
            fn div_assign(&mut self, other: S) {
                $(self.data[$index] /= other);*
            }
        }
        
        impl<S> ops::RemAssign<S> for $T where S: Scalar {
            #[inline]
            fn rem_assign(&mut self, other: S) {
                $(self.data[$index] %= other);*
            }
        }
    }
}

impl_vector_binary_assign_ops!(Vector1<S>, { 0 });
impl_vector_binary_assign_ops!(Vector2<S>, { 0, 1 });
impl_vector_binary_assign_ops!(Vector3<S>, { 0, 1, 2 });
impl_vector_binary_assign_ops!(Vector4<S>, { 0, 1, 2, 3 });


macro_rules! impl_vector_unary_ops {
    ($OpType:ident, $op:ident, $T:ty, $Output:ty, { $($index:expr),* }) => {
        impl<S> ops::$OpType for $T where S: ScalarSigned {
            type Output = $Output;

            #[inline]
            fn $op(self) -> Self::Output {
                Self::Output::new( 
                    $( self.data[$index].$op() ),* 
                )
            }
        }

        impl<S> ops::$OpType for &$T where S: ScalarSigned {
            type Output = $Output;

            #[inline]
            fn $op(self) -> Self::Output {
                Self::Output::new( 
                    $( self.data[$index].$op() ),* 
                )
            }
        }
    }
}

impl_vector_unary_ops!(Neg, neg, Vector1<S>, Vector1<S>, { 0 });
impl_vector_unary_ops!(Neg, neg, Vector2<S>, Vector2<S>, { 0, 1 });
impl_vector_unary_ops!(Neg, neg, Vector3<S>, Vector3<S>, { 0, 1, 2 });
impl_vector_unary_ops!(Neg, neg, Vector4<S>, Vector4<S>, { 0, 1, 2, 3 });


impl_coords!(X, { x });
impl_coords_deref!(Vector1, X);

impl_coords!(XY, { x, y });
impl_coords_deref!(Vector2, XY);

impl_coords!(XYZ, { x, y, z });
impl_coords_deref!(Vector3, XYZ);

impl_coords!(XYZW, { x, y, z, w });
impl_coords_deref!(Vector4, XYZW);


macro_rules! impl_vector_index_ops {
    ($IndexType:ty, $Output:ty) => {
        impl<S, const N: usize> ops::Index<$IndexType> for Vector<S, N> {
            type Output = $Output;

            #[inline]
            fn index(&self, index: $IndexType) -> &Self::Output {
                let v: &[S; N] = self.as_ref();
                &v[index]
            }
        }

        impl<S, const N: usize> ops::IndexMut<$IndexType> for Vector<S, N> {
            #[inline]
            fn index_mut(&mut self, index: $IndexType) -> &mut Self::Output {
                let v: &mut [S; N] = self.as_mut();
                &mut v[index]
            }
        }
    }
}

impl_vector_index_ops!(usize, S);
impl_vector_index_ops!(ops::Range<usize>, [S]);
impl_vector_index_ops!(ops::RangeTo<usize>, [S]);
impl_vector_index_ops!(ops::RangeFrom<usize>, [S]);
impl_vector_index_ops!(ops::RangeFull, [S]);


macro_rules! impl_as_ref_ops {
    ($VecType:ty, $RefType:ty) => {
        impl<S> AsRef<$RefType> for $VecType {
            #[inline]
            fn as_ref(&self) -> &$RefType {
                unsafe {
                    &*(self as *const $VecType as *const $RefType)
                }
            }
        }

        impl<S> AsMut<$RefType> for $VecType {
            #[inline]
            fn as_mut(&mut self) -> &mut $RefType {
                unsafe {
                    &mut *(self as *mut $VecType as *mut $RefType)
                }
            }
        }
    }
}

impl_as_ref_ops!(Vector1<S>, S);
impl_as_ref_ops!(Vector1<S>, (S,));
impl_as_ref_ops!(Vector2<S>, (S, S));
impl_as_ref_ops!(Vector3<S>, (S, S, S));
impl_as_ref_ops!(Vector4<S>, (S, S, S, S));


macro_rules! impl_magnitude {
    ($VectorN:ident) => {
        impl<S> Magnitude for $VectorN<S> where S: ScalarFloat {
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
            fn distance_squared(&self, other: &$VectorN<S>) -> Self::Output {
                (self - other).magnitude_squared()
            }
        
            #[inline]
            fn distance(&self, other: &Self) -> Self::Output {
                self.distance_squared(other).sqrt()
            }
        }
    }
}

impl_magnitude!(Vector1);
impl_magnitude!(Vector2);
impl_magnitude!(Vector3);
impl_magnitude!(Vector4);


macro_rules! impl_approx_eq_ops {
    ($T:ident, { $($index:expr),* }) => {
        impl<S> approx::AbsDiffEq for $T<S> where S: ScalarFloat {
            type Epsilon = <S as approx::AbsDiffEq>::Epsilon;
        
            #[inline]
            fn default_epsilon() -> Self::Epsilon {
                S::default_epsilon()
            }
        
            #[inline]
            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                $(S::abs_diff_eq(&self.data[$index], &other.data[$index], epsilon)) &&*
            }
        }
        
        impl<S> approx::RelativeEq for $T<S> where S: ScalarFloat {
            #[inline]
            fn default_max_relative() -> S::Epsilon {
                S::default_max_relative()
            }
        
            #[inline]
            fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
                $(S::relative_eq(&self.data[$index], &other.data[$index], epsilon, max_relative)) &&*
            }
        }
        
        impl<S> approx::UlpsEq for $T<S> where S: ScalarFloat {
            #[inline]
            fn default_max_ulps() -> u32 {
                S::default_max_ulps()
            }
        
            #[inline]
            fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
                $(S::ulps_eq(&self.data[$index], &other.data[$index], epsilon, max_ulps)) &&*
            }
        }
    }
}

impl_approx_eq_ops!(Vector1, { 0 });
impl_approx_eq_ops!(Vector2, { 0, 1 });
impl_approx_eq_ops!(Vector3, { 0, 1, 2 });
impl_approx_eq_ops!(Vector4, { 0, 1, 2, 3 });


macro_rules! impl_swizzle {
    ($name:ident() => $VectorN:ident => $Output:ident { $($i:expr),+ }) => {
        impl<S> $VectorN<S> where S: Copy {
            /// Construct a new vector from the components of the input vector.
            #[inline]
            pub fn $name(&self) -> $Output<S> {
                $Output::new(
                    $(self.data[$i]),*
                )
            }
        }
    }
}

impl_swizzle!(x() => Vector1 => Vector1 { 0 });

impl_swizzle!(x()  => Vector2 => Vector1 { 0 });
impl_swizzle!(y()  => Vector2 => Vector1 { 1 });
impl_swizzle!(xx() => Vector2 => Vector2 { 0, 0 });
impl_swizzle!(xy() => Vector2 => Vector2 { 0, 1 });
impl_swizzle!(yx() => Vector2 => Vector2 { 1, 0 });
impl_swizzle!(yy() => Vector2 => Vector2 { 1, 1 });

impl_swizzle!(x()  => Vector3 => Vector1 { 0 });
impl_swizzle!(y()  => Vector3 => Vector1 { 1 });
impl_swizzle!(z()  => Vector3 => Vector1 { 2 });
impl_swizzle!(xx() => Vector3 => Vector2 { 0, 0 });
impl_swizzle!(xy() => Vector3 => Vector2 { 0, 1 });
impl_swizzle!(xz() => Vector3 => Vector2 { 0, 2 });
impl_swizzle!(yx() => Vector3 => Vector2 { 1, 0 });
impl_swizzle!(yy() => Vector3 => Vector2 { 1, 1 });
impl_swizzle!(yz() => Vector3 => Vector2 { 1, 2 });
impl_swizzle!(zx() => Vector3 => Vector2 { 2, 0 });
impl_swizzle!(zy() => Vector3 => Vector2 { 2, 1 });
impl_swizzle!(zz() => Vector3 => Vector2 { 2, 2 });

impl_swizzle!(xxx() => Vector3 => Vector3 { 0, 0, 0 });
impl_swizzle!(xxy() => Vector3 => Vector3 { 0, 0, 1 });
impl_swizzle!(xxz() => Vector3 => Vector3 { 0, 0, 2 });
impl_swizzle!(xyx() => Vector3 => Vector3 { 0, 1, 0 });
impl_swizzle!(xyy() => Vector3 => Vector3 { 0, 1, 1 });
impl_swizzle!(xyz() => Vector3 => Vector3 { 0, 1, 2 });
impl_swizzle!(xzx() => Vector3 => Vector3 { 0, 2, 0 });
impl_swizzle!(xzy() => Vector3 => Vector3 { 0, 2, 1 });
impl_swizzle!(xzz() => Vector3 => Vector3 { 0, 2, 2 });
impl_swizzle!(yxx() => Vector3 => Vector3 { 1, 0, 0 });
impl_swizzle!(yxy() => Vector3 => Vector3 { 1, 0, 1 });
impl_swizzle!(yxz() => Vector3 => Vector3 { 1, 0, 2 });
impl_swizzle!(yyx() => Vector3 => Vector3 { 1, 1, 0 });
impl_swizzle!(yyy() => Vector3 => Vector3 { 1, 1, 1 });
impl_swizzle!(yyz() => Vector3 => Vector3 { 1, 1, 2 });
impl_swizzle!(yzx() => Vector3 => Vector3 { 1, 2, 0 });
impl_swizzle!(yzy() => Vector3 => Vector3 { 1, 2, 1 });
impl_swizzle!(yzz() => Vector3 => Vector3 { 1, 2, 2 });
impl_swizzle!(zxx() => Vector3 => Vector3 { 2, 0, 0 });
impl_swizzle!(zxy() => Vector3 => Vector3 { 2, 0, 1 });
impl_swizzle!(zxz() => Vector3 => Vector3 { 2, 0, 2 });
impl_swizzle!(zyx() => Vector3 => Vector3 { 2, 1, 0 });
impl_swizzle!(zyy() => Vector3 => Vector3 { 2, 1, 1 });
impl_swizzle!(zyz() => Vector3 => Vector3 { 2, 1, 2 });
impl_swizzle!(zzx() => Vector3 => Vector3 { 2, 2, 0 });
impl_swizzle!(zzy() => Vector3 => Vector3 { 2, 2, 1 });
impl_swizzle!(zzz() => Vector3 => Vector3 { 2, 2, 2 });


impl_swizzle!(x() => Vector4 => Vector1 { 0 });
impl_swizzle!(y() => Vector4 => Vector1 { 1 });
impl_swizzle!(z() => Vector4 => Vector1 { 2 });
impl_swizzle!(w() => Vector4 => Vector1 { 3 });

impl_swizzle!(xx() => Vector4 => Vector2 { 0, 0 });
impl_swizzle!(xy() => Vector4 => Vector2 { 0, 1 });
impl_swizzle!(xz() => Vector4 => Vector2 { 0, 2 });
impl_swizzle!(xw() => Vector4 => Vector2 { 0, 3 });
impl_swizzle!(yx() => Vector4 => Vector2 { 1, 0 });
impl_swizzle!(yy() => Vector4 => Vector2 { 1, 1 });
impl_swizzle!(yz() => Vector4 => Vector2 { 1, 2 });
impl_swizzle!(yw() => Vector4 => Vector2 { 1, 3 });
impl_swizzle!(zx() => Vector4 => Vector2 { 2, 0 });
impl_swizzle!(zy() => Vector4 => Vector2 { 2, 1 });
impl_swizzle!(zz() => Vector4 => Vector2 { 2, 2 });
impl_swizzle!(zw() => Vector4 => Vector2 { 2, 3 });
impl_swizzle!(wx() => Vector4 => Vector2 { 3, 0 });
impl_swizzle!(wy() => Vector4 => Vector2 { 3, 1 });
impl_swizzle!(wz() => Vector4 => Vector2 { 3, 2 });
impl_swizzle!(ww() => Vector4 => Vector2 { 3, 3 });

impl_swizzle!(xxx() => Vector4 => Vector3 { 0, 0, 0 });
impl_swizzle!(xxy() => Vector4 => Vector3 { 0, 0, 1 });
impl_swizzle!(xxz() => Vector4 => Vector3 { 0, 0, 2 });
impl_swizzle!(xxw() => Vector4 => Vector3 { 0, 0, 3 });
impl_swizzle!(xyx() => Vector4 => Vector3 { 0, 1, 0 });
impl_swizzle!(xyy() => Vector4 => Vector3 { 0, 1, 1 });
impl_swizzle!(xyz() => Vector4 => Vector3 { 0, 1, 2 });
impl_swizzle!(xyw() => Vector4 => Vector3 { 0, 1, 3 });
impl_swizzle!(xzx() => Vector4 => Vector3 { 0, 2, 0 });
impl_swizzle!(xzy() => Vector4 => Vector3 { 0, 2, 1 });
impl_swizzle!(xzz() => Vector4 => Vector3 { 0, 2, 2 });
impl_swizzle!(xzw() => Vector4 => Vector3 { 0, 2, 3 });
impl_swizzle!(xwx() => Vector4 => Vector3 { 0, 3, 0 });
impl_swizzle!(xwy() => Vector4 => Vector3 { 0, 3, 1 });
impl_swizzle!(xwz() => Vector4 => Vector3 { 0, 3, 2 });
impl_swizzle!(xww() => Vector4 => Vector3 { 0, 3, 3 });
impl_swizzle!(yxx() => Vector4 => Vector3 { 1, 0, 0 });
impl_swizzle!(yxy() => Vector4 => Vector3 { 1, 0, 1 });
impl_swizzle!(yxz() => Vector4 => Vector3 { 1, 0, 2 });
impl_swizzle!(yxw() => Vector4 => Vector3 { 1, 0, 3 });
impl_swizzle!(yyx() => Vector4 => Vector3 { 1, 1, 0 });
impl_swizzle!(yyy() => Vector4 => Vector3 { 1, 1, 1 });
impl_swizzle!(yyz() => Vector4 => Vector3 { 1, 1, 2 });
impl_swizzle!(yyw() => Vector4 => Vector3 { 1, 1, 3 });
impl_swizzle!(yzx() => Vector4 => Vector3 { 1, 2, 0 });
impl_swizzle!(yzy() => Vector4 => Vector3 { 1, 2, 1 });
impl_swizzle!(yzz() => Vector4 => Vector3 { 1, 2, 2 });
impl_swizzle!(yzw() => Vector4 => Vector3 { 1, 2, 3 });
impl_swizzle!(ywx() => Vector4 => Vector3 { 1, 3, 0 });
impl_swizzle!(ywy() => Vector4 => Vector3 { 1, 3, 1 });
impl_swizzle!(ywz() => Vector4 => Vector3 { 1, 3, 2 });
impl_swizzle!(yww() => Vector4 => Vector3 { 1, 3, 3 });
impl_swizzle!(zxx() => Vector4 => Vector3 { 2, 0, 0 });
impl_swizzle!(zxy() => Vector4 => Vector3 { 2, 0, 1 });
impl_swizzle!(zxz() => Vector4 => Vector3 { 2, 0, 2 });
impl_swizzle!(zxw() => Vector4 => Vector3 { 2, 0, 3 });
impl_swizzle!(zyx() => Vector4 => Vector3 { 2, 1, 0 });
impl_swizzle!(zyy() => Vector4 => Vector3 { 2, 1, 1 });
impl_swizzle!(zyz() => Vector4 => Vector3 { 2, 1, 2 });
impl_swizzle!(zyw() => Vector4 => Vector3 { 2, 1, 3 });
impl_swizzle!(zzx() => Vector4 => Vector3 { 2, 2, 0 });
impl_swizzle!(zzy() => Vector4 => Vector3 { 2, 2, 1 });
impl_swizzle!(zzz() => Vector4 => Vector3 { 2, 2, 2 });
impl_swizzle!(zzw() => Vector4 => Vector3 { 2, 2, 3 });
impl_swizzle!(zwx() => Vector4 => Vector3 { 2, 3, 0 });
impl_swizzle!(zwy() => Vector4 => Vector3 { 2, 3, 1 });
impl_swizzle!(zwz() => Vector4 => Vector3 { 2, 3, 2 });
impl_swizzle!(zww() => Vector4 => Vector3 { 2, 3, 3 });
impl_swizzle!(wxx() => Vector4 => Vector3 { 3, 0, 0 });
impl_swizzle!(wxy() => Vector4 => Vector3 { 3, 0, 1 });
impl_swizzle!(wxz() => Vector4 => Vector3 { 3, 0, 2 });
impl_swizzle!(wxw() => Vector4 => Vector3 { 3, 0, 3 });
impl_swizzle!(wyx() => Vector4 => Vector3 { 3, 1, 0 });
impl_swizzle!(wyy() => Vector4 => Vector3 { 3, 1, 1 });
impl_swizzle!(wyz() => Vector4 => Vector3 { 3, 1, 2 });
impl_swizzle!(wyw() => Vector4 => Vector3 { 3, 1, 3 });
impl_swizzle!(wzx() => Vector4 => Vector3 { 3, 2, 0 });
impl_swizzle!(wzy() => Vector4 => Vector3 { 3, 2, 1 });
impl_swizzle!(wzz() => Vector4 => Vector3 { 3, 2, 2 });
impl_swizzle!(wzw() => Vector4 => Vector3 { 3, 2, 3 });
impl_swizzle!(wwx() => Vector4 => Vector3 { 3, 3, 0 });
impl_swizzle!(wwy() => Vector4 => Vector3 { 3, 3, 1 });
impl_swizzle!(wwz() => Vector4 => Vector3 { 3, 3, 2 });
impl_swizzle!(www() => Vector4 => Vector3 { 3, 3, 3 });


impl_swizzle!(xxxx() => Vector4 => Vector4 { 0, 0, 0, 0 });
impl_swizzle!(xxxy() => Vector4 => Vector4 { 0, 0, 0, 1 });
impl_swizzle!(xxxz() => Vector4 => Vector4 { 0, 0, 0, 2 });
impl_swizzle!(xxxw() => Vector4 => Vector4 { 0, 0, 0, 3 });
impl_swizzle!(xxyx() => Vector4 => Vector4 { 0, 0, 1, 0 });
impl_swizzle!(xxyy() => Vector4 => Vector4 { 0, 0, 1, 1 });
impl_swizzle!(xxyz() => Vector4 => Vector4 { 0, 0, 1, 2 });
impl_swizzle!(xxyw() => Vector4 => Vector4 { 0, 0, 1, 3 });
impl_swizzle!(xxzx() => Vector4 => Vector4 { 0, 0, 2, 0 });
impl_swizzle!(xxzy() => Vector4 => Vector4 { 0, 0, 2, 1 });
impl_swizzle!(xxzz() => Vector4 => Vector4 { 0, 0, 2, 2 });
impl_swizzle!(xxzw() => Vector4 => Vector4 { 0, 0, 2, 3 });
impl_swizzle!(xxwx() => Vector4 => Vector4 { 0, 0, 3, 0 });
impl_swizzle!(xxwy() => Vector4 => Vector4 { 0, 0, 3, 1 });
impl_swizzle!(xxwz() => Vector4 => Vector4 { 0, 0, 3, 2 });
impl_swizzle!(xxww() => Vector4 => Vector4 { 0, 0, 3, 3 });
impl_swizzle!(xyxx() => Vector4 => Vector4 { 0, 1, 0, 0 });
impl_swizzle!(xyxy() => Vector4 => Vector4 { 0, 1, 0, 1 });
impl_swizzle!(xyxz() => Vector4 => Vector4 { 0, 1, 0, 2 });
impl_swizzle!(xyxw() => Vector4 => Vector4 { 0, 1, 0, 3 });
impl_swizzle!(xyyx() => Vector4 => Vector4 { 0, 1, 1, 0 });
impl_swizzle!(xyyy() => Vector4 => Vector4 { 0, 1, 1, 1 });
impl_swizzle!(xyyz() => Vector4 => Vector4 { 0, 1, 1, 2 });
impl_swizzle!(xyyw() => Vector4 => Vector4 { 0, 1, 1, 3 });
impl_swizzle!(xyzx() => Vector4 => Vector4 { 0, 1, 2, 0 });
impl_swizzle!(xyzy() => Vector4 => Vector4 { 0, 1, 2, 1 });
impl_swizzle!(xyzz() => Vector4 => Vector4 { 0, 1, 2, 2 });
impl_swizzle!(xyzw() => Vector4 => Vector4 { 0, 1, 2, 3 });
impl_swizzle!(xywx() => Vector4 => Vector4 { 0, 1, 3, 0 });
impl_swizzle!(xywy() => Vector4 => Vector4 { 0, 1, 3, 1 });
impl_swizzle!(xywz() => Vector4 => Vector4 { 0, 1, 3, 2 });
impl_swizzle!(xyww() => Vector4 => Vector4 { 0, 1, 3, 3 });
impl_swizzle!(xzxx() => Vector4 => Vector4 { 0, 2, 0, 0 });
impl_swizzle!(xzxy() => Vector4 => Vector4 { 0, 2, 0, 1 });
impl_swizzle!(xzxz() => Vector4 => Vector4 { 0, 2, 0, 2 });
impl_swizzle!(xzxw() => Vector4 => Vector4 { 0, 2, 0, 3 });
impl_swizzle!(xzyx() => Vector4 => Vector4 { 0, 2, 1, 0 });
impl_swizzle!(xzyy() => Vector4 => Vector4 { 0, 2, 1, 1 });
impl_swizzle!(xzyz() => Vector4 => Vector4 { 0, 2, 1, 2 });
impl_swizzle!(xzyw() => Vector4 => Vector4 { 0, 2, 1, 3 });
impl_swizzle!(xzzx() => Vector4 => Vector4 { 0, 2, 2, 0 });
impl_swizzle!(xzzy() => Vector4 => Vector4 { 0, 2, 2, 1 });
impl_swizzle!(xzzz() => Vector4 => Vector4 { 0, 2, 2, 2 });
impl_swizzle!(xzzw() => Vector4 => Vector4 { 0, 2, 2, 3 });
impl_swizzle!(xzwx() => Vector4 => Vector4 { 0, 2, 3, 0 });
impl_swizzle!(xzwy() => Vector4 => Vector4 { 0, 2, 3, 1 });
impl_swizzle!(xzwz() => Vector4 => Vector4 { 0, 2, 3, 2 });
impl_swizzle!(xzww() => Vector4 => Vector4 { 0, 2, 3, 3 });
impl_swizzle!(xwxx() => Vector4 => Vector4 { 0, 3, 0, 0 });
impl_swizzle!(xwxy() => Vector4 => Vector4 { 0, 3, 0, 1 });
impl_swizzle!(xwxz() => Vector4 => Vector4 { 0, 3, 0, 2 });
impl_swizzle!(xwxw() => Vector4 => Vector4 { 0, 3, 0, 3 });
impl_swizzle!(xwyx() => Vector4 => Vector4 { 0, 3, 1, 0 });
impl_swizzle!(xwyy() => Vector4 => Vector4 { 0, 3, 1, 1 });
impl_swizzle!(xwyz() => Vector4 => Vector4 { 0, 3, 1, 2 });
impl_swizzle!(xwyw() => Vector4 => Vector4 { 0, 3, 1, 3 });
impl_swizzle!(xwzx() => Vector4 => Vector4 { 0, 3, 2, 0 });
impl_swizzle!(xwzy() => Vector4 => Vector4 { 0, 3, 2, 1 });
impl_swizzle!(xwzz() => Vector4 => Vector4 { 0, 3, 2, 2 });
impl_swizzle!(xwzw() => Vector4 => Vector4 { 0, 3, 2, 3 });
impl_swizzle!(xwwx() => Vector4 => Vector4 { 0, 3, 3, 0 });
impl_swizzle!(xwwy() => Vector4 => Vector4 { 0, 3, 3, 1 });
impl_swizzle!(xwwz() => Vector4 => Vector4 { 0, 3, 3, 2 });
impl_swizzle!(xwww() => Vector4 => Vector4 { 0, 3, 3, 3 });
impl_swizzle!(yxxx() => Vector4 => Vector4 { 1, 0, 0, 0 });
impl_swizzle!(yxxy() => Vector4 => Vector4 { 1, 0, 0, 1 });
impl_swizzle!(yxxz() => Vector4 => Vector4 { 1, 0, 0, 2 });
impl_swizzle!(yxxw() => Vector4 => Vector4 { 1, 0, 0, 3 });
impl_swizzle!(yxyx() => Vector4 => Vector4 { 1, 0, 1, 0 });
impl_swizzle!(yxyy() => Vector4 => Vector4 { 1, 0, 1, 1 });
impl_swizzle!(yxyz() => Vector4 => Vector4 { 1, 0, 1, 2 });
impl_swizzle!(yxyw() => Vector4 => Vector4 { 1, 0, 1, 3 });
impl_swizzle!(yxzx() => Vector4 => Vector4 { 1, 0, 2, 0 });
impl_swizzle!(yxzy() => Vector4 => Vector4 { 1, 0, 2, 1 });
impl_swizzle!(yxzz() => Vector4 => Vector4 { 1, 0, 2, 2 });
impl_swizzle!(yxzw() => Vector4 => Vector4 { 1, 0, 2, 3 });
impl_swizzle!(yxwx() => Vector4 => Vector4 { 1, 0, 3, 0 });
impl_swizzle!(yxwy() => Vector4 => Vector4 { 1, 0, 3, 1 });
impl_swizzle!(yxwz() => Vector4 => Vector4 { 1, 0, 3, 2 });
impl_swizzle!(yxww() => Vector4 => Vector4 { 1, 0, 3, 3 });
impl_swizzle!(yyxx() => Vector4 => Vector4 { 1, 1, 0, 0 });
impl_swizzle!(yyxy() => Vector4 => Vector4 { 1, 1, 0, 1 });
impl_swizzle!(yyxz() => Vector4 => Vector4 { 1, 1, 0, 2 });
impl_swizzle!(yyxw() => Vector4 => Vector4 { 1, 1, 0, 3 });
impl_swizzle!(yyyx() => Vector4 => Vector4 { 1, 1, 1, 0 });
impl_swizzle!(yyyy() => Vector4 => Vector4 { 1, 1, 1, 1 });
impl_swizzle!(yyyz() => Vector4 => Vector4 { 1, 1, 1, 2 });
impl_swizzle!(yyyw() => Vector4 => Vector4 { 1, 1, 1, 3 });
impl_swizzle!(yyzx() => Vector4 => Vector4 { 1, 1, 2, 0 });
impl_swizzle!(yyzy() => Vector4 => Vector4 { 1, 1, 2, 1 });
impl_swizzle!(yyzz() => Vector4 => Vector4 { 1, 1, 2, 2 });
impl_swizzle!(yyzw() => Vector4 => Vector4 { 1, 1, 2, 3 });
impl_swizzle!(yywx() => Vector4 => Vector4 { 1, 1, 3, 0 });
impl_swizzle!(yywy() => Vector4 => Vector4 { 1, 1, 3, 1 });
impl_swizzle!(yywz() => Vector4 => Vector4 { 1, 1, 3, 2 });
impl_swizzle!(yyww() => Vector4 => Vector4 { 1, 1, 3, 3 });
impl_swizzle!(yzxx() => Vector4 => Vector4 { 1, 2, 0, 0 });
impl_swizzle!(yzxy() => Vector4 => Vector4 { 1, 2, 0, 1 });
impl_swizzle!(yzxz() => Vector4 => Vector4 { 1, 2, 0, 2 });
impl_swizzle!(yzxw() => Vector4 => Vector4 { 1, 2, 0, 3 });
impl_swizzle!(yzyx() => Vector4 => Vector4 { 1, 2, 1, 0 });
impl_swizzle!(yzyy() => Vector4 => Vector4 { 1, 2, 1, 1 });
impl_swizzle!(yzyz() => Vector4 => Vector4 { 1, 2, 1, 2 });
impl_swizzle!(yzyw() => Vector4 => Vector4 { 1, 2, 1, 3 });
impl_swizzle!(yzzx() => Vector4 => Vector4 { 1, 2, 2, 0 });
impl_swizzle!(yzzy() => Vector4 => Vector4 { 1, 2, 2, 1 });
impl_swizzle!(yzzz() => Vector4 => Vector4 { 1, 2, 2, 2 });
impl_swizzle!(yzzw() => Vector4 => Vector4 { 1, 2, 2, 3 });
impl_swizzle!(yzwx() => Vector4 => Vector4 { 1, 2, 3, 0 });
impl_swizzle!(yzwy() => Vector4 => Vector4 { 1, 2, 3, 1 });
impl_swizzle!(yzwz() => Vector4 => Vector4 { 1, 2, 3, 2 });
impl_swizzle!(yzww() => Vector4 => Vector4 { 1, 2, 3, 3 });
impl_swizzle!(ywxx() => Vector4 => Vector4 { 1, 3, 0, 0 });
impl_swizzle!(ywxy() => Vector4 => Vector4 { 1, 3, 0, 1 });
impl_swizzle!(ywxz() => Vector4 => Vector4 { 1, 3, 0, 2 });
impl_swizzle!(ywxw() => Vector4 => Vector4 { 1, 3, 0, 3 });
impl_swizzle!(ywyx() => Vector4 => Vector4 { 1, 3, 1, 0 });
impl_swizzle!(ywyy() => Vector4 => Vector4 { 1, 3, 1, 1 });
impl_swizzle!(ywyz() => Vector4 => Vector4 { 1, 3, 1, 2 });
impl_swizzle!(ywyw() => Vector4 => Vector4 { 1, 3, 1, 3 });
impl_swizzle!(ywzx() => Vector4 => Vector4 { 1, 3, 2, 0 });
impl_swizzle!(ywzy() => Vector4 => Vector4 { 1, 3, 2, 1 });
impl_swizzle!(ywzz() => Vector4 => Vector4 { 1, 3, 2, 2 });
impl_swizzle!(ywzw() => Vector4 => Vector4 { 1, 3, 2, 3 });
impl_swizzle!(ywwx() => Vector4 => Vector4 { 1, 3, 3, 0 });
impl_swizzle!(ywwy() => Vector4 => Vector4 { 1, 3, 3, 1 });
impl_swizzle!(ywwz() => Vector4 => Vector4 { 1, 3, 3, 2 });
impl_swizzle!(ywww() => Vector4 => Vector4 { 1, 3, 3, 3 });
impl_swizzle!(zxxx() => Vector4 => Vector4 { 2, 0, 0, 0 });
impl_swizzle!(zxxy() => Vector4 => Vector4 { 2, 0, 0, 1 });
impl_swizzle!(zxxz() => Vector4 => Vector4 { 2, 0, 0, 2 });
impl_swizzle!(zxxw() => Vector4 => Vector4 { 2, 0, 0, 3 });
impl_swizzle!(zxyx() => Vector4 => Vector4 { 2, 0, 1, 0 });
impl_swizzle!(zxyy() => Vector4 => Vector4 { 2, 0, 1, 1 });
impl_swizzle!(zxyz() => Vector4 => Vector4 { 2, 0, 1, 2 });
impl_swizzle!(zxyw() => Vector4 => Vector4 { 2, 0, 1, 3 });
impl_swizzle!(zxzx() => Vector4 => Vector4 { 2, 0, 2, 0 });
impl_swizzle!(zxzy() => Vector4 => Vector4 { 2, 0, 2, 1 });
impl_swizzle!(zxzz() => Vector4 => Vector4 { 2, 0, 2, 2 });
impl_swizzle!(zxzw() => Vector4 => Vector4 { 2, 0, 2, 3 });
impl_swizzle!(zxwx() => Vector4 => Vector4 { 2, 0, 3, 0 });
impl_swizzle!(zxwy() => Vector4 => Vector4 { 2, 0, 3, 1 });
impl_swizzle!(zxwz() => Vector4 => Vector4 { 2, 0, 3, 2 });
impl_swizzle!(zxww() => Vector4 => Vector4 { 2, 0, 3, 3 });
impl_swizzle!(zyxx() => Vector4 => Vector4 { 2, 1, 0, 0 });
impl_swizzle!(zyxy() => Vector4 => Vector4 { 2, 1, 0, 1 });
impl_swizzle!(zyxz() => Vector4 => Vector4 { 2, 1, 0, 2 });
impl_swizzle!(zyxw() => Vector4 => Vector4 { 2, 1, 0, 3 });
impl_swizzle!(zyyx() => Vector4 => Vector4 { 2, 1, 1, 0 });
impl_swizzle!(zyyy() => Vector4 => Vector4 { 2, 1, 1, 1 });
impl_swizzle!(zyyz() => Vector4 => Vector4 { 2, 1, 1, 2 });
impl_swizzle!(zyyw() => Vector4 => Vector4 { 2, 1, 1, 3 });
impl_swizzle!(zyzx() => Vector4 => Vector4 { 2, 1, 2, 0 });
impl_swizzle!(zyzy() => Vector4 => Vector4 { 2, 1, 2, 1 });
impl_swizzle!(zyzz() => Vector4 => Vector4 { 2, 1, 2, 2 });
impl_swizzle!(zyzw() => Vector4 => Vector4 { 2, 1, 2, 3 });
impl_swizzle!(zywx() => Vector4 => Vector4 { 2, 1, 3, 0 });
impl_swizzle!(zywy() => Vector4 => Vector4 { 2, 1, 3, 1 });
impl_swizzle!(zywz() => Vector4 => Vector4 { 2, 1, 3, 2 });
impl_swizzle!(zyww() => Vector4 => Vector4 { 2, 1, 3, 3 });
impl_swizzle!(zzxx() => Vector4 => Vector4 { 2, 2, 0, 0 });
impl_swizzle!(zzxy() => Vector4 => Vector4 { 2, 2, 0, 1 });
impl_swizzle!(zzxz() => Vector4 => Vector4 { 2, 2, 0, 2 });
impl_swizzle!(zzxw() => Vector4 => Vector4 { 2, 2, 0, 3 });
impl_swizzle!(zzyx() => Vector4 => Vector4 { 2, 2, 1, 0 });
impl_swizzle!(zzyy() => Vector4 => Vector4 { 2, 2, 1, 1 });
impl_swizzle!(zzyz() => Vector4 => Vector4 { 2, 2, 1, 2 });
impl_swizzle!(zzyw() => Vector4 => Vector4 { 2, 2, 1, 3 });
impl_swizzle!(zzzx() => Vector4 => Vector4 { 2, 2, 2, 0 });
impl_swizzle!(zzzy() => Vector4 => Vector4 { 2, 2, 2, 1 });
impl_swizzle!(zzzz() => Vector4 => Vector4 { 2, 2, 2, 2 });
impl_swizzle!(zzzw() => Vector4 => Vector4 { 2, 2, 2, 3 });
impl_swizzle!(zzwx() => Vector4 => Vector4 { 2, 2, 3, 0 });
impl_swizzle!(zzwy() => Vector4 => Vector4 { 2, 2, 3, 1 });
impl_swizzle!(zzwz() => Vector4 => Vector4 { 2, 2, 3, 2 });
impl_swizzle!(zzww() => Vector4 => Vector4 { 2, 2, 3, 3 });
impl_swizzle!(zwxx() => Vector4 => Vector4 { 2, 3, 0, 0 });
impl_swizzle!(zwxy() => Vector4 => Vector4 { 2, 3, 0, 1 });
impl_swizzle!(zwxz() => Vector4 => Vector4 { 2, 3, 0, 2 });
impl_swizzle!(zwxw() => Vector4 => Vector4 { 2, 3, 0, 3 });
impl_swizzle!(zwyx() => Vector4 => Vector4 { 2, 3, 1, 0 });
impl_swizzle!(zwyy() => Vector4 => Vector4 { 2, 3, 1, 1 });
impl_swizzle!(zwyz() => Vector4 => Vector4 { 2, 3, 1, 2 });
impl_swizzle!(zwyw() => Vector4 => Vector4 { 2, 3, 1, 3 });
impl_swizzle!(zwzx() => Vector4 => Vector4 { 2, 3, 2, 0 });
impl_swizzle!(zwzy() => Vector4 => Vector4 { 2, 3, 2, 1 });
impl_swizzle!(zwzz() => Vector4 => Vector4 { 2, 3, 2, 2 });
impl_swizzle!(zwzw() => Vector4 => Vector4 { 2, 3, 2, 3 });
impl_swizzle!(zwwx() => Vector4 => Vector4 { 2, 3, 3, 0 });
impl_swizzle!(zwwy() => Vector4 => Vector4 { 2, 3, 3, 1 });
impl_swizzle!(zwwz() => Vector4 => Vector4 { 2, 3, 3, 2 });
impl_swizzle!(zwww() => Vector4 => Vector4 { 2, 3, 3, 3 });
impl_swizzle!(wxxx() => Vector4 => Vector4 { 3, 0, 0, 0 });
impl_swizzle!(wxxy() => Vector4 => Vector4 { 3, 0, 0, 1 });
impl_swizzle!(wxxz() => Vector4 => Vector4 { 3, 0, 0, 2 });
impl_swizzle!(wxxw() => Vector4 => Vector4 { 3, 0, 0, 3 });
impl_swizzle!(wxyx() => Vector4 => Vector4 { 3, 0, 1, 0 });
impl_swizzle!(wxyy() => Vector4 => Vector4 { 3, 0, 1, 1 });
impl_swizzle!(wxyz() => Vector4 => Vector4 { 3, 0, 1, 2 });
impl_swizzle!(wxyw() => Vector4 => Vector4 { 3, 0, 1, 3 });
impl_swizzle!(wxzx() => Vector4 => Vector4 { 3, 0, 2, 0 });
impl_swizzle!(wxzy() => Vector4 => Vector4 { 3, 0, 2, 1 });
impl_swizzle!(wxzz() => Vector4 => Vector4 { 3, 0, 2, 2 });
impl_swizzle!(wxzw() => Vector4 => Vector4 { 3, 0, 2, 3 });
impl_swizzle!(wxwx() => Vector4 => Vector4 { 3, 0, 3, 0 });
impl_swizzle!(wxwy() => Vector4 => Vector4 { 3, 0, 3, 1 });
impl_swizzle!(wxwz() => Vector4 => Vector4 { 3, 0, 3, 2 });
impl_swizzle!(wxww() => Vector4 => Vector4 { 3, 0, 3, 3 });
impl_swizzle!(wyxx() => Vector4 => Vector4 { 3, 1, 0, 0 });
impl_swizzle!(wyxy() => Vector4 => Vector4 { 3, 1, 0, 1 });
impl_swizzle!(wyxz() => Vector4 => Vector4 { 3, 1, 0, 2 });
impl_swizzle!(wyxw() => Vector4 => Vector4 { 3, 1, 0, 3 });
impl_swizzle!(wyyx() => Vector4 => Vector4 { 3, 1, 1, 0 });
impl_swizzle!(wyyy() => Vector4 => Vector4 { 3, 1, 1, 1 });
impl_swizzle!(wyyz() => Vector4 => Vector4 { 3, 1, 1, 2 });
impl_swizzle!(wyyw() => Vector4 => Vector4 { 3, 1, 1, 3 });
impl_swizzle!(wyzx() => Vector4 => Vector4 { 3, 1, 2, 0 });
impl_swizzle!(wyzy() => Vector4 => Vector4 { 3, 1, 2, 1 });
impl_swizzle!(wyzz() => Vector4 => Vector4 { 3, 1, 2, 2 });
impl_swizzle!(wyzw() => Vector4 => Vector4 { 3, 1, 2, 3 });
impl_swizzle!(wywx() => Vector4 => Vector4 { 3, 1, 3, 0 });
impl_swizzle!(wywy() => Vector4 => Vector4 { 3, 1, 3, 1 });
impl_swizzle!(wywz() => Vector4 => Vector4 { 3, 1, 3, 2 });
impl_swizzle!(wyww() => Vector4 => Vector4 { 3, 1, 3, 3 });
impl_swizzle!(wzxx() => Vector4 => Vector4 { 3, 2, 0, 0 });
impl_swizzle!(wzxy() => Vector4 => Vector4 { 3, 2, 0, 1 });
impl_swizzle!(wzxz() => Vector4 => Vector4 { 3, 2, 0, 2 });
impl_swizzle!(wzxw() => Vector4 => Vector4 { 3, 2, 0, 3 });
impl_swizzle!(wzyx() => Vector4 => Vector4 { 3, 2, 1, 0 });
impl_swizzle!(wzyy() => Vector4 => Vector4 { 3, 2, 1, 1 });
impl_swizzle!(wzyz() => Vector4 => Vector4 { 3, 2, 1, 2 });
impl_swizzle!(wzyw() => Vector4 => Vector4 { 3, 2, 1, 3 });
impl_swizzle!(wzzx() => Vector4 => Vector4 { 3, 2, 2, 0 });
impl_swizzle!(wzzy() => Vector4 => Vector4 { 3, 2, 2, 1 });
impl_swizzle!(wzzz() => Vector4 => Vector4 { 3, 2, 2, 2 });
impl_swizzle!(wzzw() => Vector4 => Vector4 { 3, 2, 2, 3 });
impl_swizzle!(wzwx() => Vector4 => Vector4 { 3, 2, 3, 0 });
impl_swizzle!(wzwy() => Vector4 => Vector4 { 3, 2, 3, 1 });
impl_swizzle!(wzwz() => Vector4 => Vector4 { 3, 2, 3, 2 });
impl_swizzle!(wzww() => Vector4 => Vector4 { 3, 2, 3, 3 });
impl_swizzle!(wwxx() => Vector4 => Vector4 { 3, 3, 0, 0 });
impl_swizzle!(wwxy() => Vector4 => Vector4 { 3, 3, 0, 1 });
impl_swizzle!(wwxz() => Vector4 => Vector4 { 3, 3, 0, 2 });
impl_swizzle!(wwxw() => Vector4 => Vector4 { 3, 3, 0, 3 });
impl_swizzle!(wwyx() => Vector4 => Vector4 { 3, 3, 1, 0 });
impl_swizzle!(wwyy() => Vector4 => Vector4 { 3, 3, 1, 1 });
impl_swizzle!(wwyz() => Vector4 => Vector4 { 3, 3, 1, 2 });
impl_swizzle!(wwyw() => Vector4 => Vector4 { 3, 3, 1, 3 });
impl_swizzle!(wwzx() => Vector4 => Vector4 { 3, 3, 2, 0 });
impl_swizzle!(wwzy() => Vector4 => Vector4 { 3, 3, 2, 1 });
impl_swizzle!(wwzz() => Vector4 => Vector4 { 3, 3, 2, 2 });
impl_swizzle!(wwzw() => Vector4 => Vector4 { 3, 3, 2, 3 });
impl_swizzle!(wwwx() => Vector4 => Vector4 { 3, 3, 3, 0 });
impl_swizzle!(wwwy() => Vector4 => Vector4 { 3, 3, 3, 1 });
impl_swizzle!(wwwz() => Vector4 => Vector4 { 3, 3, 3, 2 });
impl_swizzle!(wwww() => Vector4 => Vector4 { 3, 3, 3, 3 });

