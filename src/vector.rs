use scalar::{
    Scalar,
    ScalarSigned,
    ScalarFloat,   
};
use traits::{
    Array,
    CrossProduct,
    Zero,
    ProjectOn,
    DotProduct,
    Magnitude,
    Lerp,
    Metric,
    Finite,
    Sum,
    Product,
};
use num_traits::NumCast;
use std::fmt;
use std::iter;
use std::mem;
use std::ops;


/// Construct a new one-dimensional vector. This follows the style of
/// other GLSL vector constructors even though GLSL itself lacks a
/// `vec1()` function.
#[inline]
pub fn vec1<S, T: Into<Vector1<S>>>(vector: T) -> Vector1<S> {
    vector.into()
}

/// Construct a new two-dimensional vector in the style of
/// a GLSL `vec2` constructor.
#[inline]
pub fn vec2<S, T: Into<Vector2<S>>>(vector: T) -> Vector2<S> {
    vector.into()
}

/// Construct a new three-dimensional vector in the style of
/// a GLSL `vec3` constructor.
#[inline]
pub fn vec3<S, T: Into<Vector3<S>>>(vector: T) -> Vector3<S> {
    vector.into()
}

/// Construct a new four-dimensional vector in the style of
/// a GLSL `vec4` constructor.
#[inline]
pub fn vec4<S, T: Into<Vector4<S>>>(vector: T) -> Vector4<S> {
    vector.into()
}

/// Compute the dot product between two vectors.
#[inline]
pub fn dot<W: Copy + Clone, V: DotProduct<W>>(v1: V, v2: W) -> <V as DotProduct<W>>::Output {
    V::dot(v1, v2)
}


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
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Vector1<S> {
    pub x: S,
}

impl<S> Vector1<S> {
    /// Construct a new vector.
    #[inline]
    pub const fn new(x: S) -> Vector1<S> {
        Vector1 { x: x }
    }

    /// Map an operation on the elements of a vector, returning a vector of the 
    /// new underlying type.
    pub fn map<T, F>(self, mut op: F) -> Vector1<T> where F: FnMut(S) -> T {
        Vector1 { x: op(self.x) }
    }
}

impl<S> Vector1<S> where S: NumCast + Copy {
    /// Cast a vector from one type of scalars to another type of scalars.
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
    /// the supplied value.
    #[inline]
    pub fn extend(self, y: S) -> Vector2<S> {
        Vector2::new(self.x, y)
    }
}

impl<S> Vector1<S> where S: Scalar {
    /// The unit vector representing the x-direction.
    #[inline]
    pub fn unit_x() -> Vector1<S> {
        Vector1 { x: S::one() }
    }
}

impl<S> Metric<Vector1<S>> for Vector1<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: Vector1<S>) -> S {
        let dx_squared = (to.x - self.x) * (to.x - self.x);

        dx_squared
    }
}

impl<S> Metric<&Vector1<S>> for Vector1<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: &Vector1<S>) -> S {
        let dx_squared = (to.x - self.x) * (to.x - self.x);

        dx_squared
    }
}

impl<S> Metric<Vector1<S>> for &Vector1<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: Vector1<S>) -> S {
        let dx_squared = (to.x - self.x) * (to.x - self.x);

        dx_squared
    }
}

impl<'a, 'b, S> Metric<&'a Vector1<S>> for &'b Vector1<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: &'a Vector1<S>) -> S {
        let dx_squared = (to.x - self.x) * (to.x - self.x);

        dx_squared
    }
}

impl<S> Array for Vector1<S> where S: Copy {
    type Element = S;

    #[inline]
    fn len() -> usize {
        1
    }

    #[inline]
    fn shape() -> (usize, usize) {
        (1, 1)
    }

    #[inline]
    fn as_ptr(&self) -> *const Self::Element {
        &self.x
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Self::Element {
        &mut self.x
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Element] {
        <Self as AsRef<[Self::Element; 1]>>::as_ref(self)
    }
}

impl<S> Sum for Vector1<S> where S: Scalar {
    #[inline]
    fn sum(&self) -> S {
        self.x
    }
}

impl<S> Product for Vector1<S> where S: Scalar {
    #[inline]
    fn product(&self) -> S {
        self.x
    }
}

impl<S> AsRef<[S; 1]> for Vector1<S> {
    fn as_ref(&self) -> &[S; 1] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsRef<S> for Vector1<S> {
    fn as_ref(&self) -> &S {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsRef<(S,)> for Vector1<S> {
    fn as_ref(&self) -> &(S,) {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[S; 1]> for Vector1<S> {
    fn as_mut(&mut self) -> &mut [S; 1] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<S> for Vector1<S> {
    fn as_mut(&mut self) -> &mut S {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<(S,)> for Vector1<S> {
    fn as_mut(&mut self) -> &mut (S,) {
        unsafe { mem::transmute(self) }
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

impl<S> fmt::Debug for Vector1<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector1 ")?;
        <[S; 1] as fmt::Debug>::fmt(self.as_ref(), f)
    }
}

impl<S> fmt::Display for Vector1<S> where S: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector1 [{:.2}]", self.x)
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
        unsafe { mem::transmute(v) }
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

    fn add(self, other: Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x + other.x,
        }
    }
}

impl<S> ops::Add<Vector1<S>> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn add(self, other: Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x + other.x,
        }
    }
}

impl<S> ops::Add<&Vector1<S>> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn add(self, other: &Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x + other.x,
        }
    }
}

impl<'a, 'b, S> ops::Add<&'b Vector1<S>> for &'a Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn add(self, other: &'b Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x + other.x,
        }
    }
}

impl<S> ops::Sub<Vector1<S>> for &Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn sub(self, other: Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl<S> ops::Sub<Vector1<S>> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn sub(self, other: Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl<S> ops::Sub<&Vector1<S>> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn sub(self, other: &Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl<'a, 'b, S> ops::Sub<&'b Vector1<S>> for &'a Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn sub(self, other: &'b Vector1<S>) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl<S> ops::AddAssign<Vector1<S>> for Vector1<S> where S: Scalar {
    fn add_assign(&mut self, other: Vector1<S>) {
        self.x = self.x + other.x;
    }
}

impl<S> ops::AddAssign<&Vector1<S>> for Vector1<S> where S: Scalar {
    fn add_assign(&mut self, other: &Vector1<S>) {
        self.x = self.x + other.x;
    }
}

impl<S> ops::SubAssign<Vector1<S>> for Vector1<S> where S: Scalar {
    fn sub_assign(&mut self, other: Vector1<S>) {
        self.x = self.x - other.x;
    }
}

impl<S> ops::SubAssign<&Vector1<S>> for Vector1<S> where S: Scalar {
    fn sub_assign(&mut self, other: &Vector1<S>) {
        self.x = self.x - other.x;
    }
}

impl<S> ops::Mul<S> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn mul(self, other: S) -> Self::Output {
        Vector1 {
            x: self.x * other,
        }
    }
}

impl<S> ops::Mul<S> for &Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn mul(self, other: S) -> Self::Output {
        Vector1 {
            x: self.x * other,
        }
    }
}

impl<S> ops::MulAssign<S> for Vector1<S> where S: Scalar {
    fn mul_assign(&mut self, other: S) {
        self.x *= other;
    }
}

impl<S> ops::Div<S> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn div(self, other: S) -> Self::Output {
        Vector1 {
            x: self.x / other,
        }
    }
}

impl<S> ops::Div<S> for &Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn div(self, other: S) -> Self::Output {
        Vector1 {
            x: self.x / other,
        }
    }
}

impl<S> ops::DivAssign<S> for Vector1<S> where S: Scalar {
    fn div_assign(&mut self, other: S) {
        self.x = self.x / other;
    }
}

impl<S> ops::Rem<S> for Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        
        Vector1::new(x)
    }
}

impl<S> ops::Rem<S> for &Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        
        Vector1::new(x)
    }
}

impl<S> ops::RemAssign<S> for Vector1<S> where S: Scalar {
    fn rem_assign(&mut self, other: S) {
        self.x %= other;
    }
}

impl<S> Zero for Vector1<S> where S: Scalar {
    fn zero() -> Vector1<S> {
        Vector1 { 
            x: S::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.x == S::zero()
    }
}

impl<S> DotProduct<Vector1<S>> for Vector1<S> where S: Scalar {
    type Output = S; 

    fn dot(self, other: Vector1<S>) -> Self::Output {
        self.x * other.x
    }
}

impl<S> DotProduct<&Vector1<S>> for Vector1<S> where S: Scalar {
    type Output = S; 

    fn dot(self, other: &Vector1<S>) -> Self::Output {
        self.x * other.x
    }
}

impl<S> DotProduct<Vector1<S>> for &Vector1<S> where S: Scalar {
    type Output = S; 

    fn dot(self, other: Vector1<S>) -> Self::Output {
        self.x * other.x
    }
}

impl<S> DotProduct<&Vector1<S>> for &Vector1<S> where S: Scalar {
    type Output = S; 

    fn dot(self, other: &Vector1<S>) -> Self::Output {
        self.x * other.x
    }
}


impl<S> Lerp<Vector1<S>> for Vector1<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector1<S>;

    fn lerp(self, other: Vector1<S>, amount: S) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<&Vector1<S>> for Vector1<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector1<S>;

    fn lerp(self, other: &Vector1<S>, amount: S) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<Vector1<S>> for &Vector1<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector1<S>;

    fn lerp(self, other: Vector1<S>, amount: S) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<'a, 'b, S> Lerp<&'a Vector1<S>> for &'b Vector1<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector1<S>;

    fn lerp(self, other: &'a Vector1<S>, amount: S) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Magnitude for Vector1<S> where S: ScalarFloat {
    type Output = S;
    
    fn magnitude_squared(&self) -> Self::Output {
        DotProduct::dot(self, self)
    }

    #[inline]
    fn magnitude(&self) -> Self::Output {
        S::sqrt(DotProduct::dot(self, self))
    }
    
    fn normalize(&self) -> Self {
        self / self.magnitude()
    }
    
    fn normalize_to(&self, magnitude: Self::Output) -> Self {
        self * (magnitude / self.magnitude())
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

impl<S> Finite for Vector1<S> where S: ScalarFloat {
    fn is_finite(self) -> bool {
        self.x.is_finite()
    }
}

impl<S> Finite for &Vector1<S> where S: ScalarFloat {
    fn is_finite(self) -> bool {
        self.x.is_finite()
    }
}

impl<S> ProjectOn<Vector1<S>> for Vector1<S> where S: ScalarFloat {
    type Output = Vector1<S>;

    fn project_on(self, onto: Vector1<S>) -> Vector1<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<S> ProjectOn<&Vector1<S>> for Vector1<S> where S: ScalarFloat {
    type Output = Vector1<S>;

    fn project_on(self, onto: &Vector1<S>) -> Vector1<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<S> ProjectOn<Vector1<S>> for &Vector1<S> where S: ScalarFloat {
    type Output = Vector1<S>;

    fn project_on(self, onto: Vector1<S>) -> Vector1<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<'a, 'b, S> ProjectOn<&'a Vector1<S>> for &'b Vector1<S> where S: ScalarFloat {
    type Output = Vector1<S>;

    fn project_on(self, onto: &'a Vector1<S>) -> Vector1<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}


/// A representation of two-dimensional vectors in a Euclidean space.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(C)]
pub struct Vector2<S> {
    /// The horizontal component.
    pub x: S,
    /// The vertical component.
    pub y: S,
}

impl<S> Vector2<S> {
    /// Construct a new vector.
    pub const fn new(x: S, y: S) -> Vector2<S> {
        Vector2 { 
            x: x, 
            y: y 
        }
    }

    /// Map an operation on the elements of a vector, returning a vector of the 
    /// new underlying type.
    pub fn map<T, F>(self, mut op: F) -> Vector2<T> where F: FnMut(S) -> T {
        Vector2 {
            x: op(self.x),
            y: op(self.y),
        }
    }
}

impl<S> Vector2<S> where S: NumCast + Copy {
    /// Cast a vector from one type of scalars to another type of scalars.
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
    /// supplied z value.
    #[inline]
    pub fn expand(self, z: S) -> Vector3<S> {
        Vector3::new(self.x, self.y, z)
    }

    /// Contract a two-dimensional vector to a one-dimensional vector removing
    /// the y-component.
    #[inline]
    pub fn contract(self) -> Vector1<S> {
        Vector1::new(self.x)
    }
}

impl<S> Vector2<S> where S: Scalar {
    /// Returns the unit x axis vector.
    #[inline]
    pub fn unit_x() -> Vector2<S> {
        Vector2 { 
            x: S::one(), 
            y: S::zero(),
        }
    }

    /// Returns the unit y axis vector.
    #[inline]
    pub fn unit_y() -> Vector2<S> {
        Vector2 { 
            x: S::zero(), 
            y: S::one(),
        }
    }
}

impl<S> Array for Vector2<S> where S: Copy {
    type Element = S;

    #[inline]
    fn len() -> usize {
        2
    }

    #[inline]
    fn shape() -> (usize, usize) {
        (2, 1)
    }

    #[inline]
    fn as_ptr(&self) -> *const Self::Element {
        &self.x
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Self::Element {
        &mut self.x
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Element] {
        <Self as AsRef<[Self::Element; 2]>>::as_ref(self)
    }
}

impl<S> Sum for Vector2<S> where S: Scalar {
    #[inline]
    fn sum(&self) -> S {
        self.x + self.y
    }
}

impl<S> Product for Vector2<S> where S: Scalar {
    #[inline]
    fn product(&self) -> S {
        self.x * self.y
    }
}

impl<S> AsRef<[S; 2]> for Vector2<S> {
    fn as_ref(&self) -> &[S; 2] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsRef<(S, S)> for Vector2<S> {
    fn as_ref(&self) -> &(S, S) {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[S; 2]> for Vector2<S> {
    fn as_mut(&mut self) -> &mut [S; 2] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<(S, S)> for Vector2<S> {
    fn as_mut(&mut self) -> &mut (S, S) {
        unsafe { mem::transmute(self) }
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

impl<S> fmt::Debug for Vector2<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector2 ")?;
        <[S; 2] as fmt::Debug>::fmt(self.as_ref(), f)
    }
}

impl<S> fmt::Display for Vector2<S> where S: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector2 [{:.2}, {:.2}]", self.x, self.y)
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
        unsafe { mem::transmute(v) }
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

    fn add(self, other: Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<S> ops::Add<Vector2<S>> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn add(self, other: Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<S> ops::Add<&Vector2<S>> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn add(self, other: &Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<'a, 'b, S> ops::Add<&'b Vector2<S>> for &'a Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn add(self, other: &'b Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<S> ops::Sub<Vector2<S>> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn sub(self, other: Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S> ops::Sub<Vector2<S>> for &Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn sub(self, other: Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S> ops::Sub<&Vector2<S>> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn sub(self, other: &Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<'a, 'b, S> ops::Sub<&'b Vector2<S>> for &'a Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn sub(self, other: &'b Vector2<S>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<S> ops::AddAssign<Vector2<S>> for Vector2<S> where S: Scalar {
    fn add_assign(&mut self, other: Vector2<S>) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
    }
}

impl<S> ops::AddAssign<&Vector2<S>> for Vector2<S> where S: Scalar {
    fn add_assign(&mut self, other: &Vector2<S>) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
    }
}

impl<S> ops::SubAssign<Vector2<S>> for Vector2<S> where S: Scalar {
    fn sub_assign(&mut self, other: Vector2<S>) {
        self.x = self.x - other.x;
        self.y = self.y - other.y;
    }
}

impl<S> ops::SubAssign<&Vector2<S>> for Vector2<S> where S: Scalar {
    fn sub_assign(&mut self, other: &Vector2<S>) {
        self.x = self.x - other.x;
        self.y = self.y - other.y;
    }
}

impl<S> ops::Mul<S> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn mul(self, other: S) -> Self::Output {
        Vector2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl<S> ops::Mul<S> for &Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

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
    fn mul_assign(&mut self, other: S) {
        self.x *= other;
        self.y *= other;
    }
}

impl<S> ops::Div<S> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn div(self, other: S) -> Self::Output {
        Vector2 {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

impl<S> ops::Div<S> for &Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn div(self, other: S) -> Self::Output {
        Vector2 {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

impl<S> ops::DivAssign<S> for Vector2<S> where S: Scalar {
    fn div_assign(&mut self, other: S) {
        self.x = self.x / other;
        self.y = self.y / other;
    }
}

impl<S> ops::Rem<S> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        
        Vector2::new(x, y)
    }
}

impl<S> ops::Rem<S> for &Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        
        Vector2::new(x, y)
    }
}

impl<S> ops::RemAssign<S> for Vector2<S> where S: Scalar {
    fn rem_assign(&mut self, other: S) {
        self.x %= other;
        self.y %= other;
    }
}

impl<S> Zero for Vector2<S> where S: Scalar {
    fn zero() -> Vector2<S> {
        Vector2 { 
            x: S::zero(), 
            y: S::zero(), 
        }
    }

    fn is_zero(&self) -> bool {
        self.x == S::zero() && self.y == S::zero()
    }
}

impl<S> Metric<Vector2<S>> for Vector2<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: Vector2<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
    
        dx_squared + dy_squared
    }
}

impl<S> Metric<&Vector2<S>> for Vector2<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: &Vector2<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
    
        dx_squared + dy_squared
    }
}

impl<S> Metric<Vector2<S>> for &Vector2<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: Vector2<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
    
        dx_squared + dy_squared
    }
}

impl<'a, 'b, S> Metric<&'a Vector2<S>> for &'b Vector2<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: &'a Vector2<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
    
        dx_squared + dy_squared
    }
}

impl<S> DotProduct<Vector2<S>> for Vector2<S> where S: Scalar {
    type Output = S;

    #[inline]
    fn dot(self, other: Vector2<S>) -> Self::Output {
        self.x * other.x + self.y * other.y
    }
}

impl<S> DotProduct<&Vector2<S>> for Vector2<S> where S: Scalar {
    type Output = S;

    #[inline]
    fn dot(self, other: &Vector2<S>) -> Self::Output {
        self.x * other.x + self.y * other.y
    }
}

impl<S> DotProduct<Vector2<S>> for &Vector2<S> where S: Scalar {
    type Output = S;

    #[inline]
    fn dot(self, other: Vector2<S>) -> Self::Output {
        self.x * other.x + self.y * other.y
    }
}

impl<'a, 'b, S> DotProduct<&'a Vector2<S>> for &'b Vector2<S> where S: Scalar {
    type Output = S;

    #[inline]
    fn dot(self, other: &'a Vector2<S>) -> Self::Output {
        self.x * other.x + self.y * other.y
    }
}

impl<S> Lerp<Vector2<S>> for Vector2<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector2<S>;

    fn lerp(self, other: Vector2<S>, amount: S) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<&Vector2<S>> for Vector2<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector2<S>;

    fn lerp(self, other: &Vector2<S>, amount: S) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<Vector2<S>> for &Vector2<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector2<S>;

    fn lerp(self, other: Vector2<S>, amount: S) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<'a, 'b, S> Lerp<&'a Vector2<S>> for &'b Vector2<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector2<S>;

    fn lerp(self, other: &'a Vector2<S>, amount: S) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Magnitude for Vector2<S> where S: ScalarFloat {
    type Output = S;

    fn magnitude(&self) -> Self::Output {
        Self::Output::sqrt(self.magnitude_squared())
    }

    fn magnitude_squared(&self) -> Self::Output {
        DotProduct::dot(self, self)
    }

    fn normalize(&self) -> Self {
        self / self.magnitude()
    }

    fn normalize_to(&self, magnitude: Self::Output) -> Self {
        self * (magnitude / self.magnitude())
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


impl<S> Finite for Vector2<S> where S: ScalarFloat {
    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

impl<S> Finite for &Vector2<S> where S: ScalarFloat {
    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

impl<S> ProjectOn<Vector2<S>> for Vector2<S> where S: ScalarFloat {
    type Output = Vector2<S>;

    fn project_on(self, onto: Vector2<S>) -> Vector2<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<S> ProjectOn<&Vector2<S>> for Vector2<S> where S: ScalarFloat {
    type Output = Vector2<S>;

    fn project_on(self, onto: &Vector2<S>) -> Vector2<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<S> ProjectOn<Vector2<S>> for &Vector2<S> where S: ScalarFloat {
    type Output = Vector2<S>;

    fn project_on(self, onto: Vector2<S>) -> Vector2<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<'a, 'b, S> ProjectOn<&'a Vector2<S>> for &'b Vector2<S> where S: ScalarFloat {
    type Output = Vector2<S>;

    fn project_on(self, onto: &'a Vector2<S>) -> Vector2<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}


/// A representation of three-dimensional vectors in a Euclidean space.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(C)]
pub struct Vector3<S> {
    pub x: S,
    pub y: S,
    pub z: S,
}

impl<S> Vector3<S> {
    /// Construct a new vector.
    pub const fn new(x: S, y: S, z: S) -> Vector3<S> {
        Vector3 { x: x, y: y, z: z }
    }

    /// Map an operation on the elements of a vector, returning a vector of the 
    /// new underlying type.
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
    /// Extend a three-dimensional vector to a four-dimensional vector 
    /// by supplying w-component.
    #[inline]
    pub fn extend(self, w: S) -> Vector4<S> {
        Vector4::new(self.x, self.y, self.z, w)
    }

    /// Contract a three-dimensional vector to a two-dimensional vector
    /// by removing the z-component.
    #[inline]
    pub fn contract(self) -> Vector2<S> {
        Vector2::new(self.x, self.y)
    }
}

impl<S> Vector3<S> where S: Scalar {
    /// Returns the unit x axis vector.
    #[inline]
    pub fn unit_x() -> Vector3<S> {
        Vector3 { 
            x: S::one(), 
            y: S::zero(), 
            z: S::zero(),
        }
    }

    /// Returns the unit y axis vector.
    #[inline]
    pub fn unit_y() -> Vector3<S> {
        Vector3 { 
            x: S::zero(), 
            y: S::one(), 
            z: S::zero(),
        }
    }
    
    /// Returns the unit z axis vector.
    #[inline]
    pub fn unit_z() -> Vector3<S> {
        Vector3 { 
            x: S::zero(),
            y: S::zero(), 
            z: S::one(),
        }
    }
}

impl<S> Array for Vector3<S> where S: Copy {
    type Element = S;

    #[inline]
    fn len() -> usize {
        3
    }

    #[inline]
    fn shape() -> (usize, usize) {
        (3, 1)
    }

    #[inline]
    fn as_ptr(&self) -> *const Self::Element {
        &self.x
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Self::Element {
        &mut self.x
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Element] {
        <Self as AsRef<[Self::Element; 3]>>::as_ref(self)
    }
}

impl<S> Sum for Vector3<S> where S: Scalar {
    #[inline]
    fn sum(&self) -> S {
        self.x + self.y + self.z
    }
}

impl<S> Product for Vector3<S> where S: Scalar {
    #[inline]
    fn product(&self) -> S {
        self.x * self.y * self.z
    }
}


impl<S> AsRef<[S; 3]> for Vector3<S> {
    fn as_ref(&self) -> &[S; 3] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsRef<(S, S, S)> for Vector3<S> {
    fn as_ref(&self) -> &(S, S, S) {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[S; 3]> for Vector3<S> {
    fn as_mut(&mut self) -> &mut [S; 3] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<(S, S, S)> for Vector3<S> {
    fn as_mut(&mut self) -> &mut (S, S, S) {
        unsafe { mem::transmute(self) }
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

impl<S> fmt::Debug for Vector3<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector3 ")?;
        <[S; 3] as fmt::Debug>::fmt(self.as_ref(), f)
    }
}

impl<S> fmt::Display for Vector3<S> where S: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector3 [{:.2}, {:.2}, {:.2}]", self.x, self.y, self.z)
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
        unsafe { mem::transmute(v) }
    }
}

impl<'a, S> From<&'a (S, S, S)> for &'a Vector3<S> where S: Scalar {
    #[inline]
    fn from(v: &'a (S, S, S)) -> &'a Vector3<S> {
        unsafe { mem::transmute(v) }
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

    fn sub(self, other: &'b Vector3<S>) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}


impl<S> ops::AddAssign<Vector3<S>> for Vector3<S> where S: Scalar {
    fn add_assign(&mut self, other: Vector3<S>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<S> ops::AddAssign<&Vector3<S>> for Vector3<S> where S: Scalar {
    fn add_assign(&mut self, other: &Vector3<S>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<S> ops::SubAssign<Vector3<S>> for Vector3<S> where S: Scalar {
    fn sub_assign(&mut self, other: Vector3<S>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<S> ops::SubAssign<&Vector3<S>> for Vector3<S> where S: Scalar {
    fn sub_assign(&mut self, other: &Vector3<S>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<S> ops::Mul<S> for Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

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
    fn mul_assign(&mut self, other: S) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
    }
}

impl<S> ops::Div<S> for Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

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

    fn div(self, other: S) -> Self::Output {
        Vector3 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

impl<S> ops::DivAssign<S> for Vector3<S> where S: Scalar {
    fn div_assign(&mut self, other: S) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
    }
}

impl<S> ops::Rem<S> for Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        
        Vector3::new(x, y, z)
    }
}

impl<S> ops::Rem<S> for &Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        
        Vector3::new(x, y, z)
    }
}

impl<S> ops::RemAssign<S> for Vector3<S> where S: Scalar {
    fn rem_assign(&mut self, other: S) {
        self.x %= other;
        self.y %= other;
        self.z %= other;
    }
}

impl<S> Zero for Vector3<S> where S: Scalar {
    #[inline]
    fn zero() -> Vector3<S> {
        Vector3 { 
            x: S::zero(), 
            y: S::zero(), 
            z: S::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.x == S::zero() && self.y == S::zero() && self.z == S::zero()
    }
}

impl<S> Metric<Vector3<S>> for Vector3<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: Vector3<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
        let dz_squared = (to.z - self.z) * (to.z - self.z);
    
        dx_squared + dy_squared + dz_squared
    }
}

impl<S> Metric<&Vector3<S>> for Vector3<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: &Vector3<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
        let dz_squared = (to.z - self.z) * (to.z - self.z);
    
        dx_squared + dy_squared + dz_squared
    }
}

impl<S> Metric<Vector3<S>> for &Vector3<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: Vector3<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
        let dz_squared = (to.z - self.z) * (to.z - self.z);
    
        dx_squared + dy_squared + dz_squared
    }
}

impl<'a, 'b, S> Metric<&'a Vector3<S>> for &'b Vector3<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: &Vector3<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
        let dz_squared = (to.z - self.z) * (to.z - self.z);
    
        dx_squared + dy_squared + dz_squared
    }
}

impl<S> DotProduct<Vector3<S>> for Vector3<S> where S: Scalar {
    type Output = S;

    #[inline]
    fn dot(self, other: Vector3<S>) -> Self::Output {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl<S> DotProduct<&Vector3<S>> for Vector3<S> where S: Scalar {
    type Output = S;

    #[inline]
    fn dot(self, other: &Vector3<S>) -> Self::Output {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl<S> DotProduct<Vector3<S>> for &Vector3<S> where S: Scalar {
    type Output = S;

    #[inline]
    fn dot(self, other: Vector3<S>) -> Self::Output {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl<'a, 'b, S> DotProduct<&'a Vector3<S>> for &'b Vector3<S> where S: Scalar {
    type Output = S;
    
    #[inline]
    fn dot(self, other: &'a Vector3<S>) -> Self::Output {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl<S> CrossProduct<Vector3<S>> for Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn cross(self, other: Vector3<S>) -> Self::Output {
        let x = self.y * other.z - self.z * other.y;
        let y = self.z * other.x - self.x * other.z;
        let z = self.x * other.y - self.y * other.x;

        Vector3::new(x, y, z)
    }
}

impl<S> CrossProduct<&Vector3<S>> for Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn cross(self, other: &Vector3<S>) -> Self::Output {
        let x = self.y * other.z - self.z * other.y;
        let y = self.z * other.x - self.x * other.z;
        let z = self.x * other.y - self.y * other.x;

        Vector3::new(x, y, z)
    }
}

impl<S> CrossProduct<Vector3<S>> for &Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn cross(self, other: Vector3<S>) -> Self::Output {
        let x = self.y * other.z - self.z * other.y;
        let y = self.z * other.x - self.x * other.z;
        let z = self.x * other.y - self.y * other.x;

        Vector3::new(x, y, z)
    }
}

impl<'a, 'b, S> CrossProduct<&'a Vector3<S>> for &'b Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn cross(self, other: &'a Vector3<S>) -> Self::Output {
        let x = self.y * other.z - self.z * other.y;
        let y = self.z * other.x - self.x * other.z;
        let z = self.x * other.y - self.y * other.x;

        Vector3::new(x, y, z)
    }
}

impl<S> Lerp<Vector3<S>> for Vector3<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector3<S>;

    fn lerp(self, other: Vector3<S>, amount: Self::Scalar) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<&Vector3<S>> for Vector3<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector3<S>;

    fn lerp(self, other: &Vector3<S>, amount: Self::Scalar) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<Vector3<S>> for &Vector3<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector3<S>;

    fn lerp(self, other: Vector3<S>, amount: Self::Scalar) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<'a, 'b, S> Lerp<&'a Vector3<S>> for &'b Vector3<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector3<S>;

    fn lerp(self, other: &'a Vector3<S>, amount: Self::Scalar) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Magnitude for Vector3<S> where S: ScalarFloat {
    type Output = S;

    fn magnitude(&self) -> Self::Output {
        Self::Output::sqrt(self.magnitude_squared())
    }

    fn magnitude_squared(&self) -> Self::Output {
        DotProduct::dot(self, self)
    }

    fn normalize(&self) -> Self {
        self / self.magnitude()
    }

    fn normalize_to(&self, magnitude: Self::Output) -> Self {
        self * (magnitude / self.magnitude())
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


impl<S> Finite for Vector3<S> where S: ScalarFloat {
    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl<S> Finite for &Vector3<S> where S: ScalarFloat {
    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl<S> ProjectOn<Vector3<S>> for Vector3<S> where S: ScalarFloat {
    type Output = Vector3<S>;

    fn project_on(self, onto: Vector3<S>) -> Vector3<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<S> ProjectOn<&Vector3<S>> for Vector3<S> where S: ScalarFloat {
    type Output = Vector3<S>;

    fn project_on(self, onto: &Vector3<S>) -> Vector3<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<S> ProjectOn<Vector3<S>> for &Vector3<S> where S: ScalarFloat {
    type Output = Vector3<S>;

    fn project_on(self, onto: Vector3<S>) -> Vector3<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<'a, 'b, S> ProjectOn<&'a Vector3<S>> for &'b Vector3<S> where S: ScalarFloat {
    type Output = Vector3<S>;

    fn project_on(self, onto: &'a Vector3<S>) -> Vector3<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
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
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(C)]
pub struct Vector4<S> {
    pub x: S,
    pub y: S,
    pub z: S,
    pub w: S,
}

impl<S> Vector4<S> {
    /// Construct a new four-dimensional vector.
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
    /// by removing the w-component.
    #[inline]
    pub fn contract(self) -> Vector3<S> {
        Vector3::new(self.x, self.y, self.z)
    }
}

impl<S> Vector4<S> where S: NumCast + Copy {
    /// Cast a vector from one type of scalars to another type of scalars.
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
    /// Returns the unit x axis vector.
    #[inline]
    pub fn unit_x() -> Vector4<S> {
        Vector4 { 
            x: S::one(), 
            y: S::zero(), 
            z: S::zero(), 
            w: S::zero(),
        }
    }

    /// Returns the unit y axis vector.
    #[inline]
    pub fn unit_y() -> Vector4<S> {
        Vector4 { 
            x: S::zero(), 
            y: S::one(), 
            z: S::zero(), 
            w: S::zero(),
        }
    }
    
    /// Returns the unit z axis vector.
    #[inline]
    pub fn unit_z() -> Vector4<S> {
        Vector4 { 
            x: S::zero(), 
            y: S::zero(), 
            z: S::one(), 
            w: S::zero(),
        }
    }

    /// Returns the unit w axis vector.
    #[inline]
    pub fn unit_w() -> Vector4<S> {
        Vector4 { 
            x: S::zero(), 
            y: S::zero(), 
            z: S::zero(), 
            w: S::one(),
        }
    }
}

impl<S> Array for Vector4<S> where S: Copy {
    type Element = S;

    #[inline]
    fn len() -> usize {
        4
    }

    #[inline]
    fn shape() -> (usize, usize) {
        (4, 1)
    }

    #[inline]
    fn as_ptr(&self) -> *const Self::Element {
        &self.x
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Self::Element {
        &mut self.x
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Element] {
        <Self as AsRef<[Self::Element; 4]>>::as_ref(self)
    }
}

impl<S> Sum for Vector4<S> where S: Scalar {
    #[inline]
    fn sum(&self) -> S {
        self.x + self.y + self.z + self.w
    }
}

impl<S> Product for Vector4<S> where S: Scalar {
    #[inline]
    fn product(&self) -> S {
        self.x * self.y * self.z * self.w
    } 
}

impl<S> AsRef<[S; 4]> for Vector4<S> {
    fn as_ref(&self) -> &[S; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsRef<(S, S, S, S)> for Vector4<S> {
    fn as_ref(&self) -> &(S, S, S, S) {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[S; 4]> for Vector4<S> {
    fn as_mut(&mut self) -> &mut [S; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<(S, S, S, S)> for Vector4<S> {
    fn as_mut(&mut self) -> &mut (S, S, S, S) {
        unsafe { mem::transmute(self) }
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
        unsafe { mem::transmute(v) }
    }
}

impl<'a, S> From<&'a (S, S, S, S)> for &'a Vector4<S> where S: Scalar {
    #[inline]
    fn from(v: &'a (S, S, S, S)) -> &'a Vector4<S> {
        unsafe { mem::transmute(v) }
    }
}

impl<S> fmt::Debug for Vector4<S> where S: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector4 ")?;
        <[S; 4] as fmt::Debug>::fmt(self.as_ref(), f)
    }
}

impl<S> fmt::Display for Vector4<S> where S: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector4 [{:.2}, {:.2}, {:.2}, {:.2}]", self.x, self.y, self.z, self.w)
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
    fn add_assign(&mut self, other: Vector4<S>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl<S> ops::AddAssign<&Vector4<S>> for Vector4<S> where S: Scalar {
    fn add_assign(&mut self, other: &Vector4<S>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl<S> ops::SubAssign<Vector4<S>> for Vector4<S> where S: Scalar {
    fn sub_assign(&mut self, other: Vector4<S>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl<S> ops::SubAssign<&Vector4<S>> for Vector4<S> where S: Scalar {
    fn sub_assign(&mut self, other: &Vector4<S>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl<S> ops::Mul<S> for Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

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
    fn mul_assign(&mut self, other: S) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
        self.w *= other;
    }
}

impl<S> ops::Div<S> for Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

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
    fn div_assign(&mut self, other: S) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
        self.w /= other;
    }
}

impl<S> ops::Rem<S> for Vector4<S> where S: Scalar {
    type Output = Vector4<S>;

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

    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        let w = self.w % other;
        
        Vector4::new(x, y, z, w)
    }
}

impl<S> ops::RemAssign<S> for Vector4<S> where S: Scalar {
    fn rem_assign(&mut self, other: S) {
        self.x %= other;
        self.y %= other;
        self.z %= other;
        self.w %= other;
    }
}

impl<S> Zero for Vector4<S> where S: Scalar {
    #[inline]
    fn zero() -> Vector4<S> {
        Vector4 { 
            x: S::zero(), 
            y: S::zero(), 
            z: S::zero(), 
            w: S::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.x == S::zero() && self.y == S::zero() && self.z == S::zero() && self.w == S::zero()
    }
}

impl<S> Metric<Vector4<S>> for Vector4<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: Vector4<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
        let dz_squared = (to.z - self.z) * (to.z - self.z);
        let dw_squared = (to.w - self.w) * (to.w - self.w);
    
        dx_squared + dy_squared + dz_squared + dw_squared
    }
}

impl<S> Metric<&Vector4<S>> for Vector4<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: &Vector4<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
        let dz_squared = (to.z - self.z) * (to.z - self.z);
        let dw_squared = (to.w - self.w) * (to.w - self.w);
    
        dx_squared + dy_squared + dz_squared + dw_squared
    }
}

impl<S> Metric<Vector4<S>> for &Vector4<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: Vector4<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
        let dz_squared = (to.z - self.z) * (to.z - self.z);
        let dw_squared = (to.w - self.w) * (to.w - self.w);
    
        dx_squared + dy_squared + dz_squared + dw_squared
    }
}

impl<'a, 'b, S> Metric<&'a Vector4<S>> for &'b Vector4<S> where S: ScalarFloat {
    type Output = S;

    #[inline]
    fn distance_squared(self, to: &Vector4<S>) -> Self::Output {
        let dx_squared = (to.x - self.x) * (to.x - self.x);
        let dy_squared = (to.y - self.y) * (to.y - self.y);
        let dz_squared = (to.z - self.z) * (to.z - self.z);
        let dw_squared = (to.w - self.w) * (to.w - self.w);

        dx_squared + dy_squared + dz_squared + dw_squared
    }
}

impl<S> DotProduct<Vector4<S>> for Vector4<S> where S: Scalar {
    type Output = S;
    
    fn dot(self, other: Vector4<S>) -> Self::Output {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl<S> DotProduct<&Vector4<S>> for Vector4<S> where S: Scalar {
    type Output = S;

    fn dot(self, other: &Vector4<S>) -> Self::Output {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl<S> DotProduct<Vector4<S>> for &Vector4<S> where S: Scalar {
    type Output = S;

    fn dot(self, other: Vector4<S>) -> Self::Output {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl<'a, 'b, S> DotProduct<&'a Vector4<S>> for &'b Vector4<S> where S: Scalar {
    type Output = S;
    
    fn dot(self, other: &'a Vector4<S>) -> Self::Output {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl<S> Lerp<Vector4<S>> for Vector4<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector4<S>;

    fn lerp(self, other: Vector4<S>, amount: Self::Scalar) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<&Vector4<S>> for Vector4<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector4<S>;

    fn lerp(self, other: &Vector4<S>, amount: S) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<Vector4<S>> for &Vector4<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector4<S>;

    fn lerp(self, other: Vector4<S>, amount: Self::Scalar) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<'a, 'b, S> Lerp<&'a Vector4<S>> for &'b Vector4<S> where S: Scalar {
    type Scalar = S;
    type Output = Vector4<S>;

    fn lerp(self, other: &'a Vector4<S>, amount: Self::Scalar) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Magnitude for Vector4<S> where S: ScalarFloat {
    type Output = S;

    fn magnitude(&self) -> Self::Output {
        Self::Output::sqrt(self.magnitude_squared())
    }

    fn magnitude_squared(&self) -> Self::Output {
        DotProduct::dot(self, self)
    }

    fn normalize(&self) -> Self {
        self / self.magnitude()
    }

    fn normalize_to(&self, magnitude: Self::Output) -> Self {
        self * (magnitude / self.magnitude())
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


impl<S> Finite for Vector4<S> where S: ScalarFloat {
    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
}

impl<S> Finite for &Vector4<S> where S: ScalarFloat {
    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
}

impl<S> ProjectOn<Vector4<S>> for Vector4<S> where S: ScalarFloat {
    type Output = Vector4<S>;

    fn project_on(self, onto: Vector4<S>) -> Vector4<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<S> ProjectOn<&Vector4<S>> for Vector4<S> where S: ScalarFloat {
    type Output = Vector4<S>;

    fn project_on(self, onto: &Vector4<S>) -> Vector4<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<S> ProjectOn<Vector4<S>> for &Vector4<S> where S: ScalarFloat {
    type Output = Vector4<S>;

    fn project_on(self, onto: Vector4<S>) -> Vector4<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<'a, 'b, S> ProjectOn<&'a Vector4<S>> for &'b Vector4<S> where S: ScalarFloat {
    type Output = Vector4<S>;

    fn project_on(self, onto: &'a Vector4<S>) -> Vector4<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
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

