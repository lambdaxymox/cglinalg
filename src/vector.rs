use structure::{
    Array,
    Zero,
    VectorSpace,
    //ProjectOn,
    DotProduct,
    Magnitude,
    Lerp,
    Metric,
};
use std::fmt;
use std::mem;
use std::ops;
use std::cmp;

use base::{
    Scalar,
    ScalarFloat,   
};



/// A representation of one-dimensional vectors.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Vector1<S> {
    pub x: S,
}

impl<S> Vector1<S> {
    /// Create a new vector.
    pub fn new(x: S) -> Vector1<S> {
        Vector1 { x: x }
    }
}

impl<S> Vector1<S> where S: Scalar {
    #[inline]
    pub fn unit_x() -> Vector1<S> {
        Vector1 { x: S::one() }
    }
}

impl<S> Metric<Vector1<S>> for Vector1<S> where S: ScalarFloat {
    type Metric = S;

    #[inline]
    fn distance_squared(self, to: Vector1<S>) -> S {
        let dx_2 = (to.x - self.x) * (to.x - self.x);

        dx_2
    }
}

impl<S> Metric<&Vector1<S>> for Vector1<S> where S: ScalarFloat {
    type Metric = S;

    #[inline]
    fn distance_squared(self, to: &Vector1<S>) -> S {
        let dx_2 = (to.x - self.x) * (to.x - self.x);

        dx_2
    }
}

impl<S> Metric<Vector1<S>> for &Vector1<S> where S: ScalarFloat {
    type Metric = S;

    #[inline]
    fn distance_squared(self, to: Vector1<S>) -> S {
        let dx_2 = (to.x - self.x) * (to.x - self.x);

        dx_2
    }
}

impl<'a, 'b, S> Metric<&'a Vector1<S>> for &'b Vector1<S> where S: ScalarFloat {
    type Metric = S;

    #[inline]
    fn distance_squared(self, to: &'a Vector1<S>) -> S {
        let dx_2 = (to.x - self.x) * (to.x - self.x);

        dx_2
    }
}

impl<S> Array for Vector1<S> where S: Scalar {
    type Element = S;

    #[inline]
    fn len() -> usize {
        1
    }

    #[inline]
    fn from_value(value: Self::Element) -> Self {
        Vector1::new(value)
    }

    #[inline]
    fn as_ptr(&self) -> *const Self::Element {
        &self.x
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Self::Element {
        &mut self.x
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
        write!(f, "Vector2 [{:.2}]", self.x)
    }
}

impl<S> From<S> for Vector1<S> where S: Scalar {
    #[inline]
    fn from(v: S) -> Vector1<S> {
        Vector1 { x: v }
    }
}

impl<S> From<[S; 1]> for Vector1<S> where S: Scalar {
    #[inline]
    fn from(v: [S; 1]) -> Vector1<S> {
        Vector1 { x: v[0] }
    }
}

impl<S> From<&[S; 1]> for Vector1<S> where S: Scalar {
    #[inline]
    fn from(v: &[S; 1]) -> Vector1<S> {
        Vector1 { x: v[0] }
    }
}

impl<'a, S> From<&'a [S; 1]> for &'a Vector1<S> where S: Scalar {
    #[inline]
    fn from(v: &'a [S; 1]) -> &'a Vector1<S> {
        unsafe { mem::transmute(v) }
    }
}

impl<S> ops::Neg for Vector1<S> where S: ScalarFloat {
    type Output = Vector1<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector1 { x: -self.x }
    }
}

impl<S> ops::Neg for &Vector1<S> where S: ScalarFloat {
    type Output = Vector1<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector1 { x: -self.x }
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

    fn mul(self, other: S) -> Vector1<S> {
        Vector1 {
            x: self.x * other,
        }
    }
}

impl<S> ops::Mul<S> for &Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn mul(self, other: S) -> Vector1<S> {
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

    fn div(self, other: S) -> Vector1<S> {
        Vector1 {
            x: self.x / other,
        }
    }
}

impl<S> ops::Div<S> for &Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn div(self, other: S) -> Vector1<S> {
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
        
        Vector1 { x: x }
    }
}

impl<S> ops::Rem<S> for &Vector1<S> where S: Scalar {
    type Output = Vector1<S>;

    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        
        Vector1 { x: x }
    }
}

impl<S> ops::RemAssign<S> for Vector1<S> where S: Scalar {
    fn rem_assign(&mut self, other: S) {
        self.x %= other;
    }
}

impl<S> Zero for Vector1<S> where S: Scalar {
    fn zero() -> Vector1<S> {
        Vector1 { x: S::zero() }
    }

    fn is_zero(&self) -> bool {
        self.x == S::zero()
    }
}

impl<S> VectorSpace for Vector1<S> where S: Scalar {
    type Scalar = S;
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


/*
/// Generate implementation of an operator for a pair of types.
macro_rules! impl_operator {
    // Implement a binary operator where the left-hand side is non-scalar and the 
    // right-hand side is non-scalar.
    (<$S:ident: $Constraint:ident> $Op:ident<$Rhs:ty> for $Lhs:ty {
        fn $op:ident($lhs:ident, $rhs:ident) -> $Output:ty { $body:expr }
    }) => {
        impl<$S> $Op<$Rhs> for $Lhs where $S: $Constraint {
            type Output = $Output;

            #[inline]
            fn $op(self, other: $Rhs) -> $Output {
                let $lhs = self;
                let $rhs = other; 
                $body
            }
        }

        impl<'a, $S> $Op<$Rhs> for &'a $Lhs where $S: $Constraint {
            type Output = $Output;

            #[inline]
            fn $op(self, other: $Rhs) -> $Output {
                let $lhs = self;
                let $rhs = other; 
                $body
            }
        }

        impl<'a, $S> $Op<&'a $Rhs> for $Lhs where $S: $Constraint {
            type Output = $Output;

            #[inline]
            fn $op(self, other: &'a $Rhs) -> $Output {
                let $lhs = self;
                let $rhs = other; 
                $body
            }
        }

        impl<'a, 'b, $S> $Op<&'a $Rhs> for &'b $Lhs where $S: $Constraint {
            type Output = $Output;

            #[inline]
            fn $op(self, other: &'a $Rhs) -> $Output {
                let $lhs = self;
                let $rhs = other; 
                $body
            }
        }
    };
    // Implement a binary operator where the left-hand side is non-scalar and the
    // right-hand side is scalar.
    (<$S:ident: $Constraint:ident> $Op:ident<$Rhs:ident> for $Lhs:ty {
        fn $op:ident($lhs:ident, $rhs:ident) -> $Output:ty { $body:expr }
    }) => {
        impl<$S> $Op for $Lhs where $S: $Constraint {
            type Output = $Output;
    
            #[inline]
            fn $op(self) -> $Output {
                let $lhs = self;
                let $rhs = other; 
                $body
            }
        }
    
        impl<'a, $S> $Op for &'a $Lhs where $S: $Constraint {
            type Output = $Output;
    
            #[inline]
            fn $op(self) -> $Output {
                let $lhs = self;
                let $rhs = other; 
                $body
            }
        }
    }
}
*/

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
    
    /// Compute the squared length of a vector.
    fn magnitude_squared(&self) -> Self::Output {
        DotProduct::dot(self, self)
    }

    /// Compute the norm (length) of a vector.
    #[inline]
    fn magnitude(&self) -> Self::Output {
        S::sqrt(DotProduct::dot(self, self))
    }
    
    /// Convert a vector into a unit vector.
    fn normalize(&self) -> Self {
        self / self.magnitude()
    }
    
    /// Normalize a vector with a specified magnitude.
    fn normalize_to(&self, magnitude: Self::Output) -> Self {
        self * (magnitude / self.magnitude())
    }
}


impl_mul_operator!(u8, Vector1<u8>, Vector1<u8>, { x });
impl_mul_operator!(u16, Vector1<u16>, Vector1<u16>, { x });
impl_mul_operator!(u32, Vector1<u32>, Vector1<u32>, { x });
impl_mul_operator!(u64, Vector1<u64>, Vector1<u64>, { x });
impl_mul_operator!(u128, Vector1<u128>, Vector1<u128>, { x });
impl_mul_operator!(usize, Vector1<usize>, Vector1<usize>, { x });

impl_mul_operator!(i8, Vector1<i8>, Vector1<i8>, { x });
impl_mul_operator!(i16, Vector1<i16>, Vector1<i16>, { x });
impl_mul_operator!(i32, Vector1<i32>, Vector1<i32>, { x });
impl_mul_operator!(i64, Vector1<i64>, Vector1<i64>, { x });
impl_mul_operator!(i128, Vector1<i128>, Vector1<i128>, { x });
impl_mul_operator!(isize, Vector1<isize>, Vector1<isize>, { x });

impl_mul_operator!(f32, Vector1<f32>, Vector1<f32>, { x });
impl_mul_operator!(f64, Vector1<f64>, Vector1<f64>, { x });





/// A representation of two-dimensional vectors with a Euclidean metric.
#[derive(Copy, Clone, PartialEq)]
pub struct Vector2<S> {
   pub x: S,
   pub y: S,
}

impl<S> Vector2<S> where S: Scalar {
    /// Create a new vector.
    pub fn new(x: S, y: S) -> Vector2<S> {
        Vector2 { x: x, y: y }
    }

    #[inline]
    pub fn unit_x() -> Vector2<S> {
        Vector2 { x: S::one(), y: S::zero() }
    }

    #[inline]
    pub fn unit_y() -> Vector2<S> {
        Vector2 { x: S::zero(), y: S::one() }
    }
}

impl<S> Array for Vector2<S> where S: Scalar {
    type Element = S;

    #[inline]
    fn len() -> usize {
        2
    }

    #[inline]
    fn from_value(value: Self::Element) -> Self {
        Vector2::new(value, value)
    }

    #[inline]
    fn as_ptr(&self) -> *const Self::Element {
        &self.x
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Self::Element {
        &mut self.x
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
        Vector2 { x: x, y: y }
    }
}

impl<S> From<[S; 2]> for Vector2<S> where S: Scalar {
    #[inline]
    fn from(v: [S; 2]) -> Vector2<S> {
        Vector2 { x: v[0], y: v[1] }
    }
}

impl<S> From<&[S; 2]> for Vector2<S> where S: Scalar {
    #[inline]
    fn from(v: &[S; 2]) -> Vector2<S> {
        Vector2 { x: v[0], y: v[1] }
    }
}

impl<'a, S> From<&'a [S; 2]> for &'a Vector2<S> where S: Scalar {
    #[inline]
    fn from(v: &'a [S; 2]) -> &'a Vector2<S> {
        unsafe { mem::transmute(v) }
    }
}

impl<S> ops::Neg for Vector2<S> where S: ScalarFloat {
    type Output = Vector2<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector2 { x: -self.x, y: -self.y }
    }
}

impl<S> ops::Neg for &Vector2<S> where S: ScalarFloat {
    type Output = Vector2<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector2 { x: -self.x, y: -self.y }
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

    fn mul(self, other: S) -> Vector2<S> {
        Vector2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl<S> ops::Mul<S> for &Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn mul(self, other: S) -> Vector2<S> {
        Vector2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl<S> ops::MulAssign<S> for Vector2<S> where S: Scalar {
    fn mul_assign(&mut self, other: S) {
        self.x *= other;
        self.y *= other;
    }
}

impl<S> ops::Div<S> for Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn div(self, other: S) -> Vector2<S> {
        Vector2 {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

impl<S> ops::Div<S> for &Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn div(self, other: S) -> Vector2<S> {
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
        
        Vector2 { x: x, y: y }
    }
}

impl<S> ops::Rem<S> for &Vector2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        
        Vector2 { x: x, y: y }
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
        Vector2 { x: S::zero(), y: S::zero() }
    }

    fn is_zero(&self) -> bool {
        self.x == S::zero() && self.y == S::zero()
    }
}

impl<S> Metric<Vector2<S>> for Vector2<S> where S: ScalarFloat {
    type Metric = S;

    #[inline]
    fn distance_squared(self, to: Vector2<S>) -> Self::Metric {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
    
        dx_2 + dy_2
    }
}

impl<S> Metric<&Vector2<S>> for Vector2<S> where S: ScalarFloat {
    type Metric = S;

    #[inline]
    fn distance_squared(self, to: &Vector2<S>) -> Self::Metric {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
    
        dx_2 + dy_2
    }
}

impl<S> Metric<Vector2<S>> for &Vector2<S> where S: ScalarFloat {
    type Metric = S;

    #[inline]
    fn distance_squared(self, to: Vector2<S>) -> Self::Metric {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
    
        dx_2 + dy_2
    }
}

impl<'a, 'b, S> Metric<&'a Vector2<S>> for &'b Vector2<S> where S: ScalarFloat {
    type Metric = S;

    #[inline]
    fn distance_squared(self, to: &'a Vector2<S>) -> Self::Metric {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
    
        dx_2 + dy_2
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

    /// Compute the norm (length) of a vector.
    fn magnitude(&self) -> Self::Output {
        Self::Output::sqrt(self.magnitude_squared())
    }

    /// Compute the squared length of a vector.
    fn magnitude_squared(&self) -> Self::Output {
        DotProduct::dot(self, self)
    }

    /// Convert a vector into a unit vector.
    fn normalize(&self) -> Self {
        self / self.magnitude()
    }

    /// Normalize a vector with a specified magnitude.
    fn normalize_to(&self, magnitude: Self::Output) -> Self {
        self * (magnitude / self.magnitude())
    }
}

impl_mul_operator!(u8, Vector2<u8>, Vector2<u8>, { x, y });
impl_mul_operator!(u16, Vector2<u16>, Vector2<u16>, { x, y });
impl_mul_operator!(u32, Vector2<u32>, Vector2<u32>, { x, y });
impl_mul_operator!(u64, Vector2<u64>, Vector2<u64>, { x, y });
impl_mul_operator!(u128, Vector2<u128>, Vector2<u128>, { x, y });
impl_mul_operator!(usize, Vector2<usize>, Vector2<usize>, { x, y });

impl_mul_operator!(i8, Vector2<i8>, Vector2<i8>, { x, y });
impl_mul_operator!(i16, Vector2<i16>, Vector2<i16>, { x, y });
impl_mul_operator!(i32, Vector2<i32>, Vector2<i32>, { x, y });
impl_mul_operator!(i64, Vector2<i64>, Vector2<i64>, { x, y });
impl_mul_operator!(i128, Vector2<i128>, Vector2<i128>, { x, y });
impl_mul_operator!(isize, Vector2<isize>, Vector2<isize>, { x, y });

impl_mul_operator!(f32, Vector2<f32>, Vector2<f32>, { x, y });
impl_mul_operator!(f64, Vector2<f64>, Vector2<f64>, { x, y });




/// A representation of three-dimensional vectors with a Euclidean metric.
#[derive(Copy, Clone, PartialEq)]
pub struct Vector3<S> {
    pub x: S,
    pub y: S,
    pub z: S,
}

impl<S> Vector3<S> where S: Scalar {
    /// Create a new vector.
    pub fn new(x: S, y: S, z: S) -> Vector3<S> {
        Vector3 { x: x, y: y, z: z }
    }

    #[inline]
    pub fn unit_x() -> Vector3<S> {
        Vector3 { x: S::one(), y: S::zero(), z: S::zero() }
    }

    #[inline]
    pub fn unit_y() -> Vector3<S> {
        Vector3 { x: S::zero(), y: S::one(), z: S::zero() }
    }
    
    #[inline]
    pub fn unit_z() -> Vector3<S> {
        Vector3 { x: S::zero(), y: S::zero(), z: S::one() }
    }

    /// Compute the cross product of two three-dimensional vectors. Note that
    /// with the vectors used in computer graphics (two, three, and four dimensions),
    /// the cross product is defined only in three dimensions. Also note that the 
    /// cross product is the hodge dual of the corresponding 2-vector representing 
    /// the surface element that the crossed vector is normal to. That is, 
    /// given vectors `u` and `v`, `u x v == *(u /\ v)`, where `*(.)` denotes the hodge dual.
    pub fn cross(&self, other: &Vector3<S>) -> Vector3<S> {
        let x = self.y * other.z - self.z * other.y;
        let y = self.z * other.x - self.x * other.z;
        let z = self.x * other.y - self.y * other.x;
    
        Vector3::new(x, y, z)
    }
}

impl<S> Array for Vector3<S> where S: Scalar {
    type Element = S;

    #[inline]
    fn len() -> usize {
        3
    }

    #[inline]
    fn from_value(value: Self::Element) -> Self {
        Vector3::new(value, value, value)
    }

    #[inline]
    fn as_ptr(&self) -> *const Self::Element {
        &self.x
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Self::Element {
        &mut self.x
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

impl<S> ops::Neg for Vector3<S> where S: ScalarFloat {
    type Output = Vector3<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector3 { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl<S> ops::Neg for &Vector3<S> where S: ScalarFloat {
    type Output = Vector3<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector3 { x: -self.x, y: -self.y, z: -self.z }
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

    fn mul(self, other: S) -> Vector3<S> {
        Vector3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl<S> ops::Mul<S> for &Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn mul(self, other: S) -> Vector3<S> {
        Vector3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl<S> ops::MulAssign<S> for Vector3<S> where S: Scalar {
    fn mul_assign(&mut self, other: S) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
    }
}

impl<S> ops::Div<S> for Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn div(self, other: S) -> Vector3<S> {
        Vector3 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

impl<S> ops::Div<S> for &Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn div(self, other: S) -> Vector3<S> {
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
        
        Vector3 { x: x, y: y, z: z }
    }
}

impl<S> ops::Rem<S> for &Vector3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn rem(self, other: S) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        
        Vector3 { x: x, y: y, z: z }
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
        Vector3 { x: S::zero(), y: S::zero(), z: S::zero() }
    }

    fn is_zero(&self) -> bool {
        self.x == S::zero() && self.y == S::zero() && self.z == S::zero()
    }
}

impl<S> Metric<Vector3<S>> for Vector3<S> where S: ScalarFloat {
    type Metric = S;

    #[inline]
    fn distance_squared(self, to: Vector3<S>) -> Self::Metric {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
        let dz_2 = (to.z - self.z) * (to.z - self.z);
    
        dx_2 + dy_2 + dz_2
    }
}

impl<S> Metric<&Vector3<S>> for Vector3<S> where S: ScalarFloat {
    type Metric = S;

    #[inline]
    fn distance_squared(self, to: &Vector3<S>) -> Self::Metric {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
        let dz_2 = (to.z - self.z) * (to.z - self.z);
    
        dx_2 + dy_2 + dz_2
    }
}

impl<S> Metric<Vector3<S>> for &Vector3<S> where S: ScalarFloat {
    type Metric = S;

    #[inline]
    fn distance_squared(self, to: Vector3<S>) -> Self::Metric {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
        let dz_2 = (to.z - self.z) * (to.z - self.z);
    
        dx_2 + dy_2 + dz_2
    }
}

impl<'a, 'b, S> Metric<&'a Vector3<S>> for &'b Vector3<S> where S: ScalarFloat {
    type Metric = S;

    #[inline]
    fn distance_squared(self, to: &Vector3<S>) -> Self::Metric {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
        let dz_2 = (to.z - self.z) * (to.z - self.z);
    
        dx_2 + dy_2 + dz_2
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

    /// Compute the norm (length) of a vector.
    fn magnitude(&self) -> Self::Output {
        Self::Output::sqrt(self.magnitude_squared())
    }

    /// Compute the squared length of a vector.
    fn magnitude_squared(&self) -> Self::Output {
        DotProduct::dot(self, self)
    }

    /// Convert a vector into a unit vector.
    fn normalize(&self) -> Self {
        self / self.magnitude()
    }

    /// Normalize a vector with a specified magnitude.
    fn normalize_to(&self, magnitude: Self::Output) -> Self {
        self * (magnitude / self.magnitude())
    }
}

// impl Magnitude<Vector3> for &Vector3 {}
impl_mul_operator!(u8, Vector3<u8>, Vector3<u8>, { x, y, z });
impl_mul_operator!(u16, Vector3<u16>, Vector3<u16>, { x, y, z });
impl_mul_operator!(u32, Vector3<u32>, Vector3<u32>, { x, y, z });
impl_mul_operator!(u64, Vector3<u64>, Vector3<u64>, { x, y, z });
impl_mul_operator!(u128, Vector3<u128>, Vector3<u128>, { x, y, z });
impl_mul_operator!(usize, Vector3<usize>, Vector3<usize>, { x, y, z });

impl_mul_operator!(i8, Vector3<i8>, Vector3<i8>, { x, y, z });
impl_mul_operator!(i16, Vector3<i16>, Vector3<i16>, { x, y, z });
impl_mul_operator!(i32, Vector3<i32>, Vector3<i32>, { x, y, z });
impl_mul_operator!(i64, Vector3<i64>, Vector3<i64>, { x, y, z });
impl_mul_operator!(i128, Vector3<i128>, Vector3<i128>, { x, y, z });
impl_mul_operator!(isize, Vector3<isize>, Vector3<isize>, { x, y, z });

impl_mul_operator!(f32, Vector3<f32>, Vector3<f32>, { x, y, z });
impl_mul_operator!(f64, Vector3<f64>, Vector3<f64>, { x, y, z });


/// A representation of four-dimensional vectors with a Euclidean metric.
#[derive(Copy, Clone)]
pub struct Vector4<S> {
    pub x: S,
    pub y: S,
    pub z: S,
    pub w: S,
}

impl<S> Vector4<S> where S: Scalar {
    pub fn new(x: S, y: S, z: S, w: S) -> Vector4<S> {
        Vector4 { x: x, y: y, z: z, w: w }
    }

    #[inline]
    pub fn unit_x() -> Vector4<S> {
        Vector4 { x: S::one(), y: S::zero(), z: S::zero(), w: S::zero() }
    }

    #[inline]
    pub fn unit_y() -> Vector4<S> {
        Vector4 { x: S::zero(), y: S::one(), z: S::zero(), w: S::zero() }
    }
    
    #[inline]
    pub fn unit_z() -> Vector4<S> {
        Vector4 { x: S::zero(), y: S::zero(), z: S::one(), w: S::zero() }
    }

    #[inline]
    pub fn unit_w() -> Vector4<S> {
        Vector4 { x: S::zero(), y: S::zero(), z: S::zero(), w: S::one() }
    }
}

impl<S> Array for Vector4<S> where S: Scalar {
    type Element = S;

    #[inline]
    fn len() -> usize {
        4
    }

    #[inline]
    fn from_value(value: Self::Element) -> Self {
        Vector4::new(value, value, value, value)
    }

    #[inline]
    fn as_ptr(&self) -> *const Self::Element {
        &self.x
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Self::Element {
        &mut self.x
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
/*
impl cmp::PartialEq for Vector4 {
    fn eq(&self, other: &Vector4) -> bool {
        (f32::abs(self.x - other.x) < EPSILON) &&
        (f32::abs(self.y - other.y) < EPSILON) &&
        (f32::abs(self.z - other.z) < EPSILON) &&
        (f32::abs(self.w - other.w) < EPSILON)
    }
}

impl ops::Neg for Vector4 {
    type Output = Vector4;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector4 { x: -self.x, y: -self.y, z: -self.z, w: -self.w }
    }
}

impl ops::Neg for &Vector4 {
    type Output = Vector4;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector4 { x: -self.x, y: -self.y, z: -self.z, w: -self.w }
    }
}

impl ops::Add<Vector4> for &Vector4 {
    type Output = Vector4;

    fn add(self, other: Vector4) -> Self::Output {
        Vector4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl ops::Add<Vector4> for Vector4 {
    type Output = Vector4;

    fn add(self, other: Vector4) -> Self::Output {
        Vector4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl ops::Add<&Vector4> for Vector4 {
    type Output = Vector4;

    fn add(self, other: &Vector4) -> Self::Output {
        Vector4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,   
            w: self.w + other.w,            
        }
    }
}

impl<'a, 'b> ops::Add<&'a Vector4> for &'b Vector4 {
    type Output = Vector4;

    fn add(self, other: &'a Vector4) -> Self::Output {
        Vector4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl ops::Sub<Vector4> for &Vector4 {
    type Output = Vector4;

    fn sub(self, other: Vector4) -> Self::Output {
        Vector4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl ops::Sub<Vector4> for Vector4 {
    type Output = Vector4;

    fn sub(self, other: Vector4) -> Self::Output {
        Vector4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl ops::Sub<&Vector4> for Vector4 {
    type Output = Vector4;

    fn sub(self, other: &Vector4) -> Self::Output {
        Vector4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl<'a, 'b> ops::Sub<&'b Vector4> for &'a Vector4 {
    type Output = Vector4;

    fn sub(self, other: &'b Vector4) -> Self::Output {
        Vector4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl ops::AddAssign<Vector4> for Vector4 {
    fn add_assign(&mut self, other: Vector4) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl ops::AddAssign<&Vector4> for Vector4 {
    fn add_assign(&mut self, other: &Vector4) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl ops::SubAssign<Vector4> for Vector4 {
    fn sub_assign(&mut self, other: Vector4) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl ops::SubAssign<&Vector4> for Vector4 {
    fn sub_assign(&mut self, other: &Vector4) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl ops::Mul<f32> for Vector4 {
    type Output = Vector4;

    fn mul(self, other: f32) -> Vector4 {
        Vector4 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
            w: self.w * other,
        }
    }
}

impl ops::Mul<f32> for &Vector4 {
    type Output = Vector4;

    fn mul(self, other: f32) -> Vector4 {
        Vector4 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
            w: self.w * other,
        }
    }
}

impl ops::MulAssign<f32> for Vector4 {
    fn mul_assign(&mut self, other: f32) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
        self.w *= other;
    }
}

impl ops::Div<f32> for Vector4 {
    type Output = Vector4;

    fn div(self, other: f32) -> Vector4 {
        Vector4 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
            w: self.w / other,
        }
    }
}

impl ops::Div<f32> for &Vector4 {
    type Output = Vector4;

    fn div(self, other: f32) -> Vector4 {
        Vector4 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
            w: self.w / other,
        }
    }
}

impl ops::DivAssign<f32> for Vector4 {
    fn div_assign(&mut self, other: f32) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
        self.w /= other;
    }
}

impl ops::Rem<f32> for Vector4 {
    type Output = Vector4;

    fn rem(self, other: f32) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        let w = self.w % other;

        Vector4 { x: x, y: y, z: z, w: w }
    }
}

impl ops::Rem<f32> for &Vector4 {
    type Output = Vector4;

    fn rem(self, other: f32) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        let w = self.w % other;
        
        Vector4 { x: x, y: y, z: z, w: w }
    }
}

impl ops::RemAssign<f32> for Vector4 {
    fn rem_assign(&mut self, other: f32) {
        self.x %= other;
        self.y %= other;
        self.z %= other;
        self.w %= other;
    }
}

impl Zero for Vector4 {
    #[inline]
    fn zero() -> Vector4 {
        Vector4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }
    }

    fn is_zero(&self) -> bool {
        self.x == 0.0 && self.y == 0.0 && self.z == 0.0 && self.w == 0.0
    }
}

impl Metric<Vector4> for Vector4 {
    #[inline]
    fn distance2(self, to: Vector4) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
        let dz_2 = (to.z - self.z) * (to.z - self.z);
        let dw_2 = (to.w - self.w) * (to.w - self.w);
    
        dx_2 + dy_2 + dz_2 + dw_2
    }
}

impl Metric<&Vector4> for Vector4 {
    #[inline]
    fn distance2(self, to: &Vector4) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
        let dz_2 = (to.z - self.z) * (to.z - self.z);
        let dw_2 = (to.w - self.w) * (to.w - self.w);
    
        dx_2 + dy_2 + dz_2 + dw_2
    }
}

impl Metric<Vector4> for &Vector4 {
    #[inline]
    fn distance2(self, to: Vector4) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
        let dz_2 = (to.z - self.z) * (to.z - self.z);
        let dw_2 = (to.w - self.w) * (to.w - self.w);
    
        dx_2 + dy_2 + dz_2 + dw_2
    }
}

impl<'a, 'b> Metric<&'a Vector4> for &'b Vector4 {
    #[inline]
    fn distance2(self, to: &Vector4) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
        let dz_2 = (to.z - self.z) * (to.z - self.z);
        let dw_2 = (to.w - self.w) * (to.w - self.w);

        dx_2 + dy_2 + dz_2 + dw_2
    }
}

impl DotProduct<Vector4> for Vector4 {
    fn dot(self, other: Vector4) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl DotProduct<&Vector4> for Vector4 {
    fn dot(self, other: &Vector4) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl DotProduct<Vector4> for &Vector4 {
    fn dot(self, other: Vector4) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl<'a, 'b> DotProduct<&'a Vector4> for &'b Vector4 {
    fn dot(self, other: &'a Vector4) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl Lerp<Vector4> for Vector4 {
    type Output = Vector4;

    fn lerp(self, other: Vector4, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl Lerp<&Vector4> for Vector4 {
    type Output = Vector4;

    fn lerp(self, other: &Vector4, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl Lerp<Vector4> for &Vector4 {
    type Output = Vector4;

    fn lerp(self, other: Vector4, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<'a, 'b> Lerp<&'a Vector4> for &'b Vector4 {
    type Output = Vector4;

    fn lerp(self, other: &'a Vector4, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl Magnitude<Vector4> for Vector4 {}
impl Magnitude<Vector4> for &Vector4 {}
*/

#[cfg(test)]
mod vector1_tests {
    use std::slice::Iter;
    use super::Vector1;
    use structure::Zero;


    struct TestCase {
        c: f32,
        v1: Vector1<f32>,
        v2: Vector1<f32>,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase { 
                    c: 802.3435169, v1: Vector1::from(-23.43), v2: Vector1::from(426.1),
                },
                TestCase {
                    c: 33.249539, v1: Vector1::from(27.6189), v2: Vector1::from(258.083),
                },
                TestCase {
                    c: 7.04217, v1: Vector1::from(0.0), v2: Vector1::from(0.0),
                },
            ]
        }
    }

    #[test]
    fn addition() {
        for test in test_cases().iter() {
            let expected = Vector1::from(test.v1.x + test.v2.x);
            let result = test.v1 + test.v2;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn subtraction() {
        for test in test_cases().iter() {
            let expected = Vector1::from(test.v1.x + test.v2.x);
            let result = test.v1 + test.v2;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn scalar_multiplication() {
        for test in test_cases().iter() {
            let expected = Vector1::from(test.c * test.v1.x);
            let result = test.v1 * test.c;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn scalar_division() {
        for test in test_cases().iter() {
            let expected = Vector1::from(test.v1.x / test.c);
            let result = test.v1 / test.c;
            assert_eq!(result, expected);
        }
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_array_access() {
        let v = Vector1::new(1_f32);
        assert_eq!(v[1], v[1]);
    }

    #[test]
    fn vector_times_zero_equals_zero() {
        let v = Vector1::new(1_f32);
        assert_eq!(v * 0_f32, Vector1::zero());
    }

    #[test]
    fn zero_times_vector_equals_zero() {
        let v = Vector1::new(1_f32);
        assert_eq!(0_f32 * v, Vector1::zero());
    }
}


#[cfg(test)]
mod vector2_tests {
    use std::slice::Iter;
    use structure::Zero;
    use super::Vector2;

    struct TestCase {
        c: f32,
        v1: Vector2<f32>,
        v2: Vector2<f32>,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase {
                    c: 802.3435169,
                    v1: Vector2::from((80.0,  43.569)),
                    v2: Vector2::from((6.741, 23.5724)),
                },
                TestCase {
                    c: 33.249539,
                    v1: Vector2::from((27.6189, 4.2219)),
                    v2: Vector2::from((258.083, 42.17))
                },
                TestCase {
                    c: 7.04217,
                    v1: Vector2::from((70.0,  49.0)),
                    v2: Vector2::from((89.9138, 427.46894)),
                },
                TestCase {
                    c: 61.891390,
                    v1: Vector2::from((8827.1983, 56.31)),
                    v2: Vector2::from((89.0, 936.5)),
                }
            ]
        }
    }

    #[test]
    fn test_addition() {
        for test in test_cases().iter() {
            let expected = Vector2::from((test.v1.x + test.v2.x, test.v1.y + test.v2.y));
            let result = test.v1 + test.v2;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_subtraction() {
        for test in test_cases().iter() {
            let expected = Vector2::from((test.v1.x - test.v2.x, test.v1.y - test.v2.y));
            let result = test.v1 - test.v2;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_scalar_multiplication() {
        for test in test_cases().iter() {
            let expected = Vector2::from((test.c * test.v1.x, test.c * test.v1.y));
            let result = test.v1 * test.c;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_scalar_division() {
        for test in test_cases().iter() {
            let expected = Vector2::from((test.v1.x / test.c, test.v1.y / test.c));
            let result = test.v1 / test.c;
            assert_eq!(result, expected);
        }
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_array_access() {
        let v = Vector2::new(1_f32, 2_f32);
        assert_eq!(v[2], v[2]);
    }

    #[test]
    fn vector_times_zero_equals_zero() {
        let v = Vector2::new(1_f32, 2_f32);
        assert_eq!(v * 0_f32, Vector2::zero());
    }

    #[test]
    fn zero_times_vector_equals_zero() {
        let v = Vector2::new(1_f32, 2_f32);
        assert_eq!(0_f32 * v, Vector2::zero());
    }
}

#[cfg(test)]
mod vector3_tests {
    use std::slice::Iter;
    use super::Vector3;
    use structure::Zero;


    struct TestCase {
        c: f32,
        x: Vector3<f32>,
        y: Vector3<f32>,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase {
                    c: 802.3435169,
                    x: Vector3::from((80.0,  23.43, 43.569)),
                    y: Vector3::from((6.741, 426.1, 23.5724)),
                },
                TestCase {
                    c: 33.249539,
                    x: Vector3::from((27.6189, 13.90, 4.2219)),
                    y: Vector3::from((258.083, 31.70, 42.17))
                },
                TestCase {
                    c: 7.04217,
                    x: Vector3::from((70.0,  49.0,  95.0)),
                    y: Vector3::from((89.9138, 36.84, 427.46894)),
                },
                TestCase {
                    c: 61.891390,
                    x: Vector3::from((8827.1983, 89.5049494, 56.31)),
                    y: Vector3::from((89.0, 72.0, 936.5)),
                }
            ]
        }
    }

    #[test]
    fn test_addition() {
        for test in test_cases().iter() {
            let expected = Vector3::from((test.x.x + test.y.x, test.x.y + test.y.y, test.x.z + test.y.z));
            let result = test.x + test.y;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_subtraction() {
        for test in test_cases().iter() {
            let expected = Vector3::from((test.x.x - test.y.x, test.x.y - test.y.y, test.x.z - test.y.z));
            let result = test.x - test.y;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_scalar_multiplication() {
        for test in test_cases().iter() {
            let expected = Vector3::from((test.c * test.x.x, test.c * test.x.y, test.c * test.x.z));
            let result = test.x * test.c;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_scalar_division() {
        for test in test_cases().iter() {
            let expected = Vector3::from((test.x.x / test.c, test.x.y / test.c, test.x.z / test.c));
            let result = test.x / test.c;
            assert_eq!(result, expected);
        }
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_array_access() {
        let v = Vector3::new(1_f32, 2_f32, 3_f32);
        assert_eq!(v[3], v[3]);
    }

    #[test]
    fn vector_times_zero_equals_zero() {
        let v = Vector3::new(1_f32, 2_f32, 3_f32);
        assert_eq!(v * 0_f32, Vector3::zero());
    }

    #[test]
    fn zero_times_vector_equals_zero() {
        let v = Vector3::new(1_f32, 2_f32, 3_f32);
        assert_eq!(0_f32 * v, Vector3::zero());
    }
}
/*

#[cfg(test)]
mod vector4_tests {
    use std::slice::Iter;
    use super::Vector4;

    struct TestCase {
        c: f32,
        v1: Vector4,
        v2: Vector4,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase {
                    c: 802.3435169,
                    v1: Vector4::from((80.0,  23.43, 43.569, 69.9093)),
                    v2: Vector4::from((6.741, 426.1, 23.5724, 85567.75976)),
                },
                TestCase {
                    c: 33.249539,
                    v1: Vector4::from((27.6189, 13.90, 4.2219, 91.11955)),
                    v2: Vector4::from((258.083, 31.70, 42.17, 8438.2376))
                },
                TestCase {
                    c: 7.04217,
                    v1: Vector4::from((70.0, 49.0, 95.0, 508.5602759)),
                    v2: Vector4::from((89.9138, 36.84, 427.46894, 0.5796180917)),
                },
                TestCase {
                    c: 61.891390,
                    v1: Vector4::from((8827.1983, 89.5049494, 56.31, 0.2888633714)),
                    v2: Vector4::from((89.0, 72.0, 936.5, 0.2888633714)),
                }
            ]
        }
    }

    #[test]
    fn test_addition() {
        for test in test_cases().iter() {
            let expected = Vector4::from((
                test.v1.x + test.v2.x, test.v1.y + test.v2.y,
                test.v1.z + test.v2.z, test.v1.w + test.v2.w
            ));
            let result = test.v1 + test.v2;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_subtraction() {
        for test in test_cases().iter() {
            let expected = Vector4::from((
                test.v1.x - test.v2.x, test.v1.y - test.v2.y,
                test.v1.z - test.v2.z, test.v1.w - test.v2.w
            ));
            let result = test.v1 - test.v2;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_scalar_multiplication() {
        for test in test_cases().iter() {
            let expected = Vector4::from((
                test.c * test.v1.x, test.c * test.v1.y, test.c * test.v1.z, test.c * test.v1.w
            ));
            let result = test.v1 * test.c;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_scalar_division() {
        for test in test_cases().iter() {
            let expected = Vector4::from((
                test.v1.x / test.c, test.v1.y / test.c, test.v1.z / test.c, test.v1.w / test.c
            ));
            let result = test.v1 / test.c;
            assert_eq!(result, expected);
        }
    }
}

*/