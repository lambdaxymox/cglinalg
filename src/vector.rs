use traits::{Array, Zero, Metric, ProjectOn, DotProduct, Lerp, Magnitude};
use std::fmt;
use std::mem;
use std::ops;
use std::cmp;


const EPSILON: f32 = 0.00001; 



/// A representation of one-dimensional vectors.
#[derive(Copy, Clone, PartialEq)]
pub struct Vector1 {
    pub x: f32,
}

impl Vector1 {
    ///
    /// Create a new vector.
    ///
    pub fn new(x: f32) -> Vector1 {
        Vector1 { x: x }
    }

    #[inline]
    pub fn unit_x() -> Vector1 {
        Vector1 { x: 1.0 }
    }
}

impl Metric<Vector1> for Vector1 {
    #[inline]
    fn distance2(self, to: Vector1) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);

        dx_2
    }
}

impl Metric<&Vector1> for Vector1 {
    #[inline]
    fn distance2(self, to: &Vector1) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);

        dx_2
    }
}

impl Metric<Vector1> for &Vector1 {
    #[inline]
    fn distance2(self, to: Vector1) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);

        dx_2
    }
}

impl<'a, 'b> Metric<&'a Vector1> for &'b Vector1 {
    #[inline]
    fn distance2(self, to: &'a Vector1) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);

        dx_2
    }
}

impl Array for Vector1 {
    type Element = f32;

    #[inline]
    fn len() -> usize {
        1
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

impl AsRef<[f32; 1]> for Vector1 {
    fn as_ref(&self) -> &[f32; 1] {
        unsafe { mem::transmute(self) }
    }
}

impl AsRef<f32> for Vector1 {
    fn as_ref(&self) -> &f32 {
        unsafe { mem::transmute(self) }
    }
}

impl AsRef<(f32,)> for Vector1 {
    fn as_ref(&self) -> &(f32,) {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<[f32; 1]> for Vector1 {
    fn as_mut(&mut self) -> &mut [f32; 1] {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<f32> for Vector1 {
    fn as_mut(&mut self) -> &mut f32 {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<(f32,)> for Vector1 {
    fn as_mut(&mut self) -> &mut (f32,) {
        unsafe { mem::transmute(self) }
    }
}

impl ops::Index<usize> for Vector1 {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[f32; 1] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::Range<usize>> for Vector1 {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        let v: &[f32; 1] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::RangeTo<usize>> for Vector1 {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        let v: &[f32; 1] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::RangeFrom<usize>> for Vector1 {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        let v: &[f32; 1] = self.as_ref();
        &v[index]
    }
}

impl fmt::Debug for Vector1 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector1 ")?;
        <[f32; 1] as fmt::Debug>::fmt(self.as_ref(), f)
    }
}

impl fmt::Display for Vector1 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector2 [{:.2}]", self.x)
    }
}

impl From<f32> for Vector1 {
    #[inline]
    fn from(v: f32) -> Vector1 {
        Vector1 { x: v }
    }
}

impl From<[f32; 1]> for Vector1 {
    #[inline]
    fn from(v: [f32; 1]) -> Vector1 {
        Vector1 { x: v[0] }
    }
}

impl From<&[f32; 1]> for Vector1 {
    #[inline]
    fn from(v: &[f32; 1]) -> Vector1 {
        Vector1 { x: v[0] }
    }
}

impl<'a> From<&'a [f32; 1]> for &'a Vector1 {
    #[inline]
    fn from(v: &'a [f32; 1]) -> &'a Vector1 {
        unsafe { mem::transmute(v) }
    }
}

impl ops::Neg for Vector1 {
    type Output = Vector1;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector1 { x: -self.x }
    }
}

impl ops::Neg for &Vector1 {
    type Output = Vector1;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector1 { x: -self.x }
    }
}

impl ops::Add<Vector1> for &Vector1 {
    type Output = Vector1;

    fn add(self, other: Vector1) -> Self::Output {
        Vector1 {
            x: self.x + other.x,
        }
    }
}

impl ops::Add<Vector1> for Vector1 {
    type Output = Vector1;

    fn add(self, other: Vector1) -> Self::Output {
        Vector1 {
            x: self.x + other.x,
        }
    }
}

impl ops::Add<&Vector1> for Vector1 {
    type Output = Vector1;

    fn add(self, other: &Vector1) -> Self::Output {
        Vector1 {
            x: self.x + other.x,
        }
    }
}

impl<'a, 'b> ops::Add<&'b Vector1> for &'a Vector1 {
    type Output = Vector1;

    fn add(self, other: &'b Vector1) -> Self::Output {
        Vector1 {
            x: self.x + other.x,
        }
    }
}

impl ops::Sub<Vector1> for &Vector1 {
    type Output = Vector1;

    fn sub(self, other: Vector1) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl ops::Sub<Vector1> for Vector1 {
    type Output = Vector1;

    fn sub(self, other: Vector1) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl ops::Sub<&Vector1> for Vector1 {
    type Output = Vector1;

    fn sub(self, other: &Vector1) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl<'a, 'b> ops::Sub<&'b Vector1> for &'a Vector1 {
    type Output = Vector1;

    fn sub(self, other: &'b Vector1) -> Self::Output {
        Vector1 {
            x: self.x - other.x,
        }
    }
}

impl ops::AddAssign<Vector1> for Vector1 {
    fn add_assign(&mut self, other: Vector1) {
        self.x = self.x + other.x;
    }
}

impl ops::AddAssign<&Vector1> for Vector1 {
    fn add_assign(&mut self, other: &Vector1) {
        self.x = self.x + other.x;
    }
}

impl ops::SubAssign<Vector1> for Vector1 {
    fn sub_assign(&mut self, other: Vector1) {
        self.x = self.x - other.x;
    }
}

impl ops::SubAssign<&Vector1> for Vector1 {
    fn sub_assign(&mut self, other: &Vector1) {
        self.x = self.x - other.x;
    }
}

impl ops::Mul<f32> for Vector1 {
    type Output = Vector1;

    fn mul(self, other: f32) -> Vector1 {
        Vector1 {
            x: self.x * other,
        }
    }
}

impl ops::Mul<f32> for &Vector1 {
    type Output = Vector1;

    fn mul(self, other: f32) -> Vector1 {
        Vector1 {
            x: self.x * other,
        }
    }
}

impl ops::MulAssign<f32> for Vector1 {
    fn mul_assign(&mut self, other: f32) {
        self.x *= other;
    }
}

impl ops::Div<f32> for Vector1 {
    type Output = Vector1;

    fn div(self, other: f32) -> Vector1 {
        Vector1 {
            x: self.x / other,
        }
    }
}

impl ops::Div<f32> for &Vector1 {
    type Output = Vector1;

    fn div(self, other: f32) -> Vector1 {
        Vector1 {
            x: self.x / other,
        }
    }
}

impl ops::DivAssign<f32> for Vector1 {
    fn div_assign(&mut self, other: f32) {
        self.x = self.x / other;
    }
}

impl ops::Rem<f32> for Vector1 {
    type Output = Vector1;

    fn rem(self, other: f32) -> Self::Output {
        let x = self.x % other;
        
        Vector1 { x: x }
    }
}

impl ops::Rem<f32> for &Vector1 {
    type Output = Vector1;

    fn rem(self, other: f32) -> Self::Output {
        let x = self.x % other;
        
        Vector1 { x: x }
    }
}

impl ops::RemAssign<f32> for Vector1 {
    fn rem_assign(&mut self, other: f32) {
        self.x %= other;
    }
}

impl Zero for Vector1 {
    fn zero() -> Vector1 {
        Vector1 { x: 0.0 }
    }

    fn is_zero(&self) -> bool {
        self.x == 0.0
    }
}

impl DotProduct<Vector1> for Vector1 {
    fn dot(self, other: Vector1) -> f32 {
        self.x * other.x
    }
}

impl DotProduct<&Vector1> for Vector1 {
    fn dot(self, other: &Vector1) -> f32 {
        self.x * other.x
    }
}

impl DotProduct<Vector1> for &Vector1 {
    fn dot(self, other: Vector1) -> f32 {
        self.x * other.x
    }
}

impl<'a, 'b> DotProduct<&'a Vector1> for &'b Vector1 {
    fn dot(self, other: &'a Vector1) -> f32 {
        self.x * other.x
    }
}

impl Lerp<Vector1> for Vector1 {
    type Output = Vector1;

    fn lerp(self, other: Vector1, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl Lerp<&Vector1> for Vector1 {
    type Output = Vector1;

    fn lerp(self, other: &Vector1, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl Lerp<Vector1> for &Vector1 {
    type Output = Vector1;

    fn lerp(self, other: Vector1, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<'a, 'b> Lerp<&'a Vector1> for &'b Vector1 {
    type Output = Vector1;

    fn lerp(self, other: &'a Vector1, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl Magnitude<Vector1> for Vector1 {}
impl Magnitude<Vector1> for &Vector1 {}

///
/// A representation of two-dimensional vectors with a Euclidean metric.
///
#[derive(Copy, Clone, PartialEq)]
pub struct Vector2 {
   pub x: f32,
   pub y: f32,
}

impl Vector2 {
    ///
    /// Create a new vector.
    ///
    pub fn new(x: f32, y: f32) -> Vector2 {
        Vector2 { x: x, y: y }
    }

    #[inline]
    pub fn unit_x() -> Vector2 {
        Vector2 { x: 1.0, y: 0.0 }
    }

    #[inline]
    pub fn unit_y() -> Vector2 {
        Vector2 { x: 0.0, y: 1.0 }
    }
}

impl Array for Vector2 {
    type Element = f32;

    #[inline]
    fn len() -> usize {
        2
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

impl AsRef<[f32; 2]> for Vector2 {
    fn as_ref(&self) -> &[f32; 2] {
        unsafe { mem::transmute(self) }
    }
}

impl AsRef<(f32, f32)> for Vector2 {
    fn as_ref(&self) -> &(f32, f32) {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<[f32; 2]> for Vector2 {
    fn as_mut(&mut self) -> &mut [f32; 2] {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<(f32, f32)> for Vector2 {
    fn as_mut(&mut self) -> &mut (f32, f32) {
        unsafe { mem::transmute(self) }
    }
}

impl ops::Index<usize> for Vector2 {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[f32; 2] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::Range<usize>> for Vector2 {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        let v: &[f32; 2] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::RangeTo<usize>> for Vector2 {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        let v: &[f32; 2] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::RangeFrom<usize>> for Vector2 {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        let v: &[f32; 2] = self.as_ref();
        &v[index]
    }
}

impl fmt::Debug for Vector2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector2 ")?;
        <[f32; 2] as fmt::Debug>::fmt(self.as_ref(), f)
    }
}

impl fmt::Display for Vector2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector2 [{:.2}, {:.2}]", self.x, self.y)
    }
}

impl From<(f32, f32)> for Vector2 {
    #[inline]
    fn from((x, y): (f32, f32)) -> Vector2 {
        Vector2 { x: x, y: y }
    }
}

impl From<[f32; 2]> for Vector2 {
    #[inline]
    fn from(v: [f32; 2]) -> Vector2 {
        Vector2 { x: v[0], y: v[1] }
    }
}

impl From<&[f32; 2]> for Vector2 {
    #[inline]
    fn from(v: &[f32; 2]) -> Vector2 {
        Vector2 { x: v[0], y: v[1] }
    }
}

impl<'a> From<&'a [f32; 2]> for &'a Vector2 {
    #[inline]
    fn from(v: &'a [f32; 2]) -> &'a Vector2 {
        unsafe { mem::transmute(v) }
    }
}

impl ops::Neg for Vector2 {
    type Output = Vector2;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector2 { x: -self.x, y: -self.y }
    }
}

impl ops::Neg for &Vector2 {
    type Output = Vector2;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector2 { x: -self.x, y: -self.y }
    }
}

impl ops::Add<Vector2> for &Vector2 {
    type Output = Vector2;

    fn add(self, other: Vector2) -> Self::Output {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl ops::Add<Vector2> for Vector2 {
    type Output = Vector2;

    fn add(self, other: Vector2) -> Self::Output {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl ops::Add<&Vector2> for Vector2 {
    type Output = Vector2;

    fn add(self, other: &Vector2) -> Self::Output {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<'a, 'b> ops::Add<&'b Vector2> for &'a Vector2 {
    type Output = Vector2;

    fn add(self, other: &'b Vector2) -> Self::Output {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl ops::Sub<Vector2> for &Vector2 {
    type Output = Vector2;

    fn sub(self, other: Vector2) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl ops::Sub<Vector2> for Vector2 {
    type Output = Vector2;

    fn sub(self, other: Vector2) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl ops::Sub<&Vector2> for Vector2 {
    type Output = Vector2;

    fn sub(self, other: &Vector2) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<'a, 'b> ops::Sub<&'b Vector2> for &'a Vector2 {
    type Output = Vector2;

    fn sub(self, other: &'b Vector2) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl ops::AddAssign<Vector2> for Vector2 {
    fn add_assign(&mut self, other: Vector2) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
    }
}

impl ops::AddAssign<&Vector2> for Vector2 {
    fn add_assign(&mut self, other: &Vector2) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
    }
}

impl ops::SubAssign<Vector2> for Vector2 {
    fn sub_assign(&mut self, other: Vector2) {
        self.x = self.x - other.x;
        self.y = self.y - other.y;
    }
}

impl ops::SubAssign<&Vector2> for Vector2 {
    fn sub_assign(&mut self, other: &Vector2) {
        self.x = self.x - other.x;
        self.y = self.y - other.y;
    }
}

impl ops::Mul<f32> for Vector2 {
    type Output = Vector2;

    fn mul(self, other: f32) -> Vector2 {
        Vector2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl ops::Mul<f32> for &Vector2 {
    type Output = Vector2;

    fn mul(self, other: f32) -> Vector2 {
        Vector2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl ops::MulAssign<f32> for Vector2 {
    fn mul_assign(&mut self, other: f32) {
        self.x *= other;
        self.y *= other;
    }
}

impl ops::Div<f32> for Vector2 {
    type Output = Vector2;

    fn div(self, other: f32) -> Vector2 {
        Vector2 {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

impl ops::Div<f32> for &Vector2 {
    type Output = Vector2;

    fn div(self, other: f32) -> Vector2 {
        Vector2 {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

impl ops::DivAssign<f32> for Vector2 {
    fn div_assign(&mut self, other: f32) {
        self.x = self.x / other;
        self.y = self.y / other;
    }
}

impl ops::Rem<f32> for Vector2 {
    type Output = Vector2;

    fn rem(self, other: f32) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        
        Vector2 { x: x, y: y }
    }
}

impl ops::Rem<f32> for &Vector2 {
    type Output = Vector2;

    fn rem(self, other: f32) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        
        Vector2 { x: x, y: y }
    }
}

impl ops::RemAssign<f32> for Vector2 {
    fn rem_assign(&mut self, other: f32) {
        self.x %= other;
        self.y %= other;
    }
}

impl Zero for Vector2 {
    fn zero() -> Vector2 {
        Vector2 { x: 0.0, y: 0.0 }
    }

    fn is_zero(&self) -> bool {
        self.x == 0.0 && self.y == 0.0
    }
}

impl Metric<Vector2> for Vector2 {
    #[inline]
    fn distance2(self, to: Vector2) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
    
        dx_2 + dy_2
    }
}

impl Metric<&Vector2> for Vector2 {
    #[inline]
    fn distance2(self, to: &Vector2) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
    
        dx_2 + dy_2
    }
}

impl Metric<Vector2> for &Vector2 {
    #[inline]
    fn distance2(self, to: Vector2) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
    
        dx_2 + dy_2
    }
}

impl<'a, 'b> Metric<&'a Vector2> for &'b Vector2 {
    #[inline]
    fn distance2(self, to: &'a Vector2) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
    
        dx_2 + dy_2
    }
}

impl DotProduct<Vector2> for Vector2 {
    fn dot(self, other: Vector2) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl DotProduct<&Vector2> for Vector2 {
    fn dot(self, other: &Vector2) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl DotProduct<Vector2> for &Vector2 {
    fn dot(self, other: Vector2) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl<'a, 'b> DotProduct<&'a Vector2> for &'b Vector2 {
    fn dot(self, other: &'a Vector2) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl Lerp<Vector2> for Vector2 {
    type Output = Vector2;

    fn lerp(self, other: Vector2, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl Lerp<&Vector2> for Vector2 {
    type Output = Vector2;

    fn lerp(self, other: &Vector2, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl Lerp<Vector2> for &Vector2 {
    type Output = Vector2;

    fn lerp(self, other: Vector2, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<'a, 'b> Lerp<&'a Vector2> for &'b Vector2 {
    type Output = Vector2;

    fn lerp(self, other: &'a Vector2, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

/// A representation of three-dimensional vectors with a Euclidean metric.
#[derive(Copy, Clone, PartialEq)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector3 {
    /// Create a new vector.
    pub fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x: x, y: y, z: z }
    }

    #[inline]
    pub fn unit_x() -> Vector3 {
        Vector3 { x: 1.0, y: 0.0, z: 0.0 }
    }

    #[inline]
    pub fn unit_y() -> Vector3 {
        Vector3 { x: 0.0, y: 1.0, z: 0.0 }
    }
    
    #[inline]
    pub fn unit_z() -> Vector3 {
        Vector3 { x: 0.0, y: 0.0, z: 1.0 }
    }

    /// Compute the cross product of two three-dimensional vectors. Note that
    /// with the vectors used in computer graphics (two, three, and four dimensions),
    /// the cross product is defined only in three dimensions. Also note that the 
    /// cross product is the hodge dual of the corresponding 2-vector representing 
    /// the surface element that the crossed vector is normal to. That is, 
    /// given vectors `u` and `v`, `u x v == *(u /\ v)`, where `*(.)` denotes the hodge dual.
    pub fn cross(&self, other: &Vector3) -> Vector3 {
        let x = self.y * other.z - self.z * other.y;
        let y = self.z * other.x - self.x * other.z;
        let z = self.x * other.y - self.y * other.x;
    
        Vector3::new(x, y, z)
    }
}

impl Array for Vector3 {
    type Element = f32;

    #[inline]
    fn len() -> usize {
        3
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


impl AsRef<[f32; 3]> for Vector3 {
    fn as_ref(&self) -> &[f32; 3] {
        unsafe { mem::transmute(self) }
    }
}

impl AsRef<(f32, f32, f32)> for Vector3 {
    fn as_ref(&self) -> &(f32, f32, f32) {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<[f32; 3]> for Vector3 {
    fn as_mut(&mut self) -> &mut [f32; 3] {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<(f32, f32, f32)> for Vector3 {
    fn as_mut(&mut self) -> &mut (f32, f32, f32) {
        unsafe { mem::transmute(self) }
    }
}

impl ops::Index<usize> for Vector3 {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[f32; 3] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::Range<usize>> for Vector3 {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        let v: &[f32; 3] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::RangeTo<usize>> for Vector3 {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        let v: &[f32; 3] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::RangeFrom<usize>> for Vector3 {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        let v: &[f32; 3] = self.as_ref();
        &v[index]
    }
}

impl fmt::Debug for Vector3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector3 ")?;
        <[f32; 3] as fmt::Debug>::fmt(self.as_ref(), f)
    }
}

impl fmt::Display for Vector3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector3 [{:.2}, {:.2}, {:.2}]", self.x, self.y, self.z)
    }
}

impl From<(f32, f32, f32)> for Vector3 {
    #[inline]
    fn from((x, y, z): (f32, f32, f32)) -> Vector3 {
        Vector3::new(x, y, z)
    }
}

impl From<(Vector2, f32)> for Vector3 {
    #[inline]
    fn from((v, z): (Vector2, f32)) -> Vector3 {
        Vector3::new(v.x, v.y, z)
    }
}

impl From<(&Vector2, f32)> for Vector3 {
    #[inline]
    fn from((v, z): (&Vector2, f32)) -> Vector3 {
        Vector3::new(v.x, v.y, z)
    }
}

impl From<[f32; 3]> for Vector3 {
    #[inline]
    fn from(v: [f32; 3]) -> Vector3 {
        Vector3::new(v[0], v[1], v[2])
    }
}

impl From<Vector4> for Vector3 {
    #[inline]
    fn from(v: Vector4) -> Vector3 {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl From<&Vector4> for Vector3 {
    #[inline]
    fn from(v: &Vector4) -> Vector3 {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl<'a> From<&'a [f32; 3]> for &'a Vector3 {
    #[inline]
    fn from(v: &'a [f32; 3]) -> &'a Vector3 {
        unsafe { mem::transmute(v) }
    }
}

impl<'a> From<&'a (f32, f32, f32)> for &'a Vector3 {
    #[inline]
    fn from(v: &'a (f32, f32, f32)) -> &'a Vector3 {
        unsafe { mem::transmute(v) }
    }
}

impl ops::Neg for Vector3 {
    type Output = Vector3;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector3 { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl ops::Neg for &Vector3 {
    type Output = Vector3;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector3 { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl ops::Add<Vector3> for &Vector3 {
    type Output = Vector3;

    fn add(self, other: Vector3) -> Self::Output {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl ops::Add<Vector3> for Vector3 {
    type Output = Vector3;

    fn add(self, other: Vector3) -> Self::Output {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl ops::Add<&Vector3> for Vector3 {
    type Output = Vector3;

    fn add(self, other: &Vector3) -> Self::Output {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,               
        }
    }
}

impl<'a, 'b> ops::Add<&'b Vector3> for &'a Vector3 {
    type Output = Vector3;

    fn add(self, other: &'b Vector3) -> Self::Output {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl ops::Sub<Vector3> for &Vector3 {
    type Output = Vector3;

    fn sub(self, other: Vector3) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl ops::Sub<Vector3> for Vector3 {
    type Output = Vector3;

    fn sub(self, other: Vector3) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl ops::Sub<&Vector3> for Vector3 {
    type Output = Vector3;

    fn sub(self, other: &Vector3) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,               
        }
    }
}

impl<'a, 'b> ops::Sub<&'b Vector3> for &'a Vector3 {
    type Output = Vector3;

    fn sub(self, other: &'b Vector3) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl ops::AddAssign<Vector3> for Vector3 {
    fn add_assign(&mut self, other: Vector3) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl ops::AddAssign<&Vector3> for Vector3 {
    fn add_assign(&mut self, other: &Vector3) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl ops::SubAssign<Vector3> for Vector3 {
    fn sub_assign(&mut self, other: Vector3) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl ops::SubAssign<&Vector3> for Vector3 {
    fn sub_assign(&mut self, other: &Vector3) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl ops::Mul<f32> for Vector3 {
    type Output = Vector3;

    fn mul(self, other: f32) -> Vector3 {
        Vector3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl ops::Mul<f32> for &Vector3 {
    type Output = Vector3;

    fn mul(self, other: f32) -> Vector3 {
        Vector3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl ops::MulAssign<f32> for Vector3 {
    fn mul_assign(&mut self, other: f32) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
    }
}

impl ops::Div<f32> for Vector3 {
    type Output = Vector3;

    fn div(self, other: f32) -> Vector3 {
        Vector3 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

impl ops::Div<f32> for &Vector3 {
    type Output = Vector3;

    fn div(self, other: f32) -> Vector3 {
        Vector3 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

impl ops::DivAssign<f32> for Vector3 {
    fn div_assign(&mut self, other: f32) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
    }
}

impl ops::Rem<f32> for Vector3 {
    type Output = Vector3;

    fn rem(self, other: f32) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        
        Vector3 { x: x, y: y, z: z }
    }
}

impl ops::Rem<f32> for &Vector3 {
    type Output = Vector3;

    fn rem(self, other: f32) -> Self::Output {
        let x = self.x % other;
        let y = self.y % other;
        let z = self.z % other;
        
        Vector3 { x: x, y: y, z: z }
    }
}

impl ops::RemAssign<f32> for Vector3 {
    fn rem_assign(&mut self, other: f32) {
        self.x %= other;
        self.y %= other;
        self.z %= other;
    }
}

impl Zero for Vector3 {
    #[inline]
    fn zero() -> Vector3 {
        Vector3 { x: 0.0, y: 0.0, z: 0.0 }
    }

    fn is_zero(&self) -> bool {
        self.x == 0.0 && self.y == 0.0 && self.z == 0.0
    }
}

impl Metric<Vector3> for Vector3 {
    #[inline]
    fn distance2(self, to: Vector3) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
        let dz_2 = (to.z - self.z) * (to.z - self.z);
    
        dx_2 + dy_2 + dz_2
    }
}

impl Metric<&Vector3> for Vector3 {
    #[inline]
    fn distance2(self, to: &Vector3) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
        let dz_2 = (to.z - self.z) * (to.z - self.z);
    
        dx_2 + dy_2 + dz_2
    }
}

impl Metric<Vector3> for &Vector3 {
    #[inline]
    fn distance2(self, to: Vector3) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
        let dz_2 = (to.z - self.z) * (to.z - self.z);
    
        dx_2 + dy_2 + dz_2
    }
}

impl<'a, 'b> Metric<&'a Vector3> for &'b Vector3 {
    #[inline]
    fn distance2(self, to: &Vector3) -> f32 {
        let dx_2 = (to.x - self.x) * (to.x - self.x);
        let dy_2 = (to.y - self.y) * (to.y - self.y);
        let dz_2 = (to.z - self.z) * (to.z - self.z);
    
        dx_2 + dy_2 + dz_2
    }
}

impl DotProduct<Vector3> for Vector3 {
    fn dot(self, other: Vector3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl DotProduct<&Vector3> for Vector3 {
    fn dot(self, other: &Vector3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl DotProduct<Vector3> for &Vector3 {
    fn dot(self, other: Vector3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl<'a, 'b> DotProduct<&'a Vector3> for &'b Vector3 {
    fn dot(self, other: &'a Vector3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl Lerp<Vector3> for Vector3 {
    type Output = Vector3;

    fn lerp(self, other: Vector3, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl Lerp<&Vector3> for Vector3 {
    type Output = Vector3;

    fn lerp(self, other: &Vector3, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl Lerp<Vector3> for &Vector3 {
    type Output = Vector3;

    fn lerp(self, other: Vector3, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<'a, 'b> Lerp<&'a Vector3> for &'b Vector3 {
    type Output = Vector3;

    fn lerp(self, other: &'a Vector3, amount: f32) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl Magnitude<Vector3> for Vector3 {}
impl Magnitude<Vector3> for &Vector3 {}

///
/// A representation of four-dimensional vectors with a Euclidean metric.
///
#[derive(Copy, Clone)]
pub struct Vector4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vector4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Vector4 {
        Vector4 { x: x, y: y, z: z, w: w }
    }

    #[inline]
    pub fn unit_x() -> Vector4 {
        Vector4 { x: 1.0, y: 0.0, z: 0.0, w: 0.0 }
    }

    #[inline]
    pub fn unit_y() -> Vector4 {
        Vector4 { x: 0.0, y: 1.0, z: 0.0, w: 0.0 }
    }
    
    #[inline]
    pub fn unit_z() -> Vector4 {
        Vector4 { x: 0.0, y: 0.0, z: 1.0, w: 0.0 }
    }

    #[inline]
    pub fn unit_w() -> Vector4 {
        Vector4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    }
}

impl Array for Vector4 {
    type Element = f32;

    #[inline]
    fn len() -> usize {
        4
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

impl AsRef<[f32; 4]> for Vector4 {
    fn as_ref(&self) -> &[f32; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl AsRef<(f32, f32, f32, f32)> for Vector4 {
    fn as_ref(&self) -> &(f32, f32, f32, f32) {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<[f32; 4]> for Vector4 {
    fn as_mut(&mut self) -> &mut [f32; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<(f32, f32, f32, f32)> for Vector4 {
    fn as_mut(&mut self) -> &mut (f32, f32, f32, f32) {
        unsafe { mem::transmute(self) }
    }
}

impl ops::Index<usize> for Vector4 {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[f32; 4] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::Range<usize>> for Vector4 {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        let v: &[f32; 4] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::RangeTo<usize>> for Vector4 {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        let v: &[f32; 4] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::RangeFrom<usize>> for Vector4 {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        let v: &[f32; 4] = self.as_ref();
        &v[index]
    }
}

impl From<(f32, f32, f32, f32)> for Vector4 {
    #[inline]
    fn from((x, y, z, w): (f32, f32, f32, f32)) -> Vector4 {
        Vector4::new(x, y, z, w)
    }
}

impl From<(Vector2, f32, f32)> for Vector4 {
    #[inline]
    fn from((v, z, w): (Vector2, f32, f32)) -> Vector4 {
        Vector4::new(v.x, v.y, z, w)
    }
}

impl From<(&Vector2, f32, f32)> for Vector4 {
    #[inline]
    fn from((v, z, w): (&Vector2, f32, f32)) -> Vector4 {
        Vector4::new(v.x, v.y, z, w)
    }
}

impl From<(Vector3, f32)> for Vector4 {
    #[inline]
    fn from((v, w): (Vector3, f32)) -> Vector4 {
        Vector4::new(v.x, v.y, v.z, w)
    }
}

impl From<(&Vector3, f32)> for Vector4 {
    #[inline]
    fn from((v, w): (&Vector3, f32)) -> Vector4 {
        Vector4::new(v.x, v.y, v.z, w)
    }
}

impl From<[f32; 4]> for Vector4 {
    #[inline]
    fn from(v: [f32; 4]) -> Vector4 {
        unsafe { mem::transmute(v) }
    }
}

impl From<&[f32; 4]> for Vector4 {
    #[inline]
    fn from(v: &[f32; 4]) -> Vector4 {
        Vector4::new(v[0], v[1], v[2], v[3])
    }
}

impl From<&(f32, f32, f32, f32)> for Vector4 {
    #[inline]
    fn from(v: &(f32, f32, f32, f32)) -> Vector4 {
        Vector4::new(v.0, v.1, v.2, v.3)
    }
}

impl<'a> From<&'a [f32; 4]> for &'a Vector4 {
    #[inline]
    fn from(v: &'a [f32; 4]) -> &'a Vector4 {
        unsafe { mem::transmute(v) }
    }
}

impl<'a> From<&'a (f32, f32, f32, f32)> for &'a Vector4 {
    #[inline]
    fn from(v: &'a (f32, f32, f32, f32)) -> &'a Vector4 {
        unsafe { mem::transmute(v) }
    }
}

impl fmt::Debug for Vector4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector4 ")?;
        <[f32; 4] as fmt::Debug>::fmt(self.as_ref(), f)
    }
}

impl fmt::Display for Vector4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector4 [{:.2}, {:.2}, {:.2}, {:.2}]", self.x, self.y, self.z, self.w)
    }
}

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


#[cfg(test)]
mod vec1_tests {
    use std::slice::Iter;
    use super::Vector1;


    struct TestCase {
        c: f32,
        v1: Vector1,
        v2: Vector1,
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
    fn test_addition() {
        for test in test_cases().iter() {
            let expected = Vector1::from(test.v1.x + test.v2.x);
            let result = test.v1 + test.v2;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_subtraction() {
        for test in test_cases().iter() {
            let expected = Vector1::from(test.v1.x + test.v2.x);
            let result = test.v1 + test.v2;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_scalar_multiplication() {
        for test in test_cases().iter() {
            let expected = Vector1::from(test.c * test.v1.x);
            let result = test.v1 * test.c;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_scalar_division() {
        for test in test_cases().iter() {
            let expected = Vector1::from(test.v1.x / test.c);
            let result = test.v1 / test.c;
            assert_eq!(result, expected);
        }
    }
}


#[cfg(test)]
mod vec2_tests {
    use std::slice::Iter;
    use super::Vector2;

    struct TestCase {
        c: f32,
        v1: Vector2,
        v2: Vector2,
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
}

#[cfg(test)]
mod vec3_tests {
    use std::slice::Iter;
    use super::Vector3;

    struct TestCase {
        c: f32,
        x: Vector3,
        y: Vector3,
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
}


#[cfg(test)]
mod vec4_tests {
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

