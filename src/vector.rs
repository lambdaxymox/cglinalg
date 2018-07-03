use std::fmt;
use std::mem;
use std::ops;
use std::cmp;


// Constants used to convert degrees into radians.
pub const M_PI: f32 = 3.14159265358979323846264338327950288;
pub const TAU: f32 = 2.0 * M_PI;
pub const ONE_DEG_IN_RAD: f32 = (2.0 * M_PI) / 360.0; // == 0.017444444
pub const ONE_RAD_IN_DEG: f32 = 360.0 / (2.0 * M_PI); // == 57.2957795
pub const EPSILON: f32 = 0.00001; 


///
/// Construct a new two-dimensional vector in the style of
/// a GLSL vec3 constructor.
///
#[inline]
pub fn vec2<T: Into<Vector2>>(v: T) -> Vector2 {
    v.into()
}

///
/// A representation of two-dimensional vectors, with a
/// Euclidean metric.
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

    ///
    /// Generate a zero vector.
    ///
    pub fn zero() -> Vector2 { 
        Vector2 { x: 0.0, y: 0.0 }
    }

    ///
    /// Compute the norm (length) of a vector.
    ///
    pub fn norm(&self) -> f32 {
        f32::sqrt(self.x * self.x + self.y * self.y)
    }

    ///
    /// Compute the squared norm (length) of a vector.
    ///
    pub fn norm2(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    ///
    /// Convert an arbitrary vector into a unit vector.
    ///
    pub fn normalize(&self) -> Vector2 {
        let norm_v = self.norm();
        if norm_v == 0.0 {
            return Vector2::zero();
        }

        Vector2::new(self.x / norm_v, self.y / norm_v)
    }

    ///
    /// Compute the dot product of two vectors.
    ///
    pub fn dot(&self, other: &Vector2) -> f32 {
        self.x * other.x + self.y * other.y
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
        try!(write!(f, "Vector2 "));
        <[f32; 2] as fmt::Debug>::fmt(self.as_ref(), f)
    }
}

impl fmt::Display for Vector2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:.2}, {:.2}]", self.x, self.y)
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

impl<'a> From<&'a [f32; 2]> for Vector2 {
    #[inline]
    fn from(v: &'a [f32; 2]) -> Vector2 {
        Vector2 { x: v[0], y: v[1] }
    }
}

impl<'a> From<&'a [f32; 2]> for &'a Vector2 {
    #[inline]
    fn from(v: &'a [f32; 2]) -> &'a Vector2 {
        unsafe { mem::transmute(v) }
    }
}

impl<'a> ops::Add<Vector2> for &'a Vector2 {
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

impl<'a> ops::Add<&'a Vector2> for Vector2 {
    type Output = Vector2;

    fn add(self, other: &'a Vector2) -> Self::Output {
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

impl ops::Add<f32> for Vector2 {
    type Output = Vector2;

    fn add(self, other: f32) -> Self::Output {
        Vector2 {
            x: self.x + other,
            y: self.y + other,
        }
    }
}

impl<'a> ops::Sub<Vector2> for &'a Vector2 {
    type Output = Vector2;

    fn sub(self, other: Vector2) -> Self::Output {
        Vector2 {
            x: self.x - other.y,
            y: self.y - other.y,
        }
    }
}

impl ops::Sub<Vector2> for Vector2 {
    type Output = Vector2;

    fn sub(self, other: Vector2) -> Self::Output {
        Vector2 {
            x: self.x - other.y,
            y: self.y - other.y,
        }
    }
}

impl<'a> ops::Sub<&'a Vector2> for Vector2 {
    type Output = Vector2;

    fn sub(self, other: &'a Vector2) -> Self::Output {
        Vector2 {
            x: self.x - other.y,
            y: self.y - other.y,
        }
    }
}

impl<'a, 'b> ops::Sub<&'b Vector2> for &'a Vector2 {
    type Output = Vector2;

    fn sub(self, other: &'b Vector2) -> Self::Output {
        Vector2 {
            x: self.x - other.y,
            y: self.y - other.y,
        }
    }
}

impl ops::Sub<f32> for Vector2 {
    type Output = Vector2;

    fn sub(self, other: f32) -> Self::Output {
        Vector2 {
            x: self.x - other,
            y: self.y - other,
        }
    }
}

impl ops::AddAssign<Vector2> for Vector2 {
    fn add_assign(&mut self, other: Vector2) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
    }
}

impl<'a> ops::AddAssign<&'a Vector2> for Vector2 {
    fn add_assign(&mut self, other: &'a Vector2) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
    }
}

impl<'a> ops::AddAssign<Vector2> for &'a mut Vector2 {
    fn add_assign(&mut self, other: Vector2) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
    }
}

impl<'a, 'b> ops::AddAssign<&'a Vector2> for &'b mut Vector2 {
    fn add_assign(&mut self, other: &'a Vector2) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
    }
}

impl ops::AddAssign<f32> for Vector2 {
    fn add_assign(&mut self, other: f32) {
        self.x = self.x + other;
        self.y = self.y + other;
    }
}

impl ops::SubAssign<Vector2> for Vector2 {
    fn sub_assign(&mut self, other: Vector2) {
        self.x = self.x - other.x;
        self.y = self.y - other.y;
    }
}

impl<'a> ops::SubAssign<&'a Vector2> for Vector2 {
    fn sub_assign(&mut self, other: &'a Vector2) {
        self.x = self.x - other.x;
        self.y = self.y - other.y;
    }
}

impl<'a> ops::SubAssign<Vector2> for &'a mut Vector2 {
    fn sub_assign(&mut self, other: Vector2) {
        self.x = self.x - other.x;
        self.y = self.y - other.y;
    }
}

impl<'a, 'b> ops::SubAssign<&'a Vector2> for &'b mut Vector2 {
    fn sub_assign(&mut self, other: &'a Vector2) {
        self.x = self.x - other.x;
        self.y = self.y - other.y;
    }
}

impl ops::SubAssign<f32> for Vector2 {
    fn sub_assign(&mut self, other: f32) {
        self.x = self.x - other;
        self.y = self.y - other;
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

impl<'a> ops::Mul<f32> for &'a Vector2 {
    type Output = Vector2;

    fn mul(self, other: f32) -> Vector2 {
        Vector2 {
            x: self.x * other,
            y: self.y * other,
        }
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

impl<'a> ops::Div<f32> for &'a Vector2 {
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

impl<'a> ops::DivAssign<f32> for &'a mut Vector2 {
    fn div_assign(&mut self, other: f32) {
        self.x = self.x / other;
        self.y = self.y / other;
    }
}


///
/// A representation of three-dimensional vectors, with a
/// Euclidean metric.
///
#[derive(Copy, Clone, PartialEq)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector3 {
    ///
    /// Create a new vector.
    ///
    pub fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x: x, y: y, z: z }
    }

    ///
    /// Generate a zero vector.
    ///
    pub fn zero() -> Vector3 {
        Vector3 { x: 0.0, y: 0.0, z: 0.0 }
    }
    
    ///
    /// Compute the norm (length) of a vector.
    ///
    pub fn norm(&self) -> f32 {
        f32::sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    }

    ///
    /// Compute the squared norm (length) of a vector.
    ///
    pub fn norm2(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    ///
    /// Convert an arbitrary vector into a unit vector.
    ///
    pub fn normalize(&self) -> Vector3 {
        let norm_v = self.norm();
        if norm_v == 0.0 {
            return Vector3::zero();
        }

        Vector3::new(self.x / norm_v, self.y / norm_v, self.z / norm_v)
    }

    ///
    /// Compute the dot product of two vectors.
    ///
    pub fn dot(&self, other: &Vector3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    ///
    /// Compute the cross product of two three-dimensional vectors. Note that
    /// with the vectors used in computer graphics (two, three, and four dimensions),
    /// the cross product is defined only in three dimensions. Also note that the 
    /// cross product is the hodge dual of the corresponding 2-vector representing 
    /// the surface element that the crossed vector is normal to. That is, 
    /// given vectors u and v, u x v == *(u /\ v), where *(.) denotes the hodge dual.
    ///
    pub fn cross(&self, other: &Vector3) -> Vector3 {
        let x = self.y * other.z - self.z * other.y;
        let y = self.z * other.x - self.x * other.z;
        let z = self.x * other.y - self.y * other.x;
    
        Vector3::new(x, y, z)
    }

    ///
    /// Compute the squared distance between two vectors.
    ///
    pub fn get_squared_dist(&self, to: &Vector3) -> f32 {
        let x = (to.x - self.x) * (to.x - self.x);
        let y = (to.y - self.y) * (to.y - self.y);
        let z = (to.z - self.z) * (to.z - self.z);
    
        x + y + z
    }
}

///
/// Construct a new three-dimensional vector in the style of
/// a GLSL vec3 constructor.
///
#[inline]
pub fn vec3<T: Into<Vector3>>(v: T) -> Vector3 {
    v.into()
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
        try!(write!(f, "Vector3 "));
        <[f32; 3] as fmt::Debug>::fmt(self.as_ref(), f)
    }
}

impl fmt::Display for Vector3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:.2}, {:.2}, {:.2}]", self.x, self.y, self.z)
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

impl<'a> From<(&'a Vector2, f32)> for Vector3 {
    #[inline]
    fn from((v, z): (&'a Vector2, f32)) -> Vector3 {
        Vector3::new(v.x, v.y, z)
    }
}

impl<'a> From<Vector4> for Vector3 {
    #[inline]
    fn from(v: Vector4) -> Vector3 {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl<'a> From<&'a Vector4> for Vector3 {
    #[inline]
    fn from(v: &'a Vector4) -> Vector3 {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl<'a> ops::Add<Vector3> for &'a Vector3 {
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

impl<'a> ops::Add<&'a Vector3> for Vector3 {
    type Output = Vector3;

    fn add(self, other: &'a Vector3) -> Self::Output {
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

impl ops::Add<f32> for Vector3 {
    type Output = Vector3;

    fn add(self, other: f32) -> Self::Output {
        Vector3 {
            x: self.x + other,
            y: self.y + other,
            z: self.z + other,
        }
    }
}

impl<'a> ops::Sub<Vector3> for &'a Vector3 {
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

impl<'a> ops::Sub<&'a Vector3> for Vector3 {
    type Output = Vector3;

    fn sub(self, other: &'a Vector3) -> Self::Output {
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

impl ops::Sub<f32> for Vector3 {
    type Output = Vector3;

    fn sub(self, other: f32) -> Self::Output {
        Vector3 {
            x: self.x - other,
            y: self.y - other,
            z: self.z - other,
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

impl<'a> ops::AddAssign<&'a Vector3> for Vector3 {
    fn add_assign(&mut self, other: &'a Vector3) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<'a> ops::AddAssign<Vector3> for &'a mut Vector3 {
    fn add_assign(&mut self, other: Vector3) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<'a, 'b> ops::AddAssign<&'a Vector3> for &'b mut Vector3 {
    fn add_assign(&mut self, other: &'a Vector3) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl ops::AddAssign<f32> for Vector3 {
    fn add_assign(&mut self, other: f32) {
        self.x += other;
        self.y += other;
        self.z += other;
    }
}

impl ops::SubAssign<Vector3> for Vector3 {
    fn sub_assign(&mut self, other: Vector3) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<'a> ops::SubAssign<&'a Vector3> for Vector3 {
    fn sub_assign(&mut self, other: &'a Vector3) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<'a> ops::SubAssign<Vector3> for &'a mut Vector3 {
    fn sub_assign(&mut self, other: Vector3) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<'a, 'b> ops::SubAssign<&'a Vector3> for &'b mut Vector3 {
    fn sub_assign(&mut self, other: &'a Vector3) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl ops::SubAssign<f32> for Vector3 {
    fn sub_assign(&mut self, other: f32) {
        self.x -= other;
        self.y -= other;
        self.z -= other;
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

impl<'a> ops::Mul<f32> for &'a Vector3 {
    type Output = Vector3;

    fn mul(self, other: f32) -> Vector3 {
        Vector3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
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

impl<'a> ops::Div<f32> for &'a Vector3 {
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

impl<'a> ops::DivAssign<f32> for &'a mut Vector3 {
    fn div_assign(&mut self, other: f32) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
    }
}

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

    pub fn zero() -> Vector4 {
        Vector4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }
    }
}

#[inline]
pub fn vec4<T: Into<Vector4>>(v: T) -> Vector4 {
    v.into()
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

impl<'a> From<(&'a Vector2, f32, f32)> for Vector4 {
    #[inline]
    fn from((v, z, w): (&'a Vector2, f32, f32)) -> Vector4 {
        Vector4::new(v.x, v.y, z, w)
    }
}

impl From<(Vector3, f32)> for Vector4 {
    #[inline]
    fn from((v, w): (Vector3, f32)) -> Vector4 {
        Vector4::new(v.x, v.y, v.z, w)
    }
}

impl<'a> From<(&'a Vector3, f32)> for Vector4 {
    #[inline]
    fn from((v, w): (&'a Vector3, f32)) -> Vector4 {
        Vector4::new(v.x, v.y, v.z, w)
    }
}

impl fmt::Debug for Vector4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "Vector4 "));
        <[f32; 4] as fmt::Debug>::fmt(self.as_ref(), f)
    }
}

impl fmt::Display for Vector4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:.2}, {:.2}, {:.2}, {:.2}]", self.x, self.y, self.z, self.w)
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

impl<'a> ops::Add<Vector4> for &'a Vector4 {
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

impl<'a> ops::Add<&'a Vector4> for Vector4 {
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

impl<'a, 'b> ops::Add<&'b Vector4> for &'a Vector4 {
    type Output = Vector4;

    fn add(self, other: &'b Vector4) -> Self::Output {
        Vector4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl ops::Add<f32> for Vector4 {
    type Output = Vector4;

    fn add(self, other: f32) -> Self::Output {
        Vector4 {
            x: self.x + other,
            y: self.y + other,
            z: self.z + other,
            w: self.w + other,
        }
    }
}

impl<'a> ops::Sub<Vector4> for &'a Vector4 {
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

impl<'a> ops::Sub<&'a Vector4> for Vector4 {
    type Output = Vector4;

    fn sub(self, other: &'a Vector4) -> Self::Output {
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

impl ops::Sub<f32> for Vector4 {
    type Output = Vector4;

    fn sub(self, other: f32) -> Self::Output {
        Vector4 {
            x: self.x - other,
            y: self.y - other,
            z: self.z - other,
            w: self.w - other,
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

impl<'a> ops::AddAssign<&'a Vector4> for Vector4 {
    fn add_assign(&mut self, other: &'a Vector4) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl<'a> ops::AddAssign<Vector4> for &'a mut Vector4 {
    fn add_assign(&mut self, other: Vector4) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl<'a, 'b> ops::AddAssign<&'a Vector4> for &'b mut Vector4 {
    fn add_assign(&mut self, other: &'a Vector4) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl ops::AddAssign<f32> for Vector4 {
    fn add_assign(&mut self, other: f32) {
        self.x += other;
        self.y += other;
        self.z += other;
        self.w += other;
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

impl<'a> ops::SubAssign<&'a Vector4> for Vector4 {
    fn sub_assign(&mut self, other: &'a Vector4) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl<'a> ops::SubAssign<Vector4> for &'a mut Vector4 {
    fn sub_assign(&mut self, other: Vector4) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl<'a, 'b> ops::SubAssign<&'a Vector4> for &'b mut Vector4 {
    fn sub_assign(&mut self, other: &'a Vector4) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl ops::SubAssign<f32> for Vector4 {
    fn sub_assign(&mut self, other: f32) {
        self.x -= other;
        self.y -= other;
        self.z -= other;
        self.w -= other;
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

impl<'a> ops::Mul<f32> for &'a Vector4 {
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

impl<'a> ops::Div<f32> for &'a Vector4 {
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

impl<'a> ops::DivAssign<f32> for &'a mut Vector4 {
    fn div_assign(&mut self, other: f32) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
        self.w /= other;
    }
}


///
/// The `Matrix3` type represents 3x3 matrices in column-major order.
///
#[derive(Copy, Clone, Debug)]
pub struct Matrix3 {
    m: [f32; 9],
}

impl Matrix3 {
    pub fn new(
        m11: f32, m12: f32, m13: f32, 
        m21: f32, m22: f32, m23: f32, 
        m31: f32, m32: f32, m33: f32) -> Matrix3 {

        Matrix3 {
            m: [
                m11, m12, m13, // Column 1
                m21, m22, m23, // Column 2
                m31, m32, m33  // Column 3
            ]
        }
    }

    pub fn zero() -> Matrix3 {
        Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    pub fn identity() -> Matrix3 {
        Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    }

    pub fn transpose(&self) -> Matrix3 {
        Matrix3::new(
            self.m[0], self.m[3], self.m[7],  
            self.m[1], self.m[4], self.m[8],  
            self.m[2], self.m[6], self.m[9]
        )
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.m.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.m.as_mut_ptr()
    }
}

impl fmt::Display for Matrix3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, 
            "\n[{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}]", 
            self.m[0], self.m[3], self.m[6],
            self.m[1], self.m[4], self.m[7],
            self.m[2], self.m[5], self.m[8],
        )
    }
}

#[inline]
fn mat3(m11: f32, m12: f32, m13: f32, 
        m21: f32, m22: f32, m23: f32, 
        m31: f32, m32: f32, m33: f32) -> Matrix3 {

    Matrix3::new(m11, m12, m13, m21, m22, m23, m31, m32, m33)
}

impl AsRef<[f32; 9]> for Matrix3 {
    fn as_ref(&self) -> &[f32; 9] {
        &self.m
    }
}

impl AsMut<[f32; 9]> for Matrix3 {
    fn as_mut(&mut self) -> &mut [f32; 9] {
        &mut self.m
    }
}


///
/// The `Matrix4` type represents 4x4 matrices in column-major order.
///
#[derive(Copy, Clone, Debug)]
pub struct Matrix4 {
    pub m: [f32; 16],
}

impl Matrix4 {
    pub fn new(
        m11: f32, m12: f32, m13: f32, m14: f32,
        m21: f32, m22: f32, m23: f32, m24: f32,
        m31: f32, m32: f32, m33: f32, m34: f32,
        m41: f32, m42: f32, m43: f32, m44: f32) -> Matrix4 {

        Matrix4 {
            m: [
                m11, m12, m13, m14, // Column 1
                m21, m22, m23, m24, // Column 2
                m31, m32, m33, m34, // Column 3
                m41, m42, m43, m44  // Column 4
            ]
        }
    }

    pub fn zero() -> Matrix4 {
        Matrix4::new(
            0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0
        )
    }

    pub fn identity() -> Matrix4 {
        Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0
        )
    }

    pub fn transpose(&self) -> Matrix4 {
        Matrix4::new(
            self.m[0], self.m[4], self.m[8],  self.m[12],
            self.m[1], self.m[5], self.m[9],  self.m[13], 
            self.m[2], self.m[6], self.m[10], self.m[14], 
            self.m[3], self.m[7], self.m[11], self.m[15]
        )
    }

    pub fn translate(&self, v: &Vector3) -> Matrix4 {
        let mut m_t = Matrix4::identity();
        m_t.m[12] = v.x;
        m_t.m[13] = v.y;
        m_t.m[14] = v.z;

        m_t * self
    }

    // Rotate around x axis by an angle in degrees.
    pub fn rotate_x_deg(&self, deg: f32) -> Matrix4 {
        // Convert to radians.
        let rad = deg * ONE_DEG_IN_RAD;
        let mut m_r = Matrix4::identity();
        m_r.m[5]  =  f32::cos(rad);
        m_r.m[9]  = -f32::sin(rad);
        m_r.m[6]  =  f32::sin(rad);
        m_r.m[10] =  f32::cos(rad);
    
        m_r * self
    }

    // Rotate around y axis by an angle in degrees.
    pub fn rotate_y_deg(&self, deg: f32) -> Matrix4 {
        // Convert to radians.
        let rad = deg * ONE_DEG_IN_RAD;
        let mut m_r = Matrix4::identity();
        m_r.m[0]  =  f32::cos(rad);
        m_r.m[8]  =  f32::sin(rad);
        m_r.m[2]  = -f32::sin(rad);
        m_r.m[10] =  f32::cos(rad);
    
        m_r * self
    }

    // Rotate around z axis by an angle in degrees.
    pub fn rotate_z_deg(&self, deg: f32) -> Matrix4 {
        // Convert to radians.
        let rad = deg * ONE_DEG_IN_RAD;
        let mut m_r = Matrix4::identity();
        m_r.m[0] =  f32::cos(rad);
        m_r.m[4] = -f32::sin(rad);
        m_r.m[1] =  f32::sin(rad);
        m_r.m[5] =  f32::cos(rad);
    
        m_r * self
    }

    // scale a matrix by [x, y, z]
    pub fn scale(&self, v: &Vector3) -> Matrix4 {
        let mut m_s = Matrix4::identity();
        m_s.m[0]  = v.x;
        m_s.m[5]  = v.y;
        m_s.m[10] = v.z;
    
        m_s * self
    }

    /// returns a scalar value with the determinant for a 4x4 matrix
    /// see
    /// http://www.euclideanspace.com/maths/algebra/matrix/functions/determinant/fourD/index.htm
    pub fn determinant(&self) -> f32 {
        self.m[12] * self.m[9]  * self.m[6]  * self.m[3]  -
        self.m[8]  * self.m[13] * self.m[6]  * self.m[3]  -
        self.m[12] * self.m[5]  * self.m[10] * self.m[3]  +
        self.m[4]  * self.m[13] * self.m[10] * self.m[3]  +
        self.m[8]  * self.m[5]  * self.m[14] * self.m[3]  -
        self.m[4]  * self.m[9]  * self.m[14] * self.m[3]  -
        self.m[12] * self.m[9]  * self.m[2]  * self.m[7]  +
        self.m[8]  * self.m[13] * self.m[2]  * self.m[7]  +
        self.m[12] * self.m[1]  * self.m[10] * self.m[7]  -
        self.m[0]  * self.m[13] * self.m[10] * self.m[7]  -
        self.m[8]  * self.m[1]  * self.m[14] * self.m[7]  +
        self.m[0]  * self.m[9]  * self.m[14] * self.m[7]  +
        self.m[12] * self.m[5]  * self.m[2]  * self.m[11] -
        self.m[4]  * self.m[13] * self.m[2]  * self.m[11] -
        self.m[12] * self.m[1]  * self.m[6]  * self.m[11] +
        self.m[0]  * self.m[13] * self.m[6]  * self.m[11] +
        self.m[4]  * self.m[1]  * self.m[14] * self.m[11] -
        self.m[0]  * self.m[5]  * self.m[14] * self.m[11] -
        self.m[8]  * self.m[5]  * self.m[2]  * self.m[15] +
        self.m[4]  * self.m[9]  * self.m[2]  * self.m[15] +
        self.m[8]  * self.m[1]  * self.m[6]  * self.m[15] -
        self.m[0]  * self.m[9]  * self.m[6]  * self.m[15] -
        self.m[4]  * self.m[1]  * self.m[10] * self.m[15] +
        self.m[0]  * self.m[5]  * self.m[10] * self.m[15]
    }

    pub fn is_invertible(&self) -> bool {
        self.determinant() != 0.0
    }

    pub fn inverse(&self) -> Matrix4 {
        let det = self.determinant();
        
        // A matrix with zero determinant has no inverse.
        if det == 0.0 {
            eprintln!("WARNING. Matrix has zero determinant. It cannot be inverted.");
            
            return *self;
        }

        let inv_det = 1.0 / det;

        return mat4(
            inv_det * ( self.m[9] * self.m[14] * self.m[7] - self.m[13] * self.m[10] * self.m[7] +
                                    self.m[13] * self.m[6] * self.m[11] - self.m[5] * self.m[14] * self.m[11] -
                                    self.m[9] * self.m[6] * self.m[15] + self.m[5] * self.m[10] * self.m[15] ),
            inv_det * ( self.m[13] * self.m[10] * self.m[3] - self.m[9] * self.m[14] * self.m[3] -
                                    self.m[13] * self.m[2] * self.m[11] + self.m[1] * self.m[14] * self.m[11] +
                                    self.m[9] * self.m[2] * self.m[15] - self.m[1] * self.m[10] * self.m[15] ),
            inv_det * ( self.m[5] * self.m[14] * self.m[3] - self.m[13] * self.m[6] * self.m[3] +
                                    self.m[13] * self.m[2] * self.m[7] - self.m[1] * self.m[14] * self.m[7] -
                                    self.m[5] * self.m[2] * self.m[15] + self.m[1] * self.m[6] * self.m[15] ),
            inv_det * ( self.m[9] * self.m[6] * self.m[3] - self.m[5] * self.m[10] * self.m[3] -
                                    self.m[9] * self.m[2] * self.m[7] + self.m[1] * self.m[10] * self.m[7] +
                                    self.m[5] * self.m[2] * self.m[11] - self.m[1] * self.m[6] * self.m[11] ),
            inv_det * ( self.m[12] * self.m[10] * self.m[7] - self.m[8] * self.m[14] * self.m[7] -
                                    self.m[12] * self.m[6] * self.m[11] + self.m[4] * self.m[14] * self.m[11] +
                                    self.m[8] * self.m[6] * self.m[15] - self.m[4] * self.m[10] * self.m[15] ),
            inv_det * ( self.m[8] * self.m[14] * self.m[3] - self.m[12] * self.m[10] * self.m[3] +
                                    self.m[12] * self.m[2] * self.m[11] - self.m[0] * self.m[14] * self.m[11] -
                                    self.m[8] * self.m[2] * self.m[15] + self.m[0] * self.m[10] * self.m[15] ),
            inv_det * ( self.m[12] * self.m[6] * self.m[3] - self.m[4] * self.m[14] * self.m[3] -
                                    self.m[12] * self.m[2] * self.m[7] + self.m[0] * self.m[14] * self.m[7] +
                                    self.m[4] * self.m[2] * self.m[15] - self.m[0] * self.m[6] * self.m[15] ),
            inv_det * ( self.m[4] * self.m[10] * self.m[3] - self.m[8] * self.m[6] * self.m[3] +
                                    self.m[8] * self.m[2] * self.m[7] - self.m[0] * self.m[10] * self.m[7] -
                                    self.m[4] * self.m[2] * self.m[11] + self.m[0] * self.m[6] * self.m[11] ),
            inv_det * ( self.m[8] * self.m[13] * self.m[7] - self.m[12] * self.m[9] * self.m[7] +
                                    self.m[12] * self.m[5] * self.m[11] - self.m[4] * self.m[13] * self.m[11] -
                                    self.m[8] * self.m[5] * self.m[15] + self.m[4] * self.m[9] * self.m[15] ),
            inv_det * ( self.m[12] * self.m[9] * self.m[3] - self.m[8] * self.m[13] * self.m[3] -
                                    self.m[12] * self.m[1] * self.m[11] + self.m[0] * self.m[13] * self.m[11] +
                                    self.m[8] * self.m[1] * self.m[15] - self.m[0] * self.m[9] * self.m[15] ),
            inv_det * ( self.m[4] * self.m[13] * self.m[3] - self.m[12] * self.m[5] * self.m[3] +
                                    self.m[12] * self.m[1] * self.m[7] - self.m[0] * self.m[13] * self.m[7] -
                                    self.m[4] * self.m[1] * self.m[15] + self.m[0] * self.m[5] * self.m[15] ),
            inv_det * ( self.m[8] * self.m[5] * self.m[3] - self.m[4] * self.m[9] * self.m[3] -
                                    self.m[8] * self.m[1] * self.m[7] + self.m[0] * self.m[9] * self.m[7] +
                                    self.m[4] * self.m[1] * self.m[11] - self.m[0] * self.m[5] * self.m[11] ),
            inv_det * ( self.m[12] * self.m[9] * self.m[6] - self.m[8] * self.m[13] * self.m[6] -
                                    self.m[12] * self.m[5] * self.m[10] + self.m[4] * self.m[13] * self.m[10] +
                                    self.m[8] * self.m[5] * self.m[14] - self.m[4] * self.m[9] * self.m[14] ),
            inv_det * ( self.m[8] * self.m[13] * self.m[2] - self.m[12] * self.m[9] * self.m[2] +
                                    self.m[12] * self.m[1] * self.m[10] - self.m[0] * self.m[13] * self.m[10] -
                                    self.m[8] * self.m[1] * self.m[14] + self.m[0] * self.m[9] * self.m[14] ),
            inv_det * ( self.m[12] * self.m[5] * self.m[2] - self.m[4] * self.m[13] * self.m[2] -
                                    self.m[12] * self.m[1] * self.m[6] + self.m[0] * self.m[13] * self.m[6] +
                                    self.m[4] * self.m[1] * self.m[14] - self.m[0] * self.m[5] * self.m[14] ),
            inv_det * ( self.m[4] * self.m[9] * self.m[2] - self.m[8] * self.m[5] * self.m[2] +
                                    self.m[8] * self.m[1] * self.m[6] - self.m[0] * self.m[9] * self.m[6] -
                                    self.m[4] * self.m[1] * self.m[10] + self.m[0] * self.m[5] * self.m[10] ) );
    }

    ///
    /// Compute the perspective matrix for converting from camera space to 
    /// normalized device coordinates.
    ///
    pub fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Matrix4 {
        let fov_rad = fovy * ONE_DEG_IN_RAD;
        let range = f32::tan(fov_rad * 0.5) * near;
        let sx = (2.0 * near) / (range * aspect + range * aspect);
        let sy = near / range;
        let sz = -(far + near) / (far - near);
        let pz = -(2.0 * far * near) / (far - near);
        // Make sure the bottom right corner is zero.
        let mut m = Matrix4::zero();
        m.m[0] = sx;
        m.m[5] = sy;
        m.m[10] = sz;
        m.m[14] = pz;
        m.m[11] = -1.0;
        
        m
    }

    /// 
    /// Generate a pointer to the underlying array for passing a
    /// matrix to the graphics hardware.
    ///
    pub fn as_ptr(&self) -> *const f32 {
        self.m.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.m.as_mut_ptr()
    }
}

impl fmt::Display for Matrix4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, 
            "\n[{:.2}][{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}][{:.2}]", 
            self.m[0], self.m[4], self.m[8],  self.m[12],
            self.m[1], self.m[5], self.m[9],  self.m[13],
            self.m[2], self.m[6], self.m[10], self.m[14],
            self.m[3], self.m[7], self.m[11], self.m[15]
        )
    }
}

pub fn mat4(
        m11: f32, m12: f32, m13: f32, m14: f32, 
        m21: f32, m22: f32, m23: f32, m24: f32,
        m31: f32, m32: f32, m33: f32, m34: f32,
        m41: f32, m42: f32, m43: f32, m44: f32) -> Matrix4 {

    Matrix4::new(
        m11, m12, m13, m14, 
        m21, m22, m23, m24, 
        m31, m32, m33, m34, 
        m41, m42, m43, m44
    )
}

impl AsRef<[f32; 16]> for Matrix4 {
    fn as_ref(&self) -> &[f32; 16] {
        &self.m
    }
}

impl AsMut<[f32; 16]> for Matrix4 {
    fn as_mut(&mut self) -> &mut [f32; 16] {
        &mut self.m
    }
}

impl ops::Mul<Vector4> for Matrix4 {
    type Output = Vector4;

    fn mul(self, other: Vector4) -> Self::Output {
        let x = self.m[0] * other[0] + self.m[4] * other[1] + self.m[8]  * other[2] + self.m[12] * other[3];
        let y = self.m[1] * other[0] + self.m[5] * other[1] + self.m[9]  * other[2] + self.m[13] * other[3];
        let z = self.m[2] * other[0] + self.m[6] * other[1] + self.m[10] * other[2] + self.m[14] * other[3];
        let w = self.m[3] * other[0] + self.m[7] * other[1] + self.m[11] * other[2] + self.m[15] * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl<'a> ops::Mul<&'a Matrix4> for Matrix4 {
    type Output = Matrix4;

    fn mul(self, other: &'a Matrix4) -> Matrix4 {
        let mut mm = Matrix4::zero();

        mm.m[0]  = self.m[0] * other.m[0]  + self.m[4] * other.m[1]  + self.m[8]  * other.m[2]  + self.m[12] * other.m[3];
        mm.m[1]  = self.m[1] * other.m[0]  + self.m[5] * other.m[1]  + self.m[9]  * other.m[2]  + self.m[13] * other.m[3];
        mm.m[2]  = self.m[2] * other.m[0]  + self.m[6] * other.m[1]  + self.m[10] * other.m[2]  + self.m[14] * other.m[3];
        mm.m[3]  = self.m[3] * other.m[0]  + self.m[7] * other.m[1]  + self.m[11] * other.m[2]  + self.m[15] * other.m[3];
        mm.m[4]  = self.m[0] * other.m[4]  + self.m[4] * other.m[5]  + self.m[8]  * other.m[6]  + self.m[12] * other.m[7];
        mm.m[5]  = self.m[1] * other.m[4]  + self.m[5] * other.m[5]  + self.m[9]  * other.m[6]  + self.m[13] * other.m[7];
        mm.m[6]  = self.m[2] * other.m[4]  + self.m[6] * other.m[5]  + self.m[10] * other.m[6]  + self.m[14] * other.m[7];
        mm.m[7]  = self.m[3] * other.m[4]  + self.m[7] * other.m[5]  + self.m[11] * other.m[6]  + self.m[15] * other.m[7];
        mm.m[8]  = self.m[0] * other.m[8]  + self.m[4] * other.m[9]  + self.m[8]  * other.m[10] + self.m[12] * other.m[11];
        mm.m[9]  = self.m[1] * other.m[8]  + self.m[5] * other.m[9]  + self.m[9]  * other.m[10] + self.m[13] * other.m[11];
        mm.m[10] = self.m[2] * other.m[8]  + self.m[6] * other.m[9]  + self.m[10] * other.m[10] + self.m[14] * other.m[11];
        mm.m[11] = self.m[3] * other.m[8]  + self.m[7] * other.m[9]  + self.m[11] * other.m[10] + self.m[15] * other.m[11];
        mm.m[12] = self.m[0] * other.m[12] + self.m[4] * other.m[13] + self.m[8]  * other.m[14] + self.m[12] * other.m[15];
        mm.m[13] = self.m[1] * other.m[12] + self.m[5] * other.m[13] + self.m[9]  * other.m[14] + self.m[13] * other.m[15];
        mm.m[14] = self.m[2] * other.m[12] + self.m[6] * other.m[13] + self.m[10] * other.m[14] + self.m[14] * other.m[15];
        mm.m[15] = self.m[3] * other.m[12] + self.m[7] * other.m[13] + self.m[11] * other.m[14] + self.m[15] * other.m[15];

        mm
    }
}

impl<'a, 'b> ops::Mul<&'a Matrix4> for &'b Matrix4 {
    type Output = Matrix4;

    fn mul(self, other: &'a Matrix4) -> Matrix4 {
        let mut mm = Matrix4::zero();

        mm.m[0]  = self.m[0]*other.m[0]  + self.m[4]*other.m[1]  + self.m[8]*other.m[2]   + self.m[12]*other.m[3];
        mm.m[1]  = self.m[1]*other.m[0]  + self.m[5]*other.m[1]  + self.m[9]*other.m[2]   + self.m[13]*other.m[3];
        mm.m[2]  = self.m[2]*other.m[0]  + self.m[6]*other.m[1]  + self.m[10]*other.m[2]  + self.m[14]*other.m[3];
        mm.m[3]  = self.m[3]*other.m[0]  + self.m[7]*other.m[1]  + self.m[11]*other.m[2]  + self.m[15]*other.m[3];
        mm.m[4]  = self.m[0]*other.m[4]  + self.m[4]*other.m[5]  + self.m[8]*other.m[6]   + self.m[12]*other.m[7];
        mm.m[5]  = self.m[1]*other.m[4]  + self.m[5]*other.m[5]  + self.m[9]*other.m[6]   + self.m[13]*other.m[7];
        mm.m[6]  = self.m[2]*other.m[4]  + self.m[6]*other.m[5]  + self.m[10]*other.m[6]  + self.m[14]*other.m[7];
        mm.m[7]  = self.m[3]*other.m[4]  + self.m[7]*other.m[5]  + self.m[11]*other.m[6]  + self.m[15]*other.m[7];
        mm.m[8]  = self.m[0]*other.m[8]  + self.m[4]*other.m[9]  + self.m[8]*other.m[10]  + self.m[12]*other.m[11];
        mm.m[9]  = self.m[1]*other.m[8]  + self.m[5]*other.m[9]  + self.m[9]*other.m[10]  + self.m[13]*other.m[11];
        mm.m[10] = self.m[2]*other.m[8]  + self.m[6]*other.m[9]  + self.m[10]*other.m[10] + self.m[14]*other.m[11];
        mm.m[11] = self.m[3]*other.m[8]  + self.m[7]*other.m[9]  + self.m[11]*other.m[10] + self.m[15]*other.m[11];
        mm.m[12] = self.m[0]*other.m[12] + self.m[4]*other.m[13] + self.m[8]*other.m[14]  + self.m[12]*other.m[15];
        mm.m[13] = self.m[1]*other.m[12] + self.m[5]*other.m[13] + self.m[9]*other.m[14]  + self.m[13]*other.m[15];
        mm.m[14] = self.m[2]*other.m[12] + self.m[6]*other.m[13] + self.m[10]*other.m[14] + self.m[14]*other.m[15];
        mm.m[15] = self.m[3]*other.m[12] + self.m[7]*other.m[13] + self.m[11]*other.m[14] + self.m[15]*other.m[15];

        mm
    }
}

impl ops::Mul<Matrix4> for Matrix4 {
    type Output = Matrix4;

    fn mul(self, other: Matrix4) -> Matrix4 {
        let mut mm = Matrix4::zero();

        mm.m[0]  = self.m[0]*other.m[0]  + self.m[4]*other.m[1]  + self.m[8]*other.m[2]   + self.m[12]*other.m[3];
        mm.m[1]  = self.m[1]*other.m[0]  + self.m[5]*other.m[1]  + self.m[9]*other.m[2]   + self.m[13]*other.m[3];
        mm.m[2]  = self.m[2]*other.m[0]  + self.m[6]*other.m[1]  + self.m[10]*other.m[2]  + self.m[14]*other.m[3];
        mm.m[3]  = self.m[3]*other.m[0]  + self.m[7]*other.m[1]  + self.m[11]*other.m[2]  + self.m[15]*other.m[3];
        mm.m[4]  = self.m[0]*other.m[4]  + self.m[4]*other.m[5]  + self.m[8]*other.m[6]   + self.m[12]*other.m[7];
        mm.m[5]  = self.m[1]*other.m[4]  + self.m[5]*other.m[5]  + self.m[9]*other.m[6]   + self.m[13]*other.m[7];
        mm.m[6]  = self.m[2]*other.m[4]  + self.m[6]*other.m[5]  + self.m[10]*other.m[6]  + self.m[14]*other.m[7];
        mm.m[7]  = self.m[3]*other.m[4]  + self.m[7]*other.m[5]  + self.m[11]*other.m[6]  + self.m[15]*other.m[7];
        mm.m[8]  = self.m[0]*other.m[8]  + self.m[4]*other.m[9]  + self.m[8]*other.m[10]  + self.m[12]*other.m[11];
        mm.m[9]  = self.m[1]*other.m[8]  + self.m[5]*other.m[9]  + self.m[9]*other.m[10]  + self.m[13]*other.m[11];
        mm.m[10] = self.m[2]*other.m[8]  + self.m[6]*other.m[9]  + self.m[10]*other.m[10] + self.m[14]*other.m[11];
        mm.m[11] = self.m[3]*other.m[8]  + self.m[7]*other.m[9]  + self.m[11]*other.m[10] + self.m[15]*other.m[11];
        mm.m[12] = self.m[0]*other.m[12] + self.m[4]*other.m[13] + self.m[8]*other.m[14]  + self.m[12]*other.m[15];
        mm.m[13] = self.m[1]*other.m[12] + self.m[5]*other.m[13] + self.m[9]*other.m[14]  + self.m[13]*other.m[15];
        mm.m[14] = self.m[2]*other.m[12] + self.m[6]*other.m[13] + self.m[10]*other.m[14] + self.m[14]*other.m[15];
        mm.m[15] = self.m[3]*other.m[12] + self.m[7]*other.m[13] + self.m[11]*other.m[14] + self.m[15]*other.m[15];

        mm
    }
}

impl cmp::PartialEq for Matrix4 {
    fn eq(&self, other: &Matrix4) -> bool {
        for i in 0..self.m.len() {
            if f32::abs(self.m[i] - other.m[i]) > EPSILON {
                return false;
            }
        }

        true
    }
}


