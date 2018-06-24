use std::fmt;
use std::mem;
use std::ops;


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
