use std::fmt;
use std::mem;
use std::ops;

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
    pub fn new(x: f32, y: f32) -> Vector2 {
        Vector2 { x: x, y: y }
    }

    pub fn zero() -> Vector2 { 
        Vector2 { x: 0.0, y: 0.0 }
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
