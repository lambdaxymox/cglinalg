use traits::Array;
use std::fmt;
use std::mem;
use std::ops;
use std::cmp;


///
/// A representation of one-dimensional vectors, with a
/// Euclidean metric.
///
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

    ///
    /// Generate a zero vector.
    ///
    pub fn zero() -> Vector1 {
        Vector1 { x: 0.0 }
    }

    #[inline]
    pub fn unit_x() -> Vector1 {
        Vector1 { x: 1.0 }
    }

    ///
    /// Compute the norm (length) of a vector.
    ///
    pub fn norm(&self) -> f32 {
        f32::sqrt(self.x * self.x)
    }

    ///
    /// Compute the squared norm (length) of a vector.
    ///
    pub fn norm2(&self) -> f32 {
        self.x * self.x
    }

    ///
    /// Convert an arbitrary vector into a unit vector.
    ///
    pub fn normalize(&self) -> Vector1 {
        let norm_v = self.norm();
        if norm_v == 0.0 {
            return Vector1::zero();
        }

        Vector1::new(self.x / norm_v)
    }

    ///
    /// Compute the dot product of two vectors.
    ///
    pub fn dot(&self, other: &Vector1) -> f32 {
        self.x * other.x
    }

    ///
    /// Compute the squared distance between two vectors.
    ///
    #[inline]
    pub fn distance2(&self, to: &Vector1) -> f32 {
        let x = (to.x - self.x) * (to.x - self.x);

        x
    }

    ///
    /// Compute the Euclidean distance between two vectors.
    ///
    #[inline]
    pub fn distance(&self, to: &Vector1) -> f32 {
        f32::sqrt(self.distance2(to))
    }

    ///
    /// Compute the projection for a vector onto another vector.
    ///
    #[inline]
    pub fn project(&self, onto: Vector1) -> Vector1 {
        let onto_norm2 = onto.norm2();
        let x = self.dot(&onto) / onto_norm2;

        Vector1 { x: x }
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
        try!(write!(f, "Vector1 "));
        <[f32; 1] as fmt::Debug>::fmt(self.as_ref(), f)
    }
}

impl fmt::Display for Vector1 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:.2}]", self.x)
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

impl<'a> From<&'a [f32; 1]> for Vector1 {
    #[inline]
    fn from(v: &'a [f32; 1]) -> Vector1 {
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

impl<'a> ops::Neg for &'a Vector1 {
    type Output = Vector1;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector1 { x: -self.x }
    }
}

impl<'a> ops::Add<Vector1> for &'a Vector1 {
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

impl<'a> ops::Add<&'a Vector1> for Vector1 {
    type Output = Vector1;

    fn add(self, other: &'a Vector1) -> Self::Output {
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

impl<'a> ops::Sub<Vector1> for &'a Vector1 {
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

impl<'a> ops::Sub<&'a Vector1> for Vector1 {
    type Output = Vector1;

    fn sub(self, other: &'a Vector1) -> Self::Output {
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

impl<'a> ops::AddAssign<&'a Vector1> for Vector1 {
    fn add_assign(&mut self, other: &'a Vector1) {
        self.x = self.x + other.x;
    }
}

impl<'a> ops::AddAssign<Vector1> for &'a mut Vector1 {
    fn add_assign(&mut self, other: Vector1) {
        self.x = self.x + other.x;
    }
}

impl<'a, 'b> ops::AddAssign<&'a Vector1> for &'b mut Vector1 {
    fn add_assign(&mut self, other: &'a Vector1) {
        self.x = self.x + other.x;
    }
}

impl ops::SubAssign<Vector1> for Vector1 {
    fn sub_assign(&mut self, other: Vector1) {
        self.x = self.x - other.x;
    }
}

impl<'a> ops::SubAssign<&'a Vector1> for Vector1 {
    fn sub_assign(&mut self, other: &'a Vector1) {
        self.x = self.x - other.x;
    }
}

impl<'a> ops::SubAssign<Vector1> for &'a mut Vector1 {
    fn sub_assign(&mut self, other: Vector1) {
        self.x = self.x - other.x;
    }
}

impl<'a, 'b> ops::SubAssign<&'a Vector1> for &'b mut Vector1 {
    fn sub_assign(&mut self, other: &'a Vector1) {
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

impl<'a> ops::Mul<f32> for &'a Vector1 {
    type Output = Vector1;

    fn mul(self, other: f32) -> Vector1 {
        Vector1 {
            x: self.x * other,
        }
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

impl<'a> ops::Div<f32> for &'a Vector1 {
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

impl<'a> ops::DivAssign<f32> for &'a mut Vector1 {
    fn div_assign(&mut self, other: f32) {
        self.x = self.x / other;
    }
}


#[cfg(test)]
mod tests {
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
                    c: 802.3435169,
                    v1: Vector1::from(-23.43),
                    v2: Vector1::from(426.1),
                },
                TestCase {
                    c: 33.249539,
                    v1: Vector1::from(27.6189),
                    v2: Vector1::from(258.083)
                },
                TestCase {
                    c: 7.04217,
                    v1: Vector1::from(0.0),
                    v2: Vector1::from(0.0),
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
