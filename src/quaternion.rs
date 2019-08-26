use std::fmt;
use std::mem;
use std::ops;
use std::cmp;

use crate::vector::Vector3;
use crate::matrix::{Matrix3, Matrix4};
use crate::traits::{Array, One, Zero, Metric, DotProduct, Lerp, Magnitude};


const EPSILON: f32 = 0.00001;
const M_PI: f32 = 3.14159265358979323846264338327950288;
const ONE_DEG_IN_RAD: f32 = (2.0 * M_PI) / 360.0; // == 0.017444444


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Quaternion {
    /// The scalar component.
    s: f32,
    /// The vector component.
    v: Vector3,
}

impl Quaternion {
    #[inline]
    pub fn new(s: f32, x: f32, y: f32, z: f32) -> Quaternion {
        Self::from_sv(s, Vector3::new(x, y, z))
    }

    /// Compute a quaternion from it's scalar and vector parts.
    #[inline]
    pub fn from_sv(s: f32, v: Vector3) -> Quaternion {
        Quaternion { s: s, v: v }
    }

    /// Compute a quaternion corresponding to rotating about an axis in radians.
    pub fn from_axis_rad(radians: f32, axis: Vector3) -> Quaternion {
        Quaternion::new(
            f32::cos(radians / 2.0),
            f32::sin(radians / 2.0) * axis.x,
            f32::sin(radians / 2.0) * axis.y,
            f32::sin(radians / 2.0) * axis.z,
        )
    }

    /// Computer a quaternion corresponding to rotating about an axis in degrees.
    pub fn from_axis_deg(degrees: f32, axis: Vector3) -> Quaternion {
        Self::from_axis_rad(ONE_DEG_IN_RAD * degrees, axis)
    }

    /// Compute the conjugate of a quaternion.
    pub fn conjugate(&self) -> Quaternion {
        Quaternion::from_sv(self.s, -self.v)
    }

    /// Convert a quaternion to its equivalent matrix form using .
    pub fn to_mut_mat4(&self, m: &mut Matrix4) {
        let s = self.s;
        let x = self.v.x;
        let y = self.v.y;
        let z = self.v.z;
        m.c0r0 = 1.0 - 2.0 * y * y - 2.0 * z * z;
        m.c0r1 = 2.0 * x * y + 2.0 * s * z;
        m.c0r2 = 2.0 * x * z - 2.0 * s * y;
        m.c0r3 = 0.0;
        m.c1r0 = 2.0 * x * y - 2.0 * s * z;
        m.c1r1 = 1.0 - 2.0 * x * x - 2.0 * z * z;
        m.c1r2 = 2.0 * y * z + 2.0 * s * x;
        m.c1r3 = 0.0;
        m.c2r0 = 2.0 * x * z + 2.0 * s * y;
        m.c2r1 = 2.0 * y * z - 2.0 * s * x;
        m.c2r2 = 1.0 - 2.0 * x * x - 2.0 * y * y;
        m.c2r3 = 0.0;
        m.c3r0 = 0.0;
        m.c3r1 = 0.0;
        m.c3r2 = 0.0;
        m.c3r3 = 1.0;
    }

    pub fn slerp(q: &mut Quaternion, r: &Quaternion, t: f32) -> Quaternion {
        // angle between q0-q1
        let mut cos_half_theta = q.dot(r);
        // as found here
        // http://stackoverflow.com/questions/2886606/flipping-issue-when-interpolating-rotations-using-quaternions
        // if dot product is negative then one quaternion should be negated, to make
        // it take the short way around, rather than the long way
        // yeah! and furthermore Susan, I had to recalculate the d.p. after this
        if cos_half_theta < 0.0 {
            q.s *= -1.0;
            q.v.x *= -1.0;
            q.v.y *= -1.0;
            q.v.z *= -1.0;

            cos_half_theta = q.dot(r);
        }
        // if qa=qb or qa=-qb then theta = 0 and we can return qa
        if f32::abs(cos_half_theta) >= 1.0 {
            return *q;
        }

        // Calculate temporary values
        let sin_half_theta = f32::sqrt(1.0 - cos_half_theta * cos_half_theta);
        // if theta = 180 degrees then result is not fully defined
        // we could rotate around any axis normal to qa or qb
        let mut result = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        if f32::abs(sin_half_theta) < 0.001 {
            result.s   = (1.0 - t) * q.s   + t * r.s;
            result.v.x = (1.0 - t) * q.v.x + t * r.v.x;
            result.v.y = (1.0 - t) * q.v.y + t * r.v.y;
            result.v.z = (1.0 - t) * q.v.z + t * r.v.z;

            return result;
        }
        let half_theta = f32::acos(cos_half_theta);
        let a = f32::sin((1.0 - t) * half_theta) / sin_half_theta;
        let b = f32::sin(t * half_theta) / sin_half_theta;
        
        result.s   = q.s   * a + r.s   * b;
        result.v.x = q.v.x * a + r.v.x * b;
        result.v.y = q.v.y * a + r.v.y * b;
        result.v.z = q.v.z * a + r.v.z * b;

        result
    }
}

impl Zero for Quaternion {
    fn zero() -> Quaternion {
        Quaternion::new(0.0, 0.0, 0.0, 0.0)
    }

    fn is_zero(&self) -> bool {
        self.s == 0.0 && self.v.x == 0.0 && self.v.y == 0.0 && self.v.z == 0.0
    }
}

impl One for Quaternion {
    fn one() -> Quaternion {
        Quaternion::new(1.0, 0.0, 0.0, 0.0)
    }
}

impl AsRef<[f32; 4]> for Quaternion {
    fn as_ref(&self) -> &[f32; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl AsRef<(f32, f32, f32, f32)> for Quaternion {
    fn as_ref(&self) -> &(f32, f32, f32, f32) {
        unsafe { mem::transmute(self) }
    }
}

impl From<Quaternion> for Matrix3 {
    fn from(quat: Quaternion) -> Matrix3 {
        let s = quat.s;
        let x = quat.v.x;
        let y = quat.v.y;
        let z = quat.v.z;
    
        Matrix3::new(
            1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y + 2.0 * s * z,       2.0 * x * z - 2.0 * s * y,
            2.0 * x * y - 2.0 * s * z,       1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z + 2.0 * s * x,
            2.0 * x * z + 2.0 * s * y,       2.0 * y * z - 2.0 * s * x,       1.0 - 2.0 * x * x - 2.0 * y * y,
        )
    }
}

impl From<Quaternion> for Matrix4 {
    fn from(quat: Quaternion) -> Matrix4 {
        let s = quat.s;
        let x = quat.v.x;
        let y = quat.v.y;
        let z = quat.v.z;
    
        Matrix4::new(
            1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y + 2.0 * s * z,       2.0 * x * z - 2.0 * s * y,       0.0, 
            2.0 * x * y - 2.0 * s * z,       1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z + 2.0 * s * x,       0.0, 
            2.0 * x * z + 2.0 * s * y,       2.0 * y * z - 2.0 * s * x,       1.0 - 2.0 * x * x - 2.0 * y * y, 0.0, 
            0.0,                             0.0,                             0.0,                             1.0
        )
    }
}

impl From<[f32; 4]> for Quaternion {
    #[inline]
    fn from(v: [f32; 4]) -> Quaternion {
        Quaternion::new(v[0], v[1], v[2], v[3])
    }
}

impl<'a> From<&'a [f32; 4]> for &'a Quaternion {
    #[inline]
    fn from(v: &'a [f32; 4]) -> &'a Quaternion {
        unsafe { mem::transmute(v) }
    }
}

impl From<(f32, f32, f32, f32)> for Quaternion {
    #[inline]
    fn from(v: (f32, f32, f32, f32)) -> Quaternion {
        Quaternion::new(v.0, v.1, v.2, v.3)
    }
}

impl<'a> From<&'a (f32, f32, f32, f32)> for &'a Quaternion {
    #[inline]
    fn from(v: &'a (f32, f32, f32, f32)) -> &'a Quaternion {
        unsafe { mem::transmute(v) }
    }
}

impl ops::Index<usize> for Quaternion {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[f32; 4] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::Range<usize>> for Quaternion {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        let v: &[f32; 4] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::RangeTo<usize>> for Quaternion {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        let v: &[f32; 4] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::RangeFrom<usize>> for Quaternion {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        let v: &[f32; 4] = self.as_ref();
        &v[index]
    }
}

impl ops::Index<ops::RangeFull> for Quaternion {
    type Output = [f32];

    #[inline]
    fn index(&self, index: ops::RangeFull) -> &Self::Output {
        let v: &[f32; 4] = self.as_ref();
        &v[index]
    }
}

impl fmt::Display for Quaternion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Quaternion [s: {}, v: [{}, {}, {}]]", self.s, self.v.x, self.v.y, self.v.z)
    }
}

impl ops::Neg for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn neg(self) -> Self::Output {
        Quaternion::from_sv(-self.s, -self.v)
    }
}

impl<'a> ops::Neg for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn neg(self) -> Self::Output {
        Quaternion::from_sv(-self.s, -self.v)
    }
}

impl ops::Add<Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn add(self, other: Quaternion) -> Self::Output {
        Quaternion::new(
            other.s + self.s,
            other.v.x + self.v.x, other.v.y + self.v.y, other.v.z + self.v.z,
        )
    }
}

impl<'a> ops::Add<Quaternion> for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn add(self, other: Quaternion) -> Self::Output {
        Quaternion::new(
            other.s + self.s,
            other.v.x + self.v.x, other.v.y + self.v.y, other.v.z + self.v.z,
        )
    }
}

impl<'a> ops::Add<&'a Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn add(self, other: &'a Quaternion) -> Self::Output {
        Quaternion::new(
            other.s + self.s,
            other.v.x + self.v.x, other.v.y + self.v.y, other.v.z + self.v.z,
        )
    }
}

impl<'a, 'b> ops::Add<&'a Quaternion> for &'b Quaternion {
    type Output = Quaternion;

    #[inline]
    fn add(self, other: &'a Quaternion) -> Self::Output {
        Quaternion::new(
            other.s + self.s,
            other.v.x + self.v.x, other.v.y + self.v.y, other.v.z + self.v.z,
        )
    }
}

impl ops::Sub<Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn sub(self, other: Quaternion) -> Self::Output {
        Quaternion::new(
            other.s - self.s,
            other.v.x - self.v.x, other.v.y - self.v.y, other.v.z - self.v.z,
        )
    }
}

impl<'a> ops::Sub<Quaternion> for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn sub(self, other: Quaternion) -> Self::Output {
        Quaternion::new(
            other.s - self.s,
            other.v.x - self.v.x, other.v.y - self.v.y, other.v.z - self.v.z,
        )
    }
}

impl<'a> ops::Sub<&'a Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn sub(self, other: &'a Quaternion) -> Self::Output {
        Quaternion::new(
            other.s - self.s,
            other.v.x - self.v.x, other.v.y - self.v.y, other.v.z - self.v.z,
        )
    }
}

impl<'a, 'b> ops::Sub<&'a Quaternion> for &'b Quaternion {
    type Output = Quaternion;

    #[inline]
    fn sub(self, other: &'a Quaternion) -> Self::Output {
        Quaternion::new(
            other.s - self.s,
            other.v.x - self.v.x, other.v.y - self.v.y, other.v.z - self.v.z,
        )
    }
}

impl ops::Mul<f32> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: f32) -> Quaternion {
        Quaternion::new(
            self.s * other,
            self.v.x * other, self.v.y * other, self.v.z * other,
        )
    }
}

impl ops::Mul<f32> for &Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: f32) -> Quaternion {
        Quaternion::new(
            self.s * other,
            self.v.x * other, self.v.y * other, self.v.z * other,
        )
    }
}

impl<'a> ops::Mul<Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: Quaternion) -> Self::Output {
        Quaternion::new(
            other.s * self.s   - other.v.x * self.v.x - other.v.y * self.v.y - other.v.z * self.v.z,
            other.s * self.v.x + other.v.x * self.s   - other.v.y * self.v.z + other.v.z * self.v.y,
            other.s * self.v.y + other.v.x * self.v.z + other.v.y * self.s   - other.v.z * self.v.x,
            other.s * self.v.z - other.v.x * self.v.y + other.v.y * self.v.x + other.v.z * self.s,
        )
    }
}

impl<'a> ops::Mul<&'a Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: &'a Quaternion) -> Self::Output {
        Quaternion::new(
            other.s * self.s   - other.v.x * self.v.x - other.v.y * self.v.y - other.v.z * self.v.z,
            other.s * self.v.x + other.v.x * self.s   - other.v.y * self.v.z + other.v.z * self.v.y,
            other.s * self.v.y + other.v.x * self.v.z + other.v.y * self.s   - other.v.z * self.v.x,
            other.s * self.v.z - other.v.x * self.v.y + other.v.y * self.v.x + other.v.z * self.s,
        )
    }
}

impl<'a> ops::Mul<Quaternion> for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: Quaternion) -> Self::Output {
        Quaternion::new(
            other.s * self.s   - other.v.x * self.v.x - other.v.y * self.v.y - other.v.z * self.v.z,
            other.s * self.v.x + other.v.x * self.s   - other.v.y * self.v.z + other.v.z * self.v.y,
            other.s * self.v.y + other.v.x * self.v.z + other.v.y * self.s   - other.v.z * self.v.x,
            other.s * self.v.z - other.v.x * self.v.y + other.v.y * self.v.x + other.v.z * self.s,
        )
    }
}

impl<'a, 'b> ops::Mul<&'a Quaternion> for &'b Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: &'a Quaternion) -> Self::Output {
        Quaternion::new(
            other.s * self.s   - other.v.x * self.v.x - other.v.y * self.v.y - other.v.z * self.v.z,
            other.s * self.v.x + other.v.x * self.s   - other.v.y * self.v.z + other.v.z * self.v.y,
            other.s * self.v.y + other.v.x * self.v.z + other.v.y * self.s   - other.v.z * self.v.x,
            other.s * self.v.z - other.v.x * self.v.y + other.v.y * self.v.x + other.v.z * self.s,
        )
    }
}

impl ops::Div<f32> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn div(self, other: f32) -> Quaternion {
        Quaternion::new(
            self.s / other, 
            self.v.x / other, self.v.y / other, self.v.z / other,
        )
    }
}

impl<'a> ops::Div<f32> for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn div(self, other: f32) -> Quaternion {
        Quaternion::new(
            self.s / other, 
            self.v.x / other, self.v.y / other, self.v.z / other,
        )
    }
}

impl ops::Rem<f32> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn rem(self, other: f32) -> Self::Output {
        Quaternion::new(
            self.s % other,
            self.v.x % other, self.v.y % other, self.v.z % other,
        )
    }
}

impl ops::Rem<f32> for &Quaternion {
    type Output = Quaternion;

    #[inline]
    fn rem(self, other: f32) -> Self::Output {
        Quaternion::new(
            self.s % other,
            self.v.x % other, self.v.y % other, self.v.z % other,
        )
    }
}

impl ops::AddAssign<Quaternion> for Quaternion {
    fn add_assign(&mut self, other: Quaternion) {
        self.s += other.s;
        self.v += other.v;
    }
}

impl ops::AddAssign<&Quaternion> for Quaternion {
    fn add_assign(&mut self, other: &Quaternion) {
        self.s += other.s;
        self.v += other.v;
    }
}

impl ops::SubAssign<Quaternion> for Quaternion {
    fn sub_assign(&mut self, other: Quaternion) {
        self.s -= other.s;
        self.v -= other.v;
    }
}

impl ops::SubAssign<&Quaternion> for Quaternion {
    fn sub_assign(&mut self, other: &Quaternion) {
        self.s -= other.s;
        self.v -= other.v;
    }
}

impl ops::MulAssign<f32> for Quaternion {
    fn mul_assign(&mut self, other: f32) {
        self.s *= other;
        self.v *= other;
    }
}

impl ops::DivAssign<f32> for Quaternion {
    fn div_assign(&mut self, other: f32) {
        self.s /= other;
        self.v /= other;
    }
}

impl ops::RemAssign<f32> for Quaternion {
    fn rem_assign(&mut self, other: f32) {
        self.s %= other;
        self.v %= other;
    }
}

impl Metric<Quaternion> for Quaternion {
    fn distance2(self, other: Quaternion) -> f32 {
        (self.s - other.s)     * (self.s - other.s)     + 
        (self.v.x - other.v.x) * (self.v.x - other.v.x) + 
        (self.v.x - other.v.y) * (self.v.x - other.v.y) + 
        (self.v.x - other.v.z) * (self.v.x - other.v.z)
    }
}

impl Metric<&Quaternion> for Quaternion {
    fn distance2(self, other: &Quaternion) -> f32 {
        (self.s - other.s)     * (self.s - other.s)     + 
        (self.v.x - other.v.x) * (self.v.x - other.v.x) + 
        (self.v.x - other.v.y) * (self.v.x - other.v.y) + 
        (self.v.x - other.v.z) * (self.v.x - other.v.z)
    }
}

impl Metric<Quaternion> for &Quaternion {
    fn distance2(self, other: Quaternion) -> f32 {
        (self.s - other.s)     * (self.s - other.s)     + 
        (self.v.x - other.v.x) * (self.v.x - other.v.x) + 
        (self.v.x - other.v.y) * (self.v.x - other.v.y) + 
        (self.v.x - other.v.z) * (self.v.x - other.v.z)
    }
}

impl<'a, 'b> Metric<&'a Quaternion> for &'b Quaternion {
    fn distance2(self, other: &Quaternion) -> f32 {
        (self.s - other.s)     * (self.s - other.s)     + 
        (self.v.x - other.v.x) * (self.v.x - other.v.x) + 
        (self.v.x - other.v.y) * (self.v.x - other.v.y) + 
        (self.v.x - other.v.z) * (self.v.x - other.v.z)
    }
}

impl DotProduct<Quaternion> for Quaternion {
    fn dot(self, other: Quaternion) -> f32 {
        self.s * other.s + self.v.x * other.v.x + self.v.y * other.v.y + self.v.z * other.v.z
    }
}

impl DotProduct<&Quaternion> for Quaternion {
    fn dot(self, other: &Quaternion) -> f32 {
        self.s * other.s + self.v.x * other.v.x + self.v.y * other.v.y + self.v.z * other.v.z
    }
}

impl DotProduct<Quaternion> for &Quaternion {
    fn dot(self, other: Quaternion) -> f32 {
        self.s * other.s + self.v.x * other.v.x + self.v.y * other.v.y + self.v.z * other.v.z
    }
}

impl<'a, 'b> DotProduct<&'a Quaternion> for &'b Quaternion {
    fn dot(self, other: &'a Quaternion) -> f32 {
        self.s * other.s + self.v.x * other.v.x + self.v.y * other.v.y + self.v.z * other.v.z
    }
}

impl Magnitude<Quaternion> for Quaternion {}
impl Magnitude<Quaternion> for &Quaternion {}

