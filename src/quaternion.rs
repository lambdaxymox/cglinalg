use std::fmt;
use std::mem;
use std::ops;
use std::cmp;

use crate::vector::Vector3;
use crate::matrix::{Matrix3, Matrix4};
use crate::traits::{Array, Zero, VectorSpace, MetricSpace, DotProduct, Lerp};


const EPSILON: f32 = 0.00001;
const M_PI: f32 = 3.14159265358979323846264338327950288;
const ONE_DEG_IN_RAD: f32 = (2.0 * M_PI) / 360.0; // == 0.017444444


#[derive(Copy, Clone, PartialEq)]
pub struct Quaternion {
    s: f32,
    x: f32,
    y: f32,
    z: f32,
}

impl Quaternion {
    pub fn new(s: f32, x: f32, y: f32, z: f32) -> Quaternion {
        let q = Quaternion { s: s, x: x, y: y, z: z };

        q.normalize()
    }

    pub fn normalize(&self) -> Quaternion {
        let sum = self.s * self.s + self.x * self.x + self.y * self.y + self.z * self.z;
        // NOTE: f32s have min 6 digits of precision.
        let threshold = 0.0001;
        if f32::abs(1.0 - sum) < threshold {
            return *self;
        }

        let norm = f32::sqrt(sum);
        self / norm
    }

    ///
    /// Create a zero quaterion. It is a quaternion such that 
    /// q - q = 0.
    ///
    pub fn zero() -> Quaternion {
        Quaternion { s: 0.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    ///
    /// Create a unit quaternion who is also the multiplicative identity:
    /// q * q^-1 == 1.
    ///
    pub fn one() -> Quaternion {
        Quaternion { s: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    ///
    /// Compute the inner (dot) product of two quaternions.
    ///
    pub fn dot(&self, r: &Quaternion) -> f32 {
        self.s * r.s + self.x * r.x + self.y * r.y + self.z * r.z
    }

    ///
    /// Compute the euclidean norm of a quaternion.
    ///
    pub fn norm(&self) -> f32 {
        f32::sqrt(self.s * self.s + self.x * self.x + self.y * self.y + self.z * self.z)
    }

    ///
    /// Compute the squared Euclidean norm of a quaternion.
    ///
    pub fn norm2(&self) -> f32 {
        self.s * self.s + self.x * self.x + self.y * self.y + self.z * self.z
    }

    ///
    /// Compute a quaternion from it's scalar and vector parts.
    ///
    pub fn from_sv(s: f32, v: Vector3) -> Quaternion {
        Quaternion { s: s, x: v.x, y: v.y, z: v.z }
    }

    ///
    /// Compute a quaternion corresponding to rotating about an axis in radians.
    ///
    pub fn from_axis_rad(radians: f32, axis: Vector3) -> Quaternion {
        Quaternion {
            s: f32::cos(radians / 2.0),
            x: f32::sin(radians / 2.0) * axis.x,
            y: f32::sin(radians / 2.0) * axis.y,
            z: f32::sin(radians / 2.0) * axis.z,
        }
    }

    ///
    /// Computer a quaternion corresponding to rotating about an axis in degrees.
    ///
    pub fn from_axis_deg(degrees: f32, axis: Vector3) -> Quaternion {
        Self::from_axis_rad(ONE_DEG_IN_RAD * degrees, axis)
    }

    ///
    /// Compute the conjugate of a quaternion.
    ///
    pub fn conjugate(&self) -> Quaternion {
        Quaternion { s: self.s, x: -self.x, y: -self.y, z: -self.z }
    }

    ///
    /// Convert a quaternion to its equivalent matrix form using .
    ///
    pub fn to_mut_mat4(&self, m: &mut Matrix4) {
        let s = self.s;
        let x = self.x;
        let y = self.y;
        let z = self.z;
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
            q.x *= -1.0;
            q.y *= -1.0;
            q.z *= -1.0;

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
        let mut result = Quaternion { s: 1.0, x: 0.0, y: 0.0, z: 0.0 };
        if f32::abs(sin_half_theta) < 0.001 {
            result.s = (1.0 - t) * q.s + t * r.s;
            result.x = (1.0 - t) * q.x + t * r.x;
            result.y = (1.0 - t) * q.y + t * r.y;
            result.z = (1.0 - t) * q.z + t * r.z;

            return result;
        }
        let half_theta = f32::acos(cos_half_theta);
        let a = f32::sin((1.0 - t) * half_theta) / sin_half_theta;
        let b = f32::sin(t * half_theta) / sin_half_theta;
        
        result.s = q.s * a + r.s * b;
        result.x = q.x * a + r.x * b;
        result.y = q.y * a + r.y * b;
        result.z = q.z * a + r.z * b;

        result
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
        let x = quat.x;
        let y = quat.y;
        let z = quat.z;
    
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
        let x = quat.x;
        let y = quat.y;
        let z = quat.z;
    
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

impl fmt::Debug for Quaternion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector4 ")?;
        writeln!(f, "[{}, [{}, {}, {}]]", self.s, self.x, self.y, self.z)
    }
}

impl fmt::Display for Quaternion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[{:.2}, [{:.2}, {:.2}, {:.2}]]", self.s, self.x, self.y, self.z)
    }
}

impl ops::Neg for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn neg(self) -> Self::Output {
        Quaternion { s: -self.s, x: -self.x, y: -self.y, z: -self.z }
    }
}

impl<'a> ops::Neg for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn neg(self) -> Self::Output {
        Quaternion { s: -self.s, x: -self.x, y: -self.y, z: -self.z }
    }
}

impl ops::Add<Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn add(self, other: Quaternion) -> Self::Output {
        Quaternion {
            s: other.s + self.s,
            x: other.x + self.x,
            y: other.y + self.y,
            z: other.z + self.z,
        }
    }
}

impl<'a> ops::Add<Quaternion> for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn add(self, other: Quaternion) -> Self::Output {
        Quaternion {
            s: other.s + self.s,
            x: other.x + self.x,
            y: other.y + self.y,
            z: other.z + self.z,
        }
    }
}

impl<'a> ops::Add<&'a Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn add(self, other: &'a Quaternion) -> Self::Output {
        Quaternion {
            s: other.s + self.s,
            x: other.x + self.x,
            y: other.y + self.y,
            z: other.z + self.z,
        }
    }
}

impl<'a, 'b> ops::Add<&'a Quaternion> for &'b Quaternion {
    type Output = Quaternion;

    #[inline]
    fn add(self, other: &'a Quaternion) -> Self::Output {
        Quaternion {
            s: other.s + self.s,
            x: other.x + self.x,
            y: other.y + self.y,
            z: other.z + self.z,
        }
    }
}

impl ops::Sub<Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn sub(self, other: Quaternion) -> Self::Output {
        Quaternion {
            s: other.s - self.s,
            x: other.x - self.x,
            y: other.y - self.y,
            z: other.z - self.z,
        }
    }
}

impl<'a> ops::Sub<Quaternion> for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn sub(self, other: Quaternion) -> Self::Output {
        Quaternion {
            s: other.s - self.s,
            x: other.x - self.x,
            y: other.y - self.y,
            z: other.z - self.z,
        }
    }
}

impl<'a> ops::Sub<&'a Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn sub(self, other: &'a Quaternion) -> Self::Output {
        Quaternion {
            s: other.s - self.s,
            x: other.x - self.x,
            y: other.y - self.y,
            z: other.z - self.z,
        }
    }
}

impl<'a, 'b> ops::Sub<&'a Quaternion> for &'b Quaternion {
    type Output = Quaternion;

    #[inline]
    fn sub(self, other: &'a Quaternion) -> Self::Output {
        Quaternion {
            s: other.s - self.s,
            x: other.x - self.x,
            y: other.y - self.y,
            z: other.z - self.z,
        }
    }
}

impl ops::Mul<f32> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: f32) -> Quaternion {
        Quaternion {
            s: self.s * other,
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl ops::Mul<f32> for &Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: f32) -> Quaternion {
        Quaternion {
            s: self.s * other,
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl<'a> ops::Mul<Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: Quaternion) -> Self::Output {
        Quaternion {
            s: other.s * self.s - other.x * self.x - other.y * self.y - other.z * self.z,
            x: other.s * self.x + other.x * self.s - other.y * self.z + other.z * self.y,
            y: other.s * self.y + other.x * self.z + other.y * self.s - other.z * self.x,
            z: other.s * self.z - other.x * self.y + other.y * self.x + other.z * self.s,
        }
    }
}

impl<'a> ops::Mul<&'a Quaternion> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: &'a Quaternion) -> Self::Output {
        Quaternion {
            s: other.s * self.s - other.x * self.x - other.y * self.y - other.z * self.z,
            x: other.s * self.x + other.x * self.s - other.y * self.z + other.z * self.y,
            y: other.s * self.y + other.x * self.z + other.y * self.s - other.z * self.x,
            z: other.s * self.z - other.x * self.y + other.y * self.x + other.z * self.s,
        }
    }
}

impl<'a> ops::Mul<Quaternion> for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: Quaternion) -> Self::Output {
        Quaternion {
            s: other.s * self.s - other.x * self.x - other.y * self.y - other.z * self.z,
            x: other.s * self.x + other.x * self.s - other.y * self.z + other.z * self.y,
            y: other.s * self.y + other.x * self.z + other.y * self.s - other.z * self.x,
            z: other.s * self.z - other.x * self.y + other.y * self.x + other.z * self.s,
        }
    }
}

impl<'a, 'b> ops::Mul<&'a Quaternion> for &'b Quaternion {
    type Output = Quaternion;

    #[inline]
    fn mul(self, other: &'a Quaternion) -> Self::Output {
        Quaternion {
            s: other.s * self.s - other.x * self.x - other.y * self.y - other.z * self.z,
            x: other.s * self.x + other.x * self.s - other.y * self.z + other.z * self.y,
            y: other.s * self.y + other.x * self.z + other.y * self.s - other.z * self.x,
            z: other.s * self.z - other.x * self.y + other.y * self.x + other.z * self.s,
        }
    }
}

impl ops::Div<f32> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn div(self, other: f32) -> Quaternion {
        Quaternion {
            s: self.s / other, 
            x: self.x / other, 
            y: self.y / other, 
            z: self.z / other,
        }
    }
}

impl<'a> ops::Div<f32> for &'a Quaternion {
    type Output = Quaternion;

    #[inline]
    fn div(self, other: f32) -> Quaternion {
        Quaternion {
            s: self.s / other, 
            x: self.x / other, 
            y: self.y / other, 
            z: self.z / other,
        }
    }
}

impl ops::Rem<f32> for Quaternion {
    type Output = Quaternion;

    #[inline]
    fn rem(self, other: f32) -> Self::Output {
        Quaternion {
            s: self.s % other,
            x: self.x % other,
            y: self.y % other,
            z: self.z % other,
        }
    }
}

impl ops::Rem<f32> for &Quaternion {
    type Output = Quaternion;

    #[inline]
    fn rem(self, other: f32) -> Self::Output {
        Quaternion {
            s: self.s % other,
            x: self.x % other,
            y: self.y % other,
            z: self.z % other,
        }
    }
}
