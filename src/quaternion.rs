use structure::{
    Storage,
    Zero,
    One,
    VectorSpace,
    //ProjectOn,
    DotProduct,
    Magnitude,
    Lerp,
    Metric,
};
use base::{
    Scalar,
    ScalarFloat,   
};
use matrix::{
    Matrix3, 
    Matrix4
};
use vector::Vector3;

use std::fmt;
use std::mem;
use std::ops;


const EPSILON: f32 = 0.00001;
const M_PI: f32 = 3.14159265358979323846264338327950288;
const ONE_DEG_IN_RAD: f32 = (2.0 * M_PI) / 360.0; // == 0.017444444


#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Quaternion<S> {
    /// The scalar component.
    s: S,
    /// The vector component.
    v: Vector3<S>,
}

impl<S> Quaternion<S> where S: Scalar {
    #[inline]
    pub fn new(s: S, x: S, y: S, z: S) -> Quaternion<S> {
        Self::from_sv(s, Vector3::new(x, y, z))
    }

    /// Compute a quaternion from it's scalar and vector parts.
    #[inline]
    pub fn from_sv(s: S, v: Vector3<S>) -> Quaternion<S> {
        Quaternion { s: s, v: v }
    }
}

impl<S> Quaternion<S> where S: ScalarFloat {
    /// Compute a quaternion corresponding to rotating about an axis in radians.
    pub fn from_axis_rad(radians: S, axis: Vector3<S>) -> Quaternion<S> {
        let two = S::one() + S::one();
        Quaternion::new(
            S::cos(radians / two),
            S::sin(radians / two) * axis.x,
            S::sin(radians / two) * axis.y,
            S::sin(radians / two) * axis.z,
        )
    }
    /*
    /// Computer a quaternion corresponding to rotating about an axis in degrees.
    pub fn from_axis_deg(degrees: S, axis: Vector3<S>) -> Quaternion<S> {
        Self::from_axis_rad(ONE_DEG_IN_RAD * degrees, axis)
    }
    */
    /// Compute the conjugate of a quaternion.
    pub fn conjugate(&self) -> Quaternion<S> {
        Quaternion::from_sv(self.s, -self.v)
    }

    /// Convert a quaternion to its equivalent matrix form using .
    pub fn to_mut_mat4(&self, m: &mut Matrix4<S>) {
        let s = self.s;
        let x = self.v.x;
        let y = self.v.y;
        let z = self.v.z;
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        m.c0r0 = one - two * y * y - two * z * z;
        m.c0r1 = two * x * y + two * s * z;
        m.c0r2 = two * x * z - two * s * y;
        m.c0r3 = zero;
        m.c1r0 = two * x * y - two * s * z;
        m.c1r1 = one - two * x * x - two * z * z;
        m.c1r2 = two * y * z + two * s * x;
        m.c1r3 = zero;
        m.c2r0 = two * x * z + two * s * y;
        m.c2r1 = two * y * z - two * s * x;
        m.c2r2 = one - two * x * x - two * y * y;
        m.c2r3 = zero;
        m.c3r0 = zero;
        m.c3r1 = zero;
        m.c3r2 = zero;
        m.c3r3 = one;
    }
    /*
    pub fn slerp(q: &mut Quaternion<S>, r: &Quaternion<S>, t: S) -> Quaternion<S> {
        // angle between q0-q1
        let mut cos_half_theta = q.dot(r);
        // as found here
        // http://stackoverflow.com/questions/2886606/flipping-issue-when-interpolating-rotations-using-quaternions
        // if dot product is negative then one quaternion should be negated, to make
        // it take the short way around, rather than the long way
        // yeah! and furthermore Susan, I had to recalculate the d.p. after this
        let zero = S::zero();
        let one = S::one();
        if cos_half_theta < zero {
            q.s *= -one;
            q.v.x *= -one;
            q.v.y *= -one;
            q.v.z *= -one;

            cos_half_theta = q.dot(r);
        }
        // if qa=qb or qa=-qb then theta = 0 and we can return qa
        if S::abs(cos_half_theta) >= one {
            return *q;
        }

        // Calculate temporary values
        let sin_half_theta = S::sqrt(one - cos_half_theta * cos_half_theta);
        // if theta = 180 degrees then result is not fully defined
        // we could rotate around any axis normal to qa or qb
        let mut result = Quaternion::new(one, zero, zero, zero);
        if f32::abs(sin_half_theta) < 0.001 {
            result.s   = (one - t) * q.s   + t * r.s;
            result.v.x = (one - t) * q.v.x + t * r.v.x;
            result.v.y = (one - t) * q.v.y + t * r.v.y;
            result.v.z = (one - t) * q.v.z + t * r.v.z;

            return result;
        }
        let half_theta = S::acos(cos_half_theta);
        let a = S::sin((one - t) * half_theta) / sin_half_theta;
        let b = S::sin(t * half_theta) / sin_half_theta;
        
        result.s   = q.s   * a + r.s   * b;
        result.v.x = q.v.x * a + r.v.x * b;
        result.v.y = q.v.y * a + r.v.y * b;
        result.v.z = q.v.z * a + r.v.z * b;

        result
    }
    */
}

impl<S> Zero for Quaternion<S> where S: Scalar {
    fn zero() -> Quaternion<S> {
        let zero = S::zero();
        Quaternion::new(zero, zero, zero, zero)
    }

    fn is_zero(&self) -> bool {
        let zero = S::zero();
        self.s == zero && self.v.x == zero && self.v.y == zero && self.v.z == zero
    }
}

impl<S> One for Quaternion<S> where S: Scalar {
    fn one() -> Quaternion<S> {
        let one = S::one();
        let zero = S::zero();
        Quaternion::new(one, zero, zero, zero)
    }
}

impl<S> AsRef<[S; 4]> for Quaternion<S> where S: Scalar {
    fn as_ref(&self) -> &[S; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsRef<(S, S, S, S)> for Quaternion<S> where S: Scalar {
    fn as_ref(&self) -> &(S, S, S, S) {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[S; 4]> for Quaternion<S> where S: Scalar {
    fn as_mut(&mut self) -> &mut [S; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<(S, S, S, S)> for Quaternion<S> where S: Scalar {
    fn as_mut(&mut self) -> &mut (S, S, S, S) {
        unsafe { mem::transmute(self) }
    }
}

impl<S> From<Quaternion<S>> for Matrix3<S> where S: Scalar {
    fn from(quat: Quaternion<S>) -> Matrix3<S> {
        let s = quat.s;
        let x = quat.v.x;
        let y = quat.v.y;
        let z = quat.v.z;
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
    
        Matrix3::new(
            one - two * y * y - two * z * z, two * x * y + two * s * z,       two * x * z - two * s * y,
            two * x * y - two * s * z,       one - two * x * x - two * z * z, two * y * z + two * s * x,
            two * x * z + two * s * y,       two * y * z - two * s * x,       one - two * x * x - two * y * y,
        )
    }
}

impl<S> From<Quaternion<S>> for Matrix4<S> where S: Scalar {
    fn from(quat: Quaternion<S>) -> Matrix4<S> {
        let s = quat.s;
        let x = quat.v.x;
        let y = quat.v.y;
        let z = quat.v.z;
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
    
        Matrix4::new(
            one - two * y * y - two * z * z, two * x * y + two * s * z,       two * x * z - two * s * y,       zero, 
            two * x * y - two * s * z,       one - two * x * x - two * z * z, two * y * z + two * s * x,       zero, 
            two * x * z + two * s * y,       two * y * z - two * s * x,       one - two * x * x - two * y * y, zero, 
            zero,                            zero,                            zero,                            one
        )
    }
}

impl<S> From<[S; 4]> for Quaternion<S> where S: Scalar {
    #[inline]
    fn from(v: [S; 4]) -> Quaternion<S> {
        Quaternion::new(v[0], v[1], v[2], v[3])
    }
}

impl<'a, S> From<&'a [S; 4]> for &'a Quaternion<S> where S: Scalar {
    #[inline]
    fn from(v: &'a [S; 4]) -> &'a Quaternion<S> {
        unsafe { mem::transmute(v) }
    }
}

impl<S> From<(S, S, S, S)> for Quaternion<S> where S: Scalar {
    #[inline]
    fn from(v: (S, S, S, S)) -> Quaternion<S> {
        Quaternion::new(v.0, v.1, v.2, v.3)
    }
}

impl<'a, S> From<&'a (S, S, S, S)> for &'a Quaternion<S> where S: Scalar {
    #[inline]
    fn from(v: &'a (S, S, S, S)) -> &'a Quaternion<S> {
        unsafe { mem::transmute(v) }
    }
}

impl<S> ops::Index<usize> for Quaternion<S> where S: Scalar {
    type Output = S;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[S; 4] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::Range<usize>> for Quaternion<S> where S: Scalar {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        let v: &[S; 4] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeTo<usize>> for Quaternion<S> where S: Scalar {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        let v: &[S; 4] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeFrom<usize>> for Quaternion<S> where S: Scalar {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        let v: &[S; 4] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeFull> for Quaternion<S> where S: Scalar {
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeFull) -> &Self::Output {
        let v: &[S; 4] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::IndexMut<usize> for Quaternion<S> where S: Scalar {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut S {
        let v: &mut [S; 4] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::Range<usize>> for Quaternion<S> where S: Scalar {
    #[inline]
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut [S] {
        let v: &mut [S; 4] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeTo<usize>> for Quaternion<S> where S: Scalar {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut [S] {
        let v: &mut [S; 4] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFrom<usize>> for Quaternion<S> where S: Scalar {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut [S] {
        let v: &mut [S; 4] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFull> for Quaternion<S> where S: Scalar {
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFull) -> &mut [S] {
        let v: &mut [S; 4] = self.as_mut();
        &mut v[index]
    }
}

impl<S> fmt::Display for Quaternion<S> where S: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Quaternion [s: {}, v: [{}, {}, {}]]", self.s, self.v.x, self.v.y, self.v.z)
    }
}

impl<S> ops::Neg for Quaternion<S> where S: ScalarFloat {
    type Output = Quaternion<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Quaternion::from_sv(-self.s, -self.v)
    }
}

impl<'a, S> ops::Neg for &'a Quaternion<S> where S: ScalarFloat {
    type Output = Quaternion<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Quaternion::from_sv(-self.s, -self.v)
    }
}

impl<S> ops::Add<Quaternion<S>> for Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn add(self, other: Quaternion<S>) -> Self::Output {
        Quaternion::new(
            other.s + self.s,
            other.v.x + self.v.x, other.v.y + self.v.y, other.v.z + self.v.z,
        )
    }
}

impl<'a, S> ops::Add<Quaternion<S>> for &'a Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn add(self, other: Quaternion<S>) -> Self::Output {
        Quaternion::new(
            other.s + self.s,
            other.v.x + self.v.x, other.v.y + self.v.y, other.v.z + self.v.z,
        )
    }
}

impl<'a, S> ops::Add<&'a Quaternion<S>> for Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn add(self, other: &'a Quaternion<S>) -> Self::Output {
        Quaternion::new(
            other.s + self.s,
            other.v.x + self.v.x, other.v.y + self.v.y, other.v.z + self.v.z,
        )
    }
}

impl<'a, 'b, S> ops::Add<&'a Quaternion<S>> for &'b Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn add(self, other: &'a Quaternion<S>) -> Self::Output {
        Quaternion::new(
            other.s + self.s,
            other.v.x + self.v.x, other.v.y + self.v.y, other.v.z + self.v.z,
        )
    }
}

impl<S> ops::Sub<Quaternion<S>> for Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn sub(self, other: Quaternion<S>) -> Self::Output {
        Quaternion::new(
            other.s - self.s,
            other.v.x - self.v.x, other.v.y - self.v.y, other.v.z - self.v.z,
        )
    }
}

impl<'a, S> ops::Sub<Quaternion<S>> for &'a Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn sub(self, other: Quaternion<S>) -> Self::Output {
        Quaternion::new(
            other.s - self.s,
            other.v.x - self.v.x, other.v.y - self.v.y, other.v.z - self.v.z,
        )
    }
}

impl<'a, S> ops::Sub<&'a Quaternion<S>> for Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn sub(self, other: &'a Quaternion<S>) -> Self::Output {
        Quaternion::new(
            other.s - self.s,
            other.v.x - self.v.x, other.v.y - self.v.y, other.v.z - self.v.z,
        )
    }
}

impl<'a, 'b, S> ops::Sub<&'a Quaternion<S>> for &'b Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn sub(self, other: &'a Quaternion<S>) -> Self::Output {
        Quaternion::new(
            other.s - self.s,
            other.v.x - self.v.x, other.v.y - self.v.y, other.v.z - self.v.z,
        )
    }
}

impl<S> ops::Mul<S> for Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn mul(self, other: S) -> Quaternion<S> {
        Quaternion::new(
            self.s * other,
            self.v.x * other, self.v.y * other, self.v.z * other,
        )
    }
}

impl<S> ops::Mul<S> for &Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn mul(self, other: S) -> Quaternion<S> {
        Quaternion::new(
            self.s * other,
            self.v.x * other, self.v.y * other, self.v.z * other,
        )
    }
}

impl<'a, S> ops::Mul<Quaternion<S>> for Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn mul(self, other: Quaternion<S>) -> Self::Output {
        Quaternion::new(
            other.s * self.s   - other.v.x * self.v.x - other.v.y * self.v.y - other.v.z * self.v.z,
            other.s * self.v.x + other.v.x * self.s   - other.v.y * self.v.z + other.v.z * self.v.y,
            other.s * self.v.y + other.v.x * self.v.z + other.v.y * self.s   - other.v.z * self.v.x,
            other.s * self.v.z - other.v.x * self.v.y + other.v.y * self.v.x + other.v.z * self.s,
        )
    }
}

impl<'a, S> ops::Mul<&'a Quaternion<S>> for Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn mul(self, other: &'a Quaternion<S>) -> Self::Output {
        Quaternion::new(
            other.s * self.s   - other.v.x * self.v.x - other.v.y * self.v.y - other.v.z * self.v.z,
            other.s * self.v.x + other.v.x * self.s   - other.v.y * self.v.z + other.v.z * self.v.y,
            other.s * self.v.y + other.v.x * self.v.z + other.v.y * self.s   - other.v.z * self.v.x,
            other.s * self.v.z - other.v.x * self.v.y + other.v.y * self.v.x + other.v.z * self.s,
        )
    }
}

impl<'a, S> ops::Mul<Quaternion<S>> for &'a Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn mul(self, other: Quaternion<S>) -> Self::Output {
        Quaternion::new(
            other.s * self.s   - other.v.x * self.v.x - other.v.y * self.v.y - other.v.z * self.v.z,
            other.s * self.v.x + other.v.x * self.s   - other.v.y * self.v.z + other.v.z * self.v.y,
            other.s * self.v.y + other.v.x * self.v.z + other.v.y * self.s   - other.v.z * self.v.x,
            other.s * self.v.z - other.v.x * self.v.y + other.v.y * self.v.x + other.v.z * self.s,
        )
    }
}

impl<'a, 'b, S> ops::Mul<&'a Quaternion<S>> for &'b Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn mul(self, other: &'a Quaternion<S>) -> Self::Output {
        Quaternion::new(
            other.s * self.s   - other.v.x * self.v.x - other.v.y * self.v.y - other.v.z * self.v.z,
            other.s * self.v.x + other.v.x * self.s   - other.v.y * self.v.z + other.v.z * self.v.y,
            other.s * self.v.y + other.v.x * self.v.z + other.v.y * self.s   - other.v.z * self.v.x,
            other.s * self.v.z - other.v.x * self.v.y + other.v.y * self.v.x + other.v.z * self.s,
        )
    }
}

impl<S> ops::Div<S> for Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn div(self, other: S) -> Quaternion<S> {
        Quaternion::new(
            self.s / other, 
            self.v.x / other, self.v.y / other, self.v.z / other,
        )
    }
}

impl<'a, S> ops::Div<S> for &'a Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn div(self, other: S) -> Quaternion<S> {
        Quaternion::new(
            self.s / other, 
            self.v.x / other, self.v.y / other, self.v.z / other,
        )
    }
}

impl<S> ops::Rem<S> for Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        Quaternion::new(
            self.s % other,
            self.v.x % other, self.v.y % other, self.v.z % other,
        )
    }
}

impl<S> ops::Rem<S> for &Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        Quaternion::new(
            self.s % other,
            self.v.x % other, self.v.y % other, self.v.z % other,
        )
    }
}

impl<S> ops::AddAssign<Quaternion<S>> for Quaternion<S> where S: Scalar {
    fn add_assign(&mut self, other: Quaternion<S>) {
        self.s += other.s;
        self.v += other.v;
    }
}

impl<S> ops::AddAssign<&Quaternion<S>> for Quaternion<S> where S: Scalar {
    fn add_assign(&mut self, other: &Quaternion<S>) {
        self.s += other.s;
        self.v += other.v;
    }
}

impl<S> ops::SubAssign<Quaternion<S>> for Quaternion<S> where S: Scalar {
    fn sub_assign(&mut self, other: Quaternion<S>) {
        self.s -= other.s;
        self.v -= other.v;
    }
}

impl<S> ops::SubAssign<&Quaternion<S>> for Quaternion<S> where S: Scalar {
    fn sub_assign(&mut self, other: &Quaternion<S>) {
        self.s -= other.s;
        self.v -= other.v;
    }
}

impl<S> ops::MulAssign<S> for Quaternion<S> where S: Scalar {
    fn mul_assign(&mut self, other: S) {
        self.s *= other;
        self.v *= other;
    }
}

impl<S> ops::DivAssign<S> for Quaternion<S> where S: Scalar {
    fn div_assign(&mut self, other: S) {
        self.s /= other;
        self.v /= other;
    }
}

impl<S> ops::RemAssign<S> for Quaternion<S> where S: Scalar {
    fn rem_assign(&mut self, other: S) {
        self.s %= other;
        self.v %= other;
    }
}

impl<S> Metric<Quaternion<S>> for Quaternion<S> where S: ScalarFloat {
    type Output = S;

    fn distance_squared(self, other: Quaternion<S>) -> S {
        (self.s - other.s)     * (self.s - other.s)     + 
        (self.v.x - other.v.x) * (self.v.x - other.v.x) + 
        (self.v.x - other.v.y) * (self.v.x - other.v.y) + 
        (self.v.x - other.v.z) * (self.v.x - other.v.z)
    }
}

impl<S> Metric<&Quaternion<S>> for Quaternion<S> where S: ScalarFloat {
    type Output = S;

    fn distance_squared(self, other: &Quaternion<S>) -> S {
        (self.s - other.s)     * (self.s - other.s)     + 
        (self.v.x - other.v.x) * (self.v.x - other.v.x) + 
        (self.v.x - other.v.y) * (self.v.x - other.v.y) + 
        (self.v.x - other.v.z) * (self.v.x - other.v.z)
    }
}

impl<S> Metric<Quaternion<S>> for &Quaternion<S> where S: ScalarFloat {
    type Output = S;

    fn distance_squared(self, other: Quaternion<S>) -> S {
        (self.s - other.s)     * (self.s - other.s)     + 
        (self.v.x - other.v.x) * (self.v.x - other.v.x) + 
        (self.v.x - other.v.y) * (self.v.x - other.v.y) + 
        (self.v.x - other.v.z) * (self.v.x - other.v.z)
    }
}

impl<'a, 'b, S> Metric<&'a Quaternion<S>> for &'b Quaternion<S> where S: ScalarFloat {
    type Output = S;

    fn distance_squared(self, other: &Quaternion<S>) -> S {
        (self.s - other.s)     * (self.s - other.s)     + 
        (self.v.x - other.v.x) * (self.v.x - other.v.x) + 
        (self.v.x - other.v.y) * (self.v.x - other.v.y) + 
        (self.v.x - other.v.z) * (self.v.x - other.v.z)
    }
}

impl<S> DotProduct<Quaternion<S>> for Quaternion<S> where S: Scalar {
    type Output = S;

    fn dot(self, other: Quaternion<S>) -> Self::Output {
        self.s * other.s + self.v.x * other.v.x + self.v.y * other.v.y + self.v.z * other.v.z
    }
}

impl<S> DotProduct<&Quaternion<S>> for Quaternion<S> where S: Scalar {
    type Output = S;

    fn dot(self, other: &Quaternion<S>) -> Self::Output {
        self.s * other.s + self.v.x * other.v.x + self.v.y * other.v.y + self.v.z * other.v.z
    }
}

impl<S> DotProduct<Quaternion<S>> for &Quaternion<S> where S: Scalar {
    type Output = S;

    fn dot(self, other: Quaternion<S>) -> Self::Output {
        self.s * other.s + self.v.x * other.v.x + self.v.y * other.v.y + self.v.z * other.v.z
    }
}

impl<'a, 'b, S> DotProduct<&'a Quaternion<S>> for &'b Quaternion<S> where S: Scalar {
    type Output = S;

    fn dot(self, other: &'a Quaternion<S>) -> Self::Output {
        self.s * other.s + self.v.x * other.v.x + self.v.y * other.v.y + self.v.z * other.v.z
    }
}

impl<S> Magnitude for Quaternion<S> where S: ScalarFloat {
    type Output = S;

    fn magnitude(&self) -> Self::Output {
        Self::Output::abs(self.magnitude_squared())
    }

    fn magnitude_squared(&self) -> Self::Output {
        <&Self as DotProduct<&Self>>::dot(self, self)
    }

    fn normalize(&self) -> Self {
        self / self.magnitude()
    }

    fn normalize_to(&self, magnitude: Self::Output) -> Self {
        self * (magnitude / self.magnitude())
    }
}

