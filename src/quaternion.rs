use structure::{
    Angle,
    Storage,
    Zero,
    One,
    //ProjectOn,
    DotProduct,
    Magnitude,
    Lerp,
    Nlerp,
    Metric,
    Finite,
};
use angle::{
    Radians,
    Degrees,
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


#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Quaternion<S> {
    /// The scalar component.
    s: S,
    /// The vector component.
    v: Vector3<S>,
}

impl<S> Quaternion<S> where S: Scalar {
    /// Construct a new quaternion.
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
    pub fn from_axis_rad(radians: Radians<S>, axis: Vector3<S>) -> Quaternion<S> {
        let two = S::one() + S::one();
        Quaternion::new(
            Radians::cos(radians / two),
            Radians::sin(radians / two) * axis.x,
            Radians::sin(radians / two) * axis.y,
            Radians::sin(radians / two) * axis.z,
        )
    }

    /// Compute a quaternion corresponding to rotating about an axis in degrees.
    pub fn from_axis_deg(degrees: Degrees<S>, axis: Vector3<S>) -> Quaternion<S> {
        Self::from_axis_rad(degrees.into(), axis)
    }

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

    /// Compute the inverse of a quaternion.
    ///
    /// If `self` has zero magnitude, no inverse exists for it. In this 
    /// case the function return `None`. Otherwise it returns the inverse of `self`.
    pub fn inverse(&self) -> Option<Quaternion<S>> {
        let magnitude_squared = self.magnitude_squared();
        if magnitude_squared == S::zero() {
            None
        } else {
            Some(self.conjugate() / magnitude_squared)
        }
    }

    /// Determine whether a quaternion is invertible.
    pub fn is_invertible(&self) -> bool {
        self.magnitude_squared() > S::zero()
    }

    /// Spherically linearly interpolate between two quaternions.
    pub fn slerp(&mut self, other: &Quaternion<S>, amount: S) -> Quaternion<S> {
        // angle between q0-q1
        let mut cos_half_theta = self.dot(other);
        // as found here
        // http://stackoverflow.com/questions/2886606/flipping-issue-when-interpolating-rotations-using-quaternions
        // if dot product is negative then one quaternion should be negated, to make
        // it take the short way around, rather than the long way
        // yeah! and furthermore Susan, I had to recalculate the d.p. after this
        let zero = S::zero();
        let one = S::one();
        if cos_half_theta < zero {
            self.s *= -one;
            self.v *= -one;

            cos_half_theta = self.dot(other);
        }
        // if qa=qb or qa=-qb then theta = 0 and we can return qa
        if S::abs(cos_half_theta) >= one {
            return *self;
        }

        // Calculate temporary values
        let sin_half_theta = S::sqrt(one - cos_half_theta * cos_half_theta);
        // if theta = 180 degrees then result is not fully defined
        // we could rotate around any axis normal to qa or qb
        // let mut result = Quaternion::new(one, zero, zero, zero);
        let threshold = num_traits::cast(0.001).unwrap();
        if S::abs(sin_half_theta) < threshold {
            // Linearly interpolate if the arc between quaternions is small enough.
            return self.nlerp(other, amount);
        }
        let half_theta = S::acos(cos_half_theta);
        let a = S::sin((one - amount) * half_theta) / sin_half_theta;
        let b = S::sin(amount * half_theta) / sin_half_theta;
        
        let s   = self.s   * a + other.s   * b;
        let v_x = self.v.x * a + other.v.x * b;
        let v_y = self.v.y * a + other.v.y * b;
        let v_z = self.v.z * a + other.v.z * b;

        Quaternion::new(s, v_x, v_y, v_z)
    }

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
        Quaternion::from_sv(self.s + other.s, self.v + other.v)
    }
}

impl<'a, S> ops::Add<Quaternion<S>> for &'a Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn add(self, other: Quaternion<S>) -> Self::Output {
        Quaternion::from_sv(self.s + other.s, self.v + other.v)
    }
}

impl<'a, S> ops::Add<&'a Quaternion<S>> for Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn add(self, other: &'a Quaternion<S>) -> Self::Output {
        Quaternion::from_sv(self.s + other.s, self.v + other.v)
    }
}

impl<'a, 'b, S> ops::Add<&'a Quaternion<S>> for &'b Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn add(self, other: &'a Quaternion<S>) -> Self::Output {
        Quaternion::from_sv(self.s + other.s, self.v + other.v)
    }
}

impl<S> ops::Sub<Quaternion<S>> for Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn sub(self, other: Quaternion<S>) -> Self::Output {
        Quaternion::from_sv(self.s - other.s, self.v - other.v)
    }
}

impl<'a, S> ops::Sub<Quaternion<S>> for &'a Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn sub(self, other: Quaternion<S>) -> Self::Output {
        Quaternion::from_sv(self.s - other.s, self.v - other.v)
    }
}

impl<'a, S> ops::Sub<&'a Quaternion<S>> for Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn sub(self, other: &'a Quaternion<S>) -> Self::Output {
        Quaternion::from_sv(self.s - other.s, self.v - other.v)
    }
}

impl<'a, 'b, S> ops::Sub<&'a Quaternion<S>> for &'b Quaternion<S> where S: Scalar {
    type Output = Quaternion<S>;

    #[inline]
    fn sub(self, other: &'a Quaternion<S>) -> Self::Output {
        Quaternion::from_sv(self.s - other.s, self.v - other.v)
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
        Self::Output::sqrt(Self::Output::abs(self.magnitude_squared()))
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

impl<S> Lerp<Quaternion<S>> for Quaternion<S> where S: ScalarFloat {
    type Scalar = S;
    type Output = Quaternion<S>;

    fn lerp(self, other: Quaternion<S>, amount: Self::Scalar) -> Self::Output {
        self + (other - self) * amount
    }
}

impl<S> Lerp<&Quaternion<S>> for Quaternion<S> where S: ScalarFloat {
    type Scalar = S;
    type Output = Quaternion<S>;

    fn lerp(self, other: &Quaternion<S>, amount: Self::Scalar) -> Self::Output {
        self + (other - self) * amount
    }
}

impl<S> Lerp<Quaternion<S>> for &Quaternion<S> where S: ScalarFloat {
    type Scalar = S;
    type Output = Quaternion<S>;

    fn lerp(self, other: Quaternion<S>, amount: Self::Scalar) -> Self::Output {
        self + (other - self) * amount
    }
}

impl<'a, 'b, S> Lerp<&'a Quaternion<S>> for &'b Quaternion<S> where S: ScalarFloat {
    type Scalar = S;
    type Output = Quaternion<S>;

    fn lerp(self, other: &'a Quaternion<S>, amount: Self::Scalar) -> Self::Output {
        self + (other - self) * amount
    }
}

impl<S> Nlerp<Quaternion<S>> for Quaternion<S> where S: ScalarFloat {
    type Scalar = S;
    type Output = Quaternion<S>;

    fn nlerp(self, other: Quaternion<S>, amount: S) -> Quaternion<S> {
        (self * (S::one() - amount) + other * amount).normalize()
    }
}

impl<S> Nlerp<&Quaternion<S>> for Quaternion<S> where S: ScalarFloat {
    type Scalar = S;
    type Output = Quaternion<S>;

    fn nlerp(self, other: &Quaternion<S>, amount: S) -> Quaternion<S> {
        (self * (S::one() - amount) + other * amount).normalize()
    }
}

impl<S> Nlerp<Quaternion<S>> for &Quaternion<S> where S: ScalarFloat {
    type Scalar = S;
    type Output = Quaternion<S>;

    fn nlerp(self, other: Quaternion<S>, amount: S) -> Quaternion<S> {
        (self * (S::one() - amount) + other * amount).normalize()
    }
}

impl<'a, 'b, S> Nlerp<&'a Quaternion<S>> for &'b Quaternion<S> where S: ScalarFloat {
    type Scalar = S;
    type Output = Quaternion<S>;

    fn nlerp(self, other: &'a Quaternion<S>, amount: S) -> Quaternion<S> {
        (self * (S::one() - amount) + other * amount).normalize()
    }
}

macro_rules! impl_mul_operator {
    ($Lhs:ty, $Rhs:ty, $Output:ty, { $scalar:ident, { $($field:ident),* } }) => {
        impl ops::Mul<$Rhs> for $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                <$Output>::new( self * other.$scalar, $(self * other.v.$field),*)
            }
        }

        impl<'a> ops::Mul<$Rhs> for &'a $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                <$Output>::new( self * other.$scalar, $(self * other.v.$field),*)
            }
        }
    }
}

impl_mul_operator!(u8,    Quaternion<u8>,    Quaternion<u8>,    { s, { x, y, z } });
impl_mul_operator!(u16,   Quaternion<u16>,   Quaternion<u16>,   { s, { x, y, z } });
impl_mul_operator!(u32,   Quaternion<u32>,   Quaternion<u32>,   { s, { x, y, z } });
impl_mul_operator!(u64,   Quaternion<u64>,   Quaternion<u64>,   { s, { x, y, z } });
impl_mul_operator!(u128,  Quaternion<u128>,  Quaternion<u128>,  { s, { x, y, z } });
impl_mul_operator!(usize, Quaternion<usize>, Quaternion<usize>, { s, { x, y, z } });
impl_mul_operator!(i8,    Quaternion<i8>,    Quaternion<i8>,    { s, { x, y, z } });
impl_mul_operator!(i16,   Quaternion<i16>,   Quaternion<i16>,   { s, { x, y, z } });
impl_mul_operator!(i32,   Quaternion<i32>,   Quaternion<i32>,   { s, { x, y, z } });
impl_mul_operator!(i64,   Quaternion<i64>,   Quaternion<i64>,   { s, { x, y, z } });
impl_mul_operator!(i128,  Quaternion<i128>,  Quaternion<i128>,  { s, { x, y, z } });
impl_mul_operator!(isize, Quaternion<isize>, Quaternion<isize>, { s, { x, y, z } });
impl_mul_operator!(f32,   Quaternion<f32>,   Quaternion<f32>,   { s, { x, y, z } });
impl_mul_operator!(f64,   Quaternion<f64>,   Quaternion<f64>,   { s, { x, y, z } });


impl<S> approx::AbsDiffEq for Quaternion<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        S::abs_diff_eq(&self.s, &other.s, epsilon) &&
        Vector3::abs_diff_eq(&self.v, &other.v, epsilon)
    }
}

impl<S> approx::RelativeEq for Quaternion<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.s, &other.s, epsilon, max_relative) &&
        Vector3::relative_eq(&self.v, &other.v, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Quaternion<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.s, &other.s, epsilon, max_ulps) &&
        Vector3::ulps_eq(&self.v, &other.v, epsilon, max_ulps)
    }
}

impl<S> Finite for Quaternion<S> where S: ScalarFloat {
    #[inline]
    fn is_finite(self) -> bool {
        self.s.is_finite() && self.v.is_finite()
    }

    #[inline]
    fn is_not_finite(self) -> bool {
        !self.is_finite()
    }
}



#[cfg(test)]
mod lerp_tests {
    use structure::{
        Lerp, 
        Nlerp,
        Magnitude
    };
    use super::Quaternion;


    #[test]
    fn test_nlerp() {
        let q1 = Quaternion::new(0_f64, 0_f64, 0_f64, 0_f64);
        let q2 = Quaternion::new(1_f64, 1_f64, 1_f64, 1_f64);
        let amount = 0.5;
        let result = q1.nlerp(q2, amount);
        let expected = Quaternion::new(0.5, 0.5, 0.5, 0.5);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_nlerp_should_interpolate_to_endoints_normalized() {
        let q1 = Quaternion::new(1_f64, 1_f64, 1_f64, 1_f64);
        let q2 = Quaternion::new(2_f64, 2_f64, 2_f64, 2_f64);
        
        let result = q1.nlerp(q2, 0_f64);
        let expected = q1.normalize();
        assert_eq!(result, expected);

        let result = q1.nlerp(q2, 1_f64);
        let expected = q2.normalize();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_lerp_should_interpolate_to_endoints() {
        let q1 = Quaternion::new(1_f64, 1_f64, 1_f64, 1_f64);
        let q2 = Quaternion::new(2_f64, 2_f64, 2_f64, 2_f64);
        
        let result = q1.lerp(q2, 0_f64);
        assert_eq!(result, q1);

        let result = q1.lerp(q2, 1_f64);
        assert_eq!(result, q2);
    }
}


#[cfg(test)]
mod arithmetic_tests {
    use super::Quaternion;
    use vector::Vector3;
    use structure::One;


    #[test]
    fn test_unit_axis_quaternions() {
        let i = Quaternion::from_sv(0_f64, Vector3::unit_x());
        let j = Quaternion::from_sv(0_f64, Vector3::unit_y());
        let k = Quaternion::from_sv(0_f64, Vector3::unit_z());

        let result_i = 4_f64 * i;
        let expected_i = Quaternion::from_sv(0_f64, Vector3::new(4_f64, 0_f64, 0_f64));
        let result_j = 4_f64 * j;
        let expected_j = Quaternion::from_sv(0_f64, Vector3::new(0_f64, 4_f64, 0_f64));
        let result_k = 4_f64 * k;
        let expected_k = Quaternion::from_sv(0_f64, Vector3::new(0_f64, 0_f64, 4_f64));

        assert_eq!(result_i, expected_i);
        assert_eq!(result_j, expected_j);
        assert_eq!(result_k, expected_k);
    }

    #[test]
    fn test_quaternion_addition() {
        let q1 = Quaternion::new(1, 2, 3, 4);
        let q2 = Quaternion::new(5, 6, 7, 8);
        let expected = Quaternion::new(6, 8, 10, 12);
        let result = q1 + q2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_subtraction() {
        let q1 = Quaternion::new(1, 2, 3, 4);
        let q2 = Quaternion::new(5, 6, 7, 8);
        let expected = Quaternion::new(-4, -4, -4, -4);
        let result = q1 - q2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_unit_squares() {
        let i = Quaternion::from_sv(0_f64, Vector3::unit_x());
        let j = Quaternion::from_sv(0_f64, Vector3::unit_y());
        let k = Quaternion::from_sv(0_f64, Vector3::unit_z());
        let minus_one = -Quaternion::one();

        assert_eq!(i * i, minus_one);
        assert_eq!(j * j, minus_one);
        assert_eq!(k * k, minus_one);
    }

    #[test]
    fn test_quaternion_product_of_all_unit_axis_quaternions() {
        let i = Quaternion::from_sv(0_f64, Vector3::unit_x());
        let j = Quaternion::from_sv(0_f64, Vector3::unit_y());
        let k = Quaternion::from_sv(0_f64, Vector3::unit_z());
        let minus_one = -Quaternion::one();

        assert_eq!(i * j * k, minus_one);
    }

    #[test]
    fn test_quaternion_unit_products() {
        let i = Quaternion::from_sv(0_f64, Vector3::unit_x());
        let j = Quaternion::from_sv(0_f64, Vector3::unit_y());
        let k = Quaternion::from_sv(0_f64, Vector3::unit_z());

        assert_eq!(i * j, k);
        assert_eq!(j * i, -k);
        assert_eq!(j * k, i);
        assert_eq!(k * j, -i);
        assert_eq!(k * i, j);
        assert_eq!(i * k, -j);
    }
}

#[cfg(test)]
mod magnitude_tests {
    use super::Quaternion;
    use structure::Magnitude;
    use vector::Vector3;
    use approx::relative_eq;


    #[test]
    fn unit_axis_quaternions_should_have_unit_norms() {
        let i = Quaternion::from_sv(0_f64, Vector3::unit_x());
        let j = Quaternion::from_sv(0_f64, Vector3::unit_y());
        let k = Quaternion::from_sv(0_f64, Vector3::unit_z());
    
        assert_eq!(i.magnitude(), 1_f64);
        assert_eq!(j.magnitude(), 1_f64);
        assert_eq!(k.magnitude(), 1_f64);
    }

    #[test]
    fn test_quaternion_magnitude() {
        let q = Quaternion::from_sv(3_f64, Vector3::new(34.8, 75.1939, 1.0366));
        let result = q.magnitude_squared();
        let expected = 6875.23713677;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_normalized() {
        let q = Quaternion::from_sv(3_f64, Vector3::new(34.8, 75.1939, 1.0366));
        let result = q.normalize().magnitude();
        let expected = 1_f64;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_normalized_to() {
        let q = Quaternion::from_sv(3_f64, Vector3::new(34.8, 75.1939, 1.0366));
        let magnitude = 12_f64;
        let result = q.normalize_to(magnitude).magnitude();
        let expected = magnitude;
        let tolerance = 1e-7;

        assert!(relative_eq!(result, expected, epsilon = tolerance));
    }
}

#[cfg(test)]
mod slerp_tests {
    use super::Quaternion;
    use approx::relative_eq;
    use structure::Angle;
    use angle::Degrees;
    use vector::Vector3;


    #[test]
    fn test_slerp_upper_right_quadrant() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(60_f64);
        let unit_z = Vector3::unit_z();
        let mut q1 = Quaternion::from_sv(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q2 = Quaternion::from_sv(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );
        let angle_expected = Degrees(45_f64);
        let expected = Quaternion::from_sv(
            Angle::cos(angle_expected / 2_f64), 
            Angle::sin(angle_expected / 2_f64) * unit_z
        );
        let result = q1.slerp(&q2, 0.5);

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }

    #[test]
    fn test_slerp_upper_right_quadrant1() {
        let angle1 = Degrees(20_f64);
        let angle2 = Degrees(70_f64);
        let unit_z = Vector3::unit_z();
        let mut q1 = Quaternion::from_sv(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q2 = Quaternion::from_sv(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );
        let angle_expected = Degrees(30_f64);
        let expected = Quaternion::from_sv(
            Angle::cos(angle_expected / 2_f64), 
            Angle::sin(angle_expected / 2_f64) * unit_z
        );
        let result = q1.slerp(&q2, 0.2);

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }

    #[test]
    fn test_slerp_upper_half_plane() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(150_f64);
        let unit_z = Vector3::unit_z();
        let mut q1 = Quaternion::from_sv(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q2 = Quaternion::from_sv(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );
        let angle_expected = Degrees(90_f64);
        let expected = Quaternion::from_sv(
            Angle::cos(angle_expected / 2_f64), 
            Angle::sin(angle_expected / 2_f64) * unit_z
        );
        let result = q1.slerp(&q2, 0.5);

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }

    #[test]
    fn test_slerp_negative_dot_product() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(240_f64);
        let unit_z = Vector3::unit_z();
        let mut q1 = Quaternion::from_sv(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q2 = Quaternion::from_sv(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );
        let angle_expected = Degrees(315_f64);
        let expected = Quaternion::from_sv(
            Angle::cos(angle_expected / 2_f64), 
            Angle::sin(angle_expected / 2_f64) * unit_z
        );
        let result = q1.slerp(&q2, 0.5);

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }
}

