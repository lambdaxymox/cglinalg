use structure::{
    Angle,
    Storage,
    Zero,
    One,
    ProjectOn,
    DotProduct,
    Magnitude,
    Lerp,
    Nlerp,
    Metric,
    Finite,
    Slerp,
};
use angle::{
    Radians,
    Degrees,
};
use base::{
    Scalar,
    ScalarSigned,
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
    pub s: S,
    /// The vector component.
    pub v: Vector3<S>,
}

impl<S> Quaternion<S> where S: Scalar {
    /// Construct a new quaternion.
    #[inline]
    pub fn new(s: S, x: S, y: S, z: S) -> Quaternion<S> {
        Self::from_sv(s, Vector3::new(x, y, z))
    }

    /// Compute a quaternion from its scalar and vector parts.
    #[inline]
    pub fn from_sv(s: S, v: Vector3<S>) -> Quaternion<S> {
        Quaternion { s: s, v: v }
    }

    /// Returns the unit real quaternion. 
    ///
    /// A real vector quaternion is a quaternion with zero vector part.
    pub fn unit_s() -> Quaternion<S> {
        Quaternion::from_sv(S::one(), Vector3::zero())
    }

    /// The unit pure quaternion representing the x-axis.
    ///
    /// A pure quaternion is a quaternion with zero scalar part.
    pub fn unit_x() -> Quaternion<S> {
        Quaternion::from_sv(S::zero(), Vector3::new(S::one(), S::zero(), S::zero()))
    }

    /// The unit pure quaternion representing the y-axis.
    ///
    /// A pure quaternion is a quaternion with zero scalar part.
    pub fn unit_y() -> Quaternion<S> {
        Quaternion::from_sv(S::zero(), Vector3::new(S::zero(), S::one(), S::zero()))
    }

    /// The unit pure quaternion representing the z-axis.
    ///
    /// A pure quaternion is a quaternion with zero scalar part.
    pub fn unit_z() -> Quaternion<S> {
        Quaternion::from_sv(S::zero(), Vector3::new(S::zero(), S::zero(), S::one()))
    }

    /// Convert a quaternion to its equivalent matrix form writing the terms into preallocated storage.
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
    ///
    /// Returns `true` is there exists a quaternion `r` such that `q * r = 1`.
    pub fn is_invertible(&self) -> bool {
        self.magnitude_squared() > S::zero()
    }

    /// Compute the principal argument of a quaternion.
    ///
    /// Every quaternion can be written in polar form. Let `q` be a quaternion and let 
    /// `q = qs + qv` where `qs` is the scalar part of `q` and `qv` is the vector part of `q`.
    /// The polar form of q can be written as
    /// ```text
    /// q = |q| * (cos(theta) + (qv / |qv|) * sin(theta))
    /// ```
    /// The argument of `q` is the set of angles `theta` that satisfy the relation above which 
    /// we denote `arg(q)`. The principal argument of `q` is the angle `theta` satisfying the 
    /// polar decomposition of `q` above such that `theta` lies in the closed interval `[0, pi]`. 
    /// For each elemenet of `theta` of `arg(q)`, there is an integer `n` such that
    /// ```text
    /// theta = Arg(q) + 2 * pi * n
    /// ```
    /// In the case of `theta = Arg(q)`, we have `n = 0`.
    pub fn arg(&self) -> S {
        if self.s == S::zero() {
            num_traits::cast(std::f64::consts::FRAC_PI_2).unwrap()
        } else {
            S::atan(self.v.magnitude() / self.s)
        }
    }

    /// Calculate the exponential of a quaternion.
    pub fn exp(&self) -> Quaternion<S> {
        let magnitude_v = self.v.magnitude();
        if magnitude_v == S::zero() {
            Quaternion::from_sv(self.s.exp(), Vector3::zero())
        } else {
            let exp_s = self.s.exp();
            let q_scalar = exp_s * S::cos(magnitude_v);
            let q_vector = self.v * (exp_s * S::sin(magnitude_v) / magnitude_v);
            
            Quaternion::from_sv(q_scalar, q_vector)
        }
    }

    /// Calculate the principal value of the natural logarithm of a quaternion.
    ///
    /// Just like the complex numbers, the natural logarithm of a quaternion has multiple possible
    /// values, so we return the principal value of the quaternion logarithm, defined by
    /// ```text
    /// Ln(q) = log(||q||, e) + sgn(Vec(q)) * Arg(q)
    /// ```
    /// where `Arg(q)` is the principal argument of `q`, `||q||` is the magnitude of `q`, 
    /// `Vec(q)` is the vector part of `q`, `sgn(.)` is the signum function, and `log(., e)`
    /// denotes the natural logarithm of a scalar. Returning the principal value allows us to
    /// define a unique natural logarithm for each quaternion `q`.
    pub fn ln(&self) -> Quaternion<S> {
        let magnitude_v = self.v.magnitude();
        if magnitude_v == S::zero() {
            Quaternion::from_sv(self.s.ln(), Vector3::zero())
        } else {
            let magnitude_q = self.magnitude();
            let arccos_s_over_mag_q = S::acos(self.s / magnitude_q);
            let q_scalar = S::ln(magnitude_q);
            let q_vector = self.v * (arccos_s_over_mag_q / magnitude_v);

            Quaternion::from_sv(q_scalar, q_vector)
        }
    }

    /// Calculate the power of a quaternion where the exponent is a real number.
    pub fn powf(&self, exponent: S) -> Quaternion<S> {
        let magnitude_v = self.v.magnitude();
        if (self.s == S::zero()) && (magnitude_v == S::zero()) {
            Quaternion::zero()
        } else if (self.s == S::zero()) && (magnitude_v != S::zero()) {
            let magnitude_q = self.magnitude();
            let magnitude_q_pow = magnitude_q.powf(exponent);
            let angle: S = num_traits::cast(std::f64::consts::FRAC_PI_2).unwrap();
            let q_scalar = magnitude_q_pow * S::cos(exponent * angle);
            let q_vector = self.v * (magnitude_q_pow * S::sin(exponent * angle) / magnitude_v);

            Quaternion::from_sv(q_scalar, q_vector)
        } else if (self.s != S::zero()) && (magnitude_v == S::zero()) {
            let magnitude_q = self.magnitude();
            let magnitude_q_pow = magnitude_q.powf(exponent);
            let angle = S::zero();
            let q_scalar = magnitude_q_pow * S::cos(exponent * angle);
            let q_vector = self.v * (magnitude_q_pow * S::sin(exponent * angle) / magnitude_v);

            Quaternion::from_sv(q_scalar, q_vector)
        } else {
            let magnitude_q = self.magnitude();
            let magnitude_q_pow = magnitude_q.powf(exponent);
            let angle = S::atan(magnitude_v / self.s);
            let q_scalar = magnitude_q_pow * S::cos(exponent * angle);
            let q_vector = self.v * (magnitude_q_pow * S::sin(exponent * angle) / magnitude_v);

            Quaternion::from_sv(q_scalar, q_vector)
        }
    }

    /// Compute the principal value of a quaternion raised to the power of another quaternion.
    pub fn powq(&self, exponent: Quaternion<S>) -> Quaternion<S> {
        Self::exp(&(self.ln() * exponent))
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

impl<S> Storage for Quaternion<S> where S: Scalar {
    type Element = S;

    #[inline]
    fn len() -> usize {
        4
    }

    #[inline]
    fn shape() -> (usize, usize) {
        (4, 1)
    }

    #[inline]
    fn from_value(value: Self::Element) -> Self {
        Quaternion::new(S::zero(), value, value, value)
    }

    #[inline]
    fn as_ptr(&self) -> *const Self::Element {
        &self.s
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Self::Element {
        &mut self.s
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Element] {
        <Self as AsRef<[Self::Element; 4]>>::as_ref(self)
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

impl<S> ops::Neg for Quaternion<S> where S: ScalarSigned {
    type Output = Quaternion<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Quaternion::from_sv(-self.s, -self.v)
    }
}

impl<'a, S> ops::Neg for &'a Quaternion<S> where S: ScalarSigned {
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


impl<S> Slerp<Quaternion<S>> for Quaternion<S> where S: ScalarFloat {
    type Scalar = S;
    type Output = Quaternion<S>;

    /// Spherically linearly interpolate between two unit quaternions.
    ///
    /// In the case where the angle between quaternions is `180 degrees`, the
    /// slerp function is not well defined because we can rotate about any axis
    /// normal to the plane swept out by the quaternions to get from one to the other.
    /// The vector normal to the quaternions is not unique in this case.
    fn slerp(self, other: Quaternion<S>, amount: S) -> Quaternion<S> {
        let zero = S::zero();
        let one = S::one();
        // There are two possible routes along a great circle arc between two quaternions on the three-sphere.
        // By definition the slerp function computes the shortest path between two points on a sphere, hence
        // we must determine which of two directions around the great circle arc swept out by the slerp
        // function is the shortest one. 
        let (result, cos_half_theta) = if self.dot(other) < zero {
            // If the dot product is negative, the shortest path between two points on the great circle arc
            // swept out by the quaternions runs in the opposite direction from the positive case, so we
            // must negate one of the quaterions to take the short way around instead of the long way around.
            let _result = self * -one;
            (_result, (_result).dot(other))
        } else {
            let _result = self;
            (_result, _result.dot(other))
        };

        // We have two opportunities for performance optimizations:
        //
        // If `result` == `other`, there is no curve to interpolate; the angle between `result` and 
        // `other` is zero. In this case we can return `result`.
        if S::abs(cos_half_theta) >= one {
            return result;
        }

        // If `result` == `-other` then the angle between them is 180 degrees.
        // In this case the slerp function is not well defined because we can 
        // rotate around any axis normal to the plane swept out by `result` and `other`.
        //
        // For very small angles, `sin_half_theta` is approximately equal to the angle `half_theta`.
        // That is, `sin(theta / 2) ~= theta / 2` as `theta -> 0`.
        // Therefore, we can use the sine of the angle between two quaternions to determine when to
        // approximate spherical linear interpolation with normalized linear interpolation.
        // Using the sine of the angle is also cheaper to calculate since we can derive it from the
        // cosine we already calculated instead of calculating the angle from an inverse trigonometric
        // function.
        let sin_half_theta = S::sqrt(one - cos_half_theta * cos_half_theta);
        let threshold = num_traits::cast(0.001).unwrap();
        if S::abs(sin_half_theta) < threshold {
            return result.nlerp(other, amount);
        }
        
        let half_theta = S::acos(cos_half_theta);
        let a = S::sin((one - amount) * half_theta) / sin_half_theta;
        let b = S::sin(amount * half_theta) / sin_half_theta;
        
        let s   = result.s   * a + other.s   * b;
        let v_x = result.v.x * a + other.v.x * b;
        let v_y = result.v.y * a + other.v.y * b;
        let v_z = result.v.z * a + other.v.z * b;

        Quaternion::new(s, v_x, v_y, v_z)
    }
}

impl<'a, S> Slerp<&'a Quaternion<S>> for Quaternion<S> where S: ScalarFloat {
    type Scalar = S;
    type Output = Quaternion<S>;

    /// Spherically linearly interpolate between two unit quaternions.
    ///
    /// In the case where the angle between quaternions is `180 degrees`, the
    /// slerp function is not well defined because we can rotate about any axis
    /// normal to the plane swept out by the quaternions to get from one to the other.
    /// The vector normal to the quaternions is not unique in this case.
    fn slerp(self, other: &'a Quaternion<S>, amount: S) -> Quaternion<S> {
        let zero = S::zero();
        let one = S::one();
        // There are two possible routes along a great circle arc between two quaternions on the three-sphere.
        // By definition the slerp function computes the shortest path between two points on a sphere, hence
        // we must determine which of two directions around the great circle arc swept out by the slerp
        // function is the shortest one. 
        let (result, cos_half_theta) = if self.dot(other) < zero {
            // If the dot product is negative, the shortest path between two points on the great circle arc
            // swept out by the quaternions runs in the opposite direction from the positive case, so we
            // must negate one of the quaterions to take the short way around instead of the long way around.
            let _result = self * -one;
            (_result, (_result).dot(other))
        } else {
            let _result = self;
            (_result, _result.dot(other))
        };

        // We have two opportunities for performance optimizations:
        //
        // If `result` == `other`, there is no curve to interpolate; the angle between `result` and 
        // `other` is zero. In this case we can return `result`.
        if S::abs(cos_half_theta) >= one {
            return result;
        }

        // If `result` == `-other` then the angle between them is 180 degrees.
        // In this case the slerp function is not well defined because we can 
        // rotate around any axis normal to the plane swept out by `result` and `other`.
        //
        // For very small angles, `sin_half_theta` is approximately equal to the angle `half_theta`.
        // That is, `sin(theta / 2) ~= theta / 2` as `theta -> 0`.
        // Therefore, we can use the sine of the angle between two quaternions to determine when to
        // approximate spherical linear interpolation with normalized linear interpolation.
        // Using the sine of the angle is also cheaper to calculate since we can derive it from the
        // cosine we already calculated instead of calculating the angle from an inverse trigonometric
        // function.
        let sin_half_theta = S::sqrt(one - cos_half_theta * cos_half_theta);
        let threshold = num_traits::cast(0.001).unwrap();
        if S::abs(sin_half_theta) < threshold {
            return result.nlerp(other, amount);
        }
        
        let half_theta = S::acos(cos_half_theta);
        let a = S::sin((one - amount) * half_theta) / sin_half_theta;
        let b = S::sin(amount * half_theta) / sin_half_theta;
        
        let s   = result.s   * a + other.s   * b;
        let v_x = result.v.x * a + other.v.x * b;
        let v_y = result.v.y * a + other.v.y * b;
        let v_z = result.v.z * a + other.v.z * b;

        Quaternion::new(s, v_x, v_y, v_z)
    }
}

impl<'a, S> Slerp<Quaternion<S>> for &'a Quaternion<S> where S: ScalarFloat {
    type Scalar = S;
    type Output = Quaternion<S>;

    /// Spherically linearly interpolate between two unit quaternions.
    ///
    /// In the case where the angle between quaternions is `180 degrees`, the
    /// slerp function is not well defined because we can rotate about any axis
    /// normal to the plane swept out by the quaternions to get from one to the other.
    /// The vector normal to the quaternions is not unique in this case.
    fn slerp(self, other: Quaternion<S>, amount: S) -> Quaternion<S> {
        let zero = S::zero();
        let one = S::one();
        // There are two possible routes along a great circle arc between two quaternions on the three-sphere.
        // By definition the slerp function computes the shortest path between two points on a sphere, hence
        // we must determine which of two directions around the great circle arc swept out by the slerp
        // function is the shortest one. 
        let (result, cos_half_theta) = if self.dot(other) < zero {
            // If the dot product is negative, the shortest path between two points on the great circle arc
            // swept out by the quaternions runs in the opposite direction from the positive case, so we
            // must negate one of the quaterions to take the short way around instead of the long way around.
            let _result = self * -one;
            (_result, (_result).dot(other))
        } else {
            let _result = *self;
            (_result, _result.dot(other))
        };

        // We have two opportunities for performance optimizations:
        //
        // If `result` == `other`, there is no curve to interpolate; the angle between `result` and 
        // `other` is zero. In this case we can return `result`.
        if S::abs(cos_half_theta) >= one {
            return result;
        }

        // If `result` == `-other` then the angle between them is 180 degrees.
        // In this case the slerp function is not well defined because we can 
        // rotate around any axis normal to the plane swept out by `result` and `other`.
        //
        // For very small angles, `sin_half_theta` is approximately equal to the angle `half_theta`.
        // That is, `sin(theta / 2) ~= theta / 2` as `theta -> 0`.
        // Therefore, we can use the sine of the angle between two quaternions to determine when to
        // approximate spherical linear interpolation with normalized linear interpolation.
        // Using the sine of the angle is also cheaper to calculate since we can derive it from the
        // cosine we already calculated instead of calculating the angle from an inverse trigonometric
        // function.
        let sin_half_theta = S::sqrt(one - cos_half_theta * cos_half_theta);
        let threshold = num_traits::cast(0.001).unwrap();
        if S::abs(sin_half_theta) < threshold {
            return result.nlerp(other, amount);
        }
        
        let half_theta = S::acos(cos_half_theta);
        let a = S::sin((one - amount) * half_theta) / sin_half_theta;
        let b = S::sin(amount * half_theta) / sin_half_theta;
        
        let s   = result.s   * a + other.s   * b;
        let v_x = result.v.x * a + other.v.x * b;
        let v_y = result.v.y * a + other.v.y * b;
        let v_z = result.v.z * a + other.v.z * b;

        Quaternion::new(s, v_x, v_y, v_z)
    }
}

impl<'a, 'b, S> Slerp<&'a Quaternion<S>> for &'b Quaternion<S> where S: ScalarFloat {
    type Scalar = S;
    type Output = Quaternion<S>;

    /// Spherically linearly interpolate between two unit quaternions.
    ///
    /// In the case where the angle between quaternions is `180 degrees`, the
    /// slerp function is not well defined because we can rotate about any axis
    /// normal to the plane swept out by the quaternions to get from one to the other.
    /// The vector normal to the quaternions is not unique in this case.
    fn slerp(self, other: &'a Quaternion<S>, amount: S) -> Quaternion<S> {
        let zero = S::zero();
        let one = S::one();
        // There are two possible routes along a great circle arc between two quaternions on the three-sphere.
        // By definition the slerp function computes the shortest path between two points on a sphere, hence
        // we must determine which of two directions around the great circle arc swept out by the slerp
        // function is the shortest one. 
        let (result, cos_half_theta) = if self.dot(other) < zero {
            // If the dot product is negative, the shortest path between two points on the great circle arc
            // swept out by the quaternions runs in the opposite direction from the positive case, so we
            // must negate one of the quaterions to take the short way around instead of the long way around.
            let _result = self * -one;
            (_result, (_result).dot(other))
        } else {
            let _result = *self;
            (_result, _result.dot(other))
        };

        // We have two opportunities for performance optimizations:
        //
        // If `result` == `other`, there is no curve to interpolate; the angle between `result` and 
        // `other` is zero. In this case we can return `result`.
        if S::abs(cos_half_theta) >= one {
            return result;
        }

        // If `result` == `-other` then the angle between them is 180 degrees.
        // In this case the slerp function is not well defined because we can 
        // rotate around any axis normal to the plane swept out by `result` and `other`.
        //
        // For very small angles, `sin_half_theta` is approximately equal to the angle `half_theta`.
        // That is, `sin(theta / 2) ~= theta / 2` as `theta -> 0`.
        // Therefore, we can use the sine of the angle between two quaternions to determine when to
        // approximate spherical linear interpolation with normalized linear interpolation.
        // Using the sine of the angle is also cheaper to calculate since we can derive it from the
        // cosine we already calculated instead of calculating the angle from an inverse trigonometric
        // function.
        let sin_half_theta = S::sqrt(one - cos_half_theta * cos_half_theta);
        let threshold = num_traits::cast(0.001).unwrap();
        if S::abs(sin_half_theta) < threshold {
            return result.nlerp(other, amount);
        }
        
        let half_theta = S::acos(cos_half_theta);
        let a = S::sin((one - amount) * half_theta) / sin_half_theta;
        let b = S::sin(amount * half_theta) / sin_half_theta;
        
        let s   = result.s   * a + other.s   * b;
        let v_x = result.v.x * a + other.v.x * b;
        let v_y = result.v.y * a + other.v.y * b;
        let v_z = result.v.z * a + other.v.z * b;

        Quaternion::new(s, v_x, v_y, v_z)
    }
}

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

impl<S> ProjectOn<Quaternion<S>> for Quaternion<S> where S: ScalarFloat {
    type Output = Quaternion<S>;

    fn project_on(self, onto: Quaternion<S>) -> Quaternion<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<S> ProjectOn<&Quaternion<S>> for Quaternion<S> where S: ScalarFloat {
    type Output = Quaternion<S>;

    fn project_on(self, onto: &Quaternion<S>) -> Quaternion<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<S> ProjectOn<Quaternion<S>> for &Quaternion<S> where S: ScalarFloat {
    type Output = Quaternion<S>;

    fn project_on(self, onto: Quaternion<S>) -> Quaternion<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

impl<'a, 'b, S> ProjectOn<&'a Quaternion<S>> for &'b Quaternion<S> where S: ScalarFloat {
    type Output = Quaternion<S>;

    fn project_on(self, onto: &'a Quaternion<S>) -> Quaternion<S> {
        onto * (self.dot(onto) / onto.magnitude_squared())
    }
}

