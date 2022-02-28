use crate::base::{
    Magnitude,
    Scalar,
    ScalarSigned,
    ScalarFloat,
};
use crate::angle::{
    Angle,
    Radians,
};
use core::ops;


#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Complex<S> {
    pub re: S,
    pub im: S,
}

impl<S> Complex<S> {
    #[inline]
    pub const fn new(re: S, im: S) -> Self {
        Self {
            re: re,
            im: im,
        }   
    }
}

impl<S> Complex<S>
where
    S: Scalar
{
    #[inline]
    pub fn i() -> Self {
        Self::new(S::zero(), S::one())
    }

    #[inline]
    pub fn from_real(value: S) -> Self {
        Self::new(value, S::zero())
    }

    #[inline]
    pub fn from_imaginary(value: S) -> Self {
        Self::new(S::zero(), value)
    }

    #[inline]
    pub fn zero() -> Self {
        Self::new(S::zero(), S::zero())
    }

    #[inline]
    pub fn identity() -> Self {
        Self::new(S::one(), S::zero())
    }

    #[inline]
    pub fn magnitude_squared(self) -> S {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }
}

impl<S> Complex<S>
where
    S: ScalarSigned
{
    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(self.re.clone(), -self.im.clone())
    }

    #[inline]
    pub fn scale(self, scale: S) -> Self {
        Self::new(
            self.re.clone() * scale.clone(),
            self.im.clone() * scale
        )
    }

    #[inline]
    pub fn unscale(self, scale: S) -> Self {
        let one_over_scale = S::one() / scale;

        Self::new(
            self.re.clone() * one_over_scale.clone(), 
            self.im.clone() * one_over_scale
        )
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let magnitude_squared = self.magnitude_squared();
        Self::new(
             self.re.clone() / magnitude_squared.clone(),
            -self.im.clone() / magnitude_squared
        )
    }

    #[inline]
    pub fn powi(self, power: i32) -> Self {
        unimplemented!()
    }

    #[inline]
    pub fn powu(self, power: u32) -> Self {
        unimplemented!()
    }
}

impl<S> Complex<S>
where
    S: ScalarFloat
{
    #[inline]
    pub fn magnitude(self) -> S {
        self.magnitude_squared().sqrt()
    }

    #[inline]
    pub fn arg(self) -> S {
        self.im.atan2(self.re)
    }

    #[inline]
    pub fn from_polar<A: Into<Radians<S>>>(radius: S, angle: A) -> Self {
        let _angle: Radians<S> = angle.into();
        Self::new(radius * _angle.cos(), radius * _angle.sin())
    }

    #[inline]
    pub fn from_angle<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_polar(S::one(), angle)
    }

    #[inline]
    pub fn to_polar(self) -> (S, Radians<S>) {
        (self.magnitude(), Radians(self.arg()))
    }

    #[inline]
    pub fn exp(self) -> Self {
        let exp_re = self.re.exp();
        let (sin_im, cos_im) = self.im.sin_cos();

        Self::new(exp_re * cos_im, exp_re * sin_im)
    }

    /// Calculate the principal value of the natural logarithm of a complex number.
    #[inline]
    pub fn ln(self) -> Self {
        let magnitude_self = self.magnitude();
        let arg_self = self.arg();

        Self::new(magnitude_self.ln(), arg_self)
    }

    /// Calculate the principal value of the square root of a complex number.
    #[inline]
    pub fn sqrt(self) -> Self {
        let two = S::one() + S::one();
        let magnitude = self.magnitude();
        let angle = self.arg();
        let (sin_angle_over_two, cos_angle_over_two) = (angle / two).sin_cos();

        Self::new(magnitude * cos_angle_over_two, magnitude * sin_angle_over_two)
    }
}

impl<S> ops::Add<Complex<S>> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: Complex<S>) -> Self::Output {
        Self::Output::new(self.re + other.re, self.im + other.im)
    }
}

impl<S> ops::Add<&Complex<S>> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: &Complex<S>) -> Self::Output {
        Self::Output::new(self.re + other.re, self.im + other.im)
    }
}

impl<S> ops::Add<Complex<S>> for &Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: Complex<S>) -> Self::Output {
        Self::Output::new(self.re + other.re, self.im + other.im)
    }
}

impl<'a, 'b, S> ops::Add<&'b Complex<S>> for &'a Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: &'b Complex<S>) -> Self::Output {
        Self::Output::new(self.re + other.re, self.im + other.im)
    }
}

impl<S> ops::Add<S> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: S) -> Self::Output {
        Self::Output::new(self.re + other, self.im)
    }
}

impl<S> ops::Add<&S> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: &S) -> Self::Output {
        Self::Output::new(self.re + *other, self.im)
    }
}

impl<S> ops::Add<S> for &Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: S) -> Self::Output {
        Self::Output::new(self.re + other, self.im)
    }
}

impl<'a, 'b, S> ops::Add<&'b S> for &'a Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: &'b S) -> Self::Output {
        Self::Output::new(self.re + *other, self.im)
    }
}

impl<S> ops::Sub<Complex<S>> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: Complex<S>) -> Self::Output {
        Self::Output::new(self.re - other.re, self.im - other.im)
    }
}

impl<S> ops::Sub<&Complex<S>> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: &Complex<S>) -> Self::Output {
        Self::Output::new(self.re - other.re, self.im - other.im)
    }
}

impl<S> ops::Sub<Complex<S>> for &Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: Complex<S>) -> Self::Output {
        Self::Output::new(self.re - other.re, self.im - other.im)
    }
}

impl<'a, 'b, S> ops::Sub<&'b Complex<S>> for &'a Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: &'b Complex<S>) -> Self::Output {
        Self::Output::new(self.re - other.re, self.im - other.im)
    }
}

impl<S> ops::Sub<S> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: S) -> Self::Output {
        Self::Output::new(self.re - other, self.im)
    }
}

impl<S> ops::Sub<&S> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: &S) -> Self::Output {
        Self::Output::new(self.re - *other, self.im)
    }
}

impl<S> ops::Sub<S> for &Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: S) -> Self::Output {
        Self::Output::new(self.re - other, self.im)
    }
}

impl<'a, 'b, S> ops::Sub<&'b S> for &'a Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: &'b S) -> Self::Output {
        Self::Output::new(self.re - *other, self.im)
    }
}

