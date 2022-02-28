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
    S: Clone + num_traits::Num
{
    #[inline]
    pub fn i() -> Self {
        Self::new(S::zero(), S::one())
    }

    #[inline]
    pub fn magnitude_squared(&self) -> S {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }
}

impl<S> Complex<S>
where
    S: Clone + num_traits::Num + ops::Neg<Output = S>
{
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self::new(self.re.clone(), -self.im.clone())
    }

    #[inline]
    pub fn scale(&self, scale: S) -> Self {
        Self::new(
            self.re.clone() * scale.clone(),
            self.im.clone() * scale
        )
    }

    #[inline]
    pub fn unscale(&self, scale: S) -> Self {
        let one_over_scale = S::one() / scale;

        Self::new(
            self.re.clone() * one_over_scale.clone(), 
            self.im.clone() * one_over_scale
        )
    }

    #[inline]
    pub fn inverse(&self) -> Self {
        let magnitude_squared = self.magnitude_squared();
        Self::new(
             self.re.clone() / magnitude_squared.clone(),
            -self.im.clone() / magnitude_squared
        )
    }

    #[inline]
    pub fn powi(&self, power: i32) -> Self {
        unimplemented!()
    }

    #[inline]
    pub fn powu(&self, power: u32) -> Self {
        unimplemented!()
    }
}

impl<S> Complex<S>
where
    S: num_traits::Float
{
    #[inline]
    pub fn magnitude(&self) -> S {
        self.magnitude_squared().sqrt()
    }

    #[inline]
    pub fn arg(&self) -> S {
        self.im.atan2(self.re)
    }
}

impl<S> Complex<S>
where
    S: ScalarFloat
{
    #[inline]
    pub fn from_polar<A: Into<Radians<S>>>(r: S, angle: A) -> Self {
        let theta: Radians<S> = angle.into();
        Self::new(r * theta.cos(), r * theta.sin())
    }

    #[inline]
    pub fn to_polar(&self) -> (S, Radians<S>) {
        (self.magnitude(), Radians(self.arg()))
    }
}

