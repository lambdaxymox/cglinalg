use crate::normed::Normed;
use crate::unit::Unit;
use cglinalg_complex::Complex;
use cglinalg_numeric::SimdScalarFloat;

use core::ops;


impl<S> Normed for Complex<S>
where
    S: SimdScalarFloat,
{
    type Output = S;

    fn norm_squared(&self) -> Self::Output {
        self.modulus_squared()
    }

    fn norm(&self) -> Self::Output {
        self.modulus()
    }

    fn scale(&self, scale: Self::Output) -> Self {
        self * scale
    }

    #[inline]
    fn scale_mut(&mut self, scale: Self::Output) {
        *self = self.scale(scale);
    }

    #[inline]
    fn unscale(&self, scale: Self::Output) -> Self {
        self * (Self::Output::one() / scale)
    }

    #[inline]
    fn unscale_mut(&mut self, scale: Self::Output) {
        *self = self.unscale(scale);
    }

    fn normalize(&self) -> Self {
        self * (Self::Output::one() / self.modulus())
    }

    fn normalize_mut(&mut self) -> Self::Output {
        let norm = self.modulus();
        *self = self.normalize();

        norm
    }

    fn try_normalize(&self, threshold: Self::Output) -> Option<Self> {
        let norm = self.modulus();
        if norm <= threshold {
            None
        } else {
            Some(self.normalize())
        }
    }

    fn try_normalize_mut(&mut self, threshold: Self::Output) -> Option<Self::Output> {
        let norm = self.modulus();
        if norm <= threshold {
            None
        } else {
            Some(self.normalize_mut())
        }
    }

    fn distance_squared(&self, other: &Self) -> Self::Output {
        (self - other).modulus_squared()
    }

    fn distance(&self, other: &Self) -> Self::Output {
        (self - other).modulus()
    }
}

impl<S> ops::Neg for Unit<Complex<S>>
where
    S: SimdScalarFloat,
{
    type Output = Unit<Complex<S>>;

    #[inline]
    fn neg(self) -> Self::Output {
        Unit::from_value_unchecked(-self.into_inner())
    }
}

impl<S> ops::Neg for &Unit<Complex<S>>
where
    S: SimdScalarFloat,
{
    type Output = Unit<Complex<S>>;

    #[inline]
    fn neg(self) -> Self::Output {
        Unit::from_value_unchecked(-self.into_inner())
    }
}
