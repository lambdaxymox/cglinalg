use crate::base::{
    Scalar,
    ScalarSigned,
    ScalarFloat,
};


#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Complex<S> {
    re: S,
    im: S,
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

}