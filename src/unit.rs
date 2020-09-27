use crate::traits::{
    Magnitude,
};
use core::fmt;
use core::mem;
use core::ops::{
    Deref, 
};


/// A type that represents normalized values. This type enforces the requirement that 
/// values have a unit magnitude.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Unit<T> {
    /// The underlying normalized value.
    value: T,
}

impl<T> Unit<T> where T: Magnitude {
    /// Construct a new unit value, normalizing the input value.
    #[inline]
    pub fn new(value: T) -> Self {
        Unit {
            value: value.normalize(),
        }
    }
}

impl<T> Unit<T> {
    /// Unwraps the underlying value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T> AsRef<T> for Unit<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<T> Deref for Unit<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { 
            mem::transmute(&self)
        }
    }
}

impl<T: fmt::Display> fmt::Display for Unit<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.value.fmt(f)
    }
}

