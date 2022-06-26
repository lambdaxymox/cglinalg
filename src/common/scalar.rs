use num_traits::{
    Num, 
    NumCast, 
    Float
};
use core::fmt::{
    Debug,
};
use core::ops::{
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    Neg,
};


/// A data type with this trait has the properties of a 
/// set of scalar numbers underlying vector and matrix 
/// data types.
pub trait SimdScalar 
where
    Self: Copy,
    Self: Clone,
    Self: Debug,
    Self: Num,
    Self: NumCast,
    Self: PartialOrd, 
    Self: AddAssign,
    Self: SubAssign,
    Self: MulAssign,
    Self: DivAssign,
    Self: RemAssign,
{
}

impl<T> SimdScalar for T 
where 
    T: Copy
     + Clone 
     + Debug 
     + Num 
     + NumCast 
     + PartialOrd 
     + AddAssign 
     + SubAssign 
     + MulAssign 
     + DivAssign 
     + RemAssign 
{ 
}

/// Scalar numbers with a notion of subtraction and have additive 
/// inverses. 
pub trait SimdScalarSigned 
where 
    Self: SimdScalar + Neg<Output = Self> 
{
}

impl<T> SimdScalarSigned for T 
where 
    T: SimdScalar + Neg<Output = T> 
{
}

/// Scalar numbers that have the properties of finite precision
/// floating point arithmetic.
pub trait SimdScalarFloat:
      SimdScalar
    + Float
    + approx::AbsDiffEq<Epsilon = Self>
    + approx::RelativeEq<Epsilon = Self>
    + approx::UlpsEq<Epsilon = Self>
{
}

impl<T> SimdScalarFloat for T 
where 
    T: SimdScalar 
     + Float
     + approx::AbsDiffEq<Epsilon = Self>
     + approx::RelativeEq<Epsilon = Self>
     + approx::UlpsEq<Epsilon = Self>
{
}

