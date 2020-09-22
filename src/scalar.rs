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


pub trait Scalar where
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

impl<T> Scalar for T where 
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

pub trait ScalarSigned where Self: Scalar + Neg<Output = Self> {}

impl<T> ScalarSigned for T where T: Scalar + Neg<Output = T> {}

pub trait ScalarFloat:
      Scalar
    + Float
    + approx::AbsDiffEq<Epsilon = Self>
    + approx::RelativeEq<Epsilon = Self>
    + approx::UlpsEq<Epsilon = Self>
{
}

impl<T> ScalarFloat for T where 
    T: Scalar 
     + Float
     + approx::AbsDiffEq<Epsilon = Self>
     + approx::RelativeEq<Epsilon = Self>
     + approx::UlpsEq<Epsilon = Self>
{
}
