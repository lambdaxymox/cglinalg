use num_traits::{
    Num, 
    NumCast, 
    Float
};
use approx;
use std::fmt::{
    Debug,
};
use std::ops::{
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
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

pub trait ScalarFloat:
      Scalar
    + Float
{
}

impl<T> ScalarFloat for T where 
    T: Scalar 
     + Float
     + approx::AbsDiffEq
     + approx::RelativeEq
     + approx::UlpsEq
{
}
