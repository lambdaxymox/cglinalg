#![doc = include_str!("../README.md")]
#![no_std]

#[cfg(feature = "core")]
extern crate core;

#[cfg(feature = "alloc")] 
extern crate alloc;

#[cfg(feature = "std")] 
extern crate std;

extern crate approx;
extern crate num_traits;


mod base_numeric;
mod base;
mod transform;


pub use base_numeric::*;
pub use base::*;
pub use transform::*;

