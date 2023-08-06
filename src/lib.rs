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

extern crate cglinalg_core;
extern crate cglinalg_transform;


pub use cglinalg_core::*;
pub use cglinalg_transform::*;

