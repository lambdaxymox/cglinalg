#![doc = include_str!("../README.md")]
#![no_std]

#[cfg(feature = "core")]
extern crate core;

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

extern crate approx_cmp;
extern crate num_traits;

extern crate cglinalg_core;
extern crate cglinalg_numeric;
extern crate cglinalg_transform;
extern crate cglinalg_trigonometry;

pub use cglinalg_core::*;
pub use cglinalg_numeric::*;
pub use cglinalg_transform::*;
pub use cglinalg_trigonometry::*;
