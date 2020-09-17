pub extern crate approx;
pub extern crate num_traits;


mod scalar;
mod angle;
mod prelude;
mod matrix;
mod projection;
mod point;
mod quaternion;
mod vector;
mod traits;

pub use prelude::*;
pub use angle::*;
pub use matrix::*;
pub use projection::*;
pub use quaternion::*;
pub use scalar::*;
pub use vector::*;
pub use traits::*;
pub use point::*;


mod affine;

pub use affine::*;

// Make the glm module a public module imported separately to emphasize that 
// it is optional.
pub mod glm;

