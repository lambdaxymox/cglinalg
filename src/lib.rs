pub extern crate approx;
pub extern crate num_traits;


mod scalar;
mod angle;
mod euler;
mod prelude;
mod matrix;
mod projection;
mod point;
mod quaternion;
mod vector;
mod traits;

mod reflection;
mod rotation;
mod scale;
mod shear;
mod translation;
mod affine;


pub use prelude::*;
pub use angle::*;
pub use euler::*;
pub use matrix::*;
pub use projection::*;
pub use quaternion::*;
pub use scalar::*;
pub use vector::*;
pub use traits::*;
pub use point::*;

pub use reflection::*;
pub use rotation::*;
pub use scale::*;
pub use shear::*;
pub use translation::*;
pub use affine::*;


// Make the glm module a public module imported separately to emphasize that 
// it is optional.
pub mod glm;

