//! This module re-exports all the common traits used in this library. Use this 
//! module by glob importing the prelude as
//! ```text
//! use gdmath::prelude::*;
//! ```
//! and then importing the types you want to work with as usual. This imports 
//! all the common traits to make it more convenient to work with each type. 
//! This way you do not have to import each trait individually. 
pub use crate::traits::*;

pub use crate::rotation::Rotation;
pub use crate::rotation::Rotation2;
pub use crate::rotation::Rotation3;

pub use crate::affine::AffineTransformation2D;
pub use crate::affine::AffineTransformation3D;


