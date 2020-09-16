//! This module re-exports all the common traits used in this library. Use this 
//! module by glob importing the prelude as
//! ```text
//! use gdmath::prelude::*;
//! ```
//! and then importing the types you want to work with as usual. This imports all the 
//! common traits to make it more convenient to work with each type. This way
//! you do not have to import each trait individually. 

pub use structure::*;

pub use reflection::Reflection;
pub use reflection::Reflection2;
pub use reflection::Reflection3;

pub use rotation::Rotation;
pub use rotation::Rotation2;
pub use rotation::Rotation3;

pub use scale::Scale;
pub use scale::Scale2;
pub use scale::Scale3;

pub use shear::Shear;
pub use shear::Shear2;
pub use shear::Shear3;

pub use transform::AffineTransformation2D;
pub use transform::AffineTransformation3D;

pub use translation::Translation;
pub use translation::Translation2;
pub use translation::Translation3;

