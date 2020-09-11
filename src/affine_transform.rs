use scalar::{
    Scalar,
    ScalarFloat,
};
use matrix::{
    Matrix3,
    Matrix4,
};
use vector::{
    Vector2,
    Vector3,
};
use structure::{
    One,
};
use std::fmt;

/*
pub trait AffineTransformation<M> {
    fn identity() -> Self;
    fn inverse() -> Option<M>;
    fn concatenate() -> Self;
    fn apply() -> Self;
    fn apply_inverse() -> Option<>;
}
*/

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Identity2D<S> {
    matrix: Matrix3<S>,
}

impl<S> Identity2D<S> where S: Scalar {
    #[inline]
    fn identity() -> Identity2D<S> {
        Identity2D {
            matrix: Matrix3::one(),
        }
    }
}

impl<S> AsRef<Matrix3<S>> for Identity2D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Identity2D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Identity3D<S> {
    matrix: Matrix4<S>,
}

impl<S> Identity3D<S> where S: Scalar {
    #[inline]
    pub fn identity() -> Identity3D<S> {
        Identity3D {
            matrix: Matrix4::one(),
        }
    }
}

impl<S> AsRef<Matrix4<S>> for Identity3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Identity3D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Scale2D<S> {
    matrix: Matrix3<S>,
}

impl<S> Scale2D<S> where S: Scalar {
    #[inline]
    pub fn from_vector(scale: Vector2<S>) -> Scale2D<S> {
        Scale2D {
            matrix: Matrix3::from_nonuniform_scale(scale.x, scale.y),
        }
    }

    #[inline]
    pub fn from_nonuniform_scale(sx: S, sy: S) -> Scale2D<S> {
        Scale2D {
            matrix: Matrix3::from_nonuniform_scale(sx, sy),
        }
    }

    #[inline]
    pub fn from_scale(scale: S) -> Scale2D<S> {
        Scale2D {
            matrix: Matrix3::from_scale(scale),
        }
    }
}

impl<S> AsRef<Matrix3<S>> for Scale2D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Scale2D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Scale3D<S> {
    matrix: Matrix4<S>,
}

impl<S> Scale3D<S> where S: Scalar {
    pub fn from_vector(scale: Vector3<S>) -> Scale3D<S> {
        Scale3D {
            matrix: Matrix4::from_nonuniform_scale(scale.x, scale.y, scale.z),
        }
    }

    #[inline]
    pub fn from_nonuniform_scale(sx: S, sy: S, sz: S) -> Scale3D<S> {
        Scale3D {
            matrix: Matrix4::from_nonuniform_scale(sx, sy, sz),
        }
    }

    #[inline]
    pub fn from_scale(scale: S) -> Scale3D<S> {
        Scale3D {
            matrix: Matrix4::from_scale(scale),
        }
    }
}

impl<S> AsRef<Matrix4<S>> for Scale3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Scale3D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reflection2D<S> {
    matrix: Matrix3<S>,
}

impl<S> Reflection2D<S> where S: ScalarFloat {
    fn from_normal(normal: Vector2<S>) -> Reflection2D<S> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        Reflection2D {
            matrix: Matrix3::new(
                 one - two * normal.x * normal.x, -two * normal.x * normal.y,       zero,
                -two * normal.x * normal.y,        one - two * normal.y * normal.y, zero, 
                 zero,                             zero,                            one
            )
        }
    }
}

impl<S> AsRef<Matrix3<S>> for Reflection2D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Reflection2D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reflection3D<S> {
    matrix: Matrix4<S>,
}

impl<S> Reflection3D<S> where S: ScalarFloat {
    fn from_normal(normal: Vector3<S>) -> Reflection3D<S> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        Reflection3D {
            matrix: Matrix4::new(
                 one - two * normal.x * normal.x, -two * normal.x * normal.y,       -two * normal.x * normal.z,       zero, 
                -two * normal.x * normal.y,        one - two * normal.y * normal.y, -two * normal.y * normal.z,       zero,
                -two * normal.x * normal.z,       -two * normal.y * normal.z,        one - two * normal.z * normal.z, zero,
                 zero,                             zero,                             zero,                            one
            )
        }
    }
}

impl<S> AsRef<Matrix4<S>> for Reflection3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Reflection3D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Translation2D<S> {
    matrix: Matrix3<S>,
}

impl<S> Translation2D<S> where S: Scalar {
    /// Construct a translation operator from a vector of displacements.
    #[inline]
    pub fn from_vector(distance: Vector2<S>) -> Translation2D<S> {
        Translation2D {
            matrix: Matrix3::from_translation(distance),
        }
    }

    /// This function is a synonym for `from_vector`.
    #[inline]
    pub fn from_translation(distance: Vector2<S>) -> Translation2D<S> {
        Translation2D {
            matrix: Matrix3::from_translation(distance),
        }
    }
}

impl<S> AsRef<Matrix3<S>> for Translation2D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Translation2D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Translation3D<S> {
    matrix: Matrix4<S>,
}

impl<S> Translation3D<S> where S: Scalar {
    /// Construct a translation operator from a vector of displacements.
    pub fn from_vector(distance: Vector3<S>) -> Translation3D<S> {
        Translation3D {
            matrix: Matrix4::from_translation(distance),
        }
    }

    /// This function is a synonym for `from_vector`.
    #[inline]
    pub fn from_translation(distance: Vector3<S>) -> Translation3D<S> {
        Translation3D {
            matrix: Matrix4::from_translation(distance),
        }
    }
}

impl<S> AsRef<Matrix4<S>> for Translation3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Translation3D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Shear2D<S> {
    matrix: Matrix3<S>,
}

impl<S> Shear2D<S> where S: Scalar {
    #[inline]
    pub fn from_vector(shear: Vector2<S>) -> Shear2D<S> {
        Shear2D {
            matrix: Matrix3::new(
                S::one(),  shear.y,   S::zero(),
                shear.x,   S::one(),  S::zero(),
                S::zero(), S::zero(), S::one()
            ),
        }
    }

    #[inline]
    pub fn from_shear_x(shear_y: S) -> Shear2D<S> {
        Shear2D {
            matrix: Matrix3::from_shear_x(shear_y, S::zero()),
        }
    }

    #[inline]
    pub fn from_shear_y(shear_x: S) -> Shear2D<S> {
        Shear2D {
            matrix: Matrix3::from_shear_y(shear_x, S::zero()),
        }
    }
}

impl<S> AsRef<Matrix3<S>> for Shear2D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Shear2D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Shear3D<S> {
    matrix: Matrix4<S>,
}

impl<S> Shear3D<S> where S: Scalar {
    #[inline]
    pub fn from_shear_x(shear_y: S, shear_z: S) -> Shear3D<S> {
        Shear3D {
            matrix: Matrix4::from_shear_x(shear_y, shear_z),
        }
    }

    #[inline]
    pub fn from_shear_y(shear_x: S, shear_z: S) -> Shear3D<S> {
        Shear3D {
            matrix: Matrix4::from_shear_y(shear_x, shear_z),
        }
    }

    #[inline]
    pub fn from_shear_z(shear_x: S, shear_y: S) -> Shear3D<S> {
        Shear3D {
            matrix: Matrix4::from_shear_z(shear_x, shear_y),
        }
    }
}

impl<S> AsRef<Matrix4<S>> for Shear3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Shear3D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

