use scalar::{
    Scalar,
};
use matrix::{
    Matrix3,
    Matrix4,
};
use vector::{
    Vector2,
    Vector3,
};

/*
pub trait AffineTransformation<M> {
    fn identity() -> Self;
    fn inverse() -> Option<M>;
    fn concatenate() -> Self;
    fn apply() -> Self;
    fn apply_inverse() -> Option<>;
}
*/

pub struct Scale2<S> {
    matrix: Matrix3<S>,
}

impl<S> Scale2<S> where S: Scalar {
    #[inline]
    pub fn from_vector(scale: Vector2<S>) -> Scale2<S> {
        Scale2 {
            matrix: Matrix3::from_nonuniform_scale(scale.x, scale.y),
        }
    }

    #[inline]
    pub fn from_nonuniform_scale(sx: S, sy: S) -> Scale2<S> {
        Scale2 {
            matrix: Matrix3::from_nonuniform_scale(sx, sy),
        }
    }

    #[inline]
    pub fn from_scale(scale: S) -> Scale2<S> {
        Scale2 {
            matrix: Matrix3::from_scale(scale),
        }
    }
}

pub struct Scale3<S> {
    matrix: Matrix4<S>,
}

impl<S> Scale3<S> where S: Scalar {
    pub fn from_vector(scale: Vector3<S>) -> Scale3<S> {
        Scale3 {
            matrix: Matrix4::from_nonuniform_scale(scale.x, scale.y, scale.z),
        }
    }

    #[inline]
    pub fn from_nonuniform_scale(sx: S, sy: S, sz: S) -> Scale3<S> {
        Scale3 {
            matrix: Matrix4::from_nonuniform_scale(sx, sy, sz),
        }
    }

    #[inline]
    pub fn from_scale(scale: S) -> Scale3<S> {
        Scale3 {
            matrix: Matrix4::from_scale(scale),
        }
    }
}

pub struct Reflect2<S> {
    matrix: Matrix3<S>,
}

pub struct Reflect3<S> {
    matrix: Matrix4<S>,
}

pub struct Translate2<S> {
    matrix: Matrix3<S>,
}

impl<S> Translate2<S> where S: Scalar {
    /// Construct a translation operator from a vector of displacements.
    #[inline]
    pub fn from_vector(distance: Vector2<S>) -> Translate2<S> {
        Translate2 {
            matrix: Matrix3::from_translation(distance),
        }
    }

    /// This function is a synonym for `from_vector`.
    #[inline]
    pub fn from_translation(distance: Vector2<S>) -> Translate2<S> {
        Translate2 {
            matrix: Matrix3::from_translation(distance),
        }
    }
}

pub struct Translate3<S> {
    matrix: Matrix4<S>,
}

impl<S> Translate3<S> where S: Scalar {
    /// Construct a translation operator from a vector of displacements.
    pub fn from_vector(distance: Vector3<S>) -> Translate3<S> {
        Translate3 {
            matrix: Matrix4::from_translation(distance),
        }
    }

    /// This function is a synonym for `from_vector`.
    #[inline]
    pub fn from_translation(distance: Vector3<S>) -> Translate3<S> {
        Translate3 {
            matrix: Matrix4::from_translation(distance),
        }
    }
}

pub struct Shear2<S> {
    matrix: Matrix3<S>,
}

impl<S> Shear2<S> where S: Scalar {
    #[inline]
    pub fn from_vector(shear: Vector2<S>) -> Shear2<S> {
        Shear2 {
            matrix: Matrix3::new(
                S::one(),  shear.y,   S::zero(),
                shear.x,   S::one(),  S::zero(),
                S::zero(), S::zero(), S::one()
            ),
        }
    }

    #[inline]
    pub fn from_shear_x(shear_y: S) -> Shear2<S> {
        Shear2 {
            matrix: Matrix3::from_shear_x(shear_y, S::zero()),
        }
    }

    #[inline]
    pub fn from_shear_y(shear_x: S) -> Shear2<S> {
        Shear2 {
            matrix: Matrix3::from_shear_y(shear_x, S::zero()),
        }
    }
}
/*
pub struct Shear3<S> {
    matrix: Matrix4<S>,
}

impl<S> Shear3<S> where S: Scalar {
    #[inline]
    pub fn from_shear_x(shear_x: S) -> Shear3<S> {

    }

    #[inline]
    pub fn from_shear_y(shear_y: S) -> Shear3<S> {
        
    }

    #[inline]
    pub fn from_shear_z(shear_z) -> Shear3<S> {

    }
}

*/