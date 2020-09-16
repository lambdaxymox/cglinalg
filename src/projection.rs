use scalar::{
    ScalarFloat,
};
use angle::{
    Degrees, 
    Radians,
};
use matrix::{
    Matrix4x4,
};
use traits::{
    Angle,
};


/// Compute the orthographic projection matrix for converting from camera space to
/// normalized device coordinates.
///
/// This function is equivalent to the now deprecated [glOrtho]
/// (https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml) function.
#[inline]
pub fn orthographic<S, Spec: Into<Orthographic<S>>>(spec: Spec) -> Matrix4x4<S> where S: ScalarFloat {
    Matrix4x4::from(spec.into())
}

/// Compute a perspective matrix from a view frustum.
///
/// This is the equivalent of the now deprecated [glFrustum]
/// (https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml) function.
#[inline]
pub fn frustum<S, Spec: Into<Perspective<S>>>(spec: Spec) -> Matrix4x4<S> where S: ScalarFloat {
    Matrix4x4::from(spec.into())
}

/// Compute the perspective matrix for converting from camera space to 
/// normalized device coordinates. 
///
/// This is the equivalent to the [gluPerspective] 
/// (https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml)
/// function.
#[inline]
pub fn perspective<S, Spec: Into<PerspectiveFov<S>>>(spec: Spec) -> Matrix4x4<S> where S: ScalarFloat {
    Matrix4x4::from(spec.into())
}



/// An orthographic projection with arbitrary left, right, top, bottom,
/// near, and far planes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Orthographic<S> {
    left: S,
    right: S,
    bottom: S,
    top: S,
    near: S,
    far: S,
}

impl<S> Into<Orthographic<S>> for (S, S, S, S, S, S) {
    #[inline]
    fn into(self) -> Orthographic<S> {
        match self {
            (left, right, bottom, top, near, far) => {
                Orthographic {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<S> Into<Orthographic<S>> for &(S, S, S, S, S, S) where S: Copy {
    #[inline]
    fn into(self) -> Orthographic<S> {
        match *self {
            (left, right, bottom, top, near, far) => {
                Orthographic {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<S> Into<Orthographic<S>> for [S; 6] {
    #[inline]
    fn into(self) -> Orthographic<S> {
        match self {
            [left, right, bottom, top, near, far] => {
                Orthographic {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<S> Into<Orthographic<S>> for &[S; 6] where S: Copy {
    #[inline]
    fn into(self) -> Orthographic<S> {
        match *self {
            [left, right, bottom, top, near, far] => {
                Orthographic {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

/// A perspective projection based on arbitarary left, right, bottom,
/// top, near, and far planes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Perspective<S> {
    left: S,
    right: S,
    bottom: S,
    top: S,
    near: S,
    far: S,
}

impl<S> Into<Perspective<S>> for (S, S, S, S, S, S) {
    #[inline]
    fn into(self) -> Perspective<S> {
        match self {
            (left, right, bottom, top, near, far) => {
                Perspective {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<S> Into<Perspective<S>> for &(S, S, S, S, S, S) where S: Copy {
    #[inline]
    fn into(self) -> Perspective<S> {
        match *self {
            (left, right, bottom, top, near, far) => {
                Perspective {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<S> Into<Perspective<S>> for [S; 6] {
    #[inline]
    fn into(self) -> Perspective<S> {
        match self {
            [left, right, bottom, top, near, far] => {
                Perspective {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<S> Into<Perspective<S>> for &[S; 6] where S: Copy {
    #[inline]
    fn into(self) -> Perspective<S> {
        match *self {
            [left, right, bottom, top, near, far] => {
                Perspective {
                    left: left,
                    right: right,
                    bottom: bottom,
                    top: top,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

/// A perspective projection based on the near and farplanes and the vertical
/// field-of-view angle and the horizontal/vertical aspect ratio.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerspectiveFov<S> {
    fovy: Radians<S>,
    aspect: S,
    near: S,
    far: S,
}

impl<S> Into<PerspectiveFov<S>> for (Radians<S>, S, S, S) {
    #[inline]
    fn into(self) -> PerspectiveFov<S> {
        match self {
            (fovy, aspect, near, far) => {
                PerspectiveFov {
                    fovy: fovy,
                    aspect: aspect,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<S> Into<PerspectiveFov<S>> for &(Radians<S>, S, S, S) where S: Copy {
    #[inline]
    fn into(self) -> PerspectiveFov<S> {
        match *self {
            (fovy, aspect, near, far) => {
                PerspectiveFov {
                    fovy: fovy,
                    aspect: aspect,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<S> Into<PerspectiveFov<S>> for (Degrees<S>, S, S, S) where S: ScalarFloat {
    #[inline]
    fn into(self) -> PerspectiveFov<S> {
        match self {
            (fovy, aspect, near, far) => {
                PerspectiveFov {
                    fovy: fovy.into(),
                    aspect: aspect,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<S> Into<PerspectiveFov<S>> for &(Degrees<S>, S, S, S) where S: ScalarFloat {
    #[inline]
    fn into(self) -> PerspectiveFov<S> {
        match *self {
            (fovy, aspect, near, far) => {
                PerspectiveFov {
                    fovy: fovy.into(),
                    aspect: aspect,
                    near: near,
                    far: far,
                }
            }
        }
    }
}

impl<S> From<PerspectiveFov<S>> for Matrix4x4<S> where S: ScalarFloat {
    fn from(persp: PerspectiveFov<S>) -> Matrix4x4<S> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let fov_rad = Radians::from(persp.fovy);
        let range = Angle::tan(fov_rad / two) * persp.near;
        let sx = (two * persp.near) / (range * persp.aspect + range * persp.aspect);
        let sy = persp.near / range;
        let sz = (persp.far + persp.near) / (persp.near - persp.far);
        let pz = (two * persp.far * persp.near) / (persp.near - persp.far);
        
        Matrix4x4::new(
            sx,    zero,  zero,  zero,
            zero,  sy,    zero,  zero,
            zero,  zero,  sz,   -one,
            zero,  zero,  pz,    zero
        )
    }
}

impl<S> From<Perspective<S>> for Matrix4x4<S> where S: ScalarFloat {
    fn from(persp: Perspective<S>) -> Matrix4x4<S> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;

        let c0r0 = (two * persp.near) / (persp.right - persp.left);
        let c0r1 = zero;
        let c0r2 = zero;
        let c0r3 = zero;

        let c1r0 = zero;
        let c1r1 = (two * persp.near) / (persp.top - persp.bottom);
        let c1r2 = zero;
        let c1r3 = zero;

        let c2r0 =  (persp.right + persp.left)   / (persp.right - persp.left);
        let c2r1 =  (persp.top   + persp.bottom) / (persp.top   - persp.bottom);
        let c2r2 = -(persp.far   + persp.near)   / (persp.far   - persp.near);
        let c2r3 = -one;

        let c3r0 = zero;
        let c3r1 = zero;
        let c3r2 = -(two * persp.far * persp.near) / (persp.far - persp.near);
        let c3r3 = zero;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3,
        )
    }
}

impl<S> From<Orthographic<S>> for Matrix4x4<S> where S: ScalarFloat {
    fn from(ortho: Orthographic<S>) -> Matrix4x4<S> {
        let zero = S::zero();
        let one  = S::one();
        let two = one + one;
        let sx = two / (ortho.right - ortho.left);
        let sy = two / (ortho.top - ortho.bottom);
        let sz = two / (ortho.far - ortho.near);
        let tx = (ortho.right + ortho.left) / (ortho.right - ortho.left);
        let ty = (ortho.top + ortho.bottom) / (ortho.top - ortho.bottom);
        let tz = (ortho.far + ortho.near) / (ortho.far - ortho.near);

        Matrix4x4::new(
             sx,    zero,  zero, zero,
             zero,  sy,    zero, zero,
             zero,  zero,  sz,   zero,
            -tx,   -ty,   -tz,   one
        )
    }
}

