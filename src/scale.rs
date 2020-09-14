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
use point::{
    Point2,
    Point3,
};
use affine::*;

use std::fmt;


/// The scale transformation in two dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Scale2D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix3<S>,
}

impl<S> Scale2D<S> where S: Scalar {
    /// Construct a scale transformation from a vector of scale factors.
    #[inline]
    pub fn from_vector(scale: Vector2<S>) -> Scale2D<S> {
        let matrix = Matrix3::from_affine_nonuniform_scale(scale.x, scale.y);

        Scale2D {
            matrix: matrix,
        }
    }

    /// Construct a scale transformation from a nonuniform scale across coordinates.
    #[inline]
    pub fn from_nonuniform_scale(scale_x: S, scale_y: S) -> Scale2D<S> {
        let matrix = Matrix3::from_affine_nonuniform_scale(scale_x, scale_y);

        Scale2D {
            matrix: matrix,
        }
    }

    /// Construct a scale transformation from a uniform scale factor.
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

impl<S> From<Scale2D<S>> for Matrix3<S> where S: Copy {
    fn from(transformation: Scale2D<S>) -> Matrix3<S> {
        transformation.matrix
    }
}

impl<S> From<&Scale2D<S>> for Matrix3<S> where S: Copy {
    fn from(transformation: &Scale2D<S>) -> Matrix3<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation2D<Point2<S>> for Scale2D<S> where S: Scalar {
    type Applied = Point2<S>;

    #[inline]
    fn identity() -> Scale2D<S> {
        Scale2D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2D<S>> {
        Some(Scale2D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1
        ))
    }

    #[inline]
    fn apply(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: Point2<S>) -> Option<Point2<S>> {
        let matrix = Scale2D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1
        ).matrix;
        Some(Point2::from_homogeneous( matrix * point.to_homogeneous()))
    }
}

impl<S> AffineTransformation2D<&Point2<S>> for Scale2D<S> where S: Scalar {
    type Applied = Point2<S>;

    #[inline]
    fn identity() -> Scale2D<S> {
        Scale2D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2D<S>> {
        Some(Scale2D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1
        ))
    }

    #[inline]
    fn apply(&self, point: &Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: &Point2<S>) -> Option<Point2<S>> {
        let matrix = Scale2D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1
        ).matrix;
        Some(Point2::from_homogeneous( matrix * point.to_homogeneous()))
    }
}

impl<S> AffineTransformation2D<Vector2<S>> for Scale2D<S> where S: Scalar {
    type Applied = Vector2<S>;

    #[inline]
    fn identity() -> Scale2D<S> {
        Scale2D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2D<S>> {
        Some(Scale2D::from_nonuniform_scale(S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1))
    }

    #[inline]
    fn apply(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_inverse(&self, vector: Vector2<S>) -> Option<Vector2<S>> {
        let matrix = Scale2D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1
        ).matrix;
        Some((matrix * vector.extend(S::zero())).contract())
    }
}

impl<S> AffineTransformation2D<&Vector2<S>> for Scale2D<S> where S: Scalar {
    type Applied = Vector2<S>;

    #[inline]
    fn identity() -> Scale2D<S> {
        Scale2D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2D<S>> {
        Some(Scale2D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1
        ))
    }

    #[inline]
    fn apply(&self, vector: &Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_inverse(&self, vector: &Vector2<S>) -> Option<Vector2<S>> {
        let matrix = Scale2D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1
        ).matrix;
        Some((matrix * vector.extend(S::zero())).contract())
    }
}


/// The scale transformation in three dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Scale3D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix4<S>,
}

impl<S> Scale3D<S> where S: Scalar {
    /// Construct a scale transformation from a vector of scale factors.
    pub fn from_vector(scale: Vector3<S>) -> Scale3D<S> {
        Scale3D {
            matrix: Matrix4::from_affine_nonuniform_scale(scale.x, scale.y, scale.z),
        }
    }

    /// Construct a scale transformation from a nonuniform scale across coordinates.
    #[inline]
    pub fn from_nonuniform_scale(scale_x: S, scale_y: S, scale_z: S) -> Scale3D<S> {
        Scale3D {
            matrix: Matrix4::from_affine_nonuniform_scale(scale_x, scale_y, scale_z),
        }
    }

    /// Construct a scale transformation from a uniform scale factor.
    #[inline]
    pub fn from_scale(scale: S) -> Scale3D<S> {
        Scale3D {
            matrix: Matrix4::from_affine_scale(scale),
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

impl<S> From<Scale3D<S>> for Matrix4<S> where S: Copy {
    fn from(transformation: Scale3D<S>) -> Matrix4<S> {
        transformation.matrix
    }
}

impl<S> From<&Scale3D<S>> for Matrix4<S> where S: Copy {
    fn from(transformation: &Scale3D<S>) -> Matrix4<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation3D<Point3<S>> for Scale3D<S> where S: Scalar {
    type Applied = Point3<S>;

    #[inline]
    fn identity() -> Scale3D<S> {
        Scale3D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3D<S>> {
        Some(Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1, 
            S::one() / self.matrix.c2r2
        ))
    }

    #[inline]
    fn apply(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: Point3<S>) -> Option<Point3<S>> {
        let matrix = Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1, 
            S::one() / self.matrix.c2r2
        ).matrix;
        Some(Point3::from_homogeneous( matrix * point.to_homogeneous()))
    }
}

impl<S> AffineTransformation3D<&Point3<S>> for Scale3D<S> where S: Scalar {
    type Applied = Point3<S>;

    #[inline]
    fn identity() -> Scale3D<S> {
        Scale3D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3D<S>> {
        Some(Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1, 
            S::one() / self.matrix.c2r2
        ))
    }

    #[inline]
    fn apply(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: &Point3<S>) -> Option<Point3<S>> {
        let matrix = Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1, 
            S::one() / self.matrix.c2r2
        ).matrix;
        Some(Point3::from_homogeneous( matrix * point.to_homogeneous()))
    }
}

impl<S> AffineTransformation3D<Vector3<S>> for Scale3D<S> where S: Scalar {
    type Applied = Vector3<S>;

    #[inline]
    fn identity() -> Scale3D<S> {
        Scale3D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3D<S>> {
        Some(Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1, 
            S::one() / self.matrix.c2r2
        ))
    }

    #[inline]
    fn apply(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_inverse(&self, vector: Vector3<S>) -> Option<Vector3<S>> {
        let matrix = Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1, 
            S::one() / self.matrix.c2r2
        ).matrix;
        Some((matrix * vector.extend(S::zero())).contract())
    }
}

impl<S> AffineTransformation3D<&Vector3<S>> for Scale3D<S> where S: Scalar {
    type Applied = Vector3<S>;

    #[inline]
    fn identity() -> Scale3D<S> {
        Scale3D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3D<S>> {
        Some(Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1, 
            S::one() / self.matrix.c2r2
        ))
    }

    #[inline]
    fn apply(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_inverse(&self, vector: &Vector3<S>) -> Option<Vector3<S>> {
        let matrix = Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, 
            S::one() / self.matrix.c1r1, 
            S::one() / self.matrix.c2r2
        ).matrix;
        Some((matrix * vector.extend(S::zero())).contract())
    }
}
