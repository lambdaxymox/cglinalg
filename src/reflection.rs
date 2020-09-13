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
use point::{
    Point2,
    Point3,
};
use structure::{
    One,
    Zero,
};
use affine::*;

use std::fmt;


/// A reflection transformation about a plane in two dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Reflection2D<S> {
    /// The normal vector to the plane.
    normal: Vector2<S>,
    /// The matrix representing the affine transformation.
    matrix: Matrix3<S>,
}

impl<S> Reflection2D<S> where S: ScalarFloat {
    /// Construct a new reflection transformation from the vector normal to the plane of reflection.
    #[rustfmt::skip]
    pub fn from_normal(normal: Vector2<S>) -> Reflection2D<S> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        Reflection2D {
            normal: normal,
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

impl<S> From<Reflection2D<S>> for Matrix3<S> where S: Copy {
    fn from(transformation: Reflection2D<S>) -> Matrix3<S> {
        transformation.matrix
    }
}

impl<S> From<&Reflection2D<S>> for Matrix3<S> where S: Copy {
    fn from(transformation: &Reflection2D<S>) -> Matrix3<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation2D<Point2<S>> for Reflection2D<S> where S: ScalarFloat {
    type Applied = Point2<S>;

    #[inline]
    fn identity() -> Reflection2D<S> {
        Reflection2D { 
            normal: Vector2::zero(), 
            matrix: Matrix3::one(),
        }
    }

    #[rustfmt::skip]
    #[inline]
    fn inverse(&self) -> Option<Reflection2D<S>> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let normal = self.normal;
        let inverse_det = one / (one - two * normal.x * normal.x - two * normal.y * normal.y);
        let matrix = Matrix3::new(
            one - two * normal.y * normal.y, two * normal.x * normal.y,                                   zero,
            two * normal.x * normal.y,       one - two * normal.x * normal.x - two * normal.y * normal.y, zero,
            zero,                            zero,                                                        one
        );

        Some(Reflection2D { 
            normal: normal, 
            matrix: matrix * inverse_det 
        })
    }

    #[inline]
    fn apply(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: Point2<S>) -> Option<Point2<S>> {
        let inverse_matrix = <Self as AffineTransformation2D<Point2<S>>>::inverse(&self).unwrap().matrix;
        Some(Point2::from_homogeneous( inverse_matrix * point.to_homogeneous()))
    }
}

impl<S> AffineTransformation2D<&Point2<S>> for Reflection2D<S> where S: ScalarFloat {
    type Applied = Point2<S>;

    #[inline]
    fn identity() -> Reflection2D<S> {
        Reflection2D { 
            normal: Vector2::zero(), 
            matrix: Matrix3::one(),
        }
    }

    #[rustfmt::skip]
    #[inline]
    fn inverse(&self) -> Option<Reflection2D<S>> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let normal = self.normal;
        let inverse_det = one / (one - two * normal.x * normal.x - two * normal.y * normal.y);
        let matrix = Matrix3::new(
            one - two * normal.y * normal.y, two * normal.x * normal.y,                                   zero,
            two * normal.x * normal.y,       one - two * normal.x * normal.x - two * normal.y * normal.y, zero,
            zero,                            zero,                                                        one
        );

        Some(Reflection2D { 
            normal: normal, 
            matrix: matrix * inverse_det 
        })
    }

    #[inline]
    fn apply(&self, point: &Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: &Point2<S>) -> Option<Point2<S>> {
        let inverse_matrix = <Self as AffineTransformation2D<Point2<S>>>::inverse(&self).unwrap().matrix;
        Some(Point2::from_homogeneous( inverse_matrix * point.to_homogeneous()))
    }
}

impl<S> AffineTransformation2D<Vector2<S>> for Reflection2D<S> where S: ScalarFloat {
    type Applied = Vector2<S>;

    #[inline]
    fn identity() -> Reflection2D<S> {
        Reflection2D { 
            normal: Vector2::zero(), 
            matrix: Matrix3::one(),
        }
    }

    #[rustfmt::skip]
    #[inline]
    fn inverse(&self) -> Option<Reflection2D<S>> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let normal = self.normal;
        let inverse_det = one / (one - two * normal.x * normal.x - two * normal.y * normal.y);
        let matrix = Matrix3::new(
            one - two * normal.y * normal.y, two * normal.x * normal.y,                                   zero,
            two * normal.x * normal.y,       one - two * normal.x * normal.x - two * normal.y * normal.y, zero,
            zero,                            zero,                                                        one
        );

        Some(Reflection2D { 
            normal: normal, 
            matrix: matrix * inverse_det 
        })
    }

    #[inline]
    fn apply(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.extend(S::zero())).truncate()
    }

    #[inline]
    fn apply_inverse(&self, vector: Vector2<S>) -> Option<Vector2<S>> {
        let inverse_matrix = <Self as AffineTransformation2D<Vector2<S>>>::inverse(&self).unwrap().matrix;
        Some((inverse_matrix * vector.extend(S::zero())).truncate())
    }
}

impl<S> AffineTransformation2D<&Vector2<S>> for Reflection2D<S> where S: ScalarFloat {
    type Applied = Vector2<S>;

    #[inline]
    fn identity() -> Reflection2D<S> {
        Reflection2D { 
            normal: Vector2::zero(), 
            matrix: Matrix3::one(),
        }
    }

    #[rustfmt::skip]
    #[inline]
    fn inverse(&self) -> Option<Reflection2D<S>> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let normal = self.normal;
        let inverse_det = one / (one - two * normal.x * normal.x - two * normal.y * normal.y);
        let matrix = Matrix3::new(
            one - two * normal.y * normal.y, two * normal.x * normal.y,                                   zero,
            two * normal.x * normal.y,       one - two * normal.x * normal.x - two * normal.y * normal.y, zero,
            zero,                            zero,                                                        one
        );

        Some(Reflection2D { 
            normal: self.normal, 
            matrix: matrix * inverse_det 
        })
    }

    #[inline]
    fn apply(&self, vector: &Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.extend(S::zero())).truncate()
    }

    #[inline]
    fn apply_inverse(&self, vector: &Vector2<S>) -> Option<Vector2<S>> {
        let inverse_matrix = <Self as AffineTransformation2D<Vector2<S>>>::inverse(&self).unwrap().matrix;
        Some((inverse_matrix * vector.extend(S::zero())).truncate())
    }
}


/// A reflection transformation about a plane in three dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Reflection3D<S> {
    /// The normal vector to the plane.
    normal: Vector3<S>,
    /// The matrix representing the affine transformation.
    matrix: Matrix4<S>,
}

impl<S> Reflection3D<S> where S: ScalarFloat {
    /// Construct a new reflection transformation from the vector normal to the plane of reflection.
    #[rustfmt::skip]
    pub fn from_normal(normal: Vector3<S>) -> Reflection3D<S> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        Reflection3D {
            normal: normal,
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

impl<S> From<Reflection3D<S>> for Matrix4<S> where S: Copy {
    fn from(transformation: Reflection3D<S>) -> Matrix4<S> {
        transformation.matrix
    }
}

impl<S> From<&Reflection3D<S>> for Matrix4<S> where S: Copy {
    fn from(transformation: &Reflection3D<S>) -> Matrix4<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation3D<Point3<S>> for Reflection3D<S> where S: ScalarFloat {
    type Applied = Point3<S>;

    #[inline]
    fn identity() -> Reflection3D<S> {
        Reflection3D { 
            normal: Vector3::zero(), 
            matrix: Matrix4::one(),
        }
    }

    #[rustfmt::skip]
    #[inline]
    fn inverse(&self) -> Option<Reflection3D<S>> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let normal = self.normal;
        let inverse_det = one / (one - two * normal.x * normal.x - two * normal.y * normal.y - two * normal.z * normal.z);
        let matrix = Matrix4::new(
            one - two * normal.y * normal.y - normal.z * normal.z, two * normal.x * normal.y,                                   two * normal.x * normal.z,                                   zero,
            two * normal.x * normal.y,                             one - two * normal.x * normal.x - two * normal.z * normal.z, two * normal.y * normal.z,                                   zero,
            two * normal.x * normal.z,                             two * normal.y * normal.z,                                   one - two * normal.x * normal.x - two * normal.y * normal.y, zero,
            zero,                                                  zero,                                                        zero,                                                        one
        );

        Some(Reflection3D { 
            normal: normal, 
            matrix: matrix * inverse_det,
        })
    }

    #[inline]
    fn apply(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: Point3<S>) -> Option<Point3<S>> {
        let inverse_matrix = <Self as AffineTransformation3D<Point3<S>>>::inverse(&self).unwrap().matrix;
        Some(Point3::from_homogeneous(inverse_matrix * point.to_homogeneous()))
    }
}

impl<S> AffineTransformation3D<&Point3<S>> for Reflection3D<S> where S: ScalarFloat {
    type Applied = Point3<S>;

    #[inline]
    fn identity() -> Reflection3D<S> {
        Reflection3D { 
            normal: Vector3::zero(), 
            matrix: Matrix4::one(),
        }
    }

    #[rustfmt::skip]
    #[inline]
    fn inverse(&self) -> Option<Reflection3D<S>> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let normal = self.normal;
        let inverse_det = one / (one - two * normal.x * normal.x - two * normal.y * normal.y - two * normal.z * normal.z);
        let matrix = Matrix4::new(
            one - two * normal.y * normal.y - normal.z * normal.z, two * normal.x * normal.y,                                   two * normal.x * normal.z,                                   zero,
            two * normal.x * normal.y,                             one - two * normal.x * normal.x - two * normal.z * normal.z, two * normal.y * normal.z,                                   zero,
            two * normal.x * normal.z,                             two * normal.y * normal.z,                                   one - two * normal.x * normal.x - two * normal.y * normal.y, zero,
            zero,                                                  zero,                                                        zero,                                                        one
        );

        Some(Reflection3D { 
            normal: normal, 
            matrix: matrix * inverse_det,
        })
    }

    #[inline]
    fn apply(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: &Point3<S>) -> Option<Point3<S>> {
        let inverse_matrix = <Self as AffineTransformation3D<Point3<S>>>::inverse(&self).unwrap().matrix;
        Some(Point3::from_homogeneous( inverse_matrix * point.to_homogeneous()))
    }
}

impl<S> AffineTransformation3D<Vector3<S>> for Reflection3D<S> where S: ScalarFloat {
    type Applied = Vector3<S>;

    #[inline]
    fn identity() -> Reflection3D<S> {
        Reflection3D { 
            normal: Vector3::zero(), 
            matrix: Matrix4::one(),
        }
    }

    #[rustfmt::skip]
    #[inline]
    fn inverse(&self) -> Option<Reflection3D<S>> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let normal = self.normal;
        let inverse_det = one / (one - two * normal.x * normal.x - two * normal.y * normal.y - two * normal.z * normal.z);
        let matrix = Matrix4::new(
            one - two * normal.y * normal.y - normal.z * normal.z, two * normal.x * normal.y,                                   two * normal.x * normal.z,                                   zero,
            two * normal.x * normal.y,                             one - two * normal.x * normal.x - two * normal.z * normal.z, two * normal.y * normal.z,                                   zero,
            two * normal.x * normal.z,                             two * normal.y * normal.z,                                   one - two * normal.x * normal.x - two * normal.y * normal.y, zero,
            zero,                                                  zero,                                                        zero,                                                        one
        );

        Some(Reflection3D { 
            normal: normal, 
            matrix: matrix * inverse_det,
        })
    }

    #[inline]
    fn apply(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.extend(S::zero())).truncate()
    }

    #[inline]
    fn apply_inverse(&self, vector: Vector3<S>) -> Option<Vector3<S>> {
        let inverse_matrix = <Self as AffineTransformation3D<Vector3<S>>>::inverse(&self).unwrap().matrix;
        Some((inverse_matrix * vector.extend(S::zero())).truncate())
    }
}

impl<S> AffineTransformation3D<&Vector3<S>> for Reflection3D<S> where S: ScalarFloat {
    type Applied = Vector3<S>;

    #[inline]
    fn identity() -> Reflection3D<S> {
        Reflection3D { 
            normal: Vector3::zero(), 
            matrix: Matrix4::one(),
        }
    }

    #[rustfmt::skip]
    #[inline]
    fn inverse(&self) -> Option<Reflection3D<S>> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let normal = self.normal;
        let inverse_det = one / (one - two * normal.x * normal.x - two * normal.y * normal.y - two * normal.z * normal.z);
        let matrix = Matrix4::new(
            one - two * normal.y * normal.y - normal.z * normal.z, two * normal.x * normal.y,                                   two * normal.x * normal.z,                                   zero,
            two * normal.x * normal.y,                             one - two * normal.x * normal.x - two * normal.z * normal.z, two * normal.y * normal.z,                                   zero,
            two * normal.x * normal.z,                             two * normal.y * normal.z,                                   one - two * normal.x * normal.x - two * normal.y * normal.y, zero,
            zero,                                                  zero,                                                        zero,                                                        one
        );

        Some(Reflection3D { 
            normal: normal, 
            matrix: matrix * inverse_det,
        })
    }

    #[inline]
    fn apply(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.extend(S::zero())).truncate()
    }

    #[inline]
    fn apply_inverse(&self, vector: &Vector3<S>) -> Option<Vector3<S>> {
        let inverse_matrix = <Self as AffineTransformation3D<Vector3<S>>>::inverse(&self).unwrap().matrix;
        Some((inverse_matrix * vector.extend(S::zero())).truncate())
    }
}
