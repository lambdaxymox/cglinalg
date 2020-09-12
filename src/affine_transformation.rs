use scalar::{
    Scalar,
    ScalarSigned,
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
use std::fmt;


pub trait AffineTransformation2D<V> where Self: Sized {
    type Applied;

    fn identity() -> Self;

    fn inverse(&self) -> Option<Self>;

    fn apply(&self, point: V) -> Self::Applied;

    fn apply_inverse(&self, point: V) -> Option<Self::Applied>;
}

pub trait AffineTransformation3D<V> where Self: Sized {
    type Applied;

    fn identity() -> Self;
    
    fn inverse(&self) -> Option<Self>;
    
    fn apply(&self, point: V) -> Self::Applied;
    
    fn apply_inverse(&self, point: V) -> Option<Self::Applied>;
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Identity2D<S> {
    matrix: Matrix3<S>,
}

impl<S> Identity2D<S> where S: Scalar {
    #[inline]
    pub fn identity() -> Identity2D<S> {
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

impl<S> AffineTransformation2D<Point2<S>> for Identity2D<S> where S: Scalar {
    type Applied = Point2<S>;

    #[inline]
    fn identity() -> Identity2D<S> {
        Identity2D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity2D<S>> {
        Some(Identity2D::identity())
    }

    #[inline]
    fn apply(&self, point: Point2<S>) -> Point2<S> {
        point
    }

    #[inline]
    fn apply_inverse(&self, point: Point2<S>) -> Option<Point2<S>> {
        Some(point)
    }
}

impl<S> AffineTransformation2D<&Point2<S>> for Identity2D<S> where S: Scalar {
    type Applied = Point2<S>;

    #[inline]
    fn identity() -> Identity2D<S> {
        Identity2D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity2D<S>> {
        Some(Identity2D::identity())
    }

    #[inline]
    fn apply(&self, point: &Point2<S>) -> Point2<S> {
        *point
    }

    #[inline]
    fn apply_inverse(&self, point: &Point2<S>) -> Option<Point2<S>> {
        Some(*point)
    }
}

impl<S> AffineTransformation2D<Vector2<S>> for Identity2D<S> where S: Scalar {
    type Applied = Vector2<S>;

    #[inline]
    fn identity() -> Identity2D<S> {
        Identity2D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity2D<S>> {
        Some(Identity2D::identity())
    }

    #[inline]
    fn apply(&self, point: Vector2<S>) -> Vector2<S> {
        point
    }

    #[inline]
    fn apply_inverse(&self, point: Vector2<S>) -> Option<Vector2<S>> {
        Some(point)
    }
}

impl<S> AffineTransformation2D<&Vector2<S>> for Identity2D<S> where S: Scalar {
    type Applied = Vector2<S>;

    #[inline]
    fn identity() -> Identity2D<S> {
        Identity2D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity2D<S>> {
        Some(Identity2D::identity())
    }

    #[inline]
    fn apply(&self, point: &Vector2<S>) -> Vector2<S> {
        *point
    }

    #[inline]
    fn apply_inverse(&self, point: &Vector2<S>) -> Option<Vector2<S>> {
        Some(*point)
    }
}


#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
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

impl<S> AffineTransformation3D<Point3<S>> for Identity3D<S> where S: Scalar {
    type Applied = Point3<S>;

    #[inline]
    fn identity() -> Identity3D<S> {
        Identity3D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity3D<S>> {
        Some(Identity3D::identity())
    }

    #[inline]
    fn apply(&self, point: Point3<S>) -> Point3<S> {
        point
    }

    #[inline]
    fn apply_inverse(&self, point: Point3<S>) -> Option<Point3<S>> {
        Some(point)
    }
}

impl<S> AffineTransformation3D<&Point3<S>> for Identity3D<S> where S: Scalar {
    type Applied = Point3<S>;

    #[inline]
    fn identity() -> Identity3D<S> {
        Identity3D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity3D<S>> {
        Some(Identity3D::identity())
    }

    #[inline]
    fn apply(&self, point: &Point3<S>) -> Point3<S> {
        *point
    }

    #[inline]
    fn apply_inverse(&self, point: &Point3<S>) -> Option<Point3<S>> {
        Some(*point)
    }
}

impl<S> AffineTransformation3D<Vector3<S>> for Identity3D<S> where S: Scalar {
    type Applied = Vector3<S>;

    #[inline]
    fn identity() -> Identity3D<S> {
        Identity3D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity3D<S>> {
        Some(Identity3D::identity())
    }

    #[inline]
    fn apply(&self, point: Vector3<S>) -> Vector3<S> {
        point
    }

    #[inline]
    fn apply_inverse(&self, point: Vector3<S>) -> Option<Vector3<S>> {
        Some(point)
    }
}

impl<S> AffineTransformation3D<&Vector3<S>> for Identity3D<S> where S: Scalar {
    type Applied = Vector3<S>;

    #[inline]
    fn identity() -> Identity3D<S> {
        Identity3D::identity()
    }

    #[inline]
    fn inverse(&self) -> Option<Identity3D<S>> {
        Some(Identity3D::identity())
    }

    #[inline]
    fn apply(&self, point: &Vector3<S>) -> Vector3<S> {
        *point
    }

    #[inline]
    fn apply_inverse(&self, point: &Vector3<S>) -> Option<Vector3<S>> {
        Some(*point)
    }
}


#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
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
    pub fn from_nonuniform_scale(scale_x: S, scale_y: S) -> Scale2D<S> {
        Scale2D {
            matrix: Matrix3::from_nonuniform_scale(scale_x, scale_y),
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

impl<S> AffineTransformation2D<Point2<S>> for Scale2D<S> where S: Scalar {
    type Applied = Point2<S>;

    #[inline]
    fn identity() -> Scale2D<S> {
        Scale2D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale2D<S>> {
        Some(Scale2D::from_nonuniform_scale(S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1))
    }

    #[inline]
    fn apply(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: Point2<S>) -> Option<Point2<S>> {
        let matrix = Scale2D::from_nonuniform_scale(S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1).matrix;
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
        Some(Scale2D::from_nonuniform_scale(S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1))
    }

    #[inline]
    fn apply(&self, point: &Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: &Point2<S>) -> Option<Point2<S>> {
        let matrix = Scale2D::from_nonuniform_scale(S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1).matrix;
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
        (self.matrix * vector.extend(S::zero())).truncate()
    }

    #[inline]
    fn apply_inverse(&self, vector: Vector2<S>) -> Option<Vector2<S>> {
        let matrix = Scale2D::from_nonuniform_scale(S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1).matrix;
        Some((matrix * vector.extend(S::zero())).truncate())
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
        Some(Scale2D::from_nonuniform_scale(S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1))
    }

    #[inline]
    fn apply(&self, vector: &Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.extend(S::zero())).truncate()
    }

    #[inline]
    fn apply_inverse(&self, vector: &Vector2<S>) -> Option<Vector2<S>> {
        let matrix = Scale2D::from_nonuniform_scale(S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1).matrix;
        Some((matrix * vector.extend(S::zero())).truncate())
    }
}


#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
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
    pub fn from_nonuniform_scale(scale_x: S, scale_y: S, scale_z: S) -> Scale3D<S> {
        Scale3D {
            matrix: Matrix4::from_nonuniform_scale(scale_x, scale_y, scale_z),
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

impl<S> AffineTransformation3D<Point3<S>> for Scale3D<S> where S: Scalar {
    type Applied = Point3<S>;

    #[inline]
    fn identity() -> Scale3D<S> {
        Scale3D::from_scale(S::one())
    }

    #[inline]
    fn inverse(&self) -> Option<Scale3D<S>> {
        Some(Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1, S::one() / self.matrix.c2r2
        ))
    }

    #[inline]
    fn apply(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: Point3<S>) -> Option<Point3<S>> {
        let matrix = Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1, S::one() / self.matrix.c2r2
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
            S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1, S::one() / self.matrix.c2r2
        ))
    }

    #[inline]
    fn apply(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: &Point3<S>) -> Option<Point3<S>> {
        let matrix = Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1, S::one() / self.matrix.c2r2
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
            S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1, S::one() / self.matrix.c2r2
        ))
    }

    #[inline]
    fn apply(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.extend(S::zero())).truncate()
    }

    #[inline]
    fn apply_inverse(&self, vector: Vector3<S>) -> Option<Vector3<S>> {
        let matrix = Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1, S::one() / self.matrix.c2r2
        ).matrix;
        Some((matrix * vector.extend(S::zero())).truncate())
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
            S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1, S::one() / self.matrix.c2r2
        ))
    }

    #[inline]
    fn apply(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.extend(S::zero())).truncate()
    }

    #[inline]
    fn apply_inverse(&self, vector: &Vector3<S>) -> Option<Vector3<S>> {
        let matrix = Scale3D::from_nonuniform_scale(
            S::one() / self.matrix.c0r0, S::one() / self.matrix.c1r1, S::one() / self.matrix.c2r2
        ).matrix;
        Some((matrix * vector.extend(S::zero())).truncate())
    }
}


#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Reflection2D<S> {
    normal: Vector2<S>,
    matrix: Matrix3<S>,
}

impl<S> Reflection2D<S> where S: ScalarFloat {
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

impl<S> AffineTransformation2D<Point2<S>> for Reflection2D<S> where S: ScalarFloat {
    type Applied = Point2<S>;

    #[inline]
    fn identity() -> Reflection2D<S> {
        Reflection2D { 
            normal: Vector2::zero(), 
            matrix: Matrix3::one(),
        }
    }

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


#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Reflection3D<S> {
    normal: Vector3<S>,
    matrix: Matrix4<S>,
}

impl<S> Reflection3D<S> where S: ScalarFloat {
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

impl<S> AffineTransformation3D<Point3<S>> for Reflection3D<S> where S: ScalarFloat {
    type Applied = Point3<S>;

    #[inline]
    fn identity() -> Reflection3D<S> {
        Reflection3D { 
            normal: Vector3::zero(), 
            matrix: Matrix4::one(),
        }
    }

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


#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
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

impl<S> AffineTransformation2D<Point2<S>> for Translation2D<S> where S: ScalarSigned {
    type Applied = Point2<S>;

    #[inline]
    fn identity() -> Translation2D<S> {
        Translation2D { 
            matrix: Matrix3::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation2D<S>> {
        let distance_x = self.matrix.c2r0;
        let distance_y = self.matrix.c2r1;
        let distance = Vector2::new(-distance_x, -distance_y);
        let matrix = Matrix3::from_translation(distance);
        
        Some(Translation2D {
            matrix: matrix,
        })
    }

    #[inline]
    fn apply(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: Point2<S>) -> Option<Point2<S>> {
        let inverse_matrix = <Self as AffineTransformation2D<Point2<S>>>::inverse(&self).unwrap().matrix;
        Some(Point2::from_homogeneous(inverse_matrix * point.to_homogeneous()))
    }
}

impl<S> AffineTransformation2D<&Point2<S>> for Translation2D<S> where S: ScalarSigned {
    type Applied = Point2<S>;

    #[inline]
    fn identity() -> Translation2D<S> {
        Translation2D { 
            matrix: Matrix3::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation2D<S>> {
        let distance_x = self.matrix.c2r0;
        let distance_y = self.matrix.c2r1;
        let distance = Vector2::new(-distance_x, -distance_y);
        let matrix = Matrix3::from_translation(distance);
        
        Some(Translation2D {
            matrix: matrix,
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

impl<S> AffineTransformation2D<Vector2<S>> for Translation2D<S> where S: ScalarSigned {
    type Applied = Vector2<S>;

    #[inline]
    fn identity() -> Translation2D<S> {
        Translation2D { 
            matrix: Matrix3::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation2D<S>> {
        let distance_x = self.matrix.c2r0;
        let distance_y = self.matrix.c2r1;
        let distance = Vector2::new(-distance_x, -distance_y);
        let matrix = Matrix3::from_translation(distance);
        
        Some(Translation2D {
            matrix: matrix,
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

impl<S> AffineTransformation2D<&Vector2<S>> for Translation2D<S> where S: ScalarSigned {
    type Applied = Vector2<S>;

    #[inline]
    fn identity() -> Translation2D<S> {
        Translation2D { 
            matrix: Matrix3::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation2D<S>> {
        let distance_x = self.matrix.c2r0;
        let distance_y = self.matrix.c2r1;
        let distance = Vector2::new(-distance_x, -distance_y);
        let matrix = Matrix3::from_translation(distance);
        
        Some(Translation2D {
            matrix: matrix,
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


#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
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

impl<S> AffineTransformation3D<Point3<S>> for Translation3D<S> where S: ScalarSigned {
    type Applied = Point3<S>;

    #[inline]
    fn identity() -> Translation3D<S> {
        Translation3D { 
            matrix: Matrix4::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation3D<S>> {
        let distance_x = self.matrix.c3r0;
        let distance_y = self.matrix.c3r1;
        let distance_z = self.matrix.c3r2;
        let distance = Vector3::new(-distance_x, -distance_y, -distance_z);
        let matrix = Matrix4::from_translation(distance);
        
        Some(Translation3D {
            matrix: matrix,
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

impl<S> AffineTransformation3D<&Point3<S>> for Translation3D<S> where S: ScalarSigned {
    type Applied = Point3<S>;

    #[inline]
    fn identity() -> Translation3D<S> {
        Translation3D { 
            matrix: Matrix4::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation3D<S>> {
        let distance_x = self.matrix.c3r0;
        let distance_y = self.matrix.c3r1;
        let distance_z = self.matrix.c3r2;
        let distance = Vector3::new(-distance_x, -distance_y, -distance_z);
        let matrix = Matrix4::from_translation(distance);
        
        Some(Translation3D {
            matrix: matrix,
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

impl<S> AffineTransformation3D<Vector3<S>> for Translation3D<S> where S: ScalarSigned {
    type Applied = Vector3<S>;

    #[inline]
    fn identity() -> Translation3D<S> {
        Translation3D { 
            matrix: Matrix4::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation3D<S>> {
        let distance_x = self.matrix.c3r0;
        let distance_y = self.matrix.c3r1;
        let distance_z = self.matrix.c3r2;
        let distance = Vector3::new(-distance_x, -distance_y, -distance_z);
        let matrix = Matrix4::from_translation(distance);
        
        Some(Translation3D {
            matrix: matrix,
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

impl<S> AffineTransformation3D<&Vector3<S>> for Translation3D<S> where S: ScalarSigned {
    type Applied = Vector3<S>;

    #[inline]
    fn identity() -> Translation3D<S> {
        Translation3D { 
            matrix: Matrix4::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Translation3D<S>> {
        let distance_x = self.matrix.c3r0;
        let distance_y = self.matrix.c3r1;
        let distance_z = self.matrix.c3r2;
        let distance = Vector3::new(-distance_x, -distance_y, -distance_z);
        let matrix = Matrix4::from_translation(distance);
        
        Some(Translation3D {
            matrix: matrix,
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

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
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

impl<S> AffineTransformation2D<Point2<S>> for Shear2D<S> where S: ScalarSigned {
    type Applied = Point2<S>;

    #[inline]
    fn identity() -> Shear2D<S> {
        Shear2D { 
            matrix: Matrix3::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear2D<S>> {
        let zero = S::zero();
        let one = S::one();
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_x_with_y = -self.matrix.c1r0;
        let matrix = Matrix3::new(
            one,            shear_y_with_x, zero,
            shear_x_with_y, one,            zero,
            zero,           zero,           one
        );
        
        Some(Shear2D {
            matrix: matrix,
        })
    }

    #[inline]
    fn apply(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    #[inline]
    fn apply_inverse(&self, point: Point2<S>) -> Option<Point2<S>> {
        let inverse_matrix = <Self as AffineTransformation2D<Point2<S>>>::inverse(&self).unwrap().matrix;
        Some(Point2::from_homogeneous(inverse_matrix * point.to_homogeneous()))
    }
}

impl<S> AffineTransformation2D<&Point2<S>> for Shear2D<S> where S: ScalarFloat {
    type Applied = Point2<S>;

    #[inline]
    fn identity() -> Shear2D<S> {
        Shear2D { 
            matrix: Matrix3::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear2D<S>> {
        let zero = S::zero();
        let one = S::one();
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_x_with_y = -self.matrix.c1r0;
        let matrix = Matrix3::new(
            one,            shear_y_with_x, zero,
            shear_x_with_y, one,            zero,
            zero,           zero,           one
        );
        
        Some(Shear2D {
            matrix: matrix,
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

impl<S> AffineTransformation2D<Vector2<S>> for Shear2D<S> where S: ScalarFloat {
    type Applied = Vector2<S>;

    #[inline]
    fn identity() -> Shear2D<S> {
        Shear2D { 
            matrix: Matrix3::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear2D<S>> {
        let zero = S::zero();
        let one = S::one();
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_x_with_y = -self.matrix.c1r0;
        let matrix = Matrix3::new(
            one,            shear_y_with_x, zero,
            shear_x_with_y, one,            zero,
            zero,           zero,           one
        );
        
        Some(Shear2D {
            matrix: matrix,
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

impl<S> AffineTransformation2D<&Vector2<S>> for Shear2D<S> where S: ScalarFloat {
    type Applied = Vector2<S>;

    #[inline]
    fn identity() -> Shear2D<S> {
        Shear2D { 
            matrix: Matrix3::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear2D<S>> {
        let zero = S::zero();
        let one = S::one();
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_x_with_y = -self.matrix.c1r0;
        let matrix = Matrix3::new(
            one,            shear_y_with_x, zero,
            shear_x_with_y, one,            zero,
            zero,           zero,           one
        );
        
        Some(Shear2D {
            matrix: matrix,
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


#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
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

impl<S> AffineTransformation3D<Point3<S>> for Shear3D<S> where S: ScalarSigned {
    type Applied = Point3<S>;

    #[inline]
    fn identity() -> Shear3D<S> {
        Shear3D { 
            matrix: Matrix4::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear3D<S>> {
        let zero = S::zero();
        let one = S::one();
        let shear_x_with_y = -self.matrix.c1r0;
        let shear_x_with_z = -self.matrix.c2r0;
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_y_with_z = -self.matrix.c2r1;
        let shear_z_with_x = -self.matrix.c0r2;
        let shear_z_with_y = -self.matrix.c1r2;
        let matrix = Matrix4::new(
            one,            shear_y_with_x, shear_z_with_x, zero, 
            shear_x_with_y, one,            shear_z_with_y, zero,
            shear_x_with_z, shear_y_with_z, one,            zero,
            zero,           zero,           zero,           one
        );
        
        Some(Shear3D {
            matrix: matrix,
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

impl<S> AffineTransformation3D<&Point3<S>> for Shear3D<S> where S: ScalarFloat {
    type Applied = Point3<S>;

    #[inline]
    fn identity() -> Shear3D<S> {
        Shear3D { 
            matrix: Matrix4::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear3D<S>> {
        let zero = S::zero();
        let one = S::one();
        let shear_x_with_y = -self.matrix.c1r0;
        let shear_x_with_z = -self.matrix.c2r0;
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_y_with_z = -self.matrix.c2r1;
        let shear_z_with_x = -self.matrix.c0r2;
        let shear_z_with_y = -self.matrix.c1r2;
        let matrix = Matrix4::new(
            one,            shear_y_with_x, shear_z_with_x, zero, 
            shear_x_with_y, one,            shear_z_with_y, zero,
            shear_x_with_z, shear_y_with_z, one,            zero,
            zero,           zero,           zero,           one
        );
        
        Some(Shear3D {
            matrix: matrix,
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

impl<S> AffineTransformation3D<Vector3<S>> for Shear3D<S> where S: ScalarFloat {
    type Applied = Vector3<S>;

    #[inline]
    fn identity() -> Shear3D<S> {
        Shear3D { 
            matrix: Matrix4::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear3D<S>> {
        let zero = S::zero();
        let one = S::one();
        let shear_x_with_y = -self.matrix.c1r0;
        let shear_x_with_z = -self.matrix.c2r0;
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_y_with_z = -self.matrix.c2r1;
        let shear_z_with_x = -self.matrix.c0r2;
        let shear_z_with_y = -self.matrix.c1r2;
        let matrix = Matrix4::new(
            one,            shear_y_with_x, shear_z_with_x, zero, 
            shear_x_with_y, one,            shear_z_with_y, zero,
            shear_x_with_z, shear_y_with_z, one,            zero,
            zero,           zero,           zero,           one
        );
        
        Some(Shear3D {
            matrix: matrix,
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

impl<S> AffineTransformation3D<&Vector3<S>> for Shear3D<S> where S: ScalarFloat {
    type Applied = Vector3<S>;

    #[inline]
    fn identity() -> Shear3D<S> {
        Shear3D { 
            matrix: Matrix4::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Shear3D<S>> {
        let zero = S::zero();
        let one = S::one();
        let shear_x_with_y = -self.matrix.c1r0;
        let shear_x_with_z = -self.matrix.c2r0;
        let shear_y_with_x = -self.matrix.c0r1;
        let shear_y_with_z = -self.matrix.c2r1;
        let shear_z_with_x = -self.matrix.c0r2;
        let shear_z_with_y = -self.matrix.c1r2;
        let matrix = Matrix4::new(
            one,            shear_y_with_x, shear_z_with_x, zero, 
            shear_x_with_y, one,            shear_z_with_y, zero,
            shear_x_with_z, shear_y_with_z, one,            zero,
            zero,           zero,           zero,           one
        );
        
        Some(Shear3D {
            matrix: matrix,
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

