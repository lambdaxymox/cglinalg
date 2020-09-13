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
};

use std::fmt;


/// A trait for implementing two-dimensional affine transformations.
pub trait AffineTransformation2D<V> where Self: Sized {
    /// The result of applying an affine transformation. This allows use to handle vectors, points,
    /// and pointers to them interchangeably.
    type Applied;

    /// The identity transformation for this type.
    fn identity() -> Self;

    /// Compute the inverse of an affine transformation.
    fn inverse(&self) -> Option<Self>;

    /// Apply the affine transformation to the input.
    fn apply(&self, point: V) -> Self::Applied;

    /// Apply the inverse of the affine transformation to the input.
    fn apply_inverse(&self, point: V) -> Option<Self::Applied>;
}

/// A trait for implementing three-dimensional affine transformations.
pub trait AffineTransformation3D<V> where Self: Sized {
    /// The result of applying an affine transformation. This allows use to handle vectors, points,
    /// and pointers to them interchangeably.
    type Applied;

    /// The identity transformation for this type.
    fn identity() -> Self;
    
    /// Compute the inverse of an affine transformation.
    fn inverse(&self) -> Option<Self>;
    
    /// Apply the affine transformation to the input.
    fn apply(&self, point: V) -> Self::Applied;
    
    /// Apply the inverse of the affine transformation to the input.
    fn apply_inverse(&self, point: V) -> Option<Self::Applied>;
}


/// A shearing transformation in two dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Shear2D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix3<S>,
}

impl<S> Shear2D<S> where S: Scalar {
    /// Construct a shearing transformation from a vector of shearing factors.
    #[rustfmt::skip]
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

    /// Construct a shearing transformation along the x-axis.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the y-dimension to shearing along the x-dimension.
    #[inline]
    pub fn from_shear_x(shear_x_with_y: S) -> Shear2D<S> {
        Shear2D {
            matrix: Matrix3::from_shear_x(shear_x_with_y, S::zero()),
        }
    }

    /// Construct a shearing transformation along the y-axis.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the x-dimension to shearing along the y-dimension.
    #[inline]
    pub fn from_shear_y(shear_y_with_x: S) -> Shear2D<S> {
        Shear2D {
            matrix: Matrix3::from_shear_y(shear_y_with_x, S::zero()),
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

impl<S> From<Shear2D<S>> for Matrix3<S> where S: Copy {
    fn from(transformation: Shear2D<S>) -> Matrix3<S> {
        transformation.matrix
    }
}

impl<S> From<&Shear2D<S>> for Matrix3<S> where S: Copy {
    fn from(transformation: &Shear2D<S>) -> Matrix3<S> {
        transformation.matrix
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

    #[rustfmt::skip]
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

    #[rustfmt::skip]
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

    #[rustfmt::skip]
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

    #[rustfmt::skip]
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


/// A shearing transformation in three dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Shear3D<S> {
    /// The matrix representing the affine transformation.
    matrix: Matrix4<S>,
}

impl<S> Shear3D<S> where S: Scalar {
    /// Construct a shearing transformation along the x-axis.
    ///
    /// The parameters `shear_x_with_y` and `shear_x_with_z` denote the multiplicative
    /// factors for the contributions from the y-axis and the z-axis respectively for the
    /// shearing along the x-axis.
    #[inline]
    pub fn from_shear_x(shear_x_with_y: S, shear_x_with_z: S) -> Shear3D<S> {
        Shear3D {
            matrix: Matrix4::from_shear_x(shear_x_with_y, shear_x_with_z),
        }
    }

    /// Construct a shearing transformation along the y-axis.
    ///
    /// The parameters `shear_y_with_x` and `shear_y_with_z` denote the multiplicative
    /// factors for the contributions from the x-axis and the z-axis respectively for the
    /// shearing along the y-axis.
    #[inline]
    pub fn from_shear_y(shear_y_with_x: S, shear_y_with_z: S) -> Shear3D<S> {
        Shear3D {
            matrix: Matrix4::from_shear_y(shear_y_with_x, shear_y_with_z),
        }
    }

    /// Construct a shearing transformation along the z-axis.
    ///
    /// The parameters `shear_z_with_x` and `shear_z_with_y` denote the multiplicative
    /// factors for the contributions from the x-axis and the y-axis respectively for the
    /// shearing along the z-axis.
    #[inline]
    pub fn from_shear_z(shear_z_with_x: S, shear_z_with_y: S) -> Shear3D<S> {
        Shear3D {
            matrix: Matrix4::from_shear_z(shear_z_with_x, shear_z_with_y),
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

impl<S> From<Shear3D<S>> for Matrix4<S> where S: Copy {
    fn from(transformation: Shear3D<S>) -> Matrix4<S> {
        transformation.matrix
    }
}

impl<S> From<&Shear3D<S>> for Matrix4<S> where S: Copy {
    fn from(transformation: &Shear3D<S>) -> Matrix4<S> {
        transformation.matrix
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

    #[rustfmt::skip]
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

    #[rustfmt::skip]
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

    #[rustfmt::skip]
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

    #[rustfmt::skip]
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

