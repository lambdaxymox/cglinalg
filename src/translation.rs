use scalar::{
    Scalar,
    ScalarSigned,
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
use affine::*;

use std::fmt;


/// A translation transformation in two dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Translation2D<S> {
    /// The matrix representing the affine transformation.
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

impl<S> From<Translation2D<S>> for Matrix3<S> where S: Copy {
    fn from(transformation: Translation2D<S>) -> Matrix3<S> {
        transformation.matrix
    }
}

impl<S> From<&Translation2D<S>> for Matrix3<S> where S: Copy {
    fn from(transformation: &Translation2D<S>) -> Matrix3<S> {
        transformation.matrix
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
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_inverse(&self, vector: Vector2<S>) -> Option<Vector2<S>> {
        let inverse_matrix = <Self as AffineTransformation2D<Vector2<S>>>::inverse(&self).unwrap().matrix;
        Some((inverse_matrix * vector.extend(S::zero())).contract())
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
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_inverse(&self, vector: &Vector2<S>) -> Option<Vector2<S>> {
        let inverse_matrix = <Self as AffineTransformation2D<Vector2<S>>>::inverse(&self).unwrap().matrix;
        Some((inverse_matrix * vector.extend(S::zero())).contract())
    }
}


/// A translation transformation in three dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Translation3D<S> {
    /// The matrix representing the affine transformation.
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

impl<S> From<Translation3D<S>> for Matrix4<S> where S: Copy {
    fn from(transformation: Translation3D<S>) -> Matrix4<S> {
        transformation.matrix
    }
}

impl<S> From<&Translation3D<S>> for Matrix4<S> where S: Copy {
    fn from(transformation: &Translation3D<S>) -> Matrix4<S> {
        transformation.matrix
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
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_inverse(&self, vector: Vector3<S>) -> Option<Vector3<S>> {
        let inverse_matrix = <Self as AffineTransformation3D<Vector3<S>>>::inverse(&self).unwrap().matrix;
        Some((inverse_matrix * vector.extend(S::zero())).contract())
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

    #[rustfmt::skip]
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
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_inverse(&self, vector: &Vector3<S>) -> Option<Vector3<S>> {
        let inverse_matrix = <Self as AffineTransformation3D<Vector3<S>>>::inverse(&self).unwrap().matrix;
        Some((inverse_matrix * vector.extend(S::zero())).contract())
    }
}

