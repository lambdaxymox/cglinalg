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
    InvertibleSquareMatrix,
};
use affine::*;

use std::fmt;


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform2D<S> {
    matrix: Matrix3<S>,
}

impl<S> AsRef<Matrix3<S>> for Transform2D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix3<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Transform2D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Transform2D<S>> for Matrix3<S> where S: Copy {
    fn from(transformation: Transform2D<S>) -> Matrix3<S> {
        transformation.matrix
    }
}

impl<S> From<&Transform2D<S>> for Matrix3<S> where S: Copy {
    fn from(transformation: &Transform2D<S>) -> Matrix3<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation2D<Point2<S>, Vector2<S>> for Transform2D<S> where S: ScalarFloat {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Transform2D<S> {
        Transform2D { 
            matrix: Matrix3::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Transform2D<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform2D {
                matrix: matrix
            })
        } else {
            None
        }
    }

    #[inline]
    fn apply_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_point(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> AffineTransformation2D<Point2<S>, &Vector2<S>> for Transform2D<S> where S: ScalarFloat {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Transform2D<S> {
        Transform2D { 
            matrix: Matrix3::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Transform2D<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform2D {
                matrix: matrix
            })
        } else {
            None
        }
    }

    #[inline]
    fn apply_vector(&self, vector: &Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_point(&self, point: Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> AffineTransformation2D<&Point2<S>, Vector2<S>> for Transform2D<S> where S: ScalarFloat {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Transform2D<S> {
        Transform2D { 
            matrix: Matrix3::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Transform2D<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform2D {
                matrix: matrix
            })
        } else {
            None
        }
    }

    #[inline]
    fn apply_vector(&self, vector: Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_point(&self, point: &Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<'a, 'b, S> AffineTransformation2D<&'a Point2<S>, &'b Vector2<S>> for Transform2D<S> where S: ScalarFloat {
    type OutPoint = Point2<S>;
    type OutVector = Vector2<S>;

    #[inline]
    fn identity() -> Transform2D<S> {
        Transform2D { 
            matrix: Matrix3::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Transform2D<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform2D {
                matrix: matrix
            })
        } else {
            None
        }
    }

    #[inline]
    fn apply_vector(&self, vector: &'b Vector2<S>) -> Vector2<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_point(&self, point: &'a Point2<S>) -> Point2<S> {
        Point2::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}



#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform3D<S> {
    matrix: Matrix4<S>,
}

impl<S> AsRef<Matrix4<S>> for Transform3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Transform3D<S> where S: Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> From<Transform3D<S>> for Matrix4<S> where S: Copy {
    fn from(transformation: Transform3D<S>) -> Matrix4<S> {
        transformation.matrix
    }
}

impl<S> From<&Transform3D<S>> for Matrix4<S> where S: Copy {
    fn from(transformation: &Transform3D<S>) -> Matrix4<S> {
        transformation.matrix
    }
}

impl<S> AffineTransformation3D<Point3<S>, Vector3<S>> for Transform3D<S> where S: ScalarFloat {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Transform3D<S> {
        Transform3D { 
            matrix: Matrix4::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Transform3D<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform3D {
                matrix: matrix
            })
        } else {
            None
        }
    }

    #[inline]
    fn apply_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_point(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> AffineTransformation3D<Point3<S>, &Vector3<S>> for Transform3D<S> where S: ScalarFloat {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Transform3D<S> {
        Transform3D { 
            matrix: Matrix4::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Transform3D<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform3D {
                matrix: matrix
            })
        } else {
            None
        }
    }

    #[inline]
    fn apply_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_point(&self, point: Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<S> AffineTransformation3D<&Point3<S>, Vector3<S>> for Transform3D<S> where S: ScalarFloat {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Transform3D<S> {
        Transform3D { 
            matrix: Matrix4::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Transform3D<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform3D {
                matrix: matrix
            })
        } else {
            None
        }
    }

    #[inline]
    fn apply_vector(&self, vector: Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}

impl<'a, 'b, S> AffineTransformation3D<&'a Point3<S>, &'b Vector3<S>> for Transform3D<S> where S: ScalarFloat {
    type OutPoint = Point3<S>;
    type OutVector = Vector3<S>;

    #[inline]
    fn identity() -> Transform3D<S> {
        Transform3D { 
            matrix: Matrix4::one(),
        }
    }

    #[inline]
    fn inverse(&self) -> Option<Transform3D<S>> {
        if let Some(matrix) = self.matrix.inverse() {
            Some(Transform3D {
                matrix: matrix
            })
        } else {
            None
        }
    }

    #[inline]
    fn apply_vector(&self, vector: &'b Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.extend(S::zero())).contract()
    }

    #[inline]
    fn apply_point(&self, point: &'a Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }
}


