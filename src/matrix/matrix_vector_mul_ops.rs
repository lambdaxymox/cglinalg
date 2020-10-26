use crate::base::{
    Scalar,
};
use crate::matrix::{
    Matrix1x1,
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
    Matrix1x2,
    Matrix1x3,
    Matrix1x4,
    Matrix2x3,
    Matrix2x4,
    Matrix3x2,
    Matrix3x4,
    Matrix4x2,
    Matrix4x3,
};
use crate::vector::{
    Vector1,
    Vector2,
    Vector3,
    Vector4,
};

use crate::array::*;

use core::ops;


macro_rules! impl_matrix_vector_mul_ops {
    ($MatrixMxN:ident, $VectorM:ident => $Output:ident, $dot_arr_col:ident, { $( ($col:expr, $row:expr) ),* }) => {
        impl<S> ops::Mul<$VectorM<S>> for $MatrixMxN<S> where S: Scalar {
            type Output = $Output<S>;

            #[inline]
            fn mul(self, other: $VectorM<S>) -> Self::Output {
                Self::Output::new(
                    $( $dot_arr_col(self.as_ref(), other.as_ref(), $row) ),*
                )
            }
        }

        impl<S> ops::Mul<&$VectorM<S>> for $MatrixMxN<S> where S: Scalar {
            type Output = $Output<S>;

            #[inline]
            fn mul(self, other: &$VectorM<S>) -> Self::Output {
                Self::Output::new(
                    $( $dot_arr_col(self.as_ref(), other.as_ref(), $row) ),*
                )
            }
        }

        impl<S> ops::Mul<$VectorM<S>> for &$MatrixMxN<S> where S: Scalar {
            type Output = $Output<S>;

            #[inline]
            fn mul(self, other: $VectorM<S>) -> Self::Output {
                Self::Output::new(
                    $( $dot_arr_col(self.as_ref(), other.as_ref(), $row) ),*
                )
            }
        }

        impl<'a, 'b, S> ops::Mul<&'a $VectorM<S>> for &'b $MatrixMxN<S> where S: Scalar {
            type Output = $Output<S>;

            #[inline]
            fn mul(self, other: &'a $VectorM<S>) -> Self::Output {
                Self::Output::new(
                    $( $dot_arr_col(self.as_ref(), other.as_ref(), $row) ),*
                )
            }
        }
    }
}

impl_matrix_vector_mul_ops!(
    Matrix1x1, Vector1 => Vector1, dot_array1x1_col1,
    { (0, 0) }
);
impl_matrix_vector_mul_ops!(
    Matrix2x2, Vector2 => Vector2, dot_array2x2_col2,
    { (0, 0), (0, 1) }
);
impl_matrix_vector_mul_ops!(
    Matrix3x3, Vector3 => Vector3, dot_array3x3_col3,
    { (0, 0), (0, 1), (0, 2) }
);
impl_matrix_vector_mul_ops!(
    Matrix4x4, Vector4 => Vector4, dot_array4x4_col4,
    { (0, 0), (0, 1), (0, 2), (0, 3) }
);
impl_matrix_vector_mul_ops!(
    Matrix1x2, Vector2 => Vector1, dot_array1x2_col2,
    { (0, 0) }
);
impl_matrix_vector_mul_ops!(
    Matrix1x3, Vector3 => Vector1, dot_array1x3_col3,
    { (0, 0) }
);
impl_matrix_vector_mul_ops!(
    Matrix1x4, Vector4 => Vector1, dot_array1x4_col4,
    { (0, 0) }
);
impl_matrix_vector_mul_ops!(
    Matrix2x3, Vector3 => Vector2, dot_array2x3_col3,
    { (0, 0), (0, 1) }
);
impl_matrix_vector_mul_ops!(
    Matrix3x2, Vector2 => Vector3, dot_array3x2_col2,
    { (0, 0), (0, 1), (0, 2) }
);
impl_matrix_vector_mul_ops!(
    Matrix2x4, Vector4 => Vector2, dot_array2x4_col4,
    { (0, 0), (0, 1) }
);
impl_matrix_vector_mul_ops!(
    Matrix4x2, Vector2 => Vector4, dot_array4x2_col2,
    { (0, 0), (0, 1), (0, 2), (0, 3) }
);
impl_matrix_vector_mul_ops!(
    Matrix3x4, Vector4 => Vector3, dot_array3x4_col4,
    { (0, 0), (0, 1), (0, 2) }
);
impl_matrix_vector_mul_ops!(
    Matrix4x3, Vector3 => Vector4, dot_array4x3_col3,
    { (0, 0), (0, 1), (0, 2), (0, 3) }
);
