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
use crate::matrix::array::*;

use core::ops::{
    Mul,
    Div,
    Rem,
};


macro_rules! impl_matrix_scalar_binary_ops {
    ($OpType:ident, $op:ident, $op_impl:ident, $T:ty, $Output:ty, { $( ($col:expr, $row:expr) ),* }) => {
        impl<S> $OpType<S> for $T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: S) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(self.as_ref(), other, $col, $row) ),* 
                )
            }
        }

        impl<S> $OpType<S> for &$T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: S) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(self.as_ref(), other, $col, $row) ),* 
                )
            }
        }
    }
}

impl_matrix_scalar_binary_ops!(
    Mul, mul, mul_array1x1_scalar, Matrix1x1<S>, Matrix1x1<S>, {
    (0, 0)
});
impl_matrix_scalar_binary_ops!(
    Div, div, div_array1x1_scalar, Matrix1x1<S>, Matrix1x1<S>, {
    (0, 0)
});
impl_matrix_scalar_binary_ops!(
    Rem, rem, rem_array1x1_scalar, Matrix1x1<S>, Matrix1x1<S>, {
    (0, 0)
});

impl_matrix_scalar_binary_ops!(
    Mul, mul, mul_array2x2_scalar, Matrix2x2<S>, Matrix2x2<S>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1)
});
impl_matrix_scalar_binary_ops!(
    Div, div, div_array2x2_scalar, Matrix2x2<S>, Matrix2x2<S>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1)
});
impl_matrix_scalar_binary_ops!(
    Rem, rem, rem_array2x2_scalar, Matrix2x2<S>, Matrix2x2<S>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});

impl_matrix_scalar_binary_ops!(
    Mul, mul, mul_array3x3_scalar, Matrix3x3<S>, Matrix3x3<S>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2) 
});
impl_matrix_scalar_binary_ops!(
    Div, div, div_array3x3_scalar, Matrix3x3<S>, Matrix3x3<S>, { 
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2) 
});
impl_matrix_scalar_binary_ops!(
    Rem, rem, rem_array3x3_scalar, Matrix3x3<S>, Matrix3x3<S>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});

impl_matrix_scalar_binary_ops!(
    Mul, mul, mul_array4x4_scalar, Matrix4x4<S>, Matrix4x4<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_matrix_scalar_binary_ops!(
    Div, div, div_array4x4_scalar, Matrix4x4<S>, Matrix4x4<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_matrix_scalar_binary_ops!(
    Rem, rem, rem_array4x4_scalar, Matrix4x4<S>, Matrix4x4<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 0), (3, 1), (3, 2), (3, 3)
});

impl_matrix_scalar_binary_ops!(
    Mul, mul, mul_array1x2_scalar, Matrix1x2<S>, Matrix1x2<S>, {
    (0, 0), 
    (1, 0)
});
impl_matrix_scalar_binary_ops!(
    Div, div, div_array1x2_scalar, Matrix1x2<S>, Matrix1x2<S>, {
    (0, 0),
    (1, 0)
});
impl_matrix_scalar_binary_ops!(
    Rem, rem, rem_array1x2_scalar, Matrix1x2<S>, Matrix1x2<S>, {
    (0, 0), 
    (1, 0) 
});

impl_matrix_scalar_binary_ops!(
    Mul, mul, mul_array1x3_scalar, Matrix1x3<S>, Matrix1x3<S>, { 
    (0, 0), 
    (1, 0), 
    (2, 0)
});
impl_matrix_scalar_binary_ops!(
    Div, div, div_array1x3_scalar, Matrix1x3<S>, Matrix1x3<S>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_matrix_scalar_binary_ops!(
    Rem, rem, rem_array1x3_scalar, Matrix1x3<S>, Matrix1x3<S>, {
    (0, 0),
    (1, 0),
    (2, 0)
});

impl_matrix_scalar_binary_ops!(
    Mul, mul, mul_array1x4_scalar, Matrix1x4<S>, Matrix1x4<S>, {
    (0, 0), 
    (1, 0), 
    (2, 0), 
    (3, 0)
});
impl_matrix_scalar_binary_ops!(
    Div, div, div_array1x4_scalar, Matrix1x4<S>, Matrix1x4<S>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_matrix_scalar_binary_ops!(
    Rem, rem, rem_array1x4_scalar, Matrix1x4<S>, Matrix1x4<S>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});

impl_matrix_scalar_binary_ops!(
    Mul, mul, mul_array2x3_scalar, Matrix2x3<S>, Matrix2x3<S>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_matrix_scalar_binary_ops!(
    Div, div, div_array2x3_scalar, Matrix2x3<S>, Matrix2x3<S>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1)
});
impl_matrix_scalar_binary_ops!(
    Rem, rem, rem_array2x3_scalar, Matrix2x3<S>, Matrix2x3<S>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1)
});

impl_matrix_scalar_binary_ops!(
    Mul, mul, mul_array2x4_scalar, Matrix2x4<S>, Matrix2x4<S>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_matrix_scalar_binary_ops!(
    Div, div, div_array2x4_scalar, Matrix2x4<S>, Matrix2x4<S>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_matrix_scalar_binary_ops!(
    Rem, rem, rem_array2x4_scalar, Matrix2x4<S>, Matrix2x4<S>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});

impl_matrix_scalar_binary_ops!(
    Mul, mul, mul_array3x2_scalar, Matrix3x2<S>, Matrix3x2<S>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2)
});
impl_matrix_scalar_binary_ops!(
    Div, div, div_array3x2_scalar, Matrix3x2<S>, Matrix3x2<S>, { 
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
});
impl_matrix_scalar_binary_ops!(
    Rem, rem, rem_array3x2_scalar, Matrix3x2<S>, Matrix3x2<S>, { 
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
});

impl_matrix_scalar_binary_ops!(
    Mul, mul, mul_array3x4_scalar, Matrix3x4<S>, Matrix3x4<S>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_matrix_scalar_binary_ops!(
    Div, div, div_array3x4_scalar, Matrix3x4<S>, Matrix3x4<S>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_matrix_scalar_binary_ops!(
    Rem, rem, rem_array3x4_scalar, Matrix3x4<S>, Matrix3x4<S>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});

impl_matrix_scalar_binary_ops!(
    Mul, mul, mul_array4x2_scalar, Matrix4x2<S>, Matrix4x2<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_matrix_scalar_binary_ops!(
    Div, div, div_array4x2_scalar, Matrix4x2<S>, Matrix4x2<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_matrix_scalar_binary_ops!(
    Rem, rem, rem_array4x2_scalar, Matrix4x2<S>, Matrix4x2<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});

impl_matrix_scalar_binary_ops!(
    Mul, mul, mul_array4x3_scalar, Matrix4x3<S>, Matrix4x3<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_matrix_scalar_binary_ops!(
    Div, div, div_array4x3_scalar, Matrix4x3<S>, Matrix4x3<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_matrix_scalar_binary_ops!(
    Rem, rem, rem_array4x3_scalar, Matrix4x3<S>, Matrix4x3<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});

