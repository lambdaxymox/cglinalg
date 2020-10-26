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
use crate::array::*;

use core::ops;


macro_rules! impl_matrix_matrix_mul_ops {
    ($MatrixNxM:ident, $MatrixMxP:ident => $Output:ident, [$rows:expr; $cols:expr], $dot_arr_col:ident, { $( ($col:expr, $row:expr) ),* }) => {
        impl<S> ops::Mul<$MatrixMxP<S>> for $MatrixNxM<S> where S: Scalar {
            type Output = $Output<S>;

            #[inline]
            fn mul(self, other: $MatrixMxP<S>) -> Self::Output {
                Self::Output::new(
                    $( $dot_arr_col(
                        self.as_ref(), 
                        &<$MatrixMxP<S> as AsRef<[[S; $rows]; $cols]>>::as_ref(&other)[$col],
                        $row
                    ) ),*
                )
            }
        }

        impl<S> ops::Mul<&$MatrixMxP<S>> for $MatrixNxM<S> where S: Scalar {
            type Output = $Output<S>;

            #[inline]
            fn mul(self, other: &$MatrixMxP<S>) -> Self::Output {
                Self::Output::new(
                    $( $dot_arr_col(
                        self.as_ref(), 
                        &<$MatrixMxP<S> as AsRef<[[S; $rows]; $cols]>>::as_ref(&other)[$col],
                        $row
                    ) ),*
                )
            }
        }

        impl<S> ops::Mul<$MatrixMxP<S>> for &$MatrixNxM<S> where S: Scalar {
            type Output = $Output<S>;

            #[inline]
            fn mul(self, other: $MatrixMxP<S>) -> Self::Output {
                Self::Output::new(
                    $( $dot_arr_col(
                        self.as_ref(), 
                        &<$MatrixMxP<S> as AsRef<[[S; $rows]; $cols]>>::as_ref(&other)[$col], 
                        $row
                    ) ),*
                )
            }
        }

        impl<'a, 'b, S> ops::Mul<&'a $MatrixMxP<S>> for &'b $MatrixNxM<S> where S: Scalar {
            type Output = $Output<S>;

            #[inline]
            fn mul(self, other: &'a $MatrixMxP<S>) -> Self::Output {
                Self::Output::new(
                    $( $dot_arr_col(
                        self.as_ref(), 
                        &<$MatrixMxP<S> as AsRef<[[S; $rows]; $cols]>>::as_ref(&other)[$col], 
                        $row
                    ) ),*
                )
            }
        }
    }
}

impl_matrix_matrix_mul_ops!(
    Matrix1x1, Matrix1x1 => Matrix1x1, [1; 1], dot_array1x1_col1,
    { (0, 0) }
);
impl_matrix_matrix_mul_ops!(
    Matrix1x1, Matrix1x2 => Matrix1x2, [1; 2], dot_array1x1_col1,
    { (0, 0), (1, 0) }
);

impl_matrix_matrix_mul_ops!(
    Matrix1x1, Matrix1x3 => Matrix1x3, [1; 3], dot_array1x1_col1,
    { (0, 0), (1, 0), (2, 0) }
);
impl_matrix_matrix_mul_ops!(
    Matrix1x1, Matrix1x4 => Matrix1x4, [1; 4], dot_array1x1_col1,
    { (0, 0), (1, 0), (2, 0), (3, 0) }
);

impl_matrix_matrix_mul_ops!(
    Matrix2x2, Matrix2x2 => Matrix2x2, [2; 2], dot_array2x2_col2,
    { (0, 0), (0, 1), (1, 0), (1, 1) }
);

impl_matrix_matrix_mul_ops!(
    Matrix3x3, Matrix3x3 => Matrix3x3, [3; 3], dot_array3x3_col3, { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)
});

impl_matrix_matrix_mul_ops!(
    Matrix4x4, Matrix4x4 => Matrix4x4, [4; 4], dot_array4x4_col4, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});

impl_matrix_matrix_mul_ops!(
    Matrix1x2, Matrix2x2 => Matrix1x2, [2; 2], dot_array1x2_col2,
    { (0, 0), (1, 0) }
);

impl_matrix_matrix_mul_ops!(
    Matrix1x3, Matrix3x3 => Matrix1x3, [3; 3], dot_array1x3_col3,
    { (0, 0), (1, 0), (2, 0) }
);

impl_matrix_matrix_mul_ops!(
    Matrix1x4, Matrix4x4 => Matrix1x4, [4; 4], dot_array1x4_col4,
    { (0, 0), (1, 0), (2, 0), (3, 0) }
);

impl_matrix_matrix_mul_ops!(
    Matrix2x3, Matrix3x3 => Matrix2x3, [3; 3], dot_array2x3_col3,
    { (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1) }
);
impl_matrix_matrix_mul_ops!(
    Matrix2x3, Matrix3x2 => Matrix2x2, [3; 2], dot_array2x3_col3,
    { (0, 0), (0, 1), (1, 0), (1, 1) }
);
impl_matrix_matrix_mul_ops!(
    Matrix2x2, Matrix2x3 => Matrix2x3, [2; 3], dot_array2x2_col2,
    { (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1) }
);
impl_matrix_matrix_mul_ops!(
    Matrix1x2, Matrix2x3 => Matrix1x3, [2; 3], dot_array1x2_col2,
    { (0, 0), (1, 0), (2, 0) }
);

impl_matrix_matrix_mul_ops!(
    Matrix3x2, Matrix2x2 => Matrix3x2, [2; 2], dot_array3x2_col2,
    { (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2) }
);
impl_matrix_matrix_mul_ops!(
    Matrix3x2, Matrix2x3 => Matrix3x3, [2; 3], dot_array3x2_col2,
    { (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) }
);
impl_matrix_matrix_mul_ops!(
    Matrix3x3, Matrix3x2 => Matrix3x2, [3; 2], dot_array3x3_col3,
    { (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2) }
);
impl_matrix_matrix_mul_ops!(
    Matrix1x3, Matrix3x2 => Matrix1x2, [3; 2], dot_array1x3_col3,
    { (0, 0), (1, 0) }
);

impl_matrix_matrix_mul_ops!(
    Matrix2x4, Matrix4x4 => Matrix2x4, [4; 4], dot_array2x4_col4, { 
    (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)
});
impl_matrix_matrix_mul_ops!(
    Matrix2x2, Matrix2x4 => Matrix2x4, [2; 4], dot_array2x2_col2, { 
    (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)
});
impl_matrix_matrix_mul_ops!(
    Matrix2x4, Matrix4x2 => Matrix2x2, [4; 2], dot_array2x4_col4,
    { (0, 0), (0, 1), (1, 0), (1, 1) }
);
impl_matrix_matrix_mul_ops!(
    Matrix1x2, Matrix2x4 => Matrix1x4, [2; 4], dot_array1x2_col2, 
    { (0, 0), (1, 0), (2, 0), (3, 0) }
);

impl_matrix_matrix_mul_ops!(
    Matrix4x2, Matrix2x2 => Matrix4x2, [2; 2], dot_array4x2_col2, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_matrix_matrix_mul_ops!(
    Matrix4x2, Matrix2x4 => Matrix4x4, [2; 4], dot_array4x2_col2, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 0), (3, 1), (3, 2), (3, 3)
});
impl_matrix_matrix_mul_ops!(
    Matrix4x2, Matrix2x3 => Matrix4x3, [2; 3], dot_array4x2_col2, {
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_matrix_matrix_mul_ops!(
    Matrix4x4, Matrix4x2 => Matrix4x2, [4; 2], dot_array4x4_col4, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_matrix_matrix_mul_ops!(
    Matrix1x4, Matrix4x2 => Matrix1x2, [4; 2], dot_array1x4_col4, {
    (0, 0), (1, 0)
});
impl_matrix_matrix_mul_ops!(
    Matrix3x4, Matrix4x2 => Matrix3x2, [4; 2], dot_array3x4_col4, {
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)
});

impl_matrix_matrix_mul_ops!(
    Matrix3x4, Matrix4x4 => Matrix3x4, [4; 4], dot_array3x4_col4, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_matrix_matrix_mul_ops!(
    Matrix3x4, Matrix4x3 => Matrix3x3, [4; 3], dot_array3x4_col4, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2)
});
impl_matrix_matrix_mul_ops!(
    Matrix3x3, Matrix3x4 => Matrix3x4, [3; 4], dot_array3x3_col3, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2),
    (3, 0), (3, 1), (3, 2)
});
impl_matrix_matrix_mul_ops!(
    Matrix1x3, Matrix3x4 => Matrix1x4, [3; 4], dot_array1x3_col3, {
    (0, 0), (1, 0), (2, 0), (3, 0)
});

impl_matrix_matrix_mul_ops!(
    Matrix4x3, Matrix3x3 => Matrix4x3, [3; 3], dot_array4x3_col3, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_matrix_matrix_mul_ops!(
    Matrix4x3, Matrix3x4 => Matrix4x4, [3; 4], dot_array4x3_col3, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 0), (3, 1), (3, 2), (3, 3)
});
impl_matrix_matrix_mul_ops!(
    Matrix4x4, Matrix4x3 => Matrix4x3, [4; 3], dot_array4x4_col4, {
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_matrix_matrix_mul_ops!(
    Matrix1x4, Matrix4x3 => Matrix1x3, [4; 3], dot_array1x4_col4, {
    (0, 0), (1, 0), (2, 0)
});
impl_matrix_matrix_mul_ops!(
    Matrix2x4, Matrix4x3 => Matrix2x3, [4; 3], dot_array2x4_col4, {
    (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)
});
impl_matrix_matrix_mul_ops!(
    Matrix4x3, Matrix3x2 => Matrix4x2, [3; 2], dot_array4x3_col3, {
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});

