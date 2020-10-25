use crate::base::{
    Scalar,
};
use crate::array::*;
use crate::matrix::*;

use core::ops::{
    Add,
    Sub,
};


macro_rules! impl_matrix_matrix_binary_ops {
    ($OpType:ident, $op:ident, $op_impl:ident, $T:ty, $Output:ty, { $( ($col:expr, $row:expr) ),* }) => {
        impl<S> $OpType<$T> for $T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: $T) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(self.as_ref(), other.as_ref(), $col, $row) ),* 
                )
            }
        }

        impl<S> $OpType<&$T> for $T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: &$T) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(self.as_ref(), other.as_ref(), $col, $row) ),* 
                )
            }
        }

        impl<S> $OpType<$T> for &$T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: $T) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(self.as_ref(), other.as_ref(), $col, $row) ),* 
                )
            }
        }

        impl<'a, 'b, S> $OpType<&'a $T> for &'b $T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: &'a $T) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(self.as_ref(), other.as_ref(), $col, $row) ),* 
                )
            }
        }
    }
}

impl_matrix_matrix_binary_ops!(
    Add, add, add_array1x1_array1x1, Matrix1x1<S>, Matrix1x1<S>, 
    { (0, 0) }
);
impl_matrix_matrix_binary_ops!(
    Sub, sub, sub_array1x1_array1x1, Matrix1x1<S>, Matrix1x1<S>, 
    { (0, 0) }
);

impl_matrix_matrix_binary_ops!(
    Add, add, 
    add_array2x2_array2x2, Matrix2x2<S>, Matrix2x2<S>, 
    { (0, 0), (0, 1), (1, 0), (1, 1) }
);
impl_matrix_matrix_binary_ops!(
    Sub, sub, sub_array2x2_array2x2, Matrix2x2<S>, Matrix2x2<S>, 
    { (0, 0), (0, 1), (1, 0), (1, 1) }
);

impl_matrix_matrix_binary_ops!(
    Add, add, add_array3x3_array3x3, Matrix3x3<S>, Matrix3x3<S>, { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_matrix_matrix_binary_ops!(
    Sub, sub, sub_array3x3_array3x3, Matrix3x3<S>, Matrix3x3<S>, { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});

impl_matrix_matrix_binary_ops!(
    Add, add, add_array4x4_array4x4, Matrix4x4<S>, Matrix4x4<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_matrix_matrix_binary_ops!(
    Sub, sub, sub_array4x4_array4x4, Matrix4x4<S>, Matrix4x4<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3) 
});

impl_matrix_matrix_binary_ops!(
    Add, add, add_array1x2_array1x2, Matrix1x2<S>, Matrix1x2<S>, 
    { (0, 0), (1, 0) }
);
impl_matrix_matrix_binary_ops!(
    Sub, sub, sub_array1x2_array1x2, Matrix1x2<S>, Matrix1x2<S>, 
    { (0, 0), (1, 0) }
);

impl_matrix_matrix_binary_ops!(
    Add, add, add_array1x3_array1x3, Matrix1x3<S>, Matrix1x3<S>, 
    { (0, 0), (1, 0), (2, 0) }
);
impl_matrix_matrix_binary_ops!(
    Sub, sub, sub_array1x3_array1x3, Matrix1x3<S>, Matrix1x3<S>, 
    { (0, 0), (1, 0), (2, 0) }
);

impl_matrix_matrix_binary_ops!(
    Add, add, add_array1x4_array1x4, Matrix1x4<S>, Matrix1x4<S>, 
    { (0, 0), (1, 0), (2, 0), (3, 0) }
);
impl_matrix_matrix_binary_ops!(
    Sub, sub, sub_array1x4_array1x4, Matrix1x4<S>, Matrix1x4<S>, 
    { (0, 0), (1, 0), (2, 0), (3, 0) }
);

impl_matrix_matrix_binary_ops!(
    Add, add, add_array2x3_array2x3, Matrix2x3<S>, Matrix2x3<S>, 
    { (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1) }
);
impl_matrix_matrix_binary_ops!(
    Sub, sub, sub_array2x3_array2x3, Matrix2x3<S>, Matrix2x3<S>, 
    { (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1) }
);

impl_matrix_matrix_binary_ops!(
    Add, add, add_array3x2_array3x2, Matrix3x2<S>, Matrix3x2<S>, 
    { (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2) }
);
impl_matrix_matrix_binary_ops!(
    Sub, sub, sub_array3x2_array3x2, Matrix3x2<S>, Matrix3x2<S>, 
    { (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2) }
);

impl_matrix_matrix_binary_ops!(
    Add, add, add_array2x4_array2x4, Matrix2x4<S>, Matrix2x4<S>, { 
    (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)
});
impl_matrix_matrix_binary_ops!(
    Sub, sub, sub_array2x4_array2x4, Matrix2x4<S>, Matrix2x4<S>, {
    (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)
});

impl_matrix_matrix_binary_ops!(
    Add, add, add_array4x2_array4x2, Matrix4x2<S>, Matrix4x2<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_matrix_matrix_binary_ops!(
    Sub, sub, sub_array4x2_array4x2, Matrix4x2<S>, Matrix4x2<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});

impl_matrix_matrix_binary_ops!(
    Add, add, add_array3x4_array3x4, Matrix3x4<S>, Matrix3x4<S>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_matrix_matrix_binary_ops!(
    Sub, sub, sub_array3x4_array3x4, Matrix3x4<S>, Matrix3x4<S>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});

impl_matrix_matrix_binary_ops!(
    Add, add, add_array4x3_array4x3, Matrix4x3<S>, Matrix4x3<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_matrix_matrix_binary_ops!(
    Sub, sub, sub_array4x3_array4x3, Matrix4x3<S>, Matrix4x3<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});

