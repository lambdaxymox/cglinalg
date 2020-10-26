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

use core::ops;


macro_rules! impl_matrix_binary_assign_ops {
    ($T:ty, { $( ($col:expr, $row:expr) ),* }) => {
        impl<S> ops::AddAssign<$T> for $T where S: Scalar {
            #[inline]
            fn add_assign(&mut self, other: $T) {
                $( self[$col][$row] += other[$col][$row] );*
            }
        }

        impl<S> ops::AddAssign<&$T> for $T where S: Scalar {
            #[inline]
            fn add_assign(&mut self, other: &$T) {
                $( self[$col][$row] += other[$col][$row] );*
            }
        }

        impl<S> ops::SubAssign<$T> for $T where S: Scalar {
            #[inline]
            fn sub_assign(&mut self, other: $T) {
                $( self[$col][$row] -= other[$col][$row] );*
            }
        }

        impl<S> ops::SubAssign<&$T> for $T where S: Scalar {
            #[inline]
            fn sub_assign(&mut self, other: &$T) {
                $( self[$col][$row] -= other[$col][$row] );*
            }
        }

        impl<S> ops::MulAssign<S> for $T where S: Scalar {
            #[inline]
            fn mul_assign(&mut self, other: S) {
                $( self[$col][$row] *= other );*
            }
        }
        
        impl<S> ops::DivAssign<S> for $T where S: Scalar {
            #[inline]
            fn div_assign(&mut self, other: S) {
                $( self[$col][$row] /= other );*
            }
        }
        
        impl<S> ops::RemAssign<S> for $T where S: Scalar {
            #[inline]
            fn rem_assign(&mut self, other: S) {
                $( self[$col][$row] %= other );*
            }
        }
    }
}

impl_matrix_binary_assign_ops!(
    Matrix1x1<S>, { 
    (0, 0) 
});
impl_matrix_binary_assign_ops!(
    Matrix2x2<S>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});
impl_matrix_binary_assign_ops!(
    Matrix3x3<S>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_matrix_binary_assign_ops!(
    Matrix4x4<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_matrix_binary_assign_ops!(
    Matrix1x2<S>, { 
    (0, 0),
    (1, 0)
});
impl_matrix_binary_assign_ops!(
    Matrix1x3<S>, { 
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_matrix_binary_assign_ops!(
    Matrix1x4<S>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_matrix_binary_assign_ops!(
    Matrix2x3<S>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1)
});
impl_matrix_binary_assign_ops!(
    Matrix2x4<S>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_matrix_binary_assign_ops!(
    Matrix3x2<S>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2) 
});
impl_matrix_binary_assign_ops!(
    Matrix3x4<S>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_matrix_binary_assign_ops!(
    Matrix4x2<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_matrix_binary_assign_ops!(
    Matrix4x3<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});

