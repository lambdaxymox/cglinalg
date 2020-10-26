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


macro_rules! impl_scalar_matrix_mul_ops {
    ($Lhs:ty, $Rhs:ty, $Output:ty, { $( ($col:expr, $row:expr) ),* }) => {
        impl ops::Mul<$Rhs> for $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                <$Output>::new( $(self * other[$col][$row]),* )
            }
        }

        impl<'a> ops::Mul<$Rhs> for &'a $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                <$Output>::new( $(self * other[$col][$row]),* )
            }
        }
    }
}

impl_scalar_matrix_mul_ops!(
    u8,    Matrix1x1<u8>,    Matrix1x1<u8>, { 
    (0, 0)
});
impl_scalar_matrix_mul_ops!(
    u16,   Matrix1x1<u16>,   Matrix1x1<u16>, { 
    (0, 0) 
});
impl_scalar_matrix_mul_ops!(
    u32,   Matrix1x1<u32>,   Matrix1x1<u32>, { 
    (0, 0)
});
impl_scalar_matrix_mul_ops!(
    u64,   Matrix1x1<u64>,   Matrix1x1<u64>, { 
    (0, 0)
});
impl_scalar_matrix_mul_ops!(
    u128,  Matrix1x1<u128>,  Matrix1x1<u128>, { 
    (0, 0)
});
impl_scalar_matrix_mul_ops!(
    usize, Matrix1x1<usize>, Matrix1x1<usize>, { 
    (0, 0)
});
impl_scalar_matrix_mul_ops!(
    i8,    Matrix1x1<i8>,    Matrix1x1<i8>, { 
    (0, 0) 
});
impl_scalar_matrix_mul_ops!(
    i16,   Matrix1x1<i16>,   Matrix1x1<i16>, { 
    (0, 0) 
});
impl_scalar_matrix_mul_ops!(
    i32,   Matrix1x1<i32>,   Matrix1x1<i32>, { 
    (0, 0) 
});
impl_scalar_matrix_mul_ops!(
    i64,   Matrix1x1<i64>,   Matrix1x1<i64>, { 
    (0, 0) 
});
impl_scalar_matrix_mul_ops!(
    i128,  Matrix1x1<i128>,  Matrix1x1<i128>, { 
    (0, 0) 
});
impl_scalar_matrix_mul_ops!(
    isize, Matrix1x1<isize>, Matrix1x1<isize>, { 
    (0, 0) 
});
impl_scalar_matrix_mul_ops!(
    f32,   Matrix1x1<f32>,   Matrix1x1<f32>, { 
    (0, 0) 
});
impl_scalar_matrix_mul_ops!(
    f64,   Matrix1x1<f64>,   Matrix1x1<f64>, { 
    (0, 0) 
});


impl_scalar_matrix_mul_ops!(
    u8,    Matrix2x2<u8>,    Matrix2x2<u8>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});
impl_scalar_matrix_mul_ops!(
    u16,   Matrix2x2<u16>,   Matrix2x2<u16>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});
impl_scalar_matrix_mul_ops!(
    u32,   Matrix2x2<u32>,   Matrix2x2<u32>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});
impl_scalar_matrix_mul_ops!(
    u64,   Matrix2x2<u64>,   Matrix2x2<u64>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});
impl_scalar_matrix_mul_ops!(
    u128,  Matrix2x2<u128>,  Matrix2x2<u128>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});
impl_scalar_matrix_mul_ops!(
    usize, Matrix2x2<usize>, Matrix2x2<usize>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});
impl_scalar_matrix_mul_ops!(
    i8,    Matrix2x2<i8>,    Matrix2x2<i8>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});
impl_scalar_matrix_mul_ops!(
    i16,   Matrix2x2<i16>,   Matrix2x2<i16>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});
impl_scalar_matrix_mul_ops!(
    i32,   Matrix2x2<i32>,   Matrix2x2<i32>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});
impl_scalar_matrix_mul_ops!(
    i64,   Matrix2x2<i64>,   Matrix2x2<i64>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});
impl_scalar_matrix_mul_ops!(
    i128,  Matrix2x2<i128>,  Matrix2x2<i128>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});
impl_scalar_matrix_mul_ops!(
    isize, Matrix2x2<isize>, Matrix2x2<isize>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});
impl_scalar_matrix_mul_ops!(
    f32,   Matrix2x2<f32>,   Matrix2x2<f32>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1) 
});
impl_scalar_matrix_mul_ops!(
    f64,   Matrix2x2<f64>,   Matrix2x2<f64>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1)
});

impl_scalar_matrix_mul_ops!(
    u8,    Matrix3x3<u8>,    Matrix3x3<u8>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops!(
    u16,   Matrix3x3<u16>,   Matrix3x3<u16>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops!(
    u32,   Matrix3x3<u32>,   Matrix3x3<u32>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops!(
    u64,   Matrix3x3<u64>,   Matrix3x3<u64>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops!(
    u128,  Matrix3x3<u128>,  Matrix3x3<u128>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops!(
    usize, Matrix3x3<usize>, Matrix3x3<usize>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops!(
    i8,    Matrix3x3<i8>,    Matrix3x3<i8>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops!(
    i16,   Matrix3x3<i16>,   Matrix3x3<i16>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops!(
    i32,   Matrix3x3<i32>,   Matrix3x3<i32>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops!(
    i64,   Matrix3x3<i64>,   Matrix3x3<i64>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops!(
    i128,  Matrix3x3<i128>,  Matrix3x3<i128>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops!(
    isize, Matrix3x3<isize>, Matrix3x3<isize>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops!(
    f32,   Matrix3x3<f32>,   Matrix3x3<f32>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops!(
    f64,   Matrix3x3<f64>,   Matrix3x3<f64>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2) 
});

impl_scalar_matrix_mul_ops!(
    u8,    Matrix4x4<u8>,    Matrix4x4<u8>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops!(
    u16,   Matrix4x4<u16>,   Matrix4x4<u16>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops!(
    u32,   Matrix4x4<u32>,   Matrix4x4<u32>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops!(
    u64,   Matrix4x4<u64>,   Matrix4x4<u64>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops!(
    u128,  Matrix4x4<u128>,  Matrix4x4<u128>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops!(
    usize, Matrix4x4<usize>, Matrix4x4<usize>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops!(
    i8,    Matrix4x4<i8>,    Matrix4x4<i8>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops!(
    i16,   Matrix4x4<i16>,   Matrix4x4<i16>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops!(
    i32,   Matrix4x4<i32>,   Matrix4x4<i32>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops!(
    i64,   Matrix4x4<i64>,   Matrix4x4<i64>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops!(
    i128,  Matrix4x4<i128>,  Matrix4x4<i128>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops!(
    isize, Matrix4x4<isize>, Matrix4x4<isize>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops!(
    f32,   Matrix4x4<f32>,   Matrix4x4<f32>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops!(
    f64,   Matrix4x4<f64>,   Matrix4x4<f64>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});

impl_scalar_matrix_mul_ops!(
    u8,    Matrix1x2<u8>,    Matrix1x2<u8>, { 
    (0, 0),
    (1, 0)
});
impl_scalar_matrix_mul_ops!(
    u16,   Matrix1x2<u16>,   Matrix1x2<u16>, { 
    (0, 0),
    (1, 0)
});
impl_scalar_matrix_mul_ops!(
    u32,   Matrix1x2<u32>,   Matrix1x2<u32>, {
    (0, 0), 
    (1, 0)
});
impl_scalar_matrix_mul_ops!(
    u64,   Matrix1x2<u64>,   Matrix1x2<u64>, { 
    (0, 0),
    (1, 0)
});
impl_scalar_matrix_mul_ops!(
    u128,  Matrix1x2<u128>,  Matrix1x2<u128>, { 
    (0, 0), 
    (1, 0)
});
impl_scalar_matrix_mul_ops!(
    usize, Matrix1x2<usize>, Matrix1x2<usize>, { 
    (0, 0), 
    (1, 0) 
});
impl_scalar_matrix_mul_ops!(
    i8,    Matrix1x2<i8>,    Matrix1x2<i8>, { 
    (0, 0),
    (1, 0)
});
impl_scalar_matrix_mul_ops!(
    i16,   Matrix1x2<i16>,   Matrix1x2<i16>, { 
    (0, 0), 
    (1, 0)
});
impl_scalar_matrix_mul_ops!(
    i32,   Matrix1x2<i32>,   Matrix1x2<i32>, { 
    (0, 0), 
    (1, 0)
});
impl_scalar_matrix_mul_ops!(
    i64,   Matrix1x2<i64>,   Matrix1x2<i64>, { 
    (0, 0), 
    (1, 0)
});
impl_scalar_matrix_mul_ops!(
    i128,  Matrix1x2<i128>,  Matrix1x2<i128>, { 
    (0, 0), 
    (1, 0)
});
impl_scalar_matrix_mul_ops!(
    isize, Matrix1x2<isize>, Matrix1x2<isize>, { 
    (0, 0), 
    (1, 0)
});
impl_scalar_matrix_mul_ops!(
    f32,   Matrix1x2<f32>,   Matrix1x2<f32>, { 
    (0, 0), 
    (1, 0)
});
impl_scalar_matrix_mul_ops!(
    f64,   Matrix1x2<f64>,   Matrix1x2<f64>, { 
    (0, 0), 
    (1, 0) 
});

impl_scalar_matrix_mul_ops!(
    u8,    Matrix1x3<u8>,    Matrix1x3<u8>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_scalar_matrix_mul_ops!(
    u16,   Matrix1x3<u16>,   Matrix1x3<u16>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_scalar_matrix_mul_ops!(
    u32,   Matrix1x3<u32>,   Matrix1x3<u32>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_scalar_matrix_mul_ops!(
    u64,   Matrix1x3<u64>,   Matrix1x3<u64>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_scalar_matrix_mul_ops!(
    u128,  Matrix1x3<u128>,  Matrix1x3<u128>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_scalar_matrix_mul_ops!(
    usize, Matrix1x3<usize>, Matrix1x3<usize>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_scalar_matrix_mul_ops!(
    i8,    Matrix1x3<i8>,    Matrix1x3<i8>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_scalar_matrix_mul_ops!(
    i16,   Matrix1x3<i16>,   Matrix1x3<i16>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_scalar_matrix_mul_ops!(
    i32,   Matrix1x3<i32>,   Matrix1x3<i32>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_scalar_matrix_mul_ops!(
    i64,   Matrix1x3<i64>,   Matrix1x3<i64>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_scalar_matrix_mul_ops!(
    i128,  Matrix1x3<i128>,  Matrix1x3<i128>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_scalar_matrix_mul_ops!(
    isize, Matrix1x3<isize>, Matrix1x3<isize>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_scalar_matrix_mul_ops!(
    f32,   Matrix1x3<f32>,   Matrix1x3<f32>, {
    (0, 0),
    (1, 0),
    (2, 0)
});
impl_scalar_matrix_mul_ops!(
    f64,   Matrix1x3<f64>,   Matrix1x3<f64>, {
    (0, 0),
    (1, 0),
    (2, 0)
});

impl_scalar_matrix_mul_ops!(
    u8,    Matrix1x4<u8>,    Matrix1x4<u8>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_scalar_matrix_mul_ops!(
    u16,   Matrix1x4<u16>,   Matrix1x4<u16>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_scalar_matrix_mul_ops!(
    u32,   Matrix1x4<u32>,   Matrix1x4<u32>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_scalar_matrix_mul_ops!(
    u64,   Matrix1x4<u64>,   Matrix1x4<u64>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_scalar_matrix_mul_ops!(
    u128,  Matrix1x4<u128>,  Matrix1x4<u128>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_scalar_matrix_mul_ops!(
    usize, Matrix1x4<usize>, Matrix1x4<usize>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_scalar_matrix_mul_ops!(
    i8,    Matrix1x4<i8>,    Matrix1x4<i8>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_scalar_matrix_mul_ops!(
    i16,   Matrix1x4<i16>,   Matrix1x4<i16>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_scalar_matrix_mul_ops!(
    i32,   Matrix1x4<i32>,   Matrix1x4<i32>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_scalar_matrix_mul_ops!(
    i64,   Matrix1x4<i64>,   Matrix1x4<i64>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_scalar_matrix_mul_ops!(
    i128,  Matrix1x4<i128>,  Matrix1x4<i128>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_scalar_matrix_mul_ops!(
    isize, Matrix1x4<isize>, Matrix1x4<isize>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_scalar_matrix_mul_ops!(
    f32,   Matrix1x4<f32>,   Matrix1x4<f32>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});
impl_scalar_matrix_mul_ops!(
    f64,   Matrix1x4<f64>,   Matrix1x4<f64>, {
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0)
});

impl_scalar_matrix_mul_ops!(
    u8,    Matrix2x3<u8>,    Matrix2x3<u8>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_scalar_matrix_mul_ops!(
    u16,   Matrix2x3<u16>,   Matrix2x3<u16>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_scalar_matrix_mul_ops!(
    u32,   Matrix2x3<u32>,   Matrix2x3<u32>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_scalar_matrix_mul_ops!(
    u64,   Matrix2x3<u64>,   Matrix2x3<u64>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_scalar_matrix_mul_ops!(
    u128,  Matrix2x3<u128>,  Matrix2x3<u128>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_scalar_matrix_mul_ops!(
    usize, Matrix2x3<usize>, Matrix2x3<usize>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_scalar_matrix_mul_ops!(
    i8,    Matrix2x3<i8>,    Matrix2x3<i8>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_scalar_matrix_mul_ops!(
    i16,   Matrix2x3<i16>,   Matrix2x3<i16>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_scalar_matrix_mul_ops!(
    i32,   Matrix2x3<i32>,   Matrix2x3<i32>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_scalar_matrix_mul_ops!(
    i64,   Matrix2x3<i64>,   Matrix2x3<i64>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_scalar_matrix_mul_ops!(
    i128,  Matrix2x3<i128>,  Matrix2x3<i128>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_scalar_matrix_mul_ops!(
    isize, Matrix2x3<isize>, Matrix2x3<isize>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_scalar_matrix_mul_ops!(
    f32,   Matrix2x3<f32>,   Matrix2x3<f32>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});
impl_scalar_matrix_mul_ops!(
    f64,   Matrix2x3<f64>,   Matrix2x3<f64>, {
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
});

impl_scalar_matrix_mul_ops!(
    u8,    Matrix2x4<u8>,    Matrix2x4<u8>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_scalar_matrix_mul_ops!(
    u16,   Matrix2x4<u16>,   Matrix2x4<u16>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_scalar_matrix_mul_ops!(
    u32,   Matrix2x4<u32>,   Matrix2x4<u32>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_scalar_matrix_mul_ops!(
    u64,   Matrix2x4<u64>,   Matrix2x4<u64>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1) 
});
impl_scalar_matrix_mul_ops!(
    u128,  Matrix2x4<u128>,  Matrix2x4<u128>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_scalar_matrix_mul_ops!(
    usize, Matrix2x4<usize>, Matrix2x4<usize>, { 
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_scalar_matrix_mul_ops!(
    i8,    Matrix2x4<i8>,    Matrix2x4<i8>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_scalar_matrix_mul_ops!(
    i16,   Matrix2x4<i16>,   Matrix2x4<i16>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_scalar_matrix_mul_ops!(
    i32,   Matrix2x4<i32>,   Matrix2x4<i32>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_scalar_matrix_mul_ops!(
    i64,   Matrix2x4<i64>,   Matrix2x4<i64>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_scalar_matrix_mul_ops!(
    i128,  Matrix2x4<i128>,  Matrix2x4<i128>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_scalar_matrix_mul_ops!(
    isize, Matrix2x4<isize>, Matrix2x4<isize>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_scalar_matrix_mul_ops!(
    f32,   Matrix2x4<f32>,   Matrix2x4<f32>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});
impl_scalar_matrix_mul_ops!(
    f64,   Matrix2x4<f64>,   Matrix2x4<f64>, {
    (0, 0), (0, 1), 
    (1, 0), (1, 1), 
    (2, 0), (2, 1), 
    (3, 0), (3, 1)
});

impl_scalar_matrix_mul_ops!(
    u8,    Matrix3x2<u8>,    Matrix3x2<u8>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2)
});
impl_scalar_matrix_mul_ops!(
    u16,   Matrix3x2<u16>,   Matrix3x2<u16>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2)
});
impl_scalar_matrix_mul_ops!(
    u32,   Matrix3x2<u32>,   Matrix3x2<u32>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2)
});
impl_scalar_matrix_mul_ops!(
    u64,   Matrix3x2<u64>,   Matrix3x2<u64>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2)
});
impl_scalar_matrix_mul_ops!(
    u128,  Matrix3x2<u128>,  Matrix3x2<u128>, {
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
});
impl_scalar_matrix_mul_ops!(
    usize, Matrix3x2<usize>, Matrix3x2<usize>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2)
});
impl_scalar_matrix_mul_ops!(
    i8,    Matrix3x2<i8>,    Matrix3x2<i8>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2) 
});
impl_scalar_matrix_mul_ops!(
    i16,   Matrix3x2<i16>,   Matrix3x2<i16>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2)
});
impl_scalar_matrix_mul_ops!(
    i32,   Matrix3x2<i32>,   Matrix3x2<i32>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2)
});
impl_scalar_matrix_mul_ops!(
    i64,   Matrix3x2<i64>,   Matrix3x2<i64>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2)
});
impl_scalar_matrix_mul_ops!(
    i128,  Matrix3x2<i128>,  Matrix3x2<i128>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2)
});
impl_scalar_matrix_mul_ops!(
    isize, Matrix3x2<isize>, Matrix3x2<isize>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2)
});
impl_scalar_matrix_mul_ops!(
    f32,   Matrix3x2<f32>,   Matrix3x2<f32>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2)
});
impl_scalar_matrix_mul_ops!(
    f64,   Matrix3x2<f64>,   Matrix3x2<f64>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2)
});

impl_scalar_matrix_mul_ops!(
    u8,    Matrix3x4<u8>,    Matrix3x4<u8>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_scalar_matrix_mul_ops!(
    u16,   Matrix3x4<u16>,   Matrix3x4<u16>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_scalar_matrix_mul_ops!(
    u32,   Matrix3x4<u32>,   Matrix3x4<u32>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_scalar_matrix_mul_ops!(
    u64,   Matrix3x4<u64>,   Matrix3x4<u64>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_scalar_matrix_mul_ops!(
    u128,  Matrix3x4<u128>,  Matrix3x4<u128>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_scalar_matrix_mul_ops!(
    usize, Matrix3x4<usize>, Matrix3x4<usize>, { 
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_scalar_matrix_mul_ops!(
    i8,    Matrix3x4<i8>,    Matrix3x4<i8>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_scalar_matrix_mul_ops!(
    i16,   Matrix3x4<i16>,   Matrix3x4<i16>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_scalar_matrix_mul_ops!(
    i32,   Matrix3x4<i32>,   Matrix3x4<i32>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_scalar_matrix_mul_ops!(
    i64,   Matrix3x4<i64>,   Matrix3x4<i64>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_scalar_matrix_mul_ops!(
    i128,  Matrix3x4<i128>,  Matrix3x4<i128>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_scalar_matrix_mul_ops!(
    isize, Matrix3x4<isize>, Matrix3x4<isize>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_scalar_matrix_mul_ops!(
    f32,   Matrix3x4<f32>,   Matrix3x4<f32>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});
impl_scalar_matrix_mul_ops!(
    f64,   Matrix3x4<f64>,   Matrix3x4<f64>, {
    (0, 0), (0, 1), (0, 2), 
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), 
    (3, 0), (3, 1), (3, 2)
});

impl_scalar_matrix_mul_ops!(
    u8,    Matrix4x2<u8>,    Matrix4x2<u8>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_scalar_matrix_mul_ops!(
    u16,   Matrix4x2<u16>,   Matrix4x2<u16>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_scalar_matrix_mul_ops!(
    u32,   Matrix4x2<u32>,   Matrix4x2<u32>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_scalar_matrix_mul_ops!(
    u64,   Matrix4x2<u64>,   Matrix4x2<u64>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_scalar_matrix_mul_ops!(
    u128,  Matrix4x2<u128>,  Matrix4x2<u128>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_scalar_matrix_mul_ops!(
    usize, Matrix4x2<usize>, Matrix4x2<usize>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_scalar_matrix_mul_ops!(
    i8,    Matrix4x2<i8>,    Matrix4x2<i8>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_scalar_matrix_mul_ops!(
    i16,   Matrix4x2<i16>,   Matrix4x2<i16>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_scalar_matrix_mul_ops!(
    i32,   Matrix4x2<i32>,   Matrix4x2<i32>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_scalar_matrix_mul_ops!(
    i64,   Matrix4x2<i64>,   Matrix4x2<i64>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_scalar_matrix_mul_ops!(
    i128,  Matrix4x2<i128>,  Matrix4x2<i128>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_scalar_matrix_mul_ops!(
    isize, Matrix4x2<isize>, Matrix4x2<isize>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_scalar_matrix_mul_ops!(
    f32,   Matrix4x2<f32>,   Matrix4x2<f32>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});
impl_scalar_matrix_mul_ops!(
    f64,   Matrix4x2<f64>,   Matrix4x2<f64>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3)
});

impl_scalar_matrix_mul_ops!(
    u8,    Matrix4x3<u8>,    Matrix4x3<u8>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_scalar_matrix_mul_ops!(
    u16,   Matrix4x3<u16>,   Matrix4x3<u16>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_scalar_matrix_mul_ops!(
    u32,   Matrix4x3<u32>,   Matrix4x3<u32>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_scalar_matrix_mul_ops!(
    u64,   Matrix4x3<u64>,   Matrix4x3<u64>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_scalar_matrix_mul_ops!(
    u128,  Matrix4x3<u128>,  Matrix4x3<u128>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_scalar_matrix_mul_ops!(
    usize, Matrix4x3<usize>, Matrix4x3<usize>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_scalar_matrix_mul_ops!(
    i8,    Matrix4x3<i8>,    Matrix4x3<i8>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_scalar_matrix_mul_ops!(
    i16,   Matrix4x3<i16>,   Matrix4x3<i16>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_scalar_matrix_mul_ops!(
    i32,   Matrix4x3<i32>,   Matrix4x3<i32>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_scalar_matrix_mul_ops!(
    i64,   Matrix4x3<i64>,   Matrix4x3<i64>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_scalar_matrix_mul_ops!(
    i128,  Matrix4x3<i128>,  Matrix4x3<i128>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_scalar_matrix_mul_ops!(
    isize, Matrix4x3<isize>, Matrix4x3<isize>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_scalar_matrix_mul_ops!(
    f32,   Matrix4x3<f32>,   Matrix4x3<f32>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});
impl_scalar_matrix_mul_ops!(
    f64,   Matrix4x3<f64>,   Matrix4x3<f64>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3)
});

