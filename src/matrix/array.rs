use core::ops;


#[inline(always)]
pub fn add_array_array<S, const M: usize, const N: usize>(
    arr1: &[[S; M]; N], 
    arr2: &[[S; M]; N],
    c: usize,
    r: usize
) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}

#[inline(always)]
pub fn dot_array1x1_col1<S>(arr: &[[S; 1]; 1], col: &[S; 1], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0]
}

/*
#[inline(always)]
pub fn add_array1x1_array1x1<S>(arr1: &[[S; 1]; 1], arr2: &[[S; 1]; 1], c: usize, r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}
*/

#[inline(always)]
pub fn sub_array1x1_array1x1<S>(arr1: &[[S; 1]; 1], arr2: &[[S; 1]; 1], c: usize, r: usize) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array1x1<S>(arr: &[[S; 1]; 1], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array1x1_scalar<S>(arr: &[[S; 1]; 1], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array1x1_scalar<S>(arr: &[[S; 1]; 1], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array1x1_scalar<S>(arr: &[[S; 1]; 1], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}


#[inline(always)]
pub fn dot_array1x2_col2<S>(arr: &[[S; 1]; 2], col: &[S; 2], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1]
}

/*
#[inline(always)]
pub fn add_array1x2_array1x2<S>(arr1: &[[S; 1]; 2], arr2: &[[S; 1]; 2], c: usize, r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}
*/

#[inline(always)]
pub fn sub_array1x2_array1x2<S>(arr1: &[[S; 1]; 2], arr2: &[[S; 1]; 2], c: usize, r: usize) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array1x2<S>(arr: &[[S; 1]; 2], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array1x2_scalar<S>(arr: &[[S; 1]; 2], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array1x2_scalar<S>(arr: &[[S; 1]; 2], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array1x2_scalar<S>(arr: &[[S; 1]; 2], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}

#[inline(always)]
pub fn dot_array1x3_col3<S>(arr: &[[S; 1]; 3], col: &[S; 3], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2]
}
/*
#[inline(always)]
pub fn add_array1x3_array1x3<S>(arr1: &[[S; 1]; 3], arr2: &[[S; 1]; 3], c: usize, r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}
*/
#[inline(always)]
pub fn sub_array1x3_array1x3<S>(arr1: &[[S; 1]; 3], arr2: &[[S; 1]; 3], c: usize, r: usize) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array1x3<S>(arr: &[[S; 1]; 3], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array1x3_scalar<S>(arr: &[[S; 1]; 3], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array1x3_scalar<S>(arr: &[[S; 1]; 3], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array1x3_scalar<S>(arr: &[[S; 1]; 3], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}

#[inline(always)]
pub fn dot_array1x4_col4<S>(arr: &[[S; 1]; 4], col: &[S; 4], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2] + arr[3][r] * col[3]
}
/*
#[inline(always)]
pub fn add_array1x4_array1x4<S>(arr1: &[[S; 1]; 4], arr2: &[[S; 1]; 4], c: usize, r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}
*/
#[inline(always)]
pub fn sub_array1x4_array1x4<S>(arr1: &[[S; 1]; 4], arr2: &[[S; 1]; 4], c: usize, r: usize) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array1x4<S>(arr: &[[S; 1]; 4], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array1x4_scalar<S>(arr: &[[S; 1]; 4], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array1x4_scalar<S>(arr: &[[S; 1]; 4], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array1x4_scalar<S>(arr: &[[S; 1]; 4], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}

#[inline(always)]
pub fn dot_array2x2_col2<S>(arr: &[[S; 2]; 2], col: &[S; 2], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1]
}
/*
#[inline(always)]
pub fn add_array2x2_array2x2<S>(arr1: &[[S; 2]; 2], arr2: &[[S; 2]; 2], c: usize, r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}
*/

#[inline(always)]
pub fn sub_array2x2_array2x2<S>(arr1: &[[S; 2]; 2], arr2: &[[S; 2]; 2], c: usize, r: usize) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array2x2<S>(arr: &[[S; 2]; 2], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array2x2_scalar<S>(arr: &[[S; 2]; 2], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array2x2_scalar<S>(arr: &[[S; 2]; 2], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array2x2_scalar<S>(arr: &[[S; 2]; 2], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}

#[inline(always)]
pub fn dot_array2x3_col3<S>(arr: &[[S; 2]; 3], col: &[S; 3], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2]
}
/*
#[inline(always)]
pub fn add_array2x3_array2x3<S>(arr1: &[[S; 2]; 3], arr2: &[[S; 2]; 3], c: usize, r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}
*/

#[inline(always)]
pub fn sub_array2x3_array2x3<S>(arr1: &[[S; 2]; 3], arr2: &[[S; 2]; 3], c: usize, r: usize) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array2x3<S>(arr: &[[S; 2]; 3], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array2x3_scalar<S>(arr: &[[S; 2]; 3], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array2x3_scalar<S>(arr: &[[S; 2]; 3], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array2x3_scalar<S>(arr: &[[S; 2]; 3], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}

#[inline(always)]
pub fn dot_array3x2_col2<S>(arr: &[[S; 3]; 2], col: &[S; 2], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1]
}
/*
#[inline(always)]
pub fn add_array3x2_array3x2<S>(arr1: &[[S; 3]; 2], arr2: &[[S; 3]; 2], c: usize, r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}
*/

#[inline(always)]
pub fn sub_array3x2_array3x2<S>(arr1: &[[S; 3]; 2], arr2: &[[S; 3]; 2], c: usize, r: usize) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array3x2<S>(arr: &[[S; 3]; 2], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array3x2_scalar<S>(arr: &[[S; 3]; 2], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array3x2_scalar<S>(arr: &[[S; 3]; 2], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array3x2_scalar<S>(arr: &[[S; 3]; 2], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}


#[inline(always)]
pub fn dot_array3x3_col3<S>(arr: &[[S; 3]; 3], col: &[S; 3], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2]
}
/*
#[inline(always)]
pub fn add_array3x3_array3x3<S>(arr1: &[[S; 3]; 3], arr2: &[[S; 3]; 3], c: usize, r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}
*/

#[inline(always)]
pub fn sub_array3x3_array3x3<S>(arr1: &[[S; 3]; 3], arr2: &[[S; 3]; 3], c: usize, r: usize) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array3x3<S>(arr: &[[S; 3]; 3], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array3x3_scalar<S>(arr: &[[S; 3]; 3], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array3x3_scalar<S>(arr: &[[S; 3]; 3], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array3x3_scalar<S>(arr: &[[S; 3]; 3], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}

#[inline(always)]
pub fn dot_array4x4_col4<S>(arr: &[[S; 4]; 4], col: &[S; 4], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2] + arr[3][r] * col[3]
}
/*
#[inline(always)]
pub fn add_array4x4_array4x4<S>(arr1: &[[S; 4]; 4], arr2: &[[S; 4]; 4], c: usize, r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}
*/
#[inline(always)]
pub fn sub_array4x4_array4x4<S>(arr1: &[[S; 4]; 4], arr2: &[[S; 4]; 4], c: usize, r: usize) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array4x4<S>(arr: &[[S; 4]; 4], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array4x4_scalar<S>(arr: &[[S; 4]; 4], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array4x4_scalar<S>(arr: &[[S; 4]; 4], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array4x4_scalar<S>(arr: &[[S; 4]; 4], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}

#[inline(always)]
pub fn dot_array4x2_col2<S>(arr: &[[S; 4]; 2], col: &[S; 2], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1]
}
/*
#[inline(always)]
pub fn add_array4x2_array4x2<S>(arr1: &[[S; 4]; 2], arr2: &[[S; 4]; 2], c: usize, r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}
*/

#[inline(always)]
pub fn sub_array4x2_array4x2<S>(arr1: &[[S; 4]; 2], arr2: &[[S; 4]; 2], c: usize, r: usize) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array4x2<S>(arr: &[[S; 4]; 2], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array4x2_scalar<S>(arr: &[[S; 4]; 2], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array4x2_scalar<S>(arr: &[[S; 4]; 2], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array4x2_scalar<S>(arr: &[[S; 4]; 2], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}

#[inline(always)]
pub fn dot_array2x4_col4<S>(arr: &[[S; 2]; 4], col: &[S; 4], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2] + arr[3][r] * col[3]
}
/*
#[inline(always)]
pub fn add_array2x4_array2x4<S>(arr1: &[[S; 2]; 4], arr2: &[[S; 2]; 4], c: usize, r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}
*/
#[inline(always)]
pub fn sub_array2x4_array2x4<S>(arr1: &[[S; 2]; 4], arr2: &[[S; 2]; 4], c: usize, r: usize) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array2x4<S>(arr: &[[S; 2]; 4], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array2x4_scalar<S>(arr: &[[S; 2]; 4], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array2x4_scalar<S>(arr: &[[S; 2]; 4], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array2x4_scalar<S>(arr: &[[S; 2]; 4], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}

#[inline(always)]
pub fn dot_array3x4_col4<S>(arr: &[[S; 3]; 4], col: &[S; 4], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2] + arr[3][r] * col[3]
}
/*
#[inline(always)]
pub fn add_array3x4_array3x4<S>(arr1: &[[S; 3]; 4], arr2: &[[S; 3]; 4], c: usize, r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}
*/

#[inline(always)]
pub fn sub_array3x4_array3x4<S>(arr1: &[[S; 3]; 4], arr2: &[[S; 3]; 4], c: usize, r: usize) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array3x4<S>(arr: &[[S; 3]; 4], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array3x4_scalar<S>(arr: &[[S; 3]; 4], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array3x4_scalar<S>(arr: &[[S; 3]; 4], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array3x4_scalar<S>(arr: &[[S; 3]; 4], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}

#[inline(always)]
pub fn dot_array4x3_col3<S>(arr: &[[S; 4]; 3], col: &[S; 3], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2]
}
/*
#[inline(always)]
pub fn add_array4x3_array4x3<S>(arr1: &[[S; 4]; 3], arr2: &[[S; 4]; 3], c: usize, r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S>
{
    arr1[c][r] + arr2[c][r]
}
*/
#[inline(always)]
pub fn sub_array4x3_array4x3<S>(arr1: &[[S; 4]; 3], arr2: &[[S; 4]; 3], c: usize, r: usize) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array4x3<S>(arr: &[[S; 4]; 3], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array4x3_scalar<S>(arr: &[[S; 4]; 3], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array4x3_scalar<S>(arr: &[[S; 4]; 3], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array4x3_scalar<S>(arr: &[[S; 4]; 3], other: S, c: usize, r: usize) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}
