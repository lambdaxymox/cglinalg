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
pub fn sub_array_array<S, const M: usize, const N: usize>(
    arr1: &[[S; M]; N], 
    arr2: &[[S; M]; N],
    c: usize,
    r: usize
) -> S
where
    S: Copy + ops::Sub<S, Output = S>
{
    arr1[c][r] - arr2[c][r]
}

#[inline(always)]
pub fn neg_array<S, const M: usize, const N: usize>(arr: &[[S; M]; N], c: usize, r: usize) -> S
where
    S: Copy + ops::Neg<Output = S>
{
    -arr[c][r]
}

#[inline(always)]
pub fn mul_array_scalar<S, const M: usize, const N: usize>(
    arr: &[[S; M]; N], 
    other: S, 
    c: usize, 
    r: usize
) -> S
where
    S: Copy + ops::Mul<S, Output = S>
{
    arr[c][r] * other
}

#[inline(always)]
pub fn div_array_scalar<S, const M: usize, const N: usize>(
    arr: &[[S; M]; N], 
    other: S, 
    c: usize, 
    r: usize
) -> S
where
    S: Copy + ops::Div<S, Output = S>
{
    arr[c][r] / other
}

#[inline(always)]
pub fn rem_array_scalar<S, const M: usize, const N: usize>(
    arr: &[[S; M]; N], 
    other: S, 
    c: usize, 
    r: usize
) -> S
where
    S: Copy + ops::Rem<S, Output = S>
{
    arr[c][r] % other
}

#[inline(always)]
pub fn dot_arrayMx1_col1<S, const M: usize>(arr: &[[S; M]; 1], col: &[S; 1], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0]
}

#[inline(always)]
pub fn dot_arrayMx2_col2<S, const M: usize>(arr: &[[S; M]; 2], col: &[S; 2], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1]
}

#[inline(always)]
pub fn dot_arrayMx3_col3<S, const M: usize>(arr: &[[S; M]; 3], col: &[S; 3], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2]
}

#[inline(always)]
pub fn dot_arrayMx4_col4<S, const M: usize>(arr: &[[S; M]; 4], col: &[S; 4], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2] + arr[3][r] * col[3]
}

/*
#[inline(always)]
pub fn dot_array1x1_col1<S>(arr: &[[S; 1]; 1], col: &[S; 1], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0]
}
*/
/*
#[inline(always)]
pub fn dot_array1x2_col2<S>(arr: &[[S; 1]; 2], col: &[S; 2], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1]
}
*/
/*
#[inline(always)]
pub fn dot_array1x3_col3<S>(arr: &[[S; 1]; 3], col: &[S; 3], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2]
}
*/
/*
#[inline(always)]
pub fn dot_array1x4_col4<S>(arr: &[[S; 1]; 4], col: &[S; 4], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2] + arr[3][r] * col[3]
}
*/
/*
#[inline(always)]
pub fn dot_array2x2_col2<S>(arr: &[[S; 2]; 2], col: &[S; 2], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1]
}
*/
/*
#[inline(always)]
pub fn dot_array2x3_col3<S>(arr: &[[S; 2]; 3], col: &[S; 3], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2]
}
*/
/*
#[inline(always)]
pub fn dot_array3x2_col2<S>(arr: &[[S; 3]; 2], col: &[S; 2], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1]
}
*/
/*
#[inline(always)]
pub fn dot_array3x3_col3<S>(arr: &[[S; 3]; 3], col: &[S; 3], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2]
}
*/
/*
#[inline(always)]
pub fn dot_array4x4_col4<S>(arr: &[[S; 4]; 4], col: &[S; 4], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2] + arr[3][r] * col[3]
}
*/
/*
#[inline(always)]
pub fn dot_array4x2_col2<S>(arr: &[[S; 4]; 2], col: &[S; 2], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1]
}
*/
/*
#[inline(always)]
pub fn dot_array2x4_col4<S>(arr: &[[S; 2]; 4], col: &[S; 4], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2] + arr[3][r] * col[3]
}
*/
/*
#[inline(always)]
pub fn dot_array3x4_col4<S>(arr: &[[S; 3]; 4], col: &[S; 4], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2] + arr[3][r] * col[3]
}
*/
/*
#[inline(always)]
pub fn dot_array4x3_col3<S>(arr: &[[S; 4]; 3], col: &[S; 3], r: usize) -> S
where
    S: Copy + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    arr[0][r] * col[0] + arr[1][r] * col[1] + arr[2][r] * col[2]
}
*/
