use core::ops;

/*
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
*/
#[inline(always)]
pub fn dot_array_col<S, const M: usize, const N: usize>(arr: &[[S; M]; N], col: &[S; N], r: usize) -> S
where
    S: crate::Scalar + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    // PERFORMANCE: The Rust compiler should optmize out this loop.
    let mut acc = S::zero();
    for i in 0..N {
        acc += arr[i][r] * col[i];
    }

    acc
}

