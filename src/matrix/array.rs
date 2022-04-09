// use core::ops;

/*
#[inline(always)]
pub fn dot_array_col<S, const M: usize, const N: usize>(arr: &[[S; M]; N], col: &[S; N], r: usize) -> S
where
    S: crate::Scalar + ops::Add<S, Output = S> + ops::Mul<S, Output = S>
{
    // PERFORMANCE: The Rust compiler should optmize out this loop.
    let mut result = S::zero();
    for i in 0..N {
        result += arr[i][r] * col[i];
    }

    result
}
*/
