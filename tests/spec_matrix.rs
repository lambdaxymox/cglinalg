extern crate gdmath;
extern crate num_traits;
extern crate proptest;

use proptest::prelude::*;
use gdmath::{
    Matrix2,
    Matrix3,
    Matrix4,
    Matrix, 
    Scalar,
    ScalarFloat,
};

fn any_matrix2<S>() -> impl Strategy<Value = Matrix2<S>> where S: Scalar + Arbitrary {
    any::<(S, S, S, S)>().prop_map(
    |(c0r0, c0r1, c1r0, c1r1)| Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    )
}

fn any_matrix3<S>() -> impl Strategy<Value = Matrix3<S>> where S: Scalar + Arbitrary {
    any::<((S, S, S), (S, S, S), (S, S, S))>().prop_map(
        |((c0r0, c0r1, c0r2), (c1r0, c1r1, c1r2), (c2r0, c2r1, c2r2))| {
            Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
        }
    )
}

fn any_matrix4<S>() -> impl Strategy<Value = Matrix4<S>> where S: Scalar + Arbitrary {
    any::<((S, S, S, S), (S, S, S, S), (S, S, S, S), (S, S, S, S))>().prop_map(
        |((c0r0, c0r1, c0r2, c0r3), (c1r0, c1r1, c1r2, c1r3), (c2r0, c2r1, c2r2, c2r3), (c3r0, c3r1, c3r2, c3r3))| {
            Matrix4::new(c0r0,c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2,  c2r3, c3r0, c3r1, c3r2, c3r3)
        }
    )
}

