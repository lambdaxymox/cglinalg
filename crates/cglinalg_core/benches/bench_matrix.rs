use cglinalg_core::{
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
};
use core::ops::{
    Add,
    Div,
    Mul,
    Sub,
};
use criterion::{
    criterion_group,
    criterion_main,
};
use rand::{
    Rng,
    distr::StandardUniform,
    prelude::Distribution,
};
use rand_isaac::IsaacRng;

fn gen_scalar<S>() -> S
where
    StandardUniform: Distribution<S>,
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);

    rng.random()
}

fn gen_matrix2x2<S>() -> Matrix2x2<S>
where
    StandardUniform: Distribution<S>,
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);

    Matrix2x2::new(rng.random(), rng.random(), rng.random(), rng.random())
}

fn gen_matrix3x3<S>() -> Matrix3x3<S>
where
    StandardUniform: Distribution<S>,
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);

    Matrix3x3::new(
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
    )
}

fn gen_matrix4x4<S>() -> Matrix4x4<S>
where
    StandardUniform: Distribution<S>,
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);

    Matrix4x4::new(
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
        rng.random(),
    )
}

macro_rules! bench_binop(
    ($name: ident, $scalar_type:ty, $type1:ty, $type2:ty, $generator_t1:ident, $generator_t2:ident, $binop:ident) => {
        fn $name(bh: &mut criterion::Criterion) {
            let a = $generator_t1::<$scalar_type>();
            let b = $generator_t2::<$scalar_type>();

            bh.bench_function(stringify!($name), move |bh| bh.iter(|| {
                a.$binop(b)
            }));
        }
    }
);

macro_rules! bench_unop(
    ($name:ident, $scalar_type:ty, $ty:ty, $generator:ident, $unop:ident) => {
        fn $name(bh: &mut criterion::Criterion) {
            let v = $generator::<$scalar_type>();

            bh.bench_function(stringify!($name), move |bh| bh.iter(|| {
                v.$unop()
            }));
        }
    }
);

bench_binop!(
    matrix2x2_add_matrix2x2_f32,
    f32,
    Matrix2x2<f32>,
    Matrix2x2<f32>,
    gen_matrix2x2,
    gen_matrix2x2,
    add
);
bench_binop!(
    matrix3x3_add_matrix3x3_f32,
    f32,
    Matrix3x3<f32>,
    Matrix3x3<f32>,
    gen_matrix3x3,
    gen_matrix3x3,
    add
);
bench_binop!(
    matrix4x4_add_matrix4x4_f32,
    f32,
    Matrix4x4<f32>,
    Matrix4x4<f32>,
    gen_matrix4x4,
    gen_matrix4x4,
    add
);

bench_binop!(
    matrix2x2_sub_matrix2x2_f32,
    f32,
    Matrix2x2<f32>,
    Matrix2x2<f32>,
    gen_matrix2x2,
    gen_matrix2x2,
    sub
);
bench_binop!(
    matrix3x3_sub_matrix3x3_f32,
    f32,
    Matrix3x3<f32>,
    Matrix3x3<f32>,
    gen_matrix3x3,
    gen_matrix3x3,
    sub
);
bench_binop!(
    matrix4x4_sub_matrix4x4_f32,
    f32,
    Matrix4x4<f32>,
    Matrix4x4<f32>,
    gen_matrix4x4,
    gen_matrix4x4,
    sub
);

bench_binop!(scalar_mul_matrix2x2_f32, f32, f32, Matrix2x2<f32>, gen_scalar, gen_matrix2x2, mul);
bench_binop!(scalar_mul_matrix3x3_f32, f32, f32, Matrix3x3<f32>, gen_scalar, gen_matrix3x3, mul);
bench_binop!(scalar_mul_matrix4x4_f32, f32, f32, Matrix4x4<f32>, gen_scalar, gen_matrix4x4, mul);

bench_binop!(matrix2x2_mul_scalar_f32, f32, Matrix2x2<f32>, f32, gen_matrix2x2, gen_scalar, mul);
bench_binop!(matrix3x3_mul_scalar_f32, f32, Matrix3x3<f32>, f32, gen_matrix3x3, gen_scalar, mul);
bench_binop!(matrix4x4_mul_scalar_f32, f32, Matrix4x4<f32>, f32, gen_matrix4x4, gen_scalar, mul);

bench_binop!(matrix2x2_div_scalar_f32, f32, Matrix2x2<f32>, f32, gen_matrix2x2, gen_scalar, div);
bench_binop!(matrix3x3_div_scalar_f32, f32, Matrix3x3<f32>, f32, gen_matrix3x3, gen_scalar, div);
bench_binop!(matrix4x4_div_scalar_f32, f32, Matrix4x4<f32>, f32, gen_matrix4x4, gen_scalar, div);

bench_unop!(matrix2x2_transpose_f32, f32, Matrix2x2<f32>, gen_matrix2x2, transpose);
bench_unop!(matrix3x3_transpose_f32, f32, Matrix3x3<f32>, gen_matrix3x3, transpose);
bench_unop!(matrix4x4_transpose_f32, f32, Matrix4x4<f32>, gen_matrix4x4, transpose);

bench_unop!(matrix2x2_inverse_f32, f32, Matrix2x2<f32>, gen_matrix2x2, try_inverse);
bench_unop!(matrix3x3_inverse_f32, f32, Matrix3x3<f32>, gen_matrix3x3, try_inverse);
bench_unop!(matrix4x4_inverse_f32, f32, Matrix4x4<f32>, gen_matrix4x4, try_inverse);

criterion_group!(
    matrix_benchmarks,
    matrix2x2_add_matrix2x2_f32,
    matrix3x3_add_matrix3x3_f32,
    matrix4x4_add_matrix4x4_f32,
    matrix2x2_sub_matrix2x2_f32,
    matrix3x3_sub_matrix3x3_f32,
    matrix4x4_sub_matrix4x4_f32,
    scalar_mul_matrix2x2_f32,
    scalar_mul_matrix3x3_f32,
    scalar_mul_matrix4x4_f32,
    matrix2x2_mul_scalar_f32,
    matrix3x3_mul_scalar_f32,
    matrix4x4_mul_scalar_f32,
    matrix2x2_div_scalar_f32,
    matrix3x3_div_scalar_f32,
    matrix4x4_div_scalar_f32,
    matrix2x2_transpose_f32,
    matrix3x3_transpose_f32,
    matrix4x4_transpose_f32,
    matrix2x2_inverse_f32,
    matrix3x3_inverse_f32,
    matrix4x4_inverse_f32,
);
criterion_main!(matrix_benchmarks);
