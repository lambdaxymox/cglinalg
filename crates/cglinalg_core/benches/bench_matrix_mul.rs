use cglinalg_core::{
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
    Vector2,
    Vector3,
    Vector4,
};
use core::ops::Mul;
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

fn gen_vector2<S>() -> Vector2<S>
where
    StandardUniform: Distribution<S>,
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);

    Vector2::new(rng.random(), rng.random())
}

fn gen_vector3<S>() -> Vector3<S>
where
    StandardUniform: Distribution<S>,
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);

    Vector3::new(rng.random(), rng.random(), rng.random())
}

fn gen_vector4<S>() -> Vector4<S>
where
    StandardUniform: Distribution<S>,
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);

    Vector4::new(rng.random(), rng.random(), rng.random(), rng.random())
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

bench_binop!(
    matrix2x2_mul_vector2_f32,
    f32,
    Matrix2x2<f32>,
    Vector2<f32>,
    gen_matrix2x2,
    gen_vector2,
    mul
);
bench_binop!(
    matrix3x3_mul_vector3_f32,
    f32,
    Matrix3x3<f32>,
    Vector3<f32>,
    gen_matrix3x3,
    gen_vector3,
    mul
);
bench_binop!(
    matrix4x4_mul_vector4_f32,
    f32,
    Matrix4x4<f32>,
    Vector4<f32>,
    gen_matrix4x4,
    gen_vector4,
    mul
);
bench_binop!(
    matrix2x2_mul_matrix2x2_f32,
    f32,
    Matrix2x2<f32>,
    Matrix2x2<f32>,
    gen_matrix2x2,
    gen_vector2,
    mul
);
bench_binop!(
    matrix3x3_mul_matrix3x3_f32,
    f32,
    Matrix3x3<f32>,
    Matrix3x3<f32>,
    gen_matrix3x3,
    gen_vector3,
    mul
);
bench_binop!(
    matrix4x4_mul_matrix4x4_f32,
    f32,
    Matrix4x4<f32>,
    Matrix4x4<f32>,
    gen_matrix4x4,
    gen_vector4,
    mul
);

criterion_group!(
    matrix_mul_benchmarks,
    matrix2x2_mul_vector2_f32,
    matrix3x3_mul_vector3_f32,
    matrix4x4_mul_vector4_f32,
    matrix2x2_mul_matrix2x2_f32,
    matrix3x3_mul_matrix3x3_f32,
    matrix4x4_mul_matrix4x4_f32,
);
criterion_main!(matrix_mul_benchmarks);
