use cglinalg_core::{
    Normed,
    Vector1,
    Vector2,
    Vector3,
    Vector4,
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

fn gen_vector1<S>() -> Vector1<S>
where
    StandardUniform: Distribution<S>,
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);

    Vector1::new(rng.random())
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

macro_rules! bench_binop_ref(
    ($name: ident, $scalar_type:ty, $type1:ty, $type2:ty, $generator_t1:ident, $generator_t2:ident, $binop:ident) => {
        fn $name(bh: &mut criterion::Criterion) {
            let a = $generator_t1::<$scalar_type>();
            let b = $generator_t2::<$scalar_type>();

            bh.bench_function(stringify!($name), move |bh| bh.iter(|| {
                a.$binop(&b)
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
    vector1_add_vector1_f32,
    f32,
    Vector1<f32>,
    Vector1<f32>,
    gen_vector1,
    gen_vector1,
    add
);
bench_binop!(
    vector2_add_vector2_f32,
    f32,
    Vector2<f32>,
    Vector2<f32>,
    gen_vector2,
    gen_vector2,
    add
);
bench_binop!(
    vector3_add_vector3_f32,
    f32,
    Vector3<f32>,
    Vector3<f32>,
    gen_vector3,
    gen_vector3,
    add
);
bench_binop!(
    vector4_add_vector4_f32,
    f32,
    Vector4<f32>,
    Vector4<f32>,
    gen_vector4,
    gen_vector4,
    add
);

bench_binop!(
    vector1_sub_vector1_f32,
    f32,
    Vector1<f32>,
    Vector1<f32>,
    gen_vector1,
    gen_vector1,
    sub
);
bench_binop!(
    vector2_sub_vector2_f32,
    f32,
    Vector2<f32>,
    Vector2<f32>,
    gen_vector2,
    gen_vector2,
    sub
);
bench_binop!(
    vector3_sub_vector3_f32,
    f32,
    Vector3<f32>,
    Vector3<f32>,
    gen_vector3,
    gen_vector3,
    sub
);
bench_binop!(
    vector4_sub_vector4_f32,
    f32,
    Vector4<f32>,
    Vector4<f32>,
    gen_vector4,
    gen_vector4,
    sub
);

bench_binop!(scalar_mul_vector1_f32, f32, f32, Vector1<f32>, gen_scalar, gen_vector1, mul);
bench_binop!(scalar_mul_vector2_f32, f32, f32, Vector2<f32>, gen_scalar, gen_vector2, mul);
bench_binop!(scalar_mul_vector3_f32, f32, f32, Vector3<f32>, gen_scalar, gen_vector3, mul);
bench_binop!(scalar_mul_vector4_f32, f32, f32, Vector4<f32>, gen_scalar, gen_vector4, mul);

bench_binop!(vector1_mul_scalar_f32, f32, Vector1<f32>, f32, gen_vector1, gen_scalar, mul);
bench_binop!(vector2_mul_scalar_f32, f32, Vector2<f32>, f32, gen_vector2, gen_scalar, mul);
bench_binop!(vector3_mul_scalar_f32, f32, Vector3<f32>, f32, gen_vector3, gen_scalar, mul);
bench_binop!(vector4_mul_scalar_f32, f32, Vector4<f32>, f32, gen_vector4, gen_scalar, mul);

bench_binop!(vector1_div_scalar_f32, f32, Vector1<f32>, f32, gen_vector1, gen_scalar, div);
bench_binop!(vector2_div_scalar_f32, f32, Vector2<f32>, f32, gen_vector2, gen_scalar, div);
bench_binop!(vector3_div_scalar_f32, f32, Vector3<f32>, f32, gen_vector3, gen_scalar, div);
bench_binop!(vector4_div_scalar_f32, f32, Vector4<f32>, f32, gen_vector4, gen_scalar, div);

bench_binop_ref!(
    vector1_dot_vector1_f32,
    f32,
    Vector1<f32>,
    Vector1<f32>,
    gen_vector1,
    gen_vector1,
    dot
);
bench_binop_ref!(
    vector2_dot_vector2_f32,
    f32,
    Vector2<f32>,
    Vector2<f32>,
    gen_vector2,
    gen_vector2,
    dot
);
bench_binop_ref!(
    vector3_dot_vector3_f32,
    f32,
    Vector3<f32>,
    Vector3<f32>,
    gen_vector3,
    gen_vector3,
    dot
);
bench_binop_ref!(
    vector4_dot_vector4_f32,
    f32,
    Vector4<f32>,
    Vector4<f32>,
    gen_vector4,
    gen_vector4,
    dot
);

bench_binop_ref!(
    vector3_cross_vector3_f32,
    f32,
    Vector3<f32>,
    Vector3<f32>,
    gen_vector3,
    gen_vector3,
    cross
);

bench_unop!(vector1_norm_f32, f32, Vector1<f32>, gen_vector1, norm);
bench_unop!(vector2_norm_f32, f32, Vector2<f32>, gen_vector2, norm);
bench_unop!(vector3_norm_f32, f32, Vector3<f32>, gen_vector3, norm);
bench_unop!(vector4_norm_f32, f32, Vector4<f32>, gen_vector4, norm);

bench_unop!(vector1_normalize_f32, f32, Vector1<f32>, gen_vector1, normalize);
bench_unop!(vector2_normalize_f32, f32, Vector2<f32>, gen_vector2, normalize);
bench_unop!(vector3_normalize_f32, f32, Vector3<f32>, gen_vector3, normalize);
bench_unop!(vector4_normalize_f32, f32, Vector4<f32>, gen_vector4, normalize);

criterion_group!(
    vector_benchmarks,
    vector1_add_vector1_f32,
    vector2_add_vector2_f32,
    vector3_add_vector3_f32,
    vector4_add_vector4_f32,
    vector1_sub_vector1_f32,
    vector2_sub_vector2_f32,
    vector3_sub_vector3_f32,
    vector4_sub_vector4_f32,
    scalar_mul_vector1_f32,
    scalar_mul_vector2_f32,
    scalar_mul_vector3_f32,
    scalar_mul_vector4_f32,
    vector1_mul_scalar_f32,
    vector2_mul_scalar_f32,
    vector3_mul_scalar_f32,
    vector4_mul_scalar_f32,
    vector1_div_scalar_f32,
    vector2_div_scalar_f32,
    vector3_div_scalar_f32,
    vector4_div_scalar_f32,
    vector1_dot_vector1_f32,
    vector2_dot_vector2_f32,
    vector3_dot_vector3_f32,
    vector4_dot_vector4_f32,
    vector3_cross_vector3_f32,
    vector1_norm_f32,
    vector2_norm_f32,
    vector3_norm_f32,
    vector4_norm_f32,
    vector1_normalize_f32,
    vector2_normalize_f32,
    vector3_normalize_f32,
    vector4_normalize_f32,
);
criterion_main!(vector_benchmarks);
