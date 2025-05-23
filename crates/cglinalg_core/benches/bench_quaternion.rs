use cglinalg_core::{
    Normed,
    Quaternion,
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

fn gen_quaternion<S>() -> Quaternion<S>
where
    StandardUniform: Distribution<S>,
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);

    Quaternion::new(rng.random(), rng.random(), rng.random(), rng.random())
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
    quaternion_add_quaternion_f32,
    f32,
    Quaternion<f32>,
    Quaternion<f32>,
    gen_quaternion,
    gen_quaternion,
    add
);
bench_binop!(
    quaternion_sub_quaternion_f32,
    f32,
    Quaternion<f32>,
    Quaternion<f32>,
    gen_quaternion,
    gen_quaternion,
    sub
);
bench_binop!(
    quaternion_mul_quaternion_f32,
    f32,
    Quaternion<f32>,
    Quaternion<f32>,
    gen_quaternion,
    gen_quaternion,
    mul
);
bench_binop!(
    scalar_mul_quaternion_f32,
    f32,
    f32,
    Quaternion<f32>,
    gen_scalar,
    gen_quaternion,
    mul
);
bench_binop!(
    quaternion_mul_scalar_f32,
    f32,
    Quaternion<f32>,
    f32,
    gen_quaternion,
    gen_scalar,
    mul
);
bench_binop!(
    quaternion_div_scalar_f32,
    f32,
    Quaternion<f32>,
    f32,
    gen_quaternion,
    gen_scalar,
    div
);

bench_binop_ref!(
    quaternion_dot_quaternion_f32,
    f32,
    Quaternion<f32>,
    Quaternion<f32>,
    gen_quaternion,
    gen_quaternion,
    dot
);

bench_unop!(quaternion_conjugate_f32, f32, Quaternion<f32>, gen_quaternion, conjugate);
bench_unop!(quaternion_norm_f32, f32, Quaternion<f32>, gen_quaternion, norm);
bench_unop!(quaternion_normalize_f32, f32, Quaternion<f32>, gen_quaternion, normalize);
bench_unop!(quaternion_inverse_f32, f32, Quaternion<f32>, gen_quaternion, try_inverse);
bench_unop!(quaternion_sqrt_f32, f32, Quaternion<f32>, gen_quaternion, sqrt);

criterion_group!(
    quaternion_benches,
    quaternion_add_quaternion_f32,
    quaternion_sub_quaternion_f32,
    quaternion_mul_quaternion_f32,
    scalar_mul_quaternion_f32,
    quaternion_mul_scalar_f32,
    quaternion_div_scalar_f32,
    quaternion_dot_quaternion_f32,
    quaternion_conjugate_f32,
    quaternion_norm_f32,
    quaternion_normalize_f32,
    quaternion_inverse_f32,
    quaternion_sqrt_f32,
);
criterion_main!(quaternion_benches);
