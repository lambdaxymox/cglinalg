extern crate cglinalg;
extern crate criterion;
extern crate rand;
extern crate rand_isaac;


use cglinalg::{
    Matrix,
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
};

use rand::{
    Rng, 
    prelude::Distribution,
    distributions::Standard,
};

use rand_isaac::{
    IsaacRng,
};

use criterion::{
    black_box,
    criterion_group,
    criterion_main,
};

fn gen_scalar<S>() -> S
where
    Standard: Distribution<S>
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);

    rng.gen()
}

fn gen_matrix2x2<S>() -> Matrix2x2<S> 
where 
    Standard: Distribution<S> 
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    
    Matrix2x2::new(
        rng.gen(), rng.gen(),
        rng.gen(), rng.gen()
    )
}

fn gen_matrix3x3<S>() -> Matrix3x3<S> 
where 
    Standard: Distribution<S> 
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    
    Matrix3x3::new(
        rng.gen(), rng.gen(), rng.gen(),
        rng.gen(), rng.gen(), rng.gen(),
        rng.gen(), rng.gen(), rng.gen()
    )
}

fn gen_matrix4x4<S>() -> Matrix4x4<S> 
where 
    Standard: Distribution<S> 
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    
    Matrix4x4::new(
        rng.gen(), rng.gen(), rng.gen(), rng.gen(),
        rng.gen(), rng.gen(), rng.gen(), rng.gen(),
        rng.gen(), rng.gen(), rng.gen(), rng.gen(),
        rng.gen(), rng.gen(), rng.gen(), rng.gen()
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
            let mut v = $generator::<$scalar_type>();

            bh.bench_function(stringify!($name), move |bh| bh.iter(|| {
                black_box(v.$unop())
            }));
        }
    }
);

bench_unop!(matrix2x2_is_diagonal, f32, Matrix2x2<f32>, gen_matrix2x2, is_diagonal);
bench_unop!(matrix3x3_is_diagonal, f32, Matrix3x3<f32>, gen_matrix3x3, is_diagonal);
bench_unop!(matrix4x4_is_diagonal, f32, Matrix4x4<f32>, gen_matrix4x4, is_diagonal);


criterion_group!(
    matrix_transpose_benchmarks,
    matrix2x2_is_diagonal,
    matrix3x3_is_diagonal,
    matrix4x4_is_diagonal
);
criterion_main!(matrix_transpose_benchmarks);

