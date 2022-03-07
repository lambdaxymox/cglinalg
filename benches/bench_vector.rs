extern crate cglinalg;
extern crate criterion;
extern crate rand;
extern crate rand_isaac;


use cglinalg::{
    Vector1,
    Vector2,
    Vector3,
    Vector4,
};

use core::ops::{
    Add,
    Sub,
    Mul,
    Div,
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

fn gen_vector1<S>() -> Vector1<S> 
where 
    Standard: Distribution<S> 
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    
    Vector1::new(rng.gen::<S>())
}

fn gen_vector2<S>() -> Vector2<S> 
where 
    Standard: Distribution<S> 
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    
    Vector2::new(rng.gen(), rng.gen())
}

fn gen_vector3<S>() -> Vector3<S> 
where 
    Standard: Distribution<S> 
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    
    Vector3::new(rng.gen(), rng.gen(), rng.gen())
}

fn gen_vector4<S>() -> Vector4<S> 
where 
    Standard: Distribution<S> 
{
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    
    Vector4::new(rng.gen(), rng.gen(), rng.gen(), rng.gen())
}


macro_rules! benchmark_binary_op(
    ($name: ident, $generator_t1:ident, $generator_t2:ident, $scalar_type:ty, $type1:ty, $type2:ty, $binop:ident) => {
        fn $name(bh: &mut criterion::Criterion) {
            let a = $generator_t1::<$scalar_type>();
            let b = $generator_t2::<$scalar_type>();

            bh.bench_function(stringify!($name), move |bh| bh.iter(|| {
                a.$binop(b)
            }));
        }
    }
);

benchmark_binary_op!(vector1_add_vector1, gen_vector1, gen_vector1, f32, Vector1<f32>, Vector1<f32>, add);
benchmark_binary_op!(vector2_add_vector2, gen_vector2, gen_vector2, f32, Vector2<f32>, Vector2<f32>, add);
benchmark_binary_op!(vector3_add_vector3, gen_vector3, gen_vector3, f32, Vector3<f32>, Vector3<f32>, add);
benchmark_binary_op!(vector4_add_vector4, gen_vector4, gen_vector4, f32, Vector4<f32>, Vector4<f32>, add);

benchmark_binary_op!(vector1_sub_vector1, gen_vector1, gen_vector1, f32, Vector1<f32>, Vector1<f32>, sub);
benchmark_binary_op!(vector2_sub_vector2, gen_vector2, gen_vector2, f32, Vector2<f32>, Vector2<f32>, sub);
benchmark_binary_op!(vector3_sub_vector3, gen_vector3, gen_vector3, f32, Vector3<f32>, Vector3<f32>, sub);
benchmark_binary_op!(vector4_sub_vector4, gen_vector4, gen_vector4, f32, Vector4<f32>, Vector4<f32>, sub);

benchmark_binary_op!(scalar_mul_vector1,  gen_scalar,  gen_vector1, f32, f32,          Vector1<f32>, mul);
benchmark_binary_op!(scalar_mul_vector2,  gen_scalar,  gen_vector2, f32, f32,          Vector2<f32>, mul);
benchmark_binary_op!(scalar_mul_vector3,  gen_scalar,  gen_vector3, f32, f32,          Vector3<f32>, mul);
benchmark_binary_op!(scalar_mul_vector4,  gen_scalar,  gen_vector4, f32, f32,          Vector4<f32>, mul);

benchmark_binary_op!(vector1_mul_scalar,  gen_vector1, gen_scalar,  f32, Vector1<f32>, f32,          mul);
benchmark_binary_op!(vector2_mul_scalar,  gen_vector2, gen_scalar,  f32, Vector2<f32>, f32,          mul);
benchmark_binary_op!(vector3_mul_scalar,  gen_vector3, gen_scalar,  f32, Vector3<f32>, f32,          mul);
benchmark_binary_op!(vector4_mul_scalar,  gen_vector4, gen_scalar,  f32, Vector4<f32>, f32,          mul);

benchmark_binary_op!(vector1_div_scalar,  gen_vector1, gen_scalar,  f32, Vector1<f32>, f32,          div);
benchmark_binary_op!(vector2_div_scalar,  gen_vector2, gen_scalar,  f32, Vector2<f32>, f32,          div);
benchmark_binary_op!(vector3_div_scalar,  gen_vector3, gen_scalar,  f32, Vector3<f32>, f32,          div);
benchmark_binary_op!(vector4_div_scalar,  gen_vector4, gen_scalar,  f32, Vector4<f32>, f32,          div);

criterion_group!(
    vector_benches, 
    vector1_add_vector1,
    vector2_add_vector2,
    vector3_add_vector3,
    vector4_add_vector4,
    vector1_sub_vector1,
    vector2_sub_vector2,
    vector3_sub_vector3,
    vector4_sub_vector4,
    scalar_mul_vector1,
    scalar_mul_vector2,
    scalar_mul_vector3,
    scalar_mul_vector4,
    vector1_mul_scalar,
    vector2_mul_scalar,
    vector3_mul_scalar,
    vector4_mul_scalar,
    vector1_div_scalar,
    vector2_div_scalar,
    vector3_div_scalar,
    vector4_div_scalar,
);
criterion_main!(vector_benches);

