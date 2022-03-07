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

fn generate1<S>() -> Vector1<S> where Standard: Distribution<S> {
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    
    Vector1::new(rng.gen::<S>())
}

fn generate2<S>() -> Vector2<S> where Standard: Distribution<S> {
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    
    Vector2::new(rng.gen(), rng.gen())
}

fn generate3<S>() -> Vector3<S> where Standard: Distribution<S> {
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    
    Vector3::new(rng.gen(), rng.gen(), rng.gen())
}

fn generate4<S>() -> Vector4<S> where Standard: Distribution<S> {
    use rand::SeedableRng;
    let mut rng = IsaacRng::seed_from_u64(0);
    
    Vector4::new(rng.gen(), rng.gen(), rng.gen(), rng.gen())
}


macro_rules! benchmark_binary_op(
    ($name: ident, $generator:ident, $scalar_type:ty, $type1:ty, $type2:ty, $binop:ident) => {
        fn $name(bh: &mut criterion::Criterion) {
            let a = $generator::<$scalar_type>();
            let b = $generator::<$scalar_type>();

            bh.bench_function(stringify!($name), move |bh| bh.iter(|| {
                a.$binop(b)
            }));
        }
    }
);

benchmark_binary_op!(vector1_add_vector1, generate1, f32, Vector1<f32>, Vector1<f32>, add);
benchmark_binary_op!(vector2_add_vector2, generate2, f32, Vector2<f32>, Vector2<f32>, add);
benchmark_binary_op!(vector3_add_vector3, generate3, f32, Vector3<f32>, Vector3<f32>, add);
benchmark_binary_op!(vector4_add_vector4, generate4, f32, Vector4<f32>, Vector4<f32>, add);

criterion_group!(
    vector_benches, 
    vector1_add_vector1,
    vector2_add_vector2,
    vector3_add_vector3,
    vector4_add_vector4,
);
criterion_main!(vector_benches);

