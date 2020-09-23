extern crate cglinalg;
extern crate num_traits;
extern crate proptest;


use proptest::prelude::*;
use cglinalg::{
    Vector1,
    Vector2,
    Vector3,
    Vector4, 
    Point3, 
    Scalar,
    ScalarFloat,
};
use proptest::strategy::{
    NewTree,
    Strategy,
    ValueTree,
};
use proptest::test_runner::{
    TestRunner,
};
use proptest::arbitrary::{
    Arbitrary,
};
use core::marker::{
    PhantomData 
};
use core::ops;


#[derive(Copy, Clone, Debug, PartialEq)]
struct VectorValue<VecType>(VecType);

impl<VecType> VectorValue<VecType> {
    fn into_inner(self) -> VecType {
        self.0
    }
}

#[derive(Copy, Clone, Debug)]
struct VectorStrategy<Strat, T> {
    strategy: Strat,
    _marker: PhantomData<T>,
}

impl<Strat, T> VectorStrategy<Strat, T> {
    fn new(strategy: Strat) -> VectorStrategy<Strat, T> {
        VectorStrategy {
            strategy: strategy,
            _marker: PhantomData,
        }
    }
}

#[derive(Debug)]
struct VectorValueTree<TreeT> {
    tree: TreeT,
    shrinker: usize,
    last_shrinker: Option<usize>,
}

macro_rules! vector_strategy_impl {
    ($VectorTreeN:ident, $VectorN:ident, $n:expr, { $($field:ident),* }) => {
        #[derive(Copy, Clone, Debug)]
        struct $VectorTreeN<S, STree> {
            $($field: STree,)*
            _marker: PhantomData<S>,
        }

        impl<S, STree> AsRef<[STree; $n]> for $VectorTreeN<S, STree> {
            fn as_ref(&self) -> &[STree; $n] {
                unsafe {
                    &*(self as *const $VectorTreeN<S, STree> as *const [STree; $n])
                }
            }
        }

        impl<S, STree> AsMut<[STree; $n]> for $VectorTreeN<S, STree> {
            fn as_mut(&mut self) -> &mut [STree; $n] {
                unsafe { 
                    &mut *(self as *mut $VectorTreeN<S, STree> as *mut [STree; $n])
                }
            }
        }

        impl<S, STree> ops::Index<usize> for $VectorTreeN<S, STree> {
            type Output = STree;

            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                let v: &[STree; $n] = self.as_ref();
                &v[index]
            }
        }

        impl<S, STree> ops::IndexMut<usize> for $VectorTreeN<S, STree> {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                let v: &mut [STree; $n] = self.as_mut();
                &mut v[index]
            }
        }

        impl<S, STree> ValueTree for VectorValueTree<$VectorTreeN<S, STree>> 
            where S: Scalar,
                  STree: ValueTree<Value = S>,
        {
            type Value = VectorValue<$VectorN<S>>;

            fn current(&self) -> Self::Value {
                VectorValue($VectorN::new(
                    $(self.tree.$field.current(),)* 
                ))
            }

            fn simplify(&mut self) -> bool {
                while self.shrinker < $n {
                    if self.tree[self.shrinker].simplify() {
                        self.last_shrinker = Some(self.shrinker);
                        return true;
                    } else {
                        self.shrinker += 1;
                    }
                }

                false
            }

            fn complicate(&mut self) -> bool {
                if let Some(shrinker) = self.last_shrinker {
                    self.shrinker = shrinker;
                    if self.tree[shrinker].complicate() {
                        true
                    } else {
                        self.last_shrinker = None;
                        false
                    }
                } else {
                    false
                }                                                                                                                                                   
            }
        }

        impl<Strat> Strategy for VectorStrategy<Strat, $VectorN<Strat::Value>> 
            where Strat: Strategy,
                  Strat::Value: Scalar,
        {
            type Value = VectorValue<$VectorN<Strat::Value>>;
            type Tree = VectorValueTree<$VectorTreeN<Strat::Value, Strat::Tree>>;

            fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
                Ok(VectorValueTree {
                    tree: $VectorTreeN {
                        $($field: self.strategy.new_tree(runner)?,)* 
                        _marker: PhantomData,
                    },
                    shrinker: 0,
                    last_shrinker: None,
                })
            }
        }

        impl<S> Arbitrary for VectorValue<$VectorN<S>> where S: Scalar + Arbitrary {
            type Parameters = S::Parameters;
            type Strategy = VectorStrategy<S::Strategy, $VectorN<S>>;
        
            fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
                let base = any_with::<S>(args);
                VectorStrategy::new(base)
            }
        }
    }
}

vector_strategy_impl!(VectorTree1, Vector1, 1, { tree_x });
vector_strategy_impl!(VectorTree2, Vector2, 2, { tree_x, tree_y });
vector_strategy_impl!(VectorTree3, Vector3, 3, { tree_x, tree_y, tree_z });
vector_strategy_impl!(VectorTree4, Vector4, 4, { tree_x, tree_y, tree_z, tree_w });


/// Generates the properties tests for perspective projection testing.
///
/// `$TestModuleName` is a name we give to the module we place the tests in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$ScalarType` denotes the underlying system of numbers.
/// `$Generator` is the name of a function or closure for generating examples.
/// `$UpperBound` denotes the upperbound on the range of acceptable indices.
macro_rules! perspective_projection_props {
    ($TestModuleName:ident, $ScalarType:ty) => {
        #[cfg(test)]
        mod $TestModuleName {
            use proptest::prelude::*;
            use super::VectorValue;
            use cglinalg::Vector3;

            proptest! {
                #[test]
                fn prop_perspective_projection_invertible(vv in any::<VectorValue<Vector3<$ScalarType>>>()) {
                    prop_assert!(vv == vv);
                }
            
                #[test]
                fn prop_perspective_projection_inverse_inverse_is_identity(vv in any::<VectorValue<Vector3<$ScalarType>>>()) {
                    prop_assert!(vv == vv);
                }
            }
        }
    }
}

perspective_projection_props!(perspective_f64_props, f64);

/*
extern crate cglinalg;

use cglinalg::{
    Scalar,
};
use core::marker::PhantomData;


trait Property {
    type Args;

    fn property(args: Self::Args) -> bool;
}

struct PropAdditionCommutative<T> {
    _marker: PhantomData<T>,
}

impl<S> Property for PropAdditionCommutative<S> where S: Scalar {
    type Args = (S, S);

    fn property(args: (S, S)) -> bool {
        args.0 + args.1 == args.1 + args.0
    }
}

struct PropAdditionAssociative<T> {
    _marker: PhantomData<T>,
}

impl<S> Property for PropAdditionAssociative<S> where S: Scalar {
    type Args = (S, S, S);

    fn property(args: (S, S, S)) -> bool {
        (args.0 + args.1) + args.2 == args.0 + (args.1 + args.2)
    }
}
use proptest::prelude::*;
macro_rules! props {
    ($PropT:ident, $PropFnName:ident, $ScalarT:ty) => {
    proptest! {
        #[test]
        fn $PropFnName(args in any::<<$PropT<$ScalarT> as Property>::Args>()) {
            prop_assert!($PropT::property(args));
        }
    }
    }
}

props!(PropAdditionCommutative, prop_addition_commutative, u32);
props!(PropAdditionAssociative, prop_addition_associative, u32);
*/