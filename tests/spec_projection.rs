extern crate cglinalg;
extern crate num_traits;
extern crate proptest;


use proptest::prelude::*;
use cglinalg::{
    Vector3, 
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

#[derive(Copy, Clone, Debug)]
struct VectorTree3<S, STree> {
    tree_x: STree,
    tree_y: STree,
    tree_z: STree,
    _marker: PhantomData<S>,
}

impl<S, STree> VectorTree3<S, STree> 
    where
        S: Scalar,
        STree: ValueTree<Value = S>,
{
    fn new(tree_x: STree, tree_y: STree, tree_z: STree) -> VectorTree3<S, STree> {
        VectorTree3 {
            tree_x: tree_x,
            tree_y: tree_y,
            tree_z: tree_z,
            _marker: PhantomData,
        }
    }
}

impl<S, STree> AsRef<[STree; 3]> for VectorTree3<S, STree> {
    fn as_ref(&self) -> &[STree; 3] {
        unsafe {
            &*(self as *const VectorTree3<S, STree> as *const [STree; 3])
        }
    }
}

impl<S, STree> AsMut<[STree; 3]> for VectorTree3<S, STree> {
    fn as_mut(&mut self) -> &mut [STree; 3] {
        unsafe { 
            &mut *(self as *mut VectorTree3<S, STree> as *mut [STree; 3])
        }
    }
}

impl<S, STree> ops::Index<usize> for VectorTree3<S, STree> {
    type Output = STree;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[STree; 3] = self.as_ref();
        &v[index]
    }
}

impl<S, STree> ops::IndexMut<usize> for VectorTree3<S, STree> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let v: &mut [STree; 3] = self.as_mut();
        &mut v[index]
    }
}

#[derive(Debug)]
struct VectorValueTree<TreeT> {
    tree: TreeT,
    shrinker: usize,
    last_shrinker: Option<usize>,
}

impl<S, STree> ValueTree for VectorValueTree<VectorTree3<S, STree>> 
    where 
        S: Scalar,
        STree: ValueTree<Value = S>,
{
    type Value = VectorValue<Vector3<S>>;

    fn current(&self) -> Self::Value {
        VectorValue(Vector3::new(
            self.tree.tree_x.current(), 
            self.tree.tree_y.current(), 
            self.tree.tree_z.current()
        ))
    }

    fn simplify(&mut self) -> bool {
        while self.shrinker < 3 {
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

impl<Strat> Strategy for VectorStrategy<Strat, Vector3<Strat::Value>> 
    where
        Strat: Strategy,
        Strat::Value: Scalar,
{
    type Value = VectorValue<Vector3<Strat::Value>>;
    type Tree = VectorValueTree<VectorTree3<Strat::Value, Strat::Tree>>;

    fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
        Ok(VectorValueTree {
            tree: VectorTree3::new(
                self.strategy.new_tree(runner)?, 
                self.strategy.new_tree(runner)?, 
                self.strategy.new_tree(runner)?,
            ),
            shrinker: 0,
            last_shrinker: None,
        })
    }
}

impl<S> Arbitrary for VectorValue<Vector3<S>> where S: Scalar + Arbitrary {
    type Parameters = S::Parameters;
    type Strategy = VectorStrategy<S::Strategy, Vector3<S>>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        let base = any_with::<S>(args);
        VectorStrategy::new(base)
    }
}

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
                fn prop_perspective_projection_invertible(vv in any::<super::VectorValue<Vector3<$ScalarType>>>()) {
                    prop_assert!(vv == vv);
                }
            
                #[test]
                fn prop_perspective_projection_inverse_inverse_is_identity(vv in any::<super::VectorValue<Vector3<$ScalarType>>>()) {
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