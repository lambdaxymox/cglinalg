pub trait Dim {}

pub trait DimAdd<D1: Dim, D2: Dim>: Dim {
    type Output: Dim;
}

pub trait DimSub<D1: Dim, D2: Dim>: Dim {
    type Output: Dim;
}


pub trait DimMul<D1: Dim, D2: Dim>: Dim {
    type Output: Dim;
}


#[derive(Clone, Debug)]
pub enum Const<const D: usize> {}

impl<const D: usize> Dim for Const<D> {}


pub trait DimEq<D1: Dim, D2: Dim> {
    type Representative: Dim;
}


pub enum ShapeConstraint {}

impl<D: Dim> DimEq<D, D> for ShapeConstraint {
    type Representative = D;
}

impl Dim for ShapeConstraint {}


pub trait CanMultiply<R1: Dim, C1: Dim, R2: Dim, C2: Dim>: DimEq<C1, R2> + DimEq<R2, C1> {}

impl<R1: Dim, C1: Dim, R2: Dim, C2: Dim> CanMultiply<R1, C1, R2, C2> for ShapeConstraint 
where
    ShapeConstraint: DimEq<C1, R2> + DimEq<R2, C1>
{

}

pub trait CanTransposeMultiply<R1: Dim, C1: Dim, R2: Dim, C2: Dim>: DimEq<R1, R2> + DimEq<R2, R1> {}

impl<R1: Dim, C1: Dim, R2: Dim, C2: Dim> CanTransposeMultiply<R1, C1, R2, C2> for ShapeConstraint 
where
    ShapeConstraint: DimEq<R1, R2> + DimEq<R2, R1>
{

}

pub trait CanExtend<N1: Dim, N2: Dim>: DimAdd<N1, Const<1>, Output = N2> {}

impl<N1: Dim, N2: Dim> CanExtend<N1, N2> for ShapeConstraint
where
    ShapeConstraint: DimAdd<N1, Const<1>, Output = N2>
{

}

pub trait CanContract<N1: Dim, N2: Dim>: DimSub<N1, Const<1>, Output = N2> {}

impl<N1: Dim, N2: Dim> CanContract<N1, N2> for ShapeConstraint
where
    ShapeConstraint: DimSub<N1, Const<1>, Output = N2>
{

}


macro_rules! impl_dim_add {
    ($D1:expr, $D2:expr) => {
        impl DimAdd<Const<$D1>, Const<$D2>> for ShapeConstraint {
            type Output = Const<{ $D1 + $D2 }>;
        }
    };
}

impl_dim_add!(1, 1);
impl_dim_add!(1, 2);
impl_dim_add!(1, 3);
impl_dim_add!(1, 4);
impl_dim_add!(2, 1);
impl_dim_add!(2, 2);
impl_dim_add!(2, 3);
impl_dim_add!(2, 4);
impl_dim_add!(3, 1);
impl_dim_add!(3, 2);
impl_dim_add!(3, 3);
impl_dim_add!(3, 4);
impl_dim_add!(4, 1);
impl_dim_add!(4, 2);
impl_dim_add!(4, 3);
impl_dim_add!(4, 4);


macro_rules! impl_dim_sub {
    ($D1:expr, $D2:expr) => {
        impl DimSub<Const<$D1>, Const<$D2>> for ShapeConstraint {
            type Output = Const<{ $D1 - $D2 }>;
        }
    };
}

impl_dim_sub!(1, 1);
// impl_dim_sub!(1, 2);
// impl_dim_sub!(1, 3);
// impl_dim_sub!(1, 4);
impl_dim_sub!(2, 1);
impl_dim_sub!(2, 2);
// impl_dim_sub!(2, 3);
// impl_dim_sub!(2, 4);
impl_dim_sub!(3, 1);
impl_dim_sub!(3, 2);
impl_dim_sub!(3, 3);
// impl_dim_sub!(3, 4);
impl_dim_sub!(4, 1);
impl_dim_sub!(4, 2);
impl_dim_sub!(4, 3);
impl_dim_sub!(4, 4);


macro_rules! impl_dim_mul {
    ($D1:expr, $D2:expr) => {
        impl DimMul<Const<$D1>, Const<$D2>> for ShapeConstraint {
            type Output = Const<{ $D1 * $D2 }>;
        }
    };
}

impl_dim_mul!(1, 1);
impl_dim_mul!(1, 2);
impl_dim_mul!(1, 3);
impl_dim_mul!(1, 4);
impl_dim_mul!(2, 1);
impl_dim_mul!(2, 2);
impl_dim_mul!(2, 3);
impl_dim_mul!(2, 4);
impl_dim_mul!(3, 1);
impl_dim_mul!(3, 2);
impl_dim_mul!(3, 3);
impl_dim_mul!(3, 4);
impl_dim_mul!(4, 1);
impl_dim_mul!(4, 2);
impl_dim_mul!(4, 3);
impl_dim_mul!(4, 4);


pub trait DimLt<D1: Dim, D2: Dim>: Dim {}

macro_rules! impl_dim_lt {
    ($D1:expr, $D2:expr) => {
        impl DimLt<Const<$D1>, Const<$D2>> for ShapeConstraint {
            
        }
    }
}

impl_dim_lt!(0, 1);
impl_dim_lt!(0, 2);
impl_dim_lt!(0, 3);
impl_dim_lt!(0, 4);
// impl_dim_lt!(1, 1);
impl_dim_lt!(1, 2);
impl_dim_lt!(1, 3);
impl_dim_lt!(1, 4);
// impl_dim_lt!(2, 1);
// impl_dim_lt!(2, 2);
impl_dim_lt!(2, 3);
impl_dim_lt!(2, 4);
// impl_dim_lt!(3, 1);
// impl_dim_lt!(3, 2);
// impl_dim_lt!(3, 3);
impl_dim_lt!(3, 4);
// impl_dim_lt!(4, 1);
// impl_dim_lt!(4, 2);
// impl_dim_lt!(4, 3);
// impl_dim_lt!(4, 4);


#[cfg(test)]
mod constraint_tests {
    use super::*;

    
    fn dim_lt<const M: usize, const N: usize>() -> bool
    where
        ShapeConstraint: DimLt<Const<M>, Const<N>>
    {
        M < N
    }

    fn dim_eq<const M: usize, const N: usize>() -> bool
    where
        ShapeConstraint: DimEq<Const<M>, Const<N>>
    {
        M == N
    }

    fn dim_add<const M: usize, const N: usize, const MPLUSN: usize>() -> bool
    where
        ShapeConstraint: DimAdd<Const<M>, Const<N>, Output = Const<MPLUSN>>
    {
        M + N == MPLUSN
    }

    fn dim_sub<const M: usize, const N: usize, const MMINUSN: usize>() -> bool
    where
        ShapeConstraint: DimSub<Const<M>, Const<N>, Output = Const<MMINUSN>>
    {
        M - N == MMINUSN
    }

    fn dim_mul<const M: usize, const N: usize, const MN: usize>() -> bool
    where
        ShapeConstraint: DimMul<Const<M>, Const<N>, Output = Const<MN>>
    {
        M * N == MN
    }


    #[test]
    fn test_dim_lt() {
        assert!(dim_lt::<0, 1>());
        assert!(dim_lt::<0, 2>());
        assert!(dim_lt::<0, 3>());
        assert!(dim_lt::<0, 4>());
        assert!(dim_lt::<1, 2>());
        assert!(dim_lt::<1, 3>());
        assert!(dim_lt::<1, 4>());
        assert!(dim_lt::<2, 3>());
        assert!(dim_lt::<2, 4>());
        assert!(dim_lt::<3, 4>());
    }

    #[test]
    fn test_dim_eq() {
        assert!(dim_eq::<0, 0>());
        assert!(dim_eq::<1, 1>());
        assert!(dim_eq::<2, 2>());
        assert!(dim_eq::<3, 3>());
        assert!(dim_eq::<4, 4>());
    }

    #[test]
    fn test_dim_add() {
        assert!(dim_add::<1, 1, 2>());
        assert!(dim_add::<1, 2, 3>());
        assert!(dim_add::<1, 3, 4>());
        assert!(dim_add::<1, 4, 5>());
        assert!(dim_add::<2, 1, 3>());
        assert!(dim_add::<2, 2, 4>());
        assert!(dim_add::<2, 3, 5>());
        assert!(dim_add::<2, 4, 6>());
        assert!(dim_add::<3, 1, 4>());
        assert!(dim_add::<3, 2, 5>());
        assert!(dim_add::<3, 3, 6>());
        assert!(dim_add::<3, 4, 7>());
        assert!(dim_add::<4, 1, 5>());
        assert!(dim_add::<4, 2, 6>());
        assert!(dim_add::<4, 3, 7>());
        assert!(dim_add::<4, 4, 8>());
    }

    #[test]
    fn test_dim_sub() {
        assert!(dim_sub::<1, 1, 0>());
        assert!(dim_sub::<2, 1, 1>());
        assert!(dim_sub::<2, 2, 0>());
        assert!(dim_sub::<3, 1, 2>());
        assert!(dim_sub::<3, 2, 1>());
        assert!(dim_sub::<3, 3, 0>());
        assert!(dim_sub::<4, 1, 3>());
        assert!(dim_sub::<4, 2, 2>());
        assert!(dim_sub::<4, 3, 1>());
        assert!(dim_sub::<4, 4, 0>());
    }

    #[test]
    fn test_dim_mul() {
        assert!(dim_mul::<1, 1, 1>());
        assert!(dim_mul::<1, 2, 2>());
        assert!(dim_mul::<1, 3, 3>());
        assert!(dim_mul::<1, 4, 4>());
        assert!(dim_mul::<2, 1, 2>());
        assert!(dim_mul::<2, 2, 4>());
        assert!(dim_mul::<2, 3, 6>());
        assert!(dim_mul::<2, 4, 8>());
        assert!(dim_mul::<3, 1, 3>());
        assert!(dim_mul::<3, 2, 6>());
        assert!(dim_mul::<3, 3, 9>());
        assert!(dim_mul::<3, 4, 12>());
        assert!(dim_mul::<4, 1, 4>());
        assert!(dim_mul::<4, 2, 8>());
        assert!(dim_mul::<4, 3, 12>());
        assert!(dim_mul::<4, 4, 16>());
    }
}

