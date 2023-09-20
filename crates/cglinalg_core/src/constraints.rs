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

