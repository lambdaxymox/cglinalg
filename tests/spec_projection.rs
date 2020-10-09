extern crate cglinalg;
extern crate num_traits;
extern crate proptest;


use cglinalg::{
    Degrees,
    Vector3,
    Point3,
    PerspectiveSpec,
    PerspectiveFovSpec,
    Perspective3,
    PerspectiveFov3,
    OrthographicSpec,
    Orthographic3,
    ScalarFloat,
};

use proptest::prelude::*;


fn any_vector3<S>() -> impl Strategy<Value = Vector3<S>> 
    where S: ScalarFloat + Arbitrary
{
    any::<(S, S, S)>()
        .prop_map(|(x, y, z)| {
            let modulus: S = num_traits::cast(1_000_000).unwrap();
            let vector = Vector3::new(x, y, z);

            vector % modulus
        })
        .no_shrink()
}

fn any_point3<S>() -> impl Strategy<Value = Point3<S>> 
    where S: ScalarFloat + Arbitrary 
{
    any::<(S, S, S)>()
        .prop_map(|(x, y, z)| {
            let modulus: S = num_traits::cast(1_000_000).unwrap();
            let point = Point3::new(x, y, z);

            point % modulus
        })
        .no_shrink()
}

fn any_perspective_fov_projection<S>() -> impl Strategy<Value = PerspectiveFov3<S>> 
    where S: ScalarFloat + Arbitrary
{
    any::<(S, S, S, S)>()
        .prop_map(|(_fovy, _aspect, _near, _far)| {
            let modulus: S = num_traits::cast(1_000_000).unwrap();
            let fovy = S::abs(_fovy % modulus);
            let aspect = S::abs(_aspect % modulus); 
            let near = S::abs(_near % modulus);
            let far = S::abs(_far % modulus);

            (fovy, aspect, near, far)
        })    
        .prop_map(|(fovy, aspect, near, far)| {
            let (spec_near, spec_far) = if near > far {
                (far, near)
            } else {
                (near, far)
            };
            let spec = PerspectiveFovSpec::new(
                Degrees(fovy), aspect, spec_near, spec_far
            );

            PerspectiveFov3::new(spec)
        })
        .no_shrink()
}

fn any_perspective_projection<S>() -> impl Strategy<Value = Perspective3<S>> 
    where S: ScalarFloat + Arbitrary
{
    any::<(S, S, S, S, S, S)>()
        .prop_map(|(_left, _right, _bottom, _top, _near, _far)| {
            let modulus: S = num_traits::cast(1_000_000).unwrap();
            let left = -S::abs(_left % modulus) - S::one();
            let right = S::abs(_right % modulus) + S::one();
            let bottom = -S::abs(_bottom % modulus) - S::one();
            let top = S::abs(_top % modulus) + S::one();
            let near = S::abs(_near % modulus);
            let far = S::abs(_far % modulus) + S::one();

            (left, right, bottom, top, near, far)
        })    
        .prop_map(|(left, right, bottom, top, near, far)| {
            let (spec_left, spec_right) = if left > right {
                (right, left)
            } else {
                (left, right)
            };
            let (spec_bottom, spec_top) = if bottom > top {
                (top, bottom)
            } else {
                (bottom, top)
            };
            let (spec_near, spec_far) = if near > far {
                (far, near)
            } else {
                (near, far)
            };
            let spec = PerspectiveSpec::new(
                spec_left, spec_right, spec_bottom, spec_top, spec_near, spec_far
            );

            Perspective3::new(spec)
        })
        .no_shrink()
}

fn any_orthographic_projection<S>() -> impl Strategy<Value = Orthographic3<S>>
    where S: ScalarFloat + Arbitrary
{
    any::<(S, S, S, S, S, S)>()
        .prop_map(|(_left, _right, _bottom, _top, _near, _far)| {
            let modulus: S = num_traits::cast(1_000_000).unwrap();
            let left = -S::abs(_left % modulus) - S::one();
            let right = S::abs(_right % modulus) + S::one();
            let bottom = -S::abs(_bottom % modulus) - S::one();
            let top = S::abs(_top % modulus) + S::one();
            let near = S::abs(_near % modulus);
            let far = S::abs(_far % modulus) + S::one();

            (left, right, bottom, top, near, far)
        })    
        .prop_map(|(left, right, bottom, top, near, far)| {
            let (spec_left, spec_right) = if left > right {
                (right, left)
            } else {
                (left, right)
            };
            let (spec_bottom, spec_top) = if bottom > top {
                (top, bottom)
            } else {
                (bottom, top)
            };
            let (spec_near, spec_far) = if near > far {
                (far, near)
            } else {
                (near, far)
            };
            let spec = OrthographicSpec::new(
                spec_left, spec_right, spec_bottom, spec_top, spec_near, spec_far
            );

            Orthographic3::new(spec)
        })
        .no_shrink()
}

/// Generate the property tests for perspective projection transformations.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name of the module we place the tests in to 
///    separate them from each other for each scalar type to prevent namespace
///    collisions.
/// * `$VectorN` denotes the name of the vector type.
/// * `$ScalarType` denotes the underlying system of numbers.
/// * `$ProjGen` denotes the type of the projection transformation.
/// * `$VecGen` is the name of a function or closure for generating vector 
///    examples.
/// * `$PointGen` is the name of a function for generating point examples.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! perspective_projection_props {
    ($TestModuleName:ident, $ScalarType:ty, $ProjGen:ident, $VecGen:ident, $PointGen:ident, $tolerance:expr) => {
        #[cfg(test)]
        mod $TestModuleName {
            use cglinalg::approx::{
                relative_eq,
            };
            use super::{
                $ProjGen,
                $VecGen,
                $PointGen,
            };
            use proptest::prelude::*;

            proptest! {
                /// A perspective projection over floating point scalars should
                /// be approximately invertible.
                ///
                /// Given an perspective projection transformation `P` and a vector `v`, there
                /// there is an inverse transformation `Q` such that
                /// ```text
                /// Q * P * v ~= v
                /// ```
                #[test]
                fn prop_perspective_projection_project_vector(
                    projection in $ProjGen::<$ScalarType>(), vector in $VecGen::<$ScalarType>()) {
                    
                    let projected_vector = projection.project_vector(&vector);
                    let unprojected_vector = projection.unproject_vector(&projected_vector);

                    prop_assert!(
                        relative_eq!(unprojected_vector, vector, epsilon = $tolerance),
                        "projection = {}\nvector = {}\nprojected_vector = {}\nunprojected_vector={}",
                        projection, vector, projected_vector, unprojected_vector
                    );
                }

                /// A perspective projection over floating point scalars should
                /// be approximately invertible.
                ///
                /// Given a perspective projection transformation `P` and a point `p`, there
                /// there is an inverse transformation `Q` such that
                /// ```text
                /// Q * P * p ~= p
                /// ```
                #[test]
                fn prop_perspective_projection_project_point(
                    projection in $ProjGen::<$ScalarType>(), point in $PointGen::<$ScalarType>()) {
                    
                    let projected_point = projection.project_point(&point);
                    let unprojected_point = projection.unproject_point(&projected_point);

                    prop_assert!(
                        relative_eq!(unprojected_point, point, epsilon = $tolerance),
                        "projection = {}\npoint = {}\nprojected_point = {}\nunprojected_point={}",
                        projection, point, projected_point, unprojected_point
                    );
                }

                /// The inverse of a perspective projection over floating point scalars should
                /// be approximately invertible.
                ///
                /// Given a perspective projection transformation `P^-1` and a projected vector
                /// `pv`, there is an inverse of `P^-1` that satisfies
                /// ```text
                /// P^-1 * (P^-1)^-1 * pv ~= pv
                /// ```
                #[test]
                fn prop_perspective_projection_inverse_unproject_vector(
                    projection in $ProjGen::<$ScalarType>(), 
                    vector in $VecGen::<$ScalarType>()) {
                    
                    let unprojected_vector = projection.unproject_vector(&vector);
                    let projected_vector = projection.project_vector(&unprojected_vector);

                    prop_assert!(
                        relative_eq!(projected_vector, vector, epsilon = $tolerance),
                        "projection = {}\nvector = {}\nunprojected_vector = {}\nprojected_vector={}",
                        projection, vector, unprojected_vector, projected_vector
                    );

                }

                /// The inverse of a perspective projection over floating point scalars should
                /// be approximately invertible.
                ///
                /// Given an perspective projection transformation `P^-1` and a projected point
                /// `pp`, there is an inverse of `P^-1` that satisfies
                /// ```text
                /// P^-1 * (P^-1)^-1 * pp ~= pp
                /// ```
                #[test]
                fn prop_perspective_projection_inverse_unproject_point(
                    projection in $ProjGen::<$ScalarType>(), point in $PointGen::<$ScalarType>()) {
                    
                    let unprojected_point = projection.unproject_point(&point);
                    let projected_point = projection.project_point(&unprojected_point);

                    prop_assert!(
                        relative_eq!(projected_point, point, epsilon = $tolerance),
                        "projection = {}\npoint = {}\nunprojected_point = {}\nprojected_point={}",
                        projection, point, unprojected_point, projected_point
                    );
                }
            }
        }
    }
}

perspective_projection_props!(
    perspective_f64_props, 
    f64, 
    any_perspective_projection, 
    any_vector3, 
    any_point3, 
    1e-7
);
perspective_projection_props!(
    perspective_fov_f64_props, 
    f64, 
    any_perspective_fov_projection, 
    any_vector3, 
    any_point3, 
    1e-7
);


/// Generate the property tests for orthographic projection transformation.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name of the module we place the tests in to 
///    separate them from each other for each scalar type to prevent namespace
///    collisions.
/// * `$VectorN` denotes the name of the vector type.
/// * `$ScalarType` denotes the underlying system of numbers.
/// * `$ProjGen` denotes the type of the projection transformation.
/// * `$VecGen` is the name of a function or closure for generating vector 
///    examples.
/// * `$PointGen` is the name of a function for generating point examples.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! orthographic_projection_props {
    ($TestModuleName:ident, $ScalarType:ty, $ProjGen:ident, $VecGen:ident, $PointGen:ident, $tolerance:expr) => {
        #[cfg(test)]
        mod $TestModuleName {
            use cglinalg::approx::{
                relative_eq,
            };
            use super::{
                $ProjGen,
                $VecGen,
                $PointGen,
            };
            use proptest::prelude::*;

            proptest! {
                /// An orthographic projection over floating point scalars should
                /// be approximately invertible.
                ///
                /// Given an orthographic projection transformation `P` and a vector `v`, there
                /// there is an inverse transformation `Q` such that
                /// ```text
                /// Q * P * v ~= v
                /// ```
                #[test]
                fn prop_orthographic_projection_project_vector(
                    projection in $ProjGen::<$ScalarType>(), vector in $VecGen::<$ScalarType>()) {
                    
                    let projected_vector = projection.project_vector(&vector);
                    let unprojected_vector = projection.unproject_vector(&projected_vector);

                    prop_assert!(
                        relative_eq!(unprojected_vector, vector, epsilon = $tolerance),
                        "projection = {}\nvector = {}\nprojected_vector = {}\nunprojected_vector={}",
                        projection, vector, projected_vector, unprojected_vector
                    );
                }

                /// An orthographic projection over floating point scalars should
                /// be approximately invertible.
                ///
                /// Given an orthographic projection transformation `P` and a point `p`, there
                /// there is an inverse transformation `Q` such that
                /// ```text
                /// Q * P * p ~= p
                /// ```
                #[test]
                fn prop_orthographic_projection_project_point(
                    projection in $ProjGen::<$ScalarType>(), point in $PointGen::<$ScalarType>()) {
                    
                    let projected_point = projection.project_point(&point);
                    let unprojected_point = projection.unproject_point(&projected_point);

                    prop_assert!(
                        relative_eq!(unprojected_point, point, epsilon = $tolerance),
                        "projection = {}\npoint = {}\nprojected_point = {}\nunprojected_point={}",
                        projection, point, projected_point, unprojected_point
                    );
                }

                /// An inverse of an orthographic projection over floating point scalars should
                /// be approximately invertible.
                ///
                /// Given an orthographic projection transformation `P^-1` and a projected vector 
                /// `pv`, there is an inverse of `P^-1` that satisfies
                /// ```text
                /// P^-1 * (P^-1)^-1 * pv ~= pv
                /// ```
                #[test]
                fn prop_orthographic_projection_inverse_unproject_vector(
                    projection in $ProjGen::<$ScalarType>(), 
                    vector in $VecGen::<$ScalarType>()) {
                    
                    let unprojected_vector = projection.unproject_vector(&vector);
                    let projected_vector = projection.project_vector(&unprojected_vector);

                    prop_assert!(
                        relative_eq!(projected_vector, vector, epsilon = $tolerance),
                        "projection = {}\nvector = {}\nunprojected_vector = {}\nprojected_vector={}",
                        projection, vector, unprojected_vector, projected_vector
                    );

                }

                /// An inverse of an orthographic projection over floating point scalars should
                /// be approximately invertible.
                ///
                /// Given an orthographic projection transformation `P^-1` and a projected point
                /// `pp`, there is an inverse of `P^-1` that satisfies
                /// ```text
                /// P^-1 * (P^-1)^-1 * pp ~= pp
                /// ```
                #[test]
                fn prop_orthographic_projection_inverse_unproject_point(
                    projection in $ProjGen::<$ScalarType>(), point in $PointGen::<$ScalarType>()) {
                    
                    let unprojected_point = projection.unproject_point(&point);
                    let projected_point = projection.project_point(&unprojected_point);

                    prop_assert!(
                        relative_eq!(projected_point, point, epsilon = $tolerance),
                        "projection = {}\npoint = {}\nunprojected_point = {}\nprojected_point={}",
                        projection, point, unprojected_point, projected_point
                    );
                }
            }
        }
    }
}

orthographic_projection_props!(
    orthographic_f64_props, 
    f64, 
    any_orthographic_projection, 
    any_vector3, 
    any_point3, 
    1e-7
);

