# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Change log dates follow the ISO 8601 standard (YEAR-MONTH-DAY).

## [Unreleased]

## [0.21.0] - 2024-08-26
Redesign projection matrix specification.

### Added
- Added much more documentation on the mathematical properties of the projection 
transformations.

### Changed
- Changed the way frustum projections are specified such that all plane parameters
  (left, right, bottom, top, near, far) are positive, so that the frustum parametrization
  is coordinate-system invariant.

## [0.20.0] - 2023-11-15
Integate a new approximate comparison library.

### Added
- Introduced various `try_inverse` functions that return an `Option<Self>` for 
  objects that may not have an inverse.
- Added a `SimdScalarCmp` trait to marshall together the various approximate comparison
  traits from `approx_cmp` into one trait.

### Changed
- Documentation improvements.
- Removed `approx` crate in favor of the `approx_cmp` floating point comparison crate.
- Changes in documentation's matrix notation.
- Change type signature of various `inverse` functions to return a `Self` instead of `Option<Self>`.
- Various bits of refactoring.
- Redesign point separation property tests to be less prone to catastrophic 
  cancellations causing the tests to spuriously fail.

## [0.19.1] - 2023-11-04
Some small project level improvements.

### Added
- Add `rustfmt` linting settings.

### Changed
- Move the the changelog to the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

## [0.19.0] - 2023-10-23
Major API and library redesign. Massive improvements in documentation quality
and testing quality in `cglinalg_transform`.

### Added
- Expand and immprove the test suite for the `cglinalg_transform` subcrate.
- Added property tests to every transformation in the `cglinalg_transform` subcrate.
- Introduced a type level constraint system for constraining the dimensions of matrices.
- Added vector space norms for matrix data types.
- Added complex and quaternionic trigonometry.

### Changed
- Removed the `RC` parameter from the matrix data type. The type level constraint system
  now takes care of ensuring that the underlying arrays are the correct size.
- Removed the explicit affine dimension parameter from reflection and shearing
  affine transformations though some clever redesign of the internals. This also has the 
  nice side effect of relying more on the geometric interpretation of these transformations
  instead of being just an affine matrix wrapper.
- Completely redesigned the perspective projection and orthographic projection APIs with much
  better documentation. Reorganized the internals to be more space efficient and more compatible
  with dispatching to the GPU.
- Added more exposition deriving the formulas for various data types.
- Complete redesign of the numeric stack to be more flexible.
- Reorganize internal architecture to be a collection of crates.

## [0.15.6] - 2022-06-07

### Added
- Add reflection for vectors and componentwise multiplication for matrices and 
  vectors.

## [0.15.2] - 2022-03-20
Introduction of complex numbers.

### Added
- Add complex number module.

### Changed
- Refactor the internals of quaternions to keep all coordinates in a
  `Vector4` type. This paves the way for future performance optimizations.
- General documentation and code improvements.

## [0.10.0] - 2020-10-01

### Added
- Include examples for how to use different functions in the documentation.

### Changed
- Various breaking interface renames.

## [0.9.0] - 2020-09-29

### Added
- Include Unit type for enforcing the requirement that certain vectors be
  unit vectors.
- Include more constructors for various types.

### Removed
- Remove the glm module.

## [0.8.0] - 2020-09-26

### Added
- Add affine transformations.
- Add support for Euclidean points.
- Add perspective projection and orthographic projection as transformations 
  instead of just matrices.
- Include existing library features in test suite except for affine transformations.
