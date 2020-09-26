# Computer Graphics Linear Algebra Library
## Introduction
The `cglinalg` library is a low-dimensional linear algebra library aimed 
primarily at real-time computer graphics, game engine development, and 
real-time collision detection. This library provides a strongly typed system 
for developing computer graphics applications.

The design of the library has the following goals in mind:
* **Ergonomics** -- The system should be easy to understand and use. The types 
  and the language of the documentation are designed to be understandable to 
  working graphics programmers.
* **Cross Platform** -- It should be portable to other ecosystems like 
  other C/C++ and Rust libraries. Every data type can be treated like a 
  fixed-sized array so they can be sent to across FFI boundaries.
* **Few Dependencies** -- The library should be relatively self-contained. To 
  support portability and maintainability, `cglinalg` is designed with few 
  external dependencies. The biggest dependency---`proptest`---is a development 
  dependency only.
* **Type Safety** -- Leverage Rust's type system and zero-cost abstractions 
  to ensure code correctness, abstraction, and intelligibility do not come 
  at the cost of performance.
* **Flexibility** -- The library should serve as a type-agnostic cornerstone 
  for computer graphics applications. The data types in `cglinalg` are
  generic over their scalars so they can operate on multiple scalar types.
* **Speed And Efficiency** -- Operations should be fast and efficient. SIMD 
  instructions and architecture specific optimizations should be used where 
  possible.

## Getting Started
To use library in your project, add `cglinalg` as a dependency in your 
`Cargo.toml` file:
```
[dependencies]
cglinalg = "0.8"
```
After that, place the crate declaration in either your `lib.rs` file or 
your `main.rs` file
```rust
extern crate cglinalg;
```
After that you can either import the traits and types as you need them, or you 
can simplify the type importing process by glob-importing the prelude:
```rust
use cglinalg::prelude::*;
```
This saves some extra typing when importing from the library.

## Features
`cglinalg` is a low-dimensional linear-algebra library aimed at specific 
application domains that make heavy use of computer graphics. It includes the 
most common linear algebra operations for implementing rendering algorithms, 
real-time collision detection, etc. All data types are designed to be exportable 
to external interfaces such as foreign function interfaces or external hardware. 
This serves the `cglinalg` goal to be a platform agnostic foundation for 
computer graphics applications in other languages and ecosystems as well. 
Specific features of the library include:
* Basic linear algebra with matrices and vectors up to dimension four.
* Quaternions, euler angles, and rotation matrices for doing rotations.
* All data types are parametrized to work over a large range of numerical types.
* An optional transformation system for working with affine and projective 
  transformations on points and vectors. This library distinguishes points from 
  vectors and locations in space vs. displacements in space. This matters when 
  working with working with affine transformations in homogeneous coordinates.
* Affine transformations including translation, reflection, shear, scale, 
  and rotation operations.
* Orthographic projections and perspective projections for camera models.
* The ability to treat vectors and matrices as arrays so they can be sent 
  to across API boundaries.
* Typed angles and typed angle trigonometry that statically guarantee that 
  trigonometry is done in the right units.
* The library makes heavy use of property testing via the `proptest` crate
  in addition to Rust's type system to ensure code correctness.

## Limitations On The Design
The library has design limitations for a number of reasons. 
* `cglinalg` is a low-dimensional library; it can only do linear algebra up to 
  dimensional four. If one needs high-dimensional transformations, there are other 
  libraries fit to the task rather than this one. This library is not a replacement
  for `numpy`, BLAS, or LAPACK. It is a Rust counterpart to `DirectXMath` or `glm`.
* The library is designed specifically with graphics applications in mind, which 
  tend to be mathematically simpler than other modeling and simulation applications. 
  As a consequence this library does not support most of the operations commonly used 
  in modeling and simulation tasks.
* In keeping with simplicity as one of the project goals, the underlying storage of 
  all data types in this library are statically allocated arrays. This is advantagous 
  in the low-dimensional case when the data types have small sizes, but this is a 
  limitation in the higher-dimensional case where dynamic storage allocation of storage 
  or using the heap may be desireable.

## Limitations On The Implementation
The limitations on the implementation are addressed in the project roadmap. 
The biggest one is than it does not presently leverage SIMD instructions to optimize 
operations yet.

## Acknowledgements
The Rust ecosystem has a number of linear algebra libraries including:
* [cgmath](https://crates.io/crates/cgmath) -- One of the original Rust graphics 
  mathematics libraries, and one of the most commonly used ones.
* [nalgebra](https://nalgebra.org) -- The most powerful linear algebra library 
  in the Rust ecosystem. It provides a strongly typed system for most linear 
  algebra in arbitrarily many dimensions. It is useful in many domains that do 
  heavy numerical computation, including computer graphics.
* [euclid](https://crates.io/crates/euclid) -- A collection of strongly typed 
  math tools for computer graphics with an inclination towards 2d graphics and 
  layout. This one is used in the `Servo` browser engine.
* [vecmath](https://crates.io/crates/vecmath) -- A simple and type agnostic 
  Rust library for vector math designed for reexporting.

## Project Roadmap
Major outstanding project goals include:
* Implement nonsquare nonsquare matrix types up to dimension four.
* Add support for row-major order matrices and vectors.
* Improve performance with SIMD optimizations.
* Implement swizzle operations.
