# Computer Graphics Linear Algebra Library
## Introduction
The `cglinalg` library is a low-dimensional linear algebra library aimed 
primarily at real-time computer graphics, game engine development, and 
real-time collision detection. This library provides a strongly typed system 
for developing computer graphics applications.

The design of the library has the following goals in mind:
* **Cross Platform** -- It should be portable to other ecosystems like 
  other C/C++ and Rust libraries. Every data type can be treated like a 
  fixed-sized array so they can be sent to across FFI boundaries.
* **Few Dependencies** -- To support portability and maintainability, 
  `cglinalg` is designed to use few external dependencies. This makes it simpler 
  to integrate into applications. the biggest dependency is `proptest` which is 
  not necessary to use the library in software applications, only for running 
  tests.
* **Type Safety** -- Leverage Rust's type system and zero-cost abstractions 
  to ensure code correctness, abstraction, and intelligibility do not come 
  at the cost of performance.
* **Flexibility** -- The data types in `cglinalg` are parametric over their 
  scalars so they can operate on multiple scalar types. 
* **Simplicity** -- The system should be easy to understand and use.
* **Speed And Efficiency** -- Since the main use-case for the library is real-time 
  applications, operations should be fast and efficient.
* **Ergonomics** -- The types and the language of the documentations are designed 
  to be understandable to working graphics programmers. The main prerequisite 
  for understanding the language of the library documentation is elementary 
  linear algebra. Other concepts such as quaternions are explained where possible.
* **Composability** -- The library should serve as a cornerstone for other 
  computer graphics applications.

## Getting Started
To use `cglinalg` in your project, add as a dependency in your `Cargo.toml` file:
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
This save some extra typing for importing library features.

## Features
`cglinalg` is a low-dimensional linear-algebra library aimed at specific 
application domains that make heavy use of computer graphics. It includes the 
most common linear algebra operations for implementing rendering algorithms, 
real-time collision detection, etc. All data types are designed to be exportable to 
external interfaces such as foreign function interfaces or external hardware. This 
serves the `cglinalg` goal to be a platform agnostic foundation for computer graphics 
applications in other languages and ecosystems as well. Specific features of the 
library include:
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
* The library makes heavy use of property testing via the `proptest` library 
  in addition to Rust's type systems to ensure code correctness.

## Limitations On The Design
The library has design limitations for a number of reasons. 
* `cglinalg` is a low-dimensional library; it can only do linear algebra up to 
  dimensional four. If one needs high-dimensional transformations, there are other 
  libraries fit to the task rather than this one. This library is not a replacement
  for `numpy`, BLAS, or LAPACK. It is a counterpart to `DirectXMath` or `glm`.
* The library is designed specifically with graphics applications in mind, which 
  tend to be mathematically simpler than other modeling and simulation applications. 
  As a consequence this library does not support all of the operations commonly used 
  in modeling and simulation tasks. If one needs operations such as finding eigenvalues 
  and eigenvectors or computing factorizations of matrices, this library does not 
  provide that. 
* In keeping with simplicity as one of the project goals, the underlying storage of 
  all data types in this library are statically allocated arrays. This is advantagous 
  in the low-dimensional case when the data types have small sizes, but this is a 
  limitation in the higher-dimensional case where dynamic storage allocation of storage 
  or using the heap may be desireable.

## Limitations On The Implementation
The limitations on the implementation are addressed in the project roadmap. 
The biggest one is than it does not presently leverage SIMD instructions to optimize 
operations yet.

## Other Libraries
The Rust ecosystem has a number of computer graphics libraries now. Some 
highlights include
* (cgmath)[https://crates.io/crates/cgmath] -- One of the original Rust real-time 
  game mathematics libraries, and one of the most commonly used ones.
* (nalgebra)[https://nalgebra.org] -- The most powerful linear algebra library in 
  the Rust ecosystem. It provides a strongly typed system for most linear algebra used 
  modeling and simulation in arbitrarily many dimensions, including computer graphics. 
  This system is comparable to BLAS and LAPACK.
* (euclid)[https://crates.io/crates/euclid] -- A collection of strongly typed math tools 
  for computer graphics with an inclination towards 2d graphics and layout. This one 
  is used in the `Servo` browser engine.
* (vecmath)[https://crates.io/crates/vecmath] -- A simple and type agnostic Rust library 
  for vector math designed for reexporting.

## Project Roadmap
Major outstanding project goals include:
* Complete the ontology of low-dimensional matrix types by implementing nonsquare 
  matrices.
* Make the basic types object order agnostic by adding support for row-major 
  order matrices.
* Improve performance with SIMD optimizations.
* Implement swizzle operations.
