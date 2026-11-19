# TODO
- Find a way to generalize the perspective and orthographic projection 
  transformations in the projection module to accommodate an arbitrary
  choice of local frame and normalized device coordinates, instead of
  only providing the one that OpenGL uses.
- Remove matrix dimension constraints once [generic_const_exprs](https://github.com/rust-lang/rust/issues/76560)
  stabilizes.
