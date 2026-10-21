# Fixtures

- `synthetic_l{0,1,2}.ksplat` -- produced by GaussianSplats3D's own
  `SplatBufferGenerator` from the deterministic synthetic scene in
  `examples/common/compressed_scene_fixture.h`.
- `splat_transform_minimal.ksplat` / `splat_transform_minimal.splat` -- the
  4-Gaussian `minimal.ksplat` / `minimal.splat` from
  [playcanvas/splat-transform](https://github.com/playcanvas/splat-transform)
  (`test/fixtures/splat/`, MIT licence, (c) PlayCanvas). A second tool's
  `.ksplat` writer, whose level-0 files differ from GaussianSplats3D's: the
  header's `maxSplatCount`/`sectionCount` are 0 (the real count is at offset 16)
  and the section carries a real 12-byte bucket table at compression level 0.
