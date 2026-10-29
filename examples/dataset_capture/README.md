# Dataset capture

`sabercat_dataset_capture` is a headless, deterministic offline renderer for
training-corpus generation. It accepts any scene handled by `loadScene`,
normalizes positions and scales with the scene's robust focus sphere, and
renders a fixed orbit with the full Vulkan rasterizer.

```powershell
sabercat_dataset_capture scene.spz output-directory `
  --views 32 --size 64 --vertical-fov 55 --elevation 12 --margin 1.15
```

The output contains `capture.json` plus one little-endian float32 RGBA file and
one little-endian float32 depth file per view. Matrices in JSON are row-major
camera-to-world transforms using Sabercat's right-handed, +Y-up, camera-minus-Z
convention. RGB is display-referred sRGB, premultiplied by coverage, with no background composited; alpha carries coverage. Depth is
the frontmost contributing Gaussian in normalized scene units, with zero for
pixels without a contributor.

The command records the source Gaussian count, source and rendered SH degrees,
antialiasing flag, original normalization transform, intrinsics, and exact
camera transforms. It renders SH0 by design so offline labels do not introduce
view-dependent appearance into the multi-asset training corpus.
