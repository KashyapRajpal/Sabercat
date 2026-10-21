# File formats

This document describes every file format Sabercat reads or writes:

- the scene inputs that `loadScene()` accepts,
- the cloud-v1 directory (`sabercat.cloud-v1`) and its relight sidecar (`sabercat.relight-v1`),
- the project file (`.sabercat`),
- the PLY and glTF exports.

Use it to check what a file must contain, what a loader does to it, and what a writer produces.

- [library.md](library.md) covers the C++ calls that read and write these formats.
- [studio.md](studio.md) covers the Studio menu items that use them.

## Summary

| Format | Direction | Identified by | Section |
| --- | --- | --- | --- |
| `.ply` (3DGS) | read, write | `ply` magic and extension | [PLY](#ply) |
| `.splat` | read | extension only (headerless) | [.splat](#splat) |
| `.spz` | read | extension and container | [.spz](#spz) |
| `.ksplat` | read | extension | [.ksplat](#ksplat) |
| `.sog` or `meta.json` | read | extension or file name | [.sog](#sog) |
| `.gltf`, `.glb` (`KHR_gaussian_splatting`) | read, write (`.glb`) | extension and binary magic | [glTF](#gltf-export) |
| cloud-v1 directory | read, write | `manifest.json` schema | [cloud-v1](#cloud-v1) |
| relight sidecar | read, write | `manifest.json` schema | [Relight sidecar](#relight-sidecar) |
| `.sabercat` project | read, write | `project.json` format | [Project file](#project-file) |

Input files that name a COLMAP or SfM `input.ply` are rejected with a message that says so. That file holds point positions and colours only, not Gaussians.

---

## Canonical contract

Every loader produces the same in-memory layout, whatever the file stores:

| Field | Canonical form |
| --- | --- |
| `opacity` | sigmoid-activated, in `[0, 1]` |
| `scale` | exp-activated, linear standard deviation |
| `rotation` | normalized, `xyzw` order |
| SH coefficient 0 | raw DC term (`f_dc`), not display RGB |
| SH coefficients 1..15 | band-ascending; unused slots are exactly zero |
| SH degree | taken from the file, never hard-coded |
| positions | passed through, no reordering |

Storage is structure-of-arrays with 16-byte strides for vector streams (see `include/sabercat/render_config.h` and `shaders/config.slangh`). The canonical SH layout holds 16 coefficients per Gaussian, which is degree 3.

Degree-0 colour is reconstructed as `rgb = sh0 * SH_C0 + 0.5`, with `SH_C0 = 0.28209479177387814`.

A loader never allocates a GPU resource. `Renderer::loadScene` uploads the host data.

Covariance is not stored. The GPU derives it from scale and rotation.

---

## Input formats

### Differences between formats

Each format stores some fields differently. A loader that copies another loader's behaviour gives a wrong result without an error. Check every row when you add or change a loader.

| | `.ply` | `.splat` | `.spz` |
| --- | --- | --- | --- |
| scale stored as | log | linear | log |
| loader does | `exp()` | nothing | `exp()` |
| opacity stored as | pre-sigmoid | activated u8 | pre-sigmoid |
| loader does | `sigmoid()` | `b / 255` | `sigmoid()` |
| SH DC stored as | raw `f_dc` | display RGB u8 | raw `f_dc` |
| loader does | nothing | invert | nothing |
| quaternion order | `wxyz` | `wxyz` | `xyzw` |
| loader does | reorder | reorder | nothing |
| quaternion normalized in file | no | no | effectively |
| loader does | normalize | normalize | normalize (defensive) |
| higher SH layout | channel-major | absent | coefficient-major, RGB interleaved |
| max SH degree | 3 | 0 | 4 (truncated to 3) |
| coordinate system | RDF | RDF-like | RUB |
| needs `setFlipUpAxis` | usually | usually | no |

| | glTF `KHR_gaussian_splatting` | cloud-v1 | `.ksplat` | `.sog` |
| --- | --- | --- | --- | --- |
| scale stored as | linear | linear | linear (f32 or f16) | log, codebook |
| loader does | nothing | nothing | nothing | `exp()` |
| opacity stored as | activated (f32, normalized u8 or u16) | activated | activated u8 | activated u8 |
| loader does | decode, reject outside `[0, 1]` | nothing | `b / 255` | `b / 255` |
| SH DC stored as | raw | raw (`sh0`) | display RGB u8 | raw `f_dc` (codebook) |
| loader does | nothing | nothing | invert | nothing |
| quaternion order | `xyzw` | `xyzw` | `wxyz` | `wxyz`, smallest-three |
| loader does | nothing (renormalize integers) | nothing | reorder | reorder |
| higher SH | one VEC3 accessor per coefficient | `sh_rest` `[N, K, 3]` | per band, channel-major | palette texels |
| max SH degree | 3 | 3 (higher truncated) | 2 | 3 |
| coordinates | right-handed, +Y up (no flip) | right-handed, +Y up (no flip) | as the source `.ply` | as the source `.ply` |

### PLY

Reference 3DGS `.ply` files. Only `binary_little_endian` is supported.

Properties read: `x y z`, `f_dc_0..2`, `f_rest_*`, `opacity`, `scale_0..2`, `rot_0..3`.

- `scale` is `exp(raw)` per axis.
- `opacity` is `sigmoid(raw)`.
- `rot_0` is `w`. The stored quaternion is `(w, x, y, z)`, unnormalized. The loader reorders it to `xyzw` and normalizes it.
- `f_rest` is channel-major: all R coefficients, then all G, then all B. It is not interleaved per coefficient.
- The SH degree comes from the number of `f_rest_N` properties: 0, 3, 8 or 15 per channel gives degree 0, 1, 2 or 3.
- Property offsets are resolved once per file, not once per vertex.

The loader also reads the relightable (R3DG) attributes. See [PLY export](#ply-export) for the inverse mapping.

### .splat

A headerless file of back-to-back 32-byte records. It has no magic bytes, so it can only be rejected by its contents.

```
offset  size  field
0       12    position  3 x f32
12      12    scale     3 x f32   (already linear)
24       4    color     4 x u8    RGBA, display space, DC only
28       4    rotation  4 x u8    wxyz, (b - 128) / 128
```

- `scale` is already linear. Do not apply `exp()`.
- `color` is `round((f_dc * SH_C0 + 0.5) * 255)`. Invert it to recover `f_dc`.
- `alpha` is `round(sigmoid(opacity) * 255)`. Invert it with `b / 255`.
- The rotation bytes are `wxyz`. The decoded quaternion is not unit length, so normalize it.
- Only degree-0 colour is stored. Coefficients 1..15 are zero, and `shDegree()` returns 0.

Because a `.splat` has only DC colour, SH degree and appearance LOD have no effect on it. The viewer disables SH degrees above the degree the scene carries.

The loader checks the first 64 decoded records for finite positions and positive scales. A file of the wrong type whose size divides by 32 fails here instead of rendering as noise.

### .spz

Sabercat does not parse the `.spz` container. The `spz` library (vendored at `external/spz`, pinned to v3.0.0) decodes it. `spz_reader.cpp` only adapts the decoded values.

The adapter receives unpacked floats in the pre-activation form, the same shape as `.ply`:

| spz field | form | adapter does |
| --- | --- | --- |
| positions | xyz | passed through |
| scales | xyz, log | `exp()` |
| rotations | `xyzw` | renormalized, no reorder |
| alphas | pre-sigmoid | `sigmoid()` |
| colors | raw SH DC | coefficient 0 |
| sh | coefficient-major, RGB inner (`sh[k*3 + c]`) | re-laid out to one vec4 per coefficient |

- The coordinate system is requested as RUB, which matches this renderer's +Y up and -Z forward. No `setFlipUpAxis()` is needed.
- Degree 4 input is truncated to degree 3 with a warning. The first 16 coefficients are kept.
- A decode failure returns zero points from the library. The adapter turns that into a validation error.

### .ksplat

Read from the layout used by GaussianSplats3D (`SplatBuffer.js`). That code is the only reference for the format.

- A 4096-byte file header, then 1024-byte section headers.
- Sections hold bucket lengths, bucket centres, and fixed-size records.
- Compression levels 0 (f32), 1 (u16 bucket-relative centres, f16 values) and 2 (as level 1, with u8 SH over a header min/max range).
- Degree 3 is rejected because the reference does not define a record layout for it. The maximum accepted degree is 2.

### .sog

A SOG bundle: lossless WebP images plus `meta.json` version 2. The input is a zip archive (stored or deflated, CRC checked) or an unbundled `meta.json` with its images beside it.

- Positions are 16-bit codes in log space.
- Quaternions are smallest-three, with the dropped component index in the tag byte.
- Scales and DC colour use codebooks. Higher SH uses a k-means palette.
- `meta.json` version 1 is refused.
- A streamed SOG (`lod-meta.json`) is refused by name.

### glTF `KHR_gaussian_splatting`

Reads `.gltf` and `.glb` files that use the ratified `KHR_gaussian_splatting` extension. The reader checks:

- Accessors: component types, normalized integers, `byteStride`, sparse accessors, `.glb` chunks, data URIs and external files.
- Node transforms. Positions take the full affine transform. Covariances compose exactly under uniform scale and are re-diagonalized under non-uniform scale. SH bands 1 to 3 are rotated to match the transform.
- The `ellipse` kernel is the only kernel accepted.
- `colorSpace` is `srgb_rec709_display` or `lin_rec709_display`. SH is passed through unconverted in both cases. Other values are read with a warning.
- Shear and mirroring are rejected. Unknown required extensions and partial SH degrees are rejected.
- `KHR_gaussian_splatting_compression_spz_2` is decoded through the same `spz` library. It is the only compression extension accepted.

For files from a draft of the spec that omit `kernel` and `colorSpace`, the reader uses `ellipse` and `srgb_rec709_display` and logs a warning. The defaults apply to SPZ-compressed primitives only.

### cloud-v1 directory

A cloud-v1 directory is also an input. See [cloud-v1](#cloud-v1).

---

## Canonical in-memory layout and limits

| Limit | Value |
| --- | --- |
| SH degree | 0 to 3 |
| SH coefficients per Gaussian | 16 |
| Relight visibility coefficients per Gaussian | 16 (mono SH, degree 3) |

Degree above 3 is truncated when a file is read. The cloud-v1 and PLY writers refuse a cloud with degree above 3. The glTF writer truncates it to 3.

---

## cloud-v1

cloud-v1 is the uncompressed, documented interchange form of a `GaussianCloud`. The schema id is `sabercat.cloud-v1`.

Files written before version 1.0 carry the legacy id `gaussmith.sabercat-cloud-v1`. The reader still accepts it. The writer never writes it.

### Directory layout

```
<directory>/
  manifest.json
  means.f32le.bin
  scales.f32le.bin
  rotations.f32le.bin
  opacities.f32le.bin
  sh0.f32le.bin
  sh_rest.f32le.bin        (only when shDegree > 0)
```

`CloudV1Reader::read` takes either the directory or the path to its `manifest.json`.

### Manifest

| Key | Value written | Checked on read |
| --- | --- | --- |
| `schema` | `sabercat.cloud-v1` | `sabercat.cloud-v1` or the legacy id |
| `count` | number of Gaussians | 1 to 2^28 |
| `coordinateSystem` | `right-handed-y-up-camera-minus-z` | must match |
| `quaternionOrder` | `xyzw` | must match |
| `colorSpace` | `linear-srgb` | a different value is read with a warning |
| `shDegree` | 0 to 3 | negative is an error, above 3 is truncated with a warning |
| `shLayout` | `splat-major-coefficient-major-rgb` | must match |
| `streams` | object, one entry per stream | see below |

Each stream entry has these keys:

| Key | Meaning |
| --- | --- |
| `file` | file name, relative to the manifest |
| `shape` | array shape, leading dimension is `count` |
| `scalarType` | `float32-little-endian` |
| `elementStrideBytes` | bytes of one leading element |
| `byteLength` | total bytes in the file |

### Streams

| Stream | Shape | Contents |
| --- | --- | --- |
| `means` | `[N, 4]` | position `xyz`, pad lane 0 |
| `scales` | `[N, 4]` | linear standard deviation `xyz`, pad lane 0 |
| `rotations` | `[N, 4]` | quaternion `xyzw` |
| `opacities` | `[N]` | activated opacity |
| `sh0` | `[N, 4]` | SH coefficient 0 (raw DC), pad lane 0 |
| `sh_rest` | `[N, K, 3]` | coefficients 1..K, RGB inner, `K = (shDegree + 1)^2 - 1` |

All values are little-endian `float32`. The writer writes host memory directly, so the file matches the in-memory layout.

A `confidence` stream is accepted on read and ignored, with a log message. The writer does not write it.

Unknown stream names are ignored with a warning.

### Reader checks

The reader rejects a file that would change the meaning of the data:

- A missing required stream, or a stream with a different scalar type, shape, stride or byte length.
- A file size that differs from the manifest.
- A non-finite position, SH value or opacity.
- A scale that is not positive and finite.
- An opacity outside `[0, 1]`.
- A zero-length or non-finite quaternion.
- `sh_rest` present when `shDegree` is 0.

A quaternion that is more than 1e-6 away from unit length is renormalized. A quaternion within that tolerance is kept bit for bit. Pad lanes are zeroed on read.

### Round trip

`CloudV1Writer` and `CloudV1Reader` are exact inverses. The round trip reproduces every stream bit for bit.

`CloudV1Writer::encode` returns the files in memory. `CloudV1Writer::write` writes them to a directory and creates the directory if needed.

Cloud-v1 has no antialiasing flag. The reader sets `antialiased` to false. The project file stores the flag per layer (see [Project file](#project-file)).

---

## Relight sidecar

Relight data is not part of cloud-v1. It is stored in a separate sidecar so that cloud-v1 keeps its definition. The schema id is `sabercat.relight-v1`.

### Layout

`RelightSidecar::encode` returns the files in memory. `RelightSidecar::decode` reads them into an existing cloud. The caller chooses where the files go. In a project, they go in `layers/<id>/relight/` (see [Project file](#project-file)).

### Manifest

| Key | Value |
| --- | --- |
| `schema` | `sabercat.relight-v1` |
| `count` | must equal the cloud's Gaussian count |
| `streams` | object with the streams the cloud has |

Each stream uses the same entry keys as cloud-v1. All four streams are optional. A stream the cloud does not have is not written.

| Stream | Shape | Contents |
| --- | --- | --- |
| `normals` | `[N, 4]` | unit normal `xyz`, pad lane |
| `base_color_roughness` | `[N, 4]` | linear RGB, roughness in `w` |
| `metallic` | `[N]` | metallic amount, 0 to 1 |
| `visibility` | `[N, 16]` | mono SH (degree 3) of ambient visibility |

When a stream is absent, the lighting pass uses a default:

| Stream absent | Default |
| --- | --- |
| `normals` | the shortest ellipsoid axis, facing the camera |
| `base_color_roughness` | colour from the SH DC term, roughness 0.8 |
| `metallic` | 0 |
| `visibility` | fully visible |

`decode` checks the schema, the count, each shape, byte length and scalar type. It reads every stream into temporary storage first. A rejected sidecar leaves the cloud unchanged.

---

## Project file

A `.sabercat` file is a zip archive. It holds one project: its layers, their transforms, and the application's state.

### Archive contents

```
project.json                 format, version, layers, application state
layers/<id>/                 the layer's cloud-v1 directory, as files
layers/<id>/relight/         the layer's relight sidecar, if the cloud has relight data
```

The archive uses stored (uncompressed) entries. The reader also accepts deflated entries.

Zip64 is not supported, in either direction. The reader rejects an archive with zip64 markers or more entries than a standard zip holds.

Entries the format does not name are skipped with a warning. This lets a newer writer add entries without breaking this reader.

### project.json

```json
{
  "format": "sabercat.project",
  "formatVersion": 1,
  "savedWith": { "sabercat": "1.0.0" },
  "layers": [
    {
      "id": "L1",
      "name": "Background",
      "visible": true,
      "locked": false,
      "antialiased": false,
      "transform": {
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "scale": 1.0
      }
    }
  ],
  "app": { }
}
```

| Key | Meaning |
| --- | --- |
| `format` | `sabercat.project`, required |
| `formatVersion` | `1`, required. A value above 1 is refused. |
| `savedWith.sabercat` | the Sabercat version that wrote the file |
| `layers` | array of layer objects, in draw order |
| `app` | the application's state, a JSON object. Sabercat stores it but does not read it. |

Each layer object has these keys:

| Key | Meaning |
| --- | --- |
| `id` | unique, 1 to 64 characters of `A-Z a-z 0-9 _ -`. It names the folder `layers/<id>/`. |
| `name` | display name, any UTF-8 text |
| `visible`, `locked` | flags |
| `antialiased` | the cloud's antialiasing flag, which cloud-v1 does not store |
| `transform.translation` | `[x, y, z]` |
| `transform.rotation` | quaternion `[x, y, z, w]`, unit length within 1e-4 |
| `transform.scale` | uniform scale, greater than 0 |

### Saving and loading

- `ProjectIO::save` builds and checks the whole project in memory first. An invalid project throws before any file is written.
- The file is written to `<path>.tmp` and then renamed over `<path>`. A failed save leaves an existing file unchanged.
- `ProjectIO::load` returns the clouds as host data. It does not upload them.
- A round trip is bit exact for every host stream, the SH degree, the antialiasing flag, the transform, flags, ids and names.

---

## PLY export

`PLYWriter::write` writes a 3DGS `.ply` that `PLYReader` reads back.

- The file is `binary_little_endian 1.0`.
- Every property is `float32`. There is no quantization.
- The header has a comment line `Written by Sabercat PLYWriter`.
- The file is written to `<path>.tmp` and then renamed over `<path>`.

### Properties

| Properties | Contents |
| --- | --- |
| `x y z` | position, as stored |
| `nx ny nz` | normals, or zeros when the cloud has no normals |
| `f_dc_0..2` | SH coefficient 0 |
| `f_rest_0..` | coefficients 1 and up, channel-major (all R, then G, then B) |
| `opacity` | `logit(opacity)` |
| `scale_0..2` | `log(scale)` |
| `rot_0..3` | quaternion `(w, x, y, z)` |

With relight data, the file also contains:

| Properties | Contents |
| --- | --- |
| `base_color_0..2`, `roughness` | inverse of the reader's `sigmoid * 0.77 + 0.03` and `sigmoid * 0.9 + 0.09` |
| `visibility_dc_0`, `visibility_rest_0..14` | raw visibility coefficients |
| `metallic` | raw value |

### Rules and clamps

- `opacity`, base colour and roughness go through `logit()`, which is infinite at 0 and 1. Each value is clamped to `[1e-6, 1 - 1e-6]` (`PLYWriter::kLogitEpsilon`) before the inverse. An opacity of 0 is written as `logit(1e-6)` and read back as `1e-6`.
- The number of base-colour and roughness values that were clamped is logged once as a warning.
- A relight attribute is written only when the cloud has base colour and roughness. `PLYReader` uses that attribute to recognize relightable files.
- Normals or visibility without base colour are refused. `PLYReader` would ignore them.
- `metallic` is written whenever the cloud has it.
- A scale that is not positive and finite is refused, and the error names the Gaussian.
- A non-finite value is refused.
- An empty cloud is refused.
- SH degree above 3 is refused.
- The host vectors must be populated. Call `downloadFromGPU()` first if the host copy was dropped.

---

## glTF export

`GLTFWriter::writeGLB` writes one binary glTF file (`.glb`) with the `KHR_gaussian_splatting` extension. It writes float32 accessors only. Nothing is quantized. The writer refuses any rotation, per Gaussian or on a layer transform, that is not unit length to the project tolerance (1e-4), and it names the layer and the Gaussian.

### Structure

- One mesh and one node per layer. The mesh name and the node name are both the layer name.
- The node's translation, rotation and scale are the layer's transform. The rotation is `xyzw`. The uniform scale is repeated in all three components.
- The scene contains all the nodes.
- Each mesh has one `POINTS` primitive.
- `asset.version` is `2.0`. `asset.generator` is `Sabercat` followed by the library version.
- `extensionsUsed` lists `KHR_gaussian_splatting`.
- `extensionsRequired` is not written. The ratified spec allows a loader that does not know the extension to fall back to the plain point cloud.

### Primitive attributes

| Attribute | Type | Contents |
| --- | --- | --- |
| `POSITION` | VEC3 | position (with min and max) |
| `KHR_gaussian_splatting:ROTATION` | VEC4 | quaternion `xyzw` |
| `KHR_gaussian_splatting:SCALE` | VEC3 | linear standard deviation |
| `KHR_gaussian_splatting:OPACITY` | SCALAR | activated opacity |
| `KHR_gaussian_splatting:SH_DEGREE_l_COEF_m` | VEC3 | raw SH coefficient, for `l` = 0 to the layer's degree |

The extension object on the primitive has `kernel` set to `ellipse` and `colorSpace` set to `srgb_rec709_display`.

The SH degree of each layer is `min(shDegree, 3)`. A layer of degree 0 writes only `SH_DEGREE_0_COEF_0`.

### Limits

- The writer refuses an empty layer list and a layer with no Gaussians.
- It refuses non-finite values, an opacity outside `[0, 1]`, a non-positive scale and a zero-length rotation.
- Relight data is not exported.
- The file is written to a temporary file and then renamed over the target.

### Studio exports

Studio exports the visible layers. The layer transforms are applied by the export.

- **PLY** merges the visible layers into one cloud with `mergeLayers` and writes it with `PLYWriter`. The SH degree of the merged cloud is the highest degree of the inputs. A relight stream is kept only when every visible layer has it. A stream that only some layers have is dropped with a warning.
- **glTF** writes one node per visible layer with `GLTFWriter`. Relight data is not exported.

---

## Adding an input format

1. Add a `SceneFormat` value and an extension case in `detectFormat()`. The extension decides. Magic bytes can reject a file.
2. Add `<name>_reader.h` and `<name>_reader.cpp` with a `static read(Device &, path)` function. Add the `.cpp` to `add_library(sabercat ...)` in `CMakeLists.txt`, or it will not be built.
3. Work out every row of the traps table for the new format before writing the conversion. Do not assume that it matches an existing loader.
4. The loader must not allocate GPU resources. `GaussianCloud` owns the host data and the buffers.
5. Call `cloud->setShDegree(...)` with the degree the file actually carries.
6. If the format is a versioned or compressed container, consider using a vendored decoder instead of a hand parser, as `.spz` does.
7. Add fixtures with checks that reject wrong values, and confirm that each check fails when its conversion is broken.
8. Run the forward and gaussian-cloud tests after any change to shared loader code.
9. Check the numbers from a headless program. Screen capture of Vulkan output is not a reliable check.

## Known gaps

- `.splatv` (animated or 4D Gaussians) is not supported.
- `.ksplat` files with SH degree 3 are not supported.
- SOG `meta.json` version 1 and streamed SOG (`lod-meta.json`) are not supported.
- glTF compression extensions other than `KHR_gaussian_splatting_compression_spz_2` are not supported.
- Zip64 archives are not supported for projects.
- The glTF export has no relight data, and the cloud-v1 format has no antialiasing flag. The project file stores that flag.
