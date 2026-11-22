# Comparing images with sabercat_compare

`sabercat_compare` compares two images. It prints PSNR and FLIP, and it can
compare HDR images with HDR-FLIP. It can write FLIP error maps, HDR exposure maps
and absolute-difference maps as PNG files, and a comparison report as JSON or CSV.
It also has a window for comparing images and videos interactively (split, wipe, flicker,
difference and FLIP views, a pixel probe, GPU PSNR, MSE, FLIP and HDR-FLIP, projects
and exports; see Window).
It is built from `apps/compare` when the applications are built, and it runs on
Windows only (PNG input and output go through the Windows Imaging Component).

## Command line

```powershell
sabercat_compare <reference> <test> [--flip-map out.png] [--diff-map out.png]
                 [--ppd N] [--no-flip] [--hdr] [--tone-mapper aces|hable|reinhard]
                 [--start-exposure X] [--stop-exposure Y] [--num-exposures N]
                 [--exposure-map out.png] [--report out.json|out.csv] [--help]
```

| Argument | Effect |
|----------|--------|
| `<reference>`, `<test>` | The two images: PNG, JPEG, WebP, OpenEXR (`.exr`) or raw captures (`.rgba32f`). Their sizes must match. Videos are not read here; open them in the window (see Video). |
| `--flip-map out.png` | Writes the FLIP error map as an 8-bit RGB PNG in magma. With `--hdr`, the HDR error map. |
| `--diff-map out.png` | Writes the absolute-difference map as an 8-bit RGB PNG in magma. |
| `--ppd N` | FLIP pixels per degree. A number from 1 to 1000 (inclusive); other values are a usage error (exit 2). The default is about 67 (see below). |
| `--no-flip` | Skips FLIP. Only PSNR is computed. Cannot be combined with `--flip-map`, or with `--hdr`. |
| `--hdr` | Runs HDR-FLIP on the linear values of the inputs (see below). |
| `--tone-mapper` *name* | The tone mapper of HDR-FLIP: `aces`, `hable` or `reinhard` (not case-sensitive). The default is `aces`. Needs `--hdr`. |
| `--start-exposure X` | The first exposure, in stops. Needs `--hdr`. Unset: derived from the reference. |
| `--stop-exposure Y` | The last exposure, in stops. Needs `--hdr`. Unset: derived from the reference. |
| `--num-exposures N` | The number of exposures, from 2 to 1000 (inclusive). Unset: `max(2, ceil(Y - X))`. Needs `--hdr`. |
| `--exposure-map out.png` | Writes the HDR exposure map as an 8-bit RGB PNG in magma. Needs `--hdr`. |
| `--report out.json` or `out.csv` | Writes the comparison report (see below). A `.csv` name in any case gets CSV; any other name gets JSON. |
| `--help` | Prints the usage and exits. |

Example:

```powershell
build\apps\compare\Release\sabercat_compare.exe render.png reference.png --flip-map flip.png
```

The output is a short report (the numbers here are only an example of the format):

```text
size: 1280x720
PSNR: 38.12 dB
MSE: 0.000253
FLIP: 0.021674 (ppd 67.00)
```

`PSNR: inf dB` means the two images are identical.

With `--hdr` the report has the lines `PSNR`, `MSE`, `linear MSE` and then the
HDR-FLIP line (the numbers are again only an example):

```text
size: 1024x1024
PSNR: 24.50 dB
MSE: 0.003548
linear MSE: 0.02184
FLIP (HDR, aces): 0.283547 (ppd 67.02, exposures -12.5423..0.9427, 14)
```

Exit codes: `0` on success, `1` on a failure (an unreadable file, sizes that
differ, a report that cannot be written, or a non-finite sample), and `2` on a
usage error. A usage error prints the message and the usage to stderr.

Usage errors include: an HDR option without `--hdr`, `--no-flip` with `--hdr`, a
start exposure above the stop exposure, a number of exposures outside 2 to 1000,
an exposure that is not a finite number, and an unknown tone mapper name.

### No arguments: the window

Run with no arguments, `sabercat_compare` opens its window. The window also opens
for a file count other than two, and for a scene, video or project file. Exactly
two image files, with or without CLI flags, run the command line. `--gui` opens
the window; combined with a CLI flag it is a usage error. `--selftest`,
`--validation` and `--no-validation` are window-only arguments, and a CLI flag
with any of them is a usage error.

## Window

Files given on the command line are opened in the window. A project file can be
opened from the window with Open.

**Media list.** The list holds up to 8 sources, and four rows are visible at a
time. Each row has an eye (show or hide the source), a gear (offset, background,
exposure and role: A, B or none) and Remove. A source whose file is missing shows
Locate, which picks the new file. Files can be dropped on the window, and Add
media adds files. A video is tagged `video` in its row (see Video).

**Views.** In Split, two sources are one image with a movable divider: A is left
of the divider and B right of it. Drag the handle (or the line) to move the
divider, drag elsewhere on the image to pan, and the wheel zooms. Three or four
sources show the row or the 2x2 grid of panes, with zoom and pan synced across the
panes. Overlay modes are in Single screen: wipe, flicker, diff, FLIP, the depth
and alpha maps and their differences. The single view shows A or B alone. Wipe,
flicker and difference views compare A and B; difference has a gain control. The
wipe in Single screen has the same handle: drag it to move the line. A hint line
at the top of the image names these gestures in Split. The FLIP view shows the FLIP
error map of A and B (see FLIP in the window). The zoom is a percentage, with 100%
and fit buttons. Sampling is nearest,
so zoomed pixels stay sharp. The pixel probe reads the values under the cursor. The single view also has Depth,
Alpha, Depth diff and Alpha diff maps, for scenes and depth sources (see Scene
sources).

**Sizes that differ.** The panes are padded to the largest size, and the padding
is drawn inverted so it is not mistaken for image. The difference view shows the
maximum error over the padded area. The metrics cover the overlap only. The
overlap is anchored at the top left or centered.

**Metrics.** The window computes PSNR and MSE on the GPU for the A and B pair. The
numbers match the CLI to 1e-6 relative MSE and 1e-4 dB. Premultiplied `.rgba32f`
captures are used unchanged, as the CLI does. FLIP and HDR-FLIP are computed on the
GPU as well (see FLIP in the window and HDR-FLIP in the window).

**FLIP in the window.** The single view has a FLIP view. It shows the LDR-FLIP error
map of A and B in magma, with the same padding and background rules as the other
views. In HDR mode it shows the HDR-FLIP map instead (see HDR-FLIP in the window).
The pair needs A and B both visible and Ready.

- **Pixels per degree.** The ppd control takes a value from 1 to 1000 and starts at
  the default, about 67. A value outside that range is refused, and the status line
  says so. Default sets the default again.
- **Mean.** The panel shows `FLIP` with its mean and the ppd, next to PSNR and MSE.
  In HDR mode the line also gives the tone mapper, the exposure range and the count.
  Without a FLIP result, the panel gives the reason, such as a second image that is
  missing or sources that do not overlap. NaN and infinite samples in the overlap are
  shown as a warning line, with their count.
- **Probe.** Hovering over the overlap shows the FLIP value of that pixel as
  `FLIP  <value>`. In HDR mode the value is the HDR-FLIP error of that pixel.
- **Export.** Export > FLIP map... writes the full-resolution map as a PNG over the
  padded extent. Export > Report JSON or CSV carries the FLIP mean and the ppd: in the
  JSON they are `metrics.flipMean` and the top-level `ppd`, and in the CSV they are the
  `flip_mean` and `ppd` columns. The mean is null, or an empty cell, when there is no
  FLIP result. With an HDR-FLIP result the JSON also has `"mode": "hdr"`, `toneMapper`,
  `exposures` (`start`, `stop`, `count`) and `metrics.linearMse`, and the CSV has the
  matching columns (`mode`, `tone_mapper`, `start_exposure`, `stop_exposure`,
  `exposures`, `linear_mse`).
- **When it runs.** In LDR mode, FLIP runs over the overlap of the pair. It runs again
  after a change to the sources, their offsets, visibility or roles, after a scene
  re-renders, and after a change of the ppd. Pan, zoom and changes of the view do not
  run it. HDR-FLIP runs only on request, or with Run on change (see HDR-FLIP in the
  window).
- **Padding.** Padding is the part of the padded extent outside the overlap. It shows
  the top of the magma map and is never run through FLIP.
- **Colors.** Premultiplied rgb is used as stored, as PSNR uses it. In LDR mode a
  Linear source (EXR) is clamped to [0,1] and sRGB-encoded before FLIP. In HDR mode
  the values are taken as described in HDR-FLIP in the window.
- **Non-finite samples.** In LDR mode, a NaN or infinite sample of A or B in the
  overlap is counted and treated as 0 or 1 in display space: NaN and negative infinity
  as 0, positive infinity as 1. The count is in the panel. HDR mode has its own rule
  (see HDR-FLIP in the window).
- **Memory.** LDR-FLIP needs about 68 bytes of device memory per overlap pixel, plus
  about 256 KiB. HDR-FLIP needs about 76 bytes per overlap pixel and the same fixed
  amount. When the free device memory is too small, FLIP is refused with a message
  that starts `FLIP needs about N MiB of GPU memory`, and the view shows the panes
  until the next change that makes FLIP run again.
- **Agreement.** The GPU mean agrees with the CLI's FLIP on the same pair within 5e-5.
  The self-test also checks the per-pixel error: the 99.9th percentile within 2e-3
  and the largest within 1e-2. For HDR-FLIP, the GPU mean agrees with the CLI's `--hdr`
  mean within 1e-4.

**HDR-FLIP in the window.** HDR-FLIP is the high dynamic range form of FLIP described
under What is computed. The FLIP mode row has LDR and HDR. HDR needs A and B both
visible and Ready, and at least one of them a Linear (EXR) source. Otherwise HDR is
disabled, with the tooltip "needs a Linear (EXR) source in the A/B pair". Raw
`.rgba32f` captures are Display sources in the window, so only EXR makes HDR
available. The window computes HDR-FLIP on the GPU.

- **Values.** A Linear source is used as stored, without clamping. A Display source
  (PNG, JPEG or WebP) is decoded from sRGB to linear after its values are clamped to
  [0,1]. Alpha is ignored, and premultiplied rgb is used as stored. These are the
  values of the CLI's `--hdr` path.
- **Mixed pair.** When A and B have different encodings, the panel shows a warning. In
  LDR mode the Linear source is clamped and sRGB-encoded. In HDR mode the Display
  source is decoded from sRGB to linear.
- **Non-finite samples.** A NaN or infinite sample of A or B in the overlap is counted
  per channel and for both sources, and its linear value is taken as 0, for Linear and
  for Display sources alike (so +inf of a Display source is 0, not white). The result is
  never NaN.
- **Tone mapper.** `aces` (the default), `hable` or `reinhard`, as in the CLI. The
  combo is labeled FLIP tone mapper.
- **Exposure range.** Auto is on by default. The start and stop come from A over the
  overlap, with FLIP's rule (see What is computed), and the count is
  `max(2, ceil(stop - start))`. With Auto exposure range off, Start (stops) and Stop
  (stops) fields are shown. With Auto count off, an Exposures field is shown, clamped to
  2 to 1000. A start or stop that is not finite, lies outside -126 to 126, or is above
  the stop is refused: it is not applied, the field shows the value in use again, and
  the status line says why.
- **All-black reference.** An all-black A has no range to derive. The window then
  evaluates FLIP's rule on a 1x1 gray image of FLT_EPSILON instead, so the range is a
  single point (start equal to stop) and the count is 2. The CLI refuses this case.
- **Running.** HDR-FLIP never runs per frame. Press Run HDR-FLIP to compute it, or tick
  Run on change (off by default) to compute it after each change that makes FLIP
  stale. The run is spread over frames in short steps, with a progress bar and a
  Cancel button. Any change that marks the metrics stale cancels a run and drops its
  result: a source change, an offset, a scene render, or a change of mode, ppd, tone
  mapper or range. Starting a run also drops the previous result. Cancel gives the
  reason `HDR-FLIP cancelled`. With Run on change off and no result, the panel says
  `HDR-FLIP has not run: press Run HDR-FLIP`.
- **Result.** The result line reads `FLIP (HDR, <tone>) <mean> (ppd <ppd>, exposures
  <start>..<stop>, <count>)`, with the mean to six decimals, the ppd to whole numbers
  and the exposures to four decimals.
- **PSNR and linear MSE.** In HDR mode, and whenever A or B is Linear, the PSNR and MSE
  lines are labeled "on the display transform": Linear values are clamped to [0,1] and
  sRGB-encoded, as the CLI does. After a run the panel also shows `linear MSE`, the
  mean squared difference of the linear values over the overlap, taken over the RGB
  samples. Pairs with a non-finite sample are skipped.
- **Exposure map.** Export > Exposure map PNG (full resolution)... writes the exposure
  of each pixel's largest error as a magma PNG over the padded extent, from the
  exposure index i / (count - 1). The padding shows the top of the map, as the FLIP
  map does. The window has no view of the exposure map.
- **Display controls.** A Linear source has a display exposure (EV) and a display tone
  mapper in its gear. They change only how that source is drawn. They do not change
  PSNR, MSE, FLIP or linear MSE, and they do not drop a FLIP result.
- **Deviations.** The per-pixel maximum over the exposures is a separate GPU pass. This
  is internal and not visible in the output. Raw `.rgba32f` captures have no
  "interpret as linear" setting in the window, so they stay Display sources.

**Project files.** A comparison is saved as a `*.sabercat-compare` file with New,
Open and Save. Source paths are stored relative to the project folder. A source on
another drive cannot be relative, so it is stored absolute and a load warns about
it. A project written by a newer format version is refused. The single view mode, the map
settings, the FLIP ppd, the FLIP mode and HDR settings and the scene settings and the
video and transport settings are stored too (see Scene sources and Video).

**Export.** The current view can be saved as a PNG, along with the full-resolution
difference map, the FLIP map (when there is a FLIP result), the exposure map (when
there is an HDR-FLIP result) and the padded pane images. The comparison report can be
saved as JSON or CSV. An export of a video shows the frame on screen at that moment,
and the report records its frame index and the clock (see Video).

**Self-test.** `sabercat_compare --selftest <dir>` runs the window's checks without
showing a window, and writes its output to `<dir>`. It prints one `[selftest]` line
per check and ends with `[selftest] PASS` (exit 0) or `[selftest] FAIL: ...`
(exit 1). The scene checks render a synthetic cloud as PLY and as SPZ, and compare each with an independent renderer. CTest runs it as `compare_selftest` (label `gpu`). `--validation` and
`--no-validation` set the Vulkan validation layer, as described in
[building.md](building.md#validation-layer).

**Console and files.** A launch from Explorer hides the console window that
Windows created for the process. A launch from a terminal keeps the console for
its output. The window writes no `imgui.ini`, so its layout is not kept between
runs.

**Vulkan.** `vulkan-1.dll` is delay-loaded, and it is loaded only when the window
opens. The command line works without a Vulkan runtime.

**Not in yet:** writing video files and HDR display output.

## Scene sources

A splat scene is added like an image: with Add media, or by dropping the file on
the window. A file is a scene when its extension is `.ply`, `.spz`, `.splat`,
`.ksplat`, `.sog`, `.gltf` or `.glb` (case-insensitive), or when it is named
`meta.json` (SOG). Every other file is an image. Scenes render on the window's
Vulkan device, so they need the window; the command line reads images only.

**Supported formats.** Scenes are read by the same readers as the viewer, so a
scene the viewer opens loads here too. A file that does not parse is an error
entry (see Failures).

**Render size.** Every scene renders at the same size, shown as `Render size WxH`
in a scene's settings. Width and Height apply on Enter to all scenes, between 16
and 8192 pixels per side. The first scene takes the size of the first image, or
1280x720 when there is no image, and the size is kept until a new project.

**Shared camera.** All scenes share one orbit camera, so they stay aligned. The
first scene that loads frames the camera around its bounds, unless a project
restored a camera. Over a scene pane, left-drag orbits, right-drag pans the
camera, the mouse wheel dollies, and Q and E roll. Ctrl+wheel zooms the image, and
middle-drag pans the image. Frame scene in a scene's settings frames the camera
again. On panes that show images the image controls work as before, and
double-click fits only those. The wipe line (Wipe in Single screen) and the divider
(two sources in Split) keep priority when the pointer is near them. The 100% and Fit buttons always act on the image. Only scenes
re-render when the camera moves; images do not.

**Per-scene settings.** The gear on a scene row opens these, in addition to the
image controls:

- **Up axis flip** (checkbox). Rotates the scene 180 degrees about the X axis.
- **Antialias** (checkbox). Renders with the antialiased (mip-splatting) opacity
  compensation.
- **SH cap** (combo: 0, 1, 2, 3, Full (3)). The highest spherical-harmonic degree
  used for color. 0 uses the DC color only.
- **Debug view** (combo: Final, Alpha, Depth, TileDensity, SHDegree). The scene shows
  the renderer's debug view instead of the final color. Final is the color.
- **Frame scene** and the render size (see above).

**Up axis and antialias: PLY and SPZ.** SPZ records the antialias setting in its
header, so the window takes it from the file. PLY and `.splat` cannot carry it, so
they load with antialias off, as vanilla 3DGS trains. The up axis is not in any of
the files. COLMAP and INRIA scenes are often stored Y-down, and render upside down
here, so tick Up axis flip for them. One capture saved as PLY and as SPZ can look
different until both have the same Up axis flip and Antialias. The settings panel
says so.

**Depth and alpha maps.** The single view has Depth, Alpha, Depth diff and Alpha
diff, next to A, B, Wipe, Flicker, Diff and FLIP. Depth and Alpha show one source: A, or
B when the Source B radio is picked. Depth diff and Alpha diff compare A and B, so
both must be visible.

- Depth colors the front depth of the source, the depth of the nearest splat that
  contributes to a pixel, through the magma map over the depth range. With Auto range
  on, the range is the union of the valid depths of the shown A and B sources. Turn
  it off to type Min and Max; they must be finite, with Min below Max.
- A valid depth is finite and greater than zero. NaN, infinity, zero and negative
  values are no data. They are drawn as a hatch of diagonal bands, and they never
  set the range.
- Alpha draws the coverage of the source as gray, not sRGB-encoded. NaN is zero.
- Depth diff draws the absolute difference of the two depths over the range. Where
  only one of them is valid the pixel takes the top of the map. Where neither is,
  it is hatched. A source without depth counts as invalid everywhere.
- Alpha diff draws min(1, |coverage A - coverage B| times Gain) through the magma
  map. Gain is the slider of the single view.
- Padding is the part of the padded area that lies outside the chosen source. Depth
  and Alpha hatch it. The diff modes take the top of the map there. An image
  without depth is hatched everywhere in Depth. Outside the padded area is the
  background gray, as in the other modes.

**Probe depth.** Hovering a pane reads the pixel of each source under the pointer.
A scene shows its four depth components, as `Depth: D=... cov=... front=...`: D is
the accumulated depth, cov the coverage, and front the depth of the nearest splat.
Values print with nine significant digits, and NaN and infinity print as `nan` and
`inf`. A padding pixel names no source, so nothing is read for it.

**Memory budget.** Each scene owns a renderer with its own GPU memory. Before the
second, third and fourth scene is loaded, the free device-local memory must cover
about a quarter more than the last scene used, after the memory the loaded scenes
already hold. Otherwise the scene is refused and the status reads `Not enough GPU
memory for another scene: needs about N MiB, M MiB free`. Up to four scenes can be
visible at once. That is the limit of the view, not a promise of memory: a fourth
scene can still be refused.

**Failures.** A missing file is a Missing entry, and a file that does not parse is
an Error entry. Each keeps its message in the media list, and Locate picks a new
file. A scene that fails while it renders is hidden with its message, and the other
sources keep working.

**Exports and report.** The view export, the diff map and the padded pane images
use each scene's rendered color, as the image sources use their pixels.

**Project keys.** A scene entry is saved with `"kind": "scene"`, `flipUpAxis`,
`shCap` (-1 for Full), `antialias` and `debugView` (a key: `final`, `alpha`,
`depth`, `tileDensity` or `shDegree`). The view is saved with `singleView` (`a`,
`b`, `wipe`, `flicker`, `diff`, `flip`, `depth`, `alpha`, `depthDiff` or `alphaDiff`),
`mapSource` (`a` or `b`) and `depthAuto`, and the manual range in `ranges.depth`.
The split divider is `splitDivider` in the view: a number from 0 to 1, the fraction of
the view width where the divider sits (default 0.5; a file without it loads at 0.5).
The FLIP ppd is saved as `flip.ppd` (0 for the default; a value outside 1 to 1000
loads as the default, with a notice). The FLIP mode is `flip.mode` (`ldr` or `hdr`),
and the tone mapper is `flip.toneMapper` (`aces`, `hable` or `reinhard`). The range
overrides are `flip.startExposure`, `flip.stopExposure` and `flip.numExposures`, each
written only when set, so an absent key means automatic. An override that is out of
range is dropped on load, with a notice. Run on change is not saved.
The camera is saved with `pivot`, `orientation` (w, x, y, z), `distance` and the
render size as `width` and `height`. A project written before these keys opens with
the defaults.

### Live scenes

A scene renders again when something it depends on changes: the camera, its settings (SH
cap, up axis, antialias, debug view), the render size, or its file. A scene renders at most
once per frame, however many times the window asks for it in that frame.

- **While the camera moves.** The visible scenes render on every frame of a drag. The
  metrics and the LDR-FLIP are computed at most 4 times a second, and the metrics panel
  says "stale: updating when the camera rests" above the PSNR and MSE lines until they are
  computed. The numbers shown are the last ones computed. The rate is the Live FLIP rate
  (Hz) control next to the FLIP ppd control.
- **At rest.** 150 ms after the last camera move, the scenes render once more and the
  metrics and LDR-FLIP are computed, so the numbers match the view. HDR-FLIP is never
  computed while the camera moves: with auto run on it waits for the camera to rest.
- **View menu: Live scene updates.** Off means the scenes and the metrics update only when
  the camera rests. A drag then shows the last image, and the scenes render once at the end.
- **Render size.** A size change made while the camera moves waits for the rest. A size
  that does not fit the GPU memory budget is refused with `Not enough GPU memory for a
  larger render size: needs about N MiB, M MiB free`, and the old size stays.
- **Scene files.** Once a second the window checks each scene file's write time and size.
  A changed file reloads, keeps the scene's settings and renders at the next frame. A file
  that fails to load keeps the last good render and shows the error in the scene's
  settings. It is tried again only when its write time or size changes. There is no file
  watching thread.
- **Live FLIP rate.** The rate is saved as `flip.liveHz` in project files, from 0.1 to 60
  Hz. See Project files.

Not built: file watching threads, HDR-FLIP per drag frame, camera path playback, Studio
embedding, and FLIP on video.

## Video

A video is added like an image: with Add media, or by dropping the file on the
window. A file is a video when its extension is `.mp4`, `.m4v` or `.webm`
(case-insensitive). Videos are shown in the window only. The command line reads
images, and a video file given to it opens the window.

**Formats.** MP4 and M4V with H.264 video, and WEBM with VP9 video. Audio is
ignored: the audio stream is not read and nothing is played. Other codecs are not
supported.

**VP9 extensions.** A WEBM that Media Foundation cannot open or decode is an Error
entry, with the message `WEBM needs the Windows VP9 Video Extensions (free from
Microsoft Store); the MP4 path is unaffected.`

**Decoding and color.** Media Foundation decodes the stream in software, on the CPU,
to NV12 (no hardware or DXVA decoding). The frame is then uploaded and converted on the
GPU. The conversion is always BT.709 limited range, whatever the stream is tagged:

- Y' = (Y - 16) / 219, Cb' = (Cb - 128) / 224, Cr' = (Cr - 128) / 224
- R' = Y' + 1.5748 Cr', G' = Y' - 0.187324 Cb' - 0.468124 Cr', B' = Y' + 1.8556 Cb'

Each channel is clamped to [0, 1], and alpha is 1. Chroma is nearest: each 2x2 block
of pixels uses one Cb and one Cr, with no interpolation. The output is display-encoded
R'G'B' (no transfer function is applied), so a video is compared and drawn as its
encoded values. A stream tagged BT.601 or full range is still read as BT.709 limited,
and its gear says so.

**Size, rotation and memory.** The frame is cropped to the display aperture, and the
visible size is the one used. Width and height must be even. An odd or zero size is
refused with `video size WxH is not supported (needs even width and height)`. Each side
may be up to 4096 pixels, or the device's maximum image dimension when that is larger.
A bigger video is refused with `video size WxH is above the limit of L pixels on a
side`. The GPU needs an NV12 buffer and an RGBA32F buffer for the video, plus a 25
percent margin. When the free device memory does not cover that, the video is refused
with `Not enough GPU memory for this video: needs about N MiB, M MiB free`. Rotation
metadata is ignored: frames show as stored, and the gear says `Rotation metadata (N deg)
is ignored`.

**Frames and seeking.** Frame i is shown from the time i / fps. The fps is the file's
nominal frame rate (30 when the file gives none). The frame count is the duration times
the rate, rounded, and at least 1. For a variable frame rate the count is an estimate,
and a time shows the nearest earlier frame by its timestamp. Seeking is frame-accurate.
A request that goes back, or more than 30 frames forward, seeks and then decodes forward
from the keyframe to the wanted frame. A short step forward decodes on without a seek.

**Time model.** Each video has a trim (in and out, in seconds), a loop and a time
offset. They follow the time model in `apps/compare/timeline.h`:

- Source time = global time + time offset. A positive offset starts the clip that many
  seconds into its own timeline. A negative offset holds the first frame for that many
  seconds before the clip starts.
- Without loop, source time is held in the trim range: the first frame before the in
  point, the last frame after the out point. A shorter clip holds its last frame until
  the timeline ends.
- With loop, source time wraps inside the trim range.
- The frame shown is the last frame at or before the source time.
- The global length is the largest trim out minus time offset over the visible videos.

Trim in must be at least 0, and trim out must be after trim in and at most the
duration. The trim out field shows the resolved end. To end sets the end of the clip,
which the project saves as -1, so the end follows the file. A time offset is refused
beyond 86400 seconds either way. Time offset (frames) uses the video's own frame rate.
A refused value keeps the old setting, and the status line says why.

**Per-video settings.** The gear of a video shows its size, rate, frame count and
duration on one read-only line, the rotation and color notes above, and the fields
Trim in, Trim out, To end, Loop, Time offset (s) and Time offset (frames). A field
applies when it is left after an edit.

**Transport.** When a video is in the list, a transport bar sits at the bottom of the
view. It has Play or Pause, Step back, Step forward, a seek slider, the time
(`t 1.234 s / 3.000 s`), one frame label per visible video (`A f12/90`, or the source
name when it has no role), a Speed combo (0.25x, 0.5x, 1x, 2x and 4x) and a Loop
checkbox. The state accepts any speed from 0.25 to 4. The Loop checkbox is the global
loop: it wraps the clock at the end of the timeline. Without it, the clock stops there
and playback ends. Play at the end starts again at 0.

The clock moves by real time times the speed. One step is at most 0.25 seconds, so a
stall or a minimized window never makes it jump. Step back and Step forward pause and
move the reference video by exactly one frame: A when A is a video, else B, else the
first visible video. Without a visible video the bar is greyed out, and the Play button
says `show a video` in its tooltip.

The keys are Space, which plays or pauses, and Left and Right, which step one frame back
and forward. They work when a visible video exists and no text field has the keyboard.

**Metrics during playback.** A frame change runs the metrics again for the new frame:
PSNR, MSE and LDR-FLIP, once per presented frame. When the clock passes several frames
between two presented frames, the frames in between are dropped, not queued, so the
numbers match the frame on screen.

**HDR-FLIP during playback.** HDR-FLIP never runs on a frame change. While a video is in
the A or B pair, a frame change drops the HDR result, and the panel and the transport bar
say `HDR-FLIP not run during playback`. Run on change does not start on a frame change
or while the clock runs. Press Run HDR-FLIP while paused to compute it for the frame on
screen. That result stays until the next frame change.

**Export and report.** An export uses the frame on screen at that moment. The view PNG,
the maps and the padded panes of a video show its current frame, and a report written by
the window records the frame index of each video and the clock (see Report). The window
does not write video files.

**Project.** Each video's trim, loop and time offset, and the clock, are saved (see
Project files). Playing is not saved, so a project opens paused.

**Limits.** The sources are limited as for images: eight in the list, and four visible at
a time. Audio is not played, rotation is not applied, and video files are not written.

## What is computed

**PSNR** works on the display-encoded values, as stored in the file. Each
sample is the file's encoded value divided by its maximum (255 for 8-bit, 65535
for 16-bit), so the values lie in [0,1] and the peak is 1. MSE is the mean of the
squared differences over every sample of every channel, and PSNR is
`10 log10(1 / MSE)`. No gamma conversion is applied.

For linear inputs (EXR), PSNR is computed on the display transform: the values
are clamped to [0,1] and encoded with the sRGB transfer function, which is what a
plain monitor shows. This holds with and without `--hdr`.

**FLIP** is NVIDIA's perceptual error metric, run in its low dynamic range (LDR)
form. The samples are converted from sRGB to linear RGB, and the metric compares
the images as they would look on a display at the viewing distance given by
pixels per degree (`--ppd`). FLIP's default is 0.7 m viewing distance, a 3840 px
display 0.7 m wide, which gives about 67 pixels per degree. The output is the
mean FLIP error over the image, in [0,1], where 0 means no visible difference.

**HDR-FLIP** (`--hdr`) is the high dynamic range form of FLIP. The inputs are
taken as linear RGB and are not clamped. For each exposure in the range, both
images are scaled by 2^exposure, tone mapped with the chosen tone mapper, clamped,
and compared with LDR-FLIP. Each pixel keeps the largest error over the exposures.
The mean of that map is the output, and the exposure map holds, for each pixel, the
index of the exposure that gave the largest error, scaled to [0,1].

The exposure range comes from the reference alone when `--start-exposure` and
`--stop-exposure` are not given. It follows FLIP's rule: the start is the exposure
that maps the maximum luminance to the tone mapper's knee, the stop the one that
maps the median luminance there, and the count is `max(2, ceil(stop - start))`.
The HDR line prints the range and the count that were used.

Only linear inputs (EXR) are meaningful for `--hdr`. A display-encoded input
(PNG, JPEG or WebP) is accepted, with a warning on stderr, and its values are
decoded from sRGB to linear first. The `linear MSE` line is the mean squared
difference of those linear values (non-finite pairs are skipped, though the
inputs are rejected earlier when they hold any non-finite sample).

**Error maps** are drawn with FLIP's magma color map: dark for small errors and
bright for large ones. `--flip-map` writes the per-pixel FLIP error (the HDR error
with `--hdr`), `--diff-map` writes the mean absolute difference over the color
channels, and `--exposure-map` writes the HDR exposure map.

## Inputs

- PNG files with 8 or 16 bits per channel, in gray, palette, RGB or RGB with
  alpha. Palette and gray images are expanded to RGB.
- JPEG and WebP files (display-encoded), decoded through WIC and libwebp.
- OpenEXR files (linear, half or float), and raw `.rgba32f` captures (linear,
  with their size from `capture.json`).
- MP4 and M4V files with H.264 video, and WEBM files with VP9 video, are read by the
  window only (see Video).
- Alpha is ignored. A pixel with alpha 0 is compared by its color values.
- Both images must have the same width and height. Sizes that differ are an
  error, and the tool does not resize.
- An image may hold at most `kMaxSamples` (2^31 - 1) samples, counted as decoded
  (width times height times channels). FLIP is stricter: its input may have at most
  `kMaxSamples / 3` pixels, which is the largest count its indexing allows.
- NaN and infinite samples are an error for every comparison.

**Linear input without `--hdr`.** An EXR input is compared after the display
transform (clamped to [0,1], then sRGB). The tool prints a note on stderr:

```text
note: reference.exr is linear (EXR); compared after the sRGB display transform
```

**Display input with `--hdr`.** A PNG, JPEG or WebP input is accepted with a
warning on stderr, and its values are decoded from sRGB as linear:

```text
warning: --hdr with display-encoded input 'reference.png'; values are decoded from sRGB as linear
```

## Report

`--report out.json` (or `out.csv`) writes what was compared and the numbers. It
is written after the output lines. It is deterministic: the same inputs give the
same bytes, and it holds no time stamp. A report that cannot be written is an
error (exit 1).

The JSON has a fixed key order and two-space indentation. Infinite PSNR is `null`,
and so is any value that does not apply (the tone mapper and exposures for LDR,
the linear MSE for LDR, the FLIP mean with `--no-flip`).

```json
{
  "format": "sabercat.compare-report",
  "formatVersion": 1,
  "savedWith": {
    "sabercat": "1.1.0"
  },
  "mode": "hdr",
  "sources": {
    "reference": {
      "path": "reference.exr",
      "width": 512,
      "height": 512,
      "bitDepth": 16,
      "encoding": "linear",
      "hasAlpha": false
    },
    "test": {
      "path": "test.exr",
      "width": 512,
      "height": 512,
      "bitDepth": 16,
      "encoding": "linear",
      "hasAlpha": false
    }
  },
  "overlap": {
    "x": 0,
    "y": 0,
    "w": 512,
    "h": 512
  },
  "anchor": "topLeft",
  "ppd": 67.02,
  "toneMapper": "aces",
  "exposures": {
    "start": -12.5423,
    "stop": 0.9427,
    "count": 14
  },
  "metrics": {
    "psnrDb": 24.5,
    "mse": 0.003548,
    "linearMse": 0.02184,
    "flipMean": 0.283547
  }
}
```

The example numbers are illustrative. The CSV has one header row and one data row,
with these columns in this order:

```text
mode, reference_path, reference_width, reference_height, reference_bit_depth,
reference_encoding, reference_has_alpha, test_path, test_width, test_height,
test_bit_depth, test_encoding, test_has_alpha, overlap_x, overlap_y, overlap_w,
overlap_h, anchor, ppd, tone_mapper, start_exposure, stop_exposure, exposures,
psnr_db, mse, linear_mse, flip_mean
```

Empty cells mark values that do not apply. A path with a comma or a quote is
quoted as in RFC 4180. `psnr_db` is `inf` for identical images. The `exposures`
column is the count.

The overlap is the full image and the anchor is `topLeft`, because the tool
compares images of the same size.

**Video fields.** A report written in the window with a video as A or B adds the frame
and the clock, and nothing else changes. Each video source gets `"frameIndex"` after its
other keys, the frame shown. The top level gets `"timeSeconds"` after `metrics`, the
global clock in seconds. The CSV gets `time_seconds`, `reference_frame_index` and
`test_frame_index` at the end of the header and of the row. A side that is not a video
leaves its frame cell empty. Without a video the output is the same as before, and the
format version stays 1.

## Cost

FLIP runs on the CPU with OpenMP. Its per-pixel loops run in parallel on all
cores, and the result does not depend on the number of threads: the loops write
one output per pixel and combine nothing, and the mean is summed in a fixed
order. The build turns OpenMP on for the FLIP source file only (`/openmp` on MSVC).
Working memory is about ten float images of the input size, and an HDR run holds
more, because it keeps the error and exposure maps. Plan for that on large images.
PSNR and the difference map are much cheaper.

## Decisions and limits

- **JPEG CMYK is refused.** WIC has no CMYK-to-RGB conversion, so a CMYK JPEG is an
  error.
- **EXR is scanline only.** Tiled, multi-part and deep EXR files are refused before
  any pixel is decoded.
- **HDR flags on display-encoded input are allowed.** `--hdr` with a PNG, JPEG or
  WebP input runs, with a warning on stderr, and the values are decoded from sRGB as
  linear (see Inputs).
- **Linear input without `--hdr` uses the display transform.** An EXR input is
  clamped to [0,1] and sRGB-encoded before the comparison, and a note is printed
  (see Inputs).
- **HDR stdout values.** With `--hdr`, the PSNR and MSE lines use the same display
  transform as the LDR run. The `linear MSE` line uses the raw linear values.
- **Animated WebP is refused.** Only still WebP images are read.
- **Maximum image size** is `kMaxSamples` samples, with the FLIP limit, as set out
  under Inputs.
- **Raw `.rgba32f` captures are premultiplied.** Like scene sources, a capture's rgb
  is multiplied by its coverage alpha. The CLI uses the rgb values unchanged, which
  is the capture composited over black, the renderer's default background. The GUI
  composites premultiplied sources over each source's background setting.

## Project files

A comparison can be saved as a `*.sabercat-compare` project: one UTF-8 JSON file
with the sources, camera, view, flip and timeline state. Source paths are stored
relative to the project folder, and a project holds at most 8 sources.

- **Floats are written as the shortest round-trip decimal.** Each float is written
  as the shortest decimal that reads back as the same float, so `0.1` is saved as
  `0.1`, not `0.10000000149`. Saving, loading and saving again gives byte-identical
  files. Double fields (trim, time offset, timeline) are written as they are.
- **One source with role A and one with role B.** A project holds at most one
  source with role A and one with role B. Saving refuses two of either. Loading
  keeps the first holder and clears the others to `none`, with one warning per role
  cleared, such as `role A on source 'c' cleared: source 'a' already has it`.
- **Video keys.** Every source is saved with `trimIn` and `trimOut` (seconds; `trimOut`
  is -1 for the end of the clip), `loop` and `timeOffset` (seconds). A video has
  `"kind": "video"` and its `path`. Image sources keep the same keys at their defaults.
  The clock is saved as `timeline` with `time` (global seconds), `speed` and `loop`. A
  speed outside 0.25 to 4 loads as 1, with a notice, and the time is clamped to the
  duration after the videos load. A trim or offset that does not fit its video loads as
  the whole clip or 0 s, with a notice. A malformed value in the file (a negative trim in,
  or a trim out that is not above the trim in) is a project file error. A project opens
  paused.
- **Live FLIP rate.** `flip.liveHz` is the rate, in Hz, at which the metrics and LDR-FLIP
  update while the camera moves (see Live scenes). It is written only when it is not the
  default 4 Hz, and an absent key or 0 means 4. A value outside 0.1 to 60, or one that is
  not a number, is a project file error.

## Credits

FLIP is by NVIDIA (BSD-3-Clause). The sabercat_compare build fetches a pinned
commit of the FLIP repository at configure time. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for its license text, and
[building.md](building.md) for the other dependencies.
