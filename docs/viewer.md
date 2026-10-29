# Sabercat Viewer

`sabercat_viewer` is an interactive window for Gaussian splat scenes. It
loads a scene, lets you move the camera, and shows the scene with the
renderer's debug views and lighting.

The viewer is a test and inspection tool. It is not an editor. To combine
layers or export a scene, use [Sabercat Studio](studio.md).

## Start the viewer

```powershell
sabercat_viewer.exe [scene] [flags]
```

| Argument | Effect |
|----------|--------|
| `scene` | A scene file to load at start-up. Only the first non-flag argument is used. |
| `--relight` | Start with the Relight window's lighting turned on. |
| `--borrowed` | The app creates the Vulkan device and lends it to the renderer. Without it, the renderer creates its own device. |
| `--validation` | Turn the Vulkan validation layer on, also in Release builds. |
| `--no-validation` | Turn the validation layer off, also in Debug builds. |
| `--log <path>` | Write the session log to this file instead of the default place. See [Logs and system info](#logs-and-system-info). |
| `--no-log` | Write no session log file. |

With no scene argument, the viewer loads the first scene it finds in the
`assets` folder. See [Scenes](#scenes).

Supported scene inputs:

| Format | Notes |
|--------|-------|
| `.ply` | 3D Gaussian Splatting PLY. |
| `.splat` | Binary splat format. Carries DC color only, so the scene is SH degree 0. |
| `.spz` | Compressed splat format. The file also sets antialiasing. |
| `.ksplat` | Compressed splat format. |
| `.sog` or `meta.json` | SOG scene: a set of WebP images with a `meta.json`. |
| `.gltf`, `.glb` | glTF with the `KHR_gaussian_splatting` extension. |
| `manifest.json` | A cloud-v1 directory, which is a folder with a `manifest.json`. |

## Window

The viewer draws the scene full screen. Three windows sit on top of it.

| Window | Contents |
|--------|----------|
| Sabercat Debug | Scene list, file path, debug views, camera, rendering options and LOD |
| Relight | Lighting, shadows, lights and baked visibility. See [Relight window](#relight-window). |
| About Sabercat | The logo, version, license, third-party notices, the log file, and the **Copy system info** and **Open logs folder** buttons. Opened from **Help > About**. |

The window title shows the version, with the pre-release suffix when the
build has one (for example `Sabercat Viewer 1.0.0-beta.1`). The debug window
shows the frame rate, frame time and the number of Gaussians in the scene.

## Logs and system info

Each run writes a log file to `%LOCALAPPDATA%\Sabercat\logs`, named
`viewer-<yyyymmdd-hhmmss>.log`. The newest 10 viewer logs are kept and older
ones are deleted at start-up. Every line the console shows also goes to the file.
Use `--log <path>` to write to another file, or `--no-log` to write none.

**Help > About > Copy system info** puts a plain-text block on the clipboard:
the version and git commit, the build configuration, the Windows version, the
GPU, its driver and Vulkan version, ray query and float atomics support, the
validation state and the log file path. Paste it into a bug report together
with the log file. **Open logs folder** opens the folder in Explorer.

## Camera

| Input | Action |
|-------|--------|
| Left-drag | Orbit around the pivot |
| Right-drag | Pan |
| Mouse wheel | Zoom |
| `Q` / `E` | Roll the camera left or right |

The camera orbits around a pivot. Full rotation is allowed, including past
straight up or down.

Camera input is ignored while the pointer is over a window, so dragging a
slider does not move the camera.

| Button | Action |
|--------|--------|
| Reset camera | Return to the default camera. |
| Frame scene | Move the camera back until the whole scene fits. It frames the 90th-percentile radius, so a few distant background Gaussians do not shrink the subject. |

The debug window shows the focus radius, the full radius and the camera
distance under **Frame scene**. Hover over the button for a note on how the
framing is chosen.

## Scenes

The scene list shows the scene files found under `assets`, searched
recursively. The viewer looks for `assets` in the current folder and in the
folders above it, and in the usual build output folders.

The list shows `.ply`, `.splat`, `.spz`, `.ksplat`, `.sog`, `.gltf` and `.glb`
files, and the `meta.json` of an unbundled SOG folder. cloud-v1 `manifest.json`
files are not listed.

Ways to load a scene:

- Select a name in the scene list, then click **Load**.
- Click **Browse...** and choose a scene file in the Windows Open dialog. The
  file is added to the top of the scene list, selected, shown in the path box
  and loaded. For a cloud-v1 scene, choose its `manifest.json`.
- Type or paste a path in the path box and press Enter. The path is added
  to the top of the scene list, selected, and loaded. A cloud-v1 scene can
  also be given as its folder. If there is no file or folder at the path, the
  box shows an error and nothing loads.
- Drop a file or folder onto the window. It is added to the list, selected and
  loaded.

The list keeps the paths you added for the session, most recent first, with no
duplicates. **Load** reloads the selected entry, or loads another one you pick
from the list. While the Browse
dialog is open the window keeps drawing, and **Load**, **Browse...** and Enter
in the path box wait until the dialog closes. Canceling the dialog changes
nothing.

If a dialog cannot be shown (the reason is written to the log), type the path
in the path box and press Enter instead.

A failed load shows the error in red under the buttons. Click the message
to clear it. The previous scene stays loaded.

After a load, the camera frames the new scene and the maximum SH degree is
lowered to what the new scene carries.

### Sample scene

Sabercat ships no sample scenes. Open any scene you have in a supported
format. The scene list shows what it finds under `assets\`.

## Debug views

The **Debug view** radio buttons switch what is drawn. Press `V` to step through
the same views.

| View | Shows |
|------|-------|
| Final | The rendered image. This is the normal view. |
| Alpha | Coverage, in grayscale. |
| Depth | Depth of the frontmost contributing Gaussian, in grayscale. |
| TileDensity | A heatmap of how many Gaussians land in each tile. Useful for finding dense areas. |
| SHDegree | A heatmap of the effective SH degree per pixel, from blue (degree 0) to red (degree 3), scaled by coverage. Shows the appearance LOD at work. |

## Rendering options

| Control | Effect |
|---------|--------|
| Flip up axis | Flips the scene's up axis. Use it for scenes authored with Y pointing down, which are common from COLMAP and the INRIA 3DGS convention. They render upside down otherwise. |
| Antialiased (opacity compensation) | Scales each Gaussian's opacity by its true-to-dilated footprint ratio. The file sets the default for `.spz`. `.ply` and `.splat` default to off. Turn it on for a mip-splatting `.ply` that looks oversaturated. |
| Max SH degree (SH0 to SH3) | Limits the view-dependent color. SH0 is flat color, SH3 is full quality. Degrees above the scene's own degree are disabled. |

The **Tile instances** line shows how many tile entries the last frame used,
out of the scene's Gaussian count.

## Level of detail

The LOD controls reduce the work per frame. They change only the viewer's
output. They never change a training pass. All of them are off by default.

| Control | Effect |
|---------|--------|
| Min screen radius (px) | Culls Gaussians whose projected size is below this many pixels. `0` is off. Values up to 1.5 px cull nothing. |
| Drop Gaussians (geometry LOD) | Drops Gaussians by distance and by how many are in a tile. |
| Reduce SH degree (appearance LOD) | Lowers the SH degree with the same distance and density measures. Quality falls before Gaussians disappear. |

When either LOD is on, these sliders appear:

| Slider | Meaning |
|--------|---------|
| LOD near distance | Distance below which nothing is reduced. |
| LOD far distance | Distance at which the maximum reduction applies. |
| Density reference | Tile density scale used by the density measure. |
| Max drop (distance) | Largest fraction of Gaussians the distance rule may drop. |
| Max drop (density) | Largest fraction of Gaussians the density rule may drop. |
| SH falloff (distance) | Largest fraction of SH bands the distance rule may reduce. Shown with appearance LOD. |
| SH falloff (density) | Largest fraction of SH bands the density rule may reduce. Shown with appearance LOD. |
| SH band fade width | How smoothly a band fades out before it switches off. `0` is a hard switch. `1` is continuous. |

The **SHDegree** debug view shows the appearance LOD result.

## Relight window

The Relight window turns on deferred lighting for the scene. It is off
unless you start the viewer with `--relight` or tick the box.

| Control | Effect |
|---------|--------|
| Relight (G-buffer + deferred lighting) | Turns the lighting pass on. |
| Show | What the lighting pass displays: Lit, Normal, Base color, Roughness, Metallic, Ambient occlusion, Visibility (light 0) or Depth. |
| Exposure | Logarithmic exposure scale. |
| Environment | None, Uniform or Sky. |
| Environment intensity | Scale of the environment light. |
| Shadows | Off, PCSS (soft, fast), Ray traced (reference), or Baked SH. Baked SH is available after **Bake visibility** has run for the current scene. |
| Shadow map | Shadow map size for PCSS, 256 to 4096. |
| Rays per axis, Progressive, RT quality | Ray-traced shadow settings. Progressive traces one ray per frame and becomes exact once it converges. RT quality trades accuracy for speed. |
| Shadows status line | Shows whether the shadows are reused or still converging. |

### Baked visibility and teacher samples

These need ray query. When the GPU lacks it, the section says so and the
buttons are disabled.

| Control | Effect |
|---------|--------|
| Bake directions | Number of directions used for each Gaussian's visibility bake, 16 to 1024. |
| Bake visibility | Precomputes visibility for every Gaussian. Baked SH shadows then run in real time. |
| Teacher samples | Generates visibility training samples from the last lit frame, for a trainer. |
| Visibility debug | Off, the error against the ray-traced reference, or the sample locations. |

The bake belongs to the loaded scene. Loading another scene removes it.

### Lights

| Control | Effect |
|---------|--------|
| Type, Color, Intensity | Per light. Types are Directional, Point and Rect. |
| Direction, Position, Half width, Half height | Shown for the matching light type. |
| Casts shadows | Per light. |
| Remove | Removes that light. |
| Add point light | Adds a point light near the scene, up to the maximum number of lights. |
| Reset to scene | Removes the lights and places the default lights for the scene again. |

## Troubleshooting

| Symptom | What to check |
|---------|---------------|
| Black or empty window | The scene may be outside the view. Click **Frame scene** or **Reset camera**. Check the Gaussian count. |
| Scene upside down | Tick **Flip up axis**. |
| Scene looks oversaturated (`.ply`) | Tick **Antialiased (opacity compensation)**. |
| Load error in red | Read the message. It names the reader's error, for example a bad header or an unsupported format. |
| `Validation layer ... not installed` warning | Install the Vulkan SDK, or ignore it. The viewer runs without the layer. |
| Shadow or bake controls are grayed out | The GPU lacks ray query, or you have not baked the scene yet. |
