# Sabercat Studio

`sabercat_studio` edits a scene made of layers. Each layer is one
Gaussian cloud with its own position, rotation and scale. You can show,
hide, lock, reorder, move and rename layers. You save the result as a
`.sabercat` project, or export it as one PLY or one glTF (GLB) file.

Studio is in an early stage. It does not edit individual Gaussians yet. See
the known limitations at the end of this page, and
[studio-design.md](studio-design.md) for the plan.

## Start Studio

```powershell
sabercat_studio.exe [project.sabercat | scene file ...]
```

| Argument | Effect |
|----------|--------|
| A `.sabercat` file | Opens the project. |
| Any other file | Adds it as a layer. |
| `--validation` | Turn the Vulkan validation layer on, also in Release builds. |
| `--no-validation` | Turn the validation layer off, also in Debug builds. |
| `--log <path>` | Write the session log to this file instead of the default place. See [Logs and system info](#logs-and-system-info). |
| `--no-log` | Write no session log file. The self-test writes none unless `--log` is given. |
| `--selftest <outDir> [<scene>...]` | Runs the headless self-test and exits. With no scenes it uses a synthetic project. See [Self-test](#self-test). |

You can also drop files onto the window. A `.sabercat` file opens the
project. Anything else is added as a layer.

Scene files Studio accepts: `.ply`, `.splat`, `.spz`, `.ksplat`, `.sog`
(or a SOG `meta.json`), `.gltf`, `.glb`, and a cloud-v1 `manifest.json`.
See [viewer](viewer.md#start-the-viewer) for what each format carries.

With nothing open, Studio starts with an empty project. The viewport is
empty. When every layer is hidden the viewport is empty too (internally Studio renders a single fully transparent Gaussian, because the renderer has no clear-scene call).

## Window

The menu bar holds the File and Help menus. Five windows sit below it.

| Window | Contents |
|--------|----------|
| Layers | The layer list and the buttons that change it |
| Inspector | The selected layer's transform and details |
| Viewport | Frame rate, scene rebuild time, debug view, SH degree and **Frame all** |
| Relight | The same lighting panel as the viewer. See [Relight](#relight). |
| About Sabercat | The logo, version, licence, third-party notices, the log file, and the **Copy system info** and **Open logs folder** buttons |

The window title shows the program name, the version (with the pre-release
suffix when the build has one, for example `Sabercat Studio 1.0.0-beta.1`),
the project file name, and a `*` when there are unsaved changes.

## Logs and system info

Each run writes a log file to `%LOCALAPPDATA%\Sabercat\logs`, named
`studio-<yyyymmdd-hhmmss>.log`. The newest 10 Studio logs are kept and older
ones are deleted at start-up. Use `--log <path>` to write to another file, or
`--no-log` to write none.

**Help > About > Copy system info** puts a plain-text block on the clipboard:
the version and git commit, the build configuration, the Windows version, the
GPU, its driver and Vulkan version, ray query and float atomics support, the
validation state and the log file path. Paste it into a bug report together
with the log file. **Open logs folder** opens the folder in Explorer.

## Register .sabercat files

Windows can open a `.sabercat` project in Studio when you double-click it.
Studio sets this up for your own Windows account only, so no administrator
rights are needed.

- **Help > Register .sabercat files** points `.sabercat` files at this copy of
  Studio. Project files then show the Sabercat icon, and double-clicking one
  opens it in this copy. If another copy of Studio (or another program) already
  has the association, Studio asks first.
- **Help > Unregister .sabercat files** removes the association. The item is
  offered only for this copy's own registration. Another copy's association is
  never removed.

The Help menu says which copy the association points at now. The result of
each action appears beside the menu bar.

The association is stored under `HKEY_CURRENT_USER\Software\Classes`, in the
`.sabercat` key and the `Sabercat.Project` key. It holds the full path of the
copy that registered it. If you move or delete the Studio folder, the
association still points at the old path. Run Studio from its new location and
choose **Help > Register .sabercat files** again; that replaces the old path.

Windows also keeps a per-user choice for each file type, made with **Open
with**. If double-clicking a project opens a different program, right-click the
file, choose **Open with > Choose another app**, pick Studio, and tick **Always**.

## File menu

| Item | Shortcut | Effect |
|------|----------|--------|
| New | | Starts an empty project. Asks first if there are unsaved changes. |
| Open... | `Ctrl+O` | Picks a project file in the Windows Open dialog. Asks first if there are unsaved changes. |
| Save | `Ctrl+S` | Saves to the current file. With no file yet, it opens **Save As**. |
| Save As... | `Ctrl+Shift+S` | Saves to a new `.sabercat` file, chosen in the Windows Save dialog. The name starts as the project's file name, or `untitled.sabercat`. |
| Add Layer... | | Picks a scene file in the Windows Open dialog and adds it as a new layer. A cloud-v1 scene is added by its `manifest.json`. |
| Export PLY... | | Picks where to write the visible layers as one PLY. The name starts as `<project>.ply`. |
| Export GLB... | | Picks where to write the visible layers as one GLB. The name starts as `<project>.glb`. |
| Type a Path | | Submenu: Open project, Save project as, Add layer, Export PLY and Export GLB, each in the path box (see below). |
| Quit | | Closes Studio. Asks first if there are unsaved changes. |

Closing the window also asks first if there are unsaved changes.

Open, Save As, Add Layer and the exports use the native Windows dialogs. Cancel
in a dialog does nothing. Each dialog remembers the last folder used for its
own action. Open asks about unsaved changes after you choose the file, and
before the current project is replaced.

The dialogs run in the background, so Studio keeps drawing while one is open.
While a dialog is open, the File items that open or save files, the Add button
and the Remove button are greyed out, and so are the OK and Browse buttons of
the path box. Drops onto the window wait until the dialog closes.

If a native dialog cannot be shown (the reason is written to the log), use
**File > Type a Path**. It opens the path box: type or paste the path, or click
**Browse...** to fill it from a native dialog, then press **OK** or Enter.
**Cancel** closes the box without doing anything. The box says what it expects,
for example `.sabercat` for a project and the supported scene formats for a
layer.

Errors (for example, a file that will not load, or an export with no visible
layers) appear in an Error box. Click **OK** to close it.

## Layers

The Layers table has one row per layer.

| Column | Meaning |
|--------|---------|
| Vis | Shows or hides the layer. Hidden layers are left out of the viewport and of exports. |
| Lock | Locks the layer. A locked layer cannot be removed, renamed, transformed or moved, and a neighbour cannot swap places with it. Its visibility can still change. |
| Name | The layer name. Click to select. Names come from the file name when a layer is added. |
| Gaussians | Number of Gaussians in the layer. |
| SH | The SH degree the layer carries. |
| Relight | Letters for the relight data the layer carries: `N` normals, `B` base colour and roughness, `M` metallic, `V` visibility. `-` means none. |

Buttons below the table act on the selected layer:

| Button | Effect |
|--------|--------|
| Add... | Opens the Windows Open dialog to add a scene file (or a cloud-v1 `manifest.json`) as a layer. |
| Remove | Removes the selected layer. |
| Rename... | Changes the layer name. |
| Up / Down | Moves the layer in the list. The list order is the order the layers are merged in. |

The footer shows the number of layers and how many are visible.

## Inspector

The Inspector shows the selected layer.

| Field | Meaning |
|-------|---------|
| Name and id | The layer name, and its id inside the project. The id is `(locked)` when the layer is locked. |
| Translation | Position in the scene, in scene units. |
| Rotation (deg) | Rotation as Euler angles, in degrees. The stored value is a quaternion. |
| Scale | Uniform scale. Must be above zero. |
| Reset transform | Sets translation, rotation and scale back to the identity. |
| Gaussians, SH degree, Antialiased | The layer's data. |
| Bounds | The layer's bounding box in its own space, before the transform. |

Transform fields are disabled while the layer is locked.

## Viewport

| Control | Effect |
|---------|--------|
| Frame all | Moves the camera until the whole visible scene fits. |
| Debug view | Same views as the viewer: Final, Alpha, Depth, TileDensity, SHDegree. |
| Max SH degree | Limits the view-dependent colour, 0 to 3. A layer added later may carry more degrees. |

Camera controls are the same as the viewer: left-drag to orbit, right-drag to
pan, scroll to zoom, `Q` and `E` to roll. The camera position is saved in the
project, but moving it does not mark the project as changed.

The Viewport also shows the number of Gaussians on screen and the time the
last scene rebuild took.

## Scene and rebuild

The viewport shows all visible layers together. When you add, remove, hide,
move or rescale a layer, Studio rebuilds the merged scene once per frame.

The merge runs on the CPU. For large layers the rebuild can take a noticeable
time. The Viewport shows the last rebuild time so you can see the cost.

## Relight

The Relight window is the same panel as the viewer's. Its controls are
described in [viewer.md](viewer.md#relight-window).

Relight data that a layer carries (normals, base colour, metallic and
visibility) is listed in the Layers table. The Relight window's scalar
settings (exposure, environment, shadow mode and similar) are saved in the
project. Its lights and visibility bakes are not. When you open a project,
the Relight window places its default lights again.

## Projects

A `.sabercat` file holds the whole project. It is a ZIP archive with a
`project.json` and one folder per layer. Each layer's data is stored inside
the file. The project does not refer to the original scene files, so the
project still opens if those files move.

A project file stores:

- Each layer's Gaussians, its relight data, and its transform, name, id,
  visibility and lock state.
- The viewer state: camera, SH degree, debug view, and the Relight window's
  scalar settings.

A project file does not store the Relight lights or the visibility bake. See
[formats.md](formats.md) for the layout.

Saving writes to a temporary file first, then renames it over the project
file. A failed save leaves the old file in place.

## Export

| Export | Contents |
|--------|----------|
| PLY | The visible layers merged into one binary PLY, in world space. |
| GLB | One glTF node per visible layer, with its name and transform. The Gaussians use the `KHR_gaussian_splatting` extension. |

Both exports fail with the error *No visible layers to export* when no layer
is visible. Nothing is written in that case.

Exports use the layers' current state. Hidden layers are not included.
Export format details are in [formats.md](formats.md).

## Self-test

```powershell
sabercat_studio.exe --selftest <outDir> [scene ...]
```

The self-test runs without a window. It:

1. Builds a project from the scenes and saves it. With no scenes, it builds a
   synthetic two-layer project instead (different SH degrees, one layer moved
   and carrying relight streams), so `ctest -L gpu` needs no asset files.
2. Opens the project again and checks that it matches the original exactly.
3. Exports a PLY and a GLB into `outDir`.
4. Reads the GLB back and compares it with the merged layers. With no scenes,
   the PLY is compared the same way, within the tolerances of the PLY writer
   test, and the export is refused when no layer is visible.
5. Renders one frame and checks that some pixels are covered.

It prints `studio selftest OK` and exits with 0 on success. On failure it
prints the reason and exits with 1. Its files are written to `outDir`.

The self-test needs a GPU with Vulkan 1.3.

## Known limitations

- **Layers are merged on the host.** Every scene rebuild merges the visible
  layers on the CPU. Large or many layers make rebuilds slow.
- **Lights and bakes are not saved in projects.** Reopening a project places
  the default lights again, and any visibility bake must be made again.
  Other Relight settings are saved.
- **An empty project shows nothing.** With no layers, or with every layer
  hidden, the viewport is empty (internally Studio renders a single fully transparent Gaussian, because the renderer has no clear-scene call). Exports are refused.
- **No undo.** Removing a layer, or changing a transform, cannot be undone.
- **No Gaussian editing.** Layers can be changed as a whole. Editing single
  Gaussians is planned.

## Troubleshooting

| Symptom | What to check |
|---------|---------------|
| Blank viewport | Check that a layer is visible, then click **Frame all**. |
| A layer does not appear after drop | Check the file type. Only the formats listed above are loaded. |
| *Could not build the scene* error | The renderer failed to merge or upload the layers. The message gives the reason. |
| Discard prompt on Open, New or Quit | You have unsaved changes. Yes discards the changes; No cancels. Save first if you want to keep them. |
| Transform fields greyed out | The layer is locked. Unlock it to change its transform. |
| Validation warning at start-up | The validation layer is not installed. The program runs without it. |
