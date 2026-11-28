# Sabercat Studio

`sabercat_studio` edits a scene made of layers. Each layer is one
Gaussian cloud with its own position, rotation and scale. You can show,
hide, lock, reorder, move and rename layers. You can select, transform,
delete and duplicate the Gaussians of the active layer, and undo and redo
those edits. You save the result as a `.sabercat` project, or export it as
one PLY, one glTF (GLB) or one SPZ file.

Studio is in an early stage. Editing works on one layer at a time, and some
selection and transform tools are not in yet. See [Editing](#editing) and the
known limitations at the end of this page, and
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
empty. When every layer is hidden the viewport is empty too (Studio clears the renderer's scene).

## Window

The menu bar holds the File, Edit, Window and Help menus. The windows below it
are listed in the table. The Select, Transform and History windows are shown or
hidden from the Window menu.

| Window | Contents |
|--------|----------|
| Layers | The layer list and the buttons that change it |
| Inspector | The selected layer's transform and details |
| Viewport | Frame rate, scene rebuild time, debug view, SH degree and **Frame all** |
| Relight | The same lighting panel as the viewer. See [Relight](#relight). |
| Select | The selection tools for the active layer. See [Select window](#select-window). |
| Transform | Moves, turns and scales the selection of the active layer. See [Transform window](#transform-window). |
| History | The undo steps, with a jump to any of them. See [History window](#history-window). |
| About Sabercat | The logo, version, license, third-party notices, the log file, and the **Copy system info** and **Open logs folder** buttons |

The window title shows the program name, the version (with the pre-release
suffix when the build has one, for example `Sabercat Studio 1.0.0-beta.1`),
the project file name, and a `*` when there are unsaved changes.

## Logs and system info

Each run writes a log file to `%LOCALAPPDATA%\Sabercat\logs`, named
`studio-<yyyymmdd-hhmmss>.log`. The newest 10 Studio logs are kept and older
ones are deleted at start-up. Use `--log <path>` to write to another file, or
`--no-log` to write none.

**Help > About Sabercat > Copy system info** puts a plain-text block on the clipboard:
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
| Export SPZ... | | Picks where to write the visible layers as one SPZ, merged. The name starts as `<project>.spz`. |
| Type a Path | | Submenu: Open project, Save project as, Add layer, Export PLY, Export GLB and Export SPZ, each in the path box (see below). |
| Quit | | Closes Studio. Asks first if there are unsaved changes. |

Closing the window also asks first if there are unsaved changes.

Open, Save As, Add Layer and the exports use the native Windows dialogs. Cancel
in a dialog does nothing. Each dialog remembers the last folder used for its
own action. Open asks about unsaved changes after you choose the file, and
before the current project is replaced.

The dialogs run in the background, so Studio keeps drawing while one is open.
While a dialog is open, the File items that open or save files, the Add button
and the Remove button are grayed out, and so are the OK and Browse buttons of
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
| Lock | Locks the layer. A locked layer cannot be removed, renamed, transformed or moved, and a neighbor cannot swap places with it. Its Gaussians cannot be selected or edited. Its visibility can still change. |
| Name | The layer name. Click to select. Names come from the file name when a layer is added. |
| Gaussians | Number of Gaussians in the layer. |
| SH | The SH degree the layer carries. |
| Relight | Letters for the relight data the layer carries: `N` normals, `B` base color and roughness, `M` metallic, `V` visibility. `-` means none. |

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

Transform fields are disabled while the layer is locked. Each finished drag of
the Translation, Rotation or Scale fields is one undo step, and so is **Reset
transform**.

## Viewport

| Control | Effect |
|---------|--------|
| Frame all | Moves the camera until the whole visible scene fits. |
| Debug view | Same views as the viewer: Final, Alpha, Depth, TileDensity, SHDegree. |
| Max SH degree | Limits the view-dependent color, 0 to 3. A layer added later may carry more degrees. |

Camera controls are the same as the viewer: left-drag to orbit, right-drag to
pan, scroll to zoom, `Q` and `E` to roll. The camera position is saved in the
project, but moving it does not mark the project as changed. While the
Rectangle selection tool is armed, left-drag selects instead of orbiting, and
Alt with left-drag still orbits.

The Viewport also shows the number of Gaussians on screen and the time the
last scene rebuild took.

## Editing

Editing works on the Gaussians of the active layer, the layer selected in the
Layers table. The layer must be visible and unlocked. The Edit menu, the Select
window and the Transform window all act on it. Selecting is not an edit: it
does not mark the project as changed. Transform, delete and duplicate do.

An edited layer's data lives on the GPU. Its CPU copy catches up at the next
save or export, or when you click **Frame all**.

### Edit menu

| Item | Shortcut | Effect |
|------|----------|--------|
| Undo | `Ctrl+Z` | Undoes the last step. The item names the step, for example *Undo Delete 120*. Grayed out when there is nothing to undo. |
| Redo | `Ctrl+Y` or `Ctrl+Shift+Z` | Redoes the step after the last undone one. The item names it in the same way. |
| Select All | `Ctrl+A` | Selects every Gaussian of the active layer. |
| Select None | `Ctrl+Shift+A` | Clears the selection. |
| Invert Selection | `Ctrl+I` | Selects the Gaussians that are not selected, and clears the ones that are. |
| Delete Selected | `Delete` | Deletes the selected Gaussians. |
| Duplicate | `Ctrl+D` | Appends a copy of the selected Gaussians. The copies become the selection. |

Undo and Redo do not depend on the active layer. Select All, Select None,
Invert Selection, Delete Selected and Duplicate are grayed out unless the
active layer can be edited. Delete Selected and Duplicate also need a
selection. The shortcuts do nothing while a text field has keyboard focus,
while a dialog is open, or while a popup menu is open.

### Select window

The Select window holds the tools that choose Gaussians. It starts open.

- **Tool**: **Rectangle**, **Box** or **Sphere**.
- **Op**: **Replace**, **Add** or **Subtract**. Replace makes the new shape the
  whole selection. Add adds its Gaussians to the selection. Subtract removes
  them from it.

**Rectangle** selects by dragging in the viewport with the left button. The
Gaussians whose centers project into the rectangle are selected, through all
depths, so a Gaussian hidden behind another one is selected too. While
dragging, Shift adds and Ctrl subtracts, whatever the Op is. Escape cancels the
drag, and a click without a drag selects nothing. The rectangle is projected
with the last frame that was drawn, so a rectangle selection made before any
frame has been drawn fails.

**Box** takes a **Center** and **Half extents** in scene units, along the world
axes. **Fit to layer bounds** sets both from the layer's bounds. **Apply**
selects the box. **Sphere** takes a **Center** and a **Radius**, and **Apply**
selects the sphere. While the Select window is open, the box or sphere is drawn
as a wireframe in the viewport.

Below the tools, **Selected N of M** gives the number of selected Gaussians of
the active layer, out of M. The viewport also shows *Selected N* in its top
left corner. **Clear** selects none, **All** selects every Gaussian, and
**Invert** inverts the selection.

**Highlight selection** tints the selected Gaussians. **Tint** sets the color,
and its alpha is the strength. The default is orange at strength 0.5. The
highlight is drawn in the unlit image only, so it shows with Relight off. The
window says so while Relight is on. The highlight is not drawn after the GPU
merge has failed for the session (see [Scene and rebuild](#scene-and-rebuild)).

### Transform window

The Transform window moves, turns and scales the selection of the active
layer. It needs a selection. Without one, it says so.

- **Live preview** shows the transform on the layer as you change the fields.
  Without it, the transform appears only when you click **Apply**.
- **Pivot** is the point that rotation and scale turn about: **Selection
  centroid** (the default), **Layer origin** or **World origin**.
- **Translate** moves the selection, in scene units.
- **Rotate (deg)** turns the selection about the pivot, as XYZ Euler angles in
  degrees.
- **Scale** scales the selection about the pivot. It is one value for all axes
  and must be above zero.

The rotation and scale are applied about the pivot, then the translation is
added. The transform is rigid with a uniform scale. Non-uniform scale and
shear are not in 1.1.

**Apply** commits the transform as one undo step and sets the fields back to
their identity values. It is grayed out while the fields are at identity.
**Reset fields** drops the live preview and sets the fields back to identity.
Closing the window, or changing the selection or the active layer, also drops
the live preview.

### Delete and duplicate

- **Delete Selected** removes the selected Gaussians of the active layer at
  once. The Gaussians that remain keep their order, and the layer's data is
  compacted.
- Deleting every Gaussian of a layer is refused, with the message *Every
  Gaussian is selected: remove the layer instead*. Use **Remove** in the Layers
  window to remove the layer.
- **Duplicate** appends copies of the selected Gaussians after the layer's last
  Gaussian. The copies become the selection, so the Transform window moves them
  at once.

Either command is refused, with the reason beside the menu bar, when nothing is
selected or the active layer cannot be edited.

### Undo and redo

Each edit is one step in an undo history. The Edit menu names the step that
Undo will reverse and the step that Redo will apply.

Undoable:

- Selection changes: every selection tool, Select All, Select None and Invert.
- Transform (**Apply** in the Transform window), Delete and Duplicate.
- Adding, removing and moving (Up and Down) a layer.
- Renaming a layer, and showing or hiding it.
- A layer's transform changed in the Inspector: one step per finished drag of
  the Translation, Rotation or Scale fields, and one for **Reset transform**.

Not undoable:

- Locking and unlocking a layer. A lock is a guard you set, so undo never
  changes it. The Lock toggle's tooltip says so.

Behavior:

- A new edit after an undo removes the steps that were undone.
- Undoing a transform puts back the stored values word for word. Redo applies
  the same edit again.
- Undo and Redo first drop a live transform preview.
- The history holds up to 1 GiB of host memory. When it is full, the oldest
  steps are dropped first, and the session log records each drop.
- Opening a project and starting a new project both clear the history.

### History window

The History window lists every undo step in order, oldest first. It starts
open, and the Window menu shows or hides it. The first row, *(initial state)*,
is the state before any step. Each step has its label and its size in KB.

- Applied steps show in normal text. Undone steps show greyed out. The row at
  the cursor is selected.
- Clicking a row moves the cursor to that point, undoing or redoing as needed.
  A live transform preview is dropped first.
- The footer gives the number of entries and the cursor position, the number of
  evicted steps once there are any, and the memory the history holds against
  its 1 GiB budget, as a bar.
- **Undo** and **Redo** do what the Edit menu items do.
- **Clear history** forgets every undo step. Edits already made stay as they
  are. The tooltip says so.

## Scene and rebuild

The viewport shows all visible layers together. When you add, remove, hide,
move, rescale or edit a layer, Studio rebuilds the merged scene once per frame.

The visible layers are merged on the GPU. Each layer's GPU copy is written into
one merged scene, with the layer's transform applied by a kernel, so the merge
does not read the layers back to the CPU. A layer is copied to the GPU the first
time it is needed, at the first rebuild or the first edit. The Viewport shows
the last rebuild time so you can see the cost. Measured on the
development machine (RTX 5070 Ti, Release build) with the 1.16 million Gaussian
bonsai scene: the GPU merge of one layer with a non-identity transform takes
about 1.2 ms, against about 210 to 255 ms for the host merge and upload it
replaces. `examples/edit_test` prints these timings when
`SABERCAT_EDIT_TEST_SCENE` names a scene file.

If the GPU merge fails, Studio logs one warning and uses the CPU merge for the
rest of the session. The CPU merge also runs for the PLY and SPZ exports. The
GLB export writes one glTF node per layer and does not merge.

## Relight

The Relight window is the same panel as the viewer's. Its controls are
described in [viewer.md](viewer.md#relight-window).

Relight data that a layer carries (normals, base color, metallic and
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

A project file does not store the Relight lights or the visibility bake. It
does not store selections or the undo history either. See
[formats.md](formats.md) for the layout.

Saving first drops any live transform preview, so the file holds only committed
values. It then copies every edited layer back to the CPU and writes to a
temporary file, which is renamed over the project file. A failed save leaves the
old file in place.

## Export

| Export | Contents |
|--------|----------|
| PLY | The visible layers merged into one binary PLY, in world space. |
| GLB | One glTF node per visible layer, with its name and transform. The Gaussians use the `KHR_gaussian_splatting` extension. |
| SPZ | The visible layers merged into one SPZ, in world space. The file is quantized, so positions are kept to about 1/8192. |

All three exports fail with the error *No visible layers to export* when no layer
is visible. Nothing is written in that case. SPZ also refuses a scene whose positions
lie outside about +-2048.

Exports use the layers' current state. Hidden layers are not included. Like a
save, an export first drops any live transform preview and copies the edited
layers back to the CPU. Export format details are in [formats.md](formats.md).

## Self-test

```powershell
sabercat_studio.exe --selftest <outDir> [scene ...]
```

The self-test runs without a window. It:

1. Builds a project from the scenes and saves it. With no scenes, it builds a
   synthetic two-layer project instead (different SH degrees, one layer moved
   and carrying relight streams), so `ctest -L gpu` needs no asset files.
2. Opens the project again and checks that it matches the original exactly.
3. Exports a PLY, a GLB and an SPZ into `outDir`.
4. Reads the GLB back and compares it with the merged layers. Reads the SPZ back
   and compares its positions with the merged layers, within 1.3e-4. With no scenes,
   the PLY is compared the same way, within the tolerances of the PLY writer
   test, and the export is refused when no layer is visible (PLY, GLB and SPZ).
5. Renders one frame and checks that some pixels are covered.

The self-test also exercises the editing model. It checks that the GPU layer
merge matches the host merge, and that the host fallback matches `mergeLayers`.
It runs select, transform, duplicate and delete on a layer, with undo and redo,
and checks each state bit for bit. It also saves and reopens the project after
edits, and checks that the reopened layers edit, undo and redo the same way.

It prints `studio selftest OK` and exits with 0 on success. On failure it
prints the reason and exits with 1. Its files are written to `outDir`.

The self-test needs a GPU with Vulkan 1.3.

## Known limitations

- **Layers are merged on the GPU.** Each scene rebuild merges the visible
  layers in a GPU kernel. The CPU merge remains for the PLY and SPZ exports and
  as the fallback when the GPU merge fails.
- **Lights and bakes are not saved in projects.** Reopening a project places
  the default lights again, and any visibility bake must be made again.
  Other Relight settings are saved.
- **An empty project shows nothing.** With no layers, or with every layer
  hidden, the viewport is empty (Studio clears the renderer's scene). Exports are refused.
- **Editing works on the active layer only.** Selection, transform, delete and
  duplicate apply to the active layer, which must be visible and unlocked. The
  other layers cannot be edited in place.
- **Later tools are not in 1.1.** Lasso, polygon and brush selection, an
  outline around the selection, a gizmo, and a general affine transform
  (non-uniform scale and shear) are planned. Selection and transform use the
  tools described above.
- **Editing needs about twice the GPU memory.** An edited layer is held twice on
  the GPU: in its own buffers and in the merged scene. One copy of the core
  streams of 1.16 million Gaussians is about 0.36 GB (1.16 million times 308
  bytes), before the relight streams. The figure follows from the stream sizes;
  it was not measured.
- **Bounds stay conservative after an edit.** The framing box is built from
  each layer's conservative bounds, which grow with every edit and never shrink,
  until you click **Frame all**. The same happens while a drag is in progress.
  **Frame all** copies the edited layers back to the CPU and sets the exact
  bounds. A save also copies them back, and the bounds are exact again at the
  next scene rebuild.
- **A large live preview is slower.** The live preview keeps a copy of the
  selected Gaussians' values on the host and restores it at each change. A
  selection of more than 250,000 Gaussians is still exact, but it previews more
  slowly.
- **The highlight is not shown in the lit image.** It tints the unlit color
  image, so turn Relight off to see it.
- **Undo is limited by its budget.** The history holds up to 1 GiB of host
  memory. The oldest steps are dropped first, so a long session loses its oldest
  undo steps.
## Troubleshooting

| Symptom | What to check |
|---------|---------------|
| Blank viewport | Check that a layer is visible, then click **Frame all**. |
| A layer does not appear after drop | Check the file type. Only the formats listed above are loaded. |
| *Could not build the scene* error | The renderer failed to merge or upload the layers. The message gives the reason. |
| Discard prompt on Open, New or Quit | You have unsaved changes. Yes discards the changes; No cancels. Save first if you want to keep them. |
| Transform fields grayed out | The layer is locked. Unlock it to change its transform. |
| Edit items grayed out | The active layer is hidden, locked or has no Gaussians. Delete Selected and Duplicate are also grayed out when nothing is selected. The Select window says why the layer cannot be edited. |
| Rectangle drag orbits the camera instead of selecting | Only Alt with left-drag orbits. Without Alt, the drag selects while the Select window is open, the Rectangle tool is chosen and the active layer can be edited. |
| Transform window says *Select Gaussians first* | Select the Gaussians to move with the Select window. |
| Validation warning at start-up | The validation layer is not installed. The program runs without it. |
