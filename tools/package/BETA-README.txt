Sabercat 1.0.0-beta.1 -- beta test build
=========================================

Thank you for testing Sabercat. This package is a beta: it is for trying the
apps on your own scenes and telling us what breaks, what is slow and what is
confusing.

WHAT IS SABERCAT
----------------
Sabercat is a renderer and editor for Gaussian splat scenes (3D scenes made of
many soft, semi-transparent points). The package has two apps:

  sabercat_viewer.exe   open a scene, orbit around it, relight it
  sabercat_studio.exe   layers, an inspector, project save and open, export

REQUIREMENTS
------------
  - Windows 10 or Windows 11, 64-bit
  - A GPU with a driver that supports Vulkan 1.3. If the app does not start,
    update the graphics driver first (NVIDIA, AMD or Intel download page).
  - Nothing to install. No SDK, no runtime download: the Visual C++ runtime is
    included in this folder.

HOW TO RUN
----------
  1. Unzip the archive anywhere you like (for example to your Desktop).
  2. Open the bin\Release folder and double-click sabercat_viewer.exe or
     sabercat_studio.exe.
  3. Drag a scene file onto the window to open it.

The viewer's Browse... button and the Studio's File menu open the Windows file
dialogs. In the viewer, Browse... loads the file you choose and adds it to the
scene list; Load loads whichever entry is selected in the list. Typing a path
in the path box and pressing Enter loads it. You can also drag a scene file
onto the window.

SCENE FORMATS
-------------
No scenes are included. Use your own files:
  .ply   .spz   .splat   .ksplat   .sog   .glb (glTF with KHR_gaussian_splatting)

WHAT TO TRY
-----------
  - Open a few scenes of different sizes and note how long they take to load.
  - In the viewer, click Browse... to pick a scene (a cloud-v1 folder is opened
    by its manifest.json); it loads when you click Open in the dialog.
    Check that the window keeps drawing while the dialog is open.
  - Orbit and zoom around a scene.
  - Relight it: switch the light and the shadow mode, including the ray-traced
    shadows, and see how the result changes.
  - In Studio: add and reorder layers, edit one in the inspector.
  - Save a project, close the app, and open the project again.
  - Export a scene as PLY and as GLB, and open the exported file again.
  - In Studio, choose Help > Register .sabercat files, then double-click a saved project.

SMARTSCREEN WARNING
-------------------
Windows may show "Windows protected your PC" the first time you run an app.
This beta is not code-signed. To run it, click "More info", then "Run anyway".
Only do this for the download you got from the project's GitHub releases page.

LOGS
----
Each run writes a log file to:

  %LOCALAPPDATA%\Sabercat\logs

The newest 10 logs per app are kept. In either app, open Help > About to see
the current log path and to open the logs folder. Command-line options:
  --log <path>   write the log to a file of your choice
  --no-log       do not write a log file

To collect system details for a report, open Help > About and click
"Copy system info". It copies the text to the clipboard.

HOW TO REPORT
-------------
Report problems at https://github.com/KashyapRajpal/Sabercat/issues

  - Use the "Beta feedback" template for general impressions.
  - Use the "Bug report" template for crashes and wrong results. It asks for the
    system info, which you copy from Help > About, and for the newest log file
    from the logs folder above. Please attach both.

KNOWN ISSUES
------------
  - With every layer hidden, Studio shows an empty view and no message.
  - Projects do not save lights.
  - PCSS soft shadows can look lighter than ray-traced shadows on some real
    scenes (the slope bias is tuned conservatively).
