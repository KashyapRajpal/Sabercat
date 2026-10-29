Sabercat @SABERCAT_VERSION@
=============

Sabercat is a renderer and editor for Gaussian splat scenes (3D scenes made of
many soft, semi-transparent points). The package has two apps:

  sabercat_viewer.exe   open a scene, orbit around it, relight it
  sabercat_studio.exe   layers, an inspector, project save and open, export

REQUIREMENTS
------------
  - Windows 10 or Windows 11, 64-bit
  - A GPU with a driver that supports Vulkan 1.3. If the app does not start,
    update the graphics driver first.
  - Nothing to install. No SDK, no runtime download: the Visual C++ runtime is
    included in this folder.

HOW TO RUN
----------
  1. Unzip the archive anywhere you like.
  2. Open the bin\Release folder and double-click sabercat_viewer.exe or
     sabercat_studio.exe.
  3. Drag a scene file onto the window to open it.

SCENE FORMATS
-------------
No scenes are included. Use your own files:
  .ply   .spz   .splat   .ksplat   .sog (or its meta.json)
  .gltf  .glb (glTF with KHR_gaussian_splatting)
  a cloud-v1 folder (open its manifest.json)
Studio also opens .sabercat project files. Browse... and File > Open show the
Windows file dialogs; you can also drag a file onto the window.

LOGS
----
Each run writes a log file to %LOCALAPPDATA%\Sabercat\logs. Open Help > About
in either app to see the current log path, or to open the logs folder. Use
--log <path> to choose the file, or --no-log to turn logging off.

SUPPORT
-------
Report problems at https://github.com/KashyapRajpal/Sabercat/issues. Include
the output of Help > About > Copy system info and the newest log file.

Copyright 2026 Kashyap Rajpal. Sabercat is licensed under the Apache License,
Version 2.0; see LICENSE, NOTICE and THIRD_PARTY_NOTICES.md in this folder.
