// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Headless self-test for the Studio's document layer: project save/load, PLY
// and glTF export, reload, and one forward frame. No window, no ImGui. With no
// scene files it builds a synthetic two-layer project.
#pragma once

namespace sabercat::studio
{

// `sabercat_studio.exe --selftest <outDir> [<scene>...]`. Prints
// "studio selftest OK" and returns 0 on success; prints the failure and returns 1.
// `enableValidation` is resolved by the caller (resolveValidation).
int runSelfTest(int argc, char * argv[], bool enableValidation);

} // namespace sabercat::studio
