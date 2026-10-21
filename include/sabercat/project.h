// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sabercat/layer_transform.h"

namespace sabercat
{

class Device;
class GaussianCloud;

// One layer of a Studio project (docs/studio-design.md section 4): a Gaussian
// cloud held in its OWN (asset) space, plus how it is placed in the scene.
// `cloud` carries the payload through its host vectors -- count(), shDegree(),
// antialiased(), the canonical streams and whichever relight streams exist.
struct ProjectLayer
{
    std::shared_ptr<GaussianCloud> cloud;
    // Unique within a project; becomes the archive folder layers/<id>/, so
    // only [A-Za-z0-9_-], 1..64 characters.
    std::string id;
    std::string name; // UTF-8, shown in the UI; any text
    LayerTransform transform;
    bool visible = true;
    bool locked = false;
};

// Everything a .sabercat file holds that the library understands, plus an
// opaque block the application owns.
struct Project
{
    std::vector<ProjectLayer> layers;
    // Application state (camera, render and light settings, ...) as a JSON
    // object in UTF-8 text. Stored verbatim in project.json under "app" and
    // returned as the same JSON value (re-serialised, so whitespace may
    // differ). Must parse as a JSON object; "{}" when the app has none.
    std::string appStateJson = "{}";
    // Filled in by ProjectIO::load(): the Sabercat version that saved it.
    // Ignored by save(), which always writes the running version.
    std::string savedWithVersion;
};

// The .sabercat file: a ZIP of STORED entries --
//   project.json                 format, version, layers, app state, provenance
//   layers/<id>/                 the layer as a sabercat-cloud-v1 directory
//   layers/<id>/relight/         its relight streams (sabercat.relight-v1), if any
// so an external tool can read a layer's cloud-v1 directory as it is.
class ProjectIO
{
  public:
    // Writes `project` to `path` atomically: to a temporary file beside it,
    // renamed over `path` only once complete, so a failed save leaves any
    // existing file untouched. Throws std::runtime_error (naming the layer)
    // on invalid input: a null cloud, a bad or duplicate id, host vectors
    // not sized to count(), a non-unit rotation or a scale <= 0, or app state
    // that is not a JSON object.
    static void save(const Project & project, const std::string & path);

    // Reads a project back. The clouds come back as host data only (not
    // uploaded). Round trip is bit-exact: every host stream, shDegree,
    // antialiased, the transform's floats, flags, ids and names. Throws
    // std::runtime_error naming the problem for a corrupt archive, an
    // unknown format or newer format version, or an invalid layer.
    static Project load(Device & device, const std::string & path);
};

} // namespace sabercat
