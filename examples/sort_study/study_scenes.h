// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <string>

#include "sabercat/camera.h"

namespace sabercat
{

class Device;
class GaussianCloud;

} // namespace sabercat

namespace sabercat::study
{

// One scene of the study: the cloud plus the framing used by the orbit cameras.
struct StudyScene
{
    std::string name; // file name only (never a directory), or "synthetic"
    std::shared_ptr<GaussianCloud> cloud;
    glm::vec3 center{0.0f};
    float radius = 0.0f;
    float elevationDeg = 15.0f;
    // Informational. makeStudyCamera() sets the focal length to 0.9 * height,
    // which is about 58 degrees vertical at every height.
    float fovYDeg = 58.0f;
};

// Deterministic two-sheet plus ring scene, built in code. Degree-0 SH only.
StudyScene makeSyntheticScene(Device & device);

// Loads a scene file with loadScene(), uploads it and frames it on the robust
// focus sphere. Errors name the file only.
StudyScene loadStudyScene(Device & device, const std::string & path);

// Viewer camera on an orbit around the scene center at `azimuthDeg`, at the
// scene's elevation, 1.8 radii away.
CameraUniforms makeStudyCamera(const StudyScene & scene, uint32_t width, uint32_t height, float azimuthDeg);

} // namespace sabercat::study
