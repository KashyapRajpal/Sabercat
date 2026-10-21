// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include <glm/glm.hpp>

// Internal to the renderer (not installed): the real spherical-harmonic band
// rotation and the 3x3 rotation math behind it, shared by the glTF reader's node
// transforms and layer_transform.cpp so both bake a rotation the same way.
namespace sabercat
{
namespace sh_rotation
{

// Row-major 3x3 matrix: m[row][col].
using Mat3 = std::array<std::array<double, 3>, 3>;

// A band matrix occupies the top-left (2l+1) x (2l+1) block.
constexpr int kMaxBandSize = 7;
using BandMatrix = std::array<std::array<double, kMaxBandSize>, kMaxBandSize>;

// Unit xyzw quaternion -> rotation matrix (the spec's C matrix with unit scale).
Mat3 quaternionToMatrix(double x, double y, double z, double w);

// Shepperd's method; returns a unit xyzw quaternion for a proper rotation.
std::array<double, 4> matrixToQuaternion(const Mat3 & m);

double determinant(const Mat3 & m);

// The matrix T with c_world = T * c_local for one SH band (1..3) under the
// rotation R, so the band is rotated exactly as the shader would see it.
BandMatrix shBandRotation(int band, const Mat3 & rotation);

// Rotates bands 1..degree of one Gaussian's SH coefficients in place. `coefficients`
// is the canonical 16-entry band-ascending layout (DC at index 0, band l occupying
// l^2..(l+1)^2-1); the DC term is rotation invariant and left untouched.
// `bands[l - 1]` is shBandRotation(l, R).
void rotateShCoefficients(glm::vec4 * coefficients, int degree,
                          const std::array<BandMatrix, 3> & bands);

} // namespace sh_rotation
} // namespace sabercat
