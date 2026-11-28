// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/edit_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "sabercat/render_config.h"
#include "sh_rotation.h"

namespace sabercat
{

namespace
{

// glm stores matrices column-major; sh_rotation::Mat3 is row-major.
sh_rotation::Mat3 toRowMajor(const glm::dmat3 & m)
{
    sh_rotation::Mat3 r{};
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            r[row][col] = m[col][row];
        }
    }
    return r;
}

glm::dmat3 toGlm(const sh_rotation::Mat3 & r)
{
    glm::dmat3 m(1.0);
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            m[col][row] = r[row][col];
        }
    }
    return m;
}

// The rotation of a placement's quaternion, normalized in double.
glm::dmat3 rotationOf(const glm::quat & q)
{
    const double w = q.w;
    const double x = q.x;
    const double y = q.y;
    const double z = q.z;
    const double length = std::sqrt(w * w + x * x + y * y + z * z);
    if (!(length > 0.0) || !std::isfinite(length))
    {
        throw std::invalid_argument("similarityOf: the placement's rotation is not a quaternion");
    }
    return toGlm(sh_rotation::quaternionToMatrix(x / length, y / length, z / length, w / length));
}

void requirePositive(double scale, const char * where)
{
    if (!(scale > 0.0))
    {
        throw std::invalid_argument(std::string(where) + ": the scale must be positive");
    }
}

} // namespace

glm::dvec3 Similarity::apply(const glm::dvec3 & x) const
{
    return scale * (rotation * x) + translation;
}

Similarity compose(const Similarity & a, const Similarity & b)
{
    Similarity c;
    c.rotation = a.rotation * b.rotation;
    c.translation = a.scale * (a.rotation * b.translation) + a.translation;
    c.scale = a.scale * b.scale;
    return c;
}

Similarity inverse(const Similarity & s)
{
    requirePositive(s.scale, "inverse");
    const glm::dmat3 rt = glm::transpose(s.rotation);
    Similarity inv;
    inv.rotation = rt;
    inv.scale = 1.0 / s.scale;
    inv.translation = -inv.scale * (rt * s.translation);
    return inv;
}

Similarity similarityOf(const LayerTransform & placement)
{
    requirePositive(static_cast<double>(placement.scale), "similarityOf");
    Similarity s;
    s.rotation = rotationOf(placement.rotation);
    s.translation = glm::dvec3(placement.translation);
    s.scale = static_cast<double>(placement.scale);
    return s;
}

Similarity worldEditAboutPivot(const glm::dmat3 & rotation, double scale, const glm::dvec3 & translation,
                               const glm::dvec3 & pivot)
{
    Similarity t;
    t.rotation = rotation;
    t.scale = scale;
    t.translation = pivot + translation - scale * (rotation * pivot);
    return t;
}

Similarity layerSpaceEditD(const Similarity & layer, const Similarity & worldEdit)
{
    requirePositive(layer.scale, "layerSpaceEdit");
    requirePositive(worldEdit.scale, "layerSpaceEdit");
    // L(x) = sL RL x + tL and T(y) = sT RT y + tT, so L^-1(T(L(x))) has
    // rotation RL^T RT RL, scale sT, and translation (1/sL) RL^T (sT RT tL + tT - tL).
    const glm::dmat3 rlT = glm::transpose(layer.rotation);
    Similarity s;
    s.rotation = rlT * worldEdit.rotation * layer.rotation;
    s.scale = worldEdit.scale;
    s.translation = rlT * (worldEdit.scale * (worldEdit.rotation * layer.translation) + worldEdit.translation
                           - layer.translation)
                    / layer.scale;
    return s;
}

LayerTransform layerSpaceEdit(const LayerTransform & layerPlacement, const Similarity & worldEdit)
{
    const Similarity s = layerSpaceEditD(similarityOf(layerPlacement), worldEdit);
    const std::array<double, 4> q = sh_rotation::matrixToQuaternion(toRowMajor(s.rotation)); // xyzw
    const double length = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

    LayerTransform out;
    out.rotation = glm::quat(static_cast<float>(q[3] / length), static_cast<float>(q[0] / length),
                             static_cast<float>(q[1] / length), static_cast<float>(q[2] / length));
    out.translation = glm::vec3(s.translation);
    out.scale = static_cast<float>(s.scale);
    return out;
}

glm::vec4 normalizeRect(const glm::vec2 & a, const glm::vec2 & b)
{
    return glm::vec4(std::min(a.x, b.x), std::min(a.y, b.y), std::max(a.x, b.x), std::max(a.y, b.y));
}

bool boxContains(const glm::dvec3 & p, const glm::dvec3 & center, const glm::dvec3 & halfExtents,
                 const glm::dmat3 & axes)
{
    const glm::dvec3 d = p - center;
    for (int i = 0; i < 3; ++i)
    {
        const double local = glm::dot(axes[i], d);
        // Written as "not less or equal" so a NaN is outside.
        if (!(std::abs(local) <= halfExtents[i]))
        {
            return false;
        }
    }
    return true;
}

bool sphereContains(const glm::dvec3 & p, const glm::dvec3 & center, double radius)
{
    const glm::dvec3 d = p - center;
    return radius >= 0.0 && glm::dot(d, d) <= radius * radius;
}

bool projectToPixels(const glm::dvec3 & p, const CameraUniforms & camera, glm::dvec2 & pixel,
                     double & depth)
{
    const glm::dvec4 local = glm::dmat4(camera.worldToCamera) * glm::dvec4(p, 1.0);
    const double d = -local.z;
    if (!(d > static_cast<double>(kMinDepth)) || !std::isfinite(d))
    {
        return false;
    }
    const double u = static_cast<double>(camera.fx) * local.x / d + static_cast<double>(camera.cx);
    const double v = static_cast<double>(camera.cy) - static_cast<double>(camera.fy) * local.y / d;
    if (!std::isfinite(u) || !std::isfinite(v))
    {
        return false;
    }
    pixel = glm::dvec2(u, v);
    depth = d;
    return true;
}

bool rectContains(const glm::dvec3 & p, const CameraUniforms & camera, const glm::vec4 & rect)
{
    glm::dvec2 pixel(0.0);
    double depth = 0.0;
    if (!projectToPixels(p, camera, pixel, depth))
    {
        return false;
    }
    return pixel.x >= rect.x && pixel.x <= rect.z && pixel.y >= rect.y && pixel.y <= rect.w;
}

} // namespace sabercat
