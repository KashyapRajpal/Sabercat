// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/lighting.h"

#include <cmath>

namespace sabercat
{

void shBasis16(const glm::vec3 & dir, float out[16])
{
    const float x = dir.x, y = dir.y, z = dir.z;
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    out[0] = kShC0;
    out[1] = -kShC1 * y;
    out[2] = kShC1 * z;
    out[3] = -kShC1 * x;
    out[4] = kShC2_0 * xy;
    out[5] = kShC2_1 * yz;
    out[6] = kShC2_2 * (2.0f * zz - xx - yy);
    out[7] = kShC2_3 * xz;
    out[8] = kShC2_4 * (xx - yy);
    out[9] = kShC3_0 * y * (3.0f * xx - yy);
    out[10] = kShC3_1 * xy * z;
    out[11] = kShC3_2 * y * (4.0f * zz - xx - yy);
    out[12] = kShC3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
    out[13] = kShC3_4 * x * (4.0f * zz - xx - yy);
    out[14] = kShC3_5 * z * (xx - yy);
    out[15] = kShC3_6 * x * (xx - 3.0f * yy);
}

EnvironmentLight EnvironmentLight::uniform(const glm::vec3 & radiance)
{
    // A constant function has only a DC term: L = c00 * Y00 with Y00 = kShC0.
    EnvironmentLight env;
    env.sh[0] = radiance / kShC0;
    return env;
}

EnvironmentLight EnvironmentLight::sky(const glm::vec3 & zenith, const glm::vec3 & horizon,
                                       const glm::vec3 & ground, const glm::vec3 & up)
{
    // Numerical projection: a fine latitude/longitude quadrature of the
    // radiance against each basis function (exact enough for a smooth sky;
    // done once on the host).
    EnvironmentLight env;
    double sum[9][3] = {}; // double: ~1e5 increments of ~1e-5 would drift in float
    const glm::vec3 u = glm::normalize(up);
    // Bins are uniform in cos(theta) (equal solid angle per row), so the
    // basis's polynomial terms in cos(theta) integrate to the midpoint rule's
    // O((1/kLatitude)^2) accuracy; uniform in theta it was ~1e-4 off even for a
    // constant sky.
    constexpr int kLatitude = 1024;
    constexpr int kPhi = 256;
    const float dU = 2.0f / kLatitude;
    const float dPhi = 2.0f * 3.14159265358979f / kPhi;
    for (int i = 0; i < kLatitude; ++i)
    {
        const float cosTheta = -1.0f + (static_cast<float>(i) + 0.5f) * dU;
        const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
        for (int j = 0; j < kPhi; ++j)
        {
            const float phi = (static_cast<float>(j) + 0.5f) * dPhi;
            const glm::vec3 dir(sinTheta * std::cos(phi), cosTheta, sinTheta * std::sin(phi));
            // Elevation relative to `up`, independent of the quadrature frame.
            const float h = glm::dot(dir, u);
            const glm::vec3 radiance = h >= 0.0f ? glm::mix(horizon, zenith, h)
                                                 : glm::mix(horizon, ground, std::min(1.0f, -4.0f * h));
            float basis[16];
            shBasis16(dir, basis);
            const double weight = static_cast<double>(dU) * dPhi;
            for (int k = 0; k < 9; ++k)
            {
                for (int c = 0; c < 3; ++c)
                {
                    sum[k][c] += static_cast<double>(radiance[c]) * basis[k] * weight;
                }
            }
        }
    }
    for (int k = 0; k < 9; ++k)
    {
        env.sh[k] = glm::vec3(static_cast<float>(sum[k][0]), static_cast<float>(sum[k][1]),
                              static_cast<float>(sum[k][2]));
    }
    return env;
}

glm::vec3 EnvironmentLight::radiance(const glm::vec3 & dir) const
{
    float basis[16];
    shBasis16(dir, basis);
    glm::vec3 sum(0.0f);
    for (int k = 0; k < 9; ++k)
    {
        sum += sh[k] * basis[k];
    }
    return sum * intensity;
}

} // namespace sabercat
