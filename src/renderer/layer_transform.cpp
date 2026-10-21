// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/layer_transform.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "sabercat/gaussian_cloud.h"
#include "sabercat/logger.h"
#include "sabercat/parallel_for.h"
#include "sabercat/render_config.h"
#include "sh_rotation.h"
#include "unit_quaternion.h"

namespace sabercat
{

namespace
{

// The canonical layout holds 16 coefficients, i.e. degree 3.
constexpr uint32_t kMaxDegree = 3;

// The glTF reader's unit-quaternion tolerance (gltf_reader.cpp): a float
// quaternion this close to unit length is passed through, otherwise renormalised.
constexpr float kUnitQuaternionTolerance = 1e-6f;

// Rejects a placement that cannot be baked: non-finite values, a scale that is
// not positive, or a rotation that is not unit length.
void validateTransform(const LayerTransform & transform, const std::string & where)
{
    const glm::quat & q = transform.rotation;
    const float values[] = {q.w,
                            q.x,
                            q.y,
                            q.z,
                            transform.translation.x,
                            transform.translation.y,
                            transform.translation.z,
                            transform.scale};
    for (const float value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::runtime_error(where + " has a non-finite value");
        }
    }
    if (!(transform.scale > 0.0f))
    {
        throw std::runtime_error(where + " scale must be > 0");
    }
    // A layer placement that a project would refuse to save is refused here too.
    if (!isUnitLength(q.w, q.x, q.y, q.z))
    {
        throw std::runtime_error(where + " rotation is not unit length");
    }
}

// Which optional relight streams a bake carries across.
struct StreamPresence
{
    bool normals = false;
    bool baseColorRoughness = false;
    bool metallic = false;
    bool visibility = false;
};

StreamPresence presenceOf(const GaussianCloud & cloud)
{
    const GaussianCloud & c = cloud;
    StreamPresence presence;
    presence.normals = !c.normals().empty();
    presence.baseColorRoughness = !c.baseColorRoughness().empty();
    presence.metallic = !c.metallic().empty();
    presence.visibility = !c.visibility().empty();
    return presence;
}

// A layer's node transform, decomposed exactly as the glTF reader does it for a
// node (nodeLocalMatrix, then decomposeTransform), so the baked numbers match
// what the reader produces from the same .glb.
struct BakeFrame
{
    glm::dmat4 matrix{1.0};                                      // T * R * S
    sh_rotation::Mat3 rotation{};                                // rotation factor
    std::array<double, 4> rotationQuaternion{0.0, 0.0, 0.0, 1.0}; // xyzw
    double uniformScale = 1.0;
    std::array<sh_rotation::BandMatrix, 3> shBands{}; // bands 1..3
};

BakeFrame makeFrame(const LayerTransform & transform)
{
    // nodeLocalMatrix(): matrix[c][row] = R[row][c] * s, translation in column 3.
    const glm::quat & q = transform.rotation;
    const sh_rotation::Mat3 r = sh_rotation::quaternionToMatrix(q.x, q.y, q.z, q.w);
    const double s = transform.scale;
    glm::dmat4 local(1.0);
    for (int c = 0; c < 3; ++c)
    {
        for (int row = 0; row < 3; ++row)
        {
            local[c][row] = r[row][c] * s;
        }
        local[3][c] = transform.translation[c];
    }

    BakeFrame frame;
    frame.matrix = glm::dmat4(1.0) * local; // the reader's global = parent * local

    // decomposeTransform(): column lengths, column-normalised rotation factor.
    std::array<double, 3> lengths{};
    for (int c = 0; c < 3; ++c)
    {
        lengths[c] = std::sqrt(frame.matrix[c][0] * frame.matrix[c][0]
                               + frame.matrix[c][1] * frame.matrix[c][1]
                               + frame.matrix[c][2] * frame.matrix[c][2]);
    }
    sh_rotation::Mat3 normalized{};
    for (int row = 0; row < 3; ++row)
    {
        for (int c = 0; c < 3; ++c)
        {
            normalized[row][c] = frame.matrix[c][row] / lengths[c];
        }
    }
    frame.rotationQuaternion = sh_rotation::matrixToQuaternion(normalized);
    const auto & rq = frame.rotationQuaternion;
    frame.rotation = sh_rotation::quaternionToMatrix(rq[0], rq[1], rq[2], rq[3]);
    frame.uniformScale = (lengths[0] + lengths[1] + lengths[2]) / 3.0;
    for (int band = 1; band <= 3; ++band)
    {
        frame.shBands[band - 1] = sh_rotation::shBandRotation(band, frame.rotation);
    }
    return frame;
}

// Gives `out` storage for every stream in `keep`, sized for `count` Gaussians.
void allocateRelight(GaussianCloud & out, const StreamPresence & keep, size_t count)
{
    if (keep.normals)
    {
        out.normals().assign(count, glm::vec4(0.0f));
    }
    if (keep.baseColorRoughness)
    {
        out.baseColorRoughness().assign(count, glm::vec4(0.0f));
    }
    if (keep.metallic)
    {
        out.metallic().assign(count, 0.0f);
    }
    if (keep.visibility)
    {
        out.visibility().assign(count * kVisibilityCoefficients, 0.0f);
    }
}

// Writes `source` (baked by `transform`) into `dst` at Gaussian index `first`.
// `dst` must already be sized for it, and `keep` must be a subset of the
// streams `dst` was allocated with.
void bakeLayer(const GaussianCloud & source, const LayerTransform & transform, GaussianCloud & dst,
               size_t first, const StreamPresence & keep)
{
    const GaussianCloud & src = source;
    const size_t count = src.count();
    const auto & sp = src.positions();
    const auto & ss = src.scales();
    const auto & sr = src.rotations();
    const auto & ssh = src.shCoefficients();
    const auto & sop = src.opacities();
    const auto & st = src.times();
    if (sp.size() != count || ss.size() != count || sr.size() != count || sop.size() != count
        || st.size() != count || ssh.size() != count * kShCoefficientsPerGaussian)
    {
        throw std::runtime_error("layer transform: source host streams are not sized for "
                                 + std::to_string(count) + " Gaussians");
    }
    if ((keep.normals && src.normals().size() != count)
        || (keep.baseColorRoughness && src.baseColorRoughness().size() != count)
        || (keep.metallic && src.metallic().size() != count)
        || (keep.visibility && src.visibility().size() != count * kVisibilityCoefficients))
    {
        throw std::runtime_error("layer transform: a relight stream is not sized for "
                                 + std::to_string(count) + " Gaussians");
    }

    // Coefficients above the source's degree carry nothing; they are zeroed so the
    // merged cloud is zero-padded, as the glTF reader's zero-filled storage is.
    const size_t sourceDegree = std::min(src.shDegree(), kMaxDegree);
    const size_t keptCoefficients = (sourceDegree + 1) * (sourceDegree + 1);
    const bool identity = transform.isIdentity();
    const BakeFrame frame = makeFrame(transform);
    const sh_rotation::Mat3 & rot = frame.rotation;

    auto & dp = dst.positions();
    auto & dscale = dst.scales();
    auto & dr = dst.rotations();
    auto & dsh = dst.shCoefficients();
    auto & dop = dst.opacities();
    auto & dt = dst.times();
    auto & dn = dst.normals();
    auto & dbcr = dst.baseColorRoughness();
    auto & dm = dst.metallic();
    auto & dvis = dst.visibility();

    parallelFor(count,
                [&](size_t begin, size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                    {
                        const size_t o = first + i;
                        dop[o] = sop[i];
                        dt[o] = st[i];
                        if (identity)
                        {
                            dp[o] = sp[i];
                            dscale[o] = ss[i];
                            dr[o] = sr[i];
                            for (size_t k = 0; k < kShCoefficientsPerGaussian; ++k)
                            {
                                dsh[o * kShCoefficientsPerGaussian + k] =
                                    k < keptCoefficients
                                        ? ssh[i * kShCoefficientsPerGaussian + k]
                                        : glm::vec4(0.0f);
                            }
                        }
                        else
                        {
                            // Mean: the full node matrix, as the reader does it in double.
                            const glm::vec4 p = sp[i];
                            const glm::dvec4 world =
                                frame.matrix * glm::dvec4(p.x, p.y, p.z, 1.0);
                            dp[o] = glm::vec4(static_cast<float>(world.x),
                                              static_cast<float>(world.y),
                                              static_cast<float>(world.z),
                                              0.0f);

                            // Rotation: renormalise a float quaternion only if it is off
                            // unit length (the reader's rule), then compose in double.
                            glm::vec4 q = sr[i];
                            const float lengthSquared =
                                q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
                            if (std::abs(lengthSquared - 1.0f) > kUnitQuaternionTolerance)
                            {
                                const float invLength = 1.0f / std::sqrt(lengthSquared);
                                q *= invLength;
                            }
                            const auto & a = frame.rotationQuaternion;
                            const double bx = q.x, by = q.y, bz = q.z, bw = q.w;
                            double x = a[3] * bx + a[0] * bw + a[1] * bz - a[2] * by;
                            double y = a[3] * by - a[0] * bz + a[1] * bw + a[2] * bx;
                            double z = a[3] * bz + a[0] * by - a[1] * bx + a[2] * bw;
                            double w = a[3] * bw - a[0] * bx - a[1] * by - a[2] * bz;
                            const double inv = 1.0 / std::sqrt(x * x + y * y + z * z + w * w);
                            dr[o] = glm::vec4(static_cast<float>(x * inv),
                                              static_cast<float>(y * inv),
                                              static_cast<float>(z * inv),
                                              static_cast<float>(w * inv));

                            // Scale: uniform node scale multiplies the linear scales.
                            const glm::vec4 s = ss[i];
                            dscale[o] = glm::vec4(static_cast<float>(frame.uniformScale * s.x),
                                                  static_cast<float>(frame.uniformScale * s.y),
                                                  static_cast<float>(frame.uniformScale * s.z),
                                                  0.0f);

                            // SH: DC unchanged, bands 1..degree rotated.
                            glm::vec4 coefficients[kShCoefficientsPerGaussian];
                            for (size_t k = 0; k < kShCoefficientsPerGaussian; ++k)
                            {
                                coefficients[k] =
                                    k < keptCoefficients
                                        ? glm::vec4(ssh[i * kShCoefficientsPerGaussian + k].x,
                                                    ssh[i * kShCoefficientsPerGaussian + k].y,
                                                    ssh[i * kShCoefficientsPerGaussian + k].z,
                                                    0.0f)
                                        : glm::vec4(0.0f);
                            }
                            sh_rotation::rotateShCoefficients(
                                coefficients, static_cast<int>(sourceDegree), frame.shBands);
                            for (size_t k = 0; k < kShCoefficientsPerGaussian; ++k)
                            {
                                dsh[o * kShCoefficientsPerGaussian + k] = coefficients[k];
                            }
                        }

                        if (keep.normals)
                        {
                            const glm::vec4 n = src.normals()[i];
                            if (identity)
                            {
                                dn[o] = n;
                            }
                            else
                            {
                                const double nx = n.x, ny = n.y, nz = n.z;
                                dn[o] = glm::vec4(
                                    static_cast<float>(rot[0][0] * nx + rot[0][1] * ny
                                                       + rot[0][2] * nz),
                                    static_cast<float>(rot[1][0] * nx + rot[1][1] * ny
                                                       + rot[1][2] * nz),
                                    static_cast<float>(rot[2][0] * nx + rot[2][1] * ny
                                                       + rot[2][2] * nz),
                                    n.w);
                            }
                        }
                        if (keep.baseColorRoughness)
                        {
                            dbcr[o] = src.baseColorRoughness()[i];
                        }
                        if (keep.metallic)
                        {
                            dm[o] = src.metallic()[i];
                        }
                        if (keep.visibility)
                        {
                            for (size_t k = 0; k < kVisibilityCoefficients; ++k)
                            {
                                dvis[o * kVisibilityCoefficients + k] =
                                    src.visibility()[i * kVisibilityCoefficients + k];
                            }
                        }
                    }
                });
}

} // namespace

glm::mat4 LayerTransform::matrix() const
{
    return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation)
           * glm::scale(glm::mat4(1.0f), glm::vec3(scale));
}

bool LayerTransform::isIdentity() const
{
    return rotation.w == 1.0f && rotation.x == 0.0f && rotation.y == 0.0f && rotation.z == 0.0f
           && translation == glm::vec3(0.0f) && scale == 1.0f;
}

std::shared_ptr<GaussianCloud> transformedCopy(Device & device, const GaussianCloud & source,
                                               const LayerTransform & transform)
{
    validateTransform(transform, "transformedCopy: transform");
    const size_t count = source.count();
    const StreamPresence keep = presenceOf(source);

    auto out = std::make_shared<GaussianCloud>(device);
    out->initialize(count);
    allocateRelight(*out, keep, count);
    bakeLayer(source, transform, *out, 0, keep);
    out->setShDegree(source.shDegree());
    out->setAntialiased(source.antialiased());
    out->computeBounds();
    return out;
}

std::shared_ptr<GaussianCloud> mergeLayers(
    Device & device, const std::vector<std::pair<const GaussianCloud *, LayerTransform>> & layers)
{
    if (layers.empty())
    {
        throw std::runtime_error("mergeLayers: no layers to merge");
    }

    size_t total = 0;
    uint32_t degree = 0;
    bool antialiased = true;
    StreamPresence any;
    StreamPresence all{true, true, true, true};
    for (size_t i = 0; i < layers.size(); ++i)
    {
        const auto & layer = layers[i];
        const GaussianCloud * cloud = layer.first;
        if (cloud == nullptr)
        {
            throw std::runtime_error("mergeLayers: layer " + std::to_string(i) + " has no cloud");
        }
        validateTransform(layer.second, "mergeLayers: layer " + std::to_string(i) + " transform");
        total += cloud->count();
        degree = std::max(degree, cloud->shDegree());
        antialiased = antialiased && cloud->antialiased();
        const StreamPresence presence = presenceOf(*cloud);
        any.normals = any.normals || presence.normals;
        any.baseColorRoughness = any.baseColorRoughness || presence.baseColorRoughness;
        any.metallic = any.metallic || presence.metallic;
        any.visibility = any.visibility || presence.visibility;
        all.normals = all.normals && presence.normals;
        all.baseColorRoughness = all.baseColorRoughness && presence.baseColorRoughness;
        all.metallic = all.metallic && presence.metallic;
        all.visibility = all.visibility && presence.visibility;
    }

    // A stream some layers have and others lack is dropped; say so once per stream.
    auto warnDropped = [](const char * name, bool anyHas, bool allHave)
    {
        if (anyHas && !allHave)
        {
            Logger::warning(std::string("mergeLayers: relight stream '") + name
                            + "' is missing from some layers; dropped from the merged cloud");
        }
    };
    warnDropped("normals", any.normals, all.normals);
    warnDropped("baseColorRoughness", any.baseColorRoughness, all.baseColorRoughness);
    warnDropped("metallic", any.metallic, all.metallic);
    warnDropped("visibility", any.visibility, all.visibility);

    auto out = std::make_shared<GaussianCloud>(device);
    out->initialize(total);
    allocateRelight(*out, all, total);
    size_t first = 0;
    for (const auto & layer : layers)
    {
        bakeLayer(*layer.first, layer.second, *out, first, all);
        first += layer.first->count();
    }
    out->setShDegree(degree);
    out->setAntialiased(antialiased);
    out->computeBounds();
    return out;
}

} // namespace sabercat
