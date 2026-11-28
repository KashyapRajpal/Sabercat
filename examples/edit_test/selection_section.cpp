// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sections J and K of edit_test: the C3 selection and the selection highlight.
//
//  J. Selection, on the main Device. Layer A (100,003 Gaussians) and B (4,099), at
//     identity and at transformT1(). Each selection (box, sphere, rectangle; Replace,
//     Add, Subtract) runs on the GPU and on a copy of the layer, and is checked against
//     a double CPU predicate: every Gaussian outside the boundary tolerance must match,
//     the mismatches (allowed only on the boundary) are counted and printed, and the
//     copy must select identically. Then the op combinations from a random old state,
//     the mask operations (clear, all, invert, countSelected, the masked centroid,
//     expandMaskToFlags), SelectionMask (upload, download, resize, a capacity above
//     the count) and the refusals. The rectangles run on a Renderer's device, with the
//     camera of its last forward().
//  K. The selection highlight on a Renderer: an empty mask changes nothing, a full
//     selection tints toward red, a spatial half changes some pixels, the Differentiable
//     forward ignores it, disabling and a scene change restore the image, the errors,
//     the readImage() rule, and the merged-mask boundary (the last Gaussian of A2 and
//     the first of B).
//
// Boundary tolerances: a Gaussian whose predicate margin is inside them may legitimately
// differ between the GPU (fp32) and the CPU (double) and is counted, not failed.

#include "edit_test_common.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/edit_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/highlight_settings.h"
#include "sabercat/layer_transform.h"
#include "sabercat/renderer.h"
#include "sabercat/selection_mask.h"

#include "../common/shader_dir.h"

using namespace sabercat;
using namespace edit_test;

namespace edit_test
{
namespace
{

constexpr uint32_t kWidth = 128;
constexpr uint32_t kHeight = 96;
constexpr size_t kCountA = 100003;
constexpr size_t kCountA2 = 5003;
constexpr size_t kCountB = 4099;
constexpr uint32_t kUintStride = sizeof(uint32_t);

// Boundary tolerances on the CPU margins.
constexpr double kWorldEps = 1e-4; // box and sphere margins, world units
constexpr double kPixelEps = 1e-3; // rectangle margins, pixels
constexpr double kDepthEps = 1e-4; // depth margin, world units
constexpr double kMinDepth = 0.2;  // SABERCAT_MIN_DEPTH

XorShift32 g_selRng(0xC3C3C3C3u);

// True when `fn` throws a std::exception.
template <typename Fn>
bool throwsException(Fn && fn)
{
    try
    {
        fn();
    }
    catch (const std::exception &)
    {
        return true;
    }
    return false;
}

// Camera 14 units back on +z, looking down -z (as renderer_section.cpp's).
Camera framingCamera()
{
    Camera c;
    c.width = kWidth;
    c.height = kHeight;
    c.fx = 130.0f;
    c.fy = 130.0f;
    c.cx = 0.5f * static_cast<float>(kWidth - 1);
    c.cy = 0.5f * static_cast<float>(kHeight - 1);
    c.cameraToWorld = glm::mat4(1.0f);
    c.cameraToWorld[3] = glm::vec4(0.0f, 0.0f, 14.0f, 1.0f);
    return c;
}

glm::quat identityRotation()
{
    return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
}

// --- Queries ----------------------------------------------------------------------

SelectionQuery boxQuery(glm::vec3 center, glm::quat rotation, glm::vec3 halfExtents,
                        SelectOp op)
{
    const glm::mat3 axes = glm::mat3_cast(rotation);
    SelectionQuery q;
    q.shape = SelectShape::Box;
    q.op = op;
    q.center = center;
    q.halfExtents = halfExtents;
    q.boxAxisX = axes[0];
    q.boxAxisY = axes[1];
    q.boxAxisZ = axes[2];
    return q;
}

SelectionQuery sphereQuery(glm::vec3 center, float radius, SelectOp op)
{
    SelectionQuery q;
    q.shape = SelectShape::Sphere;
    q.op = op;
    q.center = center;
    q.radius = radius;
    return q;
}

SelectionQuery rectQuery(glm::vec4 rect, const CameraUniforms & cam, SelectOp op)
{
    SelectionQuery q;
    q.shape = SelectShape::Rectangle;
    q.op = op;
    q.rect = rect;
    q.camera = cam;
    return q;
}

// --- Bit vectors --------------------------------------------------------------------

// `count` random bits, zero past count.
std::vector<uint32_t> randomWords(size_t count)
{
    std::vector<uint32_t> words((count + 31) / 32, 0u);
    for (uint32_t & w : words)
    {
        w = g_selRng.next();
    }
    if (count % 32 != 0)
    {
        words.back() &= (1u << (count % 32)) - 1u;
    }
    return words;
}

// The bits [0, count) of `words`, one byte per Gaussian.
std::vector<uint8_t> unpackBits(const std::vector<uint32_t> & words, size_t count)
{
    std::vector<uint8_t> bits(count, 0);
    for (size_t i = 0; i < count; ++i)
    {
        bits[i] = static_cast<uint8_t>((words[i >> 5] >> (i & 31u)) & 1u);
    }
    return bits;
}

// The words holding `bits`, zero past the end.
std::vector<uint32_t> packBits(const std::vector<uint8_t> & bits)
{
    std::vector<uint32_t> words((bits.size() + 31) / 32, 0u);
    for (size_t i = 0; i < bits.size(); ++i)
    {
        if (bits[i] != 0)
        {
            words[i >> 5] |= 1u << (i & 31u);
        }
    }
    return words;
}

// Set bits among the first n Gaussians of a bit vector.
size_t countPrefix(const std::vector<uint8_t> & bits, size_t n)
{
    size_t total = 0;
    for (size_t i = 0; i < n && i < bits.size(); ++i)
    {
        total += bits[i];
    }
    return total;
}

// Set bits in all of `words`.
size_t wordBits(const std::vector<uint32_t> & words)
{
    size_t total = 0;
    for (const uint32_t w : words)
    {
        for (uint32_t x = w; x != 0; x &= x - 1)
        {
            ++total;
        }
    }
    return total;
}

// Set bits at or beyond `count` in `words`.
size_t bitsBeyond(const std::vector<uint32_t> & words, size_t count)
{
    size_t total = 0;
    for (size_t i = count; i < words.size() * 32; ++i)
    {
        if (((words[i >> 5] >> (i & 31u)) & 1u) != 0u)
        {
            ++total;
        }
    }
    return total;
}

// --- CPU reference ----------------------------------------------------------------

// Each Gaussian's world position in double. Identity: the layer's own positions. Otherwise
// the positions transformedCopy() produces, which is what the merge produces in float.
std::vector<glm::dvec3> worldPositions(Device & device, const GaussianCloud & layer,
                                       const LayerTransform & placement)
{
    const size_t n = layer.count();
    std::vector<glm::dvec3> out(n);
    if (placement.isIdentity())
    {
        for (size_t i = 0; i < n; ++i)
        {
            out[i] = glm::dvec3(glm::vec3(layer.positions()[i]));
        }
        return out;
    }
    const std::shared_ptr<GaussianCloud> placed = transformedCopy(device, layer, placement);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = glm::dvec3(glm::vec3(placed->positions()[i]));
    }
    return out;
}

// One predicate over a layer: hit where the margin is >= 0, boundary where the margin
// is inside the tolerance.
struct Predicate
{
    std::vector<uint8_t> hit;
    std::vector<uint8_t> boundary;
};

Predicate boxPredicate(const SelectionQuery & q, const std::vector<glm::dvec3> & world)
{
    const glm::dvec3 center(q.center);
    const glm::dvec3 half(q.halfExtents);
    const glm::dvec3 axes[3] = {glm::dvec3(q.boxAxisX), glm::dvec3(q.boxAxisY),
                                glm::dvec3(q.boxAxisZ)};
    Predicate p;
    p.hit.resize(world.size());
    p.boundary.resize(world.size());
    for (size_t i = 0; i < world.size(); ++i)
    {
        const glm::dvec3 d = world[i] - center;
        // The smallest of h_k - |axis_k . d|: inside the box when >= 0.
        double margin = std::numeric_limits<double>::infinity();
        for (int k = 0; k < 3; ++k)
        {
            margin = std::min(margin, half[k] - std::fabs(glm::dot(axes[k], d)));
        }
        p.hit[i] = margin >= 0.0 ? 1 : 0;
        p.boundary[i] = std::fabs(margin) < kWorldEps ? 1 : 0;
    }
    return p;
}

Predicate spherePredicate(const SelectionQuery & q, const std::vector<glm::dvec3> & world)
{
    const glm::dvec3 center(q.center);
    const double radius = static_cast<double>(q.radius);
    Predicate p;
    p.hit.resize(world.size());
    p.boundary.resize(world.size());
    for (size_t i = 0; i < world.size(); ++i)
    {
        const double margin = radius - glm::length(world[i] - center);
        p.hit[i] = margin >= 0.0 ? 1 : 0;
        p.boundary[i] = std::fabs(margin) < kWorldEps ? 1 : 0;
    }
    return p;
}

// The rectangle through the query's camera, as computeProjectionGeometry does it, in double.
// Hit needs depth > 0.2 and the pixel inside the normalized rectangle, inclusive.
Predicate rectPredicate(const SelectionQuery & q, const std::vector<glm::dvec3> & world)
{
    const CameraUniforms & cam = q.camera;
    glm::dmat4 worldToCamera;
    for (int c = 0; c < 4; ++c)
    {
        for (int r = 0; r < 4; ++r)
        {
            worldToCamera[c][r] = static_cast<double>(cam.worldToCamera[c][r]);
        }
    }
    const double fx = static_cast<double>(cam.fx);
    const double fy = static_cast<double>(cam.fy);
    const double cx = static_cast<double>(cam.cx);
    const double cy = static_cast<double>(cam.cy);
    const double x0 = std::min(static_cast<double>(q.rect.x), static_cast<double>(q.rect.z));
    const double x1 = std::max(static_cast<double>(q.rect.x), static_cast<double>(q.rect.z));
    const double y0 = std::min(static_cast<double>(q.rect.y), static_cast<double>(q.rect.w));
    const double y1 = std::max(static_cast<double>(q.rect.y), static_cast<double>(q.rect.w));

    Predicate p;
    p.hit.resize(world.size());
    p.boundary.resize(world.size());
    for (size_t i = 0; i < world.size(); ++i)
    {
        const glm::dvec4 pc = worldToCamera * glm::dvec4(world[i], 1.0);
        const double depth = -pc.z;
        const double depthMargin = depth - kMinDepth;
        double pixelMargin = std::numeric_limits<double>::infinity();
        if (depthMargin > 0.0)
        {
            const double u = fx * pc.x / depth + cx;
            const double v = cy - fy * pc.y / depth;
            pixelMargin = std::min(std::min(u - x0, x1 - u), std::min(v - y0, y1 - v));
        }
        p.hit[i] = (depthMargin > 0.0 && pixelMargin >= 0.0) ? 1 : 0;
        const bool onPixelEdge = depthMargin > 0.0 && std::fabs(pixelMargin) < kPixelEps;
        p.boundary[i] = (std::fabs(depthMargin) < kDepthEps || onPixelEdge) ? 1 : 0;
    }
    return p;
}

Predicate makePredicate(const SelectionQuery & q, const std::vector<glm::dvec3> & world)
{
    switch (q.shape)
    {
    case SelectShape::Box:
        return boxPredicate(q, world);
    case SelectShape::Sphere:
        return spherePredicate(q, world);
    case SelectShape::Rectangle:
        return rectPredicate(q, world);
    }
    throw std::logic_error("makePredicate: unknown shape");
}

// What one selection did against the CPU model.
struct Tally
{
    size_t selected = 0;           // set bits after the step
    size_t boundary = 0;           // boundary Gaussians of the predicate
    size_t mismatches = 0;         // off the boundary: must be zero
    size_t boundaryMismatches = 0; // on the boundary: allowed, counted
};

// Advances the CPU model `state` by one selection and compares it with the GPU words.
// Off the boundary the expected bit is op(old, hit). On the boundary the GPU's bit is
// accepted and adopted, so later steps are checked against what the GPU holds.
Tally applyOp(const Predicate & p, SelectOp op, const std::vector<uint32_t> & gpuWords,
              std::vector<uint8_t> & state)
{
    const size_t n = state.size();
    const std::vector<uint8_t> gpu = unpackBits(gpuWords, n);
    Tally t;
    for (size_t i = 0; i < n; ++i)
    {
        const uint8_t hit = p.hit[i];
        uint8_t expected = hit;
        if (op == SelectOp::Add)
        {
            expected = static_cast<uint8_t>(state[i] | hit);
        }
        else if (op == SelectOp::Subtract)
        {
            expected = static_cast<uint8_t>(state[i] & (hit ^ 1u));
        }
        if (p.boundary[i] != 0)
        {
            ++t.boundary;
            if (gpu[i] != expected)
            {
                ++t.boundaryMismatches;
            }
            state[i] = gpu[i];
        }
        else
        {
            if (gpu[i] != expected)
            {
                ++t.mismatches;
            }
            state[i] = expected;
        }
    }
    t.selected = countPrefix(state, n);
    return t;
}

// --- Selection steps --------------------------------------------------------------

// A layer under test: the cloud, a copy of it (the same host data, uploaded again) and
// the masks of both, with the CPU model of the mask.
struct Lane
{
    Device & device;
    EditPass & edit;
    const GaussianCloud & layer;
    const GaussianCloud & copy;
    SelectionMask & mask;
    SelectionMask & copyMask;
    std::vector<uint8_t> state;
};

// One selection on the layer and on its copy, checked against the CPU predicate.
Tally runSelectStep(Lane & lane, const std::string & name, const LayerTransform & placement,
                    const SelectionQuery & q)
{
    const size_t n = lane.state.size();
    const std::vector<glm::dvec3> world = worldPositions(lane.device, lane.layer, placement);
    const Predicate pred = makePredicate(q, world);

    lane.edit.select(lane.layer, placement, q, lane.mask);
    lane.edit.select(lane.copy, placement, q, lane.copyMask);
    const std::vector<uint32_t> words = lane.mask.download(n);
    const std::vector<uint32_t> copyWords = lane.copyMask.download(n);

    const Tally t = applyOp(pred, q.op, words, lane.state);
    std::printf("  J: %s: %zu selected, %zu boundary Gaussians, %zu mismatches (%zu on the "
                "boundary)\n",
                name.c_str(), t.selected, t.boundary, t.mismatches, t.boundaryMismatches);
    check(t.mismatches == 0,
          "J: " + name + ": GPU bits equal the CPU predicate off the boundary");
    check(bitsBeyond(words, n) == 0, "J: " + name + ": no bits at or beyond the count");
    check(copyWords == words, "J: " + name + ": a copy of the layer selects identically");
    return t;
}

// --- J: selection ------------------------------------------------------------------

// The mask operations on the final selection `state` of `mask` (the layer `a`).
void checkMaskOps(Device & device, EditPass & edit, const GaussianCloud & a, SelectionMask & mask,
                  const std::vector<uint8_t> & state)
{
    const size_t n = a.count();
    const uint32_t layerCount = static_cast<uint32_t>(n);

    // countSelected against the CPU popcount, around the word boundaries.
    const uint32_t counts[] = {0, 1, 31, 32, 33, 99999, layerCount};
    for (const uint32_t c : counts)
    {
        const size_t want = countPrefix(state, c);
        const uint32_t got = edit.countSelected(mask, c);
        check(got == want, "J: countSelected(count " + std::to_string(c) + ") equals the CPU popcount ("
                               + std::to_string(got) + ", CPU " + std::to_string(want) + ")");
    }
    const std::vector<uint32_t> words = mask.download(n);
    check(bitsBeyond(words, n) == 0, "J: the selection holds no bits at or beyond the layer count");
    check(wordBits(words) == edit.countSelected(mask, layerCount),
          "J: the set bits of the downloaded words equal countSelected");

    // The masked centroid, in layer-local space, against the CPU double centroid.
    const MaskCentroid centroid = edit.maskedCentroid(a, mask);
    glm::dvec3 sum(0.0);
    size_t cpuCount = 0;
    glm::dvec3 lo(std::numeric_limits<double>::infinity());
    glm::dvec3 hi(-std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < n; ++i)
    {
        const glm::dvec3 p(glm::vec3(a.positions()[i]));
        for (int k = 0; k < 3; ++k)
        {
            lo[k] = std::min(lo[k], p[k]);
            hi[k] = std::max(hi[k], p[k]);
        }
        if (state[i] != 0)
        {
            sum += p;
            ++cpuCount;
        }
    }
    const glm::dvec3 cpuCentroid = cpuCount > 0 ? sum / static_cast<double>(cpuCount) : glm::dvec3(0.0);
    const glm::dvec3 extent = hi - lo;
    const double span = std::max(extent.x, std::max(extent.y, extent.z));
    const double centroidError = glm::length(centroid.centroid() - cpuCentroid) / span;
    std::printf("  J: masked centroid over %zu selected: relative error %.3g of the layer extent\n",
                cpuCount, centroidError);
    check(centroid.count == static_cast<uint64_t>(cpuCount),
          "J: maskedCentroid count equals the CPU count");
    check(centroidError <= 1e-4, "J: maskedCentroid within 1e-4 of the layer extent");

    SelectionMask empty(device, n);
    const MaskCentroid none = edit.maskedCentroid(a, empty);
    check(none.count == 0 && glm::length(none.centroid()) == 0.0,
          "J: an empty mask has count 0 and a zero centroid");

    // expandMaskToFlags, both polarities, against the CPU flags.
    const VkDeviceSize flagBytes = static_cast<VkDeviceSize>(n) * kUintStride;
    Buffer flags(device, flagBytes);
    for (int pass = 0; pass < 2; ++pass)
    {
        const bool keepUnselected = pass == 1;
        edit.expandMaskToFlags(mask, layerCount, keepUnselected, flags.view(kUintStride));
        std::vector<uint32_t> got(n, 0u);
        flags.download(got.data(), flagBytes);
        std::vector<uint32_t> want(n, 0u);
        for (size_t i = 0; i < n; ++i)
        {
            const uint32_t selected = state[i] != 0 ? 1u : 0u;
            want[i] = keepUnselected ? 1u - selected : selected;
        }
        check(got == want, std::string("J: expandMaskToFlags(keepUnselected ")
                               + (keepUnselected ? "true" : "false") + ") equals the CPU flags");
    }

    // invertMask over the full count, and over a shorter count that leaves bits past it.
    SelectionMask inverted(device, n);
    inverted.upload(packBits(state));
    edit.invertMask(inverted, layerCount);
    std::vector<uint8_t> complement(n, 0);
    for (size_t i = 0; i < n; ++i)
    {
        complement[i] = state[i] != 0 ? 0 : 1;
    }
    check(inverted.download(n) == packBits(complement),
          "J: invertMask over the full count equals the CPU complement");

    SelectionMask shorter(device, n);
    shorter.upload(packBits(state));
    edit.invertMask(shorter, 99999);
    std::vector<uint8_t> shortComplement(n, 0);
    for (size_t i = 0; i < 99999; ++i)
    {
        shortComplement[i] = state[i] != 0 ? 0 : 1;
    }
    const std::vector<uint32_t> shortWords = shorter.download(n);
    check(shortWords == packBits(shortComplement) && bitsBeyond(shortWords, 99999) == 0,
          "J: invertMask over 99999 flips only the bits below 99999 and leaves the rest zero");

    // clearMask, then selectAll over the full count and over 99999 of the capacity.
    edit.clearMask(mask);
    check(wordBits(mask.download(mask.capacity())) == 0, "J: clearMask leaves every word zero");

    edit.selectAll(mask, layerCount);
    std::vector<uint8_t> every(n, 1);
    check(edit.countSelected(mask, layerCount) == layerCount && mask.download(n) == packBits(every),
          "J: selectAll(count) sets exactly the bits below count");

    edit.selectAll(mask, 99999);
    const std::vector<uint32_t> partialWords = mask.download(n);
    check(edit.countSelected(mask, 99999) == 99999 && edit.countSelected(mask, layerCount) == 99999
              && bitsBeyond(partialWords, 99999) == 0,
          "J: selectAll(99999) on a mask of capacity 100003 keeps the bits past 99999 zero");
}

// The op combinations: each op with one sphere, from a random old state, against the CPU model.
void checkOpCombinations(Device & device, EditPass & edit, const GaussianCloud & a)
{
    const LayerTransform id = identityTransform();
    const std::vector<glm::dvec3> world = worldPositions(device, a, id);
    const SelectOp ops[3] = {SelectOp::Replace, SelectOp::Add, SelectOp::Subtract};
    const char * names[3] = {"Replace", "Add", "Subtract"};
    for (int k = 0; k < 3; ++k)
    {
        SelectionMask mask(device, kCountA);
        const std::vector<uint32_t> old = randomWords(kCountA);
        mask.upload(old);
        std::vector<uint8_t> state = unpackBits(old, kCountA);

        const SelectionQuery q = sphereQuery(glm::vec3(0.0f), 3.5f, ops[k]);
        const Predicate pred = makePredicate(q, world);
        edit.select(a, id, q, mask);
        const Tally t = applyOp(pred, ops[k], mask.download(kCountA), state);
        std::printf("  J: sphere %s over a random old state: %zu selected, %zu boundary, %zu "
                    "mismatches (%zu on the boundary)\n",
                    names[k], t.selected, t.boundary, t.mismatches, t.boundaryMismatches);
        check(t.mismatches == 0, std::string("J: sphere ") + names[k]
                                     + " over a random old state: GPU bits equal the CPU model "
                                       "off the boundary");
    }
}

// SelectionMask on its own: upload and download, resize, and the refusals.
void checkSelectionMask(Device & device, EditPass & edit, const GaussianCloud & a)
{
    SelectionMask m(device, kCountA);
    const std::vector<uint32_t> words = randomWords(kCountA);
    m.upload(words);
    check(m.download(kCountA) == words, "J: SelectionMask upload then download is bit-exact");

    const std::vector<uint32_t> shortWords(10, 0x5A5A5A5Au);
    m.upload(shortWords);
    const std::vector<uint32_t> back = m.download(kCountA);
    bool shortOk = back.size() == words.size();
    for (size_t w = 0; shortOk && w < back.size(); ++w)
    {
        shortOk = back[w] == (w < shortWords.size() ? shortWords[w] : 0u);
    }
    check(shortOk, "J: an upload of fewer words zeroes the rest of the mask");
    check(throwsException([&] { m.upload(std::vector<uint32_t>(m.wordCount() + 1, 0u)); }),
          "J: an upload over wordCount throws");

    m.upload(words);
    const std::vector<uint32_t> before = m.download(kCountA);
    m.resize(2 * kCountA);
    const size_t grownWords = (2 * kCountA + 31) / 32;
    check(m.capacity() == 2 * kCountA && m.wordCount() == grownWords,
          "J: resize grows the capacity and the word count");
    check(m.download(kCountA) == before, "J: resize keeps the contents");
    const std::vector<uint32_t> grown = m.download(2 * kCountA);
    bool newZero = grown.size() == grownWords;
    for (size_t w = before.size(); newZero && w < grown.size(); ++w)
    {
        newZero = grown[w] == 0u;
    }
    check(newZero, "J: resize zeroes the new words");
    m.resize(10);
    check(m.capacity() == 2 * kCountA && m.download(kCountA) == before,
          "J: a resize to a smaller capacity is a no-op");

    SelectionMask small(device, 100);
    const SelectionQuery sphere = sphereQuery(glm::vec3(0.0f), 3.5f, SelectOp::Replace);
    check(throwsException([&] { edit.select(a, identityTransform(), sphere, small); }),
          "J: select into a mask smaller than the layer throws");

    SelectionQuery rect;
    rect.shape = SelectShape::Rectangle;
    rect.rect = glm::vec4(0.0f, 0.0f, 10.0f, 10.0f);
    check(throwsException([&] { edit.select(a, identityTransform(), rect, m); }),
          "J: a Rectangle query whose camera has fx <= 0 throws");
}

// The rectangle shape: a forward of B (T1) on a Renderer, its camera, and rectangles
// selected through that camera (Replace, a swapped-corner Add, a Subtract).
void checkRectangleSelection()
{
    Renderer r(kWidth, kHeight);
    if (!r.initialize({}, true, testShaderDir()))
    {
        check(false, "J: renderer initialize (rectangle)");
        return;
    }
    Device & rd = r.sabercatDevice();
    const std::shared_ptr<GaussianCloud> bPtr =
        makeSynthetic(rd, SyntheticSpec{kCountB, 1, false, true, 0xB0B0003u});
    const GaussianCloud & b = *bPtr;
    const LayerTransform t1 = transformT1();

    r.setCamera(framingCamera());
    r.mergeLayersOnGpu({{bPtr.get(), t1}});
    r.forward(RenderMode::Viewer);
    const CameraUniforms cam = r.lastForwardCamera();
    check(cam.fx > 0.0f && cam.fy > 0.0f, "J: lastForwardCamera after a forward of B (T1)");

    EditPass edit(rd, testShaderDir());
    const std::shared_ptr<GaussianCloud> bCopy = transformedCopy(rd, b, identityTransform());
    bCopy->uploadToGPU();
    SelectionMask mask(rd, kCountB);
    SelectionMask maskCopy(rd, kCountB);
    Lane lane{rd, edit, b, *bCopy, mask, maskCopy, std::vector<uint8_t>(kCountB, 0)};

    // The middle third of the width, the full height.
    const float third = static_cast<float>(kWidth) / 3.0f;
    const float height = static_cast<float>(kHeight - 1);
    const Tally replace = runSelectStep(
        lane, "rectangle (T1), Replace", t1,
        rectQuery(glm::vec4(third, 0.0f, 2.0f * third, height), cam, SelectOp::Replace));
    check(replace.selected >= 50 && replace.selected + 50 <= kCountB,
          "J: the middle rectangle selects at least 50 and at most count - 50 ("
              + std::to_string(replace.selected) + " selected)");

    // The same rectangle with its corners swapped selects the same Gaussians.
    const std::vector<uint32_t> replaceWords = mask.download(kCountB);
    SelectionMask swapped(rd, kCountB);
    edit.select(b, t1, rectQuery(glm::vec4(2.0f * third, height, third, 0.0f), cam, SelectOp::Replace),
                swapped);
    check(swapped.download(kCountB) == replaceWords,
          "J: swapped corners select the same as the normalized rectangle");

    runSelectStep(lane, "rectangle (T1), Add with swapped corners", t1,
                  rectQuery(glm::vec4(110.0f, 80.0f, 100.0f, 60.0f), cam, SelectOp::Add));
    runSelectStep(lane, "rectangle (T1), Subtract", t1,
                  rectQuery(glm::vec4(50.0f, 30.0f, 70.0f, 50.0f), cam, SelectOp::Subtract));
}

void runSelectionSection(Device & device)
{
    EditPass edit(device, testShaderDir());
    const std::shared_ptr<GaussianCloud> aPtr =
        makeSynthetic(device, SyntheticSpec{kCountA, 3, true, false, 0xA0A0001u});
    const GaussianCloud & a = *aPtr;
    const LayerTransform id = identityTransform();
    const LayerTransform t1 = transformT1();

    const std::shared_ptr<GaussianCloud> aCopy = transformedCopy(device, a, id);
    aCopy->uploadToGPU();
    SelectionMask maskA(device, kCountA);
    SelectionMask maskACopy(device, kCountA);
    Lane lane{device, edit, a, *aCopy, maskA, maskACopy, std::vector<uint8_t>(kCountA, 0)};

    runSelectStep(lane, "box (identity), Replace", id,
                  boxQuery(glm::vec3(0.5f, -0.3f, 0.2f),
                           glm::angleAxis(0.4f, glm::normalize(glm::vec3(1.0f, 2.0f, 0.5f))),
                           glm::vec3(2.0f, 1.5f, 3.0f), SelectOp::Replace));
    runSelectStep(lane, "sphere (identity), Add", id,
                  sphereQuery(glm::vec3(-2.0f, 1.0f, 0.0f), 2.5f, SelectOp::Add));
    runSelectStep(lane, "box (identity), Subtract", id,
                  boxQuery(glm::vec3(1.0f, 0.0f, 0.0f), identityRotation(),
                           glm::vec3(1.0f, 6.0f, 6.0f), SelectOp::Subtract));
    runSelectStep(lane, "box (T1), Replace", t1,
                  boxQuery(glm::vec3(1.5f, -2.0f, 0.25f),
                           glm::angleAxis(-0.3f, glm::vec3(0.0f, 0.0f, 1.0f)),
                           glm::vec3(3.0f, 2.0f, 4.0f), SelectOp::Replace));
    runSelectStep(lane, "sphere (T1), Add", t1,
                  sphereQuery(glm::vec3(-1.0f, 0.0f, 1.0f), 3.0f, SelectOp::Add));

    checkMaskOps(device, edit, a, maskA, lane.state);
    checkOpCombinations(device, edit, a);
    checkSelectionMask(device, edit, a);
    checkRectangleSelection();
}

// --- K: selection highlight ---------------------------------------------------------

// A highlight with the red tint (1, 0, 0) at blend weight `alpha`, or disabled.
HighlightSettings highlight(bool on, float alpha)
{
    HighlightSettings s;
    s.enabled = on ? 1u : 0u;
    s.tint[0] = 1.0f;
    s.tint[1] = 0.0f;
    s.tint[2] = 0.0f;
    s.tint[3] = alpha;
    return s;
}

std::vector<float> renderViewer(Renderer & r)
{
    r.forward(RenderMode::Viewer);
    return r.readImage();
}

// Whether the pixel of two RGBA frames is bit-identical.
bool samePixel(const std::vector<float> & a, const std::vector<float> & b, size_t p)
{
    for (size_t c = 0; c < 4; ++c)
    {
        if (!sameBits(a[4 * p + c], b[4 * p + c]))
        {
            return false;
        }
    }
    return true;
}

void checkHighlightFrames(Renderer & r, Device & device, const std::vector<MergeLayer> & layers,
                          const std::shared_ptr<GaussianCloud> & a2,
                          const std::shared_ptr<GaussianCloud> & b)
{
    const LayerTransform id = identityTransform();
    EditPass edit(device, testShaderDir());
    SelectionMask maskA2(device, kCountA2);
    SelectionMask maskB(device, kCountB);
    const std::vector<BufferView> views = {maskA2.view(), maskB.view()};
    const size_t pixels = static_cast<size_t>(kWidth) * kHeight;

    // K1: the base frame, then empty masks with the highlight on (bit-identical). K7: the
    // frame ends at a setSelectionHighlight, so readImage() throws until the next forward.
    const std::vector<float> base = renderViewer(r);
    double coverage = 0.0;
    for (size_t p = 0; p < pixels; ++p)
    {
        coverage += base[4 * p + 3];
    }
    check(coverage > 0.0, "K: the base frame has coverage");

    r.setSelectionHighlight(views, highlight(true, 1.0f));
    check(throwsException([&] { r.readImage(); }),
          "K7: readImage() throws right after setSelectionHighlight, until the next forward()");
    check(bitEqual(renderViewer(r), base),
          "K1: empty masks with the highlight on leave the image bit-identical to the base");

    // K2: every A2 Gaussian selected, full red tint.
    edit.selectAll(maskA2, static_cast<uint32_t>(kCountA2));
    r.setSelectionHighlight(views, highlight(true, 1.0f));
    const std::vector<float> allA2 = renderViewer(r);
    check(!bitEqual(allA2, base), "K2: selecting all of A2 changes the image");
    bool redOk = true;
    double redBase = 0.0;
    double redAll = 0.0;
    double greenBase = 0.0;
    double greenAll = 0.0;
    for (size_t p = 0; p < pixels; ++p)
    {
        const size_t i = 4 * p;
        if (allA2[i] < base[i] - 1e-6f)
        {
            redOk = false;
        }
        if (base[i + 3] > 0.0f)
        {
            redBase += base[i];
            redAll += allA2[i];
            greenBase += base[i + 1];
            greenAll += allA2[i + 1];
        }
    }
    check(redOk, "K2: every pixel's red is at least its base red (tint red 1)");
    check(redAll > redBase, "K2: the mean red over covered pixels increased");
    check(greenAll < greenBase, "K2: the mean green over covered pixels decreased");
    r.setSelectionHighlight(views, highlight(true, 0.0f));
    check(bitEqual(renderViewer(r), base), "K2: tint alpha 0 is bit-identical to the base");

    // K3: a spatial half of A2 and of B (x <= 0), full red tint.
    edit.clearMask(maskA2);
    edit.clearMask(maskB);
    const SelectionQuery halfBox = boxQuery(glm::vec3(-5.5f, 0.0f, 0.0f), identityRotation(),
                                            glm::vec3(5.5f, 20.0f, 20.0f), SelectOp::Replace);
    edit.select(*a2, id, halfBox, maskA2);
    edit.select(*b, id, halfBox, maskB);
    r.setSelectionHighlight(views, highlight(true, 1.0f));
    const std::vector<float> half = renderViewer(r);
    size_t differ = 0;
    for (size_t p = 0; p < pixels; ++p)
    {
        if (!samePixel(half, base, p))
        {
            ++differ;
        }
    }
    std::printf("  K3: %zu of %zu pixels change under the half selection\n", differ, pixels);
    check(differ * 100 >= pixels, "K3: at least 1% of the pixels change");
    check((pixels - differ) * 100 >= pixels, "K3: at least 1% of the pixels do not change");

    // K4: the Differentiable forward ignores the highlight (the half selection is set).
    r.setSelectionHighlight({}, highlight(false, 0.0f));
    r.forward(RenderMode::Differentiable);
    const std::vector<float> diffOff = r.readImage();
    r.setSelectionHighlight(views, highlight(true, 1.0f));
    r.forward(RenderMode::Differentiable);
    const std::vector<float> diffOn = r.readImage();
    check(bitEqual(diffOn, diffOff), "K4: a Differentiable forward ignores the highlight");

    // K5: disabling restores the base image; a scene change turns the highlight off.
    r.setSelectionHighlight({}, highlight(false, 0.0f));
    check(bitEqual(renderViewer(r), base),
          "K5: disabling the highlight restores the base image bit for bit");
    r.setSelectionHighlight(views, highlight(true, 1.0f));
    r.mergeLayersOnGpu(layers);
    const std::vector<float> afterMerge = renderViewer(r);
    r.setSelectionHighlight({}, highlight(false, 0.0f));
    const std::vector<float> afterDisable = renderViewer(r);
    check(bitEqual(afterMerge, afterDisable),
          "K5: a scene change turns the highlight off (the frame equals the disabled one)");

    // K6: the errors.
    check(throwsException([&] { r.setSelectionHighlight({maskA2.view()}, highlight(true, 1.0f)); }),
          "K6: setSelectionHighlight with one mask for the two merged layers throws");
    Renderer fresh(kWidth, kHeight);
    if (fresh.initialize({}, true, testShaderDir()))
    {
        check(throwsException([&] { fresh.lastForwardCamera(); }),
              "K6: lastForwardCamera() throws on a renderer with no forward");
        check(throwsException([&] { fresh.setSelectionHighlight({}, highlight(true, 1.0f)); }),
              "K6: an enabled highlight with no merged scene throws");
    }
    else
    {
        check(false, "K6: fresh renderer initialize");
    }

    // K8: the merged-mask boundary. The last Gaussian of A2 (index 5002) and the first of
    // B (merged index 5003) are the only ones selected, by a tiny box at each position.
    edit.clearMask(maskA2);
    edit.clearMask(maskB);
    const glm::vec3 lastA2(a2->positions()[kCountA2 - 1]);
    const glm::vec3 firstB(b->positions()[0]);
    edit.select(*a2, id, boxQuery(lastA2, identityRotation(), glm::vec3(1e-4f), SelectOp::Replace),
                maskA2);
    edit.select(*b, id, boxQuery(firstB, identityRotation(), glm::vec3(1e-4f), SelectOp::Replace),
                maskB);
    check(edit.countSelected(maskA2, static_cast<uint32_t>(kCountA2)) == 1,
          "K8: only the last Gaussian of A2 (merged index 5002) is selected");
    check(edit.countSelected(maskB, static_cast<uint32_t>(kCountB)) == 1,
          "K8: only the first Gaussian of B (merged index 5003) is selected");
    r.setSelectionHighlight(views, highlight(true, 1.0f));
    const std::vector<float> boundaryFrame = renderViewer(r);
    check(boundaryFrame.size() == base.size(), "K8: the merged boundary frame renders");
}

} // namespace

void checkSelection(Device & device)
{
    runSelectionSection(device);
}

void checkHighlight(Device &)
{
    Renderer r(kWidth, kHeight);
    if (!r.initialize({}, true, testShaderDir()))
    {
        check(false, "K: renderer initialize");
        return;
    }
    Device & device = r.sabercatDevice();
    const std::shared_ptr<GaussianCloud> a2 =
        makeSynthetic(device, SyntheticSpec{kCountA2, 3, true, true, 0xA2A2002u});
    const std::shared_ptr<GaussianCloud> b =
        makeSynthetic(device, SyntheticSpec{kCountB, 1, false, true, 0xB0B0003u});
    const LayerTransform id = identityTransform();
    const std::vector<MergeLayer> layers = {{a2.get(), id}, {b.get(), id}};
    r.setCamera(framingCamera());
    r.setMaxSHDegree(3);
    r.mergeLayersOnGpu(layers);
    checkHighlightFrames(r, device, layers, a2, b);
}

} // namespace edit_test
