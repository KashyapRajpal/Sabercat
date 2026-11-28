// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Section F of edit_test: EditPass::mergeLayers against the host mergeLayers() of
// the same layers (every stream, with the Section F tolerances), planLayerMerge's
// layout per case, and the mergeLayers refusals. Device-level: no Renderer.

#include "edit_test_common.h"

#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "sabercat/device.h"
#include "sabercat/edit_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/layer_transform.h"

#include "../common/shader_dir.h"

using namespace sabercat;
using namespace edit_test;

namespace edit_test
{
namespace
{

// Layer sizes: A and A2 carry all four relight streams, B none.
constexpr size_t kCountA = 100003;
constexpr size_t kCountA2 = 5003;
constexpr size_t kCountB = 4099;

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

struct MergeCase
{
    const char * name;
    std::vector<MergeLayer> layers;
    bool identity;     // every layer at identity: bit-exact in every stream
    size_t count;      // expected count
    uint32_t shDegree; // expected max degree
    bool antialiased;  // expected AND of the layers
    bool relight;      // expected: all four relight streams (true) or none (false)
};

// Checks planLayerMerge's layout for one case, then runs EditPass::mergeLayers into a
// destination of that layout and compares every stream with the host reference.
void runMergeCase(Device & device, EditPass & editPass, const MergeCase & c)
{
    const std::string label = std::string("F ") + c.name;
    const std::shared_ptr<GaussianCloud> ref = sabercat::mergeLayers(device, c.layers);
    const MergedLayout layout = planLayerMerge(c.layers);

    check(layout.count == c.count && ref->count() == c.count,
          label + ": planLayerMerge and host count are " + std::to_string(layout.count));
    check(layout.shDegree == c.shDegree && ref->shDegree() == c.shDegree,
          label + ": shDegree is " + std::to_string(layout.shDegree));
    check(layout.antialiased == c.antialiased && ref->antialiased() == c.antialiased,
          label + ": antialiased is " + std::string(layout.antialiased ? "true" : "false"));
    const RelightPresence & p = layout.presence;
    const bool allRelight = p.normals && p.baseColorRoughness && p.metallic && p.visibility;
    const bool noRelight = !p.normals && !p.baseColorRoughness && !p.metallic && !p.visibility;
    check((c.relight && allRelight) || (!c.relight && noRelight),
          label + ": relight presence as expected");

    GaussianCloud dest(device);
    dest.allocateGPU(layout.count, 0, layout.presence);
    dest.setShDegree(layout.shDegree);
    editPass.mergeLayers(c.layers, dest);
    dest.downloadFromGPU();

    compareMergedStreams(layerSpans(c.layers), copyStreams(*ref), copyStreams(dest), c.identity,
                         label);
}

} // namespace

void checkMergeLayers(Device & device)
{
    const auto a = makeSynthetic(device, SyntheticSpec{kCountA, 3, true, false, 0xA0A0001u});
    const auto a2 = makeSynthetic(device, SyntheticSpec{kCountA2, 3, true, true, 0xA2A2002u});
    const auto b = makeSynthetic(device, SyntheticSpec{kCountB, 1, false, true, 0xB0B0003u});
    const GaussianCloud * A = a.get();
    const GaussianCloud * A2 = a2.get();
    const GaussianCloud * B = b.get();
    const LayerTransform id = identityTransform();
    const LayerTransform t1 = transformT1();
    const LayerTransform t2 = transformT2();

    EditPass editPass(device, testShaderDir());

    const std::vector<MergeCase> cases = {
        {"(1) {A:id}", std::vector<MergeLayer>{{A, id}}, true, kCountA, 3, false, true},
        {"(2) {A:id, A2:id}", std::vector<MergeLayer>{{A, id}, {A2, id}}, true,
         kCountA + kCountA2, 3, false, true},
        {"(3) {A:T1}", std::vector<MergeLayer>{{A, t1}}, false, kCountA, 3, false, true},
        {"(4) {A:T1, A2:T2}", std::vector<MergeLayer>{{A, t1}, {A2, t2}}, false,
         kCountA + kCountA2, 3, false, true},
        {"(5) {B:T2}", std::vector<MergeLayer>{{B, t2}}, false, kCountB, 1, true, false},
        {"(6) {A:T1, B:T2}", std::vector<MergeLayer>{{A, t1}, {B, t2}}, false,
         kCountA + kCountB, 3, false, false},
        {"(7) {B:id, A:id}", std::vector<MergeLayer>{{B, id}, {A, id}}, true,
         kCountA + kCountB, 3, false, false},
        {"(8) {B:T1, A2:id, A:T2}", std::vector<MergeLayer>{{B, t1}, {A2, id}, {A, t2}}, false,
         kCountB + kCountA2 + kCountA, 3, false, false},
    };

    for (const MergeCase & c : cases)
    {
        try
        {
            runMergeCase(device, editPass, c);
        }
        catch (const std::exception & e)
        {
            check(false, std::string("F ") + c.name + " threw: " + e.what());
        }
    }

    // Refusals: each throws before any GPU work.
    const std::vector<MergeLayer> single{{A, id}};
    {
        GaussianCloud wrongCount(device);
        wrongCount.allocateGPU(kCountA + 1, 0, RelightPresence{});
        check(throwsException([&] { editPass.mergeLayers(single, wrongCount); }),
              "F refusal: destination count mismatch throws");
    }
    {
        GaussianCloud self(device);
        self.allocateGPU(kCountA2, 0, RelightPresence{});
        const std::vector<MergeLayer> selfLayer{{&self, id}};
        check(throwsException([&] { editPass.mergeLayers(selfLayer, self); }),
              "F refusal: a layer that is the destination throws");
    }
    {
        GaussianCloud dest(device);
        dest.allocateGPU(1, 0, RelightPresence{});
        check(throwsException([&] { editPass.mergeLayers(std::vector<MergeLayer>{}, dest); }),
              "F refusal: an empty layer list throws");
    }
}

} // namespace edit_test
