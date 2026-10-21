// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sabercat Studio's document model (apps/studio/studio_state.h), without the
// renderer: the app-state JSON and the layer list. Layers here carry no cloud,
// so clouds (which need a Vulkan device) are covered by the GPU selftest.
#include "unit_test.h"

#include "studio_state.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <initializer_list>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

namespace sabercat::unit
{

namespace
{

using studio::StudioState;

// Holds std::cerr (where Logger::warning writes) so a test can count the lines
// the code logged. Restores the stream when it goes out of scope.
class CerrCapture
{
  public:
    CerrCapture() : m_saved(std::cerr.rdbuf(m_text.rdbuf())) {}
    ~CerrCapture()
    {
        std::cerr.rdbuf(m_saved);
    }
    CerrCapture(const CerrCapture &) = delete;
    CerrCapture & operator=(const CerrCapture &) = delete;

    // Lines that contain `needle`.
    int count(const std::string & needle) const
    {
        std::istringstream lines(m_text.str());
        std::string line;
        int matches = 0;
        while (std::getline(lines, line))
        {
            matches += line.find(needle) != std::string::npos ? 1 : 0;
        }
        return matches;
    }

  private:
    std::ostringstream m_text;
    std::streambuf * m_saved;
};

// Applies app-state text to a fresh state; `warnings` is the count of
// "Studio: app state" lines it logged, and the state's own app-state warnings
// (the ones every case checks) are all prefixed that way.
StudioState loadAppState(const std::string & text, int & warnings)
{
    StudioState state;
    CerrCapture capture;
    studio::applyAppStateJson(state, text);
    warnings = capture.count("Studio: app state");
    return state;
}

const std::string kDefaultAppState = studio::buildAppStateJson(StudioState{});

// --- App state ------------------------------------------------------------

void testAppStateRoundTrip(Context & ctx)
{
    StudioState authored;
    authored.camera.pivot = glm::vec3(0.5f, -0.25f, 2.0f);
    authored.camera.distance = 7.5f;
    authored.camera.orientation = glm::normalize(glm::angleAxis(0.7f, glm::normalize(glm::vec3(0.3f, 1.0f, 0.2f))));
    authored.shDegree = 2;
    authored.debugViewIndex = 1;
    authored.relight.enabled = true;
    authored.relight.output = 3;
    authored.relight.exposure = 1.25f;
    authored.relight.environmentPreset = 1;
    authored.relight.environmentIntensity = 2.5f;
    authored.relight.shadowMode = 2;
    authored.relight.shadowMapResolution = 2048;
    authored.relight.raySamples = 8;
    authored.relight.rtQuality = 1;
    authored.relight.progressiveShadows = false;

    const std::string text = studio::buildAppStateJson(authored);
    int warnings = -1;
    const StudioState restored = loadAppState(text, warnings);
    UNIT_CHECK(ctx, warnings == 0);
    UNIT_CHECK(ctx, restored.camera.pivot == authored.camera.pivot);
    UNIT_CHECK(ctx, restored.camera.distance == authored.camera.distance);
    UNIT_CHECK(ctx, restored.camera.orientation == authored.camera.orientation);
    UNIT_CHECK(ctx, restored.shDegree == 2 && restored.debugViewIndex == 1);
    UNIT_CHECK(ctx, restored.relight.enabled && restored.relight.output == 3);
    UNIT_CHECK(ctx, restored.relight.exposure == 1.25f && restored.relight.environmentIntensity == 2.5f);
    UNIT_CHECK(ctx, restored.relight.shadowMode == 2 && restored.relight.shadowMapResolution == 2048);
    UNIT_CHECK(ctx, restored.relight.raySamples == 8 && restored.relight.rtQuality == 1);
    UNIT_CHECK(ctx, !restored.relight.progressiveShadows);
    // Save, open, save again: the same bytes.
    UNIT_CHECK(ctx, studio::buildAppStateJson(restored) == text);
}

void testMissingKeysTakeDefaults(Context & ctx)
{
    int warnings = -1;
    const StudioState empty = loadAppState("{}", warnings);
    UNIT_CHECK(ctx, warnings == 0); // a missing key is normal in an older file
    UNIT_CHECK(ctx, studio::buildAppStateJson(empty) == kDefaultAppState);
}

void testUnknownKeysAreIgnored(Context & ctx)
{
    int warnings = -1;
    const StudioState state = loadAppState(
        R"({"version":1,"future":{"x":[1,2]},"shDegree":2,"camera":{"extra":true},"relight":{"newKnob":5}})",
        warnings);
    UNIT_CHECK(ctx, warnings == 0);
    UNIT_CHECK(ctx, state.shDegree == 2);
}

// A value of the wrong type or shape takes its default, one line each.
void testWrongTypesTakeDefaults(Context & ctx)
{
    struct Case
    {
        const char * json;
        std::string expected; // app state the case must produce
    };
    const StudioState defaults;
    const std::string defaultState = studio::buildAppStateJson(defaults);
    const Case cases[] = {
        {R"({"shDegree":"two"})", defaultState},
        {R"({"shDegree":2.5})", defaultState},
        {R"({"camera":5})", defaultState},
        {R"({"camera":{"pivot":[1,2]}})", defaultState},
        {R"({"camera":{"pivot":[1,"a",3]}})", defaultState},
        {R"({"camera":{"distance":"NaN"}})", defaultState},
        {R"({"camera":{"orientation":[1,0,0]}})", defaultState},
        {R"({"camera":{"orientation":[0,0,0,0]}})", defaultState},
        {R"({"relight":{"enabled":"yes"}})", defaultState},
        {R"({"relight":{"exposure":true}})", defaultState},
        {R"({"relight":[]})", defaultState},
    };
    for (const Case & c : cases)
    {
        int warnings = -1;
        const StudioState state = loadAppState(c.json, warnings);
        UNIT_CHECK(ctx, warnings == 1);
        UNIT_CHECK(ctx, studio::buildAppStateJson(state) == c.expected);
        if (warnings != 1)
        {
            std::cerr << "  case: " << c.json << std::endl;
        }
    }
}

// Text that is not a JSON object: one line, all defaults.
void testNonObjectAndInvalidJson(Context & ctx)
{
    const char * texts[] = {"[1,2]", "42", "\"text\"", "{", ""};
    for (const char * text : texts)
    {
        int warnings = -1;
        const StudioState state = loadAppState(text, warnings);
        UNIT_CHECK(ctx, warnings == 1);
        UNIT_CHECK(ctx, studio::buildAppStateJson(state) == kDefaultAppState);
    }
}

// The version key: newer versions still load their known keys (one line); a
// version below 1 is clamped to 1 (one line); a non-integer takes the default.
void testVersion(Context & ctx)
{
    int warnings = -1;
    StudioState state = loadAppState(R"({"version":2,"shDegree":1})", warnings);
    UNIT_CHECK(ctx, warnings == 1);
    UNIT_CHECK(ctx, state.shDegree == 1);

    state = loadAppState(R"({"version":0,"shDegree":1})", warnings);
    UNIT_CHECK(ctx, warnings == 1);
    UNIT_CHECK(ctx, state.shDegree == 1);

    state = loadAppState(R"({"version":"x","shDegree":1})", warnings);
    UNIT_CHECK(ctx, warnings == 1);
    UNIT_CHECK(ctx, state.shDegree == 1);

    loadAppState(R"({"version":1})", warnings);
    UNIT_CHECK(ctx, warnings == 0);
}

// One out-of-range value per clamp: it lands on the nearest end of the range
// the panel offers and is logged exactly once.
void testEveryClamp(Context & ctx)
{
    struct Case
    {
        const char * json;
        float expectedFloat; // compared where the field is a float
        int expectedInt;     // compared where the field is an int
        int field;           // 0 shDegree, 1 debugView, 2 output, 3 exposure, 4 envPreset,
                             // 5 envIntensity, 6 shadowMode, 7 shadowMapRes, 8 raySamples,
                             // 9 rtQuality, 10 distance
    };
    const float kMinDistance = app::OrbitCamera::kMinDistance;
    const Case cases[] = {
        {R"({"shDegree":9})", 0.0f, 3, 0},
        {R"({"shDegree":-1})", 0.0f, 0, 0},
        {R"({"debugView":99})", 0.0f, 4, 1},
        {R"({"debugView":-1})", 0.0f, 0, 1},
        {R"({"relight":{"output":8}})", 0.0f, 7, 2},
        {R"({"relight":{"output":-1}})", 0.0f, 0, 2},
        {R"({"relight":{"exposure":100}})", 8.0f, 0, 3},
        {R"({"relight":{"exposure":0}})", 0.05f, 0, 3},
        {R"({"relight":{"environmentPreset":5}})", 0.0f, 2, 4},
        {R"({"relight":{"environmentPreset":-1}})", 0.0f, 0, 4},
        {R"({"relight":{"environmentIntensity":9}})", 4.0f, 0, 5},
        {R"({"relight":{"environmentIntensity":-1}})", 0.0f, 0, 5},
        {R"({"relight":{"shadowMode":-2}})", 0.0f, 0, 6},
        {R"({"relight":{"shadowMode":4}})", 0.0f, 3, 6},
        {R"({"relight":{"shadowMapResolution":10}})", 0.0f, 256, 7},
        {R"({"relight":{"shadowMapResolution":100000}})", 0.0f, 4096, 7},
        {R"({"relight":{"raySamples":0}})", 0.0f, 1, 8},
        {R"({"relight":{"raySamples":99}})", 0.0f, 16, 8},
        {R"({"relight":{"rtQuality":7}})", 0.0f, 2, 9},
        {R"({"camera":{"distance":0}})", kMinDistance, 0, 10},
        {R"({"camera":{"distance":-3}})", kMinDistance, 0, 10},
    };
    for (const Case & c : cases)
    {
        int warnings = -1;
        const StudioState s = loadAppState(c.json, warnings);
        UNIT_CHECK(ctx, warnings == 1);
        if (warnings != 1)
        {
            std::cerr << "  case: " << c.json << std::endl;
        }
        switch (c.field)
        {
        case 0:
            UNIT_CHECK(ctx, s.shDegree == c.expectedInt);
            break;
        case 1:
            UNIT_CHECK(ctx, s.debugViewIndex == c.expectedInt);
            break;
        case 2:
            UNIT_CHECK(ctx, s.relight.output == c.expectedInt);
            break;
        case 3:
            UNIT_CHECK_NEAR(ctx, s.relight.exposure, c.expectedFloat, 1e-6);
            break;
        case 4:
            UNIT_CHECK(ctx, s.relight.environmentPreset == c.expectedInt);
            break;
        case 5:
            UNIT_CHECK_NEAR(ctx, s.relight.environmentIntensity, c.expectedFloat, 1e-6);
            break;
        case 6:
            UNIT_CHECK(ctx, s.relight.shadowMode == c.expectedInt);
            break;
        case 7:
            UNIT_CHECK(ctx, s.relight.shadowMapResolution == c.expectedInt);
            break;
        case 8:
            UNIT_CHECK(ctx, s.relight.raySamples == c.expectedInt);
            break;
        case 9:
            UNIT_CHECK(ctx, s.relight.rtQuality == c.expectedInt);
            break;
        case 10:
            UNIT_CHECK_NEAR(ctx, s.camera.distance, c.expectedFloat, 0.0);
            break;
        }
    }

    // The ends of each range are valid as they are: no clamp, no line.
    int warnings = -1;
    const StudioState edges = loadAppState(
        R"({"shDegree":0,"debugView":4,"relight":{"output":7,"exposure":8,"environmentPreset":2,)"
        R"("environmentIntensity":0,"shadowMode":3,"shadowMapResolution":4096,"raySamples":1,"rtQuality":2}})",
        warnings);
    UNIT_CHECK(ctx, warnings == 0);
    UNIT_CHECK(ctx, edges.shDegree == 0 && edges.debugViewIndex == 4 && edges.relight.output == 7);
    UNIT_CHECK(ctx, edges.relight.exposure == 8.0f && edges.relight.shadowMapResolution == 4096);
    UNIT_CHECK(ctx, edges.relight.raySamples == 1 && edges.relight.rtQuality == 2);
}

// A quaternion that is not unit length is normalised (one line); the unit one
// is kept as written, with no line.
void testOrientationNormalised(Context & ctx)
{
    int warnings = -1;
    StudioState s = loadAppState(R"({"camera":{"orientation":[2,0,0,0]}})", warnings);
    UNIT_CHECK(ctx, warnings == 1);
    UNIT_CHECK(ctx, s.camera.orientation == glm::quat(1.0f, 0.0f, 0.0f, 0.0f));

    s = loadAppState(R"({"camera":{"orientation":[1,0,0,0]}})", warnings);
    UNIT_CHECK(ctx, warnings == 0);
    UNIT_CHECK(ctx, s.camera.orientation == glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
}

// --- Layer model ----------------------------------------------------------

StudioState withLayers(std::initializer_list<const char *> ids)
{
    StudioState state;
    for (const char * id : ids)
    {
        ProjectLayer layer;
        layer.id = id;
        layer.name = std::string("layer ") + id;
        state.project.layers.push_back(layer);
    }
    return state;
}

std::vector<std::string> idsOf(const StudioState & state)
{
    std::vector<std::string> ids;
    for (const ProjectLayer & layer : state.project.layers)
    {
        ids.push_back(layer.id);
    }
    return ids;
}

void testRemove(Context & ctx)
{
    StudioState s = withLayers({"a", "b", "c"});
    s.selectedLayer = 1;
    UNIT_CHECK(ctx, removeSelectedLayer(s));
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"a", "c"}));
    UNIT_CHECK(ctx, s.selectedLayer == 1); // the layer that followed is now selected
    UNIT_CHECK(ctx, s.dirty && s.sceneDirty);

    s = withLayers({"a", "b", "c"});
    s.selectedLayer = 2;
    UNIT_CHECK(ctx, removeSelectedLayer(s));
    UNIT_CHECK(ctx, s.selectedLayer == 1); // the last layer went: select the new last

    s = withLayers({"only"});
    s.selectedLayer = 0;
    UNIT_CHECK(ctx, removeSelectedLayer(s));
    UNIT_CHECK(ctx, s.project.layers.empty() && s.selectedLayer == -1);

    // Locked, unselected and out-of-range removals change nothing and leave the flags clean.
    s = withLayers({"a", "b"});
    s.project.layers[1].locked = true;
    s.selectedLayer = 1;
    s.dirty = false;
    s.sceneDirty = false;
    UNIT_CHECK(ctx, !removeSelectedLayer(s));
    UNIT_CHECK(ctx, s.project.layers.size() == 2 && !s.dirty && !s.sceneDirty);
    s.selectedLayer = -1;
    UNIT_CHECK(ctx, !removeSelectedLayer(s));
    s.selectedLayer = 2;
    UNIT_CHECK(ctx, !removeSelectedLayer(s));
    UNIT_CHECK(ctx, s.project.layers.size() == 2);
}

void testMove(Context & ctx)
{
    StudioState s = withLayers({"a", "b", "c"});
    s.selectedLayer = 0;
    UNIT_CHECK(ctx, !moveSelectedLayer(s, -1)); // off the top
    UNIT_CHECK(ctx, !s.dirty);
    UNIT_CHECK(ctx, moveSelectedLayer(s, +1));
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"b", "a", "c"}) && s.selectedLayer == 1);
    UNIT_CHECK(ctx, s.dirty && s.sceneDirty);

    s.selectedLayer = 2;
    UNIT_CHECK(ctx, !moveSelectedLayer(s, +1)); // off the bottom
    UNIT_CHECK(ctx, moveSelectedLayer(s, -1));
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"b", "c", "a"}) && s.selectedLayer == 1);

    // Only the neighbours: any other delta, including 0, is refused.
    s = withLayers({"a", "b", "c"});
    s.selectedLayer = 0;
    UNIT_CHECK(ctx, !moveSelectedLayer(s, 2));
    UNIT_CHECK(ctx, !moveSelectedLayer(s, 0));
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"a", "b", "c"}) && s.selectedLayer == 0);

    // A locked layer does not move, and a locked neighbour is not swapped past.
    s = withLayers({"a", "b"});
    s.project.layers[0].locked = true;
    s.selectedLayer = 0;
    UNIT_CHECK(ctx, !moveSelectedLayer(s, +1));
    s.selectedLayer = 1;
    UNIT_CHECK(ctx, !moveSelectedLayer(s, -1));
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"a", "b"}));
    s.project.layers[0].locked = false;
    UNIT_CHECK(ctx, moveSelectedLayer(s, -1));
}

void testNextLayerId(Context & ctx)
{
    UNIT_CHECK(ctx, studio::nextLayerId(Project{}) == "0");

    Project p;
    auto addId = [&p](const std::string & id) {
        ProjectLayer layer;
        layer.id = id;
        p.layers.push_back(layer);
    };
    addId("0");
    addId("1");
    addId("7");
    UNIT_CHECK(ctx, studio::nextLayerId(p) == "8");

    // Gaps, non-numeric and signed or suffixed ids: only plain decimal ids count.
    p = Project{};
    for (const char * id : {"3", "x", "12", "-1", "7abc", ""})
    {
        addId(id);
    }
    UNIT_CHECK(ctx, studio::nextLayerId(p) == "13");

    // Leading zeros are still decimal.
    p = Project{};
    addId("007");
    UNIT_CHECK(ctx, studio::nextLayerId(p) == "8");

    // The largest 64-bit id: one more overflows, so the smallest unused id is
    // taken instead, and never one already in use.
    p = Project{};
    addId("18446744073709551615");
    UNIT_CHECK(ctx, studio::nextLayerId(p) == "0");
    addId("0");
    UNIT_CHECK(ctx, studio::nextLayerId(p) == "1");

    // Too large for 64 bits: not an id this function hands out, so it is ignored.
    p = Project{};
    addId("18446744073709551616");
    UNIT_CHECK(ctx, studio::nextLayerId(p) == "0");
}

void testLayerEditsAndFlags(Context & ctx)
{
    StudioState s = withLayers({"a", "b"});
    s.project.layers[0].locked = true;
    s.dirty = false;
    s.sceneDirty = false;

    // Visibility is settable on a locked layer; setting the same value is no change.
    UNIT_CHECK(ctx, studio::setLayerVisible(s, 0, true));
    UNIT_CHECK(ctx, !s.dirty && !s.sceneDirty);
    UNIT_CHECK(ctx, studio::setLayerVisible(s, 0, false));
    UNIT_CHECK(ctx, !s.project.layers[0].visible && s.dirty && s.sceneDirty);

    // The lock flag marks the project dirty but not the scene.
    s.dirty = false;
    s.sceneDirty = false;
    UNIT_CHECK(ctx, studio::setLayerLocked(s, 1, true));
    UNIT_CHECK(ctx, s.project.layers[1].locked && s.dirty && !s.sceneDirty);
    s.dirty = false;
    UNIT_CHECK(ctx, studio::setLayerLocked(s, 1, true));
    UNIT_CHECK(ctx, !s.dirty);
    UNIT_CHECK(ctx, studio::setLayerLocked(s, 1, false)); // unlocked again for the edits below
    s.dirty = false;

    // Rename and transform are refused while locked, and change nothing.
    s.sceneDirty = false;
    UNIT_CHECK(ctx, !studio::renameLayer(s, 0, "renamed"));
    UNIT_CHECK(ctx, s.project.layers[0].name == "layer a" && !s.dirty);
    LayerTransform moved;
    moved.translation = glm::vec3(1.0f, 2.0f, 3.0f);
    UNIT_CHECK(ctx, !studio::setLayerTransform(s, 0, moved));
    UNIT_CHECK(ctx, s.project.layers[0].transform.translation == glm::vec3(0.0f) && !s.sceneDirty);

    // Unlocked: rename marks the project dirty only; a transform also the scene.
    UNIT_CHECK(ctx, studio::renameLayer(s, 1, "renamed"));
    UNIT_CHECK(ctx, s.project.layers[1].name == "renamed" && s.dirty && !s.sceneDirty);
    s.dirty = false;
    UNIT_CHECK(ctx, studio::setLayerTransform(s, 1, moved));
    UNIT_CHECK(ctx, s.project.layers[1].transform.translation == glm::vec3(1.0f, 2.0f, 3.0f));
    UNIT_CHECK(ctx, s.dirty && s.sceneDirty);

    // An index out of range is refused by every edit.
    s.dirty = false;
    s.sceneDirty = false;
    UNIT_CHECK(ctx, !studio::setLayerVisible(s, 2, true));
    UNIT_CHECK(ctx, !studio::setLayerLocked(s, -1, false));
    UNIT_CHECK(ctx, !studio::renameLayer(s, 5, "x"));
    UNIT_CHECK(ctx, !studio::setLayerTransform(s, 2, moved));
    UNIT_CHECK(ctx, !s.dirty && !s.sceneDirty);
}

void testVisibleLayersAndNewProject(Context & ctx)
{
    // Layers without a cloud never contribute, visible or not.
    StudioState s = withLayers({"a", "b"});
    UNIT_CHECK(ctx, visibleLayers(s).empty());

    s.selectedLayer = 1;
    s.camera.distance = 9.0f;
    s.shDegree = 1;
    s.debugViewIndex = 3;
    s.relight.enabled = true;
    studio::newProject(s);
    UNIT_CHECK(ctx, s.project.layers.empty() && s.selectedLayer == -1);
    UNIT_CHECK(ctx, !s.dirty && s.sceneDirty && s.projectPath.empty());
    UNIT_CHECK(ctx, s.camera.distance == app::OrbitCamera{}.distance);
    UNIT_CHECK(ctx, s.shDegree == 3 && s.debugViewIndex == 0 && !s.relight.enabled);
}

} // namespace

void testAppState(Context & ctx)
{
    testAppStateRoundTrip(ctx);
    testMissingKeysTakeDefaults(ctx);
    testUnknownKeysAreIgnored(ctx);
    testWrongTypesTakeDefaults(ctx);
    testNonObjectAndInvalidJson(ctx);
    testVersion(ctx);
    testEveryClamp(ctx);
    testOrientationNormalised(ctx);
}

void testLayerModel(Context & ctx)
{
    testRemove(ctx);
    testMove(ctx);
    testNextLayerId(ctx);
    testLayerEditsAndFlags(ctx);
    testVisibleLayersAndNewProject(ctx);
}

} // namespace sabercat::unit
