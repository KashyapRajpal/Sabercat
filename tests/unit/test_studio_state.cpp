// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sabercat Studio's document model (apps/studio/studio_state.h), without the
// renderer: the app-state JSON and the layer list. Layers here carry no cloud,
// so clouds (which need a Vulkan device) are covered by the GPU selftest.
#include "unit_test.h"

#include "studio_edits.h"
#include "studio_state.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
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

// Applies app-state text to `state` (a fresh one, or one the caller reuses: every app-state
// field is assigned); `warnings` is the count of "Studio: app state" lines it logged. A
// StudioState is not copyable or movable (studio_state.h), so the caller owns it.
void loadAppState(StudioState & state, const std::string & text, int & warnings)
{
    CerrCapture capture;
    studio::applyAppStateJson(state, text);
    warnings = capture.count("Studio: app state");
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
    StudioState restored;
    loadAppState(restored, text, warnings);
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
    StudioState empty;
    loadAppState(empty, "{}", warnings);
    UNIT_CHECK(ctx, warnings == 0); // a missing key is normal in an older file
    UNIT_CHECK(ctx, studio::buildAppStateJson(empty) == kDefaultAppState);
}

void testUnknownKeysAreIgnored(Context & ctx)
{
    int warnings = -1;
    StudioState state;
    loadAppState(state,
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
        StudioState state;
        loadAppState(state, c.json, warnings);
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
        StudioState state;
        loadAppState(state, text, warnings);
        UNIT_CHECK(ctx, warnings == 1);
        UNIT_CHECK(ctx, studio::buildAppStateJson(state) == kDefaultAppState);
    }
}

// The version key: newer versions still load their known keys (one line); a
// version below 1 is clamped to 1 (one line); a non-integer takes the default.
void testVersion(Context & ctx)
{
    int warnings = -1;
    StudioState state;
    loadAppState(state, R"({"version":2,"shDegree":1})", warnings);
    UNIT_CHECK(ctx, warnings == 1);
    UNIT_CHECK(ctx, state.shDegree == 1);

    loadAppState(state, R"({"version":0,"shDegree":1})", warnings);
    UNIT_CHECK(ctx, warnings == 1);
    UNIT_CHECK(ctx, state.shDegree == 1);

    loadAppState(state, R"({"version":"x","shDegree":1})", warnings);
    UNIT_CHECK(ctx, warnings == 1);
    UNIT_CHECK(ctx, state.shDegree == 1);

    loadAppState(state, R"({"version":1})", warnings);
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
        StudioState s;
        loadAppState(s, c.json, warnings);
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
    StudioState edges;
    loadAppState(edges,
        R"({"shDegree":0,"debugView":4,"relight":{"output":7,"exposure":8,"environmentPreset":2,)"
        R"("environmentIntensity":0,"shadowMode":3,"shadowMapResolution":4096,"raySamples":1,"rtQuality":2}})",
        warnings);
    UNIT_CHECK(ctx, warnings == 0);
    UNIT_CHECK(ctx, edges.shDegree == 0 && edges.debugViewIndex == 4 && edges.relight.output == 7);
    UNIT_CHECK(ctx, edges.relight.exposure == 8.0f && edges.relight.shadowMapResolution == 4096);
    UNIT_CHECK(ctx, edges.relight.raySamples == 1 && edges.relight.rtQuality == 2);
}

// A quaternion that is not unit length is normalized (one line); the unit one
// is kept as written, with no line.
void testOrientationNormalized(Context & ctx)
{
    int warnings = -1;
    StudioState s;
    loadAppState(s, R"({"camera":{"orientation":[2,0,0,0]}})", warnings);
    UNIT_CHECK(ctx, warnings == 1);
    UNIT_CHECK(ctx, s.camera.orientation == glm::quat(1.0f, 0.0f, 0.0f, 0.0f));

    loadAppState(s, R"({"camera":{"orientation":[1,0,0,0]}})", warnings);
    UNIT_CHECK(ctx, warnings == 0);
    UNIT_CHECK(ctx, s.camera.orientation == glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
}

// --- Layer model ----------------------------------------------------------

// Makes `state` a fresh project of layers with these ids, in order (no clouds). A fresh
// project is newProject()'s: the same defaults as a new StudioState, and GPU-free.
void withLayers(StudioState & state, std::initializer_list<const char *> ids)
{
    studio::newProject(state);
    for (const char * id : ids)
    {
        ProjectLayer layer;
        layer.id = id;
        layer.name = std::string("layer ") + id;
        state.project.layers.push_back(layer);
    }
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
    StudioState s;
    withLayers(s, {"a", "b", "c"});
    s.selectedLayer = 1;
    UNIT_CHECK(ctx, removeSelectedLayer(s));
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"a", "c"}));
    UNIT_CHECK(ctx, s.selectedLayer == 1); // the layer that followed is now selected
    UNIT_CHECK(ctx, s.dirty && s.sceneDirty);

    withLayers(s, {"a", "b", "c"});
    s.selectedLayer = 2;
    UNIT_CHECK(ctx, removeSelectedLayer(s));
    UNIT_CHECK(ctx, s.selectedLayer == 1); // the last layer went: select the new last

    withLayers(s, {"only"});
    s.selectedLayer = 0;
    UNIT_CHECK(ctx, removeSelectedLayer(s));
    UNIT_CHECK(ctx, s.project.layers.empty() && s.selectedLayer == -1);

    // Locked, unselected and out-of-range removals change nothing and leave the flags clean.
    withLayers(s, {"a", "b"});
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
    StudioState s;
    withLayers(s, {"a", "b", "c"});
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

    // Only the neighbors: any other delta, including 0, is refused.
    withLayers(s, {"a", "b", "c"});
    s.selectedLayer = 0;
    UNIT_CHECK(ctx, !moveSelectedLayer(s, 2));
    UNIT_CHECK(ctx, !moveSelectedLayer(s, 0));
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"a", "b", "c"}) && s.selectedLayer == 0);

    // A locked layer does not move, and a locked neighbor is not swapped past.
    withLayers(s, {"a", "b"});
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
    StudioState s;
    withLayers(s, {"a", "b"});
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
    StudioState s;
    withLayers(s, {"a", "b"});
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

// --- Undo of layer operations ---------------------------------------------

// The transform's components as bits: -0 and +0 differ, as they should for an undo.
bool sameTransform(const LayerTransform & a, const LayerTransform & b)
{
    const float fa[8] = {a.rotation.w, a.rotation.x, a.rotation.y, a.rotation.z,
                         a.translation.x, a.translation.y, a.translation.z, a.scale};
    const float fb[8] = {b.rotation.w, b.rotation.x, b.rotation.y, b.rotation.z,
                         b.translation.x, b.translation.y, b.translation.z, b.scale};
    return std::memcmp(fa, fb, sizeof fa) == 0;
}

// A transform with distinct, non-trivial values.
LayerTransform sampleTransform(float seed)
{
    LayerTransform t;
    t.rotation = glm::normalize(glm::angleAxis(0.3f * seed, glm::normalize(glm::vec3(0.2f, 1.0f, -0.4f))));
    t.translation = glm::vec3(0.1f * seed, -0.0f, 3.25f + seed);
    t.scale = 1.5f * seed;
    return t;
}

// The project's layer list and selection, as the undo checks compare them.
struct LayerSnapshot
{
    std::string id;
    std::string name;
    bool visible;
    bool locked;
    LayerTransform transform;
};

struct ProjectSnapshot
{
    std::vector<LayerSnapshot> layers;
    int selectedLayer;
};

ProjectSnapshot snapshotOf(const StudioState & state)
{
    ProjectSnapshot snap;
    for (const ProjectLayer & layer : state.project.layers)
    {
        snap.layers.push_back(LayerSnapshot{layer.id, layer.name, layer.visible, layer.locked, layer.transform});
    }
    snap.selectedLayer = state.selectedLayer;
    return snap;
}

bool sameSnapshot(const ProjectSnapshot & a, const ProjectSnapshot & b)
{
    if (a.selectedLayer != b.selectedLayer || a.layers.size() != b.layers.size())
    {
        return false;
    }
    for (size_t i = 0; i < a.layers.size(); ++i)
    {
        const LayerSnapshot & x = a.layers[i];
        const LayerSnapshot & y = b.layers[i];
        if (x.id != y.id || x.name != y.name || x.visible != y.visible || x.locked != y.locked
            || !sameTransform(x.transform, y.transform))
        {
            return false;
        }
    }
    return true;
}

// Rename, visibility and the Inspector's transform: each undo restores the exact previous
// value, each redo the new one, and both flags are set. A drag that only calls
// setLayerTransform records nothing until commitLayerTransformEdit().
void testRenameVisibilityTransformUndo(Context & ctx)
{
    StudioState s;
    withLayers(s, {"a", "b"});
    s.selectedLayer = 0;

    UNIT_CHECK(ctx, studio::renameLayer(s, 0, "alpha"));
    UNIT_CHECK(ctx, s.edit.history.size() == 1 && s.edit.history.cursor() == 1);
    s.dirty = false;
    s.sceneDirty = false;
    UNIT_CHECK(ctx, s.edit.history.undo());
    UNIT_CHECK(ctx, s.project.layers[0].name == "layer a");
    UNIT_CHECK(ctx, s.dirty && s.sceneDirty && s.edit.history.cursor() == 0);
    s.dirty = false;
    s.sceneDirty = false;
    UNIT_CHECK(ctx, s.edit.history.redo());
    UNIT_CHECK(ctx, s.project.layers[0].name == "alpha");
    UNIT_CHECK(ctx, s.dirty && s.sceneDirty && s.edit.history.cursor() == 1);
    UNIT_CHECK(ctx, s.edit.history.label(0) == "Rename layer");

    UNIT_CHECK(ctx, studio::setLayerVisible(s, 1, false));
    UNIT_CHECK(ctx, s.edit.history.size() == 2 && s.edit.history.label(1) == "Hide layer layer b");
    UNIT_CHECK(ctx, s.edit.history.undo() && s.project.layers[1].visible);
    UNIT_CHECK(ctx, s.edit.history.redo() && !s.project.layers[1].visible);

    // The drag: every frame sets the transform (unrecorded), then the drag's end commits it.
    const LayerTransform start = s.project.layers[0].transform;
    const LayerTransform edited = sampleTransform(2.0f);
    UNIT_CHECK(ctx, studio::setLayerTransform(s, 0, sampleTransform(1.0f)));
    UNIT_CHECK(ctx, studio::setLayerTransform(s, 0, edited));
    UNIT_CHECK(ctx, s.edit.history.size() == 2);
    UNIT_CHECK(ctx, studio::commitLayerTransformEdit(s, 0, start));
    UNIT_CHECK(ctx, s.edit.history.size() == 3 && s.edit.history.cursor() == 3);
    UNIT_CHECK(ctx, s.edit.history.label(2) == "Layer transform alpha");
    s.dirty = false;
    s.sceneDirty = false;
    UNIT_CHECK(ctx, s.edit.history.undo());
    UNIT_CHECK(ctx, sameTransform(s.project.layers[0].transform, start));
    UNIT_CHECK(ctx, s.dirty && s.sceneDirty);
    UNIT_CHECK(ctx, s.edit.history.redo());
    UNIT_CHECK(ctx, sameTransform(s.project.layers[0].transform, edited));

    // An unchanged transform records nothing; -0 where +0 was is a change, bit for bit.
    UNIT_CHECK(ctx, !studio::commitLayerTransformEdit(s, 0, edited));
    UNIT_CHECK(ctx, s.edit.history.size() == 3);
    LayerTransform negativeZero;
    negativeZero.translation.x = -0.0f;
    UNIT_CHECK(ctx, studio::setLayerTransform(s, 1, negativeZero));
    UNIT_CHECK(ctx, studio::commitLayerTransformEdit(s, 1, LayerTransform{}));
    UNIT_CHECK(ctx, s.edit.history.size() == 4);
    UNIT_CHECK(ctx, s.edit.history.undo() && sameTransform(s.project.layers[1].transform, LayerTransform{}));
    UNIT_CHECK(ctx, s.edit.history.redo() && sameTransform(s.project.layers[1].transform, negativeZero));

    // Out of range, and while a command replays: nothing is pushed.
    UNIT_CHECK(ctx, !studio::commitLayerTransformEdit(s, 7, start));
    s.edit.replaying = true;
    UNIT_CHECK(ctx, !studio::commitLayerTransformEdit(s, 0, start));
    s.edit.replaying = false;
    UNIT_CHECK(ctx, s.edit.history.size() == 4);

    // A rename to the name the layer already has is no edit.
    UNIT_CHECK(ctx, studio::renameLayer(s, 1, "layer b"));
    UNIT_CHECK(ctx, s.edit.history.size() == 4);
}

// Removing the selected layer: undo puts it back at the same index with the same fields
// and selection, and with its edit state (the selection mask's bookkeeping); redo removes it.
void testRemoveUndoRedo(Context & ctx)
{
    StudioState s;
    withLayers(s, {"a", "b", "c"});
    s.project.layers[1].name = "bee";
    s.project.layers[1].visible = false;
    s.project.layers[1].transform = sampleTransform(3.0f);
    studio::LayerEditState & bEdit = s.edit.layers["b"];
    bEdit.selectedCount = 7;
    bEdit.hostStale = true;
    s.selectedLayer = 1;
    const ProjectSnapshot before = snapshotOf(s);

    UNIT_CHECK(ctx, removeSelectedLayer(s));
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"a", "c"}) && s.selectedLayer == 1);
    UNIT_CHECK(ctx, s.edit.layers.count("b") == 0);
    UNIT_CHECK(ctx, s.edit.history.size() == 1 && s.edit.history.label(0) == "Remove layer bee");

    s.dirty = false;
    s.sceneDirty = false;
    UNIT_CHECK(ctx, s.edit.history.undo());
    UNIT_CHECK(ctx, sameSnapshot(snapshotOf(s), before));
    UNIT_CHECK(ctx, s.dirty && s.sceneDirty);
    UNIT_CHECK(ctx, s.edit.layers.count("b") == 1 && s.edit.layers.at("b").selectedCount == 7
                        && s.edit.layers.at("b").hostStale);

    UNIT_CHECK(ctx, s.edit.history.redo());
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"a", "c"}) && s.selectedLayer == 1);
    UNIT_CHECK(ctx, s.edit.layers.count("b") == 0);
    UNIT_CHECK(ctx, s.edit.history.undo() && sameSnapshot(snapshotOf(s), before));
}

// Moving up, then undo restores the order and the selection; redo moves it again.
void testMoveUndoRedo(Context & ctx)
{
    StudioState s;
    withLayers(s, {"a", "b", "c"});
    s.selectedLayer = 0;
    const ProjectSnapshot start = snapshotOf(s);
    UNIT_CHECK(ctx, moveSelectedLayer(s, +1));
    const ProjectSnapshot moved = snapshotOf(s);
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"b", "a", "c"}));

    UNIT_CHECK(ctx, s.edit.history.undo());
    UNIT_CHECK(ctx, sameSnapshot(snapshotOf(s), start));
    UNIT_CHECK(ctx, s.edit.history.redo());
    UNIT_CHECK(ctx, sameSnapshot(snapshotOf(s), moved));
    UNIT_CHECK(ctx, s.edit.history.label(0) == "Move layer layer a");
}

// Adding a layer (the command as addLayerFromFile pushes it, without a cloud): undo takes
// it out and restores the selection, redo puts it back at its index.
void testAddUndoRedo(Context & ctx)
{
    StudioState s;
    withLayers(s, {"a"});
    s.selectedLayer = 0;
    ProjectLayer added;
    added.id = "b";
    added.name = "layer b";
    s.project.layers.push_back(added);
    s.selectedLayer = 1;
    s.edit.history.push(std::make_unique<studio::AddLayerCommand>(s, s.project.layers.back(), 1, 0, 1));
    UNIT_CHECK(ctx, s.edit.history.label(0) == "Add layer layer b");

    UNIT_CHECK(ctx, s.edit.history.undo());
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"a"}) && s.selectedLayer == 0);
    UNIT_CHECK(ctx, s.edit.history.redo());
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"a", "b"}) && s.selectedLayer == 1);
    UNIT_CHECK(ctx, s.project.layers[1].name == "layer b");
}

// A lock is not undoable and not recorded; a locked layer refuses a rename and records nothing;
// an undo still restores a layer that is locked now.
void testLockNotRecorded(Context & ctx)
{
    StudioState s;
    withLayers(s, {"a", "b"});
    const size_t none = s.edit.history.size();
    UNIT_CHECK(ctx, studio::setLayerLocked(s, 0, true));
    UNIT_CHECK(ctx, studio::setLayerLocked(s, 0, false));
    UNIT_CHECK(ctx, s.edit.history.size() == none);

    UNIT_CHECK(ctx, studio::renameLayer(s, 0, "alpha"));
    UNIT_CHECK(ctx, s.edit.history.size() == 1);
    UNIT_CHECK(ctx, studio::setLayerLocked(s, 0, true));
    UNIT_CHECK(ctx, s.edit.history.size() == 1);
    UNIT_CHECK(ctx, !studio::renameLayer(s, 0, "beta"));
    UNIT_CHECK(ctx, s.edit.history.size() == 1 && s.project.layers[0].name == "alpha");

    // The undo restores the name of the locked layer, and leaves the lock as it is.
    UNIT_CHECK(ctx, s.edit.history.undo());
    UNIT_CHECK(ctx, s.project.layers[0].name == "layer a" && s.project.layers[0].locked);
}

// Six mixed operations, then jumpTo(0) restores the initial project exactly and jumpTo(6)
// the final one.
void testJumpAcrossLayerOps(Context & ctx)
{
    // The selection is set by hand (as a click in the list would), so it is not part of
    // any command until a command records it: the removal below stores selection 2.
    StudioState s;
    withLayers(s, {"a", "b", "c"});
    s.selectedLayer = 2;
    const ProjectSnapshot initial = snapshotOf(s);

    UNIT_CHECK(ctx, studio::renameLayer(s, 0, "alpha"));                               // 1
    UNIT_CHECK(ctx, studio::setLayerVisible(s, 1, false));                             // 2
    const LayerTransform start = s.project.layers[0].transform;                        // 3
    UNIT_CHECK(ctx, studio::setLayerTransform(s, 0, sampleTransform(1.0f)));
    UNIT_CHECK(ctx, studio::commitLayerTransformEdit(s, 0, start));
    UNIT_CHECK(ctx, removeSelectedLayer(s));                                           // 4
    s.selectedLayer = 0;                                                               // 5
    UNIT_CHECK(ctx, moveSelectedLayer(s, +1));
    ProjectLayer added;                                                                // 6
    added.id = "d";
    added.name = "layer d";
    s.project.layers.push_back(added);
    s.selectedLayer = 2;
    s.edit.history.push(std::make_unique<studio::AddLayerCommand>(s, s.project.layers.back(), 2, 1, 2));
    UNIT_CHECK(ctx, s.edit.history.size() == 6 && s.edit.history.cursor() == 6);
    const ProjectSnapshot last = snapshotOf(s);
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"b", "a", "d"}));

    UNIT_CHECK(ctx, s.edit.history.jumpTo(0));
    UNIT_CHECK(ctx, sameSnapshot(snapshotOf(s), initial) && s.edit.history.cursor() == 0);
    UNIT_CHECK(ctx, s.edit.history.jumpTo(3));
    UNIT_CHECK(ctx, idsOf(s) == (std::vector<std::string>{"a", "b", "c"}) && s.project.layers[0].name == "alpha");
    UNIT_CHECK(ctx, s.edit.history.jumpTo(6));
    UNIT_CHECK(ctx, sameSnapshot(snapshotOf(s), last) && s.edit.history.cursor() == 6);
    UNIT_CHECK(ctx, !s.edit.replaying);
}

// newProject drops the history with the layers.
void testNewProjectClearsHistory(Context & ctx)
{
    StudioState s;
    withLayers(s, {"a"});
    UNIT_CHECK(ctx, studio::renameLayer(s, 0, "x"));
    UNIT_CHECK(ctx, s.edit.history.size() == 1);
    studio::newProject(s);
    UNIT_CHECK(ctx, s.edit.history.size() == 0 && s.edit.history.cursor() == 0);
}

// A replaying command's guard restores the flag on every exit, including an exception. An
// undo whose layer is gone throws, leaves the cursor where it was, and leaves the flag clear.
void testReplayGuard(Context & ctx)
{
    studio::StudioEdit edit;
    UNIT_CHECK(ctx, !edit.replaying);
    {
        studio::ReplayGuard outer(edit);
        UNIT_CHECK(ctx, edit.replaying);
        {
            studio::ReplayGuard inner(edit);
            UNIT_CHECK(ctx, edit.replaying);
        }
        UNIT_CHECK(ctx, edit.replaying);
    }
    UNIT_CHECK(ctx, !edit.replaying);
    try
    {
        studio::ReplayGuard guard(edit);
        throw std::runtime_error("probe");
    }
    catch (const std::runtime_error &)
    {
    }
    UNIT_CHECK(ctx, !edit.replaying);

    StudioState s;
    withLayers(s, {"a", "b"});
    UNIT_CHECK(ctx, studio::renameLayer(s, 0, "x"));
    s.project.layers.erase(s.project.layers.begin()); // the layer the command edits is gone
    bool threw = false;
    try
    {
        s.edit.history.undo();
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    UNIT_CHECK(ctx, threw && s.edit.history.cursor() == 1);
    UNIT_CHECK(ctx, !s.edit.replaying);
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
    testOrientationNormalized(ctx);
}

void testLayerModel(Context & ctx)
{
    testRemove(ctx);
    testMove(ctx);
    testNextLayerId(ctx);
    testLayerEditsAndFlags(ctx);
    testVisibleLayersAndNewProject(ctx);
    testRenameVisibilityTransformUndo(ctx);
    testRemoveUndoRedo(ctx);
    testMoveUndoRedo(ctx);
    testAddUndoRedo(ctx);
    testLockNotRecorded(ctx);
    testJumpAcrossLayerOps(ctx);
    testNewProjectClearsHistory(ctx);
    testReplayGuard(ctx);
}

} // namespace sabercat::unit
