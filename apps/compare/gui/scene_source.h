// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// A splat scene as a compare source. Each scene source owns a headless sabercat::Renderer
// that borrows the window's Vulkan device, renders the scene from the shared camera and
// exposes the result through the MediaSource contract: color, the four-float depth
// layout (depth, coverage, front depth, pad), and the host copies for exports.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "media_source.h"
#include "sabercat/buffer_view.h"
#include "sabercat/camera.h"
#include "sabercat/renderer.h"
#include "sabercat/scene_bounds.h"

namespace sabercat
{
class Device;
} // namespace sabercat

namespace sabercat::compare::gui
{

class GpuContext;

// Per-source render settings. The 4-byte fields come first; the two bools end the struct.
struct SceneSettings
{
    uint32_t width = 1280;
    uint32_t height = 720;
    // Max spherical-harmonic degree: -1 = full (3), otherwise 0..3.
    int32_t shCap = -1;
    DebugView debugView = DebugView::Final;
    bool flipUpAxis = false;
    bool antialias = false;
};

// The device-local bytes that one more pixel of render size costs each scene renderer. It
// is a planning estimate for the budget check of a larger render size, not a measurement of
// any one scene; the real cost is SceneSource::deviceBytes(). Tune it from a measured
// bytes-per-pixel figure (the live perf self-test prints one).
constexpr uint64_t kSceneBytesPerPixelEstimate = 96;

// True when the path names a scene file by extension (case-insensitive): .ply .spz
// .splat .ksplat .sog .gltf .glb, or a file named meta.json (SOG).
bool isSceneFile(const std::string & utf8Path);

class SceneSource final : public MediaSource
{
  public:
    // Creates a headless Renderer(width, height, DepthLayout::DepthCoverageFront) on the
    // host's device and loads utf8Path. The constructor renders nothing: status is Ready
    // once the scene loaded, and rendered() is false until the first render(). A missing
    // file gives status Missing; any failure gives Error with its message. Never throws.
    // The loader's antialias flag is read into antialiasFromFile(); settings.antialias is
    // applied only when applyAntialias is true.
    SceneSource(GpuContext & context, const sabercat::Renderer & host, const std::string & utf8Path,
                const SceneSettings & settings, bool applyAntialias);
    // Shuts the renderer down. The GPU must be idle (the caller's frame is done).
    ~SceneSource() override;

    SceneSource(const SceneSource &) = delete;
    SceneSource & operator=(const SceneSource &) = delete;

    project::SourceKind kind() const override
    {
        return project::SourceKind::Scene;
    }
    SourceStatus status() const override
    {
        return m_status;
    }
    const std::string & statusMessage() const override
    {
        return m_message;
    }
    uint32_t width() const override
    {
        return m_settings.width;
    }
    uint32_t height() const override
    {
        return m_settings.height;
    }
    Encoding encoding() const override
    {
        return Encoding::Display;
    }
    bool premultiplied() const override
    {
        return m_settings.debugView == sabercat::DebugView::Final;
    }
    bool hasAlpha() const override
    {
        return true;
    }

    // Zero-copy: the renderer's image for Final, else the debug view's buffer. Invalid
    // before the first successful render.
    BufferView color() const override;
    // The renderer's depth buffer: depth, coverage, front depth, pad (stride 16).
    BufferView depth() const override;
    DepthInfo depthInfo() const override
    {
        return DepthInfo{4, 2, 1};
    }
    // One 16-byte readback of the pixel's color (the same values as color()).
    bool readPixel(int32_t x, int32_t y, float rgba[4]) const override;
    // One 16-byte readback of the pixel's four depth components.
    bool readDepthRaw(int32_t x, int32_t y, float out[4], uint32_t & count) const override;
    // Lazily downloads the front depth once per render and scans it.
    bool depthRange(float & lo, float & hi) const override;
    // Lazily downloads the color and front depth once per render. Null when not rendered
    // or when the download fails.
    const ImageData * hostData() const override;

    // Settings. Each one is cheap and applied at the next render(); setSize resizes the
    // renderer at once and invalidates the views until the next render.
    // Throws std::invalid_argument unless 0 < width, height <= kMaxSourceSize.
    void setSize(uint32_t width, uint32_t height);
    void setFlipUpAxis(bool flip);
    // -1 = full (3); other values are clamped to 0..3.
    void setMaxSHDegree(int32_t shCap);
    void setAntialiased(bool antialiased);
    void setDebugView(sabercat::DebugView view);

    // True when render() would re-render: a setting changed, nothing rendered yet, or the
    // camera generation moved on.
    bool dirty(uint64_t cameraGeneration) const;
    // Renders when dirty() and Ready: setCamera, forward(Differentiable), then the debug
    // view when one is chosen (always after a successful forward). A camera whose size is
    // not the source's size is ignored (returns false, stays dirty). Any std::exception
    // gives status Error with its message and shuts the renderer down. Returns true when
    // it re-rendered.
    bool render(const sabercat::Camera & camera, uint64_t cameraGeneration);
    // True once a render succeeded (the views are valid).
    bool rendered() const
    {
        return m_rendered;
    }
    // Number of successful renders so far (tests count the camera's effect with it).
    uint64_t renderCount() const
    {
        return m_renderCount;
    }
    // Device-local bytes of this scene: its own renderer's allocator (loading included).
    // Zero once the scene has failed or been destroyed.
    uint64_t deviceBytes() const;

    // For tests. Throws std::logic_error when the scene did not load (no renderer).
    sabercat::Renderer & renderer();
    SceneSettings settings() const
    {
        return m_settings;
    }
    bool antialiasFromFile() const
    {
        return m_antialiasFromFile;
    }
    sabercat::SceneBounds bounds() const;

  private:
    // Host-side copies derived from the last render. Mutable: they are filled lazily by
    // const readers and dropped whenever the content changes.
    struct HostCache
    {
        ImageData host;
        std::vector<float> frontDepth;
        float rangeLo = 0.0f;
        float rangeHi = 0.0f;
        bool hostValid = false;
        bool frontValid = false;
        bool rangeScanned = false;
        bool rangeFound = false;
    };

    void fail(SourceStatus status, std::string message);
    void shutDownRenderer() noexcept;
    void invalidateViews();
    const std::vector<float> * frontDepth() const;

    GpuContext & m_context;
    std::unique_ptr<sabercat::Renderer> m_renderer;
    std::string m_path;
    std::string m_message;
    SceneSettings m_settings;
    sabercat::BufferView m_debugColor;
    mutable HostCache m_cache;
    uint64_t m_renderedGeneration = 0;
    uint64_t m_renderCount = 0;
    SourceStatus m_status = SourceStatus::Error;
    bool m_dirty = true;
    bool m_rendered = false;
    bool m_antialiasFromFile = false;
    bool m_rendererDown = false;
};

// Device-local memory in use (VMA usage summed over the device-local heaps), in bytes. VMA
// counts per allocator, and each scene renderer has its own, so this sees only the device's
// own allocations; SceneSource::deviceBytes() is a scene's share.
uint64_t deviceLocalAllocatedBytes(sabercat::Device & device);
// Device-local memory still free (per heap max(0, budget - usage), summed), in bytes.
uint64_t deviceLocalFreeBytes(sabercat::Device & device);
// True when neededBytes fit in the free device-local memory, after bytesInUse (the sources'
// device-local bytes, which the device's own allocator does not count, or the scenes' for
// sceneBudgetOk). Otherwise false, with the message "Not enough GPU memory for <what>: needs
// about N MiB, M MiB free". The test hook below applies to both.
bool deviceBudgetOk(sabercat::Device & device, uint64_t neededBytes, const std::string & what,
                    std::string & message, uint64_t bytesInUse = 0);
// deviceBudgetOk() for one more scene (the message says "another scene").
bool sceneBudgetOk(sabercat::Device & device, uint64_t neededBytes, std::string & message,
                   uint64_t bytesInUse = 0);
// Test hook: replaces the free bytes in deviceBudgetOk() and sceneBudgetOk() (bytesInUse is
// not subtracted from it). nullopt restores the measurement.
void setSceneBudgetOverride(std::optional<uint64_t> freeBytes);

} // namespace sabercat::compare::gui
