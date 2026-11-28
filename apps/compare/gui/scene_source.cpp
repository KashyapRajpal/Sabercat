// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "scene_source.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include <vulkan/vulkan.h>

#include "compare_layout.h"
#include "gpu_context.h"
#include "sabercat/device.h"

namespace sabercat::compare::gui
{

namespace
{

// The depth buffer holds four floats per pixel: depth, coverage, front depth, pad.
constexpr uint32_t kDepthFloats = 4;
constexpr uint32_t kFrontComponent = 2;
// One RGBA32F texel or one depth pixel, in bytes (both have stride 16).
constexpr size_t kPixelBytes = kDepthFloats * sizeof(float);
constexpr uint32_t kFullSHDegree = 3;
constexpr uint64_t kMiB = 1024ull * 1024ull;

// Test hook for sceneBudgetOk(). Empty in the app: the measured budget is used.
std::optional<uint64_t> g_budgetOverride;

// A UTF-8 string as a path, without the deprecated u8path().
std::filesystem::path pathFromUtf8(const std::string & utf8)
{
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t *>(utf8.data()), utf8.size()));
}

std::string lowerAscii(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

bool validSize(uint32_t width, uint32_t height)
{
    const auto limit = static_cast<uint32_t>(kMaxSourceSize);
    return width > 0 && height > 0 && width <= limit && height <= limit;
}

// Reads the 16 bytes of pixel (x, y) of a row-major, stride-16 view. One small readback:
// the probe runs every frame while hovering, so a full-frame copy would cost far more.
bool readPixelBytes(GpuContext & context, const BufferView & view, uint32_t width, uint32_t height, int32_t x,
                    int32_t y, float out[4])
{
    if (!view.valid() || x < 0 || y < 0 || static_cast<uint32_t>(x) >= width || static_cast<uint32_t>(y) >= height)
    {
        return false;
    }
    const VkDeviceSize offset = (static_cast<VkDeviceSize>(y) * width + static_cast<VkDeviceSize>(x)) * kPixelBytes;
    if (offset + kPixelBytes > view.size)
    {
        return false;
    }
    // readBuffer copies from view.offset, so the pixel's sub-view carries the offset.
    const BufferView pixel{view.buffer, view.offset + offset, kPixelBytes, view.stride};
    try
    {
        context.readBuffer(pixel, out, kPixelBytes);
    }
    catch (const std::exception &)
    {
        return false;
    }
    return true;
}

struct DeviceLocalTotals
{
    uint64_t allocated = 0;
    uint64_t freeBytes = 0;
};

// VMA's per-heap usage and budget, summed over the device-local heaps only.
DeviceLocalTotals readDeviceLocal(sabercat::Device & device)
{
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(device.physicalDevice(), &properties);
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(device.allocator(), budgets.data());

    DeviceLocalTotals totals;
    for (uint32_t heap = 0; heap < properties.memoryHeapCount; ++heap)
    {
        if ((properties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0)
        {
            continue;
        }
        const VmaBudget & budget = budgets[heap];
        totals.allocated += budget.usage;
        if (budget.budget > budget.usage)
        {
            totals.freeBytes += budget.budget - budget.usage;
        }
    }
    return totals;
}

} // namespace

bool isSceneFile(const std::string & utf8Path)
{
    // Only the file name counts, so a folder named like an extension does not match.
    const size_t separator = utf8Path.find_last_of("/\\");
    const std::string name =
        lowerAscii(separator == std::string::npos ? utf8Path : utf8Path.substr(separator + 1));
    if (name == "meta.json")
    {
        return true;
    }
    const size_t dot = name.find_last_of('.');
    if (dot == std::string::npos)
    {
        return false;
    }
    const std::string ext = name.substr(dot);
    return ext == ".ply" || ext == ".spz" || ext == ".splat" || ext == ".ksplat" || ext == ".sog" || ext == ".gltf"
        || ext == ".glb";
}

SceneSource::SceneSource(GpuContext & context, const sabercat::Renderer & host, const std::string & utf8Path,
                         const SceneSettings & settings, bool applyAntialias)
    : m_context(context), m_path(utf8Path), m_settings(settings)
{
    if (!validSize(settings.width, settings.height))
    {
        fail(SourceStatus::Error, "scene render size must be 1 to " + std::to_string(kMaxSourceSize) + " per side");
        return;
    }
    std::error_code ec;
    if (!std::filesystem::exists(pathFromUtf8(utf8Path), ec))
    {
        fail(SourceStatus::Missing, "file not found: " + utf8Path);
        return;
    }
    try
    {
        m_renderer = std::make_unique<sabercat::Renderer>(settings.width, settings.height,
                                                          sabercat::DepthLayout::DepthCoverageFront);
        m_renderer->initializeBorrowed(host.vulkanContext(), host.enabledDeviceFeatures(), context.shaderDir());
        m_renderer->loadScene(utf8Path.c_str());
        // The loader's own flag is read before any override. When the caller does not apply
        // the requested value, the source reports what the renderer actually uses.
        m_antialiasFromFile = m_renderer->sceneAntialiased();
        if (applyAntialias)
        {
            m_renderer->setAntialiased(settings.antialias);
        }
        m_settings.antialias = applyAntialias ? settings.antialias : m_antialiasFromFile;
        m_status = SourceStatus::Ready;
    }
    catch (const std::exception & e)
    {
        fail(SourceStatus::Error, e.what());
    }
}

SceneSource::~SceneSource()
{
    shutDownRenderer();
}

void SceneSource::fail(SourceStatus status, std::string message)
{
    m_status = status;
    m_message = std::move(message);
    invalidateViews();
    shutDownRenderer();
}

void SceneSource::shutDownRenderer() noexcept
{
    if (!m_renderer || m_rendererDown)
    {
        return;
    }
    m_rendererDown = true;
    try
    {
        m_renderer->shutdown();
    }
    catch (const std::exception &)
    {
        // Teardown failure leaves nothing to recover; the renderer is dropped with the source.
    }
}

void SceneSource::invalidateViews()
{
    m_rendered = false;
    m_debugColor = BufferView{};
    m_cache = HostCache{};
}

void SceneSource::setSize(uint32_t width, uint32_t height)
{
    if (!validSize(width, height))
    {
        throw std::invalid_argument("scene render size must be 1 to " + std::to_string(kMaxSourceSize) + " per side");
    }
    if (width == m_settings.width && height == m_settings.height)
    {
        return;
    }
    m_settings.width = width;
    m_settings.height = height;
    m_dirty = true;
    if (m_status != SourceStatus::Ready)
    {
        return;
    }
    // A resize invalidates the renderer's views until the next forward.
    invalidateViews();
    try
    {
        m_renderer->setImageSize(width, height);
    }
    catch (const std::exception & e)
    {
        fail(SourceStatus::Error, e.what());
    }
}

void SceneSource::setFlipUpAxis(bool flip)
{
    if (flip != m_settings.flipUpAxis)
    {
        m_settings.flipUpAxis = flip;
        m_dirty = true;
    }
}

void SceneSource::setMaxSHDegree(int32_t shCap)
{
    const int32_t clamped = std::clamp(shCap, -1, static_cast<int32_t>(kFullSHDegree));
    if (clamped != m_settings.shCap)
    {
        m_settings.shCap = clamped;
        m_dirty = true;
    }
}

void SceneSource::setAntialiased(bool antialiased)
{
    if (antialiased != m_settings.antialias)
    {
        m_settings.antialias = antialiased;
        m_dirty = true;
    }
}

void SceneSource::setDebugView(sabercat::DebugView view)
{
    if (view != m_settings.debugView)
    {
        m_settings.debugView = view;
        m_dirty = true;
    }
}

bool SceneSource::dirty(uint64_t cameraGeneration) const
{
    return m_dirty || !m_rendered || cameraGeneration != m_renderedGeneration;
}

bool SceneSource::render(const sabercat::Camera & camera, uint64_t cameraGeneration)
{
    if (m_status != SourceStatus::Ready || !dirty(cameraGeneration))
    {
        return false;
    }
    // The caller frames the camera at the source's own size; a mismatch waits for the next
    // frame rather than rendering with the wrong intrinsics.
    if (camera.width != m_settings.width || camera.height != m_settings.height)
    {
        return false;
    }
    try
    {
        m_renderer->setFlipUpAxis(m_settings.flipUpAxis);
        m_renderer->setMaxSHDegree(m_settings.shCap < 0 ? kFullSHDegree : static_cast<uint32_t>(m_settings.shCap));
        m_renderer->setAntialiased(m_settings.antialias);
        m_renderer->setCamera(camera);
        m_renderer->forward(sabercat::RenderMode::Differentiable);
        // Only after the forward above: renderDebugView throws without one.
        m_debugColor = m_settings.debugView == sabercat::DebugView::Final
            ? BufferView{}
            : m_renderer->renderDebugView(m_settings.debugView);
    }
    catch (const std::exception & e)
    {
        fail(SourceStatus::Error, e.what());
        return false;
    }
    m_cache = HostCache{};
    m_rendered = true;
    m_dirty = false;
    m_renderedGeneration = cameraGeneration;
    ++m_renderCount;
    return true;
}

BufferView SceneSource::color() const
{
    if (m_status != SourceStatus::Ready || !m_rendered)
    {
        return {};
    }
    if (m_settings.debugView == sabercat::DebugView::Final)
    {
        return m_renderer->imageView();
    }
    return m_debugColor;
}

BufferView SceneSource::depth() const
{
    if (m_status != SourceStatus::Ready || !m_rendered)
    {
        return {};
    }
    return m_renderer->depthView();
}

bool SceneSource::readPixel(int32_t x, int32_t y, float rgba[4]) const
{
    return readPixelBytes(m_context, color(), m_settings.width, m_settings.height, x, y, rgba);
}

bool SceneSource::readDepthRaw(int32_t x, int32_t y, float out[4], uint32_t & count) const
{
    count = 0;
    if (m_status != SourceStatus::Ready || !m_rendered)
    {
        return false;
    }
    if (!readPixelBytes(m_context, m_renderer->depthView(), m_settings.width, m_settings.height, x, y, out))
    {
        return false;
    }
    count = kDepthFloats;
    return true;
}

const std::vector<float> * SceneSource::frontDepth() const
{
    if (m_cache.frontValid)
    {
        return &m_cache.frontDepth;
    }
    if (m_status != SourceStatus::Ready || !m_rendered)
    {
        return nullptr;
    }
    const BufferView depthView = m_renderer->depthView();
    if (!depthView.valid())
    {
        return nullptr;
    }
    const size_t pixels = static_cast<size_t>(m_settings.width) * m_settings.height;
    try
    {
        std::vector<float> raw(pixels * kDepthFloats);
        m_context.readBuffer(depthView, raw.data(), raw.size() * sizeof(float));
        m_cache.frontDepth.resize(pixels);
        for (size_t i = 0; i < pixels; ++i)
        {
            m_cache.frontDepth[i] = raw[i * kDepthFloats + kFrontComponent];
        }
    }
    catch (const std::exception &)
    {
        m_cache.frontDepth.clear();
        return nullptr;
    }
    m_cache.frontValid = true;
    return &m_cache.frontDepth;
}

bool SceneSource::depthRange(float & lo, float & hi) const
{
    if (!m_cache.rangeScanned)
    {
        const std::vector<float> * front = frontDepth();
        if (front == nullptr)
        {
            return false;
        }
        bool found = false;
        float low = 0.0f;
        float high = 0.0f;
        for (const float d : *front)
        {
            if (!isValidDepth(d))
            {
                continue;
            }
            if (!found)
            {
                low = d;
                high = d;
                found = true;
            }
            else
            {
                low = std::min(low, d);
                high = std::max(high, d);
            }
        }
        m_cache.rangeFound = found;
        m_cache.rangeLo = low;
        m_cache.rangeHi = high;
        m_cache.rangeScanned = true;
    }
    if (!m_cache.rangeFound)
    {
        return false;
    }
    lo = m_cache.rangeLo;
    hi = m_cache.rangeHi;
    return true;
}

const ImageData * SceneSource::hostData() const
{
    if (m_cache.hostValid)
    {
        return &m_cache.host;
    }
    const std::vector<float> * front = frontDepth();
    const BufferView colorView = color();
    if (front == nullptr || !colorView.valid())
    {
        return nullptr;
    }
    const size_t pixels = static_cast<size_t>(m_settings.width) * m_settings.height;
    try
    {
        ImageData & host = m_cache.host;
        host.rgba.resize(pixels * 4);
        m_context.readBuffer(colorView, host.rgba.data(), host.rgba.size() * sizeof(float));
        host.depth = *front;
        host.width = m_settings.width;
        host.height = m_settings.height;
        host.bitDepth = 32;
        host.encoding = Encoding::Display;
        host.hasAlpha = true;
        host.premultiplied = m_settings.debugView == sabercat::DebugView::Final;
    }
    catch (const std::exception &)
    {
        m_cache.host = ImageData{};
        return nullptr;
    }
    m_cache.hostValid = true;
    return &m_cache.host;
}

sabercat::Renderer & SceneSource::renderer()
{
    if (!m_renderer)
    {
        throw std::logic_error("scene did not load: no renderer");
    }
    return *m_renderer;
}

sabercat::SceneBounds SceneSource::bounds() const
{
    return m_renderer ? m_renderer->sceneBounds() : sabercat::SceneBounds{};
}

uint64_t deviceLocalAllocatedBytes(sabercat::Device & device)
{
    return readDeviceLocal(device).allocated;
}

uint64_t deviceLocalFreeBytes(sabercat::Device & device)
{
    return readDeviceLocal(device).freeBytes;
}

uint64_t SceneSource::deviceBytes() const
{
    if (!m_renderer || m_rendererDown)
    {
        return 0;
    }
    return deviceLocalAllocatedBytes(m_renderer->sabercatDevice());
}

bool deviceBudgetOk(sabercat::Device & device, uint64_t neededBytes, const std::string & what, std::string & message,
                    uint64_t bytesInUse)
{
    uint64_t freeBytes = 0;
    if (g_budgetOverride)
    {
        freeBytes = *g_budgetOverride;
    }
    else
    {
        const uint64_t measured = deviceLocalFreeBytes(device);
        freeBytes = measured > bytesInUse ? measured - bytesInUse : 0;
    }
    if (freeBytes >= neededBytes)
    {
        return true;
    }
    const uint64_t neededMiB = (neededBytes + kMiB - 1) / kMiB;
    message = "Not enough GPU memory for " + what + ": needs about " + std::to_string(neededMiB) + " MiB, "
        + std::to_string(freeBytes / kMiB) + " MiB free";
    return false;
}

bool sceneBudgetOk(sabercat::Device & device, uint64_t neededBytes, std::string & message, uint64_t bytesInUse)
{
    return deviceBudgetOk(device, neededBytes, "another scene", message, bytesInUse);
}

void setSceneBudgetOverride(std::optional<uint64_t> freeBytes)
{
    g_budgetOverride = freeBytes;
}

} // namespace sabercat::compare::gui
