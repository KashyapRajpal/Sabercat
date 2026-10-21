// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Renderer-level contract, driven through the public Renderer API and checked
// numerically (no screenshots):
//
//   A. setMaxSHDegree: every cap renders exactly like a scene whose higher
//      bands are zeroed; caps above 3 clamp to 3.
//   B. setLODParams: the Viewer forward equals a ForwardPass run with the same
//      knobs; a Differentiable forward ignores them.
//   C. Antialiasing: an isolated Gaussian's alpha matches a C++ reference of the
//      dilated 2D footprint, with and without the sqrt(det ratio) compensation;
//      the flag belongs to the scene and resets on loadScene.
//   D. setFlipUpAxis: a flipped camera renders the mirrored scene (means,
//      rotations and y-parity SH mirrored) with the flip off; flip off restores
//      the original image bit-exactly.
//   E. DebugView: DebugVisualizer output against a CPU reference of each mode,
//      in both frame slots and after a resize to an odd size; presentation views
//      never change the forward image.
//   F. sceneBounds / sceneSHDegree against brute force.
//   G. Rejected calls throw with the state unchanged; forward with no scene or
//      an invalid camera is a no-op.

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/camera.h"
#include "sabercat/debug_visualizer.h"
#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/renderer.h"
#include "sabercat/scene_bounds.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 96;
constexpr uint32_t kHeight = 64;
constexpr float kFocal = 60.0f;

// Off-axis eye for the flip and debug checks: the Y flip must move the view.
const glm::vec3 kOffAxisEye(0.6f, 0.5f, 4.0f);

void require(bool ok, const std::string & what)
{
    if (!ok)
    {
        throw std::runtime_error(what);
    }
}

// Message without the " at <file>:<line>" suffix VK_VALIDATE appends.
std::string briefMessage(const std::exception & e)
{
    std::string msg = e.what();
    size_t at = msg.find(" at ");
    return at == std::string::npos ? msg : msg.substr(0, at);
}

// Expects `call` to throw; prints the (short) message as evidence.
template <typename Fn> void expectThrow(const char * label, Fn && call)
{
    try
    {
        call();
    }
    catch (const std::exception & e)
    {
        std::printf("    rejected %-40s: %s\n", label, briefMessage(e).c_str());
        return;
    }
    throw std::runtime_error(std::string("no exception: ") + label);
}

void initRenderer(Renderer & r)
{
    require(r.initialize({}, true, SHADER_DIR), "renderer initialize failed");
}

// Fixed-seed generator: the same scenes in Debug and Release.
class Rng
{
  public:
    explicit Rng(uint32_t seed) : m_state(seed) {}
    float next()
    {
        m_state = m_state * 1664525u + 1013904223u;
        return static_cast<float>(m_state >> 8) * (1.0f / 16777216.0f);
    }

  private:
    uint32_t m_state;
};

// Host arrays at the canonical strides (see GaussianHostInput).
struct Scene
{
    uint32_t count = 0;
    std::vector<float> means;     // 4 per Gaussian (w unused)
    std::vector<float> scales;    // 4 per Gaussian
    std::vector<float> rotations; // 4 per Gaussian, xyzw
    std::vector<float> opacities; // 1 per Gaussian
    std::vector<float> sh;        // 16 coefficients x 4 per Gaussian, band-ascending

    GaussianHostInput input(uint32_t shDegree, bool antialiased = false) const
    {
        GaussianHostInput in;
        in.count = count;
        in.means = means.data();
        in.scales = scales.data();
        in.rotations = rotations.data();
        in.opacities = opacities.data();
        in.shCoefficients = sh.data();
        in.shDegree = shDegree;
        in.antialiased = antialiased;
        return in;
    }
};

Scene makeScene(uint32_t count, uint32_t seed)
{
    Rng rng(seed);
    Scene s;
    s.count = count;
    s.means.resize(size_t(count) * 4);
    s.scales.resize(size_t(count) * 4);
    s.rotations.resize(size_t(count) * 4);
    s.opacities.resize(count);
    s.sh.resize(size_t(count) * kShCoefficients * 4);
    for (uint32_t i = 0; i < count; ++i)
    {
        float * m = &s.means[size_t(i) * 4];
        m[0] = (rng.next() - 0.5f) * 2.4f;
        m[1] = (rng.next() - 0.5f) * 1.8f;
        m[2] = -2.0f + 3.0f * rng.next(); // depths 3..6 from the off-axis eye
        m[3] = 0.0f;

        float * sc = &s.scales[size_t(i) * 4];
        for (int a = 0; a < 3; ++a)
        {
            sc[a] = 0.05f + 0.2f * rng.next();
        }
        sc[3] = 0.0f;

        glm::vec4 quat(rng.next() - 0.5f, rng.next() - 0.5f, rng.next() - 0.5f, rng.next() - 0.5f);
        quat = glm::normalize(quat);
        float * q = &s.rotations[size_t(i) * 4];
        q[0] = quat.x;
        q[1] = quat.y;
        q[2] = quat.z;
        q[3] = quat.w;

        s.opacities[i] = 0.3f + 0.6f * rng.next();

        // Every band carries energy, so a wrong band shows in the image.
        for (uint32_t k = 0; k < kShCoefficients; ++k)
        {
            float * c = &s.sh[(size_t(i) * kShCoefficients + k) * 4];
            const float amplitude = k == 0 ? 1.6f : 0.5f;
            for (int a = 0; a < 3; ++a)
            {
                c[a] = amplitude * (rng.next() - 0.5f);
            }
            c[3] = 0.0f;
        }
    }
    return s;
}

// SH band of coefficient k: 0 (DC), 1 (k 1-3), 2 (k 4-8), 3 (k 9-15).
uint32_t shBandOf(uint32_t k)
{
    return k == 0 ? 0u : k < 4 ? 1u : k < 9 ? 2u : 3u;
}

// Same scene with every coefficient above `degree` set to zero.
Scene zeroAbove(const Scene & s, uint32_t degree)
{
    Scene out = s;
    for (uint32_t i = 0; i < s.count; ++i)
    {
        for (uint32_t k = 0; k < kShCoefficients; ++k)
        {
            if (shBandOf(k) > degree)
            {
                float * c = &out.sh[(size_t(i) * kShCoefficients + k) * 4];
                c[0] = c[1] = c[2] = 0.0f;
            }
        }
    }
    return out;
}

// The mirror image of a scene in the plane y = 0: means.y negated; the rotation
// R becomes M R M with M = diag(1,-1,1), which for xyzw is (-x, y, -z, w); SH
// coefficients whose basis function is odd in y (bands 1, 4, 5, 9, 10, 11 in
// the band-ascending order used here) change sign.
Scene mirrorY(const Scene & s)
{
    Scene out = s;
    static const uint32_t kOddInY[] = {1, 4, 5, 9, 10, 11};
    for (uint32_t i = 0; i < s.count; ++i)
    {
        out.means[size_t(i) * 4 + 1] = -s.means[size_t(i) * 4 + 1];
        float * q = &out.rotations[size_t(i) * 4];
        q[0] = -q[0];
        q[2] = -q[2];
        for (uint32_t k : kOddInY)
        {
            float * c = &out.sh[(size_t(i) * kShCoefficients + k) * 4];
            c[0] = -c[0];
            c[1] = -c[1];
            c[2] = -c[2];
        }
    }
    return out;
}

// One Gaussian at (0, 0, -6): with the camera at the origin it projects exactly
// onto pixel (48, 32), and with an identity rotation its 2D covariance is diagonal.
Scene singleGaussian()
{
    Scene s;
    s.count = 1;
    s.means = {0.0f, 0.0f, -6.0f, 0.0f};
    s.scales = {0.6f, 0.3f, 0.2f, 0.0f};
    s.rotations = {0.0f, 0.0f, 0.0f, 1.0f};
    s.opacities = {0.8f};
    s.sh.assign(kShCoefficients * 4, 0.0f);
    s.sh[0] = 0.4f;
    s.sh[1] = 0.2f;
    s.sh[2] = -0.1f;
    return s;
}

// Identity rotation, camera at `eye` looking down -Z, principal point at the
// centre of a w x h image.
Camera makeCamera(const glm::vec3 & eye, uint32_t w, uint32_t h)
{
    Camera c;
    c.width = w;
    c.height = h;
    c.fx = c.fy = kFocal;
    c.cx = 0.5f * static_cast<float>(w);
    c.cy = 0.5f * static_cast<float>(h);
    c.cameraToWorld = glm::mat4(1.0f);
    c.cameraToWorld[3] = glm::vec4(eye, 1.0f);
    return c;
}

// The camera block the Renderer builds in Renderer::forward, so ForwardPass and
// DebugVisualizer can be driven with exactly what the Renderer sees.
CameraUniforms renderUniforms(const Camera & camera, uint32_t w, uint32_t h, uint32_t count,
                              uint32_t shDegree, bool flip, bool antialiased)
{
    glm::mat4 worldToCamera = camera.worldToCamera();
    glm::vec3 cameraPosition = glm::vec3(camera.cameraToWorld[3]);
    if (flip)
    {
        glm::mat4 f(1.0f);
        f[1][1] = -1.0f;
        worldToCamera = worldToCamera * f;
        cameraPosition.y = -cameraPosition.y;
    }
    CameraUniforms u{};
    u.worldToCamera = worldToCamera;
    u.fx = camera.fx;
    u.fy = camera.fy;
    u.cx = camera.cx;
    u.cy = camera.cy;
    u.imageWidth = w;
    u.imageHeight = h;
    u.tilesX = dispatchGroups(w, kTileSize);
    u.tilesY = dispatchGroups(h, kTileSize);
    u.gaussianCount = count;
    u.shDegree = shDegree;
    u.renderFlags = antialiased ? kRenderFlagAntialiased : 0u;
    u.cameraPositionWorld = glm::vec4(cameraPosition, 0.0f);
    return u;
}

void fillCloud(GaussianCloud & cloud, const Scene & s)
{
    cloud.initialize(s.count);
    std::memcpy(cloud.positions().data(), s.means.data(), s.means.size() * sizeof(float));
    std::memcpy(cloud.scales().data(), s.scales.data(), s.scales.size() * sizeof(float));
    std::memcpy(cloud.rotations().data(), s.rotations.data(), s.rotations.size() * sizeof(float));
    std::memcpy(cloud.opacities().data(), s.opacities.data(), s.opacities.size() * sizeof(float));
    std::memcpy(cloud.shCoefficients().data(), s.sh.data(), s.sh.size() * sizeof(float));
    cloud.uploadToGPU();
}

std::vector<float> render(Renderer & r, RenderMode mode = RenderMode::Viewer)
{
    r.forward(mode);
    return r.readImage();
}

std::vector<float> toFloats(const std::vector<glm::vec4> & px)
{
    std::vector<float> out(px.size() * 4);
    std::memcpy(out.data(), px.data(), out.size() * sizeof(float));
    return out;
}

bool bitEqual(const std::vector<float> & a, const std::vector<float> & b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

float maxAbsComponent(const glm::vec3 & v)
{
    const glm::vec3 a = glm::abs(v);
    return std::max({a.x, a.y, a.z});
}

float maxAbsComponent(const glm::vec4 & v)
{
    const glm::vec4 a = glm::abs(v);
    return std::max({a.x, a.y, a.z, a.w});
}

double maxAbsDiff(const std::vector<float> & a, const std::vector<float> & b)
{
    require(a.size() == b.size(), "image size mismatch");
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        worst = std::max(worst, std::fabs(double(a[i]) - double(b[i])));
    }
    return worst;
}

template <typename T> std::vector<T> download(const Buffer & buffer, size_t count)
{
    std::vector<T> out(count);
    buffer.download(out.data(), count * sizeof(T));
    return out;
}

// -- A. setMaxSHDegree ------------------------------------------------------

void partA()
{
    std::printf("(A) setMaxSHDegree\n");
    Renderer r(kWidth, kHeight);
    initRenderer(r);
    r.setCamera(makeCamera(kOffAxisEye, kWidth, kHeight));
    const Scene s = makeScene(60, 101);

    r.setGaussians(s.input(kShDegree));
    r.setMaxSHDegree(kShDegree);
    const std::vector<float> full = render(r);

    const uint32_t caps[] = {0, 1, 2, 3, 7};
    for (uint32_t cap : caps)
    {
        r.setGaussians(s.input(kShDegree));
        r.setMaxSHDegree(cap);
        const std::vector<float> image = render(r);

        // Reference: the same scene with the bands above the effective cap
        // zeroed, rendered at the full default cap. The extra terms are exact
        // zeros, so the two must agree bit for bit.
        const uint32_t effective = std::min(cap, kShDegree);
        r.setGaussians(zeroAbove(s, effective).input(kShDegree));
        r.setMaxSHDegree(kShDegree);
        const std::vector<float> reference = render(r);

        require(bitEqual(image, reference),
                "(A) cap differs from the zeroed-bands reference");
        std::printf("    cap %u (effective %u): bit-exact vs zeroed-bands scene, "
                    "|image - full| = %.3e\n",
                    cap, effective, maxAbsDiff(image, full));
        if (cap == 0)
        {
            require(maxAbsDiff(image, full) > 1e-2, "(A) fixture: degree 0 looks like degree 3");
        }
        if (effective == kShDegree)
        {
            require(bitEqual(image, full), "(A) cap 3 (or clamped 7) differs from the default");
        }
    }
    require(r.sceneSHDegree() == kShDegree, "(A) sceneSHDegree must not follow the cap");
    std::printf("(A) OK: caps 0-3 and clamped 7 match their references\n");
    r.shutdown();
}

// -- B. setLODParams --------------------------------------------------------

LODParams appearanceLOD()
{
    // Depths 3..6 span the whole distance range, so the SH degree steps down
    // across the scene, with a fade on each step.
    LODParams p{};
    p.enabled = 0;
    p.appearanceEnabled = 1;
    p.nearDistance = 3.0f;
    p.farDistance = 6.0f;
    p.densityReference = 1.0f;
    p.appearanceMaxDropDistance = 1.0f;
    p.appearanceMaxDropDensity = 0.0f;
    p.appearanceFadeWidth = 0.5f;
    return p;
}

void partB()
{
    std::printf("(B) setLODParams\n");
    Renderer r(kWidth, kHeight);
    initRenderer(r);
    const Camera cam = makeCamera(kOffAxisEye, kWidth, kHeight);
    r.setCamera(cam);
    const Scene s = makeScene(60, 202);
    r.setGaussians(s.input(kShDegree));

    const LODParams lod = appearanceLOD();
    r.setLODParams(LODParams{});
    const std::vector<float> plain = render(r);
    r.setLODParams(lod);
    const std::vector<float> lodImage = render(r);
    const double lodEffect = maxAbsDiff(lodImage, plain);
    require(lodEffect > 1e-3, "(B) fixture: LOD does not change the image");

    // The same knobs on a ForwardPass with the same uniforms, on the same device.
    Device & device = r.sabercatDevice();
    ForwardPass pass(device, kWidth, kHeight, SHADER_DIR);
    GaussianCloud cloud(device);
    fillCloud(cloud, s);
    pass.setLODParams(lod);
    pass.run(cloud.views(), renderUniforms(cam, kWidth, kHeight, s.count, kShDegree, false, false));
    require(bitEqual(lodImage, toFloats(pass.downloadOutputImage())),
            "(B) Renderer LOD differs from the ForwardPass result");

    // Differentiable pins the viewer-only knobs off.
    r.setLODParams(lod);
    const std::vector<float> diffWithLod = render(r, RenderMode::Differentiable);
    r.setLODParams(LODParams{});
    const std::vector<float> diffPlain = render(r, RenderMode::Differentiable);
    require(bitEqual(diffWithLod, diffPlain), "(B) Differentiable forward honours the LOD knobs");
    require(bitEqual(render(r), plain), "(B) viewer image after LOD off differs from the first render");

    std::printf("    Viewer LOD vs ForwardPass LOD: bit-exact over %u px\n", kWidth * kHeight);
    std::printf("    LOD changes the viewer image by max %.3e (fixture)\n", lodEffect);
    std::printf("    Differentiable with LOD on vs off: bit-exact\n");
    std::printf("(B) OK\n");
}

// -- C. Antialiasing ----------------------------------------------------------

// Coverage of singleGaussian() at pixel (x, y) from the projected footprint:
// the 2D covariance is diag((f s1 / z)^2, (f s2 / z)^2) plus the low-pass
// dilation on each diagonal term. Antialiased scenes multiply the opacity by
// sqrt(det before / det after dilation). Mirrors the rasterizer: contributions
// below 1/255 vanish and the alpha clamps to kMaxAlpha. `unclamped` returns the
// value before both of those.
double referenceCoverage(int x, int y, bool antialiased, double & unclamped)
{
    const double z = 6.0;
    const double ax = std::pow(kFocal * 0.6 / z, 2.0);
    const double ay = std::pow(kFocal * 0.3 / z, 2.0);
    const double cxx = ax + kCovarianceDilation;
    const double cyy = ay + kCovarianceDilation;
    const double opacity = double(0.8f);
    const double dx = x - 48.0;
    const double dy = y - 32.0;
    const double compensation = antialiased ? std::sqrt(ax * ay / (cxx * cyy)) : 1.0;
    unclamped = opacity * compensation * std::exp(-0.5 * (dx * dx / cxx + dy * dy / cyy));
    if (unclamped < 1.0 / 255.0)
    {
        return 0.0;
    }
    return std::min(unclamped, double(kMaxAlpha));
}

// Compares the alpha channel over every pixel whose unclamped reference is at
// least 0.05. Such pixels lie inside 2.3 sigma of the footprint, well within the
// tile culling radius (3 sigma), so culling cannot touch them. Tolerance 1e-5:
// the chain (covariance, conic, exponent, exp, sqrt, product) is about 30 fp32
// operations on values of magnitude <= 1, so the worst rounding is a few times
// 1e-6; 1e-5 leaves a safety margin without hiding a formula error, which would
// show at 1e-2 or more.
void checkCoverage(const std::vector<float> & image, bool antialiased, const char * label)
{
    double worst = 0.0;
    size_t inside = 0;
    for (uint32_t y = 0; y < kHeight; ++y)
    {
        for (uint32_t x = 0; x < kWidth; ++x)
        {
            double unclamped = 0.0;
            const double expected = referenceCoverage(int(x), int(y), antialiased, unclamped);
            if (unclamped < 0.05)
            {
                continue;
            }
            ++inside;
            worst = std::max(worst, std::fabs(double(image[(size_t(y) * kWidth + x) * 4 + 3]) - expected));
        }
    }
    const double centre = image[(size_t(32) * kWidth + 48) * 4 + 3];
    double centreUnclamped = 0.0;
    const double centreReference = referenceCoverage(48, 32, antialiased, centreUnclamped);
    std::printf("    %-22s: centre alpha %.6f (reference %.6f); %zu px checked, worst |alpha - ref| %.3e\n",
                label, centre, centreReference, inside, worst);
    require(inside >= 100, "(C) fixture: too few covered pixels");
    require(worst <= 1e-5, "(C) alpha differs from the footprint reference");
}

void writeSplat(const std::filesystem::path & path, const Scene & s)
{
    std::ofstream out(path, std::ios::binary);
    require(out.good(), "cannot write the .splat test file");
    for (uint32_t i = 0; i < s.count; ++i)
    {
        const float * m = &s.means[size_t(i) * 4];
        const float * sc = &s.scales[size_t(i) * 4];
        const float * q = &s.rotations[size_t(i) * 4]; // xyzw
        out.write(reinterpret_cast<const char *>(m), 3 * sizeof(float));
        out.write(reinterpret_cast<const char *>(sc), 3 * sizeof(float));
        const uint8_t color[4] = {128, 128, 128, 255};
        out.write(reinterpret_cast<const char *>(color), 4);
        // Rotation bytes are wxyz, round(q * 128 + 128) per component.
        const float wxyz[4] = {q[3], q[0], q[1], q[2]};
        uint8_t rot[4];
        for (int c = 0; c < 4; ++c)
        {
            rot[c] = static_cast<uint8_t>(std::clamp(std::lround(wxyz[c] * 128.0f + 128.0f), 0L, 255L));
        }
        out.write(reinterpret_cast<const char *>(rot), 4);
    }
}

void partC()
{
    std::printf("(C) antialiasing\n");
    Renderer r(kWidth, kHeight);
    initRenderer(r);
    r.setCamera(makeCamera(glm::vec3(0.0f), kWidth, kHeight));

    // No scene yet: the flag has nothing to attach to.
    r.setAntialiased(true);
    require(!r.sceneAntialiased(), "(C) setAntialiased with no scene must be a no-op");

    const Scene one = singleGaussian();
    r.setGaussians(one.input(0, false));
    require(!r.sceneAntialiased(), "(C) a plain scene reports antialiased");
    const std::vector<float> off = render(r);
    checkCoverage(off, false, "plain");

    r.setAntialiased(true);
    require(r.sceneAntialiased(), "(C) setAntialiased(true) not applied");
    const std::vector<float> on = render(r);
    checkCoverage(on, true, "setAntialiased(true)");
    const size_t centre = (size_t(32) * kWidth + 48) * 4 + 3;
    require(std::fabs(on[centre] - off[centre]) > 0.01, "(C) fixture: compensation invisible at the centre");

    // The per-scene input flag gives the same image as setAntialiased.
    r.setGaussians(one.input(0, true));
    require(r.sceneAntialiased(), "(C) GaussianHostInput::antialiased not applied");
    require(bitEqual(render(r), on), "(C) input flag and setAntialiased disagree");
    r.setGaussians(one.input(0, false));
    require(!r.sceneAntialiased(), "(C) a new scene kept the previous flag");

    // loadScene replaces the flag with what the file says; .splat cannot say, so off.
    r.setAntialiased(true);
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "sabercat_renderer_test_single.splat";
    writeSplat(path, one);
    r.loadScene(path.string().c_str());
    std::filesystem::remove(path);
    require(!r.sceneAntialiased(), "(C) loadScene did not reset the flag");
    require(r.gaussianCount() == 1, "(C) loaded .splat count");
    std::printf("(C) OK\n");
    r.shutdown();
}

// -- D. setFlipUpAxis -----------------------------------------------------------

// Tolerance for the flipped-camera image against the mirrored scene. The two
// reach the same arithmetic by different routes (the flip folds Y into the view
// and congruence-transforms the covariance; the mirrored scene carries negated
// means, a mirrored quaternion and negated SH), and every step is a sign change,
// so they are expected to agree exactly; the measured worst difference on the
// development GPU is 0. 1e-5 is the allowance for other drivers' fp32 handling,
// and a wrong mirror (a sign or a band) shows at 1e-2 or more.
constexpr double kFlipTolerance = 1e-5;

void rotationMatrix(const float * q, double R[3][3])
{
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    R[0][0] = 1.0 - 2.0 * (y * y + z * z);
    R[0][1] = 2.0 * (x * y - w * z);
    R[0][2] = 2.0 * (x * z + w * y);
    R[1][0] = 2.0 * (x * y + w * z);
    R[1][1] = 1.0 - 2.0 * (x * x + z * z);
    R[1][2] = 2.0 * (y * z - w * x);
    R[2][0] = 2.0 * (x * z - w * y);
    R[2][1] = 2.0 * (y * z + w * x);
    R[2][2] = 1.0 - 2.0 * (x * x + y * y);
}

// Worst |M R M - R'| over the scene in double, M = diag(1,-1,1). This is an
// exact identity for the quaternion mirror, so the bound is double rounding only.
double reflectionError(const Scene & s, const Scene & mirrored)
{
    const double m[3] = {1.0, -1.0, 1.0};
    double worst = 0.0;
    for (uint32_t i = 0; i < s.count; ++i)
    {
        double R[3][3], Rm[3][3];
        rotationMatrix(&s.rotations[size_t(i) * 4], R);
        rotationMatrix(&mirrored.rotations[size_t(i) * 4], Rm);
        for (int a = 0; a < 3; ++a)
        {
            for (int b = 0; b < 3; ++b)
            {
                worst = std::max(worst, std::fabs(m[a] * m[b] * R[a][b] - Rm[a][b]));
            }
        }
    }
    return worst;
}

double flipCheck(Renderer & r, const Scene & s, const char * label)
{
    r.setCamera(makeCamera(kOffAxisEye, kWidth, kHeight));
    r.setFlipUpAxis(false);
    r.setGaussians(s.input(kShDegree));
    const std::vector<float> plain = render(r);

    r.setFlipUpAxis(true);
    const std::vector<float> flipped = render(r);
    const double fromPlain = maxAbsDiff(flipped, plain);
    require(fromPlain > 1e-2, "(D) fixture: flip does not change the image");

    const Scene mirrored = mirrorY(s);
    r.setFlipUpAxis(false);
    r.setGaussians(mirrored.input(kShDegree));
    const std::vector<float> mirror = render(r);
    const double mirrorError = maxAbsDiff(flipped, mirror);
    require(mirrorError <= kFlipTolerance, "(D) flipped camera differs from the mirrored scene");

    // Flip off restores the original image bit-exactly.
    r.setGaussians(s.input(kShDegree));
    r.setFlipUpAxis(true);
    r.setFlipUpAxis(false);
    require(bitEqual(render(r), plain), "(D) flip off does not restore the original image");

    std::printf("    %-18s: flip vs mirrored scene worst %.3e (tol %.0e); flip changes image by %.3e; "
                "flip off bit-exact\n",
                label, mirrorError, kFlipTolerance, fromPlain);
    return mirrorError;
}

void partD()
{
    std::printf("(D) setFlipUpAxis\n");
    Renderer r(kWidth, kHeight);
    initRenderer(r);
    const Scene s = makeScene(40, 303);
    const double quaternionError = reflectionError(s, mirrorY(s));
    std::printf("    quaternion mirror (-x, y, -z, w): worst |M R M - R'| %.3e over %u Gaussians\n",
                quaternionError, s.count);
    require(quaternionError < 1e-12, "(D) the quaternion mirror is not the reflected rotation");

    flipCheck(r, s, "40 Gaussians");
    const Scene large = makeScene(60000, 304);
    flipCheck(r, large, "60000 Gaussians");
    std::printf("(D) OK\n");
    r.shutdown();
}

// -- E. DebugView -----------------------------------------------------------------

// SHDegree ramp, as debug_visualize.slang defines it.
const glm::vec3 kDegreeStops[4] = {
    glm::vec3(0.15f, 0.25f, 0.95f),
    glm::vec3(0.15f, 0.85f, 0.25f),
    glm::vec3(0.95f, 0.85f, 0.15f),
    glm::vec3(0.95f, 0.20f, 0.15f),
};

// Records DebugVisualizer::record into a frame slot and copies the display image out.
std::vector<glm::vec4> recordDisplay(Device & device, DebugVisualizer & viz, DebugView mode,
                                     const ForwardPass & pass, float depthScale,
                                     float densityScale, uint32_t frame, uint32_t w, uint32_t h)
{
    Buffer readback(device, static_cast<VkDeviceSize>(w) * h * sizeof(glm::vec4));
    device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            viz.record(cmd, mode, pass, depthScale, densityScale, frame);
            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {w, h, 1};
            vkCmdCopyImageToBuffer(cmd, viz.displayImage(frame).getHandle(),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.handle(), 1,
                                   &region);
        });
    return download<glm::vec4>(readback, size_t(w) * h);
}

bool sameBytes(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(glm::vec4)) == 0;
}

// Checks every display mode of a size against CPU references, after a
// ForwardPass run with the given camera. The Renderer must be at the same size.
void checkDebugViews(Renderer & r, Device & device, ForwardPass & pass, DebugVisualizer & viz,
                     GaussianCloud & cloud, const Scene & s, const Camera & cam, uint32_t w,
                     uint32_t h, const char * label)
{
    const CameraUniforms u = renderUniforms(cam, w, h, s.count, kShDegree, false, false);
    pass.run(cloud.views(), u);
    const std::vector<glm::vec4> out = pass.downloadOutputImage();
    require(bitEqual(render(r), toFloats(out)), "(E) ForwardPass replica differs from the Renderer");
    const std::vector<float> depth = pass.downloadDepthImage();
    const size_t n = size_t(w) * h;

    // Alpha: the coverage channel, grayscale, exact copy of the forward alpha.
    const auto alpha0 = recordDisplay(device, viz, DebugView::Alpha, pass, 1.0f, 1.0f, 0, w, h);
    const auto alpha1 = recordDisplay(device, viz, DebugView::Alpha, pass, 1.0f, 1.0f, 1, w, h);
    double worstAlpha = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double a = out[i].a;
        worstAlpha = std::max({worstAlpha, std::fabs(alpha0[i].r - a), std::fabs(alpha0[i].g - a),
                               std::fabs(alpha0[i].b - a), std::fabs(alpha0[i].a - 1.0)});
    }
    require(worstAlpha == 0.0, "(E) Alpha view differs from the forward alpha");
    require(sameBytes(alpha0, alpha1), "(E) frame slots 0 and 1 differ for the same run");

    // Depth: frontmost-contributor depth over depthScale, clamped to [0, 1].
    const float depthScale = 12.0f;
    const auto depthView = recordDisplay(device, viz, DebugView::Depth, pass, depthScale, 1.0f, 0, w, h);
    double worstDepth = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const float expected = std::clamp(depth[i] / depthScale, 0.0f, 1.0f);
        worstDepth = std::max({worstDepth, std::fabs(double(depthView[i].r) - expected),
                               std::fabs(double(depthView[i].g) - expected),
                               std::fabs(double(depthView[i].b) - expected)});
    }
    require(worstDepth <= 1e-6, "(E) Depth view differs from depth / depthScale");

    // TileDensity: instances per tile over densityScale, clamped; scales below 1
    // clamp to 1. The tile ranges must also account for every instance.
    const std::vector<TileRange> ranges =
        download<TileRange>(pass.tileRanges(), size_t(u.tilesX) * u.tilesY);
    uint64_t instances = 0;
    for (const TileRange & range : ranges)
    {
        instances += range.last - range.first;
    }
    require(instances == pass.lastInstanceCount(), "(E) tile ranges do not cover the instance list");
    for (float densityScale : {8.0f, 0.5f})
    {
        const auto tiles = recordDisplay(device, viz, DebugView::TileDensity, pass, 1.0f, densityScale, 0, w, h);
        double worst = 0.0;
        for (uint32_t y = 0; y < h; ++y)
        {
            for (uint32_t x = 0; x < w; ++x)
            {
                const TileRange & range = ranges[(y / kTileSize) * u.tilesX + x / kTileSize];
                const float count = static_cast<float>(range.last - range.first);
                const float d = std::clamp(count / std::max(densityScale, 1.0f), 0.0f, 1.0f);
                const glm::vec4 expected(d, 0.2f * (1.0f - d), 1.0f - d, 1.0f);
                const glm::vec4 & got = tiles[size_t(y) * w + x];
                worst = std::max(worst, double(maxAbsComponent(got - expected)));
            }
        }
        require(worst <= 1e-6, "(E) TileDensity view differs from the tile count reference");
        std::printf("    %-18s TileDensity densityScale %.1f: worst %.3e\n", label, densityScale, worst);
    }

    // SHDegree: with no LOD every Gaussian is evaluated at the cap, so the view is
    // one ramp stop times coverage (1 - T), i.e. the forward alpha.
    for (uint32_t cap : {0u, 3u})
    {
        pass.run(cloud.views(), renderUniforms(cam, w, h, s.count, cap, false, false));
        const std::vector<glm::vec4> capOut = pass.downloadOutputImage();
        const auto heat = recordDisplay(device, viz, DebugView::SHDegree, pass, 1.0f, 1.0f, 0, w, h);
        double worst = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const glm::vec3 expected = kDegreeStops[cap] * capOut[i].a;
            worst = std::max(worst, double(maxAbsComponent(glm::vec3(heat[i]) - expected)));
        }
        require(worst <= 1e-5, "(E) SHDegree view differs from the cap's ramp stop");
        std::printf("    %-18s SHDegree cap %u: worst %.3e\n", label, cap, worst);
    }
    std::printf("    %-18s Alpha worst %.3e (exact), Depth worst %.3e, frame slots equal\n", label,
                worstAlpha, worstDepth);
}

void partE()
{
    std::printf("(E) DebugView\n");
    Renderer r(kWidth, kHeight);
    initRenderer(r);
    r.setCamera(makeCamera(kOffAxisEye, kWidth, kHeight));
    const Scene s = makeScene(40, 404);
    r.setGaussians(s.input(kShDegree));

    // Every view is presentation-only: the forward image does not move.
    const std::vector<float> finalImage = render(r);
    const DebugView views[] = {DebugView::Alpha, DebugView::Depth, DebugView::TileDensity,
                               DebugView::SHDegree, DebugView::Final};
    for (DebugView v : views)
    {
        r.setDebugView(v);
        require(bitEqual(render(r), finalImage), "(E) a debug view changed the forward image");
    }
    r.setDebugView(DebugView::Final);
    std::printf("    all five views leave the forward image bit-exact\n");

    Device & device = r.sabercatDevice();
    ForwardPass pass(device, kWidth, kHeight, SHADER_DIR);
    GaussianCloud cloud(device);
    fillCloud(cloud, s);
    DebugVisualizer viz(device, kWidth, kHeight, SHADER_DIR);
    checkDebugViews(r, device, pass, viz, cloud, s, makeCamera(kOffAxisEye, kWidth, kHeight), kWidth,
                    kHeight, "96x64");

    // Single Gaussian at depth 6: the depth view at its centre is 6 / depthScale.
    {
        const Scene one = singleGaussian();
        GaussianCloud single(device);
        fillCloud(single, one);
        pass.run(single.views(), renderUniforms(makeCamera(glm::vec3(0.0f), kWidth, kHeight), kWidth,
                                                kHeight, 1, kShDegree, false, false));
        const auto depthView = recordDisplay(device, viz, DebugView::Depth, pass, 12.0f, 1.0f, 0, kWidth, kHeight);
        const float centre = depthView[size_t(32) * kWidth + 48].r;
        require(std::fabs(centre - 0.5f) <= 1e-6f, "(E) single-Gaussian depth view is not 6 / 12 at its centre");
        std::printf("    single Gaussian at depth 6: depth view centre %.7f (analytic 0.5)\n", centre);
    }

    // Odd size: the Renderer, the ForwardPass and the visualizer all resize.
    const uint32_t w2 = 61;
    const uint32_t h2 = 37;
    r.setImageSize(w2, h2);
    pass.resize(w2, h2);
    viz.resize(w2, h2);
    const Camera cam2 = makeCamera(kOffAxisEye, w2, h2);
    r.setCamera(cam2);
    checkDebugViews(r, device, pass, viz, cloud, s, cam2, w2, h2, "61x37 resized");
    std::printf("(E) OK\n");
}

// -- F. sceneBounds / sceneSHDegree ---------------------------------------------

struct BoundsReference
{
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
    glm::vec3 focusCenter{0.0f};
    float focusRadius = 0.0f;
};

// Brute force: exact min/max, the median per axis from a sorted copy, and the
// 90th percentile of the distances to that centre at index floor(0.9 (n - 1)).
// When that percentile is zero, the fallback is the bounding-sphere radius.
BoundsReference bruteForceBounds(const Scene & s)
{
    BoundsReference b;
    const uint32_t n = s.count;
    b.min = b.max = glm::vec3(s.means[0], s.means[1], s.means[2]);
    for (uint32_t i = 0; i < n; ++i)
    {
        const glm::vec3 p(s.means[size_t(i) * 4], s.means[size_t(i) * 4 + 1], s.means[size_t(i) * 4 + 2]);
        b.min = glm::min(b.min, p);
        b.max = glm::max(b.max, p);
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        std::vector<float> values(n);
        for (uint32_t i = 0; i < n; ++i)
        {
            values[i] = s.means[size_t(i) * 4 + axis];
        }
        std::sort(values.begin(), values.end());
        b.focusCenter[axis] = values[n / 2];
    }
    std::vector<float> distances(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        const float dx = s.means[size_t(i) * 4] - b.focusCenter.x;
        const float dy = s.means[size_t(i) * 4 + 1] - b.focusCenter.y;
        const float dz = s.means[size_t(i) * 4 + 2] - b.focusCenter.z;
        distances[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    std::sort(distances.begin(), distances.end());
    b.focusRadius = distances[static_cast<size_t>(0.9 * double(n - 1))];
    if (!(b.focusRadius > 0.0f))
    {
        b.focusRadius = 0.5f * glm::length(b.max - b.min);
    }
    return b;
}

void checkBounds(const Renderer & r, const Scene & s, const char * label)
{
    const SceneBounds got = r.sceneBounds();
    const BoundsReference ref = bruteForceBounds(s);
    require(got.valid, "(F) bounds invalid with a scene loaded");
    require(got.min == ref.min && got.max == ref.max, "(F) min/max differ from brute force");
    require(got.focusCenter == ref.focusCenter, "(F) median focus centre differs from brute force");
    const double radiusError = std::fabs(double(got.focusRadius) - double(ref.focusRadius));
    const double relative = radiusError / std::max(double(ref.focusRadius), 1e-30);
    require(radiusError <= 1e-6 * std::max(double(ref.focusRadius), 1.0),
            "(F) p90 focus radius differs from brute force");
    std::printf("    %-18s min (%.3f, %.3f, %.3f) max (%.3f, %.3f, %.3f) focus (%.4f, %.4f, %.4f) "
                "r90 %.5f (ref %.5f, rel %.1e)\n",
                label, got.min.x, got.min.y, got.min.z, got.max.x, got.max.y, got.max.z,
                got.focusCenter.x, got.focusCenter.y, got.focusCenter.z, got.focusRadius,
                ref.focusRadius, relative);
}

void setMean(Scene & s, uint32_t i, const glm::vec3 & p)
{
    s.means[size_t(i) * 4] = p.x;
    s.means[size_t(i) * 4 + 1] = p.y;
    s.means[size_t(i) * 4 + 2] = p.z;
}

void partF()
{
    std::printf("(F) sceneBounds / sceneSHDegree\n");
    Renderer r(kWidth, kHeight);
    initRenderer(r);
    {
        const SceneBounds empty = r.sceneBounds();
        require(!empty.valid && r.sceneSHDegree() == 0 && r.gaussianCount() == 0,
                "(F) an empty renderer reports a scene");
        std::printf("    no scene: valid false, sceneSHDegree 0\n");
    }

    // Scale: 100k Gaussians, brute-forced on the host.
    const Scene big = makeScene(100000, 505);
    r.setGaussians(big.input(kShDegree));
    checkBounds(r, big, "100k random");

    // Outlier shell: 90% in a unit cube, 10% on a sphere of radius 300.
    Scene shell = makeScene(1000, 506);
    Rng rng(507);
    for (uint32_t i = 0; i < shell.count; ++i)
    {
        if (i % 10 == 0)
        {
            glm::vec3 d(rng.next() - 0.5f, rng.next() - 0.5f, rng.next() - 0.5f);
            setMean(shell, i, glm::normalize(d) * 300.0f);
        }
        else
        {
            setMean(shell, i, glm::vec3(rng.next() - 0.5f, rng.next() - 0.5f, rng.next() - 0.5f) * 2.0f);
        }
    }
    r.setGaussians(shell.input(kShDegree));
    checkBounds(r, shell, "outlier shell");
    const SceneBounds b = r.sceneBounds();
    require(b.max.x > 200.0f || b.max.y > 200.0f || b.max.z > 200.0f || b.min.x < -200.0f || b.min.y < -200.0f
                || b.min.z < -200.0f,
            "(F) fixture: shell does not reach the bounds");
    require(b.focusRadius < 3.0f, "(F) the shell pulled the focus radius out");

    // One Gaussian: a degenerate sphere.
    r.setGaussians(singleGaussian().input(0));
    checkBounds(r, singleGaussian(), "one Gaussian");
    const SceneBounds one = r.sceneBounds();
    require(one.focusRadius == 0.0f && one.radius() == 0.0f && one.min == one.max,
            "(F) one Gaussian has a nonzero extent");

    // Degenerate: 95% identical, 5% far away, so the 90th percentile is zero and
    // the focus radius falls back to the bounding-sphere radius.
    Scene degenerate = makeScene(100, 508);
    for (uint32_t i = 0; i < degenerate.count; ++i)
    {
        setMean(degenerate, i, i < 95 ? glm::vec3(1.0f, 2.0f, 3.0f) : glm::vec3(10.0f, 0.0f, 0.0f));
    }
    r.setGaussians(degenerate.input(kShDegree));
    checkBounds(r, degenerate, "95% coincident");
    require(std::fabs(r.sceneBounds().focusRadius - 0.5f * glm::length(glm::vec3(9.0f, 2.0f, 3.0f))) < 1e-5f,
            "(F) fallback radius is not the bounding-sphere radius");

    // sceneSHDegree is the loaded scene's degree, not the cap.
    for (uint32_t deg = 0; deg <= kShDegree; ++deg)
    {
        r.setGaussians(makeScene(20, 509).input(deg));
        require(r.sceneSHDegree() == deg, "(F) sceneSHDegree does not match the input degree");
    }
    r.setGaussians(makeScene(20, 510).input(kShDegree));
    r.setMaxSHDegree(0);
    require(r.sceneSHDegree() == kShDegree, "(F) setMaxSHDegree changed sceneSHDegree");
    std::printf("    sceneSHDegree follows the input degree 0..3; setMaxSHDegree(0) leaves it at 3\n");

    // A loaded .splat: DC only, bounds read from the file.
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "sabercat_renderer_test_bounds.splat";
    writeSplat(path, singleGaussian());
    r.loadScene(path.string().c_str());
    std::filesystem::remove(path);
    require(r.sceneSHDegree() == 0, "(F) a .splat scene should be DC only");
    require(r.sceneBounds().min == glm::vec3(0.0f, 0.0f, -6.0f) && r.sceneBounds().max == r.sceneBounds().min,
            "(F) .splat bounds are not the file's position");
    std::printf("    .splat: sceneSHDegree 0, bounds read from the file\n");
    std::printf("(F) OK\n");
    r.shutdown();
}

// -- G. Rejected calls and no-op paths ----------------------------------------------

bool boundsEqual(const SceneBounds & a, const SceneBounds & b)
{
    return a.valid == b.valid && a.min == b.min && a.max == b.max && a.focusCenter == b.focusCenter
           && a.focusRadius == b.focusRadius;
}

void partG()
{
    std::printf("(G) rejected calls and no-op paths\n");
    {
        Renderer r(kWidth, kHeight);
        initRenderer(r);
        expectThrow("reserveGaussians, no scene", [&] { r.reserveGaussians(8); });
        expectThrow("setGaussianCount, no scene", [&] { r.setGaussianCount(1); });
        require(r.gaussianCapacity() == 0 && r.gaussianCount() == 0 && !r.sceneBounds().valid,
                "(G) no-scene state changed by a rejected call");
        r.setCamera(makeCamera(kOffAxisEye, kWidth, kHeight));
        r.forward();
        r.forward(RenderMode::Differentiable);
        expectThrow("readImage, nothing rendered", [&] { r.readImage(); });
        std::printf("    forward with no scene: no-op, no throw\n");
        r.shutdown();
    }
    {
        Renderer r(kWidth, kHeight);
        r.forward();
        std::printf("    forward before initialize: no-op, no throw\n");
    }

    Renderer r(kWidth, kHeight);
    initRenderer(r);
    const Camera cam = makeCamera(kOffAxisEye, kWidth, kHeight);
    r.setCamera(cam);
    const Scene s = makeScene(40, 606);
    r.setGaussians(s.input(kShDegree));
    const std::vector<float> before = render(r);
    const SceneBounds boundsBefore = r.sceneBounds();

    expectThrow("setImageSize(0, h)", [&] { r.setImageSize(0, kHeight); });
    expectThrow("setImageSize(w, 0)", [&] { r.setImageSize(kWidth, 0); });
    require(r.width() == kWidth && r.height() == kHeight, "(G) a rejected setImageSize changed the size");
    require(bitEqual(render(r), before), "(G) a rejected setImageSize changed the image");

    expectThrow("setGaussians, count 0", [&] {
        GaussianHostInput in = s.input(kShDegree);
        in.count = 0;
        r.setGaussians(in);
    });
    expectThrow("setGaussians, null means", [&] {
        GaussianHostInput in = s.input(kShDegree);
        in.means = nullptr;
        r.setGaussians(in);
    });
    expectThrow("setGaussians, shDegree 4", [&] { r.setGaussians(s.input(4)); });
    require(r.gaussianCount() == s.count, "(G) a rejected setGaussians replaced the scene");
    require(boundsEqual(r.sceneBounds(), boundsBefore), "(G) a rejected setGaussians changed the bounds");
    require(bitEqual(render(r), before), "(G) a rejected setGaussians changed the image");

    std::vector<float> means(size_t(s.count) * 4), scales(size_t(s.count) * 4), rotations(size_t(s.count) * 4),
        opacities(s.count), sh(size_t(s.count) * kShCoefficients * 4);
    GaussianHostOutput out;
    out.count = s.count + 1;
    out.means = means.data();
    out.scales = scales.data();
    out.rotations = rotations.data();
    out.opacities = opacities.data();
    out.shCoefficients = sh.data();
    expectThrow("readGaussians, count mismatch", [&] { r.readGaussians(out); });
    GaussianHostOutput nullOut = out;
    nullOut.count = s.count;
    nullOut.means = nullptr;
    expectThrow("readGaussians, null array", [&] { r.readGaussians(nullOut); });
    out.count = s.count;
    r.readGaussians(out);
    require(std::memcmp(means.data(), s.means.data(), means.size() * sizeof(float)) == 0
                && std::memcmp(sh.data(), s.sh.data(), sh.size() * sizeof(float)) == 0,
            "(G) readGaussians does not return the scene that was set");

    require(!r.reserveGaussians(1), "(G) reserveGaussians at or below capacity must return false");
    require(r.gaussianCapacity() == s.count, "(G) a refused reserve changed the capacity");

    // Invalid cameras: forward is a no-op and the previous image stays readable.
    r.setCamera(Camera{});
    r.forward();
    require(bitEqual(r.readImage(), before), "(G) forward with a default camera changed the image");
    Camera noFocal = cam;
    noFocal.fx = 0.0f;
    r.setCamera(noFocal);
    r.forward();
    require(bitEqual(r.readImage(), before), "(G) forward with fx = 0 changed the image");
    r.setCamera(cam);
    require(bitEqual(render(r), before), "(G) restoring the camera does not reproduce the image");
    std::printf("    all rejected calls threw with the state unchanged; invalid-camera forward is a no-op\n");
    std::printf("(G) OK\n");
    r.shutdown();
}

} // namespace

int main()
{
    int result = 0;
    try
    {
        partA();
        partB();
        partC();
        partD();
        partE();
        partF();
        partG();
        std::printf("renderer_test OK\n");
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "renderer_test FAILED: %s\n", e.what());
        result = 1;
    }
    return result;
}
