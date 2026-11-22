// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sabercat Compare compositor: one thread per view pixel, writing an RGBA8 UNORM
// storage image. The rules are the ones in apps/compare/compare_layout.h
// (windowToImage, classifyPanePixel, paddingColorFor, displayTransform,
// mapValueAt, isValidDepth, hatchValue, depthMapT, depthDiffT) and apps/compare/flip_hdr.cpp
// (displayRgb). compositor.h describes the same rules in words; keep the three in step.
//
// Rounding: the target is a UNORM format, so a float store is converted by the
// hardware with round-to-nearest. The CPU reference rounds with floor(v * 255 + 0.5);
// the two agree except on exact ties, and the selftest allows 1/255.
//
// Alpha of the composite is always 1 (the export drops alpha).
//
// Every constant and struct here mirrors compositor.cpp. The struct layouts are
// std430 with 16-byte-aligned records; compositor.cpp checks the offsets with
// static_assert.

static const uint kNoSlot = 0xFFFFFFFFu;

static const uint kModePanes = 0u;
static const uint kModeWipe = 1u;
static const uint kModeFlicker = 2u;
static const uint kModeDiff = 3u;
static const uint kModeDepth = 4u;
static const uint kModeAlpha = 5u;
static const uint kModeDepthDiff = 6u;
static const uint kModeAlphaDiff = 7u;
static const uint kModeFlip = 8u;

static const uint kToneAces = 0u;
static const uint kToneHable = 1u;

static const uint kEncodingLinear = 1u;

static const float kBackground = 0.12f;
// Light stripe of the no-data hatch; kBackground is the dark one (compare_layout.h).
static const float kHatchLight = 0.22f;
static const float kMaxLinear = 65504.0f;
static const float kMaxExposureStops = 32.0f;
static const float kPixelLimit = 1073741824.0f; // 2^30, as floorToPixel()

// One visible source. 48 bytes.
struct SourceRecord
{
    int offsetX;
    int offsetY;
    uint width;
    uint height;
    uint encoding;      // 0 Display, 1 Linear
    uint premultiplied; // 0 or 1
    uint toneMapper;    // 0 Aces, 1 Hable, 2 Reinhard
    float exposureStops;
    float4 background;  // rgb: background of premultiplied compositing; w unused
};

// One pane. 32 bytes. rect is (x, y, width, height) in view pixels.
struct PaneRecord
{
    int4 rect;
    uint slot;
    uint pad0;
    uint pad1;
    uint pad2;
};

// The whole parameter block, one element of paramsBuffer. 464 bytes.
// Depth of a source, for the depth and alpha maps, is a uint4 {stride, front, coverage, 0}:
// stride is the number of floats per pixel (0 = the source has no depth); front is the
// component the depth maps use; coverage is the component holding alpha, kNoSlot when the
// color alpha is the coverage. Sample index: (y * width + x) * stride + component.
struct CompositeParams
{
    int4 extent;          // padded extent: x, y, width, height
    uint4 view;           // view width, view height, mode, nearest
    uint4 slots;          // sourceA slot, sourceB slot, visible source count, pane count
                          // (Depth and Alpha put the chosen source in sourceA)
    float4 transform;     // panX, panY, zoom, wipe position (already clamped by the host)
    uint4 flags;          // x: wipe is vertical (1) or horizontal (0); y: map uses B (informational); rest unused
    float4 misc;          // x: diff gain; y: depth range min; z: depth range max; w unused
    PaneRecord panes[4];
    SourceRecord sources[4];
    uint4 depthA;         // depth record of slot A (of the chosen source for Depth and Alpha)
    uint4 depthB;         // depth record of slot B
    int4 flipRect;        // Flip: the overlap the FLIP map covers (x, y, width, height), canvas pixels
};

[vk::binding(0, 0)] [vk::image_format("rgba8")] RWTexture2D<float4> composite;
[vk::binding(1, 0)] StructuredBuffer<float4> source0;
[vk::binding(2, 0)] StructuredBuffer<float4> source1;
[vk::binding(3, 0)] StructuredBuffer<float4> source2;
[vk::binding(4, 0)] StructuredBuffer<float4> source3;
[vk::binding(5, 0)] StructuredBuffer<CompositeParams> paramsBuffer;
// 256 magma entries, rgb in [0,1] (byte / 255), built from colorizeMagma().
[vk::binding(6, 0)] StructuredBuffer<float4> magma;
// Depth buffers of the A and B slots, flat floats laid out by the depth records above.
// A slot without depth binds a small dummy buffer that is never read.
[vk::binding(7, 0)] StructuredBuffer<float> depthA;
[vk::binding(8, 0)] StructuredBuffer<float> depthB;
// Flip: the FLIP error map over flipRect, one float per pixel, row-major. Other modes bind a
// dummy buffer that is never read.
[vk::binding(9, 0)] StructuredBuffer<float> flipMap;

float4 loadSample(uint slot, uint index)
{
    if (slot == 0u)
    {
        return source0[index];
    }
    if (slot == 1u)
    {
        return source1[index];
    }
    if (slot == 2u)
    {
        return source2[index];
    }
    return source3[index];
}

// Clamp to [0,1]; NaN and negatives give 0 (clampUnit in compare_layout.cpp).
float clampUnit(float v)
{
    return (v > 0.0f) ? min(v, 1.0f) : 0.0f;
}

// sRGB transfer function (IEC 61966-2-1), as srgbTransfer() in compare_layout.cpp.
float srgbTransfer(float linear)
{
    return (linear <= 0.0031308f) ? 12.92f * linear : 1.055f * pow(linear, 1.0f / 2.4f) - 0.055f;
}

// Hable's filmic curve before normalization.
float hableCurve(float x)
{
    const float a = 0.15f;
    const float b = 0.50f;
    const float c = 0.10f;
    const float d = 0.20f;
    const float e = 0.02f;
    const float f = 0.30f;
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

float toneMap(float x, uint toneMapper)
{
    if (toneMapper == kToneAces)
    {
        // Narkowicz's fit of the ACES filmic curve.
        const float a = 2.51f;
        const float b = 0.03f;
        const float c = 2.43f;
        const float d = 0.59f;
        const float e = 0.14f;
        return (x * (a * x + b)) / (x * (c * x + d) + e);
    }
    if (toneMapper == kToneHable)
    {
        return hableCurve(x) / hableCurve(11.2f);
    }
    return x / (1.0f + x);
}

// displayTransform() of compare_layout.cpp. The exposure arrives already sanitized
// (NaN is 0, clamped to [-32, 32]) by the host.
float displayTransform(float linear, uint toneMapper, float exposureStops)
{
    float value = linear * exp2(exposureStops);
    value = (value > 0.0f) ? value : 0.0f;
    value = (value < kMaxLinear) ? value : kMaxLinear;
    return srgbTransfer(clampUnit(toneMap(value, toneMapper)));
}

// One display-space channel of a pane sample. Premultiplied samples are composited
// over the source background first (holdback), in the source's own encoding.
float displayChannel(float sample, float holdback, float background, SourceRecord record)
{
    const float v = sample + holdback * background;
    if (record.encoding == kEncodingLinear)
    {
        return displayTransform(v, record.toneMapper, record.exposureStops);
    }
    return clampUnit(v);
}

// Display color of the source pixel (x, y) (local integer coordinates), as paddingColorFor() sees it.
float3 displayColorAt(uint slot, SourceRecord record, uint x, uint y)
{
    const float4 s = loadSample(slot, y * record.width + x);
    const float holdback = (record.premultiplied != 0u) ? (1.0f - clampUnit(s.a)) : 0.0f;
    return float3(displayChannel(s.r, holdback, record.background.r, record),
                  displayChannel(s.g, holdback, record.background.g, record),
                  displayChannel(s.b, holdback, record.background.b, record));
}

// Bilinear display color at continuous local coordinates. Taps are clamped to the
// source rectangle, so padding is never read.
float3 bilinearColorAt(uint slot, SourceRecord record, float localX, float localY)
{
    const float tx = localX - 0.5f;
    const float ty = localY - 0.5f;
    const float fx0 = floor(tx);
    const float fy0 = floor(ty);
    const float wx = tx - fx0;
    const float wy = ty - fy0;
    const int maxX = int(record.width) - 1;
    const int maxY = int(record.height) - 1;
    const int x0 = int(fx0);
    const int y0 = int(fy0);
    const uint xa = uint(min(max(x0, 0), maxX));
    const uint xb = uint(min(max(x0 + 1, 0), maxX));
    const uint ya = uint(min(max(y0, 0), maxY));
    const uint yb = uint(min(max(y0 + 1, 0), maxY));
    const float3 c00 = displayColorAt(slot, record, xa, ya);
    const float3 c10 = displayColorAt(slot, record, xb, ya);
    const float3 c01 = displayColorAt(slot, record, xa, yb);
    const float3 c11 = displayColorAt(slot, record, xb, yb);
    return lerp(lerp(c00, c10, wx), lerp(c01, c11, wx), wy);
}

// Display value of a source pixel for Diff, as displayRgb() in flip_hdr.cpp: Display
// samples as stored, Linear samples clamped to [0,1] and sRGB encoded. Premultiplied
// rgb is used unchanged.
float3 diffDisplayAt(uint slot, SourceRecord record, uint x, uint y)
{
    const float4 s = loadSample(slot, y * record.width + x);
    if (record.encoding == kEncodingLinear)
    {
        return float3(srgbTransfer(clampUnit(s.r)), srgbTransfer(clampUnit(s.g)), srgbTransfer(clampUnit(s.b)));
    }
    return s.rgb;
}

bool inRect(int rx, int ry, int rw, int rh, int x, int y)
{
    return rw > 0 && rh > 0 && x >= rx && y >= ry && x - rx < rw && y - ry < rh;
}

// Counterpart of padding (paddingCounterpartSlot() in compositor.cpp): the other
// source of the A/B pair when it is visible, else the first other visible source.
uint counterpartSlot(uint slot, CompositeParams p)
{
    const uint a = p.slots.x;
    const uint b = p.slots.y;
    const uint count = p.slots.z;
    if (slot == a && b < count && b != slot)
    {
        return b;
    }
    if (slot == b && a < count && a != slot)
    {
        return a;
    }
    for (uint k = 0u; k < count; k++)
    {
        if (k != slot)
        {
            return k;
        }
    }
    return kNoSlot;
}

// Window pixel center mapped to image coordinates through pane geometry
// (windowToImage in compare_layout.cpp).
float imageCoordinate(float windowCenter, float paneCenter, float pan, float zoom)
{
    return pan + (windowCenter - paneCenter) / zoom;
}

int floorToPixel(float value)
{
    return int(clamp(floor(value), -kPixelLimit, kPixelLimit));
}

float3 backgroundColor()
{
    return float3(kBackground, kBackground, kBackground);
}

// Panes, Wipe and Flicker: the pane pixel rules of classifyPanePixel() and paddingColorFor().
float3 shadeView(uint px, uint py, CompositeParams p)
{
    const float wx = float(px) + 0.5f;
    const float wy = float(py) + 0.5f;
    const float zoom = p.transform.z;

    int paneIndex = -1;
    const uint paneCount = p.slots.w;
    for (uint i = 0u; i < paneCount; i++)
    {
        const int4 r = p.panes[i].rect;
        if (inRect(r.x, r.y, r.z, r.w, int(px), int(py)))
        {
            paneIndex = int(i);
            break;
        }
    }
    if (paneIndex < 0)
    {
        return backgroundColor();
    }

    const int4 paneRect = p.panes[paneIndex].rect;
    const float centerX = float(paneRect.x) + float(paneRect.z) * 0.5f;
    const float centerY = float(paneRect.y) + float(paneRect.w) * 0.5f;
    const float imageX = imageCoordinate(wx, centerX, p.transform.x, zoom);
    const float imageY = imageCoordinate(wy, centerY, p.transform.y, zoom);
    const int pixelX = floorToPixel(imageX);
    const int pixelY = floorToPixel(imageY);

    // Outside the padded extent: background.
    if (!inRect(p.extent.x, p.extent.y, p.extent.z, p.extent.w, pixelX, pixelY))
    {
        return backgroundColor();
    }

    uint slot = p.panes[paneIndex].slot;
    if (p.view.z == kModeWipe)
    {
        // A where the coordinate is less than the wipe position, B otherwise.
        const float coordinate = (p.flags.x != 0u) ? imageX : imageY;
        slot = (coordinate < p.transform.w) ? p.slots.x : p.slots.y;
    }

    const SourceRecord record = p.sources[slot];
    if (inRect(record.offsetX, record.offsetY, int(record.width), int(record.height), pixelX, pixelY))
    {
        // Image pixel.
        if (p.view.w != 0u)
        {
            return displayColorAt(slot, record, uint(pixelX - record.offsetX), uint(pixelY - record.offsetY));
        }
        return bilinearColorAt(slot, record, imageX - float(record.offsetX), imageY - float(record.offsetY));
    }

    // Padding: the inverted counterpart pixel (nearest), or background when the counterpart has none.
    const uint counterpart = counterpartSlot(slot, p);
    if (counterpart == kNoSlot)
    {
        return backgroundColor();
    }
    const SourceRecord other = p.sources[counterpart];
    const int localX = pixelX - other.offsetX;
    const int localY = pixelY - other.offsetY;
    if (localX < 0 || localY < 0 || uint(localX) >= other.width || uint(localY) >= other.height)
    {
        return backgroundColor();
    }
    return float3(1.0f, 1.0f, 1.0f) - displayColorAt(counterpart, other, uint(localX), uint(localY));
}

// ---- Map helpers (compare_layout.h: isValidDepth, hatchValue, depthMapT, depthDiffT) ----

// isValidDepth(): finite and greater than zero. NaN and infinities are no data.
bool isValidDepth(float d)
{
    return isfinite(d) && d > 0.0f;
}

// Hatch display value at a VIEW pixel: diagonal bands of 8 px between the dark and the
// light stripe, so a no-data pixel never reads as a map value.
float hatchValue(uint px, uint py)
{
    const uint band = ((px + py) >> 3u) & 1u;
    return (band != 0u) ? kHatchLight : kBackground;
}

// depthMapT(): position of d in [min, max], clamped; 0 when the span is empty.
float depthMapT(float d, float rangeMin, float rangeMax)
{
    const float span = rangeMax - rangeMin;
    if (span > 0.0f)
    {
        return clampUnit((d - rangeMin) / span);
    }
    return 0.0f;
}

// depthDiffT(): |a - b| as a fraction of the span, clamped. With an empty span any
// difference is the top of the map and no difference is the bottom.
float depthDiffT(float a, float b, float rangeMin, float rangeMax)
{
    const float diff = abs(a - b);
    const float span = rangeMax - rangeMin;
    if (span > 0.0f)
    {
        return clampUnit(diff / span);
    }
    if (diff > 0.0f)
    {
        return 1.0f;
    }
    return 0.0f;
}

// colorizeMagma() of one map value: index floor(clampUnit(t) * 255 + 0.5) into the table.
float3 magmaColor(float t)
{
    const uint index = uint(clampUnit(t) * 255.0f + 0.5f);
    return magma[index].rgb;
}

// True when the record has a front depth to map: a stride, and a front component (the host
// writes kNoSlot when the source has depth but no valid front component).
bool hasFrontDepth(uint4 depthRecord)
{
    return depthRecord.x != 0u && depthRecord.y != kNoSlot;
}

// One component of the depth sample at local pixel (x, y) of a source `width` pixels wide.
float depthSample(StructuredBuffer<float> buffer, uint4 depthRecord, uint x, uint y, uint comp, uint width)
{
    return buffer[(y * width + x) * depthRecord.x + comp];
}

// Coverage of local pixel (x, y) of a source, in [0,1]: the coverage component of the depth
// buffer when the record names one, else the color alpha. NaN is 0 (clampUnit).
float coverageAt(StructuredBuffer<float> buffer, uint4 depthRecord, uint slot, SourceRecord record, uint x, uint y)
{
    float a = 0.0f;
    if (depthRecord.x != 0u && depthRecord.z != kNoSlot)
    {
        a = depthSample(buffer, depthRecord, x, y, depthRecord.z, record.width);
    }
    else
    {
        a = loadSample(slot, y * record.width + x).a;
    }
    return clampUnit(a);
}

// Image pixel (floor of the image coordinate) of a VIEW pixel in the single-pane map modes:
// one pane covering the view (computePanes, SingleScreen), as shadeDiff() has always used.
int2 singlePanePixel(uint px, uint py, CompositeParams p)
{
    const float zoom = p.transform.z;
    const float centerX = float(p.view.x) * 0.5f;
    const float centerY = float(p.view.y) * 0.5f;
    const float imageX = imageCoordinate(float(px) + 0.5f, centerX, p.transform.x, zoom);
    const float imageY = imageCoordinate(float(py) + 0.5f, centerY, p.transform.y, zoom);
    return int2(floorToPixel(imageX), floorToPixel(imageY));
}

// ---- Map modes -------------------------------------------------------------------

// Diff: mean over RGB of |A - B| (display values) times the gain, clamped to 1, through
// the magma table. Both samples are nearest. A pixel that either source lacks is
// padding and takes the maximum (mapValueAt). Outside the padded extent: background.
float3 shadeDiff(uint px, uint py, CompositeParams p)
{
    const int2 pixel = singlePanePixel(px, py, p);
    const int pixelX = pixel.x;
    const int pixelY = pixel.y;
    if (!inRect(p.extent.x, p.extent.y, p.extent.z, p.extent.w, pixelX, pixelY))
    {
        return backgroundColor();
    }

    const uint slotA = p.slots.x;
    const uint slotB = p.slots.y;
    const SourceRecord recordA = p.sources[slotA];
    const SourceRecord recordB = p.sources[slotB];
    const bool inA = inRect(recordA.offsetX, recordA.offsetY, int(recordA.width), int(recordA.height), pixelX, pixelY);
    const bool inB = inRect(recordB.offsetX, recordB.offsetY, int(recordB.width), int(recordB.height), pixelX, pixelY);

    // mapPaddingValue() is 1.0.
    float value = 1.0f;
    if (inA && inB)
    {
        const float3 a = diffDisplayAt(slotA, recordA, uint(pixelX - recordA.offsetX), uint(pixelY - recordA.offsetY));
        const float3 b = diffDisplayAt(slotB, recordB, uint(pixelX - recordB.offsetX), uint(pixelY - recordB.offsetY));
        const float mean = (abs(a.r - b.r) + abs(a.g - b.g) + abs(a.b - b.b)) / 3.0f;
        value = min(1.0f, mean * p.misc.x);
    }
    return magmaColor(value);
}

// Depth: the front depth of the chosen source (slot A here) through the magma table.
// No data (an invalid depth, or a source without depth) and padding are hatched.
float3 shadeDepth(uint px, uint py, CompositeParams p)
{
    const int2 pixel = singlePanePixel(px, py, p);
    if (!inRect(p.extent.x, p.extent.y, p.extent.z, p.extent.w, pixel.x, pixel.y))
    {
        return backgroundColor();
    }

    const SourceRecord record = p.sources[p.slots.x];
    if (!hasFrontDepth(p.depthA) || !inRect(record.offsetX, record.offsetY, int(record.width), int(record.height), pixel.x, pixel.y))
    {
        const float h = hatchValue(px, py);
        return float3(h, h, h);
    }

    const float d = depthSample(depthA, p.depthA, uint(pixel.x - record.offsetX), uint(pixel.y - record.offsetY), p.depthA.y, record.width);
    if (!isValidDepth(d))
    {
        const float h = hatchValue(px, py);
        return float3(h, h, h);
    }
    return magmaColor(depthMapT(d, p.misc.y, p.misc.z));
}

// Alpha: the coverage of the chosen source as gray (no sRGB encoding). Padding is hatched.
float3 shadeAlpha(uint px, uint py, CompositeParams p)
{
    const int2 pixel = singlePanePixel(px, py, p);
    if (!inRect(p.extent.x, p.extent.y, p.extent.z, p.extent.w, pixel.x, pixel.y))
    {
        return backgroundColor();
    }

    const SourceRecord record = p.sources[p.slots.x];
    if (!inRect(record.offsetX, record.offsetY, int(record.width), int(record.height), pixel.x, pixel.y))
    {
        const float h = hatchValue(px, py);
        return float3(h, h, h);
    }

    const float a = coverageAt(depthA, p.depthA, p.slots.x, record, uint(pixel.x - record.offsetX), uint(pixel.y - record.offsetY));
    return float3(a, a, a);
}

// DepthDiff: |dA - dB| through the magma table where both sources have valid depth. Where
// exactly one has, the pixel takes the top of the map; where neither does, it is hatched.
// Padding (not inside both sources) takes the top too.
float3 shadeDepthDiff(uint px, uint py, CompositeParams p)
{
    const int2 pixel = singlePanePixel(px, py, p);
    if (!inRect(p.extent.x, p.extent.y, p.extent.z, p.extent.w, pixel.x, pixel.y))
    {
        return backgroundColor();
    }

    const SourceRecord recordA = p.sources[p.slots.x];
    const SourceRecord recordB = p.sources[p.slots.y];
    const bool inA = inRect(recordA.offsetX, recordA.offsetY, int(recordA.width), int(recordA.height), pixel.x, pixel.y);
    const bool inB = inRect(recordB.offsetX, recordB.offsetY, int(recordB.width), int(recordB.height), pixel.x, pixel.y);
    if (!inA || !inB)
    {
        // mapPaddingValue() is 1.0.
        return magmaColor(1.0f);
    }

    // A source without a front depth is invalid everywhere; its buffer is never read.
    float dA = 0.0f;
    float dB = 0.0f;
    bool validA = false;
    bool validB = false;
    if (hasFrontDepth(p.depthA))
    {
        dA = depthSample(depthA, p.depthA, uint(pixel.x - recordA.offsetX), uint(pixel.y - recordA.offsetY), p.depthA.y, recordA.width);
        validA = isValidDepth(dA);
    }
    if (hasFrontDepth(p.depthB))
    {
        dB = depthSample(depthB, p.depthB, uint(pixel.x - recordB.offsetX), uint(pixel.y - recordB.offsetY), p.depthB.y, recordB.width);
        validB = isValidDepth(dB);
    }

    if (validA && validB)
    {
        return magmaColor(depthDiffT(dA, dB, p.misc.y, p.misc.z));
    }
    if (validA || validB)
    {
        return magmaColor(1.0f);
    }
    const float h = hatchValue(px, py);
    return float3(h, h, h);
}

// AlphaDiff: min(1, |aA - aB| * gain) through the magma table, with the coverage of each
// source. Padding (not inside both sources) takes the top of the map.
float3 shadeAlphaDiff(uint px, uint py, CompositeParams p)
{
    const int2 pixel = singlePanePixel(px, py, p);
    if (!inRect(p.extent.x, p.extent.y, p.extent.z, p.extent.w, pixel.x, pixel.y))
    {
        return backgroundColor();
    }

    const SourceRecord recordA = p.sources[p.slots.x];
    const SourceRecord recordB = p.sources[p.slots.y];
    const bool inA = inRect(recordA.offsetX, recordA.offsetY, int(recordA.width), int(recordA.height), pixel.x, pixel.y);
    const bool inB = inRect(recordB.offsetX, recordB.offsetY, int(recordB.width), int(recordB.height), pixel.x, pixel.y);
    if (!inA || !inB)
    {
        // mapPaddingValue() is 1.0.
        return magmaColor(1.0f);
    }

    const float aA = coverageAt(depthA, p.depthA, p.slots.x, recordA, uint(pixel.x - recordA.offsetX), uint(pixel.y - recordA.offsetY));
    const float aB = coverageAt(depthB, p.depthB, p.slots.y, recordB, uint(pixel.x - recordB.offsetX), uint(pixel.y - recordB.offsetY));
    return magmaColor(min(1.0f, abs(aA - aB) * p.misc.x));
}

// FLIP: the error map over the overlap (flipRect) through the magma table. Padding (inside
// the extent, outside flipRect) takes the top of the map and is never run through FLIP.
// Outside the padded extent: background.
float3 shadeFlip(uint px, uint py, CompositeParams p)
{
    const int2 pixel = singlePanePixel(px, py, p);
    if (!inRect(p.extent.x, p.extent.y, p.extent.z, p.extent.w, pixel.x, pixel.y))
    {
        return backgroundColor();
    }

    const int4 r = p.flipRect;
    if (!inRect(r.x, r.y, r.z, r.w, pixel.x, pixel.y))
    {
        // mapPaddingValue() is 1.0.
        return magmaColor(1.0f);
    }

    const uint index = uint(pixel.y - r.y) * uint(r.z) + uint(pixel.x - r.x);
    return magmaColor(flipMap[index]);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const CompositeParams p = paramsBuffer[0];
    if (tid.x >= p.view.x || tid.y >= p.view.y)
    {
        return;
    }

    float3 color;
    if (p.view.z == kModeDiff)
    {
        color = shadeDiff(tid.x, tid.y, p);
    }
    else if (p.view.z == kModeDepth)
    {
        color = shadeDepth(tid.x, tid.y, p);
    }
    else if (p.view.z == kModeAlpha)
    {
        color = shadeAlpha(tid.x, tid.y, p);
    }
    else if (p.view.z == kModeDepthDiff)
    {
        color = shadeDepthDiff(tid.x, tid.y, p);
    }
    else if (p.view.z == kModeAlphaDiff)
    {
        color = shadeAlphaDiff(tid.x, tid.y, p);
    }
    else if (p.view.z == kModeFlip)
    {
        color = shadeFlip(tid.x, tid.y, p);
    }
    else
    {
        color = shadeView(tid.x, tid.y, p);
    }
    composite[tid.xy] = float4(color, 1.0f);
}
