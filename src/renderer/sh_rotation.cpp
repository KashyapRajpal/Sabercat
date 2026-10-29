// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sh_rotation.h"

#include <cmath>
#include <algorithm>

namespace sabercat
{
namespace sh_rotation
{

namespace
{

// --- Real SH basis, band by band ---
// Exactly the basis forward_common.slangh evaluates (and the spec's Appendix
// A: Condon-Shortley phase, m ascending), in double for the rotation solve.
constexpr double kShC1 = 0.4886025119029199;
constexpr double kShC2[5] = {1.0925484305920792,
                             -1.0925484305920792,
                             0.31539156525252005,
                             -1.0925484305920792,
                             0.5462742152960396};
constexpr double kShC3[7] = {-0.5900435899266435,
                             2.890611442640554,
                             -0.4570457994644658,
                             0.3731763325901154,
                             -0.4570457994644658,
                             1.445305721320277,
                             -0.5900435899266435};

// One SH band's basis functions at direction d, exactly as the shader
// evaluates them (DC excluded -- it is rotation invariant).
void shBasisBand(int band, double x, double y, double z, double * out)
{
    if (band == 1)
    {
        out[0] = -kShC1 * y;
        out[1] = kShC1 * z;
        out[2] = -kShC1 * x;
    }
    else if (band == 2)
    {
        out[0] = kShC2[0] * x * y;
        out[1] = kShC2[1] * y * z;
        out[2] = kShC2[2] * (2.0 * z * z - x * x - y * y);
        out[3] = kShC2[3] * x * z;
        out[4] = kShC2[4] * (x * x - y * y);
    }
    else
    {
        const double xx = x * x, yy = y * y, zz = z * z;
        out[0] = kShC3[0] * y * (3.0 * xx - yy);
        out[1] = kShC3[1] * x * y * z;
        out[2] = kShC3[2] * y * (4.0 * zz - xx - yy);
        out[3] = kShC3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
        out[4] = kShC3[4] * x * (4.0 * zz - xx - yy);
        out[5] = kShC3[5] * z * (xx - yy);
        out[6] = kShC3[6] * x * (xx - 3.0 * yy);
    }
}

// Solves the dense n x n system in place (Gauss-Jordan, partial pivoting):
// returns inverse(a). n <= 7, and `a` is a well-conditioned Gram matrix.
BandMatrix invert(BandMatrix a, int n)
{
    BandMatrix inv{};
    for (int i = 0; i < n; ++i)
    {
        inv[i][i] = 1.0;
    }
    for (int col = 0; col < n; ++col)
    {
        int pivot = col;
        for (int r = col + 1; r < n; ++r)
        {
            if (std::abs(a[r][col]) > std::abs(a[pivot][col]))
            {
                pivot = r;
            }
        }
        std::swap(a[col], a[pivot]);
        std::swap(inv[col], inv[pivot]);
        const double scale = 1.0 / a[col][col];
        for (int c = 0; c < n; ++c)
        {
            a[col][c] *= scale;
            inv[col][c] *= scale;
        }
        for (int r = 0; r < n; ++r)
        {
            if (r == col)
            {
                continue;
            }
            const double factor = a[r][col];
            for (int c = 0; c < n; ++c)
            {
                a[r][c] -= factor * a[col][c];
                inv[r][c] -= factor * inv[col][c];
            }
        }
    }
    return inv;
}

} // namespace

Mat3 quaternionToMatrix(double x, double y, double z, double w)
{
    // The spec's C matrix with unit scale (and the standard 3DGS R(q)).
    return {{{1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)},
             {2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)},
             {2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)}}};
}

// Shepperd's method; returns a unit xyzw quaternion for a proper rotation.
std::array<double, 4> matrixToQuaternion(const Mat3 & m)
{
    double x, y, z, w;
    const double trace = m[0][0] + m[1][1] + m[2][2];
    if (trace > 0.0)
    {
        const double s = std::sqrt(trace + 1.0) * 2.0;
        w = 0.25 * s;
        x = (m[2][1] - m[1][2]) / s;
        y = (m[0][2] - m[2][0]) / s;
        z = (m[1][0] - m[0][1]) / s;
    }
    else if (m[0][0] > m[1][1] && m[0][0] > m[2][2])
    {
        const double s = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
        w = (m[2][1] - m[1][2]) / s;
        x = 0.25 * s;
        y = (m[0][1] + m[1][0]) / s;
        z = (m[0][2] + m[2][0]) / s;
    }
    else if (m[1][1] > m[2][2])
    {
        const double s = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
        w = (m[0][2] - m[2][0]) / s;
        x = (m[0][1] + m[1][0]) / s;
        y = 0.25 * s;
        z = (m[1][2] + m[2][1]) / s;
    }
    else
    {
        const double s = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
        w = (m[1][0] - m[0][1]) / s;
        x = (m[0][2] + m[2][0]) / s;
        y = (m[1][2] + m[2][1]) / s;
        z = 0.25 * s;
    }
    const double inv = 1.0 / std::sqrt(x * x + y * y + z * z + w * w);
    return {x * inv, y * inv, z * inv, w * inv};
}

double determinant(const Mat3 & m)
{
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
           - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// The matrix T with c_world = T * c_local for one band, given the node
// rotation R. Derivation: an object rotated by R shows, along world direction
// d, the color its local data shows along R^T d. Each band is closed under
// rotation, so Y(R^T d) = K Y(d) for one (2l+1)^2 matrix K, and then
// c_local . Y(R^T d) = (K^T c_local) . Y(d), i.e. T = K^T.
//
// K is solved numerically from sample directions rather than from closed-form
// Wigner-D matrices: it is then correct BY CONSTRUCTION for whatever basis
// shBasisBand() (i.e. the shader) uses -- signs, Condon-Shortley phase and
// coefficient order included -- which is exactly the convention hazard the
// closed forms carry. The fit is exact up to double rounding because the data
// are consistent (the band space really is invariant).
BandMatrix shBandRotation(int band, const Mat3 & rotation)
{
    const int n = 2 * band + 1;
    const int samples = 2 * n; // over-determined for a well-conditioned Gram matrix
    BandMatrix gram{};         // Y Y^T
    BandMatrix cross{};        // Z Y^T, Z = basis at R^T d
    for (int s = 0; s < samples; ++s)
    {
        // Fibonacci sphere: deterministic, well spread, never degenerate.
        const double golden = 2.399963229728653; // pi * (3 - sqrt(5))
        const double zc = 1.0 - (2.0 * s + 1.0) / samples;
        const double r = std::sqrt(std::max(0.0, 1.0 - zc * zc));
        const double phi = golden * s;
        const double d[3] = {r * std::cos(phi), r * std::sin(phi), zc};
        // R^T d
        const double rd[3] = {
            rotation[0][0] * d[0] + rotation[1][0] * d[1] + rotation[2][0] * d[2],
            rotation[0][1] * d[0] + rotation[1][1] * d[1] + rotation[2][1] * d[2],
            rotation[0][2] * d[0] + rotation[1][2] * d[1] + rotation[2][2] * d[2]};
        double y[kMaxBandSize];
        double z[kMaxBandSize];
        shBasisBand(band, d[0], d[1], d[2], y);
        shBasisBand(band, rd[0], rd[1], rd[2], z);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                gram[i][j] += y[i] * y[j];
                cross[i][j] += z[i] * y[j];
            }
        }
    }
    const BandMatrix gramInverse = invert(gram, n);
    BandMatrix k{};
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (int m = 0; m < n; ++m)
            {
                sum += cross[i][m] * gramInverse[m][j];
            }
            k[i][j] = sum;
        }
    }
    BandMatrix t{};
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            t[i][j] = k[j][i];
        }
    }
    return t;
}
void rotateShCoefficients(glm::vec4 * coefficients, int degree,
                          const std::array<BandMatrix, 3> & bands)
{
    int offset = 1;
    for (int band = 1; band <= degree; ++band)
    {
        const int n = 2 * band + 1;
        const BandMatrix & rot = bands[band - 1];
        double rotated[kMaxBandSize][3] = {};
        for (int row = 0; row < n; ++row)
        {
            for (int col = 0; col < n; ++col)
            {
                const glm::vec4 & c = coefficients[offset + col];
                rotated[row][0] += rot[row][col] * c.x;
                rotated[row][1] += rot[row][col] * c.y;
                rotated[row][2] += rot[row][col] * c.z;
            }
        }
        for (int row = 0; row < n; ++row)
        {
            coefficients[offset + row] =
                glm::vec4(static_cast<float>(rotated[row][0]),
                          static_cast<float>(rotated[row][1]),
                          static_cast<float>(rotated[row][2]),
                          0.0f);
        }
        offset += n;
    }
}

} // namespace sh_rotation
} // namespace sabercat
