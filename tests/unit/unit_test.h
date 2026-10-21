// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Check harness for the CPU unit tests. Framework-free on purpose: the test
// binary depends only on the libraries it tests. Each check is counted; a failed
// check prints its location and expression.
#pragma once

#include <cmath>
#include <iostream>
#include <string>

namespace sabercat::unit
{

struct Context
{
    int checks = 0;
    int failures = 0;
};

#define UNIT_CHECK(ctx, expr)                                                                   \
    do                                                                                          \
    {                                                                                           \
        ++(ctx).checks;                                                                         \
        if (!(expr))                                                                            \
        {                                                                                       \
            ++(ctx).failures;                                                                   \
            std::cerr << __FILE__ << ":" << __LINE__ << ": check failed: " #expr << std::endl;  \
        }                                                                                       \
    } while (false)

// |actual - expected| <= tolerance, in double precision.
#define UNIT_CHECK_NEAR(ctx, actual, expected, tolerance)                                        \
    do                                                                                           \
    {                                                                                            \
        const double unitActual = static_cast<double>(actual);                                   \
        const double unitExpected = static_cast<double>(expected);                               \
        ++(ctx).checks;                                                                          \
        if (!(std::abs(unitActual - unitExpected) <= static_cast<double>(tolerance)))            \
        {                                                                                        \
            ++(ctx).failures;                                                                    \
            std::cerr << __FILE__ << ":" << __LINE__ << ": " #actual " = " << unitActual         \
                      << ", expected " << unitExpected << " +/- " << static_cast<double>(tolerance) \
                      << std::endl;                                                              \
        }                                                                                        \
    } while (false)

} // namespace sabercat::unit
