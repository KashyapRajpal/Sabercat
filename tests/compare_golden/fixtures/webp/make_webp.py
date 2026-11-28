# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Writes lossless_4x3.webp next to this script: a 4x3 RGBA image with distinct pixels
# and varying alpha, saved losslessly with exact=True so the RGB under transparent
# pixels is kept. Prints the expected pixel table; the unit test in
# tests/unit/test_compare_loaders.cpp carries the same values. Deterministic.
#
#   python -I make_webp.py

import os

from PIL import Image, features

WIDTH = 4
HEIGHT = 3
ALPHA = [0, 64, 128, 191, 255, 17, 200, 254, 3, 100, 230, 255]


def pixel(index):
    r = (37 * index + 11) % 256
    g = (73 * index + 5) % 256
    b = (151 * index + 97) % 256
    return (r, g, b, ALPHA[index])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    count = WIDTH * HEIGHT
    pixels = [pixel(i) for i in range(count)]
    data = bytes(value for p in pixels for value in p)
    image = Image.frombytes("RGBA", (WIDTH, HEIGHT), data)
    out = os.path.join(here, "lossless_4x3.webp")
    image.save(out, lossless=True, exact=True)

    back = Image.open(out)
    if back.format != "WEBP" or back.size != (WIDTH, HEIGHT):
        raise SystemExit("unexpected WebP header: %s %s" % (back.format, back.size))
    back = back.convert("RGBA")
    decoded = [back.getpixel((x, y)) for y in range(HEIGHT) for x in range(WIDTH)]
    if decoded != pixels:
        raise SystemExit("round trip differs from the generated pixels")

    print("wrote %s" % out)
    print("libwebp %s, Pillow %s" % (features.version("webp"), Image.__version__))
    print("index x y   r   g   b   a")
    for y in range(HEIGHT):
        for x in range(WIDTH):
            r, g, b, a = pixels[y * WIDTH + x]
            print("%5d %d %d %3d %3d %3d %3d" % (y * WIDTH + x, x, y, r, g, b, a))
    print("C++ rows (r, g, b, a), row-major, row 0 first:")
    for y in range(HEIGHT):
        row = ", ".join("{%d, %d, %d, %d}" % pixels[y * WIDTH + x] for x in range(WIDTH))
        print("    " + row + ("," if y + 1 < HEIGHT else ""))


if __name__ == "__main__":
    main()
