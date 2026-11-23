# WebP fixtures

`lossless_4x3.webp` is a 4x3 lossless RGBA WebP with distinct pixel values and
varying alpha. The unit test `compare loaders` (tests/unit/test_compare_loaders.cpp)
loads it through `loadImage()` and checks every sample against the table that
`make_webp.py` prints.

Regenerate it with:

    cd tests/compare_golden/fixtures/webp
    python -I make_webp.py

Provenance of the committed file:

- Pillow 12.1.0 (`PIL.__version__`)
- libwebp 1.6.0 (`PIL.features.version('webp')`)
- Written by `Image.save(path, lossless=True, exact=True)` from an RGBA image; the
  script reads it back and checks it before it prints anything.

The file is binary. tests/compare_golden/.gitattributes (`* -text`) keeps git from
touching its line endings.
