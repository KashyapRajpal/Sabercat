# Third-Party Notices

Sabercat is licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE).
This file lists the third-party software that Sabercat depends on, the version
pinned in this repository, how each piece is used, and the license that applies
to it. The full license text for each component follows the summary.

Sabercat's own source code is copyright 2026 Kashyap Rajpal.

## Summary

| Component | Version | How Sabercat uses it | License |
|---|---|---|---|
| zlib | 1.3.1 | Linked into libsabercat (the SOG reader inflates with it). Shipped in the install as `zlib.dll` plus its import library. | zlib |
| spz (Niantic Labs) | v3.0.0 (commit 5bf2945) | Linked into libsabercat (`.spz` reader). Static library shipped in the install. | MIT |
| zstd (Meta) | 1.5.6 | Linked into spz, so it is linked into libsabercat. Static library shipped in the install. Used under its BSD-3-Clause option. | BSD-3-Clause |
| libwebp (Google) | 1.6.0 | Linked into libsabercat (decode-only `webpdecoder`, for SOG scenes). Static library shipped in the install. | BSD-3-Clause (with PATENTS grant) |
| nlohmann/json | 3.12.0 | Header-only. Compiled into libsabercat (glTF, SOG and manifest JSON). Headers are not shipped. | MIT |
| Vulkan Memory Allocator (VMA) | 3.2.1 | Header-only, compiled into libsabercat (`src/renderer/vma_impl.cpp`) and used by public headers. From the Vulkan SDK when present, otherwise fetched at build time. Shipped in the SDK package's `include/vma/`. | MIT |
| GLM | 1.0.1 | Header-only. Used by public and internal headers. From the Vulkan SDK when present, otherwise fetched at build time. Shipped in the SDK package's `include/glm/`. | MIT |
| GLFW | 3.4 | Apps and examples only. Not part of libsabercat and not in the install. | zlib |
| Dear ImGui | 1.91.5 | Apps only (debug UI). Not part of libsabercat and not in the install. | MIT |
| Slang compiler (`slangc`) | from Vulkan SDK 1.4.309.0 | Build-time only: compiles the shaders to SPIR-V. Not linked, not shipped. | Apache-2.0 WITH LLVM-exception |
| nanobind | not applicable | Not used. spz's Python bindings are disabled (`SPZ_BUILD_PYTHON_BINDINGS OFF`). | not applicable |
| PlayCanvas splat-transform | upstream `main`, read 2026-10-20 (not pinned) | Tests only. `examples/gaussian_cloud_test/compressed_ply.cpp` is a C++ port of splat-transform's compressed-PLY encoder (`compressed-chunk.ts`, `write-compressed-ply.ts`). Not linked or shipped. | MIT |

Vulkan headers (`vulkan/vulkan.h`) come from the Vulkan SDK. They are not
vendored or shipped by Sabercat, but the public headers include them. Their
license is Apache-2.0 (Khronos); consumers obtain them from the Vulkan SDK.

The apps package (`sabercat-<version>-<sha>-win64-apps.zip`) includes the
Microsoft Visual C++ runtime DLLs (`vcruntime140.dll`, `msvcp140.dll` and
related files) so the apps start without an installer. They are redistributed
under Microsoft's redistribution terms for the Visual C++ runtime.

## Details and license texts

Versions are the ones pinned in `CMakeLists.txt`, `apps/CMakeLists.txt`, and
`external/spz` (submodule commit), and the ones actually fetched during a
configure of this tree. Each full license text below was taken from the
source named next to it.

### zlib 1.3.1

Text source: zlib 1.3.1 source tarball fetched by CMake (https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz). Upstream: https://zlib.net/ and https://github.com/madler/zlib.

```text
Copyright notice:

 (C) 1995-2022 Jean-loup Gailly and Mark Adler

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  Jean-loup Gailly        Mark Adler
  jloup@gzip.org          madler@alumni.caltech.edu
```

### spz v3.0.0

Text source: external/spz submodule at commit 5bf2945de1a003cee07133b1e495fe9c6ffdc7e7 (tag v3.0.0), https://github.com/nianticlabs/spz.

```text
MIT License

Copyright (c) 2024 Niantic Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### zstd 1.5.6 (BSD-3-Clause option)

Text source: zstd 1.5.6 source fetched by spz's CMake (https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz). zstd is dual-licensed BSD-3-Clause or GPLv2; Sabercat uses the BSD-3-Clause option and does not distribute under the GPLv2 option.

```text
BSD License

For Zstandard software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Facebook, nor Meta, nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### libwebp 1.6.0 (BSD-3-Clause)

Text source: libwebp 1.6.0 release tarball fetched by CMake, SHA256-pinned (https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz). Upstream: https://github.com/webmproject/libwebp.

```text
Copyright (c) 2010, Google Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

  * Neither the name of Google nor the names of its contributors may
    be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

```

### libwebp 1.6.0 PATENTS grant

Text source: libwebp 1.6.0 release tarball fetched by CMake (file PATENTS).

```text
Additional IP Rights Grant (Patents)
------------------------------------

"These implementations" means the copyrightable works that implement the WebM
codecs distributed by Google as part of the WebM Project.

Google hereby grants to you a perpetual, worldwide, non-exclusive, no-charge,
royalty-free, irrevocable (except as stated in this section) patent license to
make, have made, use, offer to sell, sell, import, transfer, and otherwise
run, modify and propagate the contents of these implementations of WebM, where
such license applies only to those patent claims, both currently owned by
Google and acquired in the future, licensable by Google that are necessarily
infringed by these implementations of WebM. This grant does not include claims
that would be infringed only as a consequence of further modification of these
implementations. If you or your agent or exclusive licensee institute or order
or agree to the institution of patent litigation or any other patent
enforcement activity against any entity (including a cross-claim or
counterclaim in a lawsuit) alleging that any of these implementations of WebM
or any code incorporated within any of these implementations of WebM
constitute direct or contributory patent infringement, or inducement of
patent infringement, then any patent rights granted to you under this License
for these implementations of WebM shall terminate as of the date such
litigation is filed.
```

### nlohmann/json 3.12.0 (MIT)

Text source: nlohmann/json 3.12.0 release tarball fetched by CMake (https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz). Upstream: https://github.com/nlohmann/json.

```text
MIT License 

Copyright (c) 2013-2025 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Vulkan Memory Allocator 3.2.1 (MIT)

Text source: Upstream v3.2.1 LICENSE.txt (https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/blob/v3.2.1/LICENSE.txt), byte-identical to the LICENSE.txt in the v3.2.1 source archive that the build fetches (SHA-256 5e7749504cb802427ffb7bec38a0b6a15db46ae253f00560acb3e624d9fe695c). The header shipped in the Vulkan SDK 1.4.309.0 (Include/vma/vk_mem_alloc.h, version 3.2.1) carries the same copyright and 'License: MIT'.

```text
Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### GLM 1.0.1 (MIT)

Text source: Upstream GLM 1.0.1 copying.txt (https://github.com/g-truc/glm/blob/1.0.1/copying.txt), byte-identical to the copying.txt in the 1.0.1 source archive that the build fetches (SHA-256 9f3174561fd26904b23f0db5e560971cbf9b3cbda0b280f04d5c379d03bf234c). The GLM headers in the Vulkan SDK 1.4.309.0 (GLM_VERSION 1.0.1) carry the G-Truc Creation copyright notice but not the full license text, so the text is taken from the matching upstream tag.

```text
================================================================================
OpenGL Mathematics (GLM)
--------------------------------------------------------------------------------
GLM is licensed under The Happy Bunny License or MIT License

================================================================================
The Happy Bunny License (Modified MIT License)
--------------------------------------------------------------------------------
Copyright (c) 2005 - G-Truc Creation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

Restrictions:
 By making use of the Software for military purposes, you choose to make a
 Bunny unhappy.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

================================================================================
The MIT License
--------------------------------------------------------------------------------
Copyright (c) 2005 - G-Truc Creation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### GLFW 3.4 (zlib license; apps only)

Text source: GLFW 3.4 source fetched by CMake (FetchContent, git tag 3.4, https://github.com/glfw/glfw). Only used when no system glfw3 is found, and only for apps and examples.

```text
Copyright (c) 2002-2006 Marcus Geelnard

Copyright (c) 2006-2019 Camilla Löwy

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would
   be appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not
   be misrepresented as being the original software.

3. This notice may not be removed or altered from any source
   distribution.

```

### Dear ImGui 1.91.5 (MIT; apps only)

Text source: Dear ImGui v1.91.5 source fetched by CMake (FetchContent, git tag v1.91.5, https://github.com/ocornut/imgui). Used only by the apps.

```text
The MIT License (MIT)

Copyright (c) 2014-2024 Omar Cornut

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Slang compiler (Apache-2.0 WITH LLVM-exception; build-time only)

Slang is used only at build time: `shaders/CMakeLists.txt` runs `slangc` (found
with `find_program`) to compile `.slang` kernels to SPIR-V. Nothing from Slang
is linked into libsabercat or the apps, and no Slang binary or source is
shipped in the install. The `slangc` binary comes from the Vulkan SDK
(1.4.309.0). Slang is licensed under the Apache License, Version 2.0 with the
LLVM exception; the Apache-2.0 text is in [LICENSE](LICENSE).

### nanobind (not used)

nanobind is only referenced by spz's Python bindings, which Sabercat disables
(`SPZ_BUILD_PYTHON_BINDINGS OFF`). It is not compiled, linked or shipped.

### PlayCanvas splat-transform (MIT; tests only)

Text source: upstream `LICENSE` at https://github.com/playcanvas/splat-transform (raw: https://raw.githubusercontent.com/playcanvas/splat-transform/main/LICENSE), read 2026-10-20. Used only by `examples/gaussian_cloud_test/compressed_ply.cpp`, which carries a `Portions derived from` line.

```text
Copyright (c) 2011-2026 PlayCanvas Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
