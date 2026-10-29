// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Regenerates the committed .ksplat / .sog fixtures with the formats' REFERENCE
// tools, from the synthetic source scene in compressed_scene_fixture.h.
//
// The point of these fixtures is that Sabercat did not write them: each is the
// output of the tool that defines its format, so a reader that merely agrees
// with a self-written encoder cannot pass. Generation needs node + npm, which
// the C++ tests cannot assume, hence committed outputs (all < 70 KB).
//
// Usage, in a scratch directory (nothing there belongs in the repo). Copy this
// script INTO that directory first: ES modules resolve packages relative to
// the script's own location, not the working directory.
//   npm install @mkkellogg/gaussian-splats-3d@0.4.7 @playcanvas/splat-transform@3.10.0
//   <build>/examples/ksplat_reader_test/<Config>/sabercat_ksplat_reader_test --write-source-ply source.ply
//   node generate_fixtures.mjs source.ply \
//        <repo>/examples/ksplat_reader_test/fixtures <repo>/examples/sog_reader_test/fixtures
//
// .ksplat: exactly what GaussianSplats3D's util/create-ksplat.js does
//   (PlyParser.parseToUncompressedSplatArray -> SplatBufferGenerator
//   .getStandardGenerator -> splatBuffer.bufferData), once per compression
//   level 0/1/2, SH degree 2 (the format's ceiling), default alpha threshold,
//   block size and bucket size.
// .sog: the splat-transform CLI, bundled output, k-means on the CPU so the
//   result does not depend on a GPU adapter.

import * as fs from 'fs';
import * as path from 'path';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const [sourcePly, ksplatDir, sogDir] = process.argv.slice(2);
if (!sourcePly || !ksplatDir || !sogDir) {
    console.error('usage: node generate_fixtures.mjs <source.ply> <ksplat fixture dir> <sog fixture dir>');
    process.exit(1);
}

// A Buffer from readFileSync may view a shared pool; hand the parser exactly
// the file's bytes.
const file = fs.readFileSync(sourcePly);
const plyBytes = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);

fs.mkdirSync(ksplatDir, { recursive: true });
for (const level of [0, 1, 2]) {
    const splatArray = GaussianSplats3D.PlyParser.parseToUncompressedSplatArray(plyBytes, 2);
    const generator = GaussianSplats3D.SplatBufferGenerator.getStandardGenerator(1, level, 0);
    const splatBuffer = generator.generateFromUncompressedSplatArray(splatArray);
    const out = path.join(ksplatDir, `synthetic_l${level}.ksplat`);
    fs.writeFileSync(out, Buffer.from(splatBuffer.bufferData));
    console.log(`wrote ${out} (${splatBuffer.getSplatCount()} splats)`);
}

// The package exports no package.json, so locate its CLI beside this script.
const cli = fileURLToPath(new URL('./node_modules/@playcanvas/splat-transform/bin/cli.mjs', import.meta.url));
fs.mkdirSync(sogDir, { recursive: true });
const sogOut = path.join(sogDir, 'synthetic.sog');
const result = spawnSync(process.execPath, [cli, '-q', '-w', '-g', 'cpu', sourcePly, sogOut], { stdio: 'inherit' });
if (result.status !== 0) {
    console.error('splat-transform failed');
    process.exit(1);
}
console.log(`wrote ${sogOut}`);
