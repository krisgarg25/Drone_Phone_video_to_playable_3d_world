/**
 * glb.mjs -- minimal, zero-dependency GLB (binary glTF 2.0) reader.
 *
 * Scope: everything a baked *static* collision mesh needs.
 *   - 12-byte GLB header, JSON chunk (0x4a534f47), BIN chunk (0x004e4942)
 *   - nodes/meshes/accessors/bufferViews, arbitrary node hierarchy depth
 *   - node transforms: `matrix` (column-major, as stored by glTF) or TRS
 *   - several meshes, several primitives per mesh, indexed and non-indexed
 *   - index component types 5121 (uchar) / 5123 (ushort) / 5125 (uint)
 *   - POSITION accessors: VEC3 float32, optional byteStride, optional `sparse`
 *
 * Deliberately NOT supported (none of it appears in a PlayCanvas collision GLB
 * and supporting it would only hide real errors): animation/skins/morphs,
 * embedded data-URI buffers, compression extensions, KHR_draco_mesh_compressor.
 * Anything we refuse to guess at throws instead of silently producing junk.
 *
 * Output is one concatenated triangle soup in the glTF file's own world space
 * (Y up, right handed) -- which for this repo is the same space the viewer puts
 * the collider entity in, untransformed, i.e. the space collision.json's
 * spawn/walk_path are written in.
 */

import { readFileSync } from 'node:fs';

const GLB_MAGIC = 0x46546c67; // 'glTF'
const CHUNK_JSON = 0x4e4f534a; // 'JSON'
const CHUNK_BIN = 0x004e4942; // 'BIN\0'

const COMPONENT_TYPES = {
  5120: { array: Int8Array, bytes: 1 },
  5121: { array: Uint8Array, bytes: 1 },
  5122: { array: Int16Array, bytes: 2 },
  5123: { array: Uint16Array, bytes: 2 },
  5124: { array: Int32Array, bytes: 4 },
  5125: { array: Uint32Array, bytes: 4 },
  5126: { array: Float32Array, bytes: 4 },
};

const TYPE_ELEMENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };

export class GlbError extends Error {}

/** Split a GLB buffer into its JSON + BIN chunks. */
export function parseGlb(buf) {
  if (buf.length < 12) throw new GlbError('file shorter than the 12-byte GLB header');
  const magic = buf.readUInt32LE(0);
  if (magic !== GLB_MAGIC) throw new GlbError(`not a GLB (magic 0x${magic.toString(16)}, expected 0x46546c67)`);
  const version = buf.readUInt32LE(4);
  if (version !== 2) throw new GlbError(`glTF version ${version} not supported (only 2)`);
  const totalLength = buf.readUInt32LE(8);
  if (totalLength > buf.length) throw new GlbError(`header claims ${totalLength} bytes, file has ${buf.length}`);

  let off = 12;
  let json = null;
  let bin = null;
  while (off + 8 <= totalLength) {
    const chunkLength = buf.readUInt32LE(off);
    const chunkType = buf.readUInt32LE(off + 4);
    const start = off + 8;
    if (start + chunkLength > totalLength) throw new GlbError('chunk overruns the declared GLB length');
    if (chunkType === CHUNK_JSON) {
      if (json) throw new GlbError('duplicate JSON chunk');
      // padded to a 4-byte boundary: spec says spaces, some exporters use NULs
      const text = buf.slice(start, start + chunkLength).toString('utf8').replace(/[\s\0]+$/, '');
      json = JSON.parse(text);
    } else if (chunkType === CHUNK_BIN) {
      if (bin) throw new GlbError('duplicate BIN chunk');
      bin = buf.slice(start, start + chunkLength);
    }
    // unknown chunks (e.g. the KHR_xmp XML chunk) are skipped, per spec
    off = start + chunkLength + ((4 - (chunkLength % 4)) % 4);
  }
  if (!json) throw new GlbError('no JSON chunk');
  return { json, bin: bin ?? Buffer.alloc(0) };
}

/** Read `count` elements of one accessor into a plain (unstrided) typed array. */
function readAccessor(json, bin, accessorIndex) {
  const acc = json.accessors?.[accessorIndex];
  if (!acc) throw new GlbError(`accessor ${accessorIndex} missing`);
  if (acc.type === 'MAT3' || acc.type === 'MAT2') throw new GlbError(`accessor type ${acc.type} not supported`);
  const ct = COMPONENT_TYPES[acc.componentType];
  if (!ct) throw new GlbError(`accessor ${accessorIndex}: unknown componentType ${acc.componentType}`);
  const n = TYPE_ELEMENTS[acc.type];
  if (!n) throw new GlbError(`accessor ${accessorIndex}: unknown type ${acc.type}`);
  if (acc.normalized) throw new GlbError('normalized accessors not supported');

  let out;
  if (acc.bufferView !== undefined) {
    const bv = json.bufferViews[acc.bufferView];
    if (!bv) throw new GlbError(`bufferView ${acc.bufferView} missing`);
    const stride = bv.byteStride ?? n * ct.bytes;
    const start = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    if (stride === n * ct.bytes) {
      if (start % ct.bytes !== 0) throw new GlbError(`accessor ${accessorIndex}: misaligned byteOffset ${start}`);
      out = new ct.array(bin.buffer.slice(bin.byteOffset + start, bin.byteOffset + start + acc.count * n * ct.bytes));
    } else {
      out = new ct.array(acc.count * n);
      for (let i = 0; i < acc.count; i++) {
        const src = new ct.array(bin.buffer, bin.byteOffset + start + i * stride, n);
        out.set(src, i * n);
      }
    }
  } else {
    out = new ct.array(acc.count * n); // accessor with only `sparse` data
  }

  if (acc.sparse) {
    const sc = acc.sparse;
    const iCt = COMPONENT_TYPES[sc.indices.componentType];
    if (!iCt) throw new GlbError('sparse: bad indices componentType');
    const vCt = COMPONENT_TYPES[acc.componentType];
    const ibv = json.bufferViews[sc.indices.bufferView];
    const vbv = json.bufferViews[sc.values.bufferView];
    if (!ibv || !vbv) throw new GlbError('sparse: missing bufferView');
    const idx = new iCt.array(bin.buffer, bin.byteOffset + (ibv.byteOffset ?? 0) + (sc.indices.byteOffset ?? 0), sc.count);
    const val = new vCt.array(bin.buffer, bin.byteOffset + (vbv.byteOffset ?? 0) + (sc.values.byteOffset ?? 0), sc.count * n);
    for (let i = 0; i < sc.count; i++) {
      for (let k = 0; k < n; k++) out[idx[i] * n + k] = val[i * n + k];
    }
  }
  return { data: out, components: n };
}

/** glTF TRS -> column-major 4x4. */
function trsToMatrix(t, r, s) {
  const [x, y, z, w] = r;
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  const [sx, sy, sz] = s;
  return [
    (1 - 2 * (yy + zz)) * sx, (2 * (xy + wz)) * sx, (2 * (xz - wy)) * sx, 0,
    (2 * (xy - wz)) * sy, (1 - 2 * (xx + zz)) * sy, (2 * (yz + wx)) * sy, 0,
    (2 * (xz + wy)) * sz, (2 * (yz - wx)) * sz, (1 - 2 * (xx + yy)) * sz, 0,
    t[0], t[1], t[2], 1,
  ];
}

function identity() {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

/** column-major 4x4 * 4x4 */
function mul(a, b) {
  const o = new Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

function transformPoint(m, x, y, z, out, base) {
  out[base] = m[0] * x + m[4] * y + m[8] * z + m[12];
  out[base + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
  out[base + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
}

/**
 * Load every triangle of a GLB file, baked into world space.
 *
 * @param {string} path file system path to a `.glb`
 * @param {{skipNodeNames?: string[]}} [opts]
 * @returns {{positions: Float32Array, indices: Uint32Array,
 *            stats: {nodes: number, meshes: number, primitives: number, triangles: number, vertices: number}}}
 */
export function loadGlb(path, opts = {}) {
  const { json, bin } = parseGlb(readFileSync(path));
  if (json.meshes === undefined || json.accessors === undefined) throw new GlbError('glTF has no meshes/accessors');
  for (const b of json.buffers ?? []) {
    if (b.uri) throw new GlbError(`external buffer uri "${b.uri}" not supported (GLB must be self-contained)`);
  }

  const nodes = json.nodes ?? [];
  const skip = new Set(opts.skipNodeNames ?? []);

  // world matrix per node, walking down from every root (nodes no one points at)
  const world = new Array(nodes.length).fill(null);
  const childOf = new Set();
  for (const n of nodes) for (const c of n.children ?? []) childOf.add(c);
  const localMatrix = (n) => (n.matrix ? n.matrix.slice() : trsToMatrix(n.translation ?? [0, 0, 0], n.rotation ?? [0, 0, 0, 1], n.scale ?? [1, 1, 1]));

  const stack = [];
  for (let i = 0; i < nodes.length; i++) if (!childOf.has(i)) stack.push([i, identity(), false]);
  while (stack.length) {
    const [index, parent, skipped] = stack.pop();
    const node = nodes[index];
    const self = skipped || skip.has(node.name) ? parent : mul(parent, localMatrix(node));
    world[index] = skipped || skip.has(node.name) ? null : self;
    for (const c of node.children ?? []) stack.push([c, self, skipped || skip.has(node.name)]);
  }

  const unvisited = world.findIndex((m) => m === undefined);
  if (unvisited !== -1) {
    throw new GlbError(`node graph is not a set of trees (node ${unvisited} unreachable - cycle?)`);
  }

  const positions = [];
  const indices = [];
  let primitives = 0;
  let triangles = 0;
  let meshesUsed = 0;
  const meshesSeen = new Set();

  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni];
    if (node.mesh === undefined) continue;
    const m = world[ni];
    if (m === null) continue; // pruned by skipNodeNames
    const mesh = json.meshes[node.mesh];
    if (!mesh) throw new GlbError(`node ${ni} references missing mesh ${node.mesh}`);
    if (!meshesSeen.has(node.mesh)) { meshesSeen.add(node.mesh); meshesUsed++; }
    const base = positions.length / 3;

    for (const prim of mesh.primitives) {
      if (prim.mode !== undefined && prim.mode !== 4) {
        throw new GlbError(`primitive mode ${prim.mode} is not triangles (4)`);
      }
      if (prim.extensions) throw new GlbError(`primitive has unhandled extensions: ${Object.keys(prim.extensions)}`);
      const posAcc = prim.attributes?.POSITION;
      if (posAcc === undefined) throw new GlbError('primitive without POSITION');
      const { data: pos, components } = readAccessor(json, bin, posAcc);
      if (components !== 3) throw new GlbError(`POSITION accessor must be VEC3, got ${components} components`);
      primitives++;

      for (let i = 0; i < pos.length; i += 3) {
        positions.push(0, 0, 0);
        transformPoint(m, pos[i], pos[i + 1], pos[i + 2], positions, positions.length - 3);
      }

      let idx;
      if (prim.indices !== undefined) {
        const r = readAccessor(json, bin, prim.indices);
        if (r.components !== 1) throw new GlbError('index accessor must be SCALAR');
        idx = r.data;
      } else {
        idx = new Uint32Array(pos.length / 3);
        for (let i = 0; i < idx.length; i++) idx[i] = i;
      }
      if (idx.length % 3 !== 0) throw new GlbError(`index count ${idx.length} is not a multiple of 3`);
      const vcount = pos.length / 3;
      for (let i = 0; i < idx.length; i++) {
        if (idx[i] >= vcount) throw new GlbError(`index ${idx[i]} out of range for ${vcount} vertices`);
        indices.push(base + idx[i]);
      }
      triangles += idx.length / 3;
    }
  }

  if (!triangles) throw new GlbError('GLB contained no triangles');
  return {
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices),
    stats: { nodes: nodes.length, meshes: meshesUsed, primitives, triangles, vertices: positions.length / 3 },
  };
}
