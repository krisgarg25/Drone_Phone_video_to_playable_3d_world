"""Dump vertex bounds / triangle counts of a GLB (debug helper)."""
import json
import struct
import sys

import numpy as np

CT = {5120: 'b', 5121: 'B', 5122: 'h', 5123: 'H', 5125: 'I', 5126: 'f'}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def load_glb(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'glTF', b[:4]
    _ver, total = struct.unpack('<II', b[4:12])
    off, chunks = 12, []
    while off < total:
        clen, ctype = struct.unpack('<I4s', b[off:off + 8])
        chunks.append((ctype, b[off + 8:off + 8 + clen]))
        off += 8 + clen
    gltf = json.loads(chunks[0][1].decode('utf-8'))
    return gltf, (chunks[1][1] if len(chunks) > 1 else b'')


def accessor(gltf, bin_, ai):
    a = gltf['accessors'][ai]
    v = gltf['bufferViews'][a['bufferView']]
    nc = NC[a['type']]
    fmt = CT[a['componentType']]
    itemsz = np.dtype(fmt).itemsize * nc
    start = v.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = v.get('byteStride') or itemsz
    raw = bin_[start:start + stride * (a['count'] - 1) + itemsz]
    if stride == itemsz:
        return np.frombuffer(raw, dtype=fmt).reshape(a['count'], nc)
    out = np.empty((a['count'], nc), dtype=fmt)
    for i in range(a['count']):
        out[i] = np.frombuffer(raw[i * stride:i * stride + itemsz], dtype=fmt)
    return out


def main(path):
    gltf, bin_ = load_glb(path)
    print('meshes', len(gltf.get('meshes', [])), 'nodes', len(gltf.get('nodes', [])))
    for n in gltf.get('nodes', []):
        print('  node', {k: v for k, v in n.items() if k != 'children'})
    allv, tris = [], 0
    for mi, m in enumerate(gltf['meshes']):
        for pi, prim in enumerate(m['primitives']):
            pos = accessor(gltf, bin_, prim['attributes']['POSITION']).astype(np.float64)
            allv.append(pos)
            if 'indices' in prim:
                tris += gltf['accessors'][prim['indices']]['count'] // 3
            print(f'  mesh{mi} prim{pi} verts={len(pos)} '
                  f'min={np.round(pos.min(0), 3)} max={np.round(pos.max(0), 3)}')
    V = np.vstack(allv)
    print('TOTAL verts', len(V), 'tris', tris)
    print('min', np.round(V.min(0), 3))
    print('max', np.round(V.max(0), 3))
    print('Y pct', {p: round(float(np.percentile(V[:, 1], p)), 3)
                    for p in (0, 1, 5, 25, 50, 75, 95, 99, 100)})


if __name__ == '__main__':
    main(sys.argv[1])
