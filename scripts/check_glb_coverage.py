"""Check collision GLB triangle coverage over the walk loop (XZ rasterization)."""
import json
import struct
import sys

import numpy as np

glb, x0, x1, z0, z1 = sys.argv[1], *map(float, sys.argv[2:6])
data = open(glb, "rb").read()
clen, _ = struct.unpack("<II", data[12:20])
js = json.loads(data[20:20 + clen])
bin_off = 20 + clen + 8
prim = js["meshes"][0]["primitives"][0]

acc = js["accessors"][prim["attributes"]["POSITION"]]
bv = js["bufferViews"][acc["bufferView"]]
voff = bin_off + bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
pts = np.frombuffer(data, np.uint8, count=acc["count"] * 12, offset=voff)
pts = pts.copy().view(np.float32).reshape(-1, 3)

iacc = js["accessors"][prim["indices"]]
ibv = js["bufferViews"][iacc["bufferView"]]
ioff = bin_off + ibv.get("byteOffset", 0) + iacc.get("byteOffset", 0)
n = iacc["count"]
if iacc["componentType"] == 5125:
    idx = np.frombuffer(data, np.uint32, count=n, offset=ioff)
else:
    idx = np.frombuffer(data, np.uint16, count=n, offset=ioff)
tris = pts[idx.reshape(-1, 3)]
print(f"verts {len(pts)}, tris {len(tris)}")

res = 0.5
W = int((x1 - x0) / res)
Hn = int((z1 - z0) / res)
cov = np.zeros((Hn, W), bool)
mn, mx = tris.min(1), tris.max(1)
for xa in range(W):
    xs = x0 + (xa + 0.5) * res
    hit = (mn[:, 0] <= xs) & (mx[:, 0] >= xs)
    for za in range(Hn):
        zs = z0 + (za + 0.5) * res
        if ((mn[hit, 2] <= zs) & (mx[hit, 2] >= zs)).any():
            cov[za, xa] = True

lx0, lx1, lz0, lz1 = -16, 9.4, 4.7, 28.7  # walk loop
loop = cov[int((lz0 - z0) / res):int((lz1 - z0) / res),
           int((lx0 - x0) / res):int((lx1 - x0) / res)]
print(f"walk-loop coverage: {loop.mean() * 100:.1f}%")
print(f"total footprint: {cov.mean() * 100:.1f}%")
print("tri y range", np.nanmin(tris[:, :, 1]).round(2), np.nanmax(tris[:, :, 1]).round(2))
