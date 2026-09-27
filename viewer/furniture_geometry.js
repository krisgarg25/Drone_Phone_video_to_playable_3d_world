import { placementBox } from "./workspace_core.js";

// Local component bounds: X/Z in [-.5,.5], Y in [0,1]. No downloaded assets.
const WOOD = [177, 124, 76], EDGE = [112, 76, 48], METAL = [61, 69, 74];
const FABRIC = [113, 145, 149], CUSHION = [153, 180, 178], LINEN = [223, 219, 203];
const WHITE = [239, 236, 224], CABINET = [177, 164, 140];
export function furnitureParts(kind) {
  const parts = [];
  const box = (name, min, max, color = WOOD) => parts.push({ name, min, max, color });
  const legs = (top = 0.85, inset = 0.07, thickness = 0.09, color = METAL) => {
    let i = 0;
    for (const x of [-0.5 + inset, 0.5 - inset - thickness]) {
      for (const z of [-0.5 + inset, 0.5 - inset - thickness]) {
        box(`leg-${i++}`, [x, 0, z], [x + thickness, top, z + thickness], color);
      }
    }
  };
  const shelving = (levels, bottom = 0.06, top = 1, color = CABINET) => {
    box("side-left", [-0.5, bottom, -0.5], [-0.45, top, 0.5], color);
    box("side-right", [0.45, bottom, -0.5], [0.5, top, 0.5], color);
    box("back", [-0.45, bottom, -0.5], [0.45, top, -0.46], EDGE);
    for (let i = 0; i < levels; i++) {
      const y = bottom + (top - bottom - 0.04) * i / (levels - 1);
      box(`shelf-${i}`, [-0.5, y, -0.5], [0.5, y + 0.04, 0.5], color);
    }
  };
  switch (kind) {
    case "coffee_table": case "dining_table": case "desk": case "side_table": case "stool": {
      const top = kind === "coffee_table" ? 0.84 : kind === "stool" ? 0.83 : 0.91;
      legs(top, 0.07, kind === "stool" ? 0.12 : 0.08, kind === "side_table" ? EDGE : METAL);
      box(kind === "stool" ? "seat" : "top", [-0.5, top, -0.5], [0.5, 1, 0.5]);
      if (kind === "desk") box("drawer", [0.14, 0.72, -0.4], [0.4, top, 0.39], CABINET);
      if (kind === "dining_table") {
        box("apron-front", [-0.39, 0.78, 0.32], [0.39, top, 0.38], EDGE);
        box("apron-back", [-0.39, 0.78, -0.38], [0.39, top, -0.32], EDGE);
      }
      if (kind === "side_table") box("shelf", [-0.36, 0.18, -0.36], [0.36, 0.24, 0.36], EDGE);
      if (kind === "stool") {
        box("rung-front", [-0.36, 0.28, 0.26], [0.36, 0.34, 0.33], EDGE);
        box("rung-back", [-0.36, 0.28, -0.33], [0.36, 0.34, -0.26], EDGE);
      }
      break;
    }
    case "dining_chair":
      legs(0.46, 0.05, 0.10, EDGE);
      box("seat", [-0.5, 0.46, -0.5], [0.5, 0.56, 0.5], CUSHION);
      box("back-post-left", [-0.47, 0.5, -0.48], [-0.37, 1, -0.36], EDGE);
      box("back-post-right", [0.37, 0.5, -0.48], [0.47, 1, -0.36], EDGE);
      box("back", [-0.5, 0.7, -0.5], [0.5, 1, -0.34], WOOD);
      break;
    case "sofa": case "armchair": {
      legs(0.18, 0.09, 0.08, EDGE);
      box("base", [-0.48, 0.15, -0.46], [0.48, 0.4, 0.48], FABRIC);
      box("arm-left", [-0.5, 0.2, -0.5], [-0.38, 0.76, 0.5], FABRIC);
      box("arm-right", [0.38, 0.2, -0.5], [0.5, 0.76, 0.5], FABRIC);
      box("back", [-0.38, 0.3, -0.5], [0.38, 1, -0.28], FABRIC);
      const seats = kind === "sofa" ? 3 : 1, span = 0.76 / seats;
      for (let i = 0; i < seats; i++) {
        const x = -0.38 + i * span;
        box(`cushion-seat-${i}`, [x + 0.012, 0.4, -0.26], [x + span - 0.012, 0.59, 0.46], CUSHION);
        box(`cushion-back-${i}`, [x + 0.012, 0.59, -0.29], [x + span - 0.012, 0.92, -0.14], CUSHION);
      }
      break;
    }
    case "bed_double":
      // The catalogue's length is X: pillows/headboard belong at the -X end.
      legs(0.2, 0.06, 0.09, EDGE);
      box("frame", [-0.5, 0.13, -0.5], [0.5, 0.39, 0.5], EDGE);
      box("headboard", [-0.5, 0.2, -0.5], [-0.46, 1, 0.5], WOOD);
      box("mattress", [-0.45, 0.39, -0.48], [0.49, 0.79, 0.48], LINEN);
      box("duvet", [-0.12, 0.78, -0.49], [0.49, 0.84, 0.49], FABRIC);
      box("pillow-left", [-0.42, 0.79, -0.43], [-0.19, 0.97, -0.04], WHITE);
      box("pillow-right", [-0.42, 0.79, 0.04], [-0.19, 0.97, 0.43], WHITE);
      break;
    case "bookshelf":
      shelving(6, 0, 1);
      // Small groups of books retain visible empty shelf space.
      for (let i = 0; i < 4; i++) {
        const y = 0.04 + i * 0.192;
        for (let j = 0; j < 3; j++) {
          const x = -0.38 + j * 0.085 + (i % 2) * 0.38;
          box(`book-${i}-${j}`, [x, y, -0.27], [x + 0.065, y + 0.12 + j * 0.01, 0.26], [FABRIC, EDGE, LINEN][j]);
        }
      }
      break;
    case "wardrobe":
      shelving(2, 0, 1);
      box("door-left", [-0.445, 0.045, 0.43], [-0.008, 0.955, 0.48], LINEN);
      box("door-right", [0.008, 0.045, 0.43], [0.445, 0.955, 0.48], LINEN);
      box("handle-left", [-0.07, 0.43, 0.48], [-0.05, 0.59, 0.5], METAL);
      box("handle-right", [0.05, 0.43, 0.48], [0.07, 0.59, 0.5], METAL);
      break;
    case "tv_stand":
      legs(0.2, 0.07, 0.08, METAL);
      shelving(3, 0.2, 1, WOOD);
      box("divider-left", [-0.2, 0.24, -0.46], [-0.17, 0.96, 0.46], EDGE);
      box("divider-right", [0.17, 0.24, -0.46], [0.2, 0.96, 0.46], EDGE);
      break;
    default:
      box("unknown", [-0.5, 0, -0.5], [0.5, 1, 0.5], CABINET);
  }
  return parts;
}

/** Opaque per-face byte RGBA, baked directional lighting and world-space normals. */
export function furnitureGeometry(placement) {
  const bounds = placementBox(placement);
  if (!bounds) return null;
  const positions = [], normals = [], colors = [], indices = [];
  const [L, H, D] = placement.size, [cx, cy, cz] = bounds.center;
  const angle = bounds.yaw * Math.PI / 180, ca = Math.cos(angle), sa = Math.sin(angle);
  const light = [-0.35, 0.82, 0.45], lightLength = Math.hypot(...light);
  const transform = ([x, y, z]) => [cx + x * L * ca - z * D * sa, cy + (y - 0.5) * H, cz + x * L * sa + z * D * ca];
  for (const part of furnitureParts(placement.item)) {
    const [x0, y0, z0] = part.min, [x1, y1, z1] = part.max;
    const faces = [
      [[0, -1, 0], [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]],
      [[0, 1, 0], [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]]],
      [[0, 0, -1], [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]]],
      [[0, 0, 1], [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]],
      [[-1, 0, 0], [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]],
      [[1, 0, 0], [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]]],
    ];
    for (const [n, vertices] of faces) {
      const normal = [n[0] * ca - n[2] * sa, n[1], n[0] * sa + n[2] * ca];
      const shade = 0.56 + 0.44 * Math.max(0, normal.reduce((sum, v, i) => sum + v * light[i], 0) / lightLength);
      const color = part.color.map((v, i) => {
        const tinted = placement.stale ? v * 0.8 + 30 : placement.fit?.valid === false ? v * 0.88 + [28, 9, 5][i] : v;
        return Math.max(0, Math.min(255, Math.round(tinted * shade)));
      });
      const base = positions.length / 3;
      for (const v of vertices) { positions.push(...transform(v)); normals.push(...normal); colors.push(...color, 255); }
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
  }
  return { positions, normals, colors: new Uint8Array(colors), indices };
}
