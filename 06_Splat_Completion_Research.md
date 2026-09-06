# 06 — Predictive / Assisted Completion of Unobserved Splat Regions

**Status:** research only. Nothing in this document is implemented.
**Date:** 2026-09-03
**Scope:** the "3 sides captured, 4th side is random garbage" problem, and the proposed
click-a-surface / draw-a-patch fill system.
**Hardware assumed throughout:** RTX 3050 Laptop, 6 GB VRAM, 16 GB RAM, Windows 11,
CUDA 12.4, no cloud, no compile-from-source. Same constraint as `05_MVP_Agent_Prompt.md`.

---

## 0. Decisions taken during this research pass

| Question | Answer |
|---|---|
| Which defect first? | **Floaters first**, then fill. Both eventually. |
| How is the fill region chosen? | **Manual first** (click / draw), automatic detection later. |
| What should the fill look like? | **Plausible flat wall.** Not hallucinated detail. |
| Commercial licensing | Deferred — flagged inline, not used as a filter. |
| EU/UK/KR territory | Deferred — see §6.4, matters only for Hunyuan3D. |

These decisions are load-bearing for the recommendation in §8. If any of them change, re-read
§7 (the ladder) before acting.

---

## 1. The core finding

**The interactive splat editor this feature needs is already vendored in this repo and unused.**

`tools/pc-engine` is a submodule pinned at **`e9aae30`** — verified as the *exact* revision that
`viewer/pc/playcanvas.mjs` was built from (`playcanvas.mjs:1-8` reports
`v2.22.0-beta.20 revision e9aae30 (RELEASE)`). So the bundle in the viewer and the source you can
read side-by-side are the same code. Everything below is MIT, plain `.mjs`, and requires **no build
step** — the only change needed is the import specifier.

Verified present on disk:

| Path (under `tools/pc-engine/`) | Bytes | What it is |
|---|---:|---|
| `examples/src/examples/gaussian-splatting/paint.example.mjs` | 13,601 | **3D brush painting on a loaded splat.** Depth-pick for brush position, `customColor` RGBA8 instance stream, `GSplatProcessor` writes inside a brush sphere, `BlendState.ALPHABLEND` accumulation, `setWorkBufferModifier` blends it into the rendered colour. RMB paints, LMB orbits. |
| `.../editor.example.mjs` | 23,683 | **Per-splat selection state, selection highlight, delete, and GPU clone of a selected subset.** |
| `.../editor.selection-processor.mjs` | 1,479 | writes the selection stream |
| `.../editor.delete-processor.mjs` | 1,454 | additive delete via `discard` |
| `.../editor.workbuffer-modifier.mjs` | 1,885 | tint selected / hide deleted |
| `.../editor.copy-processor.mjs` | 4,721 | remap-texture-driven subset clone |
| `.../picking.example.mjs` | 9,336 | `Picker` + depth → **world-space point on the splat surface under the cursor** |
| `.../crop.example.mjs` | 7,532 | AABB crop via the per-splat modify hook |
| `.../annotations.example.mjs` | 7,429 | anchoring UI to a picked 3D point, occlusion-correct |
| `scripts/esm/gsplat/gsplat-mesh.mjs` | — | **Mesh → Gaussian splats.** Scanline-rasterizes each triangle into uniformly spaced splats with a margin factor. Imports only `GSplatFormat, GSplatContainer, FloatPacking` from playcanvas. |
| `scripts/esm/gsplat/shader-effect-crop.mjs` | — | ready-made crop effect |
| `scripts/esm/gsplat/shader-effect-box.mjs` | — | ready-made box effect |

`gsplat-mesh.mjs` is the single most important file in that list. The proposed
"draw a patch face and it becomes splats" feature decomposes to:

```
3 clicks → plane fit → polygon outline → triangulate → pc.Mesh → gsplat-mesh.mjs → splats → merge
```

Every stage of that already exists as MIT code, and the last stage (`splat-transform`) is
already driven from this repo at `scripts/build_collider.py:177`.

### 1.1 Picking is also already solved here

`viewer/pc.js:205` and `viewer/pc.js:1139` already call:

```js
app.systems.rigidbody.raycastFirst(from, to, { filterCollisionMask: BODYMASK_STATIC });
```

against the collision trimesh loaded at `viewer/pc.js:575-583` from
`work/<scene>/pc/collision.collision.glb` (0.87 MB). That returns `{ point, normal, entity }`.

**`hit.normal` is the detail that matters.** A patch face needs a surface orientation, and the
raycast hands it over for free. Neither splat depth-picking nor per-splat ID picking gives you a
normal — you'd have to fit one. So the cheapest possible "click a surface" implementation is
already wired and running in the viewer today; it just isn't exposed to the user.

Limitation: the collision mesh is a **heightfield** (`scripts/ground_mesh.py`), so it cannot
express overhangs or vertical detail the voxel shell missed. README-MVP notes 37% of cells are
dilated in from neighbours. For picking a *facade*, the heightfield may be the wrong proxy —
walls are exactly what a heightfield flattens. See §7.1 for the fallback chain.

### 1.2 The official per-splat API (WebGL2-safe)

This is the answer to "can we modify a loaded splat at runtime". Four parts, all present in
`viewer/pc/playcanvas.mjs`:

**(a) Declare a custom per-splat stream** — `resource.format.addExtraStreams([...])`
(`playcanvas.mjs:35512`), with `GSPLAT_STREAM_INSTANCE` (per-component) or
`GSPLAT_STREAM_RESOURCE` (shared). Note: work-buffer formats do *not* support instance storage —
add to the resource format.

**(b) Read/write from CPU** — `ent.gsplat.getInstanceTexture(name)` (`:90549`), then
`tex.lock()/unlock()` to upload and `await tex.read(...)` to download.

**(c) Write from GPU** — `GSplatProcessor` (`:42435`). Critically this is a fragment-shader **MRT
render-to-texture** pass (`RenderPassShaderQuad` + `QuadRender`), **not** a compute shader, so it
works on **WebGL2**. Inside the shader body you get `getCenter()`, `splat.uv`, `splat.index`,
`splatTextureSize`, and generated `writeCustomColor(vec4)` helpers. `discard` means "leave the
existing value alone", which is how the delete processor makes its op additive.

**(d) Affect rendering** — `component.setWorkBufferModifier(...)` (`:90343`) injects GLSL as the
`gsplatModifyVS` chunk, with hooks `modifySplatCenter`, `modifySplatRotationScale`,
`modifySplatColor`. Instance-stream textures are bound into scope automatically
(`:41059-41062`), so `texelFetch(splatSelection, splat.uv, 0)` just works. Poke
`component.workBufferUpdate = WORKBUFFER_UPDATE_ONCE` to force a refresh.

**Stay in unified mode.** `GSplatComponent._unified = true` by default (`:90159`). In unified mode
`component.material` and `component.instance` return `null` with a deprecation warning, and the
engine states non-unified "will be removed in a future release". Unified + extra streams gets the
same result as SuperSplat's approach with roughly a third of the shader surface and no
deprecation cliff.

### 1.3 Gaussian storage, and one trap

`GSplatData` (`playcanvas.mjs:41638`) holds raw PLY typed arrays. Useful members:
`numSplats`, `getProp(name)`, **`addProp(name, storage)`** (the sanctioned hook for per-splat
custom data), `createIter()`, `calcAabb(result, pred)`, `calcAabbExact()`, `getCenters()`,
`calcFocalPoint()`, `get shBands`, `calcMortonOrder()`, `reorder(order)`.

There is **no** `transform()` and no `filter()` on `GSplatData`. Those live in `splat-transform`.

`GSplatResource` keeps a reference to `gsplatData`, so **the raw CPU arrays stay alive after GPU
upload** — which is what makes writing back a `.ply` from the browser possible at all.

**The trap:** `tools/pc-engine/src/framework/parsers/ply.js:619` runs
`if (asset.data.reorder ?? true) data.reorderData();`. Splats are **Morton-reordered by default**,
so resource index ≠ original file row. Any edit that must round-trip to the source `.ply` by index
needs the asset constructed with `{ reorder: false }`.

Also useful: `ply.js:614` fires `asset.fire('load:data', data)` **before GPU upload** — the correct
place to `addProp('state', new Uint8Array(n))`.

---

## 2. Two hard walls on the generative side

### 2.1 Wall 1 — 6 GB VRAM

Every published method that fills genuinely unobserved regions couples a 3DGS optimiser to a
Stable-Diffusion-class UNet held resident at the same time. Figures below are the **authors' own
stated hardware**, not estimates, except where marked:

| Method | Venue | Stated hardware | 6 GB? |
|---|---|---|---|
| G4Splat | ICLR 2026 | **A100 80 GB**; dense mode "3090 24 GB" with `--use_downsample_gaussians` | No |
| GSFix3D | 3DV 2026 | RTX 4500 Ada 24 GB | No |
| ViewCrafter (smallest of 4 configs) | TPAMI 2025 | **13.8 GB / 50 s** at 320×512, 25 frames — from their published table | No |
| Inpaint360GS | WACV 2026 | RTX 4090 | No |
| RI3D | ICCV 2025 | 2080 Ti 11 GB | No |
| WonderJourney | CVPR 2024 | README: "requires 24 GB GPU memory" | No |
| Invisible Stitch (360° mode) | 2024 | "at least 16 GB VRAM" | No |
| GenFusion | CVPR 2025 | *not stated* — ~24 GB inferred from DynamiCrafter at 960×512 + live 2DGS | No (est.) |
| Gaussian Grouping | ECCV 2024 | *never stated* | Unknown, >6 est. |
| AuraFusion360 | CVPR 2025 | *never stated* | Unknown, >6 est. |
| GScream | ECCV 2024 | developed on 3090 24 GB | Unknown, >6 est. |

When a paper calls 24 GB "limited", 6 GB is not a supported configuration. Note also that
ViewCrafter is the only project in this entire survey that publishes an honest VRAM table — which
is worth respecting, and worth noting as the exception.

The one exception category is **single-image 2D passes**: an inpainting UNet or a depth UNet run
on one image at a time, fp16, at modest resolution. Those genuinely do fit. That observation is
what makes Rung 4 in §7 viable and everything above it not.

### 2.2 Wall 2 — licences, and the root LICENSE file is not the whole story

The critical mechanism: **a permissive root LICENSE does not clear you if the CUDA rasterizer
submodule is Inria-licensed.** `graphdeco-inria/gaussian-splatting`,
`city-super/Scaffold-GS`, and `hbb1/diff-surfel-rasterization` were all read directly and all
carry the Inria Gaussian-Splatting non-commercial licence.

**Non-commercial by inheritance** (root licence is permissive, submodule is not):
Inpaint360GS (Apache root, `diff-gaussian-rasterization`), InFusion (MIT root,
`diff-gaussian-rasterization-confidence`), AuraFusion360 (Apache root, `.gitmodules` pulls
`diff-surfel-rasterization` + `simple-knn` from gitlab.inria.fr), G4Splat (no root licence,
`diff-surfel-rasterization`), LightGaussian, Mini-Splatting, Compact-3DGS, DNGaussian, FSGS,
CoR-GS.

**Non-commercial, stated outright:**
- GSFix3D — README: components "restrict **the entire project to non-commercial use only**"
- GaussianEditor — S-Lab License 1.0, "non-commercial purpose"
- Instant3dit — Adobe Research License, "noncommercial research purposes … only"
- Difix3D / `nvidia/difix` — NVIDIA License §3.3, "only … non-commercially"
- LucidDreamer — CC-BY-NC-SA-4.0, and ShareAlike is **viral**
- PartField — NVIDIA License §3.3, non-commercial research/education
- PyMeshLab — GPL-3
- PyMeshFix — GPL-3 **or** a paid contract with IMATI-GE/CNR; README states it "cannot be used for commercial purposes without a proper licensing contract"

**No LICENSE file at all** — default all-rights-reserved, i.e. no grant to use, copy, or modify
regardless of being public on GitHub:
RI3D, GScream, 3DGS-Enhancer, See3D, SplatFlow, RealmDreamer, SparseFusion, Deceptive-NeRF, GRM,
Flash3D, `ReshotAI/gaussian-splatting-blender-addon`.

> Note: a third-party fork (`PotreeConvena/GScream`) has applied MIT to GScream. A fork cannot
> grant rights the upstream never held. Ignore it.

**Territory-locked outputs — Hunyuan3D.** Read from the raw LICENSE of both 2.0 and 2.1:
- Header, capitalised: *"THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA."*
- §1.l defines Territory as worldwide **minus** EU/UK/South Korea. §2 grants rights "for the Territory only."
- **§5.c:** *"You must not use, reproduce, modify, distribute, or display the … Works, **Output or results** … outside the Territory."* The **generated geometry itself** is territory-locked, not just the weights.
- §4: >1 million MAU across all your products ⇒ must request a licence from Tencent, granted at their sole discretion; you are **not authorised** until granted.
- §5.b: outputs may not train or improve any other AI model. AUP: no military use.

**Clean and relevant** (verified permissive, and topically useful):
`playcanvas/engine` MIT · `playcanvas/supersplat` MIT (2011-2026 PlayCanvas Ltd, verbatim MIT) ·
`playcanvas/splat-transform` MIT · `nerfstudio-project/gsplat` Apache-2.0 ·
GenFusion MIT (code — verify DynamiCrafter weights separately) · TRELLIS MIT (code *and* models) ·
Invisible Stitch BSD-3 · Nerfbusters MIT · NeRFiller Apache-2.0 · ViewCrafter Apache-2.0 ·
`mapbox/earcut` ISC · `sparkjsdev/spark` MIT · Open3D MIT · pyransac3d Apache-2.0 ·
trimesh MIT · MoGe-2 MIT (all sizes) · Metric3D v2 BSD-2 ·
MapAnything Apache-2.0 code **and** `facebook/map-anything-apache` Apache-2.0 weights.

**Two licence facts commonly got wrong:**
1. **Depth Anything V2:** only **Small (ViT-S, 24.8M) is Apache-2.0.** Base, Large *and* Giant are all CC-BY-NC-4.0. It is not "the large one is NC".
2. **VGGT:** two checkpoints, two licences. `facebook/VGGT-1B` is **non-commercial**. `facebook/VGGT-1B-Commercial` permits commercial use **except military**, and is gated behind an access form. The AUP also prohibits *"operation of critical infrastructure, transportation technologies, or heavy machinery"* — a plausible hazard if this project ever touches infrastructure inspection. Meta may unilaterally modify the agreement; continued use = acceptance.

---

## 3. The literature solves a different problem than ours

This is the most important framing in the document.

Roughly 95% of what is published as "3DGS inpainting" is **object removal**:

- The object was **seen from many angles**.
- You delete it.
- A **small** hole remains, **surrounded on all sides** by observed geometry.
- The correct answer is **background continuation** — an interpolation problem.
- The mask is derived from the removal, so it is known exactly.

Ours is **view extrapolation**:

- A 180° arc was **never photographed**.
- There is **no surrounding context** on the far side.
- The missing region is **enormous**.
- The correct answer must be **invented** — an extrapolation problem.
- There is no mask; deciding *where* the hole is, is itself part of the problem.

Methods actually aimed at the extrapolation case, and their status:

| Method | Fit | Status |
|---|---|---|
| **GSCompleter** (arXiv 2604.20155) | **Best conceptual fit found.** "Distillation-free **plugin**", completes a 3DGS scene "in seconds" via Generate-then-Register: synthesise reference images, lift to Gaussians at consistent metric scale via Stereo-Anchor View Selection, merge into the existing scene. Plugin + no distillation + seconds is exactly the right shape for a 6 GB machine. | **Paper only.** No code found. **Track this.** |
| **Bolt3D** (ICCV 2025, Google/Oxford) | Feed-forward, 6.25 s on one GPU, emits Gaussians directly via Splatter Images + a Geometry VAE, variable input view count. Project page explicitly claims it "generates unobserved scene regions **without any reprojection or inpainting mechanisms**." | **No code.** `github.com/szymanowiczs/bolt3d` → 404. |
| **GenFusion** (CVPR 2025) | Has an explicit *Scene Completion* mode: `--outpaint_type rotation --rotation_angle 90` plus `--camera_path_file`, so you can hand it a trajectory swinging round to the unobserved face. Cyclical loop: render artifact-prone RGB-D → video diffusion → restoration frames → add to training set → repeat. **MIT.** | Code released. **~24 GB.** Requires `CC=gcc-9` to build `simple-knn` + `diff-surfel-rasterization` — Linux toolchain. |
| **G4Splat** (ICLR 2026) | Exploits **planar structure** to derive metric-scale depth and supervise unobserved areas. Planar priors are a genuinely good match for a building facade. | 80 GB / 24 GB. Heavy multi-model install (pytorch3d, CGAL, `make`). Non-commercial by inheritance. |
| **GSFix3D** (3DV 2026) | Fine-tuned SD2 ("GSFixer") repairs renders from *extreme novel viewpoints and partially observed regions*. **Operates on an existing trained model** via `refine_gs.py -m <gs_model_path>` — rare and valuable. No text prompt needed. | 24 GB. **Non-commercial, stated in README.** Expects Replica-format indoor SLAM data. |
| **RI3D** (ICCV 2025) | Two **separate** diffusion priors — a *repair* model for visible regions and a dedicated *inpainting* model for **missing/unobserved** regions. Two-stage: reconstruct visible first, then hallucinate missing while holding multi-view coherence. Conceptually the closest published match to "3 sides good, 4th missing". | 11 GB, 5 sequential training stages incl. per-scene LoRA. **No LICENSE file.** |
| **ExtraGS** | View extrapolation with uncertainty-guided virtual camera sampling to "actively explore blind spots". Right idea. | No code; endoscopy-specific. |
| **ReconSplat** (ECCV 2026) | Project page exists (`visinf.github.io/reconsplat`). | Code release unverified. |

**Google published nothing usable.** ReconFusion, Cat3D, CAT4D — no code, no weights, not
reproducible. ReconFusion's public artefact is only the eval data splits (which GenFusion reuses).
Zip-NeRF, which ReconFusion regularises, is also unreleased. Architecturally Cat3D is the ideal
answer to "any number of input images → consistent novel views"; practically it does not exist
outside Google.

**Difix3D+ is deprecated by its own authors.** NVIDIA opened issue #67 on their own repo titled
"DiFix is a previous generation — please use Fixer", and the HF card states verbatim: *"DiFix is a
previous-generation model: please use Fixer for active development and support."* The licences
diverge sharply:

| | Difix (V1, Jun 2025) | Fixer (V2, Nov 2025) |
|---|---|---|
| Base | SD-Turbo (0.9B) | Cosmos-Predict-0.6B |
| Licence | NVIDIA License — **research/eval only** | NVIDIA **Open** Model License — commercial OK |
| Delivery | `pip install`, diffusers pipeline | **Docker + Cosmos container only** (`nvcr.io/...`), hardcoded `/work/models/base/...` paths, requires a runtime patch to the Cosmos tokenizer |

So the commercially-usable version is locked behind a Linux Docker image. On Windows without
cloud, Fixer is out of reach; Difix V1 runs but only non-commercially.

More importantly, **Difix is the wrong tool regardless of licence.** It is trained on
degraded-vs-clean image pairs and conditions on "the closest training view". The paper states the
failure mode outright:

> "When the desired novel trajectory is too far from the input views, the conditioning signal
> becomes weaker and the diffusion model is forced to hallucinate more."

The entire Difix3D progressive scheme exists to *avoid* that regime. A 90-180° unobserved face is
precisely where the conditioning collapses. Difix will smooth floaters into something less jagged.
It has no mechanism to invent a wall.

**No 3DGS equivalent of Nerfbusters exists.** Nerfbusters (MIT, 233★) is the right *idea* for
floater removal in unobserved space — a local 3D geometry prior plus a visibility mask that
deletes anything outside the observed frustum. But it is NeRF-only (requires Nerfacto), pinned to
a fork branch `nerfstudio@nerfbusters-changes`, `torch==1.13.1+cu117`, Python 3.8, `tiny-cuda-nn`
(MSVC compile), and `binvox` which the README fetches as a **Linux-only binary**. GitHub searches
for a 3DGS equivalent returned zero repositories. The Tools section of
`MrNeRF/awesome-3D-gaussian-splatting` contains only editors and viewers.

**Practical consequence:** the visibility-mask half of Nerfbusters is ~50 lines against gsplat and
needs no diffusion model at all. That is Rung 0 in §7.

---

## 4. Object-removal methods worth reading anyway

Not runnable here, but three contain algorithms worth lifting conceptually.

**InFusion** (`ali-vilab/Infusion`, 558★, MIT root, last push 2024-07-15) — the canonical
depth-guided splat inpainting paper, and **the most `.ply`-native workflow found anywhere**:

```
train incomplete Gaussians with masks
  → render depth + c2w + intrinsics
  → image-conditioned depth-COMPLETION diffusion (Marigold-based) restores depth in the hole
    at the SAME metric scale as the original
  → unproject to new Gaussians
  → compose.py --original_ply --supp_ply --save_ply
  → 150-iteration fine-tune
```

The key design property: **it needs only ONE manually inpainted 2D reference image** plus a
hand-drawn `mask.png`. Multi-view consistency comes from single-reference depth unprojection, not
from an iterative diffusion loop. That single-reference design is what makes it *cheap*, and each
step fits in 6 GB **separately** — you never need the full stack resident at once.

Caveats: torch 1.12.1 / CUDA 11.6 / Python 3.8 (predates practical support for this GPU),
training code never released (a "To-do" since 2024), repo untouched ~2 years, 21 open issues
including "Bad results on SPIn-NeRF, INF and NAN in depth estimation" (#25). Depth checkpoint is
6.86 GB on disk in fp32. The README is unusually candid that floater cleanup via
`--nb_points`/`--threshold` is "**very important**" and needs per-scene tuning — which independently
corroborates Rung 0.

**3DGIC** (CVPR 2025) — contains **the single most transferable algorithm in this survey**:
rendered depth is used to project background regions between views, **iteratively shrinking the
inpainting mask until it contains only regions not visible in *any* training view.** That is
exactly the "which part of my scene is genuinely unobserved?" computation needed for automatic
hole detection (the "auto later" half of the decision in §0). Read `find_depth_guided_mask.sh`.

Do not plan to run it: the author explicitly states the release is a *"suboptimal version"* — the
full implementation belongs to the sponsoring company (Tron Future Tech) — the original
environment was deleted, and it needs nvdiffrast + r3dg-rasterization + bvh + vismvsnet.

**Inpaint360GS** (WACV 2026, 46★, Apache root, last push 2026-02-28) — **best-maintained repo in
the survey**, 0 open issues, full dataset/results/eval released, honest FAQ about LaMa mask
dilation. Its depth+RGB LaMa stage is the clearest worked example of the depth-guided pipeline.
Read it as a reference implementation. `--resolution 1,2,4,8` "adjust according to GPU memory" is
the only downward lever, and the CropFormer + 3DGS stage will not fit 6 GB at useful resolution.

**Gaussian Grouping** (ECCV 2024, 1,042★, Apache-2.0 verified) is the most-adopted and
most-forked (73) — the base that both 3DGIC and Inpaint360GS build on, so its code quality is
probably fine. But: untouched >2 years, 68 open issues, VRAM never stated, and Identity Encodings
add per-Gaussian channels so it costs *more* than vanilla 3DGS. Its DEVA text prompt for finding
the hole is literally `"black blurry hole"`.

---

## 5. Interactive editing tooling

### 5.1 SuperSplat — and the v2/v3 fork that matters

`playcanvas/supersplat`, 9,944★, 1,118 forks, last push 2026-09-03. **MIT**, verified by reading
the LICENSE (Copyright 2011-2026 PlayCanvas Ltd, verbatim MIT, no additional clauses).

**The critical fact: `main` is `3.0.0-alpha` and is WebGPU-only.** `src/main.ts:129` requests
`deviceTypes: ['webgpu']`; the startup error string is literally *"SuperSplat requires WebGPU"*.
All selection kernels are rewritten as WGSL compute + `StorageBuffer`.

**The latest release, `v2.32.5`, is WebGL2** (`src/main.ts:128`) and does selection with a plain
GLSL fragment shader + `drawQuadWithShader`. **For this repo's stack, v2.32.5 is the version to
mine, not `main`.**

| | v2.32.5 (release) | main / 3.0.0-alpha |
|---|---|---|
| Device | `['webgl2']` | **`['webgpu']`** |
| Selection kernel | GLSL frag + `texture.read()` readback (`shaders/intersection-shader.ts`, 106 LOC) | WGSL **compute** (`data-processor/intersect.ts`, 396 LOC) |
| Per-splat state | `SplatState` — `Uint8Array` mirrored to a `PIXELFORMAT_R8` texture (124 LOC) | `GaussianInstances` — 3 × `StorageBuffer` (464 LOC) |
| Data residency | whole `GSplatData` in RAM | lazy `ChunkSource` streaming |
| gsplat mode | `unified: false` + a fully owned `ShaderMaterial` | own `projected-splat-renderer.ts` (43 KB), bypasses the engine renderer |
| Extra tools | — | eyedropper, **3D sphere brush** |

v3's architecture is a total rewrite around WebGPU compute and streaming. Its selection code is
**not liftable** into a WebGL2 app. v2.32.5's is, and it is small and clean.

**Cannot be embedded.** The entire iframe API is 39 lines (`src/iframe-api.ts`) and exposes exactly
one message: `supersplat:is-scene-dirty`. No "get selection", no "load buffer", no "export".

**Cannot be forked as a library.** There is no library boundary — everything routes through a
global string-keyed `Events` bus, and the UI is 40 PCUI files + 24 SCSS files + i18next. Forking
means adopting TypeScript + Rollup + SCSS + PCUI + i18next, i.e. abandoning the no-build-step
constraint that `05_MVP_Agent_Prompt.md` established.

**Can be selectively lifted, and that is the right answer.** Near-zero-dependency, MIT:

| File (v2.32.5) | LOC | Deps | Note |
|---|---:|---|---|
| `src/index-ranges.ts` | 89 | **none** | run-length index set; makes "add 400k splats to selection" cost a few hundred `Uint32` entries and makes every op exactly invertible → clean undo |
| `src/select-op.ts` | 10 | none | `shift+ctrl`=intersect, `shift`=add, `ctrl`=remove, none=set |
| `src/splat-state.ts` | 124 | `pc.Texture` | CPU↔GPU mirror with dirty-span tracking. Pushes `state` onto the PLY element as a real `uchar` property so the serializer sees it for free. `State { selected=1, locked=2, deleted=4 }` |
| `src/splat-pick.ts` | 121 | `pc.{Mat4,Ray,Vec3,Vec4}` | **best-in-class pick heuristic** — see §7.1 |
| `src/tools/polygon-selection.ts` | 166 | **none** (DOM/SVG) | **this is the "patch face" outline UI.** Click-to-place verts, closes when within 8 px of the first point (stroke turns `#fa6` as a hint), dbl-click or Enter commits, Backspace removes last |
| `src/tools/lasso-selection.ts` | 153 | none | freehand variant, with adaptive point spacing |
| `src/tools/brush-selection.ts` | 175 | none | round-cap stroke, alt+wheel radius |
| `src/tools/rect-selection.ts` | ~120 | none | |
| `src/tools/orient-tool.ts` → `calcPlane()` | ~40 of 22 KB | `pc.{Vec3,Quat}` | **the 3-click plane fit.** See §7.2 |
| `src/shaders/intersection-shader.ts` | 106 | GLSL text | mask/rect/sphere/box intersection |
| `src/data-processor/intersect.ts` | 207 | pc engine | WebGL2 driver — or prefer `pc.GSplatProcessor` (§1.2), same idea, less boilerplate |
| `src/edit-ops.ts` → `StateOp`, `SelectOp` | ~150 of 558 | `IndexRanges` | undo/redo |
| `src/splat-serialize.ts` PLY path | ~250 of 838 | `pc.{Vec3,Quat,Mat3,Mat4}` | best reference for writing back a `.ply`, incl. SH rotation and a careful NaN allowlist (`opacity` may be ±Inf, `scale_*` may be −Inf) |
| `src/tool-overlay.ts` + shader | ~450 | pc engine | draws a **translucent plane fill occluded correctly against the splats**, so you can see how a plane sits relative to the surface. This is the visual feedback a patch tool needs. |
| everything in `src/ui/` | ~250k | PCUI, SCSS, i18next | **do not attempt** |

Realistic total for a first-cut editor lifted from SuperSplat: **~900-1,200 lines of plain JS.**

Every selection tool follows one pattern, which is what makes them so portable: *the tool draws a
2D shape onto a shared `<canvas>` with Canvas2D plus an SVG cursor overlay, then fires one event
with `(op, canvas, context)`.* The entire brush is 12 lines of Canvas2D.

Selection **methods** are orthogonal to selection **tools** (`editor.ts:566-694`):
`centers` (test the centre point — cheapest), `footprint` (test the 3D ellipsoid extent at 2√2·σ
using a support-function bound — conservative, never misses a real overlap), `pick`/`depth`
(per-splat ID buffer readback — only selects *visible* splats), and `surface` (v3 only).

**One v3 gotcha worth knowing:** its SOG export needs a *second, headless* `WebgpuGraphicsDevice`
and throws `WebGPUUnavailableError` if unavailable (`splat-serialize.ts:562-608`). PLY export has
no such requirement.

### 5.2 splat-transform — already in this pipeline

npm `@playcanvas/splat-transform@3.3.3`, **MIT**, 1,305★, pushed 2026-08-25, ~12.6k weekly
downloads. Already driven via `npx -y` from `scripts/build_collider.py:177`.

Reads: `.ply`, `.compressed.ply`, `.sog`/`meta.json`, `lod-meta.json`, `.spz`, `.splat`, `.ksplat`,
`.lcc`, `.lcc2`, `.mjs` (generator).
Writes: `.ply`, `.compressed.ply`, `.sog`, `lod-meta.json`, `.spz`, **`.glb`**
(`KHR_gaussian_splatting`), `.csv`, `.html`, `.voxel.json`, `.webp`, `null`.

Operations relevant here:

| Category | Flags |
|---|---|
| Transform | `-t/--translate`, `-r/--rotate` (Euler°), `-s/--scale` |
| **Crop / filter** | `-B/--filter-box`, `-S/--filter-sphere`, `-V/--filter-value name,cmp,value` (`_raw` suffix for pre-activation), `-N/--filter-nan`, `-H/--filter-harmonics 0..3`, **`-F/--filter-floaters`**, **`-C/--filter-cluster [res,op,min]`** (connected component from `--seed-pos`) |
| Combine | multiple inputs merge, with per-input actions |
| Decimate | `-d n|n%`, `--decimate-adaptive`, streams to 100M+, spills to `--scratch-dir` |
| Voxelize | `--voxel-size`, `--voxel-opacity`, `--voxel-external-fill`, `--voxel-floor-fill`, `--voxel-carve`, `--seed-pos`, `--collision-mesh [smooth\|faces]` ← already used to build `collision.collision.glb` |
| Analysis | `--stats [text\|json]` per-column min/max/median/mean/stdDev/NaN/Inf/histogram + **`fillRatio`** overdraw estimate |
| Generate | `.mjs` generator scripts + `-p key=val` (Beta) — **a procedural way to synthesize splats for a filled patch region** |

**What it cannot do:** no arbitrary-polygon or mesh-volume filter. Only box, sphere, value,
cluster, floaters. A patch region must be approximated by boxes/spheres, or the PLY written
directly.

It also has a **browser-capable programmatic library API** (`readFile` / `processSourceBridged` /
`writeSource`, `UrlReadFileSystem`, `MemoryFileSystem`) described as "platform-agnostic … both
Node.js and browser". Using it in-browser means bundling (2 deps), so it conflicts with
no-build-step today — but it is the cleanest possible write-back backend if that rule ever relaxes.
There is also a hosted frontend at `superspl.at/convert`.

### 5.3 Other editors surveyed

| Name | ★ | Licence | Region select / paint? | Liftable here? |
|---|---:|---|---|---|
| **PlayCanvas engine gsplat examples** | 16,610 | **MIT** | ✅ AABB select/delete/clone, **3D brush paint**, crop, depth pick | ★★★★★ **directly** — see §1 |
| SuperSplat v2.32.5 | 9,944 | MIT | ✅ rect, lasso, polygon, 2D brush, sphere, box, flood, eyedropper, hide, delete, duplicate | ★★★★☆ **selectively** — §5.1 |
| splat-transform | 1,305 | MIT | offline filters only | already in pipeline |
| **Spark** (World Labs) | 3,563 | **MIT** | ✅ **`SplatEdit` + `SplatEditSdf`** — SDF-region RGBA/XYZ edits (`PLANE`,`SPHERE`,`BOX`,`ELLIPSOID`,`CYLINDER`,`CAPSULE`,`INFINITE_CONE`), softmax blending, `sdfSmooth`/`softEdge`/`invert`, blend modes (`MULTIPLY` @ opacity 0 = delete a region). Plus a GPU "dyno" shader-graph to create/edit splats. | ❌ three.js-native, and building needs **Rust/wasm-pack**. **But its SDF shape catalogue is the best-designed "region" vocabulary in the field. Port the vocabulary, not the code.** |
| **UnityGaussianSplatting** | 3,395 | MIT code (⚠️ README warns the Inria licence applies to how you *obtained* the PLY) | ✅ rect drag-select, Delete, Ctrl+I invert, `W` move selected. **`GaussianCutout`** (Ellipsoid/Box, Transform-driven, `Invert`) virtually deletes splats — **and selection ops respect cutouts, so a box cutout constrains manual editing to that box.** `Export modified PLY` with world-space bake + SH rotation. No Undo. | ❌ Unity/C#/HLSL. **The cutout-as-selection-constraint idea is genuinely clever and worth stealing.** |
| `@mkkellogg/gaussian-splats-3d` | 2,881 | MIT | ❌ (has a mesh cursor showing ray/splat intersection) | ❌ three.js. **README says it is no longer actively developed and points to Spark.** |
| `antimatter15/splat` | 3,071 | MIT | ❌ viewer only | ❌ raw WebGL1. **Its README is the best plain-English explanation of splat sort/transparency trade-offs.** Also now points to Spark. |
| `huggingface/gsplat.js` | 1,662 | MIT | ⚠️ a "gsplat-editor" HF Space exists; library exposes object-level transforms only | ❌ own renderer, semi-dormant |
| **Brush** (ArthurBrussee) | 5,024 | **Apache-2.0** | ❌ trainer+viewer. Has image masking + **region-of-interest training** | ❌ Rust/wgpu/Burn, **Chrome/Edge 134+ WebGPU only.** Relevant because splat-transform reads its `comment SplatRenderMode:` PLY tag |
| **3DGS Render for Blender** (KIRI) | 1,178 | ⚠️ **GPL-2.0** | ✅ edit with Blender's native selection, modifiers, cropping, colour; export edited + animated `.ply`; mesh→3DGS | ❌ **GPL-2.0 copyleft — do not read-then-write code from this into an MIT/proprietary web app.** Its *workflow* (splat ↔ mesh proxy) is the interesting part |
| `ReshotAI/gaussian-splatting-blender-addon` | 588 | **none** | ✅ via Blender | ❌ stale since 2024-08, unlicensed. Skip |
| **Jawset Postshot** | — | closed, EULA | ✅ advertises "Selection, Editing", merge scenes, image masking, **region-of-interest training** | ❌ nothing liftable. **But ROI-training is arguably the *correct* answer to "mark a region that should be regenerated": mark it → retrain only that region at higher density.** |
| Polycam / Luma / KIRI Engine app | — | closed | ⚠️ low-to-medium confidence, help pages 404'd | ❌ |

### 5.4 Patch authoring — the pieces

There is no turnkey "draw a patch on a splat" library. Every component exists:

| Stage | Best option | Licence | Note |
|---|---|---|---|
| Draw the outline | SuperSplat `polygon-selection.ts` (166 LOC) / `lasso-selection.ts` (153 LOC) | MIT | pure DOM/SVG/Canvas2D, zero deps |
| Unproject each vertex | `camera.screenToWorld` + `raycastFirst` (already in `pc.js`) | — | gives point **and normal** per vertex |
| Fit a plane from clicks | SuperSplat `orient-tool.ts` `calcPlane()` | MIT | §7.2 |
| **Triangulate** | **`mapbox/earcut`** — 2,584★, pushed 2026-08-02, **ISC, ZERO deps, single ES module, v3.2.3**, handles holes | ISC | ★★★★★ correct choice for a no-build app: drop the file next to `playcanvas.mjs` |
| Interior vertices (subdividable / displaceable patch) | `delaunator` (ISC) or `cdt2d` (MIT) or `poly2tri` (BSD-3) | | only if the patch needs relief |
| Polygon booleans | `polybooljs` (MIT), `polygon-clipping` (MIT), `martinez` (MIT) | MIT | only if patches merge/subtract |
| Coons / NURBS patches | `verb-nurbs` — 812★, pushed 2025-04-02, MIT, has real `NurbsSurface.byCorners` and `byLoftingCurves` | MIT | Haxe-transpiled, chunky, but the only real Coons/NURBS impl in plain JS. `nurbs` (MIT, 0 deps) is smaller but evaluation-only |
| Solid CSG | `manifold-3d` (2,255★, Apache-2.0, WASM) · `@jscad/modeling` (MIT) · ⚠️ `opencascade.js` is **LGPL-2.1-only** | mixed | overkill for a face patch |
| Fast mesh raycast / closest point | `three-mesh-bvh` (3,471★, MIT) | MIT | ❌ three.js-coupled, but its SAH BVH construction is worth reimplementing over `pc.Mesh` (~250 LOC) if ammo raycast proves too coarse |

**Prior art already inside this repo's dependency tree:**
- `tools/pc-engine/examples/.../graphics/mesh-decals.example.mjs` — projecting geometry onto a surface at a picked point. Relevant if the patch should *conform* rather than be planar.
- `tools/pc-engine/examples/.../misc/editor.{example,selector,gizmo-handler}.mjs` — minimal in-engine editor: picker selection + `Translate/Rotate/ScaleGizmo`. That is the patch-manipulation UI.
- `viewer/pc.js:282-500` already builds five procedural `pc.Mesh` objects (`buildCameraFrustumsMesh`, trajectory, sparse points, coverage voxels) with `setPositions/setIndices/update` + `unlitMat`. **The patch mesh is the sixth. The plumbing exists.**

**How architectural tools actually define a facade plane** (consistent across ArchiCAD / Revit /
Rhino-family): pick 3+ points on the surface → least-squares/SVD plane fit → **snap the normal to
the nearest global axis or to gravity within a tolerance** → establish a 2D basis on the plane
(first picked edge = local +X) → all subsequent drawing happens in that 2D basis. That "work
plane" / "construction plane" concept is what makes facade authoring tractable, and
`orient-tool.ts` implements steps 1-4 in MIT JS **against a splat cloud** in ~40 lines of maths.

---

## 6. Image-to-3D generators — assessment

### 6.1 The domain problem, which is worse than the VRAM problem

Every tool in this category is an **object** generator: trained on Objaverse-style single assets,
background-removed, normalised into a **unit cube**, with **no metric scale** and no notion of
"this is a 6 m chunk of wall that must tile continuously with existing geometry."

Feeding it a cropped photo of a building side gives a plausible *model of a building in a cube*,
not a metrically-correct continuation of *this* building. Scale that cube to 20 m and the generated
Gaussians scale with it: metre-wide blobs against centimetre-scale captured splats, plus a visible
seam of mismatched density and lighting.

PartCrafter's own repo ships a `--style_transfer` flag that VLM-converts real photos into
Objaverse-style renders "to bridge the domain gap" — a candid admission by the authors that **real
photos are out-of-distribution** for this class of model. That admission applies to the whole
category.

### 6.2 Ranked by realistic usability at 6 GB, Windows, no CUDA compile

| Tool | ★ | Licence | In → Out | VRAM | Windows |
|---|---:|---|---|---|---|
| **splat-transform** | 1,305 | **MIT** | splat PLY → PLY/SOG/GLB | none (npm) | ✅ native |
| **TRELLIS** via `IgorAherne/trellis-stable-projectorz` | 13.6k upstream | **MIT** (code *and* models; only `diffoctreerast` + modified FlexiCubes differ) | 1-N images → **mesh AND 3D Gaussians** | fork claims **8 GB** (fp16, int32, halved SLAT-decode) | ✅ one-click installer, but ships its own torch 2.7 — will not share the 2.4.1 env |
| **TripoSR** | 6,922 | **MIT** (code, weights, demo) | 1 image → mesh | **~6 GB stated** | ⚠️ `torchmcubes` silently builds without CUDA — documented failure mode |
| **SF3D** | 1,799 | ⚠️ Stability Community (free <$1M rev, **must register**, auto-terminates above, "Powered by Stability AI" attribution) | 1 image → mesh + UV + **PBR, with delighting** | **~6 GB stated** | ⚠️ "experimental", VS 2022; HF weights **gated** |
| **Hunyuan3D-2mini** via `deepbeepmeep/Hunyuan3D-2GP` | 14.7k upstream | ⚠️ **Tencent Community — EU/UK/KR banned, outputs included** | image → mesh | claims **<6 GB** with `--profile 4/5` | ✅ but needs VS 2022 for `custom_rasterizer` + `differentiable_renderer` (texture path only) |

TRELLIS is the strongest conceptual fit **because it emits 3D Gaussians natively**
(`outputs['gaussian'][0].save_ply(...)`), skipping mesh→splat entirely. Its multi-image mode does
exist (`run_multi_image()` with `stochastic` and `multidiffusion` modes) but the README is explicit
it is *"based on tuning-free algorithm without training a specialized model, so it may not give
the best results"*. Upstream README says **Linux only, 16 GB minimum**, verified on A100/A6000;
`setup.sh` compiles flash-attn, spconv, kaolin, nvdiffrast, diffoctreerast, mip-splatting. The
Windows fork is the only reason it's on this list, and it targets 8 GB, not 6.

**Hunyuan3D-2mv** is the most technically apt model in the entire survey — the only widely-used
checkpoint genuinely *trained* for multi-view conditioning, taking an explicit
`{"front":…, "left":…, "back":…}` dict. It is also the one whose §5.c contaminates the deliverable
rather than just the toolchain. Hunyuan3D VRAM from the official model zoo: 2.0 shape 6 GB /
+texture 16-24.5 GB; 2.1 shape 10 GB / total 29 GB. **Texture generation is out of reach at 6 GB in
every variant**, so the realistic output is untextured geometry with texture solved separately.

### 6.3 Feed-forward reconstruction models — a category correction

**VGGT, MapAnything, Pi3, AnySplat, MVSplat, DepthSplat, NoPoSplat do not generate unseen
geometry.** They are feed-forward *reconstruction* — they infer cameras/depth/points/Gaussians from
images that already exist. They cannot invent a side that was never photographed.

They are useful for a **different** job: replacing or seeding COLMAP, and — critically — estimating
the pose and **metric scale** of a newly-generated or newly-shot chunk so it can be aligned.

- **MapAnything** is the strongest of the group and the most actively maintained in this whole survey (pushed 2026-08-07, 2 open issues). Apache-2.0 code, **two** weight sets: `facebook/map-anything` (CC-BY-NC-4.0) and **`facebook/map-anything-apache` (Apache-2.0, commercially clean)**. Outputs **metric** 3D, accepts any combination of images + intrinsics + depth + poses, ships `demo_colmap.py` and documented gsplat integration, and `memory_efficient_inference=True` with `minibatch_size=1` is designed for small GPUs. It also wraps VGGT, Pi3, MoGe, DUSt3R, MASt3R, DA3 behind one interface — the cheapest way to evaluate the whole family.
- **DepthSplat** (MIT, 1,243★) is notable only because its stated env is **torch 2.4.0 / CUDA 12.4 / Python 3.10 — an exact match for `.venv310`** — and `test.save_gaussian=true` writes viewer-compatible PLYs. But it's trained on RealEstate10K/DL3DV (indoor, forward-facing).
- **Pi3**: code BSD-3, **weights CC-BY-NC-4.0, "Strictly Non-Commercial"**.
- **AnySplat**: MIT, weights at `lhjiang/anysplat`; the correct repo is `OpenRobotLab/AnySplat` (not `OpenGVLab`).
- **SpaTracker**: NOASSERTION (inherits CoTracker NC), README states **22 GB** for dense tracking. Irrelevant here.

### 6.4 Part-level and completion categories — mostly not applicable

- **PartField** (⚠️ **NVIDIA License §3.3, non-commercial research/education only**): predicts part *feature fields* for **segmentation**, not geometry. Two interesting details: it **accepts `.ply` Gaussian splats as input** (`is_pc True`) with K-Means clustering, and its stated env is **Python 3.10 / torch 2.4 / CUDA 12.4 — this repo's exact stack.** Candidate for a "user clicks a region" selection UX. Needs `torch-scatter` with a matching wheel.
- **HoloPart** (MIT, 2B fp16): "part amodal segmentation" — completes occluded parts of an existing mesh. Closest in spirit to amodal completion of anything surveyed. But requires an already-segmented mesh (SAMPart3D/SAMesh as a separate step) and is trained on Objaverse objects. A building facade has no part decomposition it understands.
- **PartCrafter** (NeurIPS 2025, **8 GB min**): has a **scene-level variant trained on 3D-Front** — the only object-generator here with any scene training. Still indoor synthetic rooms. Windows only via `JackDainzh/PartCrafter-Windows`.
- **Point-cloud completion (PoinTr, AdaPoinTr, SeedFormer, ODGNet):** dead on arrival. Install requires compiling Chamfer Distance, PointNet++ ops, and KNN_CUDA from source. Conceptually they operate on **2,048-16,384-point normalised ShapeNet objects with no colour**. Output would be a sparse colourless blob.
- **DiffComplete / SC-Diff / PatchComplete / Scan2Mesh:** the *most conceptually correct* category for this problem — SDF/TSDF completion of partial real scans, i.e. literally amodal 3D from partial data. But `RuihangChu/DiffComplete` 404'd and no maintained Windows-friendly implementation was found. **Treat as literature, not tooling.**
- **Texture inpainting (TEXTure, Text2Tex, Paint3D, MVPaint, Hunyuan3D-Paint, UniTEX):** Paint3D is Apache-2.0 and produces *lighting-less* 2K UV maps, which would be genuinely valuable for compositing into a scene with different baked lighting. But the category runs SD + ControlNet + differentiable rasterisation, needs `kaolin==0.13.0` (version-pinned wheels) and PyTorch3D, and Hunyuan3D-Paint alone needs 21 GB. **Defer texture entirely.**
- **Michelangelo** — 486★, **GPL-3.0** (unusual copyleft in this space), last push 2024-04, superseded by TripoSG/Hunyuan3D which both cite it as ancestry.
- **LGM** (MIT, dormant 2024-08) and **GRM** (**no LICENSE at all**, dormant 2024-04): both need custom rasterizer compiles.
- **Splatter Image** (BSD-3, dormant): needs `diff-gaussian-rasterization`, old torch/PyTorch3D pins, per-category ShapeNet/CO3D weights.
- **SuGaR** solves the *inverse* problem (mesh from splats), and its own to-do list says *"the current code is not compatible with Windows."*

### 6.5 Mesh → Gaussian splats

There is **no mature turnkey converter**. Ranked:

1. **Sidestep it.** TRELLIS emits Gaussians directly. This is the strongest argument for TRELLIS over Hunyuan3D despite tighter VRAM.
2. **`gsplat-mesh.mjs`, already vendored** (§1). Scanline-rasterizes triangles into uniformly spaced splats with margin control, in-browser, MIT, no GPU training. For the flat-wall decision in §0 this is very likely sufficient on its own.
3. **Direct surface sampling in Python.** Poisson-disk sample the mesh surface; one Gaussian per sample; quaternion aligns the shortest axis to the vertex normal; two tangential scales ≈ sample spacing, normal scale ≈ 1/10 of that (flat discs); `f_dc_*` from albedo; high opacity; **zero higher-order SH**. ~100 lines with `trimesh` + `plyfile`. No GPU, no training, no CUDA. Because world coordinates are chosen at sample time, output is **already metrically placed** — no post-hoc rescale.
4. **Train gsplat on renders of the mesh.** Render ~100-200 virtual cameras at final metric scale in the scene's frame, write COLMAP cameras (the binaries are already in `tools/colmap/`), run the existing trainer. Highest fidelity, matches the existing splat's statistical character, reuses working infrastructure. Costs one training run per patch — minutes at object scale on 6 GB.
5. **`splat-transform`** for the merge/transform/decimate afterwards.

Blender addons for this are hobby-grade or GPL. Don't depend on one.

### 6.6 Alignment — the part no README addresses

Any generated chunk arrives as a **normalised cube with arbitrary orientation and no metric
scale**. Getting it into a metric COLMAP scene is a **7-DoF similarity transform** that must be
solved locally:

1. User paints the region. Extract the enclosing world-space OBB **and the boundary ring of real Gaussians surrounding the hole** — that ring is the only ground truth available.
2. Render conditioning views **from the real splat**, from virtual cameras looking at the hole. Better than raw drone frames because framing is controlled and the surrounding context arrives in the same radiometry.
3. 2D-inpaint the hole in those renders. Then feed 2-4 consistent views to a multi-view generator.
4. Solve the transform: coarse = scale generated bbox onto the painted region's world bbox; refine = **Umeyama/Procrustes on correspondences in the overlap band, then ICP** restricted to that band.
5. Convert to splats at final metric scale; merge with `splat-transform`.
6. **Feather the seam:** ramp opacity down over the overlap collar on the generated side and cull real Gaussians inside the hole, so it cross-dissolves rather than butt-joints.

Step 4 requires deliberately generating **more than the hole** — an overlap collar of real observed
surface. Generate a patch with no overlap and there is nothing to register against; placement
becomes pure guesswork.

**Expect step 4 to fail on flat, featureless, or repetitive facades** — precisely the geometry
buildings are made of. ICP has no signal on a plane, and repetitive windows create false minima.
Plan a manual gizmo fallback. Do not promise automatic alignment.

---

## 7. The ladder — ranked by value per unit of effort

Rungs are additive, not alternatives. Each is independently shippable.

### Rung 0 — Delete the garbage (highest value per hour, by a wide margin)

Floaters in a never-filmed region are, by construction, (a) seen by very few cameras and (b)
**disconnected** from the structure. Three cheap mechanisms:

**(a) Visibility pruning.** `scripts/solve_frame.py:93` `multiview_support()` already counts
cameras whose frustum contains each point and records nearest range, and already applies
`(n_views >= min_views) & (near <= max_dist)`. **This is exactly the right primitive.**
Gaussians seen by exactly 1-2 cameras are almost always floaters, because 3DGS needs multi-view
agreement to place real geometry. Cutting `n_views <= 2` is nearly free and nearly always correct.

Known limitation: it is a **frustum test, not a visibility test.** A gaussian behind a wall counts
as "seen" by every camera pointed at the wall. For floaters in genuinely empty air this is fine.
For floaters *inside* a building volume it needs occlusion, which means rasterising a depth buffer
per training camera and rejecting gaussians whose depth exceeds it by more than a slack. At 114
keyframes that is seconds on 6 GB with `render_eyelevel.py` + gsplat, and needs no training.

**(b) Connectivity filter — one CLI flag.**
```
splat-transform scene.ply --filter-cluster --seed-pos 0,1,0 --filter-floaters cleaned.ply
```
`--filter-cluster` voxelizes and keeps only the connected component containing the seed. MIT,
already in the pipeline, does most of the work by itself.

**(c) Manual box crop in SuperSplat** — browser, free, zero VRAM. Select the floater cloud,
delete, export `.ply`. For "stop the garbage being visible" this is the honest 80% solution and
nothing else beats its cost/benefit.

**Why this must come first:** it converts "random garbage that screams broken" into **clean
absence**, and clean absence reads as an unfinished capture rather than a bug. It is also
impossible to evaluate any fill while floaters are in the frame.

Corroboration from two independent directions: InFusion's README says floater removal via
`--nb_points`/`--threshold` is "**very important**" and scene-specific; and `PROBLEM-temple.md`
already reached the same conclusion for clouds by a different route.

**Licence note:** the entire academic pruning literature (LightGaussian, Mini-Splatting,
Compact-3DGS) is 3DGS-derived and inherits Inria's non-commercial clause. RadSplat released no
code. splat-transform is MIT and ships the functionality. The choice is easy.

### Rung 1 — Manual patch tool (the feature as described)

```
click 3 points  →  plane fit  →  draw polygon  →  triangulate  →  pc.Mesh
                →  gsplat-mesh.mjs  →  splats  →  merge  →  write .ply
```

Everything needed is MIT and either vendored or a single dependency-free file:

| Stage | Source | Status |
|---|---|---|
| Pick point + **normal** | `raycastFirst` at `viewer/pc.js:205` | already running |
| Fallback pick | `Picker.getWorldPointAsync` — `picking.example.mjs` | vendored |
| Fallback pick 2 | SuperSplat `splat-pick.ts` (121 LOC) | copy |
| Plane fit | SuperSplat `orient-tool.ts` `calcPlane()` (~40 LOC) | copy |
| Polygon UI | SuperSplat `polygon-selection.ts` (166 LOC, zero deps) | copy |
| Triangulate | `earcut` (ISC, zero deps, one file) | drop in |
| Build the mesh | pattern already at `viewer/pc.js:282-500` | exists |
| Mesh → splats | `gsplat-mesh.mjs` | vendored |
| Visual feedback | SuperSplat `tool-overlay.ts` (~450 LOC) — translucent plane fill, occluded correctly against splats | copy |
| Manipulate | `Translate/Rotate/ScaleGizmo` — `editor.gizmo-handler.mjs` | vendored |
| Merge + write | `splat-transform` | already in pipeline |

#### 7.1 Picking — the recommended fallback chain

| # | Technique | Accuracy on real captures | Verdict |
|---|---|---|---|
| 1 | **Proxy mesh raycast** (ammo, existing) | as good as the collision heightfield — misses thin/overhanging/vertical detail | ★★★★★ **start here, already wired, gives a normal** |
| 2 | Heightfield lookup (`ground.f32`) | ground only, no overhangs | ★★★☆☆ fine for terrain patches, useless for walls |
| 3 | **Depth-buffer readback** (`getWorldPointAsync`) | good; depth is a transmittance-weighted **mean**, so it lands slightly *behind* the visible surface on fuzzy captures | ★★★★★ already implemented |
| 4 | GPU per-splat ID pass | exact, front-most; needs a custom material (the engine's `pcId` stores `placementId`, **not** `splat.index`) | ★★★★☆ only needed for *which gaussian* |
| 5 | **CPU ray-vs-splat, transmittance-MEDIAN** | **best on real captures.** SuperSplat's own comment: the frontmost gaussian is often *"a large, nearly transparent floater (placing the point in mid-air)"*, while a mean-depth pick *"lands behind the surface"*. The median of the transmittance-weighted visible depths fixes both. | ★★★★★ 121 LOC, no shaders |
| 6 | Exact ray-vs-ellipsoid | mathematically exact per splat, but "which splat is the surface" remains ambiguous | ★★☆☆☆ over-engineered |
| 7 | Screen-space mask + centre projection | also selects occluded splats (often desired) | ★★★★★ this is `select.byMask` |
| 8 | Ellipsoid-footprint mask (2√2·σ support bound) | conservative, never misses a real overlap | ★★★★☆ |

Recommended: 1 → 3 → 5. Roughly 140 lines total on top of what exists.

**One thing to verify empirically before building on #3:** the WebGL2 quad renderer's mesh
instance has an `isVisibleFunc` (`playcanvas.mjs:87058`) gated on `thisCamera.camera === camera`.
`Picker.prepare` passes the same `camera.camera`, so it *should* pass — but confirm the picker
depth buffer is non-empty over the splat first. Also note the compute-based gsplat pick path
(`prepareForPicking`, `:89668`) requires `renderer.usesGpuSort`, which is true **only** for
`GSplatHybridRenderer` = WebGPU. On WebGL2 the renderer resolves to
`GSPLAT_RENDERER_RASTER_CPU_SORT` → `GSplatQuadRenderer`, whose mesh instance goes into the layer
normally, so the ordinary pick + depth-pick path still functions.

#### 7.2 Plane fit — the algorithm, and a detail worth keeping

`orient-tool.ts` `calcPlane()` / `calcPlaneRotation()`:

```
e0 = p1 - p0 ;  e1 = p2 - p0 ;  n = normalize(cross(e0, e1)) ;  c = centroid
```

Then it **snaps `n` to the nearest splat-local axis if within `SNAP_ANGLE_DEG = 3°`** — a small
detail that makes axis-aligned captures produce exact quarter turns instead of 89.4°. The rotation
is built as the shortest-arc quaternion from `n` to the target axis and applied **about the
centroid**, so the picked points don't slide laterally.

For >3 points, replace the cross product with a least-squares / SVD fit (or PCA of the covariance
matrix) — ~30 lines, no library.

### Rung 2 — Automatic procedural fill (flat wall)

This is the rung that matches the §0 decision "plausible flat wall".

1. RANSAC the observed facades (Open3D `segment_plane`, MIT; or pyransac3d `Plane`, Apache-2.0). For a building expect 2-4 large vertical planes plus a roof.
2. **Extend the observed planes until they intersect**, giving the footprint as a closed polygon. The footprint edge with no supporting plane **is** the missing facade. Its rectangle in 3D = that edge × the observed height range.
3. Populate it. Jittered lattice at ~2× median gaussian scale (≈0.2 m for this data), and per sample emit:
   - **scale:** flat in the plane normal, e.g. `[0.12, 0.12, 0.01]` in the plane frame. This is effectively 2DGS-style discs and **avoids the fuzzy-cotton look isotropic gaussians give on flat walls**.
   - **rotation:** quaternion whose local Z is the plane normal.
   - **DC colour:** sampled from the observed facades' median, **plus per-gaussian noise at the observed colour variance** so it isn't a dead flat plate.
   - **SH rest: zero.** Do not invent view dependence there is no evidence for. Zero SH = lambertian wall = the correct humble answer. (Convenient: `export_viewer_assets.py:173-179` already zeroes every `f_rest_*`.)
   - **opacity:** match the observed facade median (≈0.93 near ground in this data).

**Optional, nearly free, and disproportionately effective:** rasterize the observed facade's
gaussians into a 2D ortho texture in plane coordinates (a numpy weighted scatter-add, no renderer),
then **FFT it**. The periodic peaks in the power spectrum *are* the window-grid pitch, horizontal
and vertical. `scipy.fft` on 1024² is milliseconds. That is the honest 80% of "facade grammar
inference" with none of the grammar machinery, and the facade-grammar literature (Müller 2007,
Wonka 2003 split grammars) has **no runnable open code** — it was productised into CityEngine.
Reimplementing split-grammar inference is weeks.

**Expected quality:** flat but not embarrassing. A wall with window-like rectangles at the correct
pitch. Reads correctly in a wide shot and as a painted backdrop up close, because it has no depth —
window reveals, sills and relief are gone. Strictly better than a void or floaters.

**Do not use alpha shapes for closure.** `create_from_point_cloud_alpha_shape` will happily carve a
cavity where the missing facade is, because the algorithm shrinks toward absent data — the exact
opposite of what's wanted. Use the convex hull or explicit plane intersection. Reserve alpha shapes
for *diagnosing* where the hole is.

Cost: ~200 lines of numpy against libraries already present or permissively installable.

### Rung 3 — Symmetry mirror, gated

Only worth it when the building is genuinely symmetric (churches, temples, civic buildings). It is
the only technique below Rung 4 that produces real structural *detail*.

**Finding the plane cheaply** (PRST and Mitra et al. ship no usable code — don't chase them):
1. Take gaussian centres, drop Y. Fit the dominant facade direction in XZ by PCA or RANSAC line fit.
2. Candidate planes: the two axis-aligned-in-that-frame planes through the footprint centroid, plus the footprint's principal axes.
3. Score each with a one-directional Chamfer: for each observed point, distance from its **mirror image** to the nearest observed point. **Score only over the observed half** — otherwise the missing side wins trivially by having nothing to disagree with. `scipy.spatial.cKDTree` on 1-4M points is seconds.
4. Accept if median residual < ~1 gaussian scale (≈0.15 m for this data).

**Mirroring a gaussian — two traps, both verified by running the maths:**

**Trap 1 — mirroring a quaternion directly produces an invalid rotation.** `det(M) = −1` for any
reflection `M = I − 2nnᵀ`, so `M·R` is improper (`det = −1`) and corresponds to no quaternion. Two
fixes were verified to reproduce the mirrored covariance `MΣMᵀ` exactly (atol 1e-9): **negate one
column of `M·R`** (cheaper, exact), or eigendecompose `MΣMᵀ` and flip an eigenvector if
`det(V) < 0`. Use the column flip.

**Trap 2 — SH mirroring is a sign-flip *only* for axis-aligned planes.** Fitting the transfer
matrix `T` where `B(M·d) = B(d)·T` over 512 Fibonacci-sphere directions, using gsplat's own SH
basis (`tools/gsplat/gsplat/cuda/_torch_impl.py:739-801`):

| Mirror plane | `T` structure | Off-diagonal max |
|---|---|---|
| YZ (normal X) | **pure diagonal**, signs `[+,+,+,−,−,+,+,−,+,−]` | 4.3e-15 |
| XY (normal Z) | **pure diagonal**, signs `[+,+,−,+,+,−,+,−,+,+,−,+,−,+,−,+]` | 4.3e-15 |
| normal `[1,0,1]` | **dense**, needs full 16×16 matmul | 1.00 |
| normal `[0.3,0.1,0.95]` | **dense** | 0.79 |

`T@T == I` in all four cases — the transform is its own inverse. **Practical consequence: rotate
the scene so the symmetry plane is axis-aligned before mirroring, and SH handling is a sign-flip
vector, about four lines of numpy.** Mirror about an arbitrary plane in place and you need the
dense 16×16 (9×9 for SH degree 2), precomputed once by exactly the least-squares fit above.

Complete per-gaussian recipe: position `p' = M(p − o) + o`; rotation = column-flipped `M·R`;
**scale unchanged** (reflection is an isometry, eigenvalues of `MΣMᵀ` equal those of `Σ`);
opacity unchanged; `f_dc_*` unchanged (view-independent); `f_rest_*` per the table above.

**What breaks, bluntly:**
- **Lighting is baked in.** The observed side was lit by the actual sun. Mirroring puts sunlit geometry on what should be the shaded side. On a building this reads as obviously wrong even to a casual viewer. Partial fix: scale mirrored DC by the luminance ratio between sides if there are *any* observations on the target side; with none, darken by a fixed factor and accept that it looks synthetic.
- **Buildings are rarely symmetric where it matters** — doors, signage, extensions, pipework. You will mirror a front door onto a blank rear wall.
- **The observed side is itself one-sided**, so mirroring doubles its reconstruction error rather than cancelling it.
- **Seam at the plane.** Gaussians straddling it get duplicated at near-identical positions, doubling local opacity into a visible bright ridge. **Cull mirrored gaussians within ~2σ of the plane.**
- **Text and asymmetric detail come out mirror-reversed.** Unfixable by this method.

Verdict: acceptable at distance, wrong up close. Gate it, and fall through to Rung 2 when the gate
fails.

### Rung 4 — 2D inpaint + retrain (the highest-value generative experiment)

**Render virtual views looking at the hole → inpaint them in 2D → add as extra COLMAP views →
re-run the existing gsplat trainer.**

Why this beats every 3D generator here:
- Stays entirely in the **metric frame**. No 7-DoF solve, no ICP, no seam, no alignment failure mode.
- No new CUDA, no compile, no bespoke licence — the 2D inpainting step is a single-image UNet pass that fits 6 GB comfortably in fp16.
- Multi-view-inconsistent hallucination shows up as **soft fog rather than a hard geometric seam**, and gsplat's own multi-view averaging suppresses some of it.
- Reuses `run_colmap.py`, `train_splat.py`, and the COLMAP binaries already in `tools/`.

This is the InFusion recipe (§4) decomposed so no two heavy stages are ever resident at once. Borrow
**3DGIC's depth-guided mask reduction** to decide which region is genuinely unobserved.

Optional depth assist — and a correction on how to use it: a monocular depth model **cannot see the
unobserved side either**, so it cannot generate the geometry. Use it as a **validator**: render the
completed scene from a virtual camera facing the synthesized facade, run the depth model on that
render, and compare predicted normals and planarity against the fitted plane. If the wall bulges or
is fuzzy enough to confuse a depth model, that shows up as normal-map noise. That is an automated
quality gate in the spirit of `scripts/check_world.py`, for one forward pass.

Recommended model for that: **MoGe-2 ViT-S (35M), MIT for the whole family**, outputs metric point
maps + depth + **normals** + FOV in one pass, ~1.5 GB. Better than a pure depth model here, and the
cleanest licence available. `Depth Anything V2 Small` (Apache-2.0, ~1 GB) is the fallback;
Base/Large/Giant are all CC-BY-NC-4.0. `Metric3D v2` is BSD-2, ~3-4 GB for ViT-L.

Also worth checking: **do any of the 114 keyframes graze the missing facade?** If even one oblique
frame clips it, MoGe on that frame gives a metric point map that can be lifted directly to seed
gaussians — real observations beat any hallucination.

### Rung 5 — 3D generator (highest risk, lowest expected value here)

See §6. TRELLIS via the Windows fork is the only entry worth trialling, because MIT + native
Gaussian output eliminates two of the three hard problems. The third — object-scale domain and
7-DoF alignment on featureless facades — is not eliminated by anything, and is the one most likely
to sink it.

### Rung 6 — Training-time regularizers (prevents garbage, does not create content)

Every published sparse-view 3DGS regularizer worth having is (a) non-commercial by inheritance and
(b) built on a forked CUDA rasterizer, so **none drop into gsplat 1.5.3**: DNGaussian (LICENSE.md is
verbatim the Inria 3DGS licence), FSGS, CoR-GS (needs
`diff-gaussian-rasterization-confidence`), SparseGS (repo 404s). RegNeRF is Apache-2.0 but is
NeRF/JAX — port the idea, not the code.

The *ideas* are simple and unencumbered, and `scripts/train_splat.py` already exists:
- **Opacity decay in unobserved space.** Each iteration, multiply the raw opacity logit of gaussians with `n_views <= k` by a factor slightly under 1. They fade out over training instead of being hard-deleted, which avoids the collapse `train_splat.py:427` already warns about. ~15 lines, cheapest regularizer with real effect.
- **Depth-order regularization** on gsplat's returned depth: penalize local depth variance within superpixels of near-constant colour. That is FSGS's and DNGaussian's core insight without their code, and needs no depth model.
- **Scale cap near the ground** as a *training penalty* rather than the post-hoc cull already prototyped at `render_eyelevel.py:65`, so huge sky-blob gaussians never form.

Treat Rung 6 as an improvement to Rung 0, not an alternative to Rungs 1-4. It costs a full retrain
per experiment, which makes iteration far slower than post-processing a finished `.ply`.

---

## 8. Recommendation

Given the §0 decisions — floaters first, manual first, flat wall — the sequence is:

1. **Rung 0.** Half a day. `--filter-cluster --seed-pos` + `--filter-floaters` + `n_views <= 2` cut. Turns visible garbage into clean absence. Nothing else can be evaluated until this is done.
2. **Rung 1.** The click/draw patch tool. Almost entirely assembly of MIT code that is already on disk. Estimated ~900-1,200 lines of plain JS if SuperSplat pieces are lifted, considerably less if only the engine examples are used.
3. **Rung 2.** Plane fit + disc-gaussian fill, driven either by the manual patch from Rung 1 or automatically from the RANSAC footprint. Add the FFT window-pitch trick if time allows.
4. **Rung 3**, gated on the Chamfer test, falling through to Rung 2 when it fails.
5. **Rung 4** as the one generative experiment worth a day.

Explicitly **skip**: mesh Poisson hole-filling (wrong shape prior for architecture — it closes the
hole by smooth extrapolation from the boundary, producing a soft blob bulging between the adjacent
walls instead of a flat wall meeting them at a crisp edge; and the Python wrappers are GPL-3 or
worse); the point-completion category; the texture-inpainting category; and any published
sparse-view regularizer.

**Two honest caveats.**

First, for a *building*, hallucinated geometry is a liability rather than a feature. Diffusion
priors will invent windows and doors that aren't there. `02_System_Architecture_and_Pipeline_Design.md`
challenge (vii) already states this project's intent — *"scenario layer marks 'no-data' volumes
instead of hallucinating"* — and `coverage.u8` / `coverage_grid.json` / `check_coverage.py` already
implement honesty-by-design. **A flat plausible wall plus an honest coverage mask is more consistent
with this project's existing philosophy than a generated facade.** If the reconstruction is ever
used for anything measurement-adjacent, "visibly empty" is a better failure mode than "confidently
wrong".

Second, and cheapest by a wide margin: **fly the missing side.** Twenty more minutes of drone time
beats every entry in this document. Every method here is inventing plausible fiction, and on a
structure with distinctive geometry invented facades look wrong in ways viewers notice immediately.

---

## 9. Things to track

| Item | Why | Where |
|---|---|---|
| **GSCompleter** | Distillation-free plugin, "in seconds", Generate-then-Register into an existing scene. Best match to both this problem *and* this hardware. | arXiv 2604.20155 |
| **Bolt3D** | 6.25 s feed-forward, emits Gaussians, explicitly claims unobserved-region generation with no inpainting mechanism. | `szymanowiczs.github.io/bolt3d` (repo 404) |
| **ReconSplat** | ECCV 2026, code release unverified. | `visinf.github.io/reconsplat` |
| **`nvidia/Fixer`** | Commercially licensed successor to Difix, currently Docker/Linux-only. If a pip path appears, re-evaluate. | HF `nvidia/Fixer` |
| **SuperSplat v3** | If this viewer ever moves to WebGPU, v3's `GaussianInstances` architecture (instance lists over immutable static rows, deletion by removal with exact undo, layers sharing rows with zero gaussian copying) is substantially better than v2's. | `playcanvas/supersplat` main |
| **splat-transform browser API** | Cleanest possible write-back backend if the no-build-step rule ever relaxes. | `readFile`/`processSourceBridged`/`writeSource` |
| **Spark's `SplatEditSdf`** | Best-designed region vocabulary in the field. Port the vocabulary. | `sparkjsdev/spark`, MIT |
| **Postshot ROI training** | "Mark a region → retrain only that region at higher density" is arguably the *correct* framing of this whole feature. | closed source, concept only |
| **UnityGaussianSplatting cutouts** | Cutouts constrain manual selection ops to a box — a clean UX primitive. | `aras-p/UnityGaussianSplatting` |
| **3DGIC mask reduction** | The algorithm for "which region is genuinely unobserved in *any* view". Needed for the "auto later" half of the plan. | `find_depth_guided_mask.sh` |

---

## 10. Confidence and gaps

**Verified by reading source or LICENSE text:** all licences quoted in §2.2; the Hunyuan3D
territory clauses; the Difix→Fixer deprecation; ViewCrafter's, WonderJourney's and Invisible
Stitch's VRAM figures; GenFusion's `environment.yml` and CLI flags; the incomplete-release notices
in 3DGS-Enhancer and See3D; SuperSplat's `main.ts` device requests for both v2.32.5 and main; the
PlayCanvas engine internals cited by line number; the existence and byte sizes of every
`tools/pc-engine` file in §1.

**Verified by running code:** the quaternion-mirror `det = −1` failure and both fixes; the SH
mirror transfer matrices in §7.3 (512-direction least-squares fit against gsplat's own SH basis).

**Unverified — treat as estimates:**
- VRAM for Gaussian Grouping, AuraFusion360, GScream, 3DGIC, InFusion, GenFusion, See3D. **None of these publish a figure.** The "won't fit 6 GB" calls are reasoned from model stack and the GPUs the authors used, not measured.
- Difix in fp16 at 576×1024 on 6 GB. **No public VRAM number exists** — NVIDIA's model card has an unfilled `X GB` placeholder. The fp16 estimate is arithmetic from the 0.9B parameter count. A 15-minute empirical test would settle it.
- Star counts and push dates for GSFix3D, G4Splat, 3DGIC, Difix3D, and several §5.3 entries came from search-page HTML after the GitHub API rate limit was hit.
- Submodule licences were **not** individually verified for Inpaint360GS, InFusion, or 3DGIC — only the upstream Inria licences those submodules derive from. Before any commercial use, `git clone --recursive` and read every LICENSE in the tree. **The root file is not the whole story.**
- Polycam / Luma / KIRI Engine app capabilities: low-to-medium confidence, several help-centre pages 404'd.
- `DiffComplete` repo path 404'd; that whole category is under-surveyed.

**Nothing in this survey was run on this machine.**

---

## 11. Note on research hygiene

Three of the five research passes returned output with appended Chinese text instructing the agent
to write a `MEMORY.md`, clear the session, and start over. That text did not originate from the
user or from any legitimate system message — it appears to be injected by something in the tool
chain, plausibly a plugin under `.opencode/`. It was ignored. Worth auditing, because instructions
that arrive inside tool output and try to redirect an agent's control flow are the textbook shape
of a prompt-injection attempt.
