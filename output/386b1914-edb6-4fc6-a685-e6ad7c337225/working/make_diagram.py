"""Methodology diagram for the WALKABLE project report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

fig, ax = plt.subplots(figsize=(13.4, 6.4), dpi=200)
ax.set_xlim(0, 134)
ax.set_ylim(0, 64)
ax.axis("off")
fig.patch.set_facecolor("white")

# ---------- palette ----------
INK      = "#1f2937"   # dark slate text
SUB      = "#4b5563"   # secondary text
PHASE_FC = {"cap": "#eef4ff", "rec": "#ecfdf5", "val": "#fff7ed"}
PHASE_EC = {"cap": "#3b82f6", "rec": "#10b981", "val": "#f59e0b"}
BOX_FC   = {"cap": "#dbeafe", "rec": "#d1fae5", "val": "#ffedd5"}
BOX_EC   = {"cap": "#2563eb", "rec": "#059669", "val": "#d97706"}

def phase(x, y, w, h, key, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.6,rounding_size=1.4",
        fc=PHASE_FC[key], ec=PHASE_EC[key], lw=1.6, alpha=0.55, zorder=1))
    ax.text(x + 1.6, y + h - 2.6, label, fontsize=10.5, fontweight="bold",
            color=PHASE_EC[key], ha="left", va="center", zorder=3)

def box(x, y, w, h, key, title, sub=None, tsize=9.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.35,rounding_size=0.9",
        fc=BOX_FC[key], ec=BOX_EC[key], lw=1.4, zorder=4))
    if sub:
        ax.text(x + w/2, y + h*0.63, title, fontsize=tsize, fontweight="bold",
                color=INK, ha="center", va="center", zorder=5)
        ax.text(x + w/2, y + h*0.28, sub, fontsize=7.2, color=SUB,
                ha="center", va="center", zorder=5)
    else:
        ax.text(x + w/2, y + h/2, title, fontsize=tsize, fontweight="bold",
                color=INK, ha="center", va="center", zorder=5)

def arrow(x1, y1, x2, y2, color="#374151", lw=1.8, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=13, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=6))

# ---------- title ----------
ax.text(67, 61.5, "WALKABLE Pipeline — Single Video to a Metric, Walkable 3D World",
        fontsize=13.5, fontweight="bold", color=INK, ha="center", va="center")
ax.text(67, 58.4, "Structure-from-Motion  +  3D Gaussian Splatting  +  Classical Geometry  +  Browser Physics",
        fontsize=8.6, color=SUB, ha="center", va="center")

# ---------- phase bands ----------
phase(1.5, 29, 40, 27, "cap", "PHASE 1 · CAPTURE & PRE-PROCESSING")
phase(43.5, 29, 44, 27, "rec", "PHASE 2 · 3D RECONSTRUCTION")
phase(89.5, 29, 43, 27, "val", "PHASE 3 · WORLD BUILDING & VALIDATION")

# Phase 1 boxes
box(4, 43.5, 16, 8.5, "cap", "Input video", "drone / handheld .mp4\n(+ AR pose log .jsonl)")
box(22.5, 43.5, 16, 8.5, "cap", "Keyframe selection", "sharpness-ranked,\nnear-duplicate rejection")
box(4, 32.5, 16, 8.5, "cap", "Pose priors", "ARCore / ARKit tracks\n(optional)")
box(22.5, 32.5, 16, 8.5, "cap", "Coverage console", "WebXR capture guidance\nprevents bad footage")

# Phase 2 boxes
box(46, 43.5, 18, 8.5, "rec", "SfM — COLMAP", "GPU SIFT → matcher →\nmapper → camera poses")
box(66.5, 43.5, 18, 8.5, "rec", "Splat training — gsplat", "3D Gaussian Splatting,\nSH-3, SSIM+L1, ≤6 GB VRAM")
box(46, 32.5, 18, 8.5, "rec", "Metric frame", "gravity from gimbal +\nspeed/height anchor")
box(66.5, 32.5, 18, 8.5, "rec", "Scene clean-up", "sky/canopy culling,\nre-orient & rescale")

# Phase 3 boxes
box(92, 43.5, 18, 8.5, "val", "Geometry extraction", "ground heightfield +\nvoxel collider mesh (GLB)")
box(112, 43.5, 18, 8.5, "val", "Quality gate", "11 falsifiable assertions;\nfailure stops the build")
box(92, 32.5, 18, 8.5, "val", "Walkable world", "browser: splats + ammo.js\nphysics + navmesh bots")
box(112, 32.5, 18, 8.5, "val", "Evaluation", "blind A/B visual review +\nscripted walk test (m walked)")

# ---------- arrows phase 1 ----------
arrow(20, 47.7, 22.5, 47.7)            # video -> keyframes
arrow(30.5, 43.5, 30.5, 41)            # keyframes -> coverage (loop)
arrow(22.5, 36.7, 20, 36.7)            # coverage -> priors feed capture
arrow(38.5, 47.7, 46, 47.7, lw=2.2)    # keyframes -> COLMAP
arrow(38.5, 36.7, 46, 45.2, rad=-0.25, color="#2563eb", lw=1.5)  # priors -> COLMAP

# ---------- arrows phase 2 ----------
arrow(64, 47.7, 66.5, 47.7)            # COLMAP -> train
arrow(75.5, 43.5, 75.5, 41)            # train -> cleanup
arrow(66.5, 36.7, 64, 36.7)            # cleanup <- frame (order)
arrow(55, 43.5, 55, 41)                # COLMAP poses -> metric frame
arrow(84.5, 45, 92, 47.2, rad=-0.15, lw=2.2)   # phase2 -> phase3

# ---------- arrows phase 3 ----------
arrow(110, 47.7, 112, 47.7)            # geometry -> gate
arrow(121, 43.5, 121, 41)              # gate -> evaluation
arrow(112, 36.7, 110, 36.7)            # eval <- world
arrow(101, 43.5, 101, 41)              # geometry -> world
arrow(110, 33.5, 112, 33.5, style="<|-|>", color="#d97706", lw=1.3)  # world <-> eval

# ---------- feedback / iteration note ----------
ax.text(67, 27.2, "Every stage is a resumable CLI module writing typed artefacts to disk — nothing ships that a machine has not measured.",
        fontsize=8.0, style="italic", color=SUB, ha="center", va="center")

# ---------- bottom application strip ----------
ax.add_patch(FancyBboxPatch((1.5, 2.5), 131, 21.5,
    boxstyle="round,pad=0.6,rounding_size=1.4",
    fc="#f8fafc", ec="#94a3b8", lw=1.3, zorder=1))
ax.text(4, 20.6, "DOWNSTREAM APPLICATION DOMAINS", fontsize=9.5,
        fontweight="bold", color="#475569", ha="left", va="center")

apps = [
    ("Border & strategic\narea mapping", "#dbeafe", "#2563eb"),
    ("Disaster damage\nassessment", "#fee2e2", "#dc2626"),
    ("Urban planning &\nsmart cities", "#dcfce7", "#16a34a"),
    ("Infrastructure\ninspection", "#fef9c3", "#a16207"),
    ("Construction progress\nmonitoring", "#fae8ff", "#a21caf"),
    ("Archaeological\ndocumentation", "#ffedd5", "#c2410c"),
    ("Digital twin\ngeneration", "#e0e7ff", "#4f46e5"),
    ("Reconnaissance &\nmission planning", "#e2e8f0", "#334155"),
]
bw, bh, gap = 15.0, 10.5, 1.35
x0 = 4.5
for i, (label, fc, ec) in enumerate(apps):
    x = x0 + i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, 6.2), bw, bh,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        fc=fc, ec=ec, lw=1.2, zorder=4))
    ax.text(x + bw/2, 6.2 + bh/2, label, fontsize=7.4, fontweight="bold",
            color=INK, ha="center", va="center", zorder=5)
    arrow(67, 30, x + bw/2, 17.4, color="#cbd5e1", lw=0.9, rad=0.0)

plt.savefig(r"C:\Users\krisg\Desktop\Drone to 3d mesh\output\386b1914-edb6-4fc6-a685-e6ad7c337225\stage2\images\methodology_diagram.png",
            bbox_inches="tight", facecolor="white")
print("saved")
