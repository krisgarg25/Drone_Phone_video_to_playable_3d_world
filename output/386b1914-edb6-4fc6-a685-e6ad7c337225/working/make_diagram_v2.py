"""Methodology diagram (Figure 1) for the WALKABLE project report.

Authored at the size it is printed (6.6 in wide, the A4 text column), so the figure's
point sizes are the document's point sizes. Every label is measured after layout and
shrunk until it fits its box, so nothing is clipped.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

U = 0.01  # inches per data unit
XMAX = 660
YMAX = 734
fig, ax = plt.subplots(figsize=(XMAX * U, YMAX * U), dpi=300)
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.axis("off")
ax.set_position([0, 0, 1, 1])
fig.patch.set_facecolor("white")

INK, SUB = "#1f2937", "#4b5563"
PHASE_FC = {"cap": "#eef4ff", "rec": "#ecfdf5", "val": "#fff7ed"}
PHASE_EC = {"cap": "#3b82f6", "rec": "#10b981", "val": "#f59e0b"}
BOX_FC = {"cap": "#dbeafe", "rec": "#d1fae5", "val": "#ffedd5"}
BOX_EC = {"cap": "#2563eb", "rec": "#059669", "val": "#d97706"}

BAND_X, BAND_W, PAD = 14.0, 632.0, 14.0
INNER_X0, INNER_W = BAND_X + PAD, BAND_W - 2 * PAD
COL_GAP = 36.0
BW = (INNER_W - COL_GAP) / 2
BH = 48.0
CL, CR = INNER_X0, INNER_X0 + BW + COL_GAP
renderer = fig.canvas.get_renderer()
too_wide = []


def fit(t, max_units, floor=4.8):
    limit = max_units * U * fig.dpi
    size = t.get_fontproperties().get_size()
    while size > floor and t.get_window_extent(renderer).width > limit:
        size -= 0.2
        t.set_fontsize(size)
    if t.get_window_extent(renderer).width > limit:
        too_wide.append(t.get_text().replace("\n", " / "))


def text(x, y, s, max_units=None, **kw):
    t = ax.text(x, y, s, **kw)
    if max_units:
        fit(t, max_units)
    return t


def band(yb, h, key, label):
    ax.add_patch(FancyBboxPatch((BAND_X, yb), BAND_W, h,
                                boxstyle="round,pad=0,rounding_size=8",
                                fc=PHASE_FC[key], ec=PHASE_EC[key], lw=1.3,
                                alpha=0.6, zorder=1))
    text(BAND_X + PAD, yb + h - 13, label, BAND_W - 2 * PAD,
         fontsize=9.2, fontweight="bold", color=PHASE_EC[key], ha="left", va="center", zorder=3)


def box(x, yb, key, title, sub):
    ax.add_patch(FancyBboxPatch((x, yb), BW, BH,
                                boxstyle="round,pad=0,rounding_size=5",
                                fc=BOX_FC[key], ec=BOX_EC[key], lw=1.1, zorder=4))
    text(x + BW / 2, yb + BH * 0.71, title, BW - 14, fontsize=8.2, fontweight="bold",
         color=INK, ha="center", va="center", zorder=5)
    text(x + BW / 2, yb + BH * 0.29, sub, BW - 14, fontsize=6.9, color=SUB,
         ha="center", va="center", linespacing=1.35, zorder=5)


def arrow(x1, y1, x2, y2, color="#374151", lw=1.5, style="-|>", ms=11):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=ms, lw=lw, color=color, zorder=6))


def relay(y_from, y_to, label):
    arrow(330, y_from, 330, y_to, color="#334155", lw=2.4, ms=15)
    text(344, (y_from + y_to) / 2, label, 270, fontsize=6.9, style="italic",
         color=SUB, ha="left", va="center", zorder=6)


# --------------------------------------------------------------------------- header
text(330, 720, "WALKABLE: Single Video to a Metric, Walkable 3D World", 640,
     fontsize=12.5, fontweight="bold", color=INK, ha="center", va="center")
text(330, 705, "Structure-from-Motion  +  3D Gaussian Splatting  +  Classical Geometry  +  Browser Physics", 640,
     fontsize=7.6, color=SUB, ha="center", va="center")
text(330, 692, "Every stage is a resumable CLI module writing typed artefacts to disk — nothing ships that a machine has not measured.", 640,
     fontsize=6.9, style="italic", color=SUB, ha="center", va="center")

H = 156
R1, R2 = H - 26 - BH, 20.0  # row offsets from band bottom
B1, B2, B3 = 526, 338, 150

# ------------------------------------------------------------------------ Phase 1
band(B1, H, "cap", "PHASE 1 · CAPTURE & PRE-PROCESSING")
box(CL, B1 + R1, "cap", "Input video", "drone / handheld .mp4\n(+ optional AR pose log .jsonl)")
box(CR, B1 + R1, "cap", "Keyframe selection", "OpenCV decode · Laplacian sharpness\nnear-duplicate rejection")
box(CL, B1 + R2, "cap", "Pose priors", "ARCore / ARKit tracks\n(injected as pose priors, optional)")
box(CR, B1 + R2, "cap", "Coverage console", "WebXR capture guidance —\nquality decided before filming")
arrow(CL + BW, B1 + R1 + BH / 2, CR, B1 + R1 + BH / 2)
arrow(CR + BW / 2, B1 + R1, CR + BW / 2, B1 + R2 + BH)
arrow(CR, B1 + R2 + BH / 2, CL + BW, B1 + R2 + BH / 2)
relay(B1, B2 + H, "keyframes + pose priors")

# ------------------------------------------------------------------------ Phase 2
band(B2, H, "rec", "PHASE 2 · 3D RECONSTRUCTION")
box(CL, B2 + R1, "rec", "SfM — COLMAP", "GPU SIFT → matching →\nbundle adjustment → camera poses")
box(CR, B2 + R1, "rec", "Splat training — gsplat", "3D Gaussian Splatting, SH-3, SSIM+L1\n— converges within 6 GB VRAM")
box(CL, B2 + R2, "rec", "Metric frame", "gravity from gimbal attitude\nscale from one speed/height anchor")
box(CR, B2 + R2, "rec", "Scene clean-up", "sky / cloud-sea culling by height\nbimodality, then re-orient & rescale")
arrow(CL + BW, B2 + R1 + BH / 2, CR, B2 + R1 + BH / 2)
arrow(CR + BW / 2, B2 + R1, CR + BW / 2, B2 + R2 + BH)
arrow(CR, B2 + R2 + BH / 2, CL + BW, B2 + R2 + BH / 2)
arrow(CL + BW / 2, B2 + R1, CL + BW / 2, B2 + R2 + BH, color="#059669", lw=1.1)
relay(B2, B3 + H, "poses, splats, metric frame")

# ------------------------------------------------------------------------ Phase 3
band(B3, H, "val", "PHASE 3 · WORLD BUILDING & VALIDATION")
box(CL, B3 + R1, "val", "Geometry extraction", "ground heightfield + voxelised\ncollision shell exported as GLB")
box(CR, B3 + R1, "val", "Quality gate", "11 falsifiable assertions —\na failure stops the build")
box(CL, B3 + R2, "val", "Walkable world", "PlayCanvas + ammo.js/Bullet:\nfirst-person walking, navmesh agents")
box(CR, B3 + R2, "val", "Evaluation", "blinded A/B visual review +\nscripted walk test (metres walked)")
arrow(CL + BW, B3 + R1 + BH / 2, CR, B3 + R1 + BH / 2)
arrow(CR + BW / 2, B3 + R1, CR + BW / 2, B3 + R2 + BH)
arrow(CR, B3 + R2 + BH / 2, CL + BW, B3 + R2 + BH / 2, style="<|-|>", color="#d97706", lw=1.1, ms=9)
arrow(CL + BW / 2, B3 + R1, CL + BW / 2, B3 + R2 + BH, color="#d97706", lw=1.1)

# ------------------------------------------------------------------ domain strip
SB, SH = 14, 108
ax.add_patch(FancyBboxPatch((BAND_X, SB), BAND_W, SH,
                            boxstyle="round,pad=0,rounding_size=8",
                            fc="#f8fafc", ec="#94a3b8", lw=1.1, zorder=1))
text(BAND_X + PAD, SB + SH - 13, "DOWNSTREAM APPLICATION DOMAINS", BAND_W - 2 * PAD,
     fontsize=8.4, fontweight="bold", color="#475569", ha="left", va="center", zorder=3)
arrow(330, B3, 330, SB + SH, color="#64748b", lw=1.6, ms=12)

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
CH, C_GAP = 30.0, 16.0
CW = (INNER_W - 3 * C_GAP) / 4
for i, (label, fc, ec) in enumerate(apps):
    x = INNER_X0 + (i % 4) * (CW + C_GAP)
    y = SB + 48 - (i // 4) * (CH + 8)
    ax.add_patch(FancyBboxPatch((x, y), CW, CH,
                                boxstyle="round,pad=0,rounding_size=4",
                                fc=fc, ec=ec, lw=1.0, zorder=4))
    text(x + CW / 2, y + CH / 2, label, CW - 10, fontsize=6.8, fontweight="bold",
         color=INK, ha="center", va="center", linespacing=1.3, zorder=5)

OUT = r"C:\Users\krisg\Desktop\Drone to 3d mesh\output\386b1914-edb6-4fc6-a685-e6ad7c337225\stage2\images\methodology_diagram.png"
fig.savefig(OUT, dpi=300, facecolor="white")
print("saved", OUT)
print("labels still overflowing:", too_wide or "none")
