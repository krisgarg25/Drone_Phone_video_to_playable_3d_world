"""
SIH26158 ONEPASS — offline deck builder (python-pptx).

Idempotent: always rebuilds from the pristine template.
Run:  python build_deck.py
"""
import shutil
import os
import pptx
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

BASE = r'c:\Users\krisg\Desktop\Drone to 3d mesh'
TEMPLATE = os.path.join(BASE, 'ppt', 'SIH2026-IDEA-Presentation-Format.pptx')
OUT = os.path.join(BASE, 'ppt', 'SIH26158_ONEPASS_Idea_Deck.pptx')

# --- palette (BRAND matches the template's own #0070C0 footer) ---
BRAND     = RGBColor(0x00, 0x70, 0xC0)
BRAND_DK  = RGBColor(0x00, 0x4C, 0x80)
NAVY      = RGBColor(0x1A, 0x3B, 0x6B)
PALE      = RGBColor(0xE8, 0xF2, 0xFA)
PALE2     = RGBColor(0xF4, 0xF8, 0xFC)
CRIMSON   = RGBColor(0xC8, 0x10, 0x2E)
INK       = RGBColor(0x1A, 0x1A, 0x1A)
GREY      = RGBColor(0x59, 0x59, 0x59)
RULE      = RGBColor(0xD0, 0xD7, 0xDE)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)

F_TITLE = 'Times New Roman'
F_BODY  = 'Arial'

# content box (below title placeholder, above footer bar at 6.95)
L = 0.36
R = 12.97
TOP = 1.30
BOT = 6.82


def solid(shape, rgb):
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb
    shape.line.fill.background()


def outline(shape, rgb, pt=0.75):
    shape.line.color.rgb = rgb
    shape.line.width = Pt(pt)


def rect(slide, l, t, w, h, fill=None, line=None, lw=0.75,
         shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(l), Inches(t), Inches(w), Inches(h))
    s.shadow.inherit = False
    if fill is None:
        s.fill.background()
    else:
        solid(s, fill)
    if line is None:
        s.line.fill.background()
    else:
        outline(s, line, lw)
    if s.has_text_frame:
        s.text_frame.clear()
        s.text_frame.word_wrap = True
    return s


def tb(slide, l, t, w, h, text, size=11, bold=False, color=INK,
       font=F_BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
       space_after=0, line_spacing=None, italic=False):
    """Add a textbox. `text` may be a str or list of (str, dict-overrides)."""
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor

    lines = text if isinstance(text, list) else [text]
    for i, ln in enumerate(lines):
        over = {}
        if isinstance(ln, tuple):
            ln, over = ln
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = over.get('align', align)
        p.space_after = Pt(over.get('space_after', space_after))
        if line_spacing:
            p.line_spacing = line_spacing
        r = p.add_run()
        r.text = ln
        f = r.font
        f.size = Pt(over.get('size', size))
        f.bold = over.get('bold', bold)
        f.italic = over.get('italic', italic)
        f.name = over.get('font', font)
        f.color.rgb = over.get('color', color)
    return box


def section_label(slide, l, t, w, text):
    """Pointer label — text kept VERBATIM from the SIH template."""
    tb(slide, l, t, w, 0.26, text, size=10.5, bold=True,
       color=BRAND_DK, font=F_BODY)
    return rect(slide, l, t + 0.27, w, 0.018, fill=BRAND)


def stat(slide, l, t, w, h, value, label, vsize=21):
    """Stat callout: big number over a short caption, on a pale panel."""
    p = rect(slide, l, t, w, h, fill=PALE)
    tb(slide, l + 0.10, t + 0.11, w - 0.20, 0.38, value, size=vsize,
       bold=True, color=BRAND_DK, font=F_TITLE)
    tb(slide, l + 0.10, t + 0.52, w - 0.20, h - 0.60, label, size=8.5,
       color=GREY, line_spacing=0.95)
    return p


def numbered(slide, l, t, w, items, gap=0.58, num_size=11, head_size=10.5,
             body_size=9.5, body_w_off=0.30):
    """items = [(head, body), ...] rendered as 1. / 2. / 3."""
    for i, (head, body) in enumerate(items):
        y = t + i * gap
        tb(slide, l, y + 0.01, 0.30, 0.24, '%d.' % (i + 1), size=num_size,
           bold=True, color=BRAND, font=F_TITLE)
        tb(slide, l + 0.30, y, w - 0.30, 0.22, head, size=head_size,
           bold=True, color=INK)
        tb(slide, l + 0.30, y + 0.21, w - 0.30, gap - 0.24, body,
           size=body_size, color=GREY, line_spacing=0.95)


def delete_slide(prs, index):
    lst = prs.slides._sldIdLst
    for s in list(lst):
        if s.get('id') == prs.slides[index].slide_id:
            lst.remove(s)
            return
    lst.remove(list(lst)[index])


def find(slide, name):
    for sh in slide.shapes:
        if sh.name == name:
            return sh
    return None


def clear_pointer_box(slide, name='TextBox 8'):
    """Remove the template's grouped pointer textbox; each pointer is
    re-added verbatim as its own section label."""
    sh = find(slide, name)
    if sh is not None:
        sh._element.getparent().remove(sh._element)


# ───────────────────────────── SLIDE 1 — TITLE PAGE ─────────────────────────
def build_slide1(slide):
    old = find(slide, 'TextBox 9')
    if old is not None:
        old._element.getparent().remove(old._element)

    # Idea title
    tb(slide, L, 2.30, 6.55, 0.70,
       'ONEPASS', size=34, bold=True, color=BRAND_DK, font=F_TITLE)
    tb(slide, L, 3.02, 6.55, 0.52,
       'Georeferenced, metrically accurate 3D from a single drone pass',
       size=13, bold=True, color=INK, line_spacing=1.0)
    rect(slide, L, 3.62, 2.10, 0.045, fill=CRIMSON)

    # SIH registration fields — filled where known, left blank where not
    fields = [
        ('Problem Statement ID', 'SIH26158'),
        ('Problem Statement Title',
         'AI-enabled single-pass drone video to a georeferenced, '
         'metrically accurate 3D model'),
        ('Theme', ''),
        ('PS Category', 'Software'),
        ('Team ID', ''),
        ('Team Name (Registered on portal)', ''),
    ]
    y = 3.76
    for label, value in fields:
        tb(slide, L, y, 6.55, 0.20, label, size=9.5, bold=True, color=BRAND_DK)
        if value:
            tb(slide, L, y + 0.20, 6.55, 0.34, value, size=12, bold=True,
               color=INK, line_spacing=0.95)
            # long values wrap to two lines — give them the room
            y += 0.74 if len(value) > 46 else 0.54
        else:
            rect(slide, L, y + 0.38, 5.60, 0.012, fill=RULE)
            tb(slide, L + 5.72, y + 0.22, 0.80, 0.20, 'fill in', size=8,
               italic=True, color=CRIMSON)
            y += 0.50
    return slide


# ───────────────────────────── SLIDE 2 — IDEA TITLE ─────────────────────────
P_SOLUTION = 'Proposed Solution (Describe your Idea/Solution/Prototype)'
P_DETAIL   = 'Detailed explanation of the proposed solution'
P_ADDRESS  = ' How it addresses the problem'
P_INNOV    = 'Innovation and uniqueness of the solution '


def build_slide2(slide):
    clear_pointer_box(slide)

    q1l, q1t, q1w = L, TOP, 6.05
    q2l, q2t, q2w = 6.75, TOP, 6.22
    q3l, q3t = L, 4.14
    q4l, q4t = 6.75, 4.14

    # column + row dividers
    rect(slide, 6.60, TOP, 0.012, 2.60, fill=RULE)
    rect(slide, L, 4.02, 12.61, 0.012, fill=RULE)

    # Q1 — Proposed Solution + stats
    section_label(slide, q1l, q1t, q1w, P_SOLUTION)
    tb(slide, q1l, q1t + 0.36, q1w, 0.78,
       'ONEPASS converts one UAV video plus its GPS log into Earth-referenced, '
       'measurable 3D. Telemetry drives the solve instead of being stamped on '
       'afterwards.', size=11, line_spacing=1.02)

    tiles = [('68', 'frames kept of 288'),
             ('1.57M', 'fused points, one real run'),
             ('287/287', 'cameras matched'),
             ('5 of 6', 'official formats delivered')]
    tw, gap = 1.44, 0.09
    for i, (v, lab) in enumerate(tiles):
        stat(slide, q1l + i * (tw + gap), 2.52, tw, 0.86, v, lab, vsize=17)

    # Q2 — Detailed explanation
    section_label(slide, q2l, q2t, q2w, P_DETAIL)
    numbered(slide, q2l, q2t + 0.36, q2w, [
        ('CAPTURE INTELLIGENCE',
         'Reads video and GPX / DJI .srt telemetry. Baseline-aware keyframe '
         'planning spends the frame budget where parallax is won.'),
        ('METRIC RECONSTRUCTION',
         'GPS enters bundle adjustment as per-camera priors with per-axis '
         'uncertainty; robust Sim(3) into local ENU, then dense stereo.'),
        ('EVIDENCE GATING',
         'Blur, compression and motion measured as separate gates before '
         'matching. 4.85% of pixels vetoed; reprojection error better by 7.8%.'),
        ('GEOREFERENCED DELIVERY',
         'CRS derived from scene origin; mesh, cloud, DSM, ortho and a '
         'walkable browser scene all from the same pass.'),
    ], gap=0.55, head_size=9.5, body_size=8.5)

    # Q3 — How it addresses the problem
    section_label(slide, q3l, q3t, q1w, P_ADDRESS)
    rows = [
        ('Terrain and structures', 'Poisson mesh + fused dense cloud', 'Delivered'),
        ('Facades and rooftops', 'Oblique/nadir plan, baseline-ranked frames', 'Mechanism'),
        ('Roads and infrastructure', 'Metric DSM, ortho, position table', 'Delivered'),
        ('Vegetation and obstacles', 'Static kept, motion masked', 'Measured'),
        ('Textured mesh / point cloud', 'PLY + LAS now, UV atlas open', 'Partial'),
    ]
    y = q3t + 0.42
    for target, ret, status in rows:
        tb(slide, q3l, y + 0.02, 1.95, 0.36, target, size=8.5, bold=True,
           color=INK, line_spacing=0.92)
        tb(slide, q3l + 2.00, y + 0.02, 2.75, 0.36, ret, size=8.5,
           color=GREY, line_spacing=0.92)
        col = BRAND_DK if status in ('Delivered', 'Measured') else CRIMSON
        tb(slide, q3l + 4.80, y + 0.02, 1.20, 0.24, status, size=8.5,
           bold=True, color=col, align=PP_ALIGN.RIGHT)
        y += 0.44
        rect(slide, q3l, y - 0.05, q1w, 0.008, fill=RULE)

    # Q4 — Innovation and uniqueness
    section_label(slide, q4l, q4t, q2w, P_INNOV)
    numbered(slide, q4l, q4t + 0.40, q2w, [
        ('GPS AS PRIORS',
         'Enters bundle adjustment with per-axis uncertainty — never an EPSG '
         'stamp pasted onto the output afterwards.'),
        ('BASELINE-AWARE SELECTION',
         'The one lever that improves geometry and speed at the same time.'),
        ('BLUR AND COMPRESSION SPLIT',
         'Two failure modes measured and gated separately, not merged into a '
         'single score.'),
        ('GEOMETRY THAT CANNOT LIE',
         'Measured, weak and unobserved surfaces stay separate. No inpainting '
         'entry point exists in the code.'),
    ], gap=0.55, head_size=9.5, body_size=8.5)
    return slide


P_TECH  = ('Technologies to be used (e.g. programming languages, '
           'frameworks, hardware)')
P_METH  = ('Methodology and process for implementation '
           '(Flow Charts/Images/ working prototype)')
P_FEAS  = 'Analysis of the feasibility of the idea'
P_RISK  = 'Potential challenges and risks'
P_STRAT = 'Strategies for overcoming these challenges'


# ─────────────────────────── SLIDE 3 — TECHNICAL APPROACH ───────────────────
def build_slide3(slide):
    clear_pointer_box(slide)

    # ---- Methodology: 8-stage flow band ----
    section_label(slide, L, TOP, 12.61, P_METH)
    stages = [
        ('INGEST', 'video + GPS'), ('PREPARE', 'gates + plan'),
        ('FRAMES', '68 of 288'),   ('PRIORS', 'GPS to poses'),
        ('MAP', '239.7 s'),        ('DENSE', '1.57M pts'),
        ('MESH', 'Poisson'),       ('DELIVER', '5 of 6 fmts'),
    ]
    n = len(stages)
    sw = (12.61 - (n - 1) * 0.10) / n
    for i, (name, sub) in enumerate(stages):
        x = L + i * (sw + 0.10)
        col = BRAND_DK if i % 2 == 0 else BRAND
        box = rect(slide, x, TOP + 0.36, sw, 0.62, fill=col,
                   shape=MSO_SHAPE.CHEVRON)
        tf = box.text_frame
        tf.margin_left = tf.margin_right = Inches(0.03)
        tf.margin_top = tf.margin_bottom = Emu(0)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = name
        r.font.size = Pt(9.5); r.font.bold = True
        r.font.name = F_BODY; r.font.color.rgb = WHITE
        tb(slide, x, TOP + 1.02, sw, 0.22, sub, size=7.5, color=GREY,
           align=PP_ALIGN.CENTER)

    # ---- Left: Technologies table ----
    section_label(slide, L, 2.68, 6.60, P_TECH)
    rows = [
        ('Language / runtime', 'Python, CUDA, PyTorch, Docker + conda'),
        ('Reconstruction core', 'COLMAP 4.1.1: pose-prior mapper, PatchMatch, Poisson'),
        ('Geometry / alignment', 'Robust Sim(3) + RANSAC, WGS84 to local ENU'),
        ('Frame intelligence', 'ffmpeg decode; Laplacian, Tenengrad, blocking scores'),
        ('Telemetry', 'GPX, DJI .srt, Pilot CSV, contract CSV readers'),
        ('Output products', 'PLY, LAS, GeoTIFF, glTF/GLB, FBX, WGS84 table'),
        ('Interface / QA', 'PlayCanvas viewer, Next.js console, headless-Chrome walk'),
    ]
    y = 3.10
    for k, v in rows:
        tb(slide, L, y, 1.75, 0.30, k, size=8.5, bold=True, color=BRAND_DK,
           line_spacing=0.92)
        tb(slide, L + 1.82, y, 4.78, 0.30, v, size=8.5, color=INK,
           line_spacing=0.92)
        y += 0.31
        rect(slide, L, y - 0.05, 6.60, 0.008, fill=RULE)

    # ---- Right: measured levers chart ----
    cl = 7.30
    section_label(slide, cl, 2.68, 5.67,
                  'Measured speed-up on one RTX 3050 6 GB laptop (\u00d7)')
    bars = [
        ('Baseline-aware frame selection', 4.2, BRAND_DK),
        ('Sequential vs exhaustive match', 3.9, BRAND_DK),
        ('Dense: fast profile', 2.6, BRAND),
        ('Dense: budget profile', 3.9, BRAND),
    ]
    bx, barmax = cl + 2.42, 2.70
    y = 3.14
    for label, val, col in bars:
        tb(slide, cl, y + 0.01, 2.36, 0.24, label, size=8, color=INK)
        rect(slide, bx, y, barmax * (val / 5.0), 0.22, fill=col)
        tb(slide, bx + barmax * (val / 5.0) + 0.06, y + 0.01, 0.62, 0.22,
           '%.1f\u00d7' % val, size=8.5, bold=True, color=BRAND_DK)
        y += 0.34
    tx = bx + barmax * (3.0 / 5.0)
    rect(slide, tx, 3.10, 0.014, 1.34, fill=CRIMSON)
    tb(slide, tx - 0.60, 4.46, 2.10, 0.34,
       '3.0\u00d7 needed to fit 900 s', size=7.5, bold=True, color=CRIMSON,
       line_spacing=0.92)
    tb(slide, cl, 4.82, 5.67, 0.34,
       'First bar is estimated from measured per-image rates; the other three '
       'are measured stage times.', size=7.5, italic=True, color=GREY,
       line_spacing=0.95)

    # ---- Bottom evidence strip ----
    rect(slide, L, 5.34, 12.61, 1.44, fill=PALE2)
    rect(slide, L, 5.34, 0.06, 1.44, fill=BRAND)
    ev = [('4:30', 'demo video'), ('11/11', 'CPU suites green'),
          ('151', 'survey tests'), ('1,405 MiB', 'peak VRAM'),
          ('65.3 m', 'walked, 0 falls'), ('8/8', 'cells at 2 m')]
    ew = 2.02
    for i, (v, lab) in enumerate(ev):
        x = L + 0.22 + i * ew
        tb(slide, x, 5.54, ew - 0.12, 0.34, v, size=15, bold=True,
           color=BRAND_DK, font=F_TITLE)
        tb(slide, x, 5.90, ew - 0.12, 0.32, lab, size=8, color=GREY,
           line_spacing=0.95)
    tb(slide, L + 0.22, 6.30, 12.20, 0.38,
       'Working prototype: repository with continuous integration and a '
       '4 minute 30 second recorded demo.', size=9, bold=True, color=INK)
    return slide


# ─────────────────────── SLIDE 4 — FEASIBILITY AND VIABILITY ────────────────
def build_slide4(slide):
    clear_pointer_box(slide)

    # ---- Left: official desired output vs position ----
    section_label(slide, L, TOP, 6.90, P_FEAS)
    hy = TOP + 0.36
    for dx, txt in ((0, 'Parameter'), (2.10, 'Official target'),
                    (4.25, 'Position today')):
        tb(slide, L + dx, hy, 2.40, 0.22, txt, size=8, bold=True,
           color=BRAND_DK)
    rect(slide, L, hy + 0.22, 6.90, 0.014, fill=BRAND)

    comp = [
        ('Reconstruction', '3D mesh / point cloud', 'Delivered'),
        ('Processing time', '< 15 min per 10-min video', 'Not met'),
        ('Spatial accuracy', '\u2264 1 m', 'Unproven'),
        ('Coverage', 'Entire visible scene', 'Partial'),
        ('Output formats', '6 formats specified', '5 of 6'),
        ('Visualisation', 'Web or desktop viewer', 'Delivered'),
    ]
    y = hy + 0.32
    for param, target, pos in comp:
        ok = pos == 'Delivered'
        tb(slide, L, y, 2.05, 0.32, param, size=8.5, bold=True, color=INK,
           line_spacing=0.92)
        tb(slide, L + 2.10, y, 2.10, 0.32, target, size=8.5, color=GREY,
           line_spacing=0.92)
        tb(slide, L + 4.25, y, 2.65, 0.32, pos, size=8.5, bold=True,
           color=BRAND_DK if ok else CRIMSON, line_spacing=0.92)
        y += 0.36
        rect(slide, L, y - 0.06, 6.90, 0.008, fill=RULE)

    tb(slide, L, y + 0.06, 6.90, 0.50,
       'Against the brief\u2019s eight key challenges: 1 fulfilled with '
       'measurement, 5 mechanism-complete awaiting data, 1 partial by design, '
       '1 not met on this hardware.', size=8, italic=True, color=GREY,
       line_spacing=0.98)

    # ---- Right: official evaluation weights ----
    cl = 7.72
    section_label(slide, cl, TOP, 5.25,
                  'Official evaluation criteria \u2014 weight and evidence')
    weights = [
        ('Accuracy', 30, 'Mechanism complete, unproven on qualifying data'),
        ('Completeness', 20, 'Observability measured, 8 of 8 cells at 2 m'),
        ('Speed', 20, '2.6\u20134.2\u00d7 levers, time gate still unmet'),
        ('Innovation', 15, 'Delivered and distinct from survey-first pipelines'),
        ('Scalability', 10, 'Runs on one 6 GB laptop, peak 1,405 MiB'),
        ('Usability', 5, 'Walkable viewer, 65.3 m, 16 of 16 waypoints'),
    ]
    y = TOP + 0.40
    barx, barmax = cl + 1.55, 1.45
    for name, w, ev in weights:
        tb(slide, cl, y, 1.50, 0.24, name, size=8.5, bold=True, color=INK)
        rect(slide, barx, y + 0.035, barmax * (w / 30.0), 0.17, fill=BRAND)
        tb(slide, barx + barmax + 0.06, y, 0.50, 0.24, str(w), size=8.5,
           bold=True, color=BRAND_DK)
        tb(slide, cl, y + 0.23, 5.25, 0.22, ev, size=7.5, color=GREY)
        y += 0.44

    rect(slide, cl, y + 0.04, 5.25, 0.76, fill=PALE)
    rect(slide, cl, y + 0.04, 0.06, 0.76, fill=CRIMSON)
    tb(slide, cl + 0.20, y + 0.13, 4.92, 0.60,
       'Accuracy, completeness and speed carry 70 of 100 points \u2014 and all '
       'three sit inside the reconstruction core. Usability is worth 5. That '
       'weighting is why the build order went to geometry before the viewer.',
       size=8.5, color=INK, line_spacing=0.96)

    # ---- Bottom band: risks -> strategies ----
    rect(slide, L, 5.02, 12.61, 0.014, fill=RULE)
    section_label(slide, L, 5.14, 6.05, P_RISK)
    section_label(slide, 6.90, 5.14, 6.07, P_STRAT)
    pairs = [
        ('Dense stereo needs ~3\u00d7 more speed at a 270-frame aerial budget',
         'Three measured levers behind one profile flag plus windowed submaps; '
         'one 10-minute flight run twice proves it'),
        ('Consumer GNSS is 1\u20133 m and worse vertically',
         'GPS enters as uncertainty-weighted priors with takeoff-anchored '
         'heights; RTK/PPK kept as a separate declared track'),
        ('Occluded faces cannot be imaged from a single path',
         'The hole is measured and published with its excluded area; '
         'generative inpainting deliberately absent'),
        ('Exports never read by a third-party reader',
         'Add GDAL, PDAL and laspy validation gates to CI before the '
         'qualifying flight'),
    ]
    y = 5.60
    for risk, strat in pairs:
        tb(slide, L, y, 6.05, 0.32, risk, size=8, bold=True, color=INK,
           line_spacing=0.92)
        tb(slide, 6.90, y, 6.07, 0.32, strat, size=8, color=GREY,
           line_spacing=0.92)
        y += 0.33
    return slide


P_IMPACT = 'Potential impact on the target audience'
P_BENEF  = ('Benefits of the solution (social, economic, environmental, etc.)')


# ─────────────────────────── SLIDE 5 — IMPACT AND BENEFITS ──────────────────
def build_slide5(slide):
    clear_pointer_box(slide)

    # ---- Left: target audience ----
    section_label(slide, L, TOP, 7.40, P_IMPACT)
    tb(slide, L, TOP + 0.36, 7.40, 0.44,
       'One engine turns the footage a mission already captures into a '
       'measurable world \u2014 offline, on hardware a team already carries.',
       size=10.5, line_spacing=1.02)

    missions = [
        ('BORDER AND PERIMETER', 'Hostile terrain mapped without a survey crew entering it'),
        ('DISASTER FIRST RESPONSE', 'Damage model before the first team lands'),
        ('FACILITIES AND TELEMETRY', 'Existing sites digitised from footage already flown'),
        ('MISSION REHEARSAL', 'Walkable scene of the objective area before approach'),
    ]
    y = TOP + 0.90
    for head, body in missions:
        rect(slide, L, y + 0.03, 0.06, 0.40, fill=BRAND)
        tb(slide, L + 0.18, y, 2.55, 0.22, head, size=8.5, bold=True,
           color=BRAND_DK)
        tb(slide, L + 0.18, y + 0.21, 7.15, 0.24, body, size=9, color=GREY)
        y += 0.50

    # two hero verticals
    y += 0.06
    heroes = [
        ('Military reconnaissance',
         'AVAILABLE TODAY \u2014 a 65.3 m walkable terrain scene, 16 of 16 '
         'waypoints, 0 falls, with position, ground, obstacle and flyable data.',
         'ONE BUILD AWAY \u2014 enemy indicators (helmet, vehicle, tent, gun) and '
         'full mission-rehearsal data that flying alone cannot supply.'),
        ('Urban planning',
         'AVAILABLE TODAY \u2014 georeferenced buildings, roads, vegetation and '
         'utilities from real scenes; 5 of 6 official formats in one run.',
         'ONE BUILD AWAY \u2014 semantic layers and a text legend.'),
    ]
    for name, now, later in heroes:
        rect(slide, L, y, 7.40, 0.78, fill=PALE2)
        rect(slide, L, y, 0.06, 0.78, fill=BRAND_DK)
        tb(slide, L + 0.18, y + 0.07, 7.10, 0.20, name, size=9.5, bold=True,
           color=INK)
        tb(slide, L + 0.18, y + 0.28, 7.10, 0.24, now, size=8,
           color=BRAND_DK, line_spacing=0.94)
        tb(slide, L + 0.18, y + 0.52, 7.10, 0.24, later, size=8,
           color=CRIMSON, line_spacing=0.94)
        y += 0.86

    tb(slide, L, y + 0.04, 7.40, 0.34,
       'Supporting: disaster response, AI-in-the-loop flight optimisation, '
       'digital twins, historical preservation, forestry and agriculture, '
       'inspection.', size=8, italic=True, color=GREY, line_spacing=0.95)

    # ---- Right: benefits ----
    cl, cw = 7.90, 5.07
    rect(slide, 7.76, TOP, 0.012, 5.40, fill=RULE)
    section_label(slide, cl, TOP, cw, P_BENEF)

    benefits = [
        ('SOCIAL',
         'First damage model in minutes for search and rescue, with no cell '
         'tower and no cloud. The terrain of a landslide or a flood is walked '
         'remotely before anyone walks it for real.'),
        ('ECONOMIC',
         'One flight instead of a survey sortie and a ground-control crew. '
         'Developed on a 6 GB laptop peaking at 1,405 MiB \u2014 no workstation '
         'to buy, no ground station to establish.'),
        ('STRATEGIC AND ENVIRONMENTAL',
         'Runs fully offline for contested or air-gapped sites. Licence-clean '
         'and reproducible from one command. Fewer repeat sorties over the '
         'same ground.'),
    ]
    y = TOP + 0.42
    for head, body in benefits:
        rect(slide, cl, y, cw, 1.32, fill=PALE2)
        rect(slide, cl, y, 0.06, 1.32, fill=BRAND)
        tb(slide, cl + 0.20, y + 0.12, cw - 0.38, 0.22, head, size=9.5,
           bold=True, color=BRAND_DK)
        tb(slide, cl + 0.20, y + 0.38, cw - 0.38, 0.86, body, size=9,
           color=INK, line_spacing=1.0)
        y += 1.44

    rect(slide, cl, y + 0.02, cw, 0.60, fill=PALE)
    tb(slide, cl + 0.18, y + 0.12, cw - 0.34, 0.44,
       'Target audience: field engineers, disaster-response and border-mapping '
       'agencies, survey and construction firms, facility and '
       'autonomous-operations teams.', size=8.5, bold=True, color=BRAND_DK,
       line_spacing=0.96)
    return slide


P_REFS = 'Details / Links of the reference and research work'


# ─────────────────────── SLIDE 6 — RESEARCH AND REFERENCES ──────────────────
def build_slide6(slide):
    clear_pointer_box(slide)

    section_label(slide, L, TOP, 12.61, P_REFS)

    groups = [
        ('RECONSTRUCTION AND MAPPING', [
            ('COLMAP SfM and MVS', 'colmap.github.io'),
            ('OpenDroneMap photogrammetry', 'github.com/OpenDroneMap/ODM'),
            ('Single-pass UAV trajectory planning',
             'github.com/ch1bo/drone-reconstruction'),
        ]),
        ('FEED-FORWARD GEOMETRY AND TRACKING', [
            ('DPVO deep patch visual odometry', 'github.com/princeton-vl/DPVO'),
            ('Pi3 equivariant reconstruction', 'github.com/yyfz/Pi3'),
            ('Depth-Anything-3',
             'github.com/ByteDance-Seed/Depth-Anything-3'),
            ('MapAnything metric 3D mapping',
             'github.com/facebookresearch/map-anything'),
            ('VGGT-Long online reconstruction',
             'github.com/DengKaiCQ/VGGT-Long'),
        ]),
        ('RENDERING, PHYSICS AND EVALUATION', [
            ('gsplat Gaussian splatting stack', 'docs.gsplat.studio'),
            ('MuJoCo physics engine', 'mujoco.org'),
            ('evo trajectory evaluation', 'github.com/MichaelGrupp/evo'),
            ('RGB-D and completion datasets', 'dornhelge.github.io'),
        ]),
        ('THIS SUBMISSION\u2019S EVIDENCE TRAIL', [
            ('Problem statement and desired output', 'SIH26158 PDF, pp. 38\u201340'),
            ('Evaluation and improvement report',
             '09_SIH26158_Evaluation...2026-09-22'),
            ('Evidence map, wiring ledger, findings',
             'docs/readiness/04, 02, 01'),
            ('Prototype and demo',
             'github.com/krisgarg25/Drone_Phone_video...'),
        ]),
    ]

    cw = 3.06
    for gi, (head, items) in enumerate(groups):
        x = L + gi * (cw + 0.11)
        rect(slide, x, TOP + 0.38, cw, 0.30, fill=BRAND_DK)
        tb(slide, x + 0.10, TOP + 0.44, cw - 0.20, 0.20, head, size=7.5,
           bold=True, color=WHITE)
        y = TOP + 0.80
        for title, link in items:
            tb(slide, x + 0.02, y, cw - 0.14, 0.36, title, size=8.5,
               bold=True, color=INK, line_spacing=0.94)
            tb(slide, x + 0.02, y + 0.34, cw - 0.14, 0.34, link, size=7.5,
               color=BRAND, line_spacing=0.94)
            y += 0.74
            rect(slide, x + 0.02, y - 0.10, cw - 0.14, 0.008, fill=RULE)

    # ---- Evidence accountability band ----
    by = 5.60
    rect(slide, L, by, 12.61, 1.22, fill=PALE2)
    rect(slide, L, by, 0.06, 1.22, fill=CRIMSON)
    tb(slide, L + 0.22, by + 0.12, 12.20, 0.24,
       'EVIDENCE ACCOUNTABILITY', size=9.5, bold=True, color=BRAND_DK)
    tb(slide, L + 0.22, by + 0.40, 12.20, 0.74,
       'Measured figures are stage times, frame counts, point counts and '
       'errors read from machine-generated reports. Anything not yet proven '
       'is labelled a target and stated as unproven. The 1.14 m hold-out '
       'figure is reported against a drifting phone inertial reference and is '
       'presented as reference-dependent rather than as a pass. No total '
       'evaluation score is shown, because only the accuracy and completeness '
       'terms could be computed.', size=8.5, color=INK, line_spacing=1.0)
    return slide


# ───────────────────────────────── MAIN ─────────────────────────────────────
def main():
    shutil.copyfile(TEMPLATE, OUT)
    prs = pptx.Presentation(OUT)

    build_slide1(prs.slides[0])
    build_slide2(prs.slides[1])
    build_slide3(prs.slides[2])
    build_slide4(prs.slides[3])
    build_slide5(prs.slides[4])
    build_slide6(prs.slides[5])

    # Slide 7 is the instructions page — the template says it may be deleted.
    delete_slide(prs, 6)

    prs.save(OUT)
    print('WROTE: %s' % OUT)
    return OUT


if __name__ == '__main__':
    main()



