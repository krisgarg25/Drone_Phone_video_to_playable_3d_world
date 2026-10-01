"""Animated illustrations for the "not shown here" feature cards.

Each one is a small SVG scene (viewBox 320 x 170) that shows what the feature does, drawn in
the app's own line style, with a looping GSAP sub-timeline. They are diagrams, not results:
no numbers, only the names of the things the feature measures.

    svg, js = illo(kind, prefix, t0, dur)

`svg` goes inside <svg viewBox="0 0 320 170">; `js` runs inside a composition script where
`tl` and `$` exist, and plays from t0 for dur seconds. Every loop has a finite repeat count.
"""
import math

A = "hsl(36 100% 57%)"
INK = "hsl(220 25% 96%)"
DIM = "hsl(218 12% 62%)"
LINE = "hsl(222 9% 32%)"
FAINT = "hsl(222 9% 22%)"
GRN = "hsl(156 62% 52%)"
RED = "hsl(4 82% 62%)"
BLUE = "hsl(205 85% 62%)"
LBL = 'font-family="Geist Mono, monospace" font-size="13"'


def _loop(p, t0, dur, period, body, delay=0.35):
    """A finite repeating sub-timeline `s` starting at t0."""
    rep = max(0, int((dur - 0.2) // (period + delay)))
    return (f"          {{ const s = gsap.timeline({{ repeat: {rep}, repeatDelay: {delay} }});\n"
            f"{body}"
            f"            tl.add(s, {t0:.3f}); }}\n")


def ripple(p, name, x, y, col=A):
    return (f'<circle id="{p}-{name}" cx="{x}" cy="{y}" r="4" fill="none" stroke="{col}" stroke-width="2" opacity="0"/>',
            f's.fromTo($("#{p}-{name}"), {{ attr: {{ r: 3 }}, opacity: 1 }}, {{ attr: {{ r: 16 }}, opacity: 0, duration: 0.55, ease: "power2.out" }}, ')


def ground(y=150, x0=16, x1=304):
    return f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{LINE}" stroke-width="2" stroke-linecap="round"/>'


def coverage(p, t0, dur):
    cols, rows, cw, ch, gx, gy = 10, 4, 28, 18, 20, 84
    levels = [0.62, 0.35, 0.8, 0.25, 0.5, 0.7, 0.3, 0.55, 0.85, 0.4]
    cells = []
    for c in range(cols):
        for r in range(rows):
            cells.append(f'<rect class="{p}-c{c}" x="{gx + c * cw + 1}" y="{gy + r * ch + 1}" width="{cw - 2}" height="{ch - 2}" rx="2" '
                         f'fill="{A}" fill-opacity="0" stroke="{FAINT}" stroke-width="1"/>')
    svg = (f'<g>{"".join(cells)}</g>'
           f'<path id="{p}-fr" d="M0 0 L-26 {gy - 30 + 60} L26 {gy - 30 + 60} Z" transform="translate(20 24)" fill="{A}" fill-opacity=".14"/>'
           f'<g id="{p}-dr" transform="translate(20 24)"><circle r="6" fill="{INK}"/><line x1="-12" y1="0" x2="12" y2="0" stroke="{INK}" stroke-width="2"/></g>'
           f'<text x="20" y="166" fill="{DIM}" {LBL}>seen well</text><rect x="92" y="157" width="40" height="8" rx="2" fill="{A}" fill-opacity=".8"/>'
           f'<text x="150" y="166" fill="{DIM}" {LBL}>seen once</text><rect x="228" y="157" width="40" height="8" rx="2" fill="{A}" fill-opacity=".25"/>')
    body = (f'            s.fromTo([$("#{p}-dr"), $("#{p}-fr")], {{ x: 20 }}, {{ x: 300, duration: 2.8, ease: "none" }}, 0);\n')
    for c in range(cols):
        body += (f'            s.fromTo($(".{p}-c{c}"), {{ attr: {{ "fill-opacity": 0 }} }}, {{ attr: {{ "fill-opacity": {levels[c]} }}, duration: 0.35, '
                 f'stagger: {{ each: 0.05, from: "random" }}, ease: "power2.out" }}, {0.1 + c * 0.27:.2f});\n')
    body += f'            s.to($("[class^={p}-c]"), {{ attr: {{ "fill-opacity": 0 }}, duration: 0.5 }}, 3.3);\n'
    return svg, _loop(p, t0, dur, 3.8, body)


def tilt(p, t0, dur):
    ax, ay = 160, 150
    r = 78
    ang = math.radians(9)
    ex, ey = ax + r * math.sin(ang), ay - r * math.cos(ang)
    rp, rj = ripple(p, "rp", ax, ay)
    svg = (ground() + f'<line x1="{ax}" y1="{ay}" x2="{ax}" y2="22" stroke="{DIM}" stroke-width="1.5" stroke-dasharray="4 5"/>'
           f'<g id="{p}-pole"><line x1="{ax}" y1="{ay}" x2="{ax}" y2="26" stroke="{INK}" stroke-width="6" stroke-linecap="round"/>'
           f'<line x1="{ax - 16}" y1="44" x2="{ax + 16}" y2="44" stroke="{INK}" stroke-width="4" stroke-linecap="round"/></g>'
           f'<path id="{p}-arc" d="M{ax} {ay - r} A{r} {r} 0 0 1 {ex:.1f} {ey:.1f}" fill="none" stroke="{A}" stroke-width="3" stroke-linecap="round" '
           f'stroke-dasharray="20" stroke-dashoffset="20"/>'
           f'<text id="{p}-th" x="{ax + 22}" y="{ay - r - 6}" fill="{A}" font-family="Geist, sans-serif" font-size="22" font-weight="700" opacity="0">θ</text>'
           f'<text id="{p}-lb" x="{ax + 44}" y="{ay - r - 6}" fill="{DIM}" {LBL} opacity="0">lean</text>' + rp)
    body = (f'            {rj}0.1);\n'
            f'            s.fromTo($("#{p}-pole"), {{ rotation: 0 }}, {{ rotation: 9, svgOrigin: "{ax} {ay}", duration: 1.1, ease: "back.out(1.4)" }}, 0.35);\n'
            f'            s.fromTo($("#{p}-arc"), {{ attr: {{ "stroke-dashoffset": 20 }} }}, {{ attr: {{ "stroke-dashoffset": 0 }}, duration: 0.9, ease: "power2.out" }}, 0.55);\n'
            f'            s.fromTo([$("#{p}-th"), $("#{p}-lb")], {{ opacity: 0, y: 6 }}, {{ opacity: 1, y: 0, duration: 0.4, stagger: 0.1 }}, 1.0);\n'
            f'            s.to($("#{p}-pole"), {{ rotation: 0, svgOrigin: "{ax} {ay}", duration: 0.7, ease: "power2.inOut" }}, 2.9);\n'
            f'            s.to($("#{p}-arc"), {{ attr: {{ "stroke-dashoffset": 20 }}, duration: 0.5 }}, 2.9);\n'
            f'            s.to([$("#{p}-th"), $("#{p}-lb")], {{ opacity: 0, duration: 0.3 }}, 2.9);\n')
    return svg, _loop(p, t0, dur, 3.7, body)


def cable(p, t0, dur):
    r1, j1 = ripple(p, "r1", 52, 44)
    r2, j2 = ripple(p, "r2", 268, 44)
    svg = (ground() + f'<line x1="52" y1="150" x2="52" y2="40" stroke="{INK}" stroke-width="5" stroke-linecap="round"/>'
           f'<line x1="268" y1="150" x2="268" y2="40" stroke="{INK}" stroke-width="5" stroke-linecap="round"/>'
           f'<path id="{p}-cb" d="M52 44 Q160 44 268 44" fill="none" stroke="{A}" stroke-width="3" stroke-linecap="round"/>'
           f'<line id="{p}-sag" x1="160" y1="44" x2="160" y2="84" stroke="{A}" stroke-width="2" stroke-dasharray="3 4" opacity="0"/>'
           f'<line id="{p}-clr" x1="160" y1="88" x2="160" y2="148" stroke="{GRN}" stroke-width="2.5" opacity="0"/>'
           f'<text id="{p}-t1" x="168" y="70" fill="{A}" {LBL} opacity="0">sag</text>'
           f'<text id="{p}-t2" x="168" y="124" fill="{GRN}" {LBL} opacity="0">clearance</text>' + r1 + r2)
    body = (f'            {j1}0.1);\n            {j2}0.5);\n'
            f'            s.fromTo($("#{p}-cb"), {{ attr: {{ d: "M52 44 Q160 44 268 44" }} }}, {{ attr: {{ d: "M52 44 Q160 124 268 44" }}, duration: 1.2, ease: "elastic.out(1, 0.55)" }}, 0.75);\n'
            f'            s.fromTo($("#{p}-sag"), {{ opacity: 0, scaleY: 0, svgOrigin: "160 44" }}, {{ opacity: 1, scaleY: 1, duration: 0.5 }}, 1.4);\n'
            f'            s.fromTo($("#{p}-clr"), {{ opacity: 0, scaleY: 0, svgOrigin: "160 148" }}, {{ opacity: 1, scaleY: 1, duration: 0.5 }}, 1.7);\n'
            f'            s.fromTo([$("#{p}-t1"), $("#{p}-t2")], {{ opacity: 0, x: -6 }}, {{ opacity: 1, x: 0, duration: 0.4, stagger: 0.25 }}, 1.6);\n'
            f'            s.to([$("#{p}-sag"), $("#{p}-clr"), $("#{p}-t1"), $("#{p}-t2")], {{ opacity: 0, duration: 0.35 }}, 3.3);\n'
            f'            s.to($("#{p}-cb"), {{ attr: {{ d: "M52 44 Q160 44 268 44" }}, duration: 0.5, ease: "power2.inOut" }}, 3.3);\n')
    return svg, _loop(p, t0, dur, 3.9, body)


def moved(p, t0, dur):
    old = "M12 128 C 70 128, 104 92, 150 92 C 196 92, 226 118, 308 114"
    new = "M12 128 C 70 128, 104 58, 150 60 C 196 62, 226 122, 308 114"
    svg = (f'<path d="{old}" fill="none" stroke="{DIM}" stroke-width="2.5" stroke-dasharray="5 5"/>'
           f'<path id="{p}-n" d="{old}" fill="none" stroke="{A}" stroke-width="3" stroke-linecap="round"/>'
           f'<ellipse id="{p}-hl" cx="150" cy="76" rx="54" ry="30" fill="{A}" fill-opacity=".12" stroke="{A}" stroke-width="1.5" opacity="0"/>'
           f'<path id="{p}-ar" d="M150 88 V64 M143 71 L150 64 L157 71" fill="none" stroke="{INK}" stroke-width="2" stroke-linecap="round" opacity="0"/>'
           f'<text x="16" y="160" fill="{DIM}" {LBL}>older scan</text><text x="120" y="160" fill="{A}" {LBL}>this scan</text>'
           f'<text id="{p}-mv" x="210" y="54" fill="{A}" font-family="Geist, sans-serif" font-size="15" font-weight="650" opacity="0">moved</text>')
    body = (f'            s.fromTo($("#{p}-n"), {{ attr: {{ d: "{old}" }} }}, {{ attr: {{ d: "{new}" }}, duration: 1.1, ease: "power3.inOut" }}, 0.2);\n'
            f'            s.fromTo($("#{p}-hl"), {{ opacity: 0, scale: 0.6, svgOrigin: "150 76" }}, {{ opacity: 1, scale: 1, duration: 0.6, ease: "back.out(2)" }}, 1.2);\n'
            f'            s.fromTo([$("#{p}-ar"), $("#{p}-mv")], {{ opacity: 0, y: 8 }}, {{ opacity: 1, y: 0, duration: 0.45, stagger: 0.12 }}, 1.4);\n'
            f'            s.to([$("#{p}-hl"), $("#{p}-ar"), $("#{p}-mv")], {{ opacity: 0, duration: 0.35 }}, 3.2);\n'
            f'            s.to($("#{p}-n"), {{ attr: {{ d: "{old}" }}, duration: 0.6, ease: "power2.inOut" }}, 3.2);\n')
    return svg, _loop(p, t0, dur, 3.9, body)


def seasons(p, t0, dur):
    st = ["M12 150 L12 60 C 60 58, 110 52, 150 70 C 190 88, 240 110, 308 118 L308 150 Z",
          "M12 150 L12 64 C 60 64, 104 66, 146 82 C 186 98, 238 114, 308 120 L308 150 Z",
          "M12 150 L12 70 C 58 72, 100 84, 140 98 C 182 112, 236 120, 308 124 L308 150 Z"]
    names = ["spring", "summer", "autumn"]
    svg = (f'<path d="{st[0]}" fill="none" stroke="{DIM}" stroke-width="1.5" stroke-dasharray="4 5"/>'
           f'<path id="{p}-s" d="{st[0]}" fill="{A}" fill-opacity=".16" stroke="{A}" stroke-width="2.5"/>'
           + "".join(f'<text id="{p}-n{k}" x="228" y="40" fill="{INK}" font-family="Geist, sans-serif" font-size="16" font-weight="650" opacity="0">{n}</text>'
                     for k, n in enumerate(names))
           + f'<circle cx="206" cy="35" r="10" fill="none" stroke="{DIM}" stroke-width="1.5"/>'
             f'<line id="{p}-hand" x1="206" y1="35" x2="206" y2="28" stroke="{A}" stroke-width="2" stroke-linecap="round"/>'
             f'<text x="16" y="166" fill="{DIM}" {LBL}>the same slope, season by season</text>')
    body = f'            s.fromTo($("#{p}-hand"), {{ rotation: 0 }}, {{ rotation: 360, svgOrigin: "206 35", duration: 3.6, ease: "none" }}, 0);\n'
    for k in range(3):
        a = 0.1 + k * 1.2
        body += f'            s.fromTo($("#{p}-n{k}"), {{ opacity: 0, y: 8 }}, {{ opacity: 1, y: 0, duration: 0.3 }}, {a:.2f});\n'
        body += f'            s.to($("#{p}-n{k}"), {{ opacity: 0, y: -8, duration: 0.25 }}, {a + 1.0:.2f});\n'
        if k:
            body += f'            s.to($("#{p}-s"), {{ attr: {{ d: "{st[k]}" }}, duration: 0.9, ease: "power2.inOut" }}, {a - 0.2:.2f});\n'
    body += f'            s.to($("#{p}-s"), {{ attr: {{ d: "{st[0]}" }}, duration: 0.4, ease: "power2.inOut" }}, 3.5);\n'
    return svg, _loop(p, t0, dur, 3.9, body)


def restore(p, t0, dur):
    full = "M40 140 V78 H92 V36 H128 V78 H196 V96 H280 V140"
    svg = (ground(140) +
           f'<path d="M40 140 V104 L58 96 L70 108 L92 92 V78 L112 70 L118 90 L128 84 V110 L150 118 L176 108 L196 118 V140 Z" fill="{DIM}" fill-opacity=".4" stroke="{DIM}" stroke-width="1.5"/>'
           f'<clipPath id="{p}-cp"><rect id="{p}-cr" x="0" y="0" width="0" height="170"/></clipPath>'
           f'<path d="{full}" fill="{A}" fill-opacity=".06" stroke="{A}" stroke-width="2.5" stroke-dasharray="7 6" clip-path="url(#{p}-cp)"/>'
           f'<g id="{p}-pen"><path d="M0 0 l-6 -16 l6 -4 l6 4 z" fill="{A}"/></g>'
           f'<g id="{p}-tag" opacity="0"><rect x="206" y="26" width="92" height="28" rx="14" fill="hsl(222 12% 12%)" stroke="{A}" stroke-width="1.5"/>'
           f'<text x="224" y="45" fill="{A}" font-family="Geist, sans-serif" font-size="14" font-weight="650">a guess</text></g>')
    body = (f'            s.fromTo($("#{p}-cr"), {{ attr: {{ width: 0 }} }}, {{ attr: {{ width: 300 }}, duration: 2.0, ease: "power1.inOut" }}, 0.1);\n'
            f'            s.fromTo($("#{p}-pen"), {{ x: 40, y: 78, opacity: 1 }}, {{ keyframes: [{{ x: 92, y: 40, duration: 0.6 }}, {{ x: 128, y: 40, duration: 0.3 }}, {{ x: 196, y: 80, duration: 0.5 }}, {{ x: 280, y: 98, duration: 0.6 }}], ease: "none" }}, 0.1);\n'
            f'            s.to($("#{p}-pen"), {{ opacity: 0, duration: 0.2 }}, 2.1);\n'
            f'            s.fromTo($("#{p}-tag"), {{ opacity: 0, y: 8 }}, {{ opacity: 1, y: 0, duration: 0.45, ease: "back.out(2)" }}, 2.0);\n'
            f'            s.to([$("#{p}-tag")], {{ opacity: 0, duration: 0.3 }}, 3.4);\n'
            f'            s.to($("#{p}-cr"), {{ attr: {{ width: 0 }}, duration: 0.4 }}, 3.4);\n')
    return svg, _loop(p, t0, dur, 3.9, body)


def changed(p, t0, dur):
    def houses(broken):
        out = []
        for k, x in enumerate((34, 104, 196, 250)):
            if broken and k == 2:
                out.append(f'<path d="M{x} 132 l8 -10 l7 6 l9 -12 l6 9 l10 -4 v11 z" fill="{RED}" fill-opacity=".55"/>')
            else:
                out.append(f'<path d="M{x} 132 v-26 l20 -16 l20 16 v26 z" fill="{INK}" fill-opacity=".85"/>')
        return "".join(out)
    svg = (f'<rect x="10" y="20" width="300" height="130" rx="8" fill="hsl(222 12% 11%)"/>'
           f'<g>{houses(False)}</g>'
           f'<clipPath id="{p}-cp"><rect id="{p}-cr" x="10" y="20" width="0" height="130"/></clipPath>'
           f'<g clip-path="url(#{p}-cp)"><rect x="10" y="20" width="300" height="130" rx="8" fill="hsl(222 12% 13%)"/>{houses(True)}</g>'
           f'<line id="{p}-wl" x1="10" y1="16" x2="10" y2="154" stroke="{A}" stroke-width="3"/>'
           f'<circle id="{p}-ring" cx="216" cy="122" r="22" fill="none" stroke="{A}" stroke-width="2.5" opacity="0"/>'
           f'<text x="14" y="168" fill="{DIM}" {LBL}>before the event</text><text x="222" y="168" fill="{A}" {LBL}>after</text>')
    body = (f'            s.fromTo($("#{p}-cr"), {{ attr: {{ width: 0 }} }}, {{ attr: {{ width: 300 }}, duration: 1.6, ease: "power2.inOut" }}, 0.2);\n'
            f'            s.fromTo($("#{p}-wl"), {{ x: 0 }}, {{ x: 300, duration: 1.6, ease: "power2.inOut" }}, 0.2);\n'
            f'            s.fromTo($("#{p}-ring"), {{ opacity: 0, scale: 1.8, svgOrigin: "216 122" }}, {{ opacity: 1, scale: 1, duration: 0.5, ease: "back.out(2)" }}, 1.5);\n'
            f'            s.to($("#{p}-ring"), {{ scale: 1.15, svgOrigin: "216 122", duration: 0.4, yoyo: true, repeat: 3, ease: "sine.inOut" }}, 2.0);\n'
            f'            s.to($("#{p}-ring"), {{ opacity: 0, duration: 0.3 }}, 3.6);\n'
            f'            s.to($("#{p}-cr"), {{ attr: {{ width: 0 }}, duration: 0.01 }}, 3.9);\n'
            f'            s.to($("#{p}-wl"), {{ x: 0, duration: 0.01 }}, 3.9);\n')
    return svg, _loop(p, t0, dur, 3.95, body)


def damage(p, t0, dur):
    xs = (34, 130, 226)
    cols = (GRN, A, RED)
    names = ("intact", "damaged", "collapsed")
    svg = ground(132)
    for k, x in enumerate(xs):
        shape = f"M{x} 132 V64 H{x + 60} V132 Z"
        svg += (f'<path id="{p}-b{k}" d="{shape}" fill="{cols[k]}" fill-opacity="0" stroke="{INK}" stroke-width="2" stroke-linejoin="round"/>'
                f'<text id="{p}-l{k}" x="{x}" y="156" fill="{cols[k]}" {LBL} opacity="0">{names[k]}</text>')
    svg += f'<line id="{p}-scan" x1="20" y1="44" x2="20" y2="136" stroke="{A}" stroke-width="2" opacity=".9"/>'
    rubble = f"M{xs[2]} 132 V116 L{xs[2] + 12} 104 L{xs[2] + 22} 114 L{xs[2] + 34} 96 L{xs[2] + 44} 110 L{xs[2] + 60} 104 V132 Z"
    body = f'            s.fromTo($("#{p}-scan"), {{ x: 0 }}, {{ x: 280, duration: 2.2, ease: "none" }}, 0);\n'
    for k in range(3):
        a = 0.35 + k * 0.62
        body += (f'            s.fromTo($("#{p}-b{k}"), {{ attr: {{ "fill-opacity": 0 }} }}, {{ attr: {{ "fill-opacity": 0.6 }}, duration: 0.35 }}, {a:.2f});\n'
                 f'            s.fromTo($("#{p}-l{k}"), {{ opacity: 0, y: 6 }}, {{ opacity: 1, y: 0, duration: 0.3 }}, {a + 0.1:.2f});\n')
    body += (f'            s.to($("#{p}-b1"), {{ rotation: -4, svgOrigin: "{xs[1]} 132", duration: 0.4, ease: "back.out(3)" }}, 1.05);\n'
             f'            s.fromTo($("#{p}-b2"), {{ attr: {{ d: "M{xs[2]} 132 V64 L{xs[2] + 12} 64 L{xs[2] + 22} 64 L{xs[2] + 34} 64 L{xs[2] + 44} 64 L{xs[2] + 60} 64 V132 Z" }} }}, {{ attr: {{ d: "{rubble}" }}, duration: 0.5, ease: "bounce.out" }}, 1.6);\n'
             f'            s.to([$("#{p}-b0"), $("#{p}-b1"), $("#{p}-b2")], {{ attr: {{ "fill-opacity": 0 }}, duration: 0.4 }}, 3.3);\n'
             f'            s.to($("#{p}-b1"), {{ rotation: 0, svgOrigin: "{xs[1]} 132", duration: 0.4 }}, 3.3);\n'
             f'            s.to($("[id^={p}-l]"), {{ opacity: 0, duration: 0.3 }}, 3.3);\n')
    # the collapsed building's path resets with the fromTo on the next cycle
    svg = svg.replace(f'd="M{xs[2]} 132 V64 H{xs[2] + 60} V132 Z"',
                      f'd="M{xs[2]} 132 V64 L{xs[2] + 12} 64 L{xs[2] + 22} 64 L{xs[2] + 34} 64 L{xs[2] + 44} 64 L{xs[2] + 60} 64 V132 Z"')
    return svg, _loop(p, t0, dur, 3.8, body)


def debris(p, t0, dur):
    mound = "M40 140 C 70 140, 90 70, 150 66 C 208 62, 236 132, 280 140 Z"
    lasso = "M28 144 C 40 90, 110 44, 160 50 C 222 56, 266 108, 292 146 Z"
    svg = (ground(140) + f'<clipPath id="{p}-mc"><path d="{mound}"/></clipPath>'
           f'<path d="{mound}" fill="{DIM}" fill-opacity=".3" stroke="{DIM}" stroke-width="1.5"/>'
           f'<rect id="{p}-lv" x="30" y="140" width="260" height="90" fill="{A}" fill-opacity=".55" clip-path="url(#{p}-mc)"/>'
           f'<path id="{p}-ls" d="{lasso}" fill="none" stroke="{A}" stroke-width="2.5" stroke-linejoin="round" stroke-dasharray="620" stroke-dashoffset="620"/>'
           f'<text id="{p}-m3" x="226" y="40" fill="{A}" font-family="Geist, sans-serif" font-size="18" font-weight="700" opacity="0">m³</text>')
    body = (f'            s.fromTo($("#{p}-ls"), {{ attr: {{ "stroke-dashoffset": 620 }} }}, {{ attr: {{ "stroke-dashoffset": 0 }}, duration: 1.2, ease: "power2.inOut" }}, 0.1);\n'
            f'            s.fromTo($("#{p}-lv"), {{ attr: {{ y: 140 }} }}, {{ attr: {{ y: 60 }}, duration: 1.3, ease: "power2.out" }}, 1.2);\n'
            f'            s.fromTo($("#{p}-m3"), {{ opacity: 0, scale: 0.5, svgOrigin: "236 34" }}, {{ opacity: 1, scale: 1, duration: 0.45, ease: "back.out(2.5)" }}, 2.0);\n'
            f'            s.to([$("#{p}-m3")], {{ opacity: 0, duration: 0.3 }}, 3.4);\n'
            f'            s.to($("#{p}-lv"), {{ attr: {{ y: 140 }}, duration: 0.4 }}, 3.4);\n'
            f'            s.to($("#{p}-ls"), {{ attr: {{ "stroke-dashoffset": 620 }}, duration: 0.4 }}, 3.4);\n')
    return svg, _loop(p, t0, dur, 3.9, body)


def people(p, t0, dur):
    svg = (f'<rect x="10" y="14" width="300" height="136" rx="8" fill="hsl(222 12% 11%)"/>'
           + "".join(f'<line x1="{x}" y1="14" x2="{x}" y2="150" stroke="{FAINT}" stroke-width="1"/>' for x in range(40, 310, 30))
           + "".join(f'<line x1="10" y1="{y}" x2="310" y2="{y}" stroke="{FAINT}" stroke-width="1"/>' for y in range(34, 150, 30))
           + f'<path d="M10 104 C 90 100, 150 70, 310 64" fill="none" stroke="{LINE}" stroke-width="10" stroke-linecap="round"/>')
    who = [("p", 60, 60), ("p", 96, 128), ("p", 232, 110), ("p", 270, 32), ("v", 70, 101), ("v", 200, 78)]
    body = ""
    for k, (kind, x, y) in enumerate(who):
        if kind == "p":
            svg += f'<circle id="{p}-w{k}" cx="{x}" cy="{y}" r="6" fill="{INK}" opacity="0"/>'
        else:
            svg += f'<rect id="{p}-w{k}" x="{x - 8}" y="{y - 5}" width="16" height="10" rx="2" fill="{A}" opacity="0"/>'
        svg += f'<circle id="{p}-rw{k}" cx="{x}" cy="{y}" r="6" fill="none" stroke="{A if kind == "v" else INK}" stroke-width="1.5" opacity="0"/>'
        a = 0.2 + k * 0.3
        body += (f'            s.fromTo($("#{p}-w{k}"), {{ opacity: 0, scale: 0, svgOrigin: "{x} {y}" }}, {{ opacity: 1, scale: 1, duration: 0.35, ease: "back.out(3)" }}, {a:.2f});\n'
                 f'            s.fromTo($("#{p}-rw{k}"), {{ opacity: 1, attr: {{ r: 6 }} }}, {{ opacity: 0, attr: {{ r: 20 }}, duration: 0.6 }}, {a:.2f});\n')
    body += (f'            s.to($("#{p}-w4"), {{ x: 110, y: -24, duration: 1.8, ease: "power1.inOut" }}, 1.4);\n'
             f'            s.to($("#{p}-w5"), {{ x: 70, y: -10, duration: 1.8, ease: "power1.inOut" }}, 1.6);\n'
             f'            s.to($("[id^={p}-w]"), {{ opacity: 0, duration: 0.3 }}, 3.5);\n'
             f'            s.to([$("#{p}-w4"), $("#{p}-w5")], {{ x: 0, y: 0, duration: 0.01 }}, 3.85);\n')
    svg += (f'<circle cx="22" cy="164" r="4" fill="{INK}"/><text x="32" y="168" fill="{DIM}" {LBL}>person</text>'
            f'<rect x="100" y="159" width="12" height="9" rx="2" fill="{A}"/><text x="118" y="168" fill="{DIM}" {LBL}>vehicle</text>')
    return svg, _loop(p, t0, dur, 3.9, body)


def firstmap(p, t0, dur):
    cols, rows, cw, ch = 7, 3, 40, 36
    gx, gy = 20, 24
    order = []
    for r in range(rows):
        cs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        order += [(r, c) for c in cs]
    tints = [0.35, 0.55, 0.45, 0.7, 0.4, 0.6, 0.5]
    svg = ""
    for r in range(rows):
        for c in range(cols):
            svg += (f'<rect id="{p}-t{r}{c}" x="{gx + c * cw + 1}" y="{gy + r * ch + 1}" width="{cw - 2}" height="{ch - 2}" rx="3" '
                    f'fill="hsl({90 + (c * 7 + r * 11) % 40} 30% {30 + (c + r) % 3 * 6}%)" opacity="0"/>'
                    f'<rect x="{gx + c * cw + 1}" y="{gy + r * ch + 1}" width="{cw - 2}" height="{ch - 2}" rx="3" fill="none" stroke="{FAINT}"/>')
    pts = []
    for r, c in order:
        pts.append((gx + c * cw + cw / 2, gy + r * ch + ch / 2))
    path = "M" + " L".join(f"{x:.0f} {y:.0f}" for x, y in pts)
    svg += (f'<path id="{p}-path" d="{path}" fill="none" stroke="{A}" stroke-width="2" stroke-dasharray="1400" stroke-dashoffset="1400" opacity=".8"/>'
            f'<circle id="{p}-dr" cx="0" cy="0" r="6" fill="{INK}"/>'
            f'<text x="20" y="160" fill="{DIM}" {LBL}>the map fills in while the drone flies</text>')
    step = 2.9 / len(order)
    kf = ", ".join(f"{{ x: {x:.0f}, y: {y:.0f}, duration: {step:.3f} }}" for x, y in pts[1:])
    body = (f'            s.fromTo($("#{p}-dr"), {{ x: {pts[0][0]:.0f}, y: {pts[0][1]:.0f} }}, {{ keyframes: [{kf}], ease: "none" }}, 0);\n'
            f'            s.fromTo($("#{p}-path"), {{ attr: {{ "stroke-dashoffset": 1400 }} }}, {{ attr: {{ "stroke-dashoffset": {1400 - (cols - 1) * rows * cw - (rows - 1) * ch:.0f} }}, duration: 2.9, ease: "none" }}, 0);\n')
    for k, (r, c) in enumerate(order):
        body += f'            s.fromTo($("#{p}-t{r}{c}"), {{ opacity: 0 }}, {{ opacity: {tints[c]:.2f}, duration: 0.3 }}, {k * step:.3f});\n'
    body += f'            s.to($("[id^={p}-t]"), {{ opacity: 0, duration: 0.4 }}, 3.4);\n'
    return svg, _loop(p, t0, dur, 3.9, body)


def stockpile(p, t0, dur):
    before = "M50 140 C 80 140, 110 96, 160 94 C 210 92, 240 140, 270 140 Z"
    now = "M36 140 C 70 140, 100 58, 160 54 C 220 50, 252 140, 284 140 Z"
    svg = (ground(140) + f'<path id="{p}-now" d="{before}" fill="{A}" fill-opacity=".28" stroke="{A}" stroke-width="2.5"/>'
           f'<path d="{before}" fill="none" stroke="{INK}" stroke-width="1.8" stroke-dasharray="5 5"/>'
           f'<text x="16" y="164" fill="{DIM}" {LBL}>last flight</text><text x="110" y="164" fill="{A}" {LBL}>this flight</text>'
           f'<g id="{p}-d" opacity="0"><rect x="208" y="20" width="96" height="30" rx="15" fill="hsl(222 12% 12%)" stroke="{A}" stroke-width="1.5"/>'
           f'<text x="222" y="40" fill="{A}" font-family="Geist, sans-serif" font-size="14" font-weight="650">Δ volume</text></g>')
    body = (f'            s.fromTo($("#{p}-now"), {{ attr: {{ d: "{before}" }} }}, {{ attr: {{ d: "{now}" }}, duration: 1.4, ease: "power3.inOut" }}, 0.2);\n'
            f'            s.fromTo($("#{p}-d"), {{ opacity: 0, y: 8 }}, {{ opacity: 1, y: 0, duration: 0.45, ease: "back.out(2)" }}, 1.4);\n'
            f'            s.to($("#{p}-now"), {{ attr: {{ "fill-opacity": 0.5 }}, duration: 0.4, yoyo: true, repeat: 2 }}, 1.7);\n'
            f'            s.to($("#{p}-d"), {{ opacity: 0, duration: 0.3 }}, 3.3);\n'
            f'            s.to($("#{p}-now"), {{ attr: {{ d: "{before}" }}, duration: 0.5, ease: "power2.inOut" }}, 3.3);\n')
    return svg, _loop(p, t0, dur, 3.9, body)


def cutfill(p, t0, dur):
    yd = 96
    cut = f"M20 {yd} C 50 {yd}, 70 52, 110 52 C 140 52, 150 {yd}, 164 {yd} Z"
    fil = f"M164 {yd} C 184 {yd}, 200 136, 240 136 C 270 136, 290 {yd}, 304 {yd} Z"
    terr = f"M20 {yd} C 50 {yd}, 70 52, 110 52 C 140 52, 150 {yd}, 164 {yd} C 184 {yd}, 200 136, 240 136 C 270 136, 290 {yd}, 304 {yd}"
    svg = (f'<path id="{p}-c" d="{cut}" fill="{A}" fill-opacity=".45"/>'
           f'<path id="{p}-f" d="{fil}" fill="{BLUE}" fill-opacity=".4"/>'
           f'<path d="{terr}" fill="none" stroke="{INK}" stroke-width="2.5"/>'
           f'<line id="{p}-dl" x1="20" y1="{yd}" x2="304" y2="{yd}" stroke="{A}" stroke-width="2" stroke-dasharray="7 5"/>'
           f'<text id="{p}-tc" x="90" y="44" fill="{A}" font-family="Geist, sans-serif" font-size="15" font-weight="650" opacity="0">cut</text>'
           f'<text id="{p}-tf" x="226" y="158" fill="{BLUE}" font-family="Geist, sans-serif" font-size="15" font-weight="650" opacity="0">fill</text>'
           f'<text x="250" y="{yd - 8}" fill="{A}" {LBL}>design</text>')
    body = (f'            s.fromTo($("#{p}-dl"), {{ scaleX: 0, svgOrigin: "20 {yd}" }}, {{ scaleX: 1, duration: 0.8, ease: "power2.inOut" }}, 0.1);\n'
            f'            s.fromTo($("#{p}-c"), {{ scaleY: 0, svgOrigin: "0 {yd}" }}, {{ scaleY: 1, duration: 0.7, ease: "back.out(1.6)" }}, 0.8);\n'
            f'            s.fromTo($("#{p}-f"), {{ scaleY: 0, svgOrigin: "0 {yd}" }}, {{ scaleY: 1, duration: 0.7, ease: "back.out(1.6)" }}, 1.2);\n'
            f'            s.fromTo([$("#{p}-tc"), $("#{p}-tf")], {{ opacity: 0, y: 6 }}, {{ opacity: 1, y: 0, duration: 0.35, stagger: 0.4 }}, 1.1);\n'
            f'            s.to([$("#{p}-c"), $("#{p}-f")], {{ scaleY: 0, svgOrigin: "0 {yd}", duration: 0.4 }}, 3.3);\n'
            f'            s.to([$("#{p}-tc"), $("#{p}-tf")], {{ opacity: 0, duration: 0.3 }}, 3.3);\n')
    return svg, _loop(p, t0, dur, 3.9, body)


def builtdesign(p, t0, dur):
    def box(x, y, w, h, d):
        return (f"M{x} {y + d} h{w} v{h} h{-w} Z M{x} {y + d} l{d} {-d} h{w} l{-d} {d} M{x + w} {y + d} l{d} {-d} v{h} l{-d} {d}")
    design = box(84, 30, 110, 96, 26)
    svg = (f'<path d="{design}" fill="none" stroke="{A}" stroke-width="2" stroke-dasharray="6 5" stroke-linejoin="round"/>'
           f'<path id="{p}-b" d="{box(96, 38, 110, 96, 26)}" fill="{INK}" fill-opacity=".1" stroke="{INK}" stroke-width="2" stroke-linejoin="round"/>'
           f'<line id="{p}-sc" x1="60" y1="20" x2="60" y2="150" stroke="{GRN}" stroke-width="2" opacity="0"/>'
           f'<path id="{p}-dv" d="M194 88 H206 M200 82 L206 88 L200 94" fill="none" stroke="{RED}" stroke-width="2.5" stroke-linecap="round" opacity="0"/>'
           f'<rect id="{p}-hl" x="190" y="60" width="22" height="84" rx="4" fill="{RED}" fill-opacity=".18" stroke="{RED}" stroke-width="1.5" opacity="0"/>'
           f'<text x="16" y="164" fill="{A}" {LBL}>design</text><text x="80" y="164" fill="{INK}" {LBL}>as built</text>'
           f'<text id="{p}-off" x="232" y="94" fill="{RED}" font-family="Geist, sans-serif" font-size="14" font-weight="650" opacity="0">off design</text>')
    body = (f'            s.fromTo($("#{p}-b"), {{ opacity: 0, y: 16 }}, {{ opacity: 1, y: 0, duration: 0.7, ease: "power3.out" }}, 0.1);\n'
            f'            s.fromTo($("#{p}-sc"), {{ opacity: 1, x: 0 }}, {{ opacity: 1, x: 200, duration: 1.4, ease: "power1.inOut" }}, 0.6);\n'
            f'            s.to($("#{p}-sc"), {{ opacity: 0, duration: 0.2 }}, 2.0);\n'
            f'            s.fromTo([$("#{p}-hl"), $("#{p}-dv"), $("#{p}-off")], {{ opacity: 0 }}, {{ opacity: 1, duration: 0.35, stagger: 0.12 }}, 1.6);\n'
            f'            s.to($("#{p}-hl"), {{ attr: {{ "fill-opacity": 0.4 }}, duration: 0.35, yoyo: true, repeat: 3 }}, 2.0);\n'
            f'            s.to([$("#{p}-hl"), $("#{p}-dv"), $("#{p}-off"), $("#{p}-b")], {{ opacity: 0, duration: 0.35 }}, 3.4);\n')
    return svg, _loop(p, t0, dur, 3.9, body)


KIND = {
    "Camera coverage": coverage, "Pole tilt": tilt, "Cable sag": cable, "What moved": moved,
    "Seasonal change": seasons, "Restoration idea": restore, "What changed": changed,
    "Building damage": damage, "Debris volume": debris, "People & vehicles": people, "First map": firstmap,
    "Stockpile volume": stockpile, "Cut & fill": cutfill, "Built vs design": builtdesign,
}


def illo(name, p, t0, dur):
    return KIND[name](p, t0, dur)
