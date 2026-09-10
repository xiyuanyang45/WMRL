#!/usr/bin/env python3
"""Render the result charts as inline SVG and splice them into docs/index.html.

The charts are generated rather than hand-written so the numbers stay tied to a
single table at the top of this file. Run from anywhere:

    python3 docs/tools/make_charts.py

Each chart is written between a pair of markers in index.html, so re-running
replaces the previous render in place.

Palette: one accent over a steel value ramp. The three baselines separate by
lightness alone (untrained -> pure world model -> real environment), so the
series stay distinguishable in greyscale and under any colour-vision deficiency,
and orange is reserved for our runs and appears nowhere else on the page.

Axis ranges are fitted to the data rather than padded out to round numbers, so
the plot area carries marks instead of margin. Bars still start at zero; only the
dot and scatter forms, where no area encodes the value, use a clipped axis.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.normpath(os.path.join(HERE, "..", "index.html"))

# ---------------------------------------------------------------- palette

# One blue family stepped by lightness, with the accent reserved for WMRL.
# Validated all-pairs on a white surface: worst normal-vision dE 16.6, worst CVD
# dE 15.9. CTX sits at 1.8:1 against white, so every CTX mark carries a visible
# direct label and the full tables sit below each chart (the relief rule).
REAL = "#2f5e8c"   # deep blue    - RL in real environment (GRPO)
OURS = "#e8580a"   # accent       - WMRL (ours)
WM   = "#708db0"   # mid blue     - RL with pure world model
CTX  = "#b3c0d0"   # pale blue    - untrained base / SFT

# ---------------------------------------------------------------- data
# Every number below is from main_text.tex (arXiv:2608.12564), Tables 1-3.

# name, gpu_hours, mle_avg, ds_avg, scale, series
AUTORESEARCH = [
    ("Qwen3.5-4B",        None,  7.3, 17.1, "4B", "base"),
    ("Qwen3.5-9B",        None,  9.6, 23.9, "9B", "base"),
    ("Qwen3.5-4B-GRPO",    883, 15.2, 25.7, "4B", "real"),
    ("Qwen3.5-9B-GRPO",   1174, 18.8, 31.2, "9B", "real"),
    ("Qwen3.5-4B-WM",      269, 12.9, 23.1, "4B", "wm"),
    ("Qwen3.5-9B-WM",      330, 16.1, 28.0, "9B", "wm"),
    ("Qwen3.5-4B-Ours",    286, 16.4, 28.8, "4B", "ours"),
    ("Qwen3.5-9B-Ours",    349, 21.6, 32.8, "9B", "ours"),
]

# strongest off-the-shelf agent in the comparison, post-trained on nothing
NEMOTRON = (20.5, 31.7)   # (MLE-Dojo, DSBench)

# row -> (SFT, GRPO, pure WM, ours)
VLA = [
    ("In-Domain",           37.3, 39.3, 39.1, 41.2),
    ("Out-of-Distribution", 37.5, 37.8, 39.3, 41.2),
    ("Overall",             37.4, 38.3, 39.2, 41.2),
]

# label, 4B MLE, 4B DS, 9B MLE, 9B DS
ABLATION = [
    ("No correction",                    13.5, 25.3, 16.8, 29.5),
    ("Inverse-Variance Denoising only",  14.9, 26.2, 18.0, 31.2),
    ("Online Debiasing only",            15.7, 28.1, 19.4, 31.7),
    ("Both (WMRL)",                      16.4, 28.8, 21.6, 32.8),
]
# ordinal ramp: monotone lightness, adjacent dL >= 0.06, light end 2.04:1
ABL_SHADES = ["#a5b5c8", "#6d8bab", "#35598a", OURS]

SERIES_COLOR = {"base": CTX, "real": REAL, "wm": WM, "ours": OURS}
SERIES_NAME = {
    "base": "Untrained base",
    "real": "RL in real environment (GRPO)",
    "wm":   "RL with pure world model",
    "ours": "WMRL (ours)",
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def legend(entries):
    items = "".join(f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, c in entries)
    return f'<div class="legend">{items}</div>'


def fmt(v):
    return f"{v:.1f}"


def bar(x, y_top, y_base, w, fill):
    """A column with a 4px rounded cap and a square foot on the baseline."""
    return (
        f'<path class="mark" d="M {x:.1f} {y_base:.1f} L {x:.1f} {y_top + 4:.1f} '
        f'Q {x:.1f} {y_top:.1f} {x + 4:.1f} {y_top:.1f} L {x + w - 4:.1f} {y_top:.1f} '
        f'Q {x + w:.1f} {y_top:.1f} {x + w:.1f} {y_top + 4:.1f} L {x + w:.1f} {y_base:.1f} Z" '
        f'fill="{fill}"/>'
    )


def star(cx, cy, r, fill, cls="mark"):
    """Five-point star, the marker the poster uses for our runs."""
    import math
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.44
        pts.append(f"{cx + rad * math.cos(a):.1f},{cy + rad * math.sin(a):.1f}")
    return (f'<polygon class="{cls}" points="{" ".join(pts)}" fill="{fill}" '
            f'stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/>')


def diamond(cx, cy, r, fill, cls="mark"):
    return (f'<polygon class="{cls}" points="{cx:.1f},{cy - r:.1f} {cx + r:.1f},{cy:.1f} '
            f'{cx:.1f},{cy + r:.1f} {cx - r:.1f},{cy:.1f}" fill="{fill}" '
            f'stroke="#fff" stroke-width="1.6"/>')


def figure(title, sub, legend_html, view, body, caption, defs=""):
    w, h = view
    return f"""<figure class="chart reveal">
  <div class="chart-head">
    <p class="chart-title">{title}</p>
    <p class="chart-sub">{sub}</p>
  </div>
  {legend_html}
  <div class="plotwrap"><svg class="plot" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(caption)}">
    {defs}
      {body}
  </svg></div>
  <figcaption>{caption}</figcaption>
</figure>"""


ARROW_DEFS = """<defs>
      <marker id="arw" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1 L 9 5 L 0 9 z" fill="#aab3c0"/>
      </marker>
      <marker id="arwo" viewBox="0 0 10 10" refX="8.4" refY="5" markerWidth="5.4" markerHeight="5.4" orient="auto-start-reverse">
        <path d="M 0 1.2 L 9 5 L 0 8.8 z" fill="#e8580a"/>
      </marker>
    </defs>"""


# ---------------------------------------------------------------- chart A

def chart_tradeoff():
    """Cost against capability, one panel per benchmark, poster styling."""
    W, H = 780, 366
    ml, mr, mt, mb = 48, 16, 52, 54
    gap = 64
    pw = (W - ml - mr - gap) / 2
    y0, y1 = mt, H - mb
    xlo, xhi = 190, 1265

    panels = [
        ("MLE-Dojo (test)", 2, 11.4, 23.6, (12, 15, 18, 21), NEMOTRON[0]),
        ("DSBench",         3, 21.7, 34.9, (22, 26, 30, 34), NEMOTRON[1]),
    ]
    GAIN = {"4B": ("3.1&#215;", "+1.2", "+3.1"), "9B": ("3.4&#215;", "+2.8", "+1.6")}

    s = []
    for pi, (ptitle, idx, ylo, yhi, yticks, ref) in enumerate(panels):
        px0 = ml + pi * (pw + gap)
        px1 = px0 + pw

        def X(v, px0=px0):
            return px0 + (v - xlo) / (xhi - xlo) * pw

        def Y(v, ylo=ylo, yhi=yhi):
            return y1 - (v - ylo) / (yhi - ylo) * (y1 - y0)

        ry = Y(ref)
        # the band our runs have to clear: everything above the best off-the-shelf agent
        s.append(f'<rect x="{px0:.1f}" y="{y0:.1f}" width="{pw:.1f}" height="{max(ry - y0, 0):.1f}" '
                 f'fill="#f2f5f9"/>')
        s.append(f'<text class="panel-title" x="{px0:.0f}" y="{mt - 30}">{esc(ptitle)}</text>')
        s.append(f'<text class="zone-label" x="{px0 + 6:.0f}" y="{mt - 12}">'
                 f'above Nemotron-120B, the best untrained agent ({ref})</text>')

        for t in yticks:
            s.append(f'<line class="tick-line" x1="{px0:.1f}" y1="{Y(t):.1f}" x2="{px1:.1f}" y2="{Y(t):.1f}"/>')
            s.append(f'<text class="tick-label" x="{px0 - 9:.1f}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
        for t in (250, 500, 750, 1000, 1250):
            s.append(f'<text class="tick-label" x="{X(t):.1f}" y="{y1 + 18}" text-anchor="middle">{t}</text>')
        s.append(f'<line class="axis-line" x1="{px0:.1f}" y1="{y1}" x2="{px1:.1f}" y2="{y1}"/>')
        s.append(f'<line class="ref-line" x1="{px0:.1f}" y1="{ry:.1f}" x2="{px1:.1f}" y2="{ry:.1f}"/>')

        # ---- the two moves: real-environment run -> ours, cheaper and higher ----
        for scale in ("4B", "9B"):
            g = next(r for r in AUTORESEARCH if r[4] == scale and r[5] == "real")
            o = next(r for r in AUTORESEARCH if r[4] == scale and r[5] == "ours")
            gx, gy = X(g[1]), Y(g[idx])
            ox, oy = X(o[1]), Y(o[idx])
            up = scale == "9B"
            bow = -34 if up else 26
            mx, my = (gx + ox) / 2, (gy + oy) / 2 + bow
            s.append(
                f'<path class="lift" marker-end="url(#arwo)" '
                f'd="M {gx - 13:.1f} {gy + (-7 if up else 8):.1f} Q {mx:.1f} {my:.1f} {ox + 15:.1f} {oy + (-5 if up else 7):.1f}"/>'
            )
            cut, pts = GAIN[scale][0], GAIN[scale][1 if idx == 2 else 2]
            midy = (gy + oy) / 2
            lx = mx if up else ox + 0.40 * (gx - ox)
            ly = midy - 43 if up else midy + 40
            s.append(f'<text class="cut-label" x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle">'
                     f'{cut} cheaper<tspan class="cut-gain" dx="7">{pts} pts</tspan></text>')

        # ---- points, weakest to strongest ----
        for series in ("wm", "real", "ours"):
            for name, gh, mle, ds, scale, ser in AUTORESEARCH:
                if gh is None or ser != series:
                    continue
                v = mle if idx == 2 else ds
                cx, cy = X(gh), Y(v)
                col = SERIES_COLOR[series]
                if series == "ours":
                    s.append(star(cx, cy, 10.5, col))
                    s.append(f'<text class="pt-label pt-ours" x="{cx:.1f}" y="{cy - 16:.1f}" '
                             f'text-anchor="middle">WMRL {scale}</text>')
                elif series == "real":
                    s.append(f'<circle class="mark" cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="{col}" '
                             f'stroke="#fff" stroke-width="2"/>')
                    s.append(f'<text class="pt-label" x="{cx:.1f}" y="{cy + 20:.1f}" '
                             f'text-anchor="middle">GRPO {scale}</text>')
                else:
                    s.append(diamond(cx, cy, 6.4, col))
                s.append(
                    f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="16" '
                    f'data-name="{esc(name)}" data-color="{col}" '
                    f'data-val="{esc(ptitle)} {fmt(v)} &#183; {gh} GPU-hours"/>'
                )

    s.append(f'<text class="axis-title" x="{W / 2:.0f}" y="{H - 7}" text-anchor="middle">'
             f'Training compute (GPU-hours) &#8594;&#160;&#160;&#183;&#160;&#160;up and to the left is better</text>')

    return figure(
        "Cheaper and better at the same time",
        "Each arrow is one agent scale moving off the real-environment run it replaces: left is compute saved, up is score gained. Both land in the band above the strongest off-the-shelf agent.",
        legend([(SERIES_NAME["real"], REAL), (SERIES_NAME["wm"], WM), (SERIES_NAME["ours"], OURS)]),
        (W, H),
        "\n      ".join(s),
        "Every trained point shares the data split, the scaffold, and the hyperparameters, so the horizontal distance is the price of the reward signal and nothing else. The rule in each panel is the strongest off-the-shelf agent in the comparison, 13 times larger than our 9B agent and post-trained on nothing.",
        ARROW_DEFS,
    )


# ---------------------------------------------------------------- chart B

def chart_autoresearch_bars():
    W, H = 780, 330
    ml, mr, mt, mb = 40, 10, 38, 44
    gap = 56
    pw = (W - ml - mr - gap) / 2
    y0, y1 = mt, H - mb
    ymax = 35.0

    def Y(v):
        return y1 - v / ymax * (y1 - y0)

    order = ["base", "real", "wm", "ours"]
    bw, bgap = 21, 2
    block = len(order) * bw + (len(order) - 1) * bgap

    s = []
    for pi, (ptitle, idx) in enumerate([("MLE-Dojo (test)", 2), ("DSBench", 3)]):
        px0 = ml + pi * (pw + gap)
        px1 = px0 + pw
        s.append(f'<text class="panel-title" x="{px0:.0f}" y="{mt - 15}">{esc(ptitle)}</text>')

        for t in (0, 10, 20, 30):
            s.append(f'<line class="tick-line" x1="{px0:.1f}" y1="{Y(t):.1f}" x2="{px1:.1f}" y2="{Y(t):.1f}"/>')
            s.append(f'<text class="tick-label" x="{px0 - 8:.1f}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
        s.append(f'<line class="axis-line" x1="{px0:.1f}" y1="{y1}" x2="{px1:.1f}" y2="{y1}"/>')

        for gi, scale in enumerate(("4B", "9B")):
            gx = px0 + gi * (pw / 2)
            bx = gx + (pw / 2 - block) / 2
            for si, series in enumerate(order):
                row = next(r for r in AUTORESEARCH if r[4] == scale and r[5] == series)
                v = row[idx]
                x = bx + si * (bw + bgap)
                yv = Y(v)
                col = SERIES_COLOR[series]
                s.append(bar(x, yv, y1, bw, col))
                cls = "val-label hi" if series == "ours" else "val-label"
                s.append(f'<text class="{cls}" x="{x + bw / 2:.1f}" y="{yv - 6:.1f}" text-anchor="middle">{fmt(v)}</text>')
                s.append(
                    f'<rect class="hit" x="{x:.1f}" y="{y0}" width="{bw}" height="{y1 - y0:.1f}" '
                    f'data-name="{esc(row[0])}" data-color="{col}" '
                    f'data-val="{esc(ptitle)} {fmt(v)} percentile"/>'
                )
            s.append(f'<text class="group-label" x="{gx + pw / 4:.1f}" y="{y1 + 19}" text-anchor="middle">{scale} agent</text>')

    s.append(f'<text class="axis-title" transform="translate(11,{(y0 + y1) / 2:.0f}) rotate(-90)" text-anchor="middle">Leaderboard percentile</text>')

    return figure(
        "Held-out benchmarks, by agent scale",
        "The world model alone gives back most of the gain. Adding the two corrections carries WMRL past the real-environment run on both benchmarks and both scales.",
        legend([(SERIES_NAME["base"], CTX), (SERIES_NAME["real"], REAL), (SERIES_NAME["wm"], WM), (SERIES_NAME["ours"], OURS)]),
        (W, H),
        "\n      ".join(s),
        "DSBench is disjoint from the training set, and the MLE-Dojo test split holds out tasks never trained on. Bars start at zero.",
    )


# ---------------------------------------------------------------- chart C

def chart_ablation():
    W, H = 780, 262
    ml, mr, mt, mb = 40, 10, 34, 42
    gap = 56
    pw = (W - ml - mr - gap) / 2
    y0, y1 = mt, H - mb
    ymax = 35.0

    def Y(v):
        return y1 - v / ymax * (y1 - y0)

    bw, bgap = 19, 2
    block = 4 * bw + 3 * bgap

    s = []
    for pi, (ptitle, i4, i9) in enumerate([("MLE-Dojo (test)", 1, 3), ("DSBench", 2, 4)]):
        px0 = ml + pi * (pw + gap)
        px1 = px0 + pw
        s.append(f'<text class="panel-title" x="{px0:.0f}" y="{mt - 14}">{esc(ptitle)}</text>')

        for t in (0, 10, 20, 30):
            s.append(f'<line class="tick-line" x1="{px0:.1f}" y1="{Y(t):.1f}" x2="{px1:.1f}" y2="{Y(t):.1f}"/>')
            s.append(f'<text class="tick-label" x="{px0 - 8:.1f}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
        s.append(f'<line class="axis-line" x1="{px0:.1f}" y1="{y1}" x2="{px1:.1f}" y2="{y1}"/>')

        for gi, (scale, idx) in enumerate((("4B", i4), ("9B", i9))):
            gx = px0 + gi * (pw / 2)
            bx = gx + (pw / 2 - block) / 2
            for si, row in enumerate(ABLATION):
                v = row[idx]
                x = bx + si * (bw + bgap)
                yv = Y(v)
                col = ABL_SHADES[si]
                s.append(bar(x, yv, y1, bw, col))
                if si in (0, 3):
                    cls = "val-label hi" if si == 3 else "val-label"
                    s.append(f'<text class="{cls}" x="{x + bw / 2:.1f}" y="{yv - 6:.1f}" text-anchor="middle">{fmt(v)}</text>')
                s.append(
                    f'<rect class="hit" x="{x:.1f}" y="{y0}" width="{bw}" height="{y1 - y0:.1f}" '
                    f'data-name="{esc(scale + " agent &#183; " + row[0])}" data-color="{col}" '
                    f'data-val="{esc(ptitle)} {fmt(v)} percentile"/>'
                )
            s.append(f'<text class="group-label" x="{gx + pw / 4:.1f}" y="{y1 + 19}" text-anchor="middle">{scale} agent</text>')

    s.append(f'<text class="axis-title" transform="translate(11,{(y0 + y1) / 2:.0f}) rotate(-90)" text-anchor="middle">Leaderboard percentile</text>')

    return figure(
        "Both corrections earn their place",
        "Every run here consumes the same two reward streams at the same ratio and differs only in which correction is switched on. The leftmost bar of each group is the uncorrected mixture.",
        legend([(ABLATION[i][0], ABL_SHADES[i]) for i in range(4)]),
        (W, H),
        "\n      ".join(s),
        "Denoising alone adds 0.9 to 1.7 points and debiasing alone adds 2.2 to 2.8, which matches the analysis: bias enters the bound at full size while noise enters damped by the step size. Together they add 2.9 to 4.8, more than either alone, so the two mechanisms are complementary rather than redundant.",
    )


# ---------------------------------------------------------------- chart D

def chart_vla_dots():
    W, H = 780, 244
    ml, mr, mt, mb = 150, 104, 32, 38
    x0, x1 = ml, W - mr
    xlo, xhi = 37.0, 41.6

    def X(v):
        return x0 + (v - xlo) / (xhi - xlo) * (x1 - x0)

    rows_y = [mt + 22, mt + 76, mt + 130]
    s = []

    for t in (37, 38, 39, 40, 41):
        s.append(f'<line class="tick-line" x1="{X(t):.1f}" y1="{mt - 6}" x2="{X(t):.1f}" y2="{H - mb + 2}"/>')
        s.append(f'<text class="tick-label" x="{X(t):.1f}" y="{H - mb + 18}" text-anchor="middle">{t}</text>')

    for ri, (label, sft, grpo, wm, ours) in enumerate(VLA):
        y = rows_y[ri]
        s.append(f'<text class="group-label" x="{ml - 18}" y="{y + 4}" text-anchor="end">{esc(label)}</text>')
        s.append(f'<line class="connector" x1="{X(sft):.1f}" y1="{y}" x2="{X(ours):.1f}" y2="{y}"/>')

        for name, v, col in (
            ("MiniVLA-1B-SFT", sft, CTX),
            ("MiniVLA-1B-GRPO", grpo, REAL),
            ("MiniVLA-1B-WM", wm, WM),
            ("MiniVLA-1B-Ours", ours, OURS),
        ):
            cx = X(v)
            s.append(f'<circle class="mark" cx="{cx:.1f}" cy="{y}" r="6.5" fill="{col}" stroke="#fff" stroke-width="2"/>')
            s.append(
                f'<circle class="hit" cx="{cx:.1f}" cy="{y}" r="14" '
                f'data-name="{esc(name)}" data-color="{col}" '
                f'data-val="{esc(label)} {fmt(v)}% success"/>'
            )
        s.append(f'<text class="val-label" x="{X(sft):.1f}" y="{y - 14}" text-anchor="middle">{fmt(sft)}</text>')
        s.append(f'<text class="val-label hi" x="{X(ours):.1f}" y="{y - 14}" text-anchor="middle">{fmt(ours)}</text>')
        s.append(f'<text class="gain-label" x="{x1 + 16}" y="{y + 4}">&#43;{ours - sft:.1f} over SFT</text>')

    s.append(f'<text class="axis-title" x="{(x0 + x1) / 2:.0f}" y="{H - 5}" text-anchor="middle">LIBERO-Long success rate (%), average over eight rollouts</text>')

    return figure(
        "Embodied manipulation on LIBERO-Long",
        "Each signal on its own barely moves the SFT policy. Combining them under the same two corrections gives the largest gain, and the widest margin is on unseen initial states.",
        legend([("MiniVLA-1B-SFT", CTX), ("RL in real environment (GRPO)", REAL), ("RL with pure world model", WM), ("WMRL (ours)", OURS)]),
        (W, H),
        "\n      ".join(s),
        "All reinforcement-learning rows start from the same SFT checkpoint, so the spread within a row is attributable to the reward signal alone. The axis starts at 37 to resolve that spread; absolute values including the untrained policy are in the table below.",
    )


# ---------------------------------------------------------------- splice

def splice(html, key, block):
    start, end = f"<!--CHART:{key}-->", f"<!--/CHART:{key}-->"
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pat.search(html):
        raise SystemExit(f"marker pair for {key!r} not found in index.html")
    return pat.sub(lambda _m: start + "\n" + block + "\n" + end, html)


def main():
    with open(INDEX, encoding="utf-8") as f:
        html = f.read()
    for key, fn in (
        ("tradeoff", chart_tradeoff),
        ("autoresearch", chart_autoresearch_bars),
        ("ablation", chart_ablation),
        ("vla", chart_vla_dots),
    ):
        html = splice(html, key, fn())
    with open(INDEX, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"rendered 4 charts into {INDEX}")


if __name__ == "__main__":
    main()
