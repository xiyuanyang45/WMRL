#!/usr/bin/env python3
"""Render the result charts as inline SVG and splice them into docs/index.html.

The charts are generated rather than hand-written so the numbers stay tied to a
single table at the top of this file. Run from anywhere:

    python3 docs/tools/make_charts.py

Each chart is written between a pair of markers in index.html, so re-running
replaces the previous render in place.

Palette: slots 1-3 of the reference categorical palette, validated all-pairs on
a light surface (worst CVD dE 9.2, worst normal-vision dE 24.0). Aqua sits below
3:1 against white, so every aqua mark carries a visible direct label and the full
tables live directly below each chart.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.normpath(os.path.join(HERE, "..", "index.html"))

# ---------------------------------------------------------------- palette

REAL = "#2a78d6"   # slot 1 blue   - RL in real environment (GRPO)
OURS = "#eb6834"   # slot 2 orange - WMRL (ours)
WM   = "#1baf7a"   # slot 3 aqua   - RL with pure world model
CTX  = "#a8afba"   # de-emphasis gray - untrained base / SFT

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

NEMOTRON_MLE = 20.5   # Nemotron-120B-A12B, no post-training

# row -> (SFT, GRPO, pure WM, ours)
VLA = [
    ("In-Domain",           37.3, 39.3, 39.1, 41.2),
    ("Out-of-Distribution", 37.5, 37.8, 39.3, 41.2),
    ("Overall",             37.4, 38.3, 39.2, 41.2),
]

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
    items = "".join(
        f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, c in entries
    )
    return f'<div class="legend">{items}</div>'


def fmt(v):
    return f"{v:.1f}"


# ---------------------------------------------------------------- chart A

def chart_tradeoff():
    W, H = 780, 442
    ml, mr, mt, mb = 66, 26, 26, 58
    x0, x1 = ml, W - mr
    y0, y1 = mt, H - mb
    xmax = 1300.0
    ylo, yhi = 5.0, 23.5

    def X(v):
        return x0 + v / xmax * (x1 - x0)

    def Y(v):
        return y1 - (v - ylo) / (yhi - ylo) * (y1 - y0)

    s = []
    # gridlines
    for t in (5, 10, 15, 20):
        s.append(f'<line class="tick-line" x1="{x0}" y1="{Y(t):.1f}" x2="{x1}" y2="{Y(t):.1f}"/>')
        s.append(f'<text class="tick-label" x="{x0 - 10}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    for t in (0, 250, 500, 750, 1000, 1250):
        s.append(f'<text class="tick-label" x="{X(t):.1f}" y="{y1 + 20}" text-anchor="middle">{t}</text>')
    s.append(f'<line class="axis-line" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')

    # reference line: the best off-the-shelf agent, which needs no training at all
    ry = Y(NEMOTRON_MLE)
    s.append(f'<line class="ref-line" x1="{x0}" y1="{ry:.1f}" x2="{x1 - 4}" y2="{ry:.1f}"/>')
    s.append(f'<text class="ref-label" x="{x0 + 6}" y="{ry - 7:.1f}">Nemotron-120B-A12B, no post-training ({NEMOTRON_MLE})</text>')

    # connectors: same scale, GRPO -> ours (down in cost, up in score)
    for scale in ("4B", "9B"):
        g = next(r for r in AUTORESEARCH if r[4] == scale and r[5] == "real")
        o = next(r for r in AUTORESEARCH if r[4] == scale and r[5] == "ours")
        gx, gy = X(g[1]), Y(g[2])
        ox, oy = X(o[1]), Y(o[2])
        cx, cy = (gx + ox) / 2, min(gy, oy) - 34
        s.append(
            f'<path class="connector" marker-end="url(#arw)" '
            f'd="M {gx - 13:.1f} {gy - 5:.1f} Q {cx:.1f} {cy:.1f} {ox + 15:.1f} {oy - 6:.1f}"/>'
        )

    # points
    for name, gh, mle, _ds, scale, series in AUTORESEARCH:
        if gh is None:
            continue
        cx, cy = X(gh), Y(mle)
        col = SERIES_COLOR[series]
        s.append(
            f'<circle class="mark" cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{col}" '
            f'stroke="#fff" stroke-width="2"/>'
        )
        if series == "wm":
            s.append(f'<text class="pt-label" x="{cx + 13:.1f}" y="{cy + 4:.1f}" text-anchor="start">{scale}</text>')
        else:
            dy = -15 if series == "ours" else 21
            s.append(f'<text class="pt-label" x="{cx:.1f}" y="{cy + dy:.1f}" text-anchor="middle">{scale}</text>')
        s.append(
            f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="16" '
            f'data-name="{esc(name)}" data-color="{col}" '
            f'data-val="{fmt(mle)} percentile &#183; {gh} GPU-hours"/>'
        )

    # axis titles and a reading hint
    s.append(f'<text class="axis-title" x="{(x0 + x1) / 2:.0f}" y="{H - 16}" text-anchor="middle">Training compute (GPU-hours)</text>')
    s.append(f'<text class="axis-title" transform="translate(16,{(y0 + y1) / 2:.0f}) rotate(-90)" text-anchor="middle">MLE-Dojo (test) percentile</text>')
    s.append(f'<text class="hint-label" x="{x1}" y="{y0 + 10}" text-anchor="end">&#8598; cheaper and better</text>')

    body = "\n      ".join(s)
    return f"""<figure class="chart reveal">
  <div class="chart-head">
    <p class="chart-title">Compute against capability on held-out ML research tasks</p>
    <p class="chart-sub">Up and to the left is better. At both scales WMRL sits above the real-environment run it replaces, on roughly a third of the compute.</p>
  </div>
  {legend([(SERIES_NAME['real'], REAL), (SERIES_NAME['wm'], WM), (SERIES_NAME['ours'], OURS)])}
  <div class="plotwrap"><svg class="plot" viewBox="0 0 {W} {H}" role="img" aria-label="Scatter plot of training compute in GPU-hours against MLE-Dojo held-out percentile. WMRL reaches 16.4 at 4B and 21.6 at 9B using 286 and 349 GPU-hours, while real-environment GRPO reaches 15.2 and 18.8 using 883 and 1174 GPU-hours.">
    <defs>
      <marker id="arw" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1 L 9 5 L 0 9 z" fill="#c3c9d2"/>
      </marker>
    </defs>
      {body}
  </svg></div>
  <figcaption>Every trained point uses the same data split, scaffold, and hyperparameters. The horizontal rule is the strongest off-the-shelf agent in the comparison, which is 13&#215; larger than our 9B agent and does no post-training at all.</figcaption>
</figure>"""


# ---------------------------------------------------------------- chart B

def chart_autoresearch_bars():
    W, H = 780, 408
    ml, mr, mt, mb = 46, 14, 46, 52
    gap = 62
    pw = (W - ml - mr - gap) / 2
    y0, y1 = mt, H - mb
    ymax = 40.0

    def Y(v):
        return y1 - v / ymax * (y1 - y0)

    order = [("base", 2), ("real", 3), ("wm", 4), ("ours", 6)]  # (series, tuple index)
    bw, bgap = 19, 2
    block = len(order) * bw + (len(order) - 1) * bgap

    s = []
    panels = [("MLE-Dojo (test)", 2), ("DSBench", 3)]

    for pi, (ptitle, idx) in enumerate(panels):
        px0 = ml + pi * (pw + gap)
        px1 = px0 + pw
        s.append(f'<text class="group-label" x="{px0:.0f}" y="{mt - 20}" style="font-size:13.5px;fill:#14171c">{esc(ptitle)}</text>')

        for t in (0, 10, 20, 30, 40):
            s.append(f'<line class="tick-line" x1="{px0}" y1="{Y(t):.1f}" x2="{px1:.1f}" y2="{Y(t):.1f}"/>')
            if pi == 0:
                s.append(f'<text class="tick-label" x="{px0 - 10}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
        s.append(f'<line class="axis-line" x1="{px0}" y1="{y1}" x2="{px1:.1f}" y2="{y1}"/>')

        for gi, scale in enumerate(("4B", "9B")):
            gx = px0 + gi * (pw / 2)
            bx = gx + (pw / 2 - block) / 2
            for si, (series, _ti) in enumerate(order):
                row = next(r for r in AUTORESEARCH if r[4] == scale and r[5] == series)
                v = row[idx]
                x = bx + si * (bw + bgap)
                yv = Y(v)
                col = SERIES_COLOR[series]
                s.append(
                    f'<path class="mark" d="M {x:.1f} {y1} L {x:.1f} {yv + 4:.1f} '
                    f'Q {x:.1f} {yv:.1f} {x + 4:.1f} {yv:.1f} L {x + bw - 4:.1f} {yv:.1f} '
                    f'Q {x + bw:.1f} {yv:.1f} {x + bw:.1f} {yv + 4:.1f} L {x + bw:.1f} {y1} Z" '
                    f'fill="{col}"/>'
                )
                cls = "val-label hi" if series == "ours" else "val-label"
                s.append(f'<text class="{cls}" x="{x + bw / 2:.1f}" y="{yv - 6:.1f}" text-anchor="middle">{fmt(v)}</text>')
                s.append(
                    f'<rect class="hit" x="{x:.1f}" y="{y0}" width="{bw}" height="{y1 - y0:.1f}" '
                    f'data-name="{esc(row[0])}" data-color="{col}" '
                    f'data-val="{esc(ptitle)} {fmt(v)} percentile"/>'
                )
            s.append(f'<text class="group-label" x="{gx + pw / 4:.1f}" y="{y1 + 21}" text-anchor="middle">{scale} agent</text>')

    s.append(f'<text class="axis-title" transform="translate(13,{(y0 + y1) / 2:.0f}) rotate(-90)" text-anchor="middle">Leaderboard percentile</text>')

    body = "\n      ".join(s)
    return f"""<figure class="chart reveal">
  <div class="chart-head">
    <p class="chart-title">Held-out ML research benchmarks, by agent scale</p>
    <p class="chart-sub">The pure world model alone gives back most of the gain. Adding the two corrections carries WMRL past the real-environment run on both benchmarks and both scales.</p>
  </div>
  {legend([(SERIES_NAME['base'], CTX), (SERIES_NAME['real'], REAL), (SERIES_NAME['wm'], WM), (SERIES_NAME['ours'], OURS)])}
  <div class="plotwrap"><svg class="plot" viewBox="0 0 {W} {H}" role="img" aria-label="Grouped bar chart of leaderboard percentile on MLE-Dojo and DSBench for the untrained base, real-environment GRPO, pure world model, and WMRL, at 4B and 9B scale. Full values are in the table below.">
      {body}
  </svg></div>
  <figcaption>DSBench is disjoint from the training set, and the MLE-Dojo test split holds out tasks never trained on. Bars start at zero; exact values are in the table below.</figcaption>
</figure>"""


# ---------------------------------------------------------------- chart C

def chart_vla_dots():
    W, H = 780, 296
    ml, mr, mt, mb = 168, 96, 46, 46
    x0, x1 = ml, W - mr
    xlo, xhi = 36.9, 41.9

    def X(v):
        return x0 + (v - xlo) / (xhi - xlo) * (x1 - x0)

    rows_y = [mt + 26, mt + 96, mt + 166]
    s = []

    for t in (37, 38, 39, 40, 41):
        s.append(f'<line class="tick-line" x1="{X(t):.1f}" y1="{mt - 8}" x2="{X(t):.1f}" y2="{H - mb + 4}"/>')
        s.append(f'<text class="tick-label" x="{X(t):.1f}" y="{H - mb + 22}" text-anchor="middle">{t}</text>')

    for ri, (label, sft, grpo, wm, ours) in enumerate(VLA):
        y = rows_y[ri]
        s.append(f'<text class="group-label" x="{ml - 20}" y="{y + 4}" text-anchor="end">{esc(label)}</text>')
        # connector spanning the row's span, drawn under the dots
        s.append(f'<line class="connector" x1="{X(sft):.1f}" y1="{y}" x2="{X(ours):.1f}" y2="{y}"/>')

        pts = [
            ("MiniVLA-1B-SFT", sft, CTX),
            ("MiniVLA-1B-GRPO", grpo, REAL),
            ("MiniVLA-1B-WM", wm, WM),
            ("MiniVLA-1B-Ours", ours, OURS),
        ]
        for name, v, col in pts:
            cx = X(v)
            s.append(f'<circle class="mark" cx="{cx:.1f}" cy="{y}" r="6.5" fill="{col}" stroke="#fff" stroke-width="2"/>')
            s.append(
                f'<circle class="hit" cx="{cx:.1f}" cy="{y}" r="15" '
                f'data-name="{esc(name)}" data-color="{col}" '
                f'data-val="{esc(label)} {fmt(v)}% success"/>'
            )
        # label only the two ends of the story
        s.append(f'<text class="val-label" x="{X(sft):.1f}" y="{y - 15}" text-anchor="middle">{fmt(sft)}</text>')
        s.append(f'<text class="val-label hi" x="{X(ours):.1f}" y="{y - 15}" text-anchor="middle">{fmt(ours)}</text>')
        s.append(
            f'<text class="pt-label" x="{x1 + 18}" y="{y + 4}" style="fill:#196b24">'
            f'&#43;{ours - sft:.1f} over SFT</text>'
        )

    s.append(f'<text class="axis-title" x="{(x0 + x1) / 2:.0f}" y="{H - 8}" text-anchor="middle">LIBERO-Long success rate (%), average over eight rollouts</text>')

    body = "\n      ".join(s)
    return f"""<figure class="chart reveal">
  <div class="chart-head">
    <p class="chart-title">Embodied manipulation on LIBERO-Long</p>
    <p class="chart-sub">Each signal on its own barely moves the SFT policy. Combining them under the same two corrections gives the largest gain, and the widest margin is on unseen initial states.</p>
  </div>
  {legend([("MiniVLA-1B-SFT", CTX), ("RL in real environment (GRPO)", REAL), ("RL with pure world model", WM), ("WMRL (ours)", OURS)])}
  <div class="plotwrap"><svg class="plot" viewBox="0 0 {W} {H}" role="img" aria-label="Dot plot of LIBERO-Long success rate for SFT, real-environment GRPO, pure world model, and WMRL across in-domain, out-of-distribution, and overall initial states. WMRL reaches 41.2 percent in all three, between 3.7 and 3.9 points above SFT.">
      {body}
  </svg></div>
  <figcaption>All reinforcement-learning rows start from the same SFT checkpoint, so the spread within a row is attributable to the reward signal. The axis starts at 36 to resolve the spread; absolute values including the untrained policy are in the table below.</figcaption>
</figure>"""


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
    html = splice(html, "tradeoff", chart_tradeoff())
    html = splice(html, "autoresearch", chart_autoresearch_bars())
    html = splice(html, "vla", chart_vla_dots())
    with open(INDEX, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"rendered 3 charts into {INDEX}")


if __name__ == "__main__":
    main()
