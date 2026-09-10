# Front-end handoff

Notes for anyone refining the visual design of this page. The layout and styling
are open to a full redesign. The items under "Do not break" are the ones that
will silently produce a wrong or broken page if changed.

## Where things are

Repo root: `/Users/yxy/Desktop/research/autoresearch/WMRL`
Remote: `https://github.com/xiyuanyang45/WMRL`
Live: `https://xiyuanyang45.github.io/WMRL/`

| Path | What it is |
|---|---|
| `docs/index.html` | The whole page. 756 lines, no framework, no build step. |
| `docs/style.css` | All styling. 557 lines, plain CSS, custom properties at the top. |
| `docs/app.js` | Sticky nav, scroll reveal, chart tooltips. Progressive enhancement only. |
| `docs/tools/make_charts.py` | Generates the four result charts as inline SVG. |
| `docs/static/` | Figures lifted from the paper, plus the two affiliation logos. |
| `docs/.nojekyll` | Stops GitHub from running Jekyll over the directory. Leave it. |

## Running it

```bash
python3 -m http.server 8321 --directory docs
```

Then open `http://localhost:8321`. There is nothing to install or compile.

## Deploying

GitHub Pages serves the `main` branch's `/docs` directory. Push to `main` and it
rebuilds in about 30 seconds. Nothing else is wired up.

## The charts

The four `<svg>` blocks in `index.html` are generated, not hand-written. Each one
lives between a marker pair:

```html
<!--CHART:tradeoff-->     ... <!--/CHART:tradeoff-->
<!--CHART:autoresearch--> ... <!--/CHART:autoresearch-->
<!--CHART:ablation-->     ... <!--/CHART:ablation-->
<!--CHART:vla-->          ... <!--/CHART:vla-->
```

Editing the SVG inside a marker pair works, but the next run of the generator
overwrites it. To change a chart durably, edit `make_charts.py` and re-run:

```bash
python3 docs/tools/make_charts.py
```

It rewrites each block in place. The generator only emits geometry and class
names; every color, font size, and stroke weight comes from the `.tick-line`,
`.val-label`, `.panel-title`, etc. rules in `style.css`, so most visual tuning can
happen in CSS alone.

Replacing the charts with a JS charting library is fine, with one caveat: they
currently render without JavaScript, which matters because this page is a paper
artifact people will save and print.

## Do not break

**The numbers.** Every value in the charts and tables comes from the paper
(arXiv:2608.12564), Tables 1 to 3. The chart data lives in one block at the top
of `make_charts.py`; the table values are inline in `index.html`. They must stay
in sync with each other and with the paper. Do not round, re-derive, or
"clean up" a value.

**Bar charts start at zero.** The two bar charts do; the scatter and the dot plot
use clipped axes deliberately, because no area encodes the value there. Do not
clip a bar axis to make a difference look larger.

**Light theme only.** `fig1.png`, `fig2.png`, and both logos are transparent PNGs
authored for a white background. A dark mode would render them as dark-on-dark.
If you want dark mode, the paper figures need re-exporting first.

**The categorical palette.** Blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`,
plus a de-emphasis gray. These are slots 1 to 3 of a palette validated for
colorblind separation on a white surface (worst all-pairs CVD ΔE 9.2, worst
normal-vision ΔE 24.0). Orange is always WMRL. If you re-color, re-validate
rather than picking by eye, and keep the assignment stable across all four
charts.

**Author names must not break mid-name.** Each is wrapped in `.au`
(`display:inline-block; white-space:nowrap`). Wrapping happens between names.

**One shared measure.** Prose, figures, charts, and tables all span the same
left and right edge (the `.wrap` container). An earlier version capped prose
narrower than the figures and it read as broken. Keep the edges aligned.

**Tables scroll rather than squeeze.** `.tablewrap` is `overflow-x:auto`, and
below 780px the charts do the same via `.plotwrap`. The page body must never
scroll horizontally.

## Known weak spots

Worth attention in a redesign:

- The overall look is competent but plain. The hero is the only part with any
  visual identity.
- The four charts sit in near-identical bordered cards and read as a wall of
  similar boxes when scrolled past.
- The two result tables are long and dense. They exist partly as the
  accessible "table view" for the charts, so they should stay in some form,
  but they could collapse, or become progressive disclosure.
- The teaser figure and the method figure are both full-width PNGs from the
  paper and cannot be restyled, only framed better.
- The `.strip` of four LIBERO frames is a plain grid and does nothing to convey
  that it is a rollout over time.
