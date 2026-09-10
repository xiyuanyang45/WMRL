# README figures

These are the paper figures flattened onto an opaque white background.

The originals in `docs/static/` are transparent PNGs, which is correct for the
project page: it is a white page, and transparency lets the figures sit on it
without a seam. GitHub is different. It renders README images straight onto its
own page background, which is near-black in dark mode, and these figures are
mostly black text. On dark mode the transparent versions lose their panel
titles, axis labels and legends entirely.

So the README uses these instead. Regenerate them after changing a figure:

```bash
python - <<'PY'
from PIL import Image
MARGIN = 28
for src, dst in [("docs/static/fig1.png", "assets/teaser.png"),
                 ("docs/static/fig2.png", "assets/method.png")]:
    im = Image.open(src).convert("RGBA")
    canvas = Image.new("RGB", (im.width + 2*MARGIN, im.height + 2*MARGIN), "white")
    canvas.paste(im, (MARGIN, MARGIN), im)
    canvas.save(dst, optimize=True)
PY
```

The margin keeps the figure from running into whatever surrounds it.
