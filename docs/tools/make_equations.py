#!/usr/bin/env python3
"""Render the theory equations to SVG and write them into docs/static/.

The SVGs are committed, so this only needs re-running when the maths changes.
It needs a LaTeX install (pdflatex) and poppler (pdftocairo); pdftocairo emits
glyphs as paths, so the result carries no font dependency and scales cleanly.

    python3 docs/tools/make_equations.py

Colour roles match the rest of the page: red marks an error term we are trying
to kill, green marks the factor that divides it.
"""

import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "static"))

PRE = r"""\documentclass[border=1pt,varwidth=34cm]{standalone}
\usepackage{amsmath,amssymb}
\usepackage[T1]{fontenc}
\usepackage[sc]{mathpazo}
\usepackage{xcolor}
\definecolor{bad}{RGB}{165,49,49}
\definecolor{good}{RGB}{26,109,53}
\definecolor{ink}{RGB}{19,23,32}
\color{ink}
\begin{document}
%s
\end{document}
"""

EQ = {
    "eq-thm1": r"$\displaystyle J^\star-\mathbb{E}[J(\theta_T)]\;\le\;\varepsilon(T)\;+\;"
               r"O\big({\color{bad}M^{2}B^{2}}\big)\;+\;O\big({\color{bad}\gamma V_{WM}}\big)$",
    "eq-thm2": r"$\displaystyle J^\star-\mathbb{E}[J(\theta_T)]\;\le\;\varepsilon(T)\;+\;"
               r"\widetilde{O}\bigg(\frac{{\color{bad}M^{2}B^{2}}}{{\color{good}1+T/T_0}}\bigg)\;+\;"
               r"O\bigg(\frac{{\color{bad}\gamma V_{WM}}}{{\color{good}1+V_{WM}/V_E}}\bigg)$",
    "eq-eps":  r"$\displaystyle \varepsilon(T)=\big(1-\tfrac{\gamma\mu}{4}\big)^{T}\!\Delta_0$",
    "eq-den1": r"$\displaystyle {\color{good}1+T/T_0}$",
    "eq-den2": r"$\displaystyle {\color{good}1+V_{WM}/V_E}$",
}


def render(name, body):
    d = os.path.join("/tmp/wmrl-eq", name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "e.tex"), "w").write(PRE % body)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "e.tex"],
                       cwd=d, capture_output=True, text=True)
    if not os.path.exists(os.path.join(d, "e.pdf")):
        print(r.stdout[-1500:]); sys.exit(f"pdflatex failed for {name}")
    dst = os.path.join(OUT, name + ".svg")
    subprocess.run(["pdftocairo", "-svg", os.path.join(d, "e.pdf"), dst], check=True)
    return dst


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for k, v in EQ.items():
        p = render(k, v)
        import re
        head = open(p).read(400)
        w = re.search(r'width="([\d.]+)"', head).group(1)
        h = re.search(r'height="([\d.]+)"', head).group(1)
        print(f"  {k:10s} {float(w):7.1f} x {float(h):5.1f}  -> {os.path.relpath(p, HERE)}")
