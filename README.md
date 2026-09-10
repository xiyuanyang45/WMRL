# WMRL — Scaling Automatic Research Agents via World Models

> **Project page:** https://xiyuanyang45.github.io/WMRL/
> **Paper:** [arXiv:2608.12564](https://arxiv.org/abs/2608.12564)

RL for AutoResearch agents is bottlenecked by the environment, not the model: agent
generation amortizes across trajectories through batching, while every candidate
solution needs its own isolated sandbox on real machine time. As trajectories scale,
execution dominates the training cost.

**WMRL** replaces environment execution with a world model that predicts the execution
outcome in a few forward passes, so grading scales as gracefully as generation. Because
the predicted reward is corrupted, WMRL keeps a thin stream of real execution (~10% of
groups) as an *anchor signal* and spends it on two mechanisms, one per error term:

| Mechanism | Target | What it does |
|---|---|---|
| **Online Debiasing** | the `O(B²)` bias term | fits a monotone map `f̂` (isotonic regression) on anchor score pairs and recasts world model scores through it, refit each step to track drift |
| **Inverse-Variance Denoising** | the `O(σ²)` noise term | fuses the anchor and world model gradient estimates weighted by inverse variance, attaining a variance strictly below either stream alone |

Both are proven to strictly improve the convergence guarantee.

## Results

| | GPU-hours | MLE-Dojo (test) Avg | DSBench Avg |
|---|---|---|---|
| Qwen3.5-4B-GRPO (real env) | 883 | 15.2 | 25.7 |
| **Qwen3.5-4B-WMRL** | **286** (3.1× less) | **16.4** | **28.8** |
| Qwen3.5-9B-GRPO (real env) | 1174 | 18.8 | 31.2 |
| **Qwen3.5-9B-WMRL** | **349** (3.4× less) | **21.6** | **32.8** |

Leaderboard percentile (%), higher is better; both benchmarks hold out tasks never
trained on. The post-trained 4B agent surpasses off-the-shelf Kimi-48B-A3B and the 9B
agent surpasses Nemotron-120B-A12B. The recipe also transfers to embodied VLA
post-training, lifting LIBERO-Long success by 3.8 points. Full tables are on the
[project page](https://xiyuanyang45.github.io/WMRL/).

## Code

A conceptual, community reference implementation of the two correction mechanisms is in
preparation and will be released here. The production training stack used for the paper
is not part of this release.

## Repository layout

```
docs/
  index.html         # the page, served by GitHub Pages from main /docs
  style.css
  app.js             # sticky nav, scroll reveal, chart tooltips
  static/            # figures from the paper
  tools/
    make_charts.py   # renders the three result charts as inline SVG
```

The result charts are generated, not hand-written, so every number traces back to
one table at the top of `docs/tools/make_charts.py`. After editing that table:

```bash
python3 docs/tools/make_charts.py
```

which re-splices the SVG into `docs/index.html` in place.

## Citation

```bibtex
@article{yang2026wmrl,
  title   = {Scaling Automatic Research Agents via World Models},
  author  = {Yang, Xiyuan and Sarwar, Sheikh and Cheng, Jingru and Shi, Zhan and
             Li, Duanshun and Chen, Huiyuan and Zhang, Haiyang and Fan, Xing and
             Guo, Chenlei and He, Jingrui and Liao, Zhenyu},
  journal = {arXiv preprint arXiv:2608.12564},
  year    = {2026}
}
```
