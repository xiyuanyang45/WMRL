#!/usr/bin/env python
"""Compute MLE-Dojo leaderboard-percentile (LB%ile, fails=0) per category from a scores.json.

per-task pct = valid_rate * mean_best_pos_valid   (mean_best_pos_valid null -> 0)   [= mean pos over ALL traj, fails=0]
category cell = simple mean of per-task pct over that category's tasks (each task = avg@G)
MLE Avg       = simple mean over all 14 tasks.

Usage: lbpct_mle.py <scores.json> [<scores.json> ...]
"""
import json, sys

CATS = {
    "tabular": ["GiveMeSomeCredit", "forest-cover-type-kernels-only", "novozymes-enzyme-stability-prediction",
                "integer-sequence-learning", "afsis-soil-properties"],
    "text":    ["quora-question-pairs", "AI4Code", "jigsaw-unintended-bias-in-toxicity-classification",
                "quora-insincere-questions-classification"],
    "image":   ["uw-madison-gi-tract-image-segmentation", "invasive-species-monitoring", "kuzushiji-recognition",
                "bengaliai-cv19", "cassava-leaf-disease-classification"],
}


def pct_of(ts):
    mbp = ts.get("mean_best_pos_valid")
    return ts["valid_rate"] * (mbp if mbp is not None else 0.0)


def report(path):
    d = json.load(open(path))
    tasks = d["tasks"]
    per = {t: pct_of(ts) for t, ts in tasks.items()}
    print(f"\n== {d.get('tag', path)} ==")
    allp = []
    for cat, names in CATS.items():
        ps = [per[t] for t in names if t in per]
        allp += ps
        print(f"  {cat:8s} ({len(ps)}): {100*sum(ps)/len(ps):5.1f}%   " +
              " ".join(f"{t.split('-')[0][:10]}={100*per[t]:.1f}" for t in names if t in per))
    print(f"  MLE-Avg  ({len(allp)}): {100*sum(allp)/len(allp):5.1f}%")


for p in sys.argv[1:]:
    report(p)
