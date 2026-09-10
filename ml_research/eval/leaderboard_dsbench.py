#!/usr/bin/env python
"""DSBench leaderboard-percentile (LB%ile, fails=0) per metric-category, from scores.json.

Category by the task's Kaggle METRIC (from data_prep/dsbench.json tags):
  BinCls  = AUC / ROC-AUC
  MultiCls= accuracy / Cohen-kappa / F1
  Regress = RMSE / RMSLE / MAE / SMAPE / MSE / ...
  Other   = custom metric
per-task pct = valid_rate * (mean_best_pos_valid or 0). Category = mean over its tasks. DS-Avg = mean over all.

Usage: lbpct_dsbench.py <scores.json> [<scores.json> ...]   (first arg may be VALIDATE to print the category map+counts)
"""
import json, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = [l.strip() for l in open(f"{ROOT}/eval_ood/dsbench_tasks.txt") if l.strip()]
DB = json.load(open(f"{ROOT}/data_prep/dsbench.json"))


def categorize(slug):
    tags = " ".join(DB.get(slug, {}).get("tags", [])).lower()
    # classify by the STANDARD metric first; "Custom Metric" -> Other only if no standard metric tag is present.
    if any(k in tags for k in ["roc", "auc", "area under"]):
        return "BinCls"
    if any(k in tags for k in ["rmse", "root mean squared", "mean squared", "absolute error", "smape",
                                "logarithmic error", "mean absolute", "rmsle", " mae", "gini"]):
        return "Regress"
    if any(k in tags for k in ["kappa", "accuracy", "f1", "f-score", "categorization", "log loss", "logloss"]):
        return "MultiCls"
    return "Other"


CATMAP = {t: categorize(t) for t in TASKS}
CATS = ["BinCls", "MultiCls", "Regress", "Other"]


def pct_of(ts):
    mbp = ts.get("mean_best_pos_valid")
    return ts["valid_rate"] * (mbp if mbp is not None else 0.0)


def report(path):
    d = json.load(open(path))
    tasks = d["tasks"]
    per = {t: pct_of(ts) for t, ts in tasks.items()}
    print(f"\n== {d.get('tag', path)} ==")
    allp = []
    for cat in CATS:
        names = [t for t in TASKS if CATMAP[t] == cat and t in per]
        ps = [per[t] for t in names]
        allp += ps
        print(f"  {cat:9s} ({len(ps):2d}): {100*sum(ps)/len(ps):5.1f}%" if ps else f"  {cat:9s} (0): n/a")
    print(f"  DS-Avg    ({len(allp):2d}): {100*sum(allp)/len(allp):5.1f}%")


if sys.argv[1] == "VALIDATE":
    from collections import Counter
    c = Counter(CATMAP.values())
    print("category counts:", dict(c), "| total", sum(c.values()))
    for cat in CATS:
        print(f"  {cat}: {[t for t in TASKS if CATMAP[t]==cat]}")
    sys.argv.pop(1)
for p in sys.argv[1:]:
    report(p)
