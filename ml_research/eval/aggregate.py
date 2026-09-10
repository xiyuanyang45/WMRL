#!/usr/bin/env python3
"""Aggregate the 9 held-out OOD eval jobs' scores.json (S3 runs/ood_eval/<EXP>_step<STEP>/) into a table + figure
into multi_node_rl/analysis/. Run AFTER the jobs finish:
    python aggregate_ood.py

Primary figure: held-out OOD mean reward vs TRAINING STEP, one line per run (3 runs x {20,40,60}).
Secondary (best-effort): the same y vs CUMULATIVE REAL SANDBOX GRADES at that step — derived from each run's
training log on S3 (TIMELINE.jsonl / env_server grade counters) if available; gracefully skipped if not derivable.

No GPU, no model load — pure S3 read + matplotlib. Writes:
    analysis/ood_held_out_reward_vs_step.png
    analysis/ood_held_out_table.md       (per-comp + overall mean reward / valid-rate / n, per cell)
    analysis/ood_scores.json             (merged raw scores for downstream use)
"""
import os, json, io
import boto3

S3_BKT = "${WMRL_STORE_BUCKET}"
S3_PFX = "wmrl"
ANALYSIS = "/home/sagemaker-user/xiyuan_work_dir/auto_research/multi_node_rl/analysis"
EXPS = ["verl_4b_k4_v2", "verl_4b_hybrid_async_se4_long", "verl_4b_hybrid_async_sbxw2_long"]
STEPS = [20, 40, 60]
TASKS = ["random-acts-of-pizza", "tabular-playground-series-dec-2021", "new-york-city-taxi-fare-prediction",
         "dogs-vs-cats-redux-kernels-edition", "dog-breed-identification"]

s3 = boto3.client("s3")


def s3_get_json(key):
    try:
        return json.loads(s3.get_object(Bucket=S3_BKT, Key=key)["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        if "NoSuchKey" in str(e) or "404" in str(e):
            return None
        raise


def s3_get_text(key):
    try:
        return s3.get_object(Bucket=S3_BKT, Key=key)["Body"].read().decode(errors="ignore")
    except Exception:
        return None


def load_all():
    """Return {(exp, step): scores_dict}. Missing cells are skipped (job not done) with a warning."""
    out = {}
    for exp in EXPS:
        for step in STEPS:
            tag = f"{exp}_step{step}"
            key = f"{S3_PFX}/runs/ood_eval/{tag}/scores.json"
            sc = s3_get_json(key)
            if sc is None:
                print(f"[aggregate] MISSING scores.json for {tag} (job not finished?)")
                continue
            out[(exp, step)] = sc
    return out


def cumulative_grades(exp, step):
    """Best-effort: cumulative REAL sandbox grades at `step` from the training run's S3 logs. The disagg sandbox
    grade count isn't logged as a single counter, so we approximate via TIMELINE.jsonl 'staleness/produced' if
    present, else None (the secondary panel is skipped). Returns int or None."""
    tl = s3_get_text(f"{S3_PFX}/runs/{exp}/TIMELINE.jsonl")
    if not tl:
        return None
    rows = [json.loads(l) for l in tl.splitlines() if l.strip()]
    # find the timeline row at/just before this training step; use total_generated as a proxy if present.
    best = None
    for r in rows:
        try:
            s = int(r.get("step") or 0)
        except Exception:
            continue
        if s <= step:
            best = r
    if not best:
        return None
    for k in ("total_generated", "produced"):
        if best.get(k) not in (None, "?"):
            try:
                return int(best[k])
            except Exception:
                pass
    return None


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    data = load_all()
    json.dump({f"{e}_step{s}": v for (e, s), v in data.items()},
              open(f"{ANALYSIS}/ood_scores.json", "w"), indent=2)

    # ---- table ----
    lines = ["# Held-out OOD eval — per-cell scores", "",
             "Mean reward = mean traj_reward (best position-score over the trajectory's K turns); valid-rate = "
             "fraction of trajs with >=1 `valid` turn; n = trajs (= G samples x 5 comps).", ""]
    for exp in EXPS:
        lines.append(f"## {exp}")
        lines.append("")
        lines.append("| step | overall reward | valid-rate | n | " + " | ".join(t[:18] for t in TASKS) + " |")
        lines.append("|---|---|---|---|" + "---|" * len(TASKS))
        for step in STEPS:
            sc = data.get((exp, step))
            if sc is None:
                lines.append(f"| {step} | (missing) | | | " + " | ".join("" for _ in TASKS) + " |")
                continue
            ov = sc["overall"]
            cells = []
            for t in TASKS:
                ts = sc["tasks"].get(t)
                cells.append(f"{ts['mean_reward']:.3f} (v{ts['valid_rate']:.2f})" if ts else "-")
            lines.append(f"| {step} | {ov['mean_reward']:.3f} | {ov['valid_rate']:.2f} | {ov['n']} | "
                         + " | ".join(cells) + " |")
        lines.append("")
    open(f"{ANALYSIS}/ood_held_out_table.md", "w").write("\n".join(lines))
    print(f"[aggregate] wrote {ANALYSIS}/ood_held_out_table.md")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have_cum = {}
    for exp in EXPS:
        for step in STEPS:
            c = cumulative_grades(exp, step)
            if c is not None:
                have_cum[(exp, step)] = c
    npanels = 2 if have_cum else 1
    fig, axes = plt.subplots(1, npanels, figsize=(7 * npanels, 5), squeeze=False)
    ax = axes[0][0]
    colors = {"verl_4b_k4_v2": "C0", "verl_4b_hybrid_async_se4_long": "C1",
              "verl_4b_hybrid_async_sbxw2_long": "C2"}
    for exp in EXPS:
        xs, ys, vr = [], [], []
        for step in STEPS:
            sc = data.get((exp, step))
            if sc is None:
                continue
            xs.append(step); ys.append(sc["overall"]["mean_reward"]); vr.append(sc["overall"]["valid_rate"])
        if xs:
            ax.plot(xs, ys, "o-", color=colors[exp], label=exp)
            for x, y, v in zip(xs, ys, vr):
                ax.annotate(f"v{v:.2f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("training step"); ax.set_ylabel("held-out OOD mean reward")
    ax.set_title("Held-out OOD reward vs training step\n(5 unseen comps, G=6, K=4, GPU grading)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    if have_cum:
        ax2 = axes[0][1]
        for exp in EXPS:
            xs, ys = [], []
            for step in STEPS:
                if (exp, step) in have_cum and data.get((exp, step)):
                    xs.append(have_cum[(exp, step)]); ys.append(data[(exp, step)]["overall"]["mean_reward"])
            if xs:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                ax2.plot([xs[i] for i in order], [ys[i] for i in order], "o-", color=colors[exp], label=exp)
        ax2.set_xlabel("cumulative samples generated at step (proxy for real grades)")
        ax2.set_ylabel("held-out OOD mean reward")
        ax2.set_title("Held-out OOD reward vs cumulative training rollouts")
        ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)

    fig.tight_layout()
    out = f"{ANALYSIS}/ood_held_out_reward_vs_step.png"
    fig.savefig(out, dpi=130)
    print(f"[aggregate] wrote {out}")


if __name__ == "__main__":
    main()
