#!/usr/bin/env python
"""Prep MLE-bench-Lite competitions into OUR task layout so the EXISTING pipeline can run them.

We reuse ONLY MLE-bench's data + official leaderboard. Prompt construction, interact loop and
sandbox stay exactly as in training — that is the user's hard requirement, so this script writes the
same on-disk shape our MLE-Dojo/DSBench tasks already have and changes nothing upstream:

    data/<task>/data/public/{description.txt, train.csv, test.csv, sample_submission.csv, ...}
    data/<task>/data/private/{test_answer.csv, public_leaderboard.csv, private_leaderboard.csv}

DESCRIPTION IS COPIED VERBATIM (description.md -> description.txt, byte-for-byte). This is the
lesson DSBench cost us: normalizing headers made the prompt LESS like training, not more. The
MLE-Dojo training set is itself a mix of `###` and plain headers, so verbatim IS the faithful path.

Leaderboard: MLE-bench ships kaggle's full leaderboard (cols scoreNullable,teamId,...,score); our
harness wants `rank,score`. We sort by score in the competition's own direction (derived from the
leaderboard itself, exactly as mlebench's Grader.is_lower_better does) and emit rank,score.

Run:  mleb-python data_prep/mlebench_prep.py --all       (or --comp <slug>)
"""
import argparse, shutil, sys
from pathlib import Path

MNR = Path("/home/sagemaker-user/xiyuan_work_dir/auto_research/multi_node_rl")
MLEB_DATA = Path("/home/sagemaker-user/mleb-baseline/data")
MLEB_PKG = Path("/home/sagemaker-user/mleb-baseline/mle-bench/mlebench/competitions")

# FINAL SET = 19 (user, 2026-07-26). 22 Lite minus:
#   siim-isic-melanoma-classification  — obtainable but ~100GB, deferred
#   the-icml-2013-whale-challenge…     — Kaggle-CLOSED (403 + submissions_disabled), unobtainable
#   detecting-insults-in-social-…      — Kaggle-CLOSED; we hold a May-2026 snapshot but a reader
#                                        could not re-download it, so it is excluded for reproducibility
COMPS = [
    "aerial-cactus-identification", "aptos2019-blindness-detection", "denoising-dirty-documents",
    "dog-breed-identification", "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection", "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification", "mlsp-2013-birds", "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors", "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza", "ranzcr-clip-catheter-line-classification",
    "spooky-author-identification", "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022", "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
]

# Still in our RL TRAINING set — disclosed in the appendix, not hidden.
TRAIN_OVERLAP = {"aerial-cactus-identification", "jigsaw-toxic-comment-classification-challenge",
                 "leaf-classification", "nomad2018-predict-transparent-conductors",
                 "spooky-author-identification"}


def prep_one(comp: str, force: bool = False) -> dict:
    import pandas as pd
    src = MLEB_DATA / comp / "prepared"
    if not (src / "public").is_dir():
        return {"comp": comp, "ok": False, "why": "no prepared/public"}
    # ⚠ NAMESPACE: `mleb__<comp>`, NEVER `data/<comp>`.
    # 5 of the 19 competitions ALREADY exist as TRAINING task dirs (they are in our RL train set),
    # and their description.txt is the MLE-Dojo prose (e.g. spooky: 2046 B, "### Description")
    # while MLE-bench ships a different description.md (6281 B) for the same competition.
    # Writing here would either (a) leave the 19 with two different description sources — 5 MLE-Dojo
    # vs 14 MLE-bench, i.e. an internally inconsistent benchmark — or (b) corrupt training data by
    # overwriting it. A separate namespace gives all 19 the SAME (MLE-bench) description and leaves
    # every training dir byte-untouched.
    # (Verified safe to reuse the split itself: our existing test_answer.csv for spooky is md5-identical
    #  to MLE-bench's prepared/private/test.csv, so the two harnesses agree on the held-out set.)
    dst = MNR / "data" / f"mleb__{comp}" / "data"
    if dst.exists() and not force:
        return {"comp": comp, "ok": True, "why": "already prepped (use --force)"}
    pub_o, pri_o = dst / "public", dst / "private"
    pub_o.mkdir(parents=True, exist_ok=True); pri_o.mkdir(parents=True, exist_ok=True)

    # ---- public: everything MLE-bench exposes to the agent, description renamed VERBATIM ----
    n_pub = 0
    for p in sorted((src / "public").iterdir()):
        if p.is_dir():
            if not (pub_o / p.name).exists():
                (pub_o / p.name).symlink_to(p)      # image dirs can be tens of GB — link, don't copy
            n_pub += 1
            continue
        out = pub_o / ("description.txt" if p.name == "description.md" else p.name)
        if not out.exists():
            shutil.copy2(p, out)
        n_pub += 1
    if not (pub_o / "description.txt").exists():
        return {"comp": comp, "ok": False, "why": "no description.md in prepared/public"}
    # Some competitions ship the data zipped and/or with non-canonical names
    # (denoising-dirty-documents & random-acts-of-pizza: `sampleSubmission.csv`;
    #  text-normalization-*: `en_sample_submission_2.csv.zip`).
    # UNZIP IN PLACE KEEPING THE ORIGINAL NAMES — the description prose refers to those names, and
    # renaming would leave the agent reading about files it cannot see. Then ADD a canonical
    # `sample_submission.csv` copy purely so build_overview can derive the SUBMISSION CONTRACT
    # (its dedup marks the duplicate as "same schema as ...", so the prompt does not double-count).
    import zipfile
    for z in sorted(pub_o.glob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(pub_o)
            z.unlink()
        except Exception as e:
            return {"comp": comp, "ok": False, "why": f"unzip {z.name} failed: {str(e)[:80]}"}
    if not (pub_o / "sample_submission.csv").exists():
        alt = [q for q in sorted(pub_o.iterdir())
               if q.suffix == ".csv" and "sample" in q.name.lower() and "submission" in q.name.lower()]
        if not alt:
            return {"comp": comp, "ok": False,
                    "why": f"no sample-submission file (saw {[q.name for q in pub_o.iterdir()]})"}
        shutil.copy2(alt[0], pub_o / "sample_submission.csv")

    # ---- private: the held-out answers ----
    # MLE-bench names the held-out labels `test.csv` for most competitions but `answers.csv` for some
    # (text-normalization-*), whose private dir ALSO carries a sample_submission.csv — so "the only
    # csv" is not a safe rule. Resolve by explicit preference, and only then fall back.
    ans = next((p for p in ((src / "private" / "test.csv"), (src / "private" / "answers.csv"))
                if p.is_file()), None)
    if ans is None:
        cands = [q for q in sorted((src / "private").iterdir())
                 if q.suffix == ".csv" and "sample" not in q.name.lower()]
        if len(cands) != 1:
            return {"comp": comp, "ok": False, "why": f"ambiguous private answers: {[c.name for c in cands]}"}
        ans = cands[0]
    if not (pri_o / "test_answer.csv").exists():
        shutil.copy2(ans, pri_o / "test_answer.csv")

    # ---- leaderboard -> rank,score (direction inferred from the leaderboard itself) ----
    lb_src = MLEB_PKG / comp / "leaderboard.csv"
    if not lb_src.is_file():
        return {"comp": comp, "ok": False, "why": "no leaderboard.csv in mlebench package"}
    lb = pd.read_csv(lb_src)
    s = lb["score"].dropna()
    lower_better = bool(s.iloc[0] < s.iloc[-1])          # same rule as mlebench Grader.is_lower_better
    s = s.sort_values(ascending=lower_better).reset_index(drop=True)
    out_lb = pd.DataFrame({"rank": range(1, len(s) + 1), "score": s.values})
    for name in ("public_leaderboard.csv", "private_leaderboard.csv"):
        out_lb.to_csv(pri_o / name, index=False)

    return {"comp": comp, "ok": True, "n_public": n_pub, "n_leaderboard": len(out_lb),
            "lower_better": lower_better, "in_train": comp in TRAIN_OVERLAP}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--comp")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    todo = COMPS if a.all else ([a.comp] if a.comp else [])
    if not todo:
        sys.exit("give --all or --comp <slug>")
    ok = bad = 0
    for c in todo:
        r = prep_one(c, a.force)
        ok, bad = (ok + 1, bad) if r["ok"] else (ok, bad + 1)
        print(("OK   " if r["ok"] else "FAIL ") + f"{c:<48} " +
              " ".join(f"{k}={v}" for k, v in r.items() if k not in ("comp", "ok")))
    print(f"\nprepped {ok} ok, {bad} failed, of {len(todo)}")
