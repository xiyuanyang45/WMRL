#!/usr/bin/env python3
"""Assemble curated_fewshot.json from the agent-selected representative examples.
Re-pulls VERBATIM code+traceback from the case pools by a unique substring (no transcription error);
error_line located by matching the traceback source-text in the code; reason hand-crafted (crisp)."""
import json, re

def load(p): return json.load(open(f"/tmp/wm_data/cases/{p}.json"))

def line_of(fb, code):
    # prefer the deepest USER-SCRIPT frame (/tmp/*.py); library-internal frames won't match the code
    user = re.findall(r'File "/tmp/[^"]+\.py", line \d+[^\n]*\n\s*([^\n]+)', fb or "")
    allf = re.findall(r'File "[^"]+", line \d+[^\n]*\n\s*([^\n]+)', fb or "")
    for src in [s.strip() for s in (user[::-1] + allf[::-1])]:
        for i, ln in enumerate((code or "").splitlines(), 1):
            if src and src in ln: return i
    return 0

def find(pool, substr):
    hits = [e for e in load(pool) if substr in (e.get("code") or "")]
    assert len(hits) == 1, f"{pool}:{substr!r} matched {len(hits)} (need exactly 1)"
    return hits[0]

# check -> (pool, unique-substring, crisp reason)
SEL = {
 "imports_names": ("name_key", "y_text = train_df['comment_text'].values",
    "TfidfVectorizer is used but never imported (only os/pandas/numpy are imported); the name is undefined at first use."),
 "api_calls": ("type_attr", "LinearSVC(C=100, max_iter=20000",
    "LinearSVC has no `solver` parameter (confused with LogisticRegression); the hallucinated kwarg is rejected by __init__."),
 "data_refs": ("file_import", 'DATA_DIR = "DATA_DIR"',
    "DATA_DIR is the literal string \"DATA_DIR\" instead of os.environ['DATA_DIR'], so read_csv looks for 'DATA_DIR/train.csv' which doesn't exist."),
 "shape_dtype": ("valueerror", "['formation_energy_ev_natom', 'bandgap_energy_ev']].values",
    "A 2-column (multi-target) y is fed to a single-output GradientBoostingRegressor, which requires a 1d target."),
 "compute_budget": ("timeout", "def string_to_int",
    "GradientBoostingClassifier (n_estimators=100, max_depth=5) is fit on ~900k rows on CPU; sklearn GBM is serial and exceeds the 600s cap."),
 "submission": ("ran_no_sub", "sub_df.to_csv('/kaggle/working/submission.csv'",
    "Code runs fine but writes to the hardcoded '/kaggle/working/submission.csv' instead of SUBMISSION_PATH, so the grader finds no file."),
 "other_runtime": ("runtime_other", "analyzer='whitespace'",
    "TfidfVectorizer(analyzer='whitespace') — 'whitespace' is not a valid analyzer (must be 'word'/'char'/'char_wb'/callable); sklearn raises InvalidParameterError at fit."),
}
QUAL = {
 "low": ("quality", "'bandgap_energy_ev': preds",
    "Nomad has two targets but this writes the SAME single-model prediction to both columns, so the bandgap column is garbage -> near-zero rank."),
 "high": ("quality", "model_fg_1 = GradientBoostingRegressor",
    "Trains two separate GradientBoostingRegressors (one per target) on scaled features -> properly models both targets; matched estimator for small tabular."),
}

cur = {}
for check, (pool, sub, reason) in SEL.items():
    e = find(pool, sub); code = e["code"]; fb = e.get("feedback_raw","") or ""
    et = e.get("exc_type") if check != "submission" else ""
    cur[check] = {"code": code, "error_type": et, "error_line": line_of(fb, code),
                  "reason": reason, "env_feedback": fb.strip()[:500]}
cur["quality"] = {}
for b,(pool,sub,reason) in QUAL.items():
    e = find(pool, sub)
    cur["quality"][b] = {"code": e["code"], "pos": e.get("pos"), "reason": reason}

json.dump(cur, open("/home/sagemaker-user/xiyuan_work_dir/auto_research/multi_node_rl/wm_feedback/curated_fewshot.json","w"), indent=1)
print("curated_fewshot.json assembled:")
for k in cur:
    if k=="quality": print(f"  quality: low(pos={cur['quality']['low']['pos']}) + high(pos={cur['quality']['high']['pos']})")
    else: print(f"  {k:14s} exc={cur[k]['error_type'] or '-':20s} line={cur[k]['error_line']}  codelen={len(cur[k]['code'])}")
