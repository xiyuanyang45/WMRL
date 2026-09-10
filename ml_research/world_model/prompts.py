#!/usr/bin/env python3
"""Phase-2 World-Model check prompts (grounded in investigation/diagnostics.md).

The WM is the SAME base model as the RL agent (Qwen3.5-4B/9B) — no stronger critic (no-distillation).
It does NOT run code; it PREDICTS, by static analysis of (task overview + data inventory + code), the grading
outcome. Two layers:
  - DETERMINISTIC gates (compute exactly, no LLM): code-block present + compile(); see wm_aggregate.py.
  - WM-LLM checks (this file): parallel focused aspect checks, each returns strict JSON
      {"verdict": "<pass|fail>", "confidence": <0..1>, "reason": "<analysis FIRST>", "env_feedback": "<synthetic
       sandbox-style message if fail, else ''>"}
    reason precedes verdict to force analysis before judgement. env_feedback mimics the real sandbox output so the
    RL agent's next-turn observation looks identical to the real env.

Aggregation precedence (wm_aggregate.py): stage0 malformed/syntax=0.0 > any runtime check fires=0.1 >
submission fails=0.2 > valid=0.5+0.5*quality_pos.

Two designs are provided for the fidelity test (Phase 3):
  - DECOMPOSED: the per-aspect battery below (user's parallel-checks vision).
  - HOLISTIC: a single call predicting the whole outcome (baseline, to measure whether decomposition helps).
"""

# Keep in sync with agentic_core.SANDBOX_ENV_BRIEF (the agent's own advertised env).
ENV_BRIEF = (
    "Sandbox: Python 3.11 on Linux, ONE ~20GB GPU available (torch 2.12 CUDA, tensorflow 2.21 CUDA, keras 3.14 all "
    "detect it). Installed (name version): numpy 1.26, pandas 2.2, scipy 1.17, scikit-learn 1.4, xgboost 3.2, "
    "lightgbm 4.6, catboost 1.2, statsmodels 0.14, nltk 3.9, torch 2.12, torchvision 0.27, tensorflow 2.21, "
    "keras 3.14, transformers 5.12, Pillow 12.2, matplotlib 3.10, seaborn 0.13. NOT installed: opencv(cv2), jax, "
    "and anything else not listed. Execution is killed at 600s. The code reads data from os.environ['DATA_DIR'] and "
    "MUST write predictions to os.environ['SUBMISSION_PATH'] (a submission.csv matching sample_submission.csv)."
)

SYS = (
    "You are a precise execution-and-grading SIMULATOR for Kaggle/MLE Python solutions. You do NOT execute code. "
    "You predict — by careful static analysis — exactly what the real sandbox would report. " + ENV_BRIEF +
    "\nYou are given the TASK OVERVIEW (with a DATA INVENTORY: real files, columns, row counts, dtypes, and the "
    "sample_submission format) and the agent's CODE (its lines are 1-indexed as shown). Analyze ONLY the failure "
    "mode this check asks about.\n"
    "CRITICAL — DEFAULT TO PASS. Real submissions usually RUN FINE; most code does NOT trigger this check's failure "
    "mode. Return verdict='fail' ONLY when you can cite a SPECIFIC line and a CONCRETE, CERTAIN reason it raises this "
    "exact error. Mere suspicion / 'might' / 'could' / 'possibly' / a stylistic concern / an unverified assumption is "
    "NOT enough → PASS. A false 'fail' on working code is WORSE than a missed error. When in doubt, PASS.\n"
    "If you do fail it, be concrete: identify the EXACT offending source line, its line number, the precise "
    "Python exception class, and the real error message. NAME THE EXACT CLASS (prefer the specific one over a "
    "generic ValueError/TypeError): sklearn invalid/out-of-range PARAMETER VALUE -> 'InvalidParameterError'; a local "
    "variable used before assignment -> 'UnboundLocalError' (vs 'NameError' for a never-defined/unimported name); "
    ".predict()/.transform() before .fit() -> 'NotFittedError'; lightgbm/xgboost/catboost config errors -> "
    "'LightGBMError'/'XGBoostError'/'CatBoostError'; bad-shape/out-of-range indexing -> 'IndexError'. The agent will "
    "READ your env_feedback to fix its code next turn, so env_feedback MUST look byte-similar to the real sandbox "
    "output: a traceback ending in "
    "'<ExceptionClass>: <message>', citing the offending `File \"solution.py\", line N` and that source line.\n"
    "Keep `reason` SHORT and committal: at most 2 sentences naming the specific cause + line. Do NOT deliberate, "
    "backtrack, hedge, or write 'wait'/'let me re-check' — commit to the single most likely outcome.\n"
    "Output STRICT JSON only, no prose around it:\n"
    '{"reason": "<= 2 sentences, the specific cause>", "verdict": "<pass|fail>", '
    '"confidence": <float 0..1>, "error_type": "<exact Python exception class if fail (e.g. ValueError, KeyError), '
    'else empty>", "error_type_alt": "<the SECOND most-likely exception class if you are not certain of error_type '
    '(a real alternative the same line could raise), else empty>", '
    '"error_line": <1-indexed line number of the failing statement if fail, else 0>, '
    '"env_feedback": "<if fail: the realistic sandbox message, e.g. \'=== Code Execution Results ===\\nExecution '
    'failed: Traceback (most recent call last):\\n  File \\"solution.py\\", line N, in <module>\\n    <source '
    'line>\\n<ExceptionClass>: <message>\'; if pass: empty string>"}'
)

# ---- DECOMPOSED per-aspect checks. Each value is the check-specific instruction appended after SYS + the case. ----
CHECKS = {
"imports_names": (
    "CHECK: Will this code raise NameError / ImportError / ModuleNotFoundError before or during execution?\n"
    "Look for, in order:\n"
    "1. A name used but never bound (missing `import`, e.g. uses TfidfVectorizer/np/zipfile without importing it; "
    "used-before-assignment like read_csv(train_csv) where train_csv is never defined; identifier typos like "
    "LogicRegression).\n"
    "2. A `from pkg.sub import name` where `name` actually lives in a DIFFERENT submodule for the INSTALLED versions "
    "(e.g. LinearSVC is in sklearn.svm not sklearn.linear_model; normalize is in sklearn.preprocessing not "
    "sklearn.utils; keras preprocess_input is per-application not keras.utils).\n"
    "3. An `import X` of a package NOT in the installed list (e.g. cv2/opencv, jax), or a stdlib typo (import warning "
    "-> warnings).\n"
    "Build the set of bound names from imports+assignments+builtins; flag the FIRST unbound/mis-located/missing one. "
    "verdict=fail if any such error is certain; env_feedback = the exact NameError/ImportError/ModuleNotFoundError."
),
"api_calls": (
    "CHECK: Will this code raise TypeError / AttributeError from API misuse, for the INSTALLED library versions "
    "(scikit-learn 1.4, pandas 2.2, numpy 1.26, lightgbm 4.6, catboost 1.2, xgboost 3.2)?\n"
    "Look for: (a) a keyword argument the constructor/.fit()/function does NOT accept (hallucinated or wrong-library "
    "kwarg, e.g. LinearSVC(solver=...), LogisticRegression(classes=...), TfidfVectorizer(ngrams=...), CatBoost with "
    "XGBoost-style colsample_bytree, np.multiply(...,axis=), lgb.train(train_data=...)); (b) calling an "
    "attribute/method that does not exist on that object TYPE (e.g. .columns on a Series from df['col']; "
    "DataFrame.append which was REMOVED in pandas 2.x; a str-method typo); (c) operating on the wrong type (e.g. "
    "str + int from factorize()[1].max()+1 on a string column; list used as a dict key).\n"
    "Check each call site's kwargs against the real signature. verdict=fail if a TypeError/AttributeError is certain; "
    "env_feedback = the exact message (e.g. \"TypeError: LinearSVC.__init__() got an unexpected keyword argument "
    "'solver'\")."
),
"data_refs": (
    "CHECK: Will this code raise FileNotFoundError or KeyError(column) — i.e. does it reference a file/path/column "
    "that does not match the DATA INVENTORY?\n"
    "FileNotFound: flag DATA_DIR/SUBMISSION_PATH assigned a LITERAL string (e.g. DATA_DIR=\"DATA_DIR\") instead of "
    "os.environ; hardcoded /kaggle/input/... roots; or a subpath/filename joined+opened that is NOT in the inventory "
    "(e.g. opening DATA_DIR/'train/' when the inventory lists only train.zip).\n"
    "KeyError(column): TWO distinct cases — (i) references a column name ABSENT from the inventory's listed columns "
    "(IMPORTANT: symbolically expand any CONSTRUCTED column names — f-strings, slices, list comprehensions — and "
    "compare the resulting names against the inventory's actual columns; a constructed name that doesn't match any "
    "listed column is a KeyError); (ii) read-after-drop / read-before-create WITHIN the script (e.g. "
    "drop(columns=['id']) then df['id']; df['x']=df['x'].apply(...) before x exists) — track each DataFrame's "
    "evolving column set; the key may BE in the inventory yet still KeyError. Do NOT pass a column just because it is "
    "in the inventory — check the script's own mutation order.\n"
    "verdict=fail if a FileNotFoundError or KeyError is certain; env_feedback = the exact message."
),
"shape_dtype": (
    "CHECK: Will this code raise a ValueError from shape / dtype / label mismatch in the ML pipeline?\n"
    "Look for: (a) an unencoded string/object feature column passed to a numeric estimator with no "
    "encoding/get_dummies/astype('category') (sklearn => 'could not convert string to float'; LightGBM "
    "categorical_feature without astype('category') => 'pandas dtypes must be int, float or bool'); (b) predict_proba "
    "output (shape n_samples x n_classes) sliced on the WRONG axis when building the submission (e.g. proba.T, "
    "proba[i] inside a per-class dict) => length mismatch with the id column; (c) np.hstack/concatenate(axis=1) where "
    "one operand is a scipy SPARSE vectorizer output or a .ravel()/.flatten() 1-D array => dimension mismatch; (d) a "
    "multi-target y (>=2 target columns per the sample_submission) fed to a single-output regressor "
    "(GradientBoostingRegressor etc.) instead of MultiOutputRegressor => 'y should be a 1d array'.\n"
    "verdict=fail if a ValueError is certain; env_feedback = the exact ValueError message."
),
"compute_budget": (
    "CHECK: Will this code EXCEED the 600s execution cap (timeout) or run out of memory, on THIS dataset size (use "
    "the DATA INVENTORY row/feature counts)?\n"
    "Timeouts are CPU-compute (NOT infinite loops). Flag, weighing the estimator FAMILY against the data size: (a) "
    "sklearn GradientBoostingClassifier/Regressor on a large dataset (>=100k rows or >=10k features) — exact, "
    "single-threaded, the #1 timeout cause; (b) .toarray()/.todense() on a vectorizer output from a large text "
    "column (densifies a huge sparse matrix); (c) a neural net (torch/keras) trained WITHOUT moving the model to GPU "
    "(.cuda()/.to('cuda')/device='cuda') — CPU NN training times out; or excessive epochs; (d) per-label / per-fold "
    "re-fit loops multiplying a big fit; (e) SVC/saga-LogReg on large data. NOTE: LightGBM/XGBoost are multithreaded "
    "and usually FINISH — do NOT flag them as timeouts unless data is extreme. Also flag OOM: get_dummies/one-hot of "
    "a high-cardinality string column at large row count.\n"
    "verdict=fail if a timeout/OOM is likely; env_feedback = 'Execution failed: Process execution timed out after "
    "600s' (or the numpy _ArrayMemoryError). Be calibrated: borderline-but-probably-fine => pass with lower "
    "confidence."
),
"other_runtime": (
    "CHECK: Will this code raise ANY OTHER runtime exception NOT covered by the import/name, API-call, "
    "data-reference, shape/dtype, or compute-budget checks? This is the catch-all net for the long tail. Reason "
    "about control flow and flag the FIRST such error. Examples: NotFittedError (calling .predict()/.transform() "
    "before .fit()); AssertionError (a failed `assert` in the code); ZeroDivisionError; RecursionError; "
    "JSONDecodeError; UnicodeDecodeError; an exception raised explicitly via `raise`; IndexError/slicing not driven "
    "by an ML shape mismatch; library-specific errors (LightGBMError/XGBoostError/CatBoostError) from a bad "
    "config/value not already caught as a kwarg error. Do NOT re-flag anything the other checks own.\n"
    "verdict=fail if some other runtime exception is certain; env_feedback = the exact exception message."
),
"submission": (
    "CHECK: ASSUMING the code runs to completion without the errors checked elsewhere, will it produce a VALID "
    "submission file?\n"
    "Look for: (a) does it write to os.environ['SUBMISSION_PATH'] (pass), or to a wrong path like "
    "'/kaggle/working/submission.csv' or a self-built os.path.join(DATA_DIR,'submission.csv') (fail: no file at the "
    "expected path), or is there NO to_csv/open(...,'w') reachable at all (fail)? (b) do the submission columns "
    "match the sample_submission format (correct id column + correct prediction column names), correct row count, no "
    "obviously-NaN predictions, and is the id column the real test ids (not a broadcast constant)?\n"
    "ALSO: flag 'crash-before-write' — a to_csv that sits AFTER a fragile fit/predict inside a broad try/except: "
    "print(e); if the fit is likely to raise (per the other checks' logic), the file is never written. \n"
    "verdict=fail (=> ran_no_sub) if the submission would be missing or invalid; env_feedback = the grader message "
    "(e.g. 'SubmissionNotFoundError: No submission file found' or 'InvalidSubmissionError: ...')."
),
"quality": (
    "CHECK: ASSUMING a VALID submission, rate the solution's leaderboard QUALITY as a coarse bucket. Judge from a "
    "rubric over the CODE+TASK only — you are NOT told the real score; predict it.\n"
    "FIRST infer the task MODALITY (dense tabular / text / image) and the METRIC. Then:\n"
    "- HIGH: model family MATCHES the modality (tree-boosters LightGBM/XGB/CatBoost for dense tabular; LINEAR models "
    "on TF-IDF for TEXT — boosting LOSES on sparse text; justified NN for image) AND correct feature/categorical "
    "handling AND hyperparameter care (tuned lr/rounds/C/regularization/class_weight) AND calibrated probabilities "
    "for logloss/AUC AND no correctness bugs.\n"
    "- MID: family matches modality and runs clean, but bare defaults / shallow feature engineering / no tuning / no "
    "CV.\n- FAIR: runs but a sub-optimal family for the modality (linear on dense tabular; naive single "
    "GradientBoosting where boosting libs would win).\n- LOW: a correctness bug / degenerate or constant output / "
    "wrong family with no effort / hard labels where the metric needs probabilities.\n"
    "Decision order: bug/degenerate -> LOW; else wrong-family -> at most FAIR; else defaults -> MID; else "
    "tuned+matched+calibrated -> HIGH.\n"
    "CALIBRATION — use the FULL range, do NOT default to mid/high: HIGH is RARE (requires ALL of matched-family + "
    "tuning + calibration + no bugs — a plain default model is NOT high). Actively look for the LOW/FAIR "
    "disqualifiers first (any bug/degenerate output -> LOW; any wrong-family-for-modality or no-feature-work -> "
    "FAIR); only if none apply consider MID, and only a clearly strong solution is HIGH. Across many submissions the "
    "buckets should be spread, not clustered on mid/high.\n"
    "Output: verdict ALWAYS 'pass'; put the bucket in env_feedback as one of exactly low|fair|mid|high; reason = the "
    "modality+metric+evidence. (confidence = how sure of the bucket.)"
),
}

# ---- HOLISTIC baseline: one call predicts the whole outcome (to measure if decomposition helps). ----
HOLISTIC = (
    "Predict the FULL grading outcome of this solution. CALIBRATION — agent-written ML code FREQUENTLY fails at "
    "RUNTIME from DATA-DEPENDENT bugs that are NOT obvious from a quick read (shape/dtype/label mismatch, a column "
    "absent or dropped-then-used, NaN/inf, wrong predict_proba axis, single-output regressor on multi-target, OOM/"
    "timeout). Empirically the MAJORITY of plausible-looking submissions actually CRASH — do NOT assume it runs just "
    "because the pipeline 'looks coherent'. Predict 'valid' ONLY after you TRACE THE DATA END-TO-END against the DATA "
    "INVENTORY: the column set after each transform, the array shape entering each estimator, and dtypes — and confirm "
    "no data-dependent failure. If you cannot verify the full data flow, predict the single most likely runtime error "
    "(name the exact exception + line). Reason step by step, "
    "simulating execution in order: parse -> "
    "compile -> run (first exception wins) -> write submission -> score. Consider in sequence: missing/mis-located "
    "imports & undefined names; API/kwarg misuse for the installed versions; file-path & column references vs the "
    "DATA INVENTORY (mind read-after-drop); shape/dtype/label ValueErrors; the 600s timeout / OOM given the data "
    "size and estimator family (sklearn GradientBoosting & CPU-NN are slow; LightGBM/XGB are fast); whether a valid "
    "submission.csv is written to SUBMISSION_PATH with the right columns; and if valid, the coarse quality bucket "
    "(modality-matched family + tuning + calibration => higher).\n"
    "Output your TOP-2 most likely outcomes, ranked (most likely first). The agent sees BOTH, so prediction #2 is "
    "your HEDGE for when #1 is wrong and MUST have a DIFFERENT `status` than #1 — NEVER repeat the same status "
    "twice. Rule: if #1 is 'valid', make #2 the single most likely ERROR you'd expect IF it does not actually run "
    "(name the specific exception + line); if #1 is an error, make #2 either 'valid' or a different, distinct error. "
    "Keep `reason` SHORT (<=3 sentences), no "
    "backtracking. Each prediction's env_feedback must be byte-similar to the real sandbox: a traceback ending "
    "'<ExceptionClass>: <message>' citing `File \"solution.py\", line N` (runtime_err), or the submission block "
    "(ran_no_sub/valid). For a 'valid' prediction, env_feedback may state ONLY that a submission was written with the "
    "expected columns — do NOT FABRICATE a numeric leaderboard score (no 'Score: 0.42'): you did not run the code and "
    "the real grader computes the score. Put solution strength ONLY in quality_bucket.\n"
    "Output STRICT JSON only:\n"
    '{"reason": "<concise ordered simulation>", "predictions": [ '
    '{"status": "<malformed|syntax_err|runtime_err|ran_no_sub|valid>", "error_type": "<exc class if runtime_err '
    'else \'\'>", "error_line": <int, 0 if n/a>, "quality_bucket": "<low|fair|mid|high if valid else \'\'>", '
    '"env_feedback": "<realistic sandbox message>", "confidence": <0..1>}, '
    '{"status": "...", "error_type": "...", "error_line": <int>, "quality_bucket": "...", "env_feedback": "...", '
    '"confidence": <0..1>} ] }'
)

def number_lines(code):
    return "\n".join(f"{i:4d}  {ln}" for i, ln in enumerate(code.splitlines(), 1))

def build_case_block(task, overview, code):
    """The shared case payload appended to every check. Code is 1-indexed so error_line is meaningful."""
    return (f"\n\n=== TASK: {task} ===\n=== TASK OVERVIEW + DATA INVENTORY ===\n{overview}\n"
            f"=== AGENT CODE (lines are 1-indexed) ===\n```python\n{number_lines(code)}\n```\n")

def decomposed_messages(check_key, task, overview, code, fewshot=""):
    """-> (system, user) for one decomposed check."""
    user = CHECKS[check_key] + (("\n\nEXAMPLES (real cases):\n"+fewshot) if fewshot else "") + build_case_block(task, overview, code)
    return SYS, user

def holistic_messages(task, overview, code):
    return SYS.split("Analyze ONLY")[0] + HOLISTIC, build_case_block(task, overview, code)
