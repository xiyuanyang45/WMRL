#!/usr/bin/env python3
"""
Core building blocks for multi-turn agentic RLVR. Pure functions, no GPU/vLLM — independently testable.
  - build_system_prompt: self-contained instruction (what actions exist, how to call, the goal)
  - build_turn_prompt(tok, task, overview, history): full chat prompt for the current turn,
    history = list of {role, content} for prior turns (model outputs + env observations)
  - parse_action(text): parse the model's output into ('request_info'|'execute_code', payload)
"""
import re, os

# Optional cap on the MASKED prompt context (system+task+history), shared by rollout AND trainer so both
# build byte-for-byte identical prompts even when long histories must be trimmed. 0/unset = no cap.
MAX_PROMPT_TOKENS = int(os.environ.get("MAX_PROMPT_TOKENS", "0")) or None

# Thinking on/off — a SINGLE module-level flag that BOTH rollout and trainer read (via build_turn_prompt's
# default), so they always agree (on-policy). enable_thinking=True => Qwen3.5 emits <think>reasoning</think>
# then the action; =False => template inserts an empty <think></think> and the model writes the action
# DIRECTLY (no reasoning), so every trained token is the action/code (no signal dilution across reasoning).
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "1") == "1"

# ---- action protocol (text-based, NOT vLLM tool-call: simpler to mask + train) ----
# DESIGN NOTE / repo alignment: the MLE-Dojo reference agent (mledojo/agent/mleagent) frames the goal as
# "maximizing the Position Score in limited steps" and, after EVERY successful action, shows a `reflection`
# template that says "Optimize your Model/Parameters/Data Processing/Algorithm for continuous improvement".
# Our earlier prompt OMITTED that — it only said "if errored, fix it" + "simple beats fancy", which never
# pushed the agent to improve a valid submission. Trajectory analysis confirmed the consequence: after a
# valid sub the model's later turns netted +0.001 rank change (no improvement) and broke the code 55% of
# the time. This template restores the reference's iterative-improvement intent (plus an explicit
# keep-your-best-working-solution discipline to cut self-breakage), while keeping our text-action format.
SYSTEM_TEMPLATE = """You are an expert ML engineer competing to MAXIMIZE your leaderboard Position Score over
UP TO {K} attempts (Position Score = your percentile rank, higher = better; earned ONLY when your code runs and
writes a VALID submission). Work in two phases:

PHASE 1 — LAND A VALID SUBMISSION FIRST (turn 1): write a SIMPLE solution you are confident RUNS end-to-end and
writes SUBMISSION_PATH in the exact required format. A valid submission is your safety net — keep it simple.

PHASE 2 — IMPROVE INCREMENTALLY (later turns): START FROM YOUR LAST WORKING CODE (it is in the history above),
copy it, and change EXACTLY ONE thing per turn — e.g. add/transform features, change preprocessing (missing
values, scaling, encoding), tune a hyperparameter, try a stronger model, ensemble two models, or use
cross-validation to choose between two options. Keep everything else identical. If a change ERRORS or scores
WORSE, discard it and try a DIFFERENT single change. Never rewrite from scratch once something works.

ENVIRONMENT CONTRACT (these Python variables are ALREADY defined when your code runs):
- DATA_DIR: absolute path to the folder with the data files listed below — load every input from here,
  e.g. pd.read_csv(os.path.join(DATA_DIR, 'train.csv')).
- SUBMISSION_PATH: absolute path to write your submission CSV to, e.g. df.to_csv(SUBMISSION_PATH, index=False).
A modern NVIDIA GPU (~20 GB) is available (torch.cuda.is_available() is True; TensorFlow/Keras detect it too) —
use it for heavier models when it can raise your rank, keep GPU memory under ~20 GB, and fall back to CPU if
something fails to import or does not fit. Do NOT hardcode '/kaggle/...', '/content/...', or any other path —
use ONLY DATA_DIR and SUBMISSION_PATH.
{ENV_BRIEF}

EACH TURN, respond in TWO parts, in this order:
1) ANALYSIS (2-4 short sentences of plain text, FIRST): look at the MOST RECENT result in the history.
   If it ERRORED, name exactly what went wrong (e.g. "KeyError on column X") and the ONE change that fixes it;
   if it was VALID, name the ONE change you will make to raise the score; on turn 1, state your simple baseline
   plan in one sentence. Keep it SHORT — not an essay, not code.
2) CODE: a SINGLE fenced Python block — a complete, self-contained script:
```python
# full script: load from DATA_DIR, train, predict, write SUBMISSION_PATH
```
Put nothing after the code (or, to ask for more info instead, emit just <action>request_info</action>).

After your code runs, the environment returns your print() output and — if a valid submission was written —
your leaderboard Position and Raw Score; otherwise the full error traceback (long output truncated to ~1000
tokens, head and tail kept). The DATA BRIEF below is AUTO-EXTRACTED from the actual files (real schemas, dtypes,
row counts, image dimensions, zip/dir contents) — it is complete and trustworthy, so you can model from turn 1;
you may still print() (df.head(), value_counts(), os.listdir) to explore. You keep the FULL history of every
previous attempt and its result, so always build on your best working one.

RULES:
- Keep a valid submission as your fallback; only replace it with code that RUNS and scores BETTER.
- The submission CSV must match the required format EXACTLY (see the SUBMISSION CONTRACT below).
- Keep code efficient so it finishes within the time limit."""


# Sandbox runtime brief injected into the system prompt. HARDCODED (not queried at runtime) so the rollout and the
# trainer render byte-identical prompts (on-policy). KEEP IN SYNC with envs/kaggle-grade — the env the model's code
# actually runs in. Bounded length by construction (one short paragraph).
SANDBOX_ENV_BRIEF = (
    "Runtime: Python 3.11 on Linux (this is the sandbox where YOUR code executes). Installed libraries you may "
    "import (name version): numpy 1.26, pandas 2.2, scipy 1.17, scikit-learn 1.4, xgboost 3.2, lightgbm 4.6, "
    "catboost 1.2, statsmodels 0.14, nltk 3.9, torch 2.12 (CUDA), torchvision 0.27, tensorflow 2.21 (CUDA), "
    "keras 3.14, transformers 5.12, Pillow 12.2, matplotlib 3.10, seaborn 0.13. Anything NOT in this list is not "
    "installed (e.g. no opencv, no jax) — do not import it, or the run will error.")


_PROMPT_ADDON = None


def _prompt_addon():
    """Optional model-specific guidance appended to the system prompt. Env PROMPT_ADDON_FILE names a file
    (absolute, or relative to this repo root); unset/empty => "" so every existing path renders BYTE-IDENTICAL
    to the historical prompt. Cached after first read; missing file fails loud (a silently-absent addon would
    invalidate the experiment the flag was set for)."""
    global _PROMPT_ADDON
    if _PROMPT_ADDON is None:
        p = os.environ.get("PROMPT_ADDON_FILE", "")
        if not p:
            _PROMPT_ADDON = ""
        else:
            import pathlib
            fp = pathlib.Path(p) if os.path.isabs(p) else pathlib.Path(__file__).parent / p
            _PROMPT_ADDON = "\n\n" + fp.read_text().strip()
    return _PROMPT_ADDON


def build_system_prompt(K):
    return SYSTEM_TEMPLATE.format(K=K, ENV_BRIEF=SANDBOX_ENV_BRIEF) + _prompt_addon()


def build_turn_prompt(tok, task, overview, history, K, enable_thinking=None, max_prompt_tokens=-1):
    """history: list of {"role":"assistant"/"user", "content": str} for prior turns.
       assistant = model's prior action text; user = env observation (masked in loss).

    enable_thinking=None (the default) => use the module ENABLE_THINKING flag, so rollout and trainer (both
    call this with the default) ALWAYS pick the same mode => on-policy. Pass an explicit bool only to override.

    On-policy guarantee: an optional token budget (max_prompt_tokens; -1 => module MAX_PROMPT_TOKENS) trims
    the OLDEST (assistant,user) history pairs — the system prompt and the first task message stay PINNED —
    until the prompt fits the budget. The rollout and the trainer call THIS SAME function with the SAME
    budget, so they produce byte-for-byte identical prompts even when trimming; the trainer therefore never
    front-truncates a context the model never generated under. budget=None disables it."""
    if enable_thinking is None:
        enable_thinking = ENABLE_THINKING
    budget = MAX_PROMPT_TOKENS if max_prompt_tokens == -1 else max_prompt_tokens
    sys_msg = {"role": "system", "content": build_system_prompt(K)}
    first = {"role": "user", "content": f"Competition: {task}\n\n{overview}\n\nBegin."}
    hist = list(history)

    def render(h):
        msgs = [sys_msg, first] + h
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=enable_thinking)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompt = render(hist)
    if budget:
        # drop the OLDEST (assistant,user) pair until within budget; system + first task msg stay pinned
        while len(hist) >= 2 and len(tok(prompt, add_special_tokens=False).input_ids) > budget:
            hist = hist[2:]
            prompt = render(hist)
    return prompt


def _tok_head(tok, text, n):
    ids = tok(text, add_special_tokens=False).input_ids
    return text if len(ids) <= n else tok.decode(ids[:n], skip_special_tokens=False) + " …"


def _hsize(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f}{u}"
        n /= 1024


def _compress_cols(cols, dtypes):
    """['f_00'..'f_30','id'] -> ['id (int)', 'f_00…f_30 (31 cols, float)'] — patterned runs collapsed so a
    194-column frame stays readable while NO information is lost (range + count + dtype kept)."""
    runs = {}
    for c in cols:
        m = re.match(r"^(.*?)(\d+)$", str(c))
        if m:
            runs.setdefault(m.group(1), []).append(c)
    grouped = {p: cs for p, cs in runs.items() if len(cs) >= 4}
    flat = {c for cs in grouped.values() for c in cs}
    res, done = [], set()
    for c in cols:
        if c in flat:
            for p, cs in grouped.items():
                if c in cs and p not in done:
                    done.add(p)
                    dt = dtypes.get(cs[0], "")
                    res.append(f"{cs[0]}…{cs[-1]} ({len(cs)} cols, {dt})")
                    break
        else:
            res.append(f"{c} ({dtypes.get(c, '?')})")
    return res


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".ppm"}


def _img_summary(openers):
    """openers: list of zero-arg callables, each returning a PIL.Image (lazy). Sample up to 3 → a dims/mode
    string ('RGB, 374×500; 500×375 → VARIABLE size' or 'L, 28×28 (uniform)'). FAIL LOUD: if image files are
    present but NONE can be opened, raise — an image task's prompt is incomplete without dimensions."""
    from PIL import Image  # noqa: F401
    dims, modes, err = [], [], None
    for op in openers[:3]:
        try:
            im = op()
            dims.append(tuple(im.size)); modes.append(im.mode)   # size = (W, H)
        except Exception as e:
            err = e
    if not dims:
        raise RuntimeError(f"image files present but none could be opened ({type(err).__name__ if err else '?'}: "
                           f"{str(err)[:120]}) — refusing to emit an image prompt without dimensions")
    mode = modes[0] if len(set(modes)) == 1 else "/".join(sorted(set(modes)))
    if len(set(dims)) == 1:
        w, h = dims[0]; return f"{mode}, {w}×{h} (uniform)"
    return f"{mode}, " + "; ".join(f"{w}×{h}" for w, h in dims) + " → VARIABLE size"


def _csv_brief(tok, p, max_cols_full=24):
    """One tabular file -> schema line(s): n_rows × n_cols, every column with dtype (patterned runs
    compressed), and EXAMPLE VALUES for string/object columns (dtype traps like a 10-char string feature
    among floats must be VISIBLE up front). FAIL LOUD if the file cannot be parsed as a table."""
    import pandas as pd
    try:
        sample = pd.read_csv(p, nrows=300, sep=None, engine="python")
    except Exception as _sniff_err:
        # sep=None runs pandas' delimiter SNIFFER, which is unreliable on SINGLE-COLUMN files: with no
        # delimiter present it guesses, and on a column of hex ids (aptos2019 test.csv = one `id_code`
        # per line) it picked a hex letter, yielding "Expected 3 fields in line 11, saw 4".
        # Retry with the plain comma reader before failing. This is STRICTLY ADDITIVE — any file the
        # sniffer already handled still takes the original path, so existing task prompts are
        # byte-identical and training parity is preserved.
        try:
            sample = pd.read_csv(p, nrows=300)
        except Exception:
            raise RuntimeError(
                f"CSV {p.name} unreadable as a table (sniffer: {type(_sniff_err).__name__}: "
                f"{str(_sniff_err)[:100]}; comma fallback also failed) — "
                f"refusing to emit a prompt missing this file's schema") from _sniff_err
    if p.stat().st_size < 300 * 1024 * 1024:
        with open(p, "rb") as fh:
            n_rows = sum(1 for _ in fh) - 1
        rows = f"{n_rows} rows"
    else:
        rows = f"~{int(p.stat().st_size / max(len(sample.to_csv(index=False)) / max(len(sample), 1), 1))} rows (est)"
    dt = {c: ("str" if str(t) == "object" else re.sub(r"\d+", "", str(t))) for c, t in sample.dtypes.items()}
    cols = _compress_cols(list(sample.columns), dt)
    if len(cols) > max_cols_full:    # uncompressible wide frame: a 100-name dump is ~950 tok of low value —
        cols = cols[:12] + [f"… (+{len(cols)-12} more — read this file's header row for the full list)"]
    line = f"{p.name}: {rows} × {len(sample.columns)} cols: [" + ", ".join(cols) + "]"
    objs = [c for c in sample.columns if dt.get(c) == "str"][:4]
    exs = []
    for c in objs:
        v = sample[c].dropna()
        if len(v):
            ex = re.sub(r"\s+", " ", str(v.iloc[0]))[:32]      # collapse whitespace/escapes: token bombs
            exs.append(f"{c}: e.g. {ex!r}")
    if exs:
        line += "\n    string-column examples -> " + "; ".join(exs)
    return line


def _zip_brief(p):
    import zipfile, collections as _c, io
    try:
        zf = zipfile.ZipFile(p); names = zf.namelist()
    except Exception as e:
        raise RuntimeError(f"ZIP {p.name} unreadable ({type(e).__name__}) — refusing incomplete prompt") from e
    files = sorted(n for n in names if not n.endswith("/"))   # sorted: zip-order independent (byte-identical)
    ext = _c.Counter(("." + n.rsplit(".", 1)[-1].lower()) if "." in n.rsplit("/", 1)[-1] else "(dir)" for n in files)
    top = ", ".join(f"{k}×{v}" for k, v in ext.most_common(3))
    line = f"{p.name}: ZIP with {len(names)} entries ({top}); e.g. {files[:2]}"
    imgs = sorted(n for n in files if ("." + n.rsplit(".", 1)[-1].lower()) in IMG_EXTS)[:3]  # sorted: byte-identical
    if imgs:
        from PIL import Image
        line += f"  [images: {_img_summary([(lambda n=n: Image.open(io.BytesIO(zf.read(n)))) for n in imgs])}]"
    return line


def _dir_brief(d):
    import collections as _c
    entries = sorted(d.iterdir(), key=lambda e: e.name)   # sorted: filesystem-order independent (byte-identical)
    ext = _c.Counter((e.suffix.lower() or ("(dir)" if e.is_dir() else "(noext)")) for e in entries)
    top = ", ".join(f"{k}×{v}" for k, v in ext.most_common(3))
    line = f"{d.name}/: DIRECTORY with {len(entries)} entries ({top}); e.g. {[e.name for e in entries[:2]]}"
    cand, scanned = [], 0                                # recursive (class-subfolders common); bounded both ways
    for q in d.rglob("*"):
        scanned += 1
        if q.suffix.lower() in IMG_EXTS:
            cand.append(q)
        if len(cand) >= 64 or scanned >= 4000:           # enough to sort-sample; or stop deep walk (perf guard)
            break
    if cand:
        cand.sort(key=lambda x: x.as_posix())            # sorted: deterministic 3 → byte-identical across pods
        from PIL import Image
        line += f"  [images: {_img_summary([(lambda q=q: Image.open(q)) for q in cand[:3]])}]"
    return line


# Soft, standardized per-modality hint (ONE injected per task). Facts the agent can't introspect + a LIGHT
# starting-point suggestion (italic clause) — deliberately NOT a fixed recipe, so RL still discovers the method.
APPROACH = {
    "tabular": ("=== TABULAR TASK ===\nPredict the target column(s) for each test row (schema + submission "
                "contract above). A simple tree-based or linear model is a safe first submission — then improve."),
    "text":    ("=== TEXT / NLP TASK ===\nThe predictive signal is mainly in the free-text column(s) above — "
                "featurize the text to use it. A simple bag-of-words model is a fine starting point — then improve."),
    "image":   ("=== IMAGE TASK ===\nInputs are image files (directories/zips with sampled dimensions in the "
                "inventory above); labels are given above (a labels CSV or encoded in the image filenames). Write "
                "one prediction per test image in the submission format; the GPU is available. A small CNN, or a "
                "quick downscaled baseline, lands a valid submission first — then improve."),
}


def _detect_modality(pub, has_image):
    """Pure introspection (no per-task hand-info): image if any data dir/zip holds image files; else text if a
    train CSV has a genuinely free-text column (mean cell length > 40 chars); else tabular."""
    if has_image:
        return "image"
    import pandas as pd
    trains = sorted(pub.glob("*train*.csv")) or sorted(pub.glob("*.csv"))
    for p in trains[:1]:
        try:
            s = pd.read_csv(p, nrows=200, sep=None, engine="python")
        except Exception:
            continue
        for c in s.columns:
            v = s[c].dropna().astype(str)
            if len(v) and v.str.len().mean() > 40:
                return "text"
    return "tabular"


def build_overview(tok, task, data_dir, desc_tokens=900, max_cols_full=24, max_tokens=3000):
    """v2 (2026-06-11) — the COMPLETE auto-generated task brief. Design bar (user): the brief alone must be
    sufficient, with high probability, for a strong model to write a VALID submission in ONE shot — and the
    extraction is pure introspection (no per-task hand-info), so it generalizes to any future MLE benchmark.
    Everything is TOKEN-budgeted (char caps starved the description). Err on MORE info, not less.

    Sections: DESCRIPTION (head, token-capped; the Evaluation section is force-included if the cap cut it)
    · DATA INVENTORY (recursive: every file with size; CSVs with full schema/dtypes/row-counts + string-col
    example values; ZIP contents; directory contents) · TARGETS (columns present in train but not test)
    · SUBMISSION CONTRACT (exact columns + example rows). Single source for trainer+rollout+eval (on-policy).
    """
    import pandas as pd
    from pathlib import Path
    pub = Path(data_dir) / "public"

    # MLE-Dojo/DSBench ship `description.txt`; MLE-bench ships `description.md`. Prefer .txt so
    # EVERY existing task keeps byte-identical prompt construction (training parity is required);
    # .md is only consulted when .txt is absent.
    _dsc = pub / "description.txt"
    if not _dsc.exists():
        _dsc = pub / "description.md"
    if not _dsc.exists():
        raise RuntimeError(f"no description.txt/.md under {pub} — refusing incomplete prompt")
    full_desc = _dsc.read_text()
    desc = _tok_head(tok, full_desc, desc_tokens)
    m = re.search(r"^#+\s*Evaluation\b.*?(?=^#+\s|\Z)", full_desc, re.M | re.S)
    if m and m.group(0)[:80] not in desc:               # eval metric got cut off -> force-include its head
        desc += "\n\n=== EVALUATION (from description) ===\n" + _tok_head(tok, m.group(0), 220)

    inv, csvs = [], {}
    skip = {"description.txt", "description.md"}
    for p in sorted(pub.iterdir()):
        if p.name in skip:
            continue
        if p.is_dir():
            inv.append(_dir_brief(p))
        elif p.suffix.lower() == ".zip":
            inv.append(_zip_brief(p))
        elif p.suffix.lower() in (".csv", ".tsv"):
            try:
                hdr = list(pd.read_csv(p, nrows=0).columns)
            except Exception:
                hdr = None
            dup = next((n for n, c in csvs.items() if hdr is not None and c == hdr), None)
            if dup:
                inv.append(f"{p.name}: same schema as {dup}")     # e.g. sample_submission_null.csv
            else:
                inv.append(_csv_brief(tok, p, max_cols_full))
            if hdr is not None:
                csvs[p.name] = hdr
        elif p.suffix.lower() in (".json", ".jsonl"):
            try:
                first = open(p).readline()
                import json as _j
                keys = list(_j.loads(first).keys()) if p.suffix == ".jsonl" else list(_j.load(open(p)).keys())[:20]
                inv.append(f"{p.name}: {_hsize(p.stat().st_size)}, keys like {keys[:12]}")
            except Exception:
                inv.append(f"{p.name}: {_hsize(p.stat().st_size)}")
        else:
            inv.append(f"{p.name}: {_hsize(p.stat().st_size)}")

    tr = next((c for n, c in csvs.items() if "train" in n.lower()), None)
    te = next((c for n, c in csvs.items() if "test" in n.lower()), None)
    targets = ""
    if tr and te:
        only = [c for c in tr if c not in te]
        if only:
            targets = ("\n=== TARGET COLUMNS (in train but NOT in test — these are what you predict) ===\n"
                       + str(only))

    ss_p = pub / "sample_submission.csv"
    sub = ""
    if ss_p.exists():
        ss = pd.read_csv(ss_p, nrows=2)
        cols = list(ss.columns)
        if len(cols) <= 15:
            disp = ss.head(2).astype(str).apply(lambda s: s.str.slice(0, 24))   # cell cap: to_string pads
            # Mark which column(s) hold the PREDICTION (= submission cols absent from test.csv) vs which are
            # id/echoed (present in test). Auto-derived, robust across all task types (binary/multi/regression).
            note = ""
            if te is not None:                          # only when we know the test schema can we split pred vs echoed
                pred = [c for c in cols if c not in te]
                echoed = [c for c in cols if c in te]
                if pred:
                    note = f"\nFILL the prediction column(s) with your model output: {pred}."
                    if echoed:
                        note += f" COPY the column(s) {echoed} straight from the test set (id / passthrough)."
            sub = (f"\n=== SUBMISSION CONTRACT (write EXACTLY this format to SUBMISSION_PATH) ===\n"
                   f"columns = {cols}{note}\nExample rows (cells truncated for display):\n"
                   + disp.to_string(index=False))
        else:
            shown = cols[:6] + ["...", *cols[-3:]]
            sub = (f"\n=== SUBMISSION CONTRACT (write to SUBMISSION_PATH) ===\n{len(cols)} columns "
                   f"(column 0 = '{cols[0]}' id; the other {len(cols)-1} are target columns like {shown}); "
                   f"write one row per test id, filling every target column.")

    # ---- FAIL LOUD: refuse to emit a prompt that does not fully describe the data + output contract ----
    if not full_desc.strip():
        raise RuntimeError(f"[{task}] description.txt is empty — refusing incomplete prompt")
    if not inv:
        raise RuntimeError(f"[{task}] DATA INVENTORY empty (no data files under {pub}) — refusing incomplete prompt")
    if not sub:
        raise RuntimeError(f"[{task}] no SUBMISSION CONTRACT (sample_submission.csv missing under {pub}) — "
                           f"cannot define the output format; refusing incomplete prompt")

    has_image = any("[images:" in x for x in inv)        # set by _dir_brief/_zip_brief when image dims sampled
    approach = "\n\n" + APPROACH[_detect_modality(pub, has_image)]

    inv_str = _tok_head(tok, "\n".join(inv), 800)   # per-part length cap: a many-file dir can't dominate the brief
    ov = (desc + "\n\n=== DATA INVENTORY (everything under DATA_DIR) ===\n" + inv_str
          + targets + sub)
    ids = tok(ov, add_special_tokens=False).input_ids   # generous backstop only — should rarely fire
    if len(ids) > max_tokens:
        ov = tok.decode(ids[:max_tokens], skip_special_tokens=False) + "\n…(overview truncated)"
    return ov + approach            # modality hint appended AFTER the backstop so it is never truncated


def parse_action(text):
    """Return (action, payload). action in {'request_info','execute_code'}.
       Robust to unclosed code fences (truncation). Priority: explicit request_info tag,
       else a python code block, else treat raw as code."""
    # 1. explicit request_info action tag
    if re.search(r"<action>\s*request_info\s*</action>", text, re.IGNORECASE):
        # but if there's ALSO a code block, the model chose to execute → prefer code
        if not re.search(r"```python", text):
            return ("request_info", None)
    # 2. closed python block
    m = re.findall(r"```python\s*\n?(.*?)```", text, flags=re.DOTALL)
    if m:
        return ("execute_code", m[0])
    # 3. closed generic block
    m = re.findall(r"```[a-zA-Z]*\s*\n?(.*?)```", text, flags=re.DOTALL)
    if m:
        return ("execute_code", m[0])
    # 4. UNCLOSED python fence (truncated at max_tokens)
    idx = text.find("```python")
    if idx != -1:
        rest = text[idx+len("```python"):]
        return ("execute_code", rest.split("```")[0].lstrip("\n"))
    idx = text.find("```")
    if idx != -1:
        rest = re.sub(r"^[a-zA-Z]*\n", "", text[idx+3:], count=1)
        return ("execute_code", rest.split("```")[0])
    # 5. request_info tag even if alone
    if re.search(r"request_info", text, re.IGNORECASE):
        return ("request_info", None)
    # 6. no fence, no action tag → the model emitted NO valid action. This is DATA (the policy must learn
    #    to emit a proper action), surfaced explicitly as 'malformed' — NOT silently treated as code.
    return ("malformed", None)


# ---- token-level multi-turn rollout helpers (shared by the rollout host: chat-template turn wrapping + obs capping) ----
def _derive_turn_wrap(tok, enable_thinking):
    """Derive the chat-template pieces between turns DIRECTLY from the tokenizer (so we never re-render/re-tokenize
    the model's generation, and we reproduce the gen-prompt's <think></think> primer exactly).
    Returns (ASST_OPEN_PRIMER, IM_END, USER_OPEN). The qwen <|im_*|> markers are stable; the primer is derived."""
    def render(msgs):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=enable_thinking)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    txt = render([{"role": "user", "content": "x"}])
    marker = "<|im_start|>assistant\n"
    asst_open_primer = marker + txt[txt.rindex(marker) + len(marker):]   # "<|im_start|>assistant\n" + PRIMER
    return asst_open_primer, "<|im_end|>\n", "<|im_start|>user\n"


def _cap_obs(tok, obs, max_tokens):
    """Bound the env observation (execution feedback) so a giant traceback can't blow the fixed context budget and
    break the forced-K-turns guarantee. Keep the HEAD (error type/first lines) + the TAIL (where it failed); drop the
    middle. These tokens are MASKED in the loss — capping them changes only what the model READS, not what it trains
    on; and the SAME capped text is used for the token build, so rollout and trainer stay byte-for-byte identical."""
    if not max_tokens:
        return obs
    ids = tok(obs, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return obs
    head = tok.decode(ids[: max_tokens * 2 // 3], skip_special_tokens=False)
    tail = tok.decode(ids[-(max_tokens // 3):], skip_special_tokens=False)
    return head + "\n…(feedback truncated to fit context)…\n" + tail


