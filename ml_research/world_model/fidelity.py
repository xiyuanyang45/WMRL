#!/usr/bin/env python3
"""Phase-3 fidelity harness: does the WM (Qwen3.5) reproduce the REAL grader's verdict, without running code?

For each held-out real case (code + real status + real pos), query the WM with:
  - DECOMPOSED: the 7 parallel aspect checks (prompts.CHECKS) -> wm_aggregate.aggregate_decomposed
  - HOLISTIC:   one call (prompts.HOLISTIC)                      -> wm_aggregate.aggregate_holistic
Then compare predicted status/reward vs real. Reports: status confusion matrix, per-check fidelity, and
(for valid cases) predicted quality-bucket vs real pos correlation.

Run (one model per process; distribute models across GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python fidelity.py --model <9b> --n 1058 --design both --out res_9b.json
Style: parse failures are COUNTED (a fidelity metric: does the WM emit parseable JSON?), not silently dropped.
"""
import argparse, json, re, sys, collections, os
sys.path.insert(0, os.path.dirname(__file__))
import prompts as P
import wm_aggregate as A

def extract_json(text):
    """Robust: strip <think>..</think>, take the last balanced {...} that json-parses. None if unparseable."""
    if not text: return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    # try fenced ```json
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cands = []
    if m: cands.append(m.group(1))
    # all balanced top-level braces (greedy from each '{')
    depth=0; start=None
    for i,c in enumerate(text):
        if c=='{':
            if depth==0: start=i
            depth+=1
        elif c=='}':
            depth-=1
            if depth==0 and start is not None: cands.append(text[start:i+1])
    for c in reversed(cands):
        try: return json.loads(c)
        except Exception: continue
    return None

def build_all(cases, design, fewshot):
    """-> list of (case_idx, kind, system, user). kind in {'holistic'} U CHECKS keys."""
    reqs=[]
    for ci,c in enumerate(cases):
        ov = (c.get("overview") or "")[:4000]; code=(c.get("code") or "")[:8000]; task=c.get("task","")
        if design in ("decomposed","both"):
            for k in P.CHECKS:
                fs = fewshot.get(k,"") if fewshot else ""
                s,u = P.decomposed_messages(k, task, ov, code, fs)
                reqs.append((ci,k,s,u))
        if design in ("holistic","both"):
            s,u = P.holistic_messages(task, ov, code)
            reqs.append((ci,"holistic",s,u))
    return reqs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True); ap.add_argument("--n",type=int,default=1058)
    ap.add_argument("--design",choices=["decomposed","holistic","both"],default="both")
    ap.add_argument("--tp",type=int,default=1); ap.add_argument("--think",action="store_true")
    ap.add_argument("--fewshot",action="store_true")
    ap.add_argument("--offset",type=int,default=0)
    ap.add_argument("--fidelity",default="/tmp/wm_data/fidelity.json")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    cases=json.load(open(args.fidelity))[args.offset:args.offset+args.n]
    fewshot=None
    if args.fewshot:
        import mk_fewshot; fewshot=mk_fewshot.build()
        print(f"[harness] few-shot ENABLED ({sum(len(v) for v in fewshot.values())} chars of demos)",flush=True)
    reqs=build_all(cases,args.design,fewshot)
    print(f"[harness] {len(cases)} cases -> {len(reqs)} WM queries (design={args.design}, think={args.think})",flush=True)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    # GUIDED JSON + maxLength caps: the 9B rambles in `reason` (truncating the JSON); a hard maxLength bounds each
    # output so it is short, fast, and ALWAYS closes -> kills truncation (303) and malformed (121). Small bounded
    # pieces, massively parallel (vLLM continuous batching). xgrammar backend (prebuilt, no nvcc JIT).
    CHECK_SCHEMA={"type":"object","properties":{
        "reason":{"type":"string","maxLength":280},"verdict":{"type":"string","enum":["pass","fail"]},
        "confidence":{"type":"number"},"error_type":{"type":"string","maxLength":40},
        "error_type_alt":{"type":"string","maxLength":40},
        "error_line":{"type":"integer"},"env_feedback":{"type":"string","maxLength":420}},
        "required":["reason","verdict","confidence","error_type","error_type_alt","error_line","env_feedback"]}
    PRED={"type":"object","properties":{
        "status":{"type":"string","enum":["malformed","syntax_err","runtime_err","ran_no_sub","valid"]},
        "error_type":{"type":"string","maxLength":40},"error_line":{"type":"integer"},
        "quality_bucket":{"type":"string","enum":["low","fair","mid","high",""]},
        "env_feedback":{"type":"string","maxLength":420},"confidence":{"type":"number"}},
        "required":["status","error_type","error_line","quality_bucket","env_feedback","confidence"]}
    HOLI_SCHEMA={"type":"object","properties":{
        "reason":{"type":"string","maxLength":300},
        "predictions":{"type":"array","items":PRED,"minItems":1,"maxItems":2}},
        "required":["reason","predictions"]}
    llm=LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
            gpu_memory_utilization=0.90, max_model_len=12288, trust_remote_code=True, enforce_eager=True)
    # Qwen3.5 non-thinking preset (HF card); think is DROPPED (too slow + conflicts with guided). Bounded output
    # -> max_tokens=1024 is ample and fast.
    def mk_sp(kind):
        sch=HOLI_SCHEMA if kind=="holistic" else CHECK_SCHEMA
        return SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.0,
                              max_tokens=1024, seed=0, structured_outputs=StructuredOutputsParams(json=sch))
    sps=[mk_sp(kind) for (_,kind,_,_) in reqs]
    convos=[[{"role":"system","content":s},{"role":"user","content":u}] for _,_,s,u in reqs]
    outs=llm.chat(convos, sps)
    texts=[o.outputs[0].text for o in outs]

    # collect per-case
    bycase=collections.defaultdict(dict)
    parse_fail=collections.Counter()
    for (ci,kind,_,_),txt in zip(reqs,texts):
        j=extract_json(txt)
        if j is None: parse_fail[kind]+=1
        bycase[ci][kind]={"raw":txt[:1200],"json":j}

    results=[]
    for ci,c in enumerate(cases):
        rec={"task":c["task"],"real_status":c["status"],"real_exc":c.get("exc_type"),"real_pos":c.get("pos"),
             "code_head":(c.get("code") or "")[:200],
             "raw":{k:bycase[ci][k]["raw"] for k in bycase[ci]}}
        text_with_fence=f"```python\n{c.get('code','')}\n```"
        if args.design in ("decomposed","both"):
            checks={}
            for k in P.CHECKS:
                j=bycase[ci].get(k,{}).get("json")
                if j and ("verdict" in j or k=="quality"): checks[k]=j
            try:
                # ROBUST: aggregate over whatever parsed (a missing check = no opinion), confidence-gated.
                rec["decomposed"]=A.aggregate_decomposed(text_with_fence, checks)
                rec["decomposed"]["n_parsed"]=len(checks)
            except Exception as e:
                rec["decomposed"]={"status":"AGG_ERROR","err":str(e)}
        if args.design in ("holistic","both"):
            j=bycase[ci].get("holistic",{}).get("json")
            if j and (j.get("predictions") or j.get("status")):
                try: rec["holistic"]=A.aggregate_holistic(text_with_fence,j)
                except Exception as e: rec["holistic"]={"status":"AGG_ERROR","err":str(e)}
            else:
                rec["holistic"]={"status":"PARSE_FAIL"}
        results.append(rec)

    json.dump({"model":args.model,"design":args.design,"think":args.think,"n":len(cases),
               "parse_fail":dict(parse_fail),"results":results},
              open(args.out,"w"), indent=1)
    # quick console summary
    print(f"[harness] parse failures by kind: {dict(parse_fail)}",flush=True)
    def norm_exc(s): return (s or "").split(".")[-1].replace("Error","").replace("Exception","").lower()
    for design in (["decomposed","holistic"] if args.design=="both" else [args.design]):
        usable=[r for r in results if design in r and r[design].get("status") in
                ("malformed","syntax_err","runtime_err","ran_no_sub","valid")]
        match=sum(1 for r in usable if r[design]["status"]==r["real_status"])
        def st2(r):
            cs=r[design].get("cand_status") or [r[design]["status"]]
            return r["real_status"] in cs
        match2=sum(1 for r in usable if st2(r))
        # ladder-stage match (any-runtime collapses); reward MAE
        import numpy as np
        rew_real={"malformed":0.0,"syntax_err":0.0,"runtime_err":0.1,"ran_no_sub":0.2}
        def rr(r): return rew_real.get(r["real_status"], 0.5+0.5*({"low":.05,"fair":.20,"mid":.45,"high":.75}.get(
            "mid" if r["real_pos"] is None else ("low" if r["real_pos"]<.1 else "fair" if r["real_pos"]<.3 else "mid" if r["real_pos"]<.6 else "high"),.45)))
        mae=np.mean([abs(r[design].get("reward",0)-rr(r)) for r in usable]) if usable else float("nan")
        # FEEDBACK fidelity: on real runtime_err cases predicted runtime, does exc_type match (top-1 and @candidates)?
        rt=[r for r in usable if r["real_status"]=="runtime_err" and r[design]["status"]=="runtime_err"]
        exc1=sum(1 for r in rt if norm_exc(r[design].get("error_type"))==norm_exc(r["real_exc"]))
        def cand_hit(r):
            cands=r[design].get("cand_errors") or [{"error_type":r[design].get("error_type")}]
            return any(norm_exc(c.get("error_type"))==norm_exc(r["real_exc"]) for c in cands)
        excC=sum(1 for r in rt if cand_hit(r))
        print(f"[{design}] usable={len(usable)}/{len(results)}  status@1={100*match/max(1,len(usable)):.0f}% "
              f"status@2={100*match2/max(1,len(usable)):.0f}%  reward-MAE={mae:.3f}  "
              f"exc@1={exc1}/{len(rt)} ({100*exc1/max(1,len(rt)):.0f}%)  "
              f"exc@cand={excC}/{len(rt)} ({100*excC/max(1,len(rt)):.0f}%)",flush=True)
    print(f"[harness] wrote {args.out}",flush=True)

if __name__=="__main__":
    main()
