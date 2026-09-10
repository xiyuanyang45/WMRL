"""v11 one-demo SFT: LoRA BC of MiniVLA-1B on the 10 replayed LIBERO-Long demo_0 trajectories
(data from demo_replay_l10.py — pixel-identical to the RL rollout pipeline).

Protocol (cited): one-trajectory-per-task SFT cold start = SimpleVLA-RL (17.3% base, 7B) and
RIPT-VLA (<4% base, 20M), both on LIBERO-Long; SFT trained to convergence on the demo set
(RIPT practice) = fixed 5 epochs here, per-epoch adapters saved, LAST epoch is the SFT model.
LoRA config identical to the RL runs (r16 alpha32, same target modules). lr is mini-swept
{1e-4, 5e-4} by the caller (two invocations); pick by 10-task probe, symmetric for all arms.

Usage: bc_sft_l10.py --gpu G --lr 1e-4 --data data/bc_l10_demo0.pt --out sft_l10/lr1e-4
"""
import argparse, os, sys, time, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("PRISMATIC_DATA_ROOT", os.path.join(ROOT, "models"))
os.environ.setdefault("HF_TOKEN", "hf_placeholder"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.join(ROOT, "third_party/openvla-mini"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--data", default="data/bc_l10_demo0.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--microbatch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=3)   # effective batch 24
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig
    import minivla_common as mc

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = f"cuda:{args.gpu}"
    out = os.path.join(ROOT, args.out); os.makedirs(out, exist_ok=True)

    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                          target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                          task_type="CAUSAL_LM")
    vla, unnorm = mc.load_minivla(dev, lora_cfg=lora_cfg)
    A = vla.get_action_dim(unnorm)
    trainable = [p for p in vla.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"trainable {sum(p.numel() for p in trainable):,}  lr={args.lr}", flush=True)

    pairs = []
    for df in args.data.split(","):
        tasks = torch.load(os.path.join(ROOT, df), weights_only=False)
        pairs += [p for t in tasks for p in t["pairs"]]
        print(f"loaded {df}: cumulative {len(pairs)} pairs", flush=True)

    # microbatches are bucketed BY TASK: instructions tokenize to different lengths across
    # tasks, and the padding-free batched-logprob path requires equal-length input_ids
    # (same reason the RL trainer only batches same-task rollouts).
    by_task = {}
    for i, p in enumerate(pairs):
        by_task.setdefault(p["instruction"], []).append(i)

    def epoch_microbatches():
        mbs = []
        for ins, lst in by_task.items():
            li = lst[:]; random.shuffle(li)
            for s in range(0, len(li), args.microbatch):
                mbs.append(li[s:s + args.microbatch])
        random.shuffle(mbs)
        return mbs

    step = 0
    for ep in range(args.epochs):
        mbs = epoch_microbatches()
        ep_loss, ep_acc, n_seen = 0.0, 0.0, 0
        for s in range(0, len(mbs), args.accum):
            group = mbs[s:s + args.accum]
            chunk = [i for mb in group for i in mb]
            opt.zero_grad(set_to_none=True)
            for mb_idx in group:
                mb = [pairs[i] for i in mb_idx]
                ids_l, pv_l, tok_l = [], [], []
                for p in mb:
                    ids, pv, _ = mc.build_inputs(vla, p["image"], p["instruction"], dev)
                    ids_l.append(ids); pv_l.append(pv)
                    tok_l.append(torch.as_tensor(p["action_tokens"], device=dev))
                ids_b = torch.cat(ids_l, dim=0)
                pv_b = ({k: torch.cat([q[k] for q in pv_l], dim=0) for k in pv_l[0]}
                        if isinstance(pv_l[0], dict) else torch.cat(pv_l, dim=0))
                tok_b = torch.stack(tok_l, dim=0)
                lp = mc.per_token_logprobs_batched(vla, ids_b, pv_b, tok_b, A, grad=True)  # [B, A]
                loss = -(lp.mean()) * (len(mb) / len(chunk))
                loss.backward()
                ep_loss += float(-lp.mean().detach()) * len(mb); n_seen += len(mb)
            opt.step(); step += 1
        torch.cuda.empty_cache()
        ep_dir = os.path.join(out, f"epoch_{ep}")
        if os.path.isdir(ep_dir):
            import shutil; shutil.rmtree(ep_dir)
        vla.llm_backbone.llm.save_pretrained(ep_dir)
        print(f"epoch {ep}: mean_nll={ep_loss / max(n_seen,1):.4f} opt_steps={step}", flush=True)
    vla.llm_backbone.llm.save_pretrained(os.path.join(out, "final"))
    print("SFT_DONE", out, flush=True)


if __name__ == "__main__":
    main()
