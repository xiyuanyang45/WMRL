#!/usr/bin/env python
"""3-arm token-GRPO trainer for MiniVLA-1B on LIBERO-90 — Phase-2b two-step alignment compare.

One trainer, three reward modes (--reward_mode); rollout collection + teacher-forced log-prob +
clipped GRPO update are IDENTICAL to scripts/grpo_minivla_par.py (arm 1 == that baseline). Only the
per-trajectory REWARD (and, for arm 3, the recal+ADAW that shape the advantage) differ:

  1. sparse       : reward = LIBERO check_success (0/1). == the live baseline. Robometer never loaded.
  2. dense_raw    : reward = raw Robometer-4B dense signal (potential-based-shaped progress, telescoped
                    Phi_T - Phi_0). Biased, uncalibrated -> the ablation that gets reward-hacked
                    (over-scores partial-progress-then-collapse failures; phase-2a REVIEW).
  3. dense_calib  : ("Ours") dense Robometer signal as PRIMARY per-step reward, ALIGNED to the sparse
                    success scale by (a) online monotone recal f (bucketed isotonic PAV, vla_align)
                    fit on (dense, sparse) anchor pairs and applied to the dense reward BEFORE it enters
                    the GRPO advantage, and (b) ADAW inverse-variance up-weight of anchor advantages.
  4. sparse_scarce: SCARCE-LABEL control (the regime our method is designed for): only ~anchor_ratio of
                    rollouts per group (at random) receive their sparse label; the rest are UNGRADED —
                    dropped from the GRPO advantage entirely (not zero-reward); groups renormalize over
                    graded members only, groups with <2 graded are skipped. Simulates costly success-
                    labeling (real-robot regime). dense_calib with anchor_ratio<1 + anchor_reward=sparse
                    + adaw=1 is the matching "ours" arm (recal fits on the same scarce labels; ADAW's
                    inverse-variance up-weight of the labeled rollouts is now meaningful, not an lr-confound).

Dense reward is computed OUT-OF-PROCESS by scripts/robometer_score_worker.py (envs/rbm, transformers
4.57.1) sharded across --worker_gpus, because Robometer needs a different env than the VLA stack.

Rollout render note: EGL on this box is SOFTWARE llvmpipe (CPU) with a single EGL device — the container
has no NVIDIA graphics capability (REVIEW_phase0). MUJOCO_EGL_DEVICE_ID must stay 0; rollout speed scales
with the NUMBER OF WORKER PROCESSES (CPU render parallelism + per-GPU policy compute), not with EGL ids.

Reference for the math: ml_research/grading/grade_store.py + docs/docs/docs/ALGORITHM.md (WMRL).
"""
import os, sys, json, time, csv, argparse, random, subprocess, glob, shutil
from datetime import datetime
import numpy as np
ROOT = "/home/sagemaker-user/xiyuan_work_dir/auto_research/vla_pilot/libero_rl"
# learner shares its GPU with rollout workers: keep allocator compact (2026-07-17 step-6 OOMs:
# learner cache creep + 26GB worker peak broke 40GB — see REVIEW_speedup.md)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("PRISMATIC_DATA_ROOT", os.path.join(ROOT, "models"))
os.environ.setdefault("HF_TOKEN", "hf_placeholder"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# vq_bet_official supplies `vqvae`, which prismatic's ACTION_TOKENIZERS imports lazily when the VQ
# (action-chunking) tokenizer is constructed. The rollout worker and bc_sft_vq have always inserted
# this; the TRAINER never did, because until v22 it only ever loaded the BASE model and that path
# never touches the VQ tokenizer. Missing it killed all of v22 at load_minivla with
# `ModuleNotFoundError: No module named 'vqvae'`. Harmless for base runs (nothing imports it).
sys.path.insert(0, os.path.join(ROOT, "third_party/vq_bet_official"))
sys.path.insert(0, os.path.join(ROOT, "third_party/openvla-mini"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import vla_align as va

_RBM_FAILS = [0]


def log(*a): print(f"[{datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)


def subsample_frames(queries, n):
    """Uniformly subsample a trajectory's per-step frames to n (Robometer training window = 8)."""
    imgs = [q["image"] for q in queries]
    if len(imgs) == 0:
        return np.zeros((1, 224, 224, 3), np.uint8)
    if len(imgs) > n:
        idx = np.linspace(0, len(imgs) - 1, n).round().astype(int)
        imgs = [imgs[i] for i in idx]
    return np.stack(imgs, 0).astype(np.uint8)


def score_dense(rollouts, worker_gpus, dense_frames, tmp_dir, model_path):
    """Shard trajectories across worker_gpus, score each with Robometer (envs/rbm subprocess), return
    {traj_index: {"prog":[...], "succ":[...]}}. Sequential w.r.t. the rollout phase (reuses the GPUs)."""
    if os.path.isdir(tmp_dir): shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    shards = {g: [] for g in worker_gpus}
    for i in range(len(rollouts)):
        shards[worker_gpus[i % len(worker_gpus)]].append(i)
    procs = []
    for g, idxs in shards.items():
        if not idxs: continue
        sd = os.path.join(tmp_dir, f"score_gpu{g}"); os.makedirs(sd, exist_ok=True)
        manifest = []
        for i in idxs:
            frames = subsample_frames(rollouts[i]["queries"], dense_frames)
            npy = f"traj_{i:04d}.npy"; np.save(os.path.join(sd, npy), frames)
            manifest.append({"id": str(i), "task": rollouts[i]["instruction"], "npy": npy})
        json.dump(manifest, open(os.path.join(sd, "manifest.json"), "w"))
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(g)
        # CONTAINER FIX (07-25): the training scheduler images ship a system cuDNN9 in /lib/x86_64-linux-gnu that
        # SHADOWS envs/rbm's bundled cuDNN -> "Could not load symbol cudnnGetLibConfig ... undefined
        # symbol" at the first compute_batch_outputs -> scorer dies -> dense=0. No system cuDNN
        # locally, which is why this never reproduced here. Put the env's own NVIDIA libs FIRST.
        _nvd = os.path.join(ROOT, "envs/rbm/lib/python3.10/site-packages/nvidia")
        if os.path.isdir(_nvd):
            _libs = [os.path.join(_nvd, d, "lib") for d in sorted(os.listdir(_nvd))
                     if os.path.isdir(os.path.join(_nvd, d, "lib"))]
            if _libs:
                env["LD_LIBRARY_PATH"] = ":".join(_libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
        cmd = ["./envs/rbm/bin/python", "scripts/robometer_score_worker.py",
               "--shard_dir", sd, "--gpu", str(g), "--model_path", model_path]
        lf = open(os.path.join(sd, "rbm.log"), "w")
        procs.append((g, subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT), sd))
    scores = {}
    for g, p, sd in procs:
        rc = p.wait()
        sp = os.path.join(sd, "scores.json")
        if rc != 0 or not os.path.isfile(sp):
            log(f"WARNING rbm scorer gpu{g} rc={rc} missing={not os.path.isfile(sp)}")
            _RBM_FAILS[0] += 1
            # surface the scorer's actual error (its stdout/stderr log) so it reaches train.log->S3
            _rlog = os.path.join(sd, "rbm.log")
            if os.path.isfile(_rlog):
                _lines = open(_rlog, errors="replace").read().splitlines()
                # the per-layer "RG: True" dump floods the tail; keep only real error signal
                _sig = [l for l in _lines if any(k in l for k in (
                    "Traceback", "Error", "error", "Exception", "raise ", "assert",
                    "CUDA", "out of memory", "Killed", "No module", "not found", "cannot"))
                    and "RG: True" not in l]
                _keep = (_sig[-15:] if _sig else _lines[-15:])
                log("RBM_ERR[gpu%d] (%d lines, %d signal):\n%s" % (g, len(_lines), len(_sig), "\n".join(_keep)))
            continue
        for k, v in json.load(open(sp)).items():
            scores[int(k)] = v
    if _RBM_FAILS[0] and not scores:
        raise RuntimeError(
            f"FATAL: ALL rbm scorers failed ({_RBM_FAILS[0]}) and no dense scores were produced. "
            "This is the dense=0 bug (missing phase2a_gonogo/robometer in payload, or model/env "
            "issue). Refusing to train a WM arm with a dead world-model reward.")
    return scores


def main():
    ap = argparse.ArgumentParser()
    # --- identical knobs to grpo_minivla_par.py (arm 1 reproduces the baseline exactly) ---
    ap.add_argument("--suite", default="libero_90")
    ap.add_argument("--task_ids", default="1,4")
    ap.add_argument("--group_size", type=int, default=16)
    ap.add_argument("--num_train_steps", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--num_wait", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--clip_eps_high", type=float, default=None,
                    help="asymmetric UPPER clip bound (DAPO/SimpleVLA-RL 'clip-higher': they relax "
                         "1.2 -> 1.28 to keep low-probability exploratory actions alive). Default "
                         "None = symmetric (== clip_eps), byte-identical to all prior runs.")
    ap.add_argument("--full_ft", type=int, default=0,
                    help="1 = FULL fine-tune of the LLM backbone: merge any loaded LoRA into the base "
                         "and unfreeze all backbone params (~0.5B; AdamW states fit easily on 80G, "
                         "target 8x80GB node). Checkpoints then save the whole LLM (~1GB each) and workers "
                         "auto-detect full dirs (minivla_common). Community delta: SimpleVLA-RL "
                         "full-fine-tunes; our 8.8M LoRA r16 may simply lack capacity. Default 0.")
    ap.add_argument("--adv_no_std", type=int, default=0,
                    help="1 = advantage is (r - group mean) WITHOUT /std (Dr.GRPO): with binary "
                         "rewards the std term inflates updates for near-deterministic groups and "
                         "shrinks them for the informative p~0.5 ones. Default 0 = classic GRPO.")
    # ★ PPO inner epochs. With 1 (the default, and what every run before 2026-07-26 used) old_lp is
    # recomputed from the SAME weights in the SAME pass, so ratio == 1 IDENTICALLY, the clip never
    # binds (clipfrac=0, approx_kl=0 in every logged step) and lr is the only step-size control —
    # which is why the v21 lr sweep could stop the collapse but produced exactly zero learning.
    # With E>1 old_lp is computed ONCE before the first epoch and FROZEN, so epochs 2..E see a real
    # ratio and the trust region actually exists. E=1 keeps behaviour byte-identical.
    ap.add_argument("--inner_epochs", type=int, default=1)
    ap.add_argument("--loss_steps_per_rollout", type=int, default=12)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--adv_eps", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:1")                 # AVOID GPU0/GPU2 (live baseline)
    ap.add_argument("--worker_gpus", default="1,3,4,5,6,7")       # 6 allowed GPUs; live baseline holds 0,2
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rl_runs/phase2b_run")
    ap.add_argument("--save_samples_iters", type=int, default=3)
    ap.add_argument("--sample_img_every", type=int, default=6)
    # --- phase-2b: reward mode + dense/recal/ADAW knobs ---
    ap.add_argument("--reward_mode", choices=["sparse", "dense_raw", "dense_calib", "sparse_scarce"], default="sparse")
    ap.add_argument("--dense_model", default="jesbu1/robometer-4b-fft-libero")
    ap.add_argument("--dense_frames", type=int, default=8, help="frames/traj fed to Robometer (=training window)")
    ap.add_argument("--dense_feature", choices=["potential", "final", "max", "success"], default="potential")
    ap.add_argument("--smooth_k", type=int, default=3, help="frame smoothing for the dense scalar endpoints")
    # recal (Step 1) — same defaults as grade_store / configs/ours
    ap.add_argument("--recal_min_pairs", type=int, default=200)
    ap.add_argument("--recal_refit_every", type=int, default=64)
    ap.add_argument("--recal_bins", type=int, default=10)
    # anchor stream: free-anchor regime -> anchor_ratio=1.0 (recal de-biases, ADAW weak).
    # SCARCE regime: anchor_ratio=0.15 == labeled fraction, used by BOTH sparse_scarce (graded fraction)
    # and dense_calib (recal-pair + ADAW-anchor fraction; pair with --anchor_reward sparse --adaw 1).
    ap.add_argument("--anchor_ratio", type=float, default=1.0)
    ap.add_argument("--anchor_reward", choices=["calib", "sparse", "mix"], default="calib")
    ap.add_argument("--init_adapter", default=None,
                    help="warm-start: load an existing LoRA adapter dir (trainable) instead of a fresh LoRA")
    ap.add_argument("--anchor_mix_w", type=float, default=2.0, help="w in (w*gt + f(dense))/(1+w) for anchor_reward=mix")
    # ADAW (Step 2). DEFAULT OFF: at anchor_ratio=1.0 a uniform w_t on ALL advantages == a hidden lr
    # multiplier (GRPO already unit-normalizes advantage) -> confounds the compare. ADAW does genuine
    # RELATIVE reweighting only when anchor_ratio<1 (scarce-anchor ablation). See PHASE2B_PLAN §ADAW.
    ap.add_argument("--adaw", type=int, default=0, help="1=apply ADAW advantage up-weight; 0=off (default)")
    ap.add_argument("--adaw_wmax", type=float, default=4.0)
    ap.add_argument("--adaw_halflife", type=float, default=64)
    ap.add_argument("--adaw_warmup", type=int, default=32)
    ap.add_argument("--adaw_target_w", type=float, default=2.0)
    # v4 (group-64) knobs — defaults keep v3-and-earlier behavior byte-identical:
    ap.add_argument("--worker_script", default="scripts/minivla_rollout_worker_batched.py")
    ap.add_argument("--variant", default="base",
                    help="base | vq. vq = MiniVLA-VQ action chunking: a residual-VQ codebook packs an "
                         "8-step action chunk into the SAME 7 discrete tokens the base model emits, so "
                         "the GRPO per-token-logprob path is unchanged and A stays 7. Requires a "
                         "VQ-aware worker (--worker_script scripts/minivla_rollout_worker_mt.py); the "
                         "batched worker has no variant support and would silently roll out the BASE "
                         "policy against VQ-trained weights.")
    ap.add_argument("--batched_update", type=int, default=0,
                    help="1 = microbatched teacher-forced update (same math, ~4x faster)")
    ap.add_argument("--update_microbatch", type=int, default=8)
    ap.add_argument("--save_ckpt_every", type=int, default=0,
                    help=">0: save the LoRA adapter every N steps to out/ckpts/step_XX (35MB each) "
                         "for best-checkpoint (early-stopping) selection")
    args = ap.parse_args()

    # Fail loud rather than train against a policy we are not rolling out: only the mt worker reads
    # --variant. Pairing vq with the batched worker would roll out the BASE decode path (1 action per
    # 7 tokens) while the trainer updates VQ chunk weights (8 actions per 7 tokens) — the reward would
    # be measured on a different policy than the gradient improves.
    if args.variant != "base" and "worker_mt" not in args.worker_script:
        raise SystemExit(f"--variant {args.variant} requires the VQ-aware worker; got "
                         f"--worker_script {args.worker_script}. Use scripts/minivla_rollout_worker_mt.py")

    # asymmetric upper clip bound (clip-higher); None keeps symmetric behaviour byte-identical
    _clip_hi = args.clip_eps_high if args.clip_eps_high is not None else args.clip_eps

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import torch
    from peft import LoraConfig
    import minivla_common as mc

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    DEV = args.device
    out = os.path.join(ROOT, args.out) if not args.out.startswith("/") else args.out
    os.makedirs(out, exist_ok=True)
    samples_root = os.path.join(out, "phase2b_samples"); os.makedirs(samples_root, exist_ok=True)
    shard_dir = os.path.join(out, "_shards"); os.makedirs(shard_dir, exist_ok=True)
    score_dir = os.path.join(out, "_dense_scores")
    adapter_dir = os.path.join(out, "_adapter_current")
    csv_path = os.path.join(out, "reward_curve.csv")
    with open(os.path.join(out, "config.json"), "w") as f: json.dump(vars(args), f, indent=2)
    task_ids = [int(x) for x in args.task_ids.split(",")]
    worker_gpus = [int(x) for x in args.worker_gpus.split(",")]
    dense_on = args.reward_mode in ("dense_raw", "dense_calib")

    log(f"=== reward_mode={args.reward_mode}  device={DEV}  worker_gpus={worker_gpus} ===")
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                          target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                          task_type="CAUSAL_LM")
    if args.init_adapter:
        vla, unnorm = mc.load_minivla(DEV, adapter_dir=args.init_adapter, variant=args.variant)
    else:
        vla, unnorm = mc.load_minivla(DEV, lora_cfg=lora_cfg, variant=args.variant)
    if args.full_ft:
        # merge any LoRA into the base so the update is a single full-parameter model, then unfreeze
        # the entire LLM backbone. save_pretrained on the merged plain HF model saves the FULL llm,
        # which minivla_common's full-dir detection loads back in workers/evals.
        if hasattr(vla.llm_backbone.llm, "merge_and_unload"):
            vla.llm_backbone.llm = vla.llm_backbone.llm.merge_and_unload()
        for p in vla.llm_backbone.llm.parameters():
            p.requires_grad_(True)
        log(f"full_ft: LLM backbone unfrozen "
            f"({sum(p.numel() for p in vla.llm_backbone.llm.parameters())/1e6:.0f}M params trainable)")
    A = vla.get_action_dim(unnorm)
    trainable = [p for p in vla.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    # lora_B abs-sum: fresh PEFT init is exactly 0 -> nonzero PROVES the warm-start weights loaded
    bsum = sum(float(p.detach().abs().sum()) for n, p in vla.llm_backbone.llm.named_parameters() if "lora_B" in n)
    log(f"trainable LoRA params: {sum(p.numel() for p in trainable):,}  action_dim={A}  "
        f"lora_B_abs_sum={bsum:.3f} ({'warm-start '+args.init_adapter if args.init_adapter else 'fresh'})")

    # persistent two-step-alignment state (dense_calib only) — survives across train steps
    recal = va.OnlineRecal(args.recal_min_pairs, args.recal_refit_every, args.recal_bins)
    adaw = va.Adaw(args.adaw_halflife, args.adaw_warmup, args.adaw_wmax, args.adaw_target_w)

    def q_logprob(q, grad):
        img = q["image"]; instr = q["instruction"]; toks = torch.as_tensor(q["action_tokens"], device=DEV)
        input_ids, pv, _ = mc.build_inputs(vla, img, instr, DEV)
        return mc.per_token_logprobs(vla, input_ids, pv, toks, A, grad=grad)

    csv_fields = ["train_step","reward_mode","n_rollouts","mean_reward","success_rate","mean_dense","n_groups",
                  "n_groups_with_var","n_anchor","recal_nfit","recal_reduction","adaw_w","mean_abs_adv","loss",
                  "grad_norm","approx_kl","clipfrac","lr","temperature","n_queries","collect_s","score_s",
                  "update_s","wall_s","timestamp"]
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f: csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    def make_specs(tstep):
        # init_idx = g % 16: training NEVER touches init states 16-49 (the held-out eval pool),
        # for any group_size. group_size>16 => multiple sampled rollouts per training init.
        specs = []
        for tid in task_ids:
            for g in range(args.group_size):
                specs.append((tid, g % 16, args.seed*100003 + tstep*997 + tid*131 + g))
        return specs

    global_opt = 0
    for tstep in range(args.num_train_steps):
        t0 = time.time()
        if os.path.isdir(adapter_dir): shutil.rmtree(adapter_dir)
        vla.llm_backbone.llm.save_pretrained(adapter_dir)
        specs = make_specs(tstep)
        # shard by (task, <=32-spec chunk), round-robin over workers: same-task rollouts batch
        # padding-free (REVIEW_speedup.md); 32-cap balances load and keeps GPU mem ~26 GB at
        # group 64 (gate-3). For group<=32 a task = one chunk => identical to shard-by-task.
        by_task = {}
        for sp in specs: by_task.setdefault(sp[0], []).append(sp)
        chunks = []
        for tid in sorted(by_task):
            sps = by_task[tid]
            for s in range(0, len(sps), 32): chunks.append(sps[s:s+32])
        # v11 multitask OOM fix (2026-07-22): ONE TASK-CHUNK PER WORKER PROCESS, executed in
        # rounds of <=1 process per worker GPU. A single process cycling 5 task scenes
        # accumulated renderer/env GPU memory and OOM'd by step 2 (killed sparse+raw arms);
        # a fresh process per task bounds memory to the known-good single-task footprint.
        # Cost: model reload per task (~2 min); step time ~65 min at 10 tasks / 2 GPUs.
        queues = {g: [] for g in worker_gpus}
        for k, ch in enumerate(chunks):
            queues[worker_gpus[k % len(worker_gpus)]].append(ch)
        for f in glob.glob(os.path.join(shard_dir, "*.pt")): os.remove(f)
        rollouts = []
        rounds = max(len(q) for q in queues.values())
        for rnd in range(rounds):
            procs = []
            for g in worker_gpus:
                if rnd >= len(queues[g]): continue
                sp = queues[g][rnd]
                outpt = os.path.join(shard_dir, f"shard_gpu{g}_r{rnd}.pt")
                env_g = dict(os.environ)
                env_g["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"   # cuda:{g} -> physical g
                env_g["MUJOCO_EGL_DEVICE_ID"] = str(g)
                cmd = ["./envs/vla/bin/python", args.worker_script, "--gpu", str(g),
                       "--adapter", adapter_dir, "--suite", args.suite, "--rollouts", json.dumps(sp),
                       "--temperature", str(args.temperature), "--top_p", str(args.top_p),
                       "--max_steps", str(args.max_steps), "--num_wait", str(args.num_wait), "--out", outpt,
                       "--variant", args.variant]
                lf = open(os.path.join(shard_dir, f"worker_gpu{g}_r{rnd}.log"), "w")
                procs.append((g, subprocess.Popen(cmd, env=env_g, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT), outpt))
            for g, p, outpt in procs:
                rc = p.wait()
                if rc != 0 or not os.path.isfile(outpt):
                    log(f"WARNING worker gpu{g} round{rnd} rc={rc} missing={not os.path.isfile(outpt)}")
                    # Surface the worker's own stdout/stderr into train.log, mirroring what the RBM
                    # scorer branch above already does. Without this a cloud worker failure is
                    # UNDIAGNOSABLE: the worker log lives under _shards/, which the job entry's
                    # `aws s3 sync --exclude '_shards/*'` used to drop, so v22 showed only the bare
                    # rc=1 line while every arm silently produced zero training steps.
                    _wlog = os.path.join(shard_dir, f"worker_gpu{g}_r{rnd}.log")
                    if os.path.isfile(_wlog):
                        _lines = open(_wlog, errors="replace").read().splitlines()
                        _sig = [l for l in _lines if any(k in l for k in (
                            "Traceback", "Error", "error", "Exception", "raise ", "assert",
                            "CUDA", "out of memory", "Killed", "No module", "not found", "cannot"))]
                        _keep = (_sig[-15:] if _sig else _lines[-15:])
                        log("WORKER_ERR[gpu%d r%d] (%d lines, %d signal):\n%s"
                            % (g, rnd, len(_lines), len(_sig), "\n".join(_keep)))
                    continue
                rollouts.extend(torch.load(outpt))
        collect_s = time.time() - t0
        if not rollouts: log("FATAL: no rollouts collected; aborting."); break

        # ---- REWARD ASSEMBLY (the only place the 3 arms differ) ----
        for r in rollouts: r["sparse"] = 1.0 if r["success"] else 0.0
        score_s = 0.0; mean_dense = 0.0
        if dense_on:
            ts = time.time()
            scores = score_dense(rollouts, worker_gpus, args.dense_frames, score_dir, args.dense_model)
            score_s = time.time() - ts
            for i, r in enumerate(rollouts):
                sc = scores.get(i)
                prog = sc["prog"] if sc else []; succ = sc["succ"] if sc else []
                r["dense"] = va.dense_scalar(prog, succ, mode=args.dense_feature, smooth_k=args.smooth_k)
            mean_dense = float(np.mean([r["dense"] for r in rollouts]))

        groups = {}
        for r in rollouts: groups.setdefault(r["task_id"], []).append(r)

        n_anchor = 0
        if args.reward_mode == "sparse":
            for r in rollouts: r["reward"] = r["sparse"]
        elif args.reward_mode == "sparse_scarce":
            # costly-label regime: only ~anchor_ratio of each group (at random) gets its sparse label;
            # the rest are UNGRADED (reward=None) -> DROPPED from the group advantage (not zero-reward).
            rng = random.Random(args.seed*7919 + tstep)
            for tid, rs in groups.items():
                k = max(1, int(round(args.anchor_ratio * len(rs))))
                idx = set(rng.sample(range(len(rs)), min(k, len(rs))))
                for j, r in enumerate(rs):
                    r["is_anchor"] = (j in idx)
                    r["reward"] = r["sparse"] if j in idx else None
                    if j in idx: n_anchor += 1
        elif args.reward_mode == "dense_raw":
            for r in rollouts: r["reward"] = r["dense"]
        elif args.reward_mode == "dense_calib":
            # 1) designate anchors (all, by default: sparse is free here) & fit recal f on anchor pairs
            rng = random.Random(args.seed*7919 + tstep)
            for tid, rs in groups.items():
                if args.anchor_ratio >= 1.0:
                    idx = set(range(len(rs)))
                else:
                    k = max(1, int(round(args.anchor_ratio * len(rs))))
                    idx = set(rng.sample(range(len(rs)), min(k, len(rs))))
                for j, r in enumerate(rs): r["is_anchor"] = (j in idx)
            for r in rollouts:
                if r["is_anchor"]:
                    recal.push(r["dense"], r["sparse"]); n_anchor += 1
            # 2) reward assembly. "mix" = FAITHFUL WMRL/B port (v8 fix, 2026-07-19): anchored
            # rollouts blend TRUE label with calibrated dense, r=(w*gt + f(dense))/(1+w) — restores
            # the success-imitation advantage that pure f(dense) blunts (case analysis: calib gave
            # successes adv +1.26 vs sparse's +1.72; MLE w-A/B: reward monotone in w, w=2 default).
            for r in rollouts:
                if r["is_anchor"] and args.anchor_reward == "sparse":
                    r["reward"] = r["sparse"]
                elif r["is_anchor"] and args.anchor_reward == "mix":
                    r["reward"] = (args.anchor_mix_w * r["sparse"] + recal.apply(r["dense"])) / (1.0 + args.anchor_mix_w)
                else:
                    r["reward"] = recal.apply(r["dense"])

        # ---- GRPO group-normalized advantage over GRADED members (== baseline when all graded) ----
        # reward=None (sparse_scarce ungraded) => dropped: no 'advantage' key, excluded from the loss;
        # groups with <2 graded members are skipped whole. Logic + unit test: vla_align.group_advantages.
        gstats = va.group_advantages(groups, adv_eps=args.adv_eps, min_graded=2,
                                     no_std=bool(args.adv_no_std))
        n_var = gstats["n_var"]
        graded_rw = [r["reward"] for r in rollouts if r.get("reward") is not None]
        mean_reward = float(np.mean(graded_rw)) if graded_rw else 0.0    # over GRADED rollouts
        success_rate = float(np.mean([r["sparse"] for r in rollouts]))   # oracle true task metric, ALL rollouts

        # ---- Step 2: ADAW up-weight anchor advantages by w_t = clip(1+c*eta2,1,wmax) ----
        adaw_w = 1.0
        if args.reward_mode == "dense_calib" and args.adaw:
            for tid, rs in groups.items():
                anch = [r for r in rs if r.get("is_anchor")]
                if len(anch) >= 2:
                    e2 = va.group_eta2([recal.apply(r["dense"]) for r in anch], [r["sparse"] for r in anch])
                    if e2 is not None: adaw.push_eta2(e2)
            adaw.maybe_calibrate()
            adaw_w = adaw.w_t()
            for r in rollouts:
                if r.get("is_anchor") and "advantage" in r: r["advantage"] *= adaw_w

        samples = []; K = args.loss_steps_per_rollout
        for r in rollouts:
            if "advantage" not in r: continue   # ungraded/dropped (scarce) or member of a skipped group
            qs = r["queries"]
            idxs = sorted(random.sample(range(len(qs)), K)) if (K and len(qs) > K) else range(len(qs))
            for i in idxs: samples.append((r["queries"][i], r["advantage"]))
        mean_abs_adv = float(np.mean([abs(a) for _, a in samples])) if samples else 0.0

        # ---- teacher-forced clipped GRPO update (identical to baseline) ----
        tu = time.time()
        loss_val = gn_val = kl_val = clip_val = 0.0; nz = 0
        opt.zero_grad(set_to_none=True); vla.train()
        if args.batched_update:
            # microbatched path: rows grouped by instruction (same prompt length -> stackable);
            # math identical to the serial path (single epoch, per-row token-mean, /len(samples)).
            by_instr = {}
            for i, (q, adv) in enumerate(samples):
                if adv == 0.0: continue
                by_instr.setdefault(q["instruction"], []).append(i)
            MB = max(1, args.update_microbatch)
            _OLD_LP = {}                       # frozen reference logprobs, filled on inner epoch 0
            for _ep in range(args.inner_epochs):
                if _ep:                        # epoch 0's zero_grad already happened above
                    opt.zero_grad(set_to_none=True)
                for instr, idx_list in by_instr.items():
                    if _ep == 0:
                        random.shuffle(idx_list)   # fix the chunking on epoch 0 so cache keys hold
                    for s in range(0, len(idx_list), MB):
                        chunk = idx_list[s:s+MB]
                        ids_l, pv_l, tok_l, adv_l = [], [], [], []
                        for i in chunk:
                            q, adv = samples[i]
                            ids, pv, _ = mc.build_inputs(vla, q["image"], q["instruction"], DEV)
                            ids_l.append(ids); pv_l.append(pv)
                            tok_l.append(torch.as_tensor(q["action_tokens"], device=DEV))
                            adv_l.append(adv)
                        ids_b = torch.cat(ids_l, dim=0)
                        pv_b = ({k: torch.cat([p[k] for p in pv_l], dim=0) for k in pv_l[0]}
                                if isinstance(pv_l[0], dict) else torch.cat(pv_l, dim=0))
                        tok_b = torch.stack(tok_l, dim=0)
                        _key = (instr, s)
                        if _ep == 0:                       # compute ONCE, then freeze
                            with torch.no_grad():
                                _OLD_LP[_key] = mc.per_token_logprobs_batched(
                                    vla, ids_b, pv_b, tok_b, A, grad=False).detach()
                        old_lp_b = _OLD_LP[_key]
                        new_lp_b = mc.per_token_logprobs_batched(vla, ids_b, pv_b, tok_b, A, grad=True)
                        ratio = torch.exp(new_lp_b - old_lp_b)                     # [B,A]
                        adv_t = torch.tensor(adv_l, device=DEV, dtype=ratio.dtype).unsqueeze(1)
                        loss_tok = -torch.min(ratio*adv_t, torch.clamp(ratio, 1-args.clip_eps, 1+_clip_hi)*adv_t)
                        loss_rows = loss_tok.mean(dim=1)                            # per-row token-mean
                        (loss_rows.sum() / max(1, len(samples))).backward()
                        nz += len(chunk)
                        loss_val += float(loss_rows.sum().item())
                        with torch.no_grad():
                            kl_val += float((old_lp_b-new_lp_b).mean(dim=1).sum().item())
                            clip_val += float(((ratio-1).abs() > args.clip_eps).float().mean(dim=1).sum().item())
                # PPO step per INNER EPOCH. Without stepping here, E epochs would accumulate into
                # one update, the weights would never move between epochs, old_lp would still equal
                # new_lp and ratio would stay 1 — the trust region would not exist. inner_epochs=1
                # falls through to the original single step below, byte-identical to every prior run.
                if args.inner_epochs > 1:
                    gn_val = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
                    opt.step(); global_opt += 1
        else:
            old_lp = []
            with torch.no_grad():
                for q, _ in samples: old_lp.append(q_logprob(q, grad=False).detach())
            order = list(range(len(samples))); random.shuffle(order)
            for i in order:
                q, adv = samples[i]
                if adv == 0.0: continue
                nz += 1
                new_lp = q_logprob(q, grad=True)
                ratio = torch.exp(new_lp - old_lp[i])
                adv_t = torch.tensor(adv, device=DEV, dtype=ratio.dtype)
                loss_tok = -torch.min(ratio*adv_t, torch.clamp(ratio, 1-args.clip_eps, 1+_clip_hi)*adv_t)
                (loss_tok.mean() / max(1, len(samples))).backward()
                loss_val += float(loss_tok.mean().item())
                with torch.no_grad():
                    kl_val += float((old_lp[i]-new_lp).mean().item())
                    clip_val += float(((ratio-1).abs() > args.clip_eps).float().mean().item())
        if args.batched_update and args.inner_epochs > 1:
            gn = torch.tensor(gn_val)                 # already stepped once per inner epoch above
        else:
            gn = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step(); global_opt += 1
        vla.eval()
        nz = max(1, nz); loss_val/=nz; kl_val/=nz; clip_val/=nz; gn_val=float(gn)
        update_s = time.time() - tu; wall = time.time() - t0

        row = {"train_step": tstep, "reward_mode": args.reward_mode, "n_rollouts": len(rollouts),
               "mean_reward": round(mean_reward,4), "success_rate": round(success_rate,4),
               "mean_dense": round(mean_dense,4), "n_groups": len(groups), "n_groups_with_var": n_var,
               "n_anchor": n_anchor, "recal_nfit": recal.nfit,
               "recal_reduction": round(recal.reduction,4) if recal.reduction is not None else "",
               "adaw_w": round(adaw_w,4), "mean_abs_adv": round(mean_abs_adv,4), "loss": round(loss_val,6),
               "grad_norm": round(gn_val,6), "approx_kl": round(kl_val,6), "clipfrac": round(clip_val,4),
               "lr": args.lr, "temperature": args.temperature, "n_queries": len(samples),
               "collect_s": round(collect_s,1), "score_s": round(score_s,1), "update_s": round(update_s,1),
               "wall_s": round(wall,1), "timestamp": datetime.now().isoformat(timespec="seconds")}
        with open(csv_path, "a", newline="") as f: csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
        log(f"STEP {tstep} [{args.reward_mode}]: success={success_rate:.3f} reward={mean_reward:.3f} "
            f"dense={mean_dense:.3f} groups_var={n_var}/{len(groups)} |adv|={mean_abs_adv:.3f} "
            f"recal(nfit={recal.nfit},red={recal.reduction}) adaw_w={adaw_w:.3f} "
            f"loss={loss_val:.4f} collect={collect_s:.0f}s score={score_s:.0f}s update={update_s:.0f}s")

        if tstep < args.save_samples_iters:
            _save_samples(samples_root, tstep, rollouts, args)

        vla.llm_backbone.llm.save_pretrained(os.path.join(out, "lora_adapter_latest"))
        if args.save_ckpt_every > 0 and (tstep + 1) % args.save_ckpt_every == 0:
            vla.llm_backbone.llm.save_pretrained(os.path.join(out, "ckpts", f"step_{tstep:02d}"))
        del rollouts, samples
        if not args.batched_update: del old_lp   # only the serial path defines it
        torch.cuda.empty_cache()   # return learner cache to the driver: next step's worker needs ~26GB on this same GPU

    log(f"DONE. {global_opt} optimizer steps.  ({args.reward_mode})")


def _save_samples(samples_root, tstep, rollouts, args):
    from PIL import Image
    it_dir = os.path.join(samples_root, f"iter_{tstep:02d}"); os.makedirs(it_dir, exist_ok=True)
    idx = []
    rollouts = rollouts[:48]   # cap disk: at group 64 full dumps would be ~8 GB/iter
    for ri, r in enumerate(rollouts):
        rd = os.path.join(it_dir, f"rollout_{ri:02d}_task{r['task_id']:02d}_init{r['init_idx']:02d}_"
                                  f"{'SUCCESS' if r['success'] else 'fail'}"); os.makedirs(rd, exist_ok=True)
        for qi, q in enumerate(r["queries"]):
            if qi % args.sample_img_every == 0:
                Image.fromarray(q["image"]).save(os.path.join(rd, f"query_{qi:03d}_image.png"))
        meta = {"iter": tstep, "reward_mode": args.reward_mode, "task_id": r["task_id"], "init_idx": r["init_idx"],
                "instruction": r["instruction"], "sparse_success": r["sparse"], "success": r["success"],
                "dense_reward": r.get("dense"), "is_anchor": r.get("is_anchor"),
                "final_reward_into_GRPO": r.get("reward"),
                "group_normalized_advantage": r.get("advantage"),   # null => UNGRADED/dropped (scarce mode)
                "in_loss": ("advantage" in r),
                "num_env_steps": r["num_env_steps"]}
        with open(os.path.join(rd, "rollout.json"), "w") as f: json.dump(meta, f, indent=2)
        idx.append({"rollout": ri, "task_id": r["task_id"], "init_idx": r["init_idx"], "sparse": r["sparse"],
                    "dense": r.get("dense"), "reward": r.get("reward"), "is_anchor": r.get("is_anchor"),
                    "advantage": r.get("advantage"), "in_loss": ("advantage" in r),
                    "dir": os.path.basename(rd)})
    with open(os.path.join(it_dir, "index.json"), "w") as f:
        json.dump({"train_step": tstep, "reward_mode": args.reward_mode, "rollouts": idx}, f, indent=2)


if __name__ == "__main__":
    main()
