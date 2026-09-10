#!/usr/bin/env python
"""Pure ACTOR: collect LIBERO rollouts for OpenVLA at a given LoRA adapter, dump to disk.

Loads base OpenVLA + (optional) LoRA adapter, runs the assigned (task,init) rollouts
with temperature sampling, and torch.saves a compact per-step record for the learner.
The learner (grpo_train_par.py) recomputes pixel_values from the stored consumed image
(processor is deterministic) and computes all log-probs itself. Clean actor/learner split.

Caller MUST export: MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0,1,..,7
(the Phase-0 software-EGL gotcha: one EGL device, torch pinned per-worker via --gpu).
"""
import os, io, json, time, argparse
import numpy as np

def log(*a): print("[worker]", *a, flush=True)

def preprocess_agentview(agentview_256, resize=224):
    from PIL import Image
    img = np.asarray(agentview_256)[::-1, ::-1]
    pil = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=95); buf.seek(0)
    pil = Image.open(buf).convert("RGB").resize((resize, resize), Image.LANCZOS)
    return np.asarray(pil)

def normalize_gripper_action(a, binarize=True):
    a = np.asarray(a, dtype=np.float64).copy()
    a[..., -1] = 2*(a[..., -1]-0.0)/(1.0-0.0)-1
    if binarize: a[..., -1] = np.sign(a[..., -1])
    return a
def invert_gripper_action(a):
    a = np.asarray(a).copy(); a[..., -1] = -a[..., -1]; return a

EMPTY = 29871

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--adapter", default="NONE")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--rollouts", required=True, help="json list of [task_id, init_idx, seed]")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=128)
    ap.add_argument("--num_wait", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    DEV = f"cuda:{args.gpu}"
    proc = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.ckpt, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)
    ds = os.path.join(args.ckpt, "dataset_statistics.json")
    if os.path.isfile(ds):
        with open(ds) as f: vla.norm_stats = json.load(f)
    if args.adapter and args.adapter != "NONE":
        from peft import PeftModel
        vla = PeftModel.from_pretrained(vla, args.adapter)
        # need norm_stats/get_action_* on the wrapped model:
    base = vla.base_model.model if hasattr(vla, "base_model") else vla
    A = base.get_action_dim(args.suite)
    bin_centers = base.bin_centers.copy(); vdeto = int(base.vocab_size)
    st = base.get_action_stats(args.suite)
    a_low = np.array(st["q01"]); a_high = np.array(st["q99"]); a_mask = np.array(st.get("mask", np.ones_like(a_low, bool)))
    vla = vla.to(DEV).eval()

    def decode(tokens):
        disc = np.clip(vdeto - tokens - 1, 0, bin_centers.shape[0]-1)
        norm = bin_centers[disc]
        act = np.where(a_mask, 0.5*(norm+1)*(a_high-a_low)+a_low, norm)
        return invert_gripper_action(normalize_gripper_action(act, True))

    bdict = benchmark.get_benchmark_dict(); suite = bdict[args.suite]()
    specs = json.loads(args.rollouts)
    out = []
    for (task_id, init_idx, seed) in specs:
        torch.manual_seed(seed)
        task = suite.get_task(task_id)
        lang = task.language
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(0)
        inits = suite.get_task_init_states(task_id)
        prompt = f"In: What action should the robot take to {lang.lower()}?\nOut:"
        env.reset(); obs = env.set_init_state(inits[init_idx])
        steps = []; success = False; t = 0; t0 = time.time()
        while t < args.max_steps + args.num_wait:
            if t < args.num_wait:
                obs, r, d, i = env.step([0,0,0,0,0,0,-1]); t += 1; continue
            consumed = preprocess_agentview(obs["agentview_image"], 224)
            enc = proc(prompt, Image.fromarray(consumed).convert("RGB"))
            input_ids = enc["input_ids"].to(DEV)
            pv = enc["pixel_values"].to(DEV, dtype=torch.bfloat16)
            attn = enc.get("attention_mask")
            attn = attn.to(DEV) if attn is not None else torch.ones_like(input_ids)
            if not bool((input_ids[:, -1] == EMPTY).all()):
                input_ids = torch.cat([input_ids, torch.tensor([[EMPTY]], device=DEV)], 1)
                attn = torch.cat([attn, torch.ones((1,1), dtype=attn.dtype, device=DEV)], 1)
            with torch.no_grad():
                g = vla.generate(input_ids, pixel_values=pv, attention_mask=attn,
                                 max_new_tokens=A, do_sample=True, temperature=args.temperature,
                                 top_p=args.top_p, return_dict_in_generate=True, output_scores=False)
            atok = g.sequences[0, -A:].detach().cpu().numpy().astype(np.int64)
            act = decode(atok)
            steps.append({"consumed_img": consumed.astype(np.uint8),
                          "gen_input_ids": input_ids[0].detach().cpu().numpy().astype(np.int64),
                          "action_tokens": atok, "executed_action": act.tolist()})
            obs, r, d, i = env.step(act.tolist())
            if bool(d) or bool(env.check_success()):
                success = True; break
            t += 1
        reward = 1.0 if success else 0.0
        out.append({"task_id": task_id, "init_idx": init_idx, "prompt": prompt,
                    "reward": reward, "success": success, "num_steps": len(steps), "steps": steps})
        log(f"gpu{args.gpu} task{task_id} init{init_idx} success={success} steps={len(steps)} {time.time()-t0:.0f}s")
        env.close() if hasattr(env, "close") else None
    torch.save(out, args.out)
    log(f"gpu{args.gpu} wrote {len(out)} rollouts -> {args.out}")

if __name__ == "__main__":
    main()
