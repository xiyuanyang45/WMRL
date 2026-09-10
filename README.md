# WMRL — Scaling Automatic Research Agents via World Models

[**Paper**](https://arxiv.org/abs/2608.12564) · [**Project page**](https://xiyuanyang45.github.io/WMRL/)

Reinforcement learning on world model rewards, corrected by a thin stream of
real execution.

Training a research agent with RL is bottlenecked by the environment, not the
model. Agent generation batches, so extra trajectories are nearly free.
Execution does not: every candidate solution needs its own sandbox on real
machine time. Past a certain scale, grading *is* the training cost.

WMRL hands the grading to a world model that predicts the outcome instead of
measuring it. That makes grading batch like generation, and removes the
bottleneck. The predicted reward is wrong in two separable ways, and each gets
one mechanism:

| | corrects | how |
|---|---|---|
| **Online Debiasing** | the systematic error | a monotone map fit online against anchor pairs, refit as the error drifts |
| **Inverse-Variance Denoising** | the random error | fuses the anchor and world model streams weighted by inverse variance |

Both are driven by the **anchor stream**: about a tenth of groups are graded by
both the world model and real execution, and those pairs are the only ground
truth in the loop.

At 4B and 9B this trains on roughly a third of the compute and still scores
higher on both held-out benchmarks. The 9B agent beats an off-the-shelf agent
thirteen times its size.

## What is here

```
wmrl/              the algorithm. numpy only, no framework, no GPU
tests/             65 tests, including three simulated nodes shaking hands
ml_research/       track 1: AutoResearch agents on MLE-Dojo and DSBench
embodied/          track 2: VLA post-training on LIBERO-Long
docs/              the project page
```

`wmrl/` is the part worth reading first. It is four short modules and depends on
nothing but numpy, so the method can be lifted into another codebase without
taking any of the rest.

```python
from wmrl import AnchorScheduler, InverseVarianceWeighter, OnlineDebiaser

debias   = OnlineDebiaser(min_pairs=200, refit_every=64)
weighter = InverseVarianceWeighter(target_weight=2.0)
anchors  = AnchorScheduler(fraction=0.10, min_per_step=1)

for step in training_steps:
    anchor_ids = set(anchors.select(n_groups))
    for gid, group in enumerate(groups):
        if gid in anchor_ids:                  # buy ground truth
            r_env, r_wm = sandbox(group), world_model(group)
            debias.push_many(r_wm, r_env)      # feeds the calibration
            weighter.observe_group(debias.apply(r_wm), r_env)
            scores, source = r_env, "env"
        else:                                  # cheap, corrected
            scores, source = debias.apply(world_model(group)), "wm"
```

Before it has seen enough anchor pairs the calibration is exactly the identity
and the anchor weight is exactly 1, so a run without ground truth degrades to
training on raw world model rewards rather than to something undefined.

## Running a training job

A run is three nodes, one per role:

| node | role | what it does |
|---|---|---|
| 0 | `trainer` | policy optimisation and rollout generation |
| 1 | `world_model` | one inference engine per GPU, serving predicted rewards |
| 2 | `sandbox` | executes candidate solutions for the anchor stream |

The split is the point. Rollout and world model inference batch together;
sandbox execution cannot batch at all, so it is fenced onto its own node where
it can be scaled or starved without touching the other two.

```bash
./ml_research/run.sh setup                 # dependencies, then the test suite
./ml_research/run.sh data                  # MLE-Dojo and DSBench, resumable

export WMRL_HOSTS=node0,node1,node2        # order is role order
export WMRL_STORE=/mnt/shared/wmrl         # any path all three mount
export WMRL_RUN_ID=wmrl-9b-$(date +%F)

./ml_research/run.sh plan                  # check each node agrees who it is
./ml_research/run.sh train                 # run this on all three
./ml_research/run.sh eval checkpoints/step-155
```

`plan` starts nothing and is the cheap way to catch a host list that does not
match what the nodes call themselves. Nothing reads a scheduler-specific
variable, so the same commands work under Slurm, a manual ssh loop, or a managed
service.

More nodes are fine: any node past the third takes the `sandbox` role and widens
the anchor stream.

## Reproducing the paper

`ml_research/configs/wmrl_9b.yaml` and `wmrl_4b.yaml` are the settings behind the
two rows of Table 1, down to the 45 training competitions in
`configs/tasks_train.txt`. Worth noticing: every correction setting is identical
across the two scales. The mechanisms were not retuned per model size.

The 9B run is 3200 steps and about 20 hours on three 8-GPU nodes.

## What this release is not

The production training stack this work ran on is not public. This is a clean
reimplementation of the method and the pipeline around it, built to be read and
re-run rather than to be byte-identical to the internal runs. Baselines are not
included: the repository implements one method.

`ml_research/cluster/audit_release.py` checks the tree for infrastructure detail
that should not ship, by class rather than by literal. It runs clean, and it
runs in CI.

## Citation

```bibtex
@article{yang2026wmrl,
  title   = {Scaling Automatic Research Agents via World Models},
  author  = {Yang, Xiyuan and Sarwar, Sheikh and Cheng, Jingru and Shi, Zhan and
             Li, Duanshun and Chen, Huiyuan and Zhang, Haiyang and Fan, Xing and
             Guo, Chenlei and He, Jingrui and Liao, Zhenyu},
  journal = {arXiv preprint arXiv:2608.12564},
  year    = {2026}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
