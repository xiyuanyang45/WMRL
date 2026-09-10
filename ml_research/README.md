# AutoResearch track

Post-training a coding agent that reads a research question, writes an
experiment, runs it, and iterates. Benchmarks are MLE-Dojo for training and the
in-distribution test split, DSBench for out of distribution.

## Layout

```
run.sh              the only command you need
cluster/            topology, shared storage, rendezvous, the launcher
agent/              the scaffold: rollout, environment, turn loop, scheduler
world_model/        the prediction server, its prompts, few-shot curation
grading/            sandbox execution server, grader clients, anchor transport
eval/               leaderboard-percentile scoring for both benchmarks
data/               download and preparation
configs/            the two runs from the paper, and the training task list
```

## Before a run

**Dependencies.** `run.sh setup` installs what this repository needs and runs the
test suite. The training loop additionally needs an inference backend and an RL
trainer. Those are not pinned in `requirements.txt` because they are large,
CUDA-version-sensitive, and change fast; installing them is the part of setup
that depends on your cluster rather than on this code.

These runs were built against vLLM for inference and verl for the RL loop, with
the trainer entry point selectable through `WMRL_TRAINER_ENTRY`. A different
trainer can be substituted: the seam is
`ml_research/cluster/train_entry.py`, which translates this repository's config
into that trainer's flags and points its reward hook at
`ml_research/grading/reward.py`.

**Data.** `run.sh data` prepares each competition into a public half the agent
sees and a private half it never does. It is resumable and reports per-task
failures at the end instead of aborting. Most failures are competition rules
that have to be accepted on the competition page with the same Kaggle account
before the data can be downloaded; accept them and re-run.

**Weights.** The same base model serves as both the agent and the world model.
That is deliberate: with no stronger model anywhere in the loop, an improvement
cannot be distillation from one.

## The three roles

Every node runs `run.sh train`. Each resolves its own role from the position of
its hostname in `WMRL_HOSTS`.

- **trainer** waits for the other two to publish their server URLs, then
  optimises against them. When it stops, for any reason, it marks the run done,
  which is how the other nodes learn to exit.
- **world_model** starts one inference engine per GPU, all serving the agent's
  backbone. One engine per card rather than one tensor-parallel engine across
  cards: predictions are short, so throughput is what matters.
- **sandbox** starts the execution server. Its slot count bounds the anchor
  stream, and therefore bounds how quickly the calibration can track drift.

A trainer that comes up without a sandbox stops rather than proceeding. Without
an anchor stream neither correction can run, and the run would otherwise look
healthy while quietly training on uncorrected rewards.

## Where the method actually enters

Two places, and they are both short.

`cluster/train_entry.py` holds `CorrectedGrader`, which owns the anchor
schedule, the calibration map, and the anchor weight for the whole run. Each
step it decides which groups pay for execution, grades them, and spends the
resulting pairs on both corrections before anything else sees the scores.

`grading/reward.py` is the hook the RL trainer calls per finished trajectory.

Everything else in this directory is the machinery around those: getting
trajectories generated, getting them graded, and getting the numbers back out.

## Logging

Four numbers per step are worth watching, all of them printed by
`CorrectedGrader.stats()`:

- `anchor_fraction` — drifts below target when the sandbox concurrency cap
  binds, which is the signal that the sandbox pool is setting the pace.
- `calibrated` — false until enough anchor pairs exist. While it is false the
  run is training on raw world model rewards.
- `bias_reduction` — how much within-group bias the current map removes.
- `anchor_weight` — 1.0 until calibrated, then rises with measured disagreement.
