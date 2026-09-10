# Embodied track

The same two corrections, applied to something that is not a coding agent at
all: post-training a vision-language-action policy on LIBERO-Long.

This exists to show the mechanisms are not specific to AutoResearch. Nothing in
`wmrl/` knows what a trajectory contains, so the only thing that changes here is
what plays each part.

| | AutoResearch | Embodied |
|---|---|---|
| agent | a coding agent | MiniVLA-1B |
| world model | the agent's own backbone, prompted for the outcome | Robometer, an off-the-shelf progress predictor |
| cheap reward | predicted leaderboard score | per-frame progress, collapsed to one scalar |
| anchor | real sandbox execution | the one sparse success the simulator returns |

The asymmetry is sharper here than in the AutoResearch setting, which makes it a
good test. The dense signal is available on every frame of every rollout; the
sparse one arrives once, at the end. Either alone barely moves the policy: RL on
the sparse outcome adds 0.9 points over the SFT baseline and RL on the raw dense
signal adds 1.8. Combined under the two corrections it adds 3.8, and the widest
margin is on initial states the policy never saw in training.

## Running it

Single node, unlike the AutoResearch track: the simulator is cheap enough that
fencing it onto its own machine buys nothing.

```bash
./embodied/run.sh setup
./embodied/run.sh data
./embodied/run.sh sft      # every RL run starts here
./embodied/run.sh train
./embodied/run.sh eval
```

The SFT step is not optional and not a formality. All the reported RL numbers
start from the same behaviour-cloned checkpoint, which is what makes the spread
between reward signals attributable to the signal rather than to where each run
happened to start.

## Dependencies not pinned here

- **LIBERO** — the benchmark and its demonstrations.
- **MiniVLA** — the 1B policy, pretrained on LIBERO-90.
- **Robometer** — the progress predictor used as the world model. Upstream, not
  vendored: it is a separate piece of work with its own license, and copying it
  in would obscure that.

Headless rendering needs a working EGL or OSMesa stack. On a fresh machine that
is reliably the first thing to fail, and it fails in ways that look like a
policy bug rather than a graphics one.

## Collapsing a per-frame signal to one number

Robometer scores frames; GRPO needs one scalar per trajectory. The default is
potential-based shaping, the last-k mean of progress minus the first-k mean.
Under this loop's per-trajectory advantage the shaped return telescopes to
exactly that difference.

The alternatives are selectable and worth knowing about, because the choice is
where reward hacking enters. Taking the peak progress rewards a policy that
touches a good state and then falls apart, which is precisely the behaviour a
sparse anchor catches and a dense signal alone does not.
