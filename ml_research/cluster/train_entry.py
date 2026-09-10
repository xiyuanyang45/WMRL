#!/usr/bin/env python3
"""The trainer process: wires the corrections into the RL loop and starts it.

Called by :mod:`ml_research.cluster.launch` on the trainer node once the world
model and sandbox servers have registered. Its job is threefold.

**Decide who pays for real execution.** Each step, the anchor scheduler picks
which groups go to the sandbox instead of the world model. That choice is the
whole cost model: everything else in the step is cheap.

**Correct the cheap rewards.** Anchor groups come back with both a predicted and
a measured score. Those pairs feed the calibration map, and the map is applied
to every world-model-graded score before advantages are formed. The same pairs
drive the anchor weight.

**Hand the rest to the RL trainer.** Policy optimisation, sharding, checkpoint
management and rollout scheduling are not this project's contribution, so they
are delegated to `verl <https://github.com/volcengine/verl>`_ rather than
reimplemented. This module generates its configuration and supervises it.

The reward path is the seam between the two. verl calls
:func:`reward_fn` for each finished trajectory; that function is where the
corrections live, and it is short on purpose.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

from ml_research.cluster.launch import load_config
from ml_research.grading.client import GraderPool
from wmrl import AnchorScheduler, InverseVarianceWeighter, OnlineDebiaser

REPO = pathlib.Path(__file__).resolve().parents[2]

__all__ = ["CorrectedGrader", "build_verl_overrides", "main"]


class CorrectedGrader:
    """Grades a group, and keeps the two corrections up to date while doing it.

    One instance lives for the whole run on the trainer. It owns the anchor
    schedule, the calibration map, and the anchor weight, which is why those
    three stay consistent with each other: they are all driven by the same
    stream of anchor pairs.
    """

    def __init__(self, cfg: dict, world_model_urls: list[str], sandbox_urls: list[str]):
        self.world_model = GraderPool(world_model_urls, name="world_model")
        self.sandbox = GraderPool(sandbox_urls, name="sandbox")

        groups_per_step = int(cfg.get("train_batch_size", 8))
        min_anchor = int(cfg.get("anchor_groups_per_step", 1))

        self.anchors = AnchorScheduler(
            fraction=min_anchor / max(1, groups_per_step),
            min_per_step=min_anchor,
            max_concurrent=int(cfg.get("anchor_concurrency_cap", 24)),
            seed=int(cfg.get("seed", 0)),
        )
        self.debias = OnlineDebiaser(
            min_pairs=int(cfg.get("recalibration_min_pairs", 200)),
            refit_every=int(cfg.get("recalibration_refit_every", 64)),
            bins=int(cfg.get("recalibration_bins", 10)),
        )
        self.weighter = InverseVarianceWeighter(
            half_life=float(cfg.get("anchor_weight_half_life", 64)),
            warmup=int(cfg.get("anchor_weight_warmup", 32)),
            w_max=float(cfg.get("anchor_weight_max", 4.0)),
            target_weight=float(cfg.get("anchor_weight_target", 2.0)),
        )
        self.step = 0

    def grade_step(self, groups):
        """Grade one step's groups. ``groups`` is a list of lists of trajectories.

        Returns ``(scores, sources)``: a per-group list of scores, and a
        per-group ``"env"`` or ``"wm"`` saying which stream produced it.
        """
        anchor_ids = set(self.anchors.select(len(groups), in_flight=self.sandbox.in_flight))
        scores, sources = [], []

        for gid, trajs in enumerate(groups):
            if gid in anchor_ids:
                # Anchor group: pay for execution, and spend the pairs it yields
                # on both corrections before anything else sees them.
                r_env = self.sandbox.grade(trajs)
                r_wm = self.world_model.grade(trajs)
                self.debias.push_many(r_wm, r_env)
                self.weighter.observe_group(self.debias.apply(r_wm), r_env)
                scores.append(r_env)
                sources.append("env")
            else:
                scores.append(self.debias.apply(self.world_model.grade(trajs)))
                sources.append("wm")

        self.step += 1
        return scores, sources

    def stats(self) -> dict:
        """What to log each step. All four move, and all four are diagnostic."""
        return {
            "step": self.step,
            "anchor_fraction": round(self.anchors.realized_fraction, 4),
            "calibrated": self.debias.calibrated,
            "calibration_fits": self.debias.n_fits,
            "bias_reduction": (
                round(self.debias.reduction, 4) if self.debias.reduction is not None else None
            ),
            "anchor_weight": round(self.weighter.weight(), 3),
            "disagreement": (
                round(self.weighter.disagreement, 4)
                if self.weighter.disagreement is not None
                else None
            ),
        }


def build_verl_overrides(cfg: dict, log_path: str) -> list[str]:
    """Translate our config into the trainer's override flags.

    Kept as one function so the mapping between what this repository documents
    and what the underlying trainer is actually told is readable in one place.
    """
    rollout_gpus = str(cfg.get("rollout_gpus", "0,1,2,3")).split(",")
    train_gpus = str(cfg.get("train_gpus", "4,5,6,7")).split(",")

    ov = [
        f"actor_rollout_ref.model.path={cfg['model_path']}",
        f"actor_rollout_ref.actor.optim.lr={cfg.get('learning_rate', 1e-6)}",
        f"actor_rollout_ref.actor.kl_loss_coef={cfg.get('kl_beta', 0.04)}",
        f"actor_rollout_ref.actor.entropy_coeff={cfg.get('entropy_coef', 0.002)}",
        f"actor_rollout_ref.rollout.n={cfg.get('group_size', 8)}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg.get('rollout_tensor_parallel', 2)}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg.get('rollout_gpu_util', 0.65)}",
        f"actor_rollout_ref.rollout.max_new_tokens={cfg.get('max_new_tokens_per_turn', 4096)}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={cfg.get('param_offload', False)}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={cfg.get('optimizer_offload', False)}",
        f"trainer.total_epochs={cfg.get('epochs', 1)}",
        f"trainer.total_training_steps={cfg.get('steps', 3200)}",
        f"trainer.save_freq={cfg.get('save_every', 5)}",
        f"trainer.n_gpus_per_node={len(rollout_gpus) + len(train_gpus)}",
        f"trainer.nnodes=1",  # the trainer role occupies one node; servers are separate
        f"trainer.default_local_dir={cfg.get('checkpoint_dir', 'checkpoints')}",
        f"data.train_batch_size={cfg.get('train_batch_size', 8)}",
        "algorithm.adv_estimator=grpo",
        f"custom_reward_function.path={REPO / 'ml_research' / 'grading' / 'reward.py'}",
        f"custom_reward_function.name=reward_fn",
    ]

    if cfg.get("async_rollout", True):
        # Bounded staleness rather than none: waiting for a full batch of
        # finished trajectories would give back the pipelining the world model
        # was introduced to buy.
        ov += [
            f"async_training.staleness_threshold={cfg.get('staleness_threshold', 9)}",
            f"async_training.sync_every_steps={cfg.get('sync_every_steps', 2)}",
            f"async_training.partial_rollout={cfg.get('partial_rollout', True)}",
            f"async_training.max_concurrent_samples={cfg.get('max_concurrent_samples', 12)}",
            f"async_training.max_queue_size={cfg.get('max_queue_size', 24)}",
        ]

    return ov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--print-overrides", action="store_true",
                    help="print what the trainer would be told, and stop")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    wm_urls = [u for u in os.environ.get("WMRL_WORLD_MODEL_URLS", "").split(",") if u]
    sbx_urls = [u for u in os.environ.get("WMRL_SANDBOX_URLS", "").split(",") if u]

    overrides = build_verl_overrides(cfg, args.log or "trainer.log")

    if args.print_overrides:
        print(json.dumps({"world_model": wm_urls, "sandbox": sbx_urls,
                          "overrides": overrides}, indent=2))
        return 0

    if not wm_urls or not sbx_urls:
        raise SystemExit("WMRL_WORLD_MODEL_URLS and WMRL_SANDBOX_URLS must both be set; "
                         "launch.py sets them after rendezvous")

    # The grader is constructed here and reached by the reward function through
    # the environment, because the trainer owns the process that calls it.
    os.environ["WMRL_GRADER_CONFIG"] = args.config

    entry = os.environ.get("WMRL_TRAINER_ENTRY", "verl.trainer.main_ppo")
    cmd = [sys.executable, "-m", entry] + overrides
    print(f"[train] {' '.join(cmd[:4])} ... ({len(overrides)} overrides)", flush=True)

    log = open(args.log, "a") if args.log else None
    try:
        return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT if log else None)
    finally:
        if log:
            log.close()


if __name__ == "__main__":
    sys.exit(main())
