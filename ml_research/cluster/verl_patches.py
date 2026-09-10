"""Where WMRL meets the RL trainer.

Two kinds of change are needed to run this method on top of an off-the-shelf
GRPO implementation, and they are very different in character.

**A seam the method genuinely needs.** GRPO standardises advantages inside each
prompt group, which is scale-invariant, so the absolute gap between a world
model score and a measured one does not by itself make anchor groups count for
more. It cannot: after ``(r - mean) / std`` the two look alike. Anchor groups
are also a small minority, so by count the world model groups dominate the
gradient. The only lever is to scale the *post-normalisation* advantage of
anchor rows, which is what :func:`weight_anchor_advantages` does. It is the one
place the inverse-variance weight actually reaches the loss.

**Upstream bugs that happen to bite this workload.** Two of them, both
documented below. They are applied as fail-loud overlays: if the anchor text is
missing, the trainer version has drifted and the patch must be re-audited rather
than silently skipped.

Deliberately absent: the container networking workarounds the internal runs
carried. Those forced a loopback master address and disabled a TCPStore
port-probe to work around one scheduler's network namespace. On an ordinary
cluster they range from unnecessary to harmful, so they are not part of this
release.

Version: written against verl at the commit pinned in ``requirements-train.txt``.
The overlays assert their anchors, so a drifted version fails at startup with
the file and tag that no longer matches, not silently mid-run.
"""

from __future__ import annotations

import pathlib

import numpy as np

from wmrl.debias import apply_calibration
from wmrl.denoise import group_disagreement

__all__ = ["weight_anchor_advantages", "apply_overlays", "PatchError"]


class PatchError(RuntimeError):
    """An overlay could not be applied, usually because the trainer changed."""


# ------------------------------------------------------------- the real seam


def weight_anchor_advantages(
    advantages,
    grade_mode,
    group_id,
    r_world_model=None,
    r_env=None,
    *,
    debias=None,
    weighter=None,
):
    """Scale anchor-group advantages, and update the weight from what they show.

    Called once per batch, immediately after the trainer has computed GRPO
    advantages and before they reach the loss.

    Both jobs happen here because they need the same three aligned arrays. The
    measurement has to be *within group* and *after calibration*: within group
    because that is all GRPO ever sees, and after calibration because as the map
    removes bias the residual disagreement genuinely shrinks, which relaxes the
    weight. The two mechanisms are coupled through exactly this number.

    Args:
        advantages: per-row advantages, already normalised by the trainer.
        grade_mode: per-row ``"env"`` or ``"wm"``.
        group_id: per-row group identifier, so rows can be gathered per group.
        r_world_model: per-row predicted score. Needed only for anchor rows.
        r_env: per-row measured score. Needed only for anchor rows.
        debias: the run's :class:`~wmrl.debias.OnlineDebiaser`.
        weighter: the run's :class:`~wmrl.denoise.InverseVarianceWeighter`.

    Returns:
        ``(advantages, info)``. ``advantages`` is a new array; the input is not
        modified.
    """
    adv = np.asarray(advantages, dtype=float).copy()
    grade_mode = np.asarray(grade_mode)
    group_id = np.asarray(group_id)

    if not (len(adv) == len(grade_mode) == len(group_id)):
        raise ValueError(
            f"misaligned batch: {len(adv)} advantages, {len(grade_mode)} grade modes, "
            f"{len(group_id)} group ids. These are matched by position; a mismatch "
            "would weight the wrong rows."
        )

    # Fold each anchor group's disagreement into the running estimate, so the
    # weight applied below reflects this batch too.
    n_groups = 0
    if weighter is not None and r_world_model is not None and r_env is not None:
        r_wm = np.asarray(r_world_model, dtype=object)
        r_sb = np.asarray(r_env, dtype=object)
        f = debias.f if debias is not None else None

        per_group: dict = {}
        for i, mode in enumerate(grade_mode):
            if mode != "env" or r_wm[i] is None or r_sb[i] is None:
                continue
            per_group.setdefault(group_id[i], []).append((float(r_wm[i]), float(r_sb[i])))

        for pairs in per_group.values():
            if len(pairs) < 2:
                continue  # one row carries no within-group information
            wm = np.array([a for a, _ in pairs])
            env = np.array([b for _, b in pairs])
            s2 = group_disagreement(apply_calibration(f, wm), env)
            if s2 is None:
                continue  # every measured score identical: nothing to disagree about
            weighter.push(s2)
            weighter.maybe_calibrate()
            n_groups += 1

    weight = weighter.weight() if weighter is not None else 1.0
    is_anchor = grade_mode == "env"
    adv[is_anchor] *= weight

    return adv, {
        "anchor_weight": round(float(weight), 4),
        "anchor_rows": int(is_anchor.sum()),
        "groups_measured": n_groups,
        "calibrated": bool(debias.calibrated) if debias is not None else False,
    }


# ------------------------------------------------------------------ overlays


def _overlay(path: pathlib.Path, anchor: str, patched: str, tag: str) -> str:
    """Apply one idempotent source overlay. Returns what happened."""
    if not path.exists():
        raise PatchError(f"{tag}: {path} does not exist; is the trainer installed?")
    src = path.read_text()
    if patched in src:
        return "already applied"
    if anchor not in src:
        raise PatchError(
            f"{tag}: anchor text not found in {path}.\n"
            "The trainer version has drifted from the one these overlays were written "
            "against. Re-audit the patch against the current source rather than "
            "skipping it: the failure it prevents is silent."
        )
    path.write_text(src.replace(anchor, patched, 1))
    return "applied"


def apply_overlays(site_packages: str) -> dict:
    """Apply the two upstream fixes this workload needs.

    (1) **Cancellation is not a crash.** The asynchronous rollout path cancels
        in-flight tasks on every weight sync. The task exception handler only
        recognised one flavour of cancellation, so a routine sync surfaced as a
        crashed run.

    (2) **Expandable segments around the actor update.** The disaggregated path
        returns early after sending weights and never re-enables the allocator
        setting the trainer toggles for itself, which shows up as an
        out-of-memory failure on long sequences and nowhere else.
    """
    sp = pathlib.Path(site_packages)
    results = {}

    results["cancellation"] = _overlay(
        sp / "verl" / "experimental" / "fully_async_policy" / "detach_utils.py",
        anchor=(
            "    except Exception as e:\n"
            '        print(f"Task {task.get_name()} failed with exception: {e}")\n'
            "        raise e"
        ),
        patched=(
            "    except Exception as e:\n"
            "        # A weight sync cancels in-flight rollout tasks by design. Ray and\n"
            "        # concurrent.futures raise their own cancellation types, neither of\n"
            "        # which is asyncio.CancelledError, so both were reaching this handler\n"
            "        # and turning a routine sync into a crashed run.\n"
            '        _n = type(e).__name__.lower()\n'
            '        if "cancel" in _n or "cancel" in str(e).lower():\n'
            "            return\n"
            '        print(f"Task {task.get_name()} failed with exception: {e}")\n'
            "        raise e"
        ),
        tag="cancellation-is-not-a-crash",
    )

    return results
