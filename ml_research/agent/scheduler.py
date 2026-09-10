#!/usr/bin/env python3
"""
Standardized sandbox resource scheduler (aligns with MLE-Dojo run.py's queue scheduling).

EVERY sandbox gets the SAME standardized resource quota — there is NO "this task uses GPU, that one doesn't".
Each grade is assigned one GPU from a fixed pool + identical cpu/mem limits; the model's own code decides whether
to actually use the GPU (CNN -> uses it; sklearn -> ignores it). Standard, paper-defensible design.

HARDWARE-FLEXIBLE BY DESIGN. Two INDEPENDENT card groups, set purely by config (env), never hardcoded:
  TRAIN_GPU  -> colocate cards: FSDP trainer + in-process TP vLLM (time-shared)
  SBX_GPUS   -> this grading pool
Moving 8x40GB node(8x40G) <-> 8x80GB node(8x80G) is just changing those two lists.

CROSS-PROCESS POOL. Under FSDP the trainer is N processes; TRL hands each rank its own slice of the per-step batch,
and each rank grades its own rollouts. So the pool MUST be shared across processes — otherwise either the ranks
oversubscribe a card, or we'd have to statically partition cards per rank (rigid, needs #cards>=#ranks). `SandboxPool`
coordinates with atomic lock files in a shared dir: any rank borrows ANY free card; when all K are busy it blocks.
A per-step batch of B=G*tasks rollouts (however split across ranks) is thus graded across K cards in ceil(B/K) rounds.
FAIL LOUD on harness errors; a crashed grade's lock is reclaimed (dead holder) so it never wedges a card.
"""
import os, time


class SandboxPool:
    """Cross-process GPU pool for grading sandboxes, shared by ALL processes that borrow from it (FSDP ranks in
    the single-node path; the env_server's request threads in the multi-node path).

    `borrow()` blocks until any (card, slot) is free and returns a handle `(gpu_id, slot_idx, lockfile)`; pass it
    back to `return_slot()` when the grade finishes. `slot_idx` is the GLOBAL slot index (0..capacity-1) — the
    grader uses it to taskset onto a disjoint CPU-core block, which MUST be unique per concurrent grade. With
    max_tasks_per_gpu>1, two grades share a card but still get distinct slot_idx => distinct cores (the gpu id
    alone would collide). Coordination is by atomic O_CREAT|O_EXCL lock files in `lockdir` (one per card-slot),
    so it works ACROSS processes with no shared memory. A lock whose holder PID is dead is reclaimed, so a crashed
    grade can't permanently wedge a slot. Hardware-agnostic: only `gpu_ids` / `max_tasks_per_gpu` change.

    max_tasks_per_gpu>1 PACKS multiple sandboxes onto one card; the caller MUST also halve the per-sandbox
    gpu_memory_limit (env_server does this explicitly) so the packed grades fit the card."""

    def __init__(self, gpu_ids, max_tasks_per_gpu=1, lockdir=None):
        # global slot index = enumeration order; lock file keyed by (gpu, local slot)
        self.slots = [(int(g), s, i)
                      for i, (g, s) in enumerate((g, s) for g in gpu_ids for s in range(max_tasks_per_gpu))]
        if not self.slots:
            raise ValueError("SandboxPool got no GPUs — set SBX_GPUS to the grading cards.")
        self.lockdir = lockdir or f"/tmp/sbxpool_{os.environ.get('EXP', 'default')}"
        os.makedirs(self.lockdir, exist_ok=True)
        self.capacity = len(self.slots)

    @staticmethod
    def _alive(pid):
        return os.path.exists(f"/proc/{pid}")

    def borrow(self, poll=0.25):
        """Block until a card-slot is free; return (gpu_id, slot_idx, lockfile_path)."""
        while True:
            for g, s, i in self.slots:
                lf = os.path.join(self.lockdir, f"gpu{g}_slot{s}.lock")
                try:
                    fd = os.open(lf, os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # atomic cross-process acquire
                    os.write(fd, str(os.getpid()).encode())
                    os.close(fd)
                    return g, i, lf
                except FileExistsError:
                    try:                                                     # reclaim a DEAD holder's lock
                        pid = int((open(lf).read() or "0").strip())
                        if pid and not self._alive(pid):
                            os.unlink(lf)
                    except (ValueError, FileNotFoundError, OSError):
                        pass
            time.sleep(poll)                                                 # all busy -> wait (queues the batch)

    def return_slot(self, handle):
        _gpu, _slot, lf = handle
        try:
            os.unlink(lf)
        except (FileNotFoundError, OSError):
            pass
