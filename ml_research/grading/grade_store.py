"""Cross-actor grade bridge for the async grader (fully_async backend).

WHY: fully_async's streaming RewardLoopWorker populates rm_scores by calling compute_score(response_text, task). It
runs in a DIFFERENT Ray actor than our AgentLoop and on a verl-internal trajectory representation that does NOT carry
our WM/sandbox grade (verified via mledojo_reward DIAG: extra_info has only task/response; turn_scores/tool_rewards/
rollout_reward_scores empty, our extra_fields dropped). So we bridge out-of-band: the AgentLoop PUTs its grade keyed
by a stable hash of the decoded response; mledojo_reward.compute_score GETs it by the SAME key (the reward loop's
solution_str == decode(skip_special=True) of the very same response). Named detached Ray actor, shared across the
AgentLoopWorker and RewardLoopWorker actors in the rollouter's Ray cluster.

KEY = sha1 of the decoded response (NOT python hash(), which is per-process salted). Collision caveat: two distinct
trajectories with byte-identical responses but different graders would collide (rare for full multi-turn responses);
acceptable. A store miss -> mledojo_reward logs it + falls back, so the hit-rate is observable.
"""
import hashlib
import os

_STORE = None   # cached handle (per worker process)


def grade_key(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", "ignore")).hexdigest()


def _make_actor():
    import ray

    @ray.remote(num_cpus=0)
    class _GradeStore:
        def __init__(self):
            self._d = {}
            # ADAW pair store — separate from _d (never touched by put_grade/get_grade)
            self._pairs = {}            # resp_key -> (r_wm, r_sbx)
            self._eta2  = None          # EMA of within-group WM-sbx disagreement (η²)
            self._neta  = 0             # number of EMA updates
            self._c     = None          # calibrated ADAW gain (frozen after warmup)
            # WMRL (wmrl): online monotone recalibration f: r_wm -> r̂ ≈ r_sbx.
            # _rc_pairs = anchor (r_wm, r_sbx) buffer; _rc_fx/_rc_fy = fitted monotone step map for np.interp
            # (None until first fit -> recal is identity -> E ≡ B). bucketed-isotonic (PAV), numpy-only, deterministic.
            self._rc_pairs = []         # list of (r_wm, r_sbx); capped FIFO
            self._rc_fx = None          # bin centers (increasing) for np.interp
            self._rc_fy = None          # monotone non-decreasing fitted r̂ per bin
            self._rc_since = 0          # anchor pairs seen since last fit
            self._rc_nfit  = 0          # number of refits done
            self._rc_braw  = None       # last fit's within-group b̄ BEFORE recal (diagnostic)
            self._rc_brecal = None      # last fit's within-group b̄ AFTER recal (diagnostic)
            # an abandoned variant (an abandoned variant): fit φ(z_wm)=E[z_sbx|z_wm] in STANDARDIZED (within-group z) space, applied to
            # WM-group ADVANTAGES (not rewards) so std-norm GRPO cannot undo it. Isotonic in z-space; (px,py) for np.interp.
            self._zpairs   = []         # list of (z_wm, z_sbx) standardized anchor pairs; FIFO-capped
            self._phi_px   = None       # bin centers in z_wm (increasing) for np.interp
            self._phi_py   = None       # monotone non-decreasing E[z_sbx|z_wm] per bin
            self._phi_since = 0         # z-pairs seen since last φ fit
            self._phi_nfit  = 0         # number of φ refits
            self._phi_R     = None      # last fit's corr(z_wm, z_sbx) (WM↔sandbox rank corr; bounds 1-R² variance)
            self._phi_resid = None      # last fit's E[(φ(z_wm)-z_sbx)²] (should track 1-R²)

        def put(self, k, v):
            self._d[k] = float(v)

        def get(self, k):
            return self._d.get(k)

        def size(self):
            return len(self._d)

        # -- ADAW transport --
        def put_adaw_pair(self, k, r_wm, r_sbx):
            self._pairs[k] = (float(r_wm), float(r_sbx))

        def pop_adaw_pairs(self, keys):
            return {k: self._pairs.pop(k, None) for k in keys}

        def push_eta2(self, x, half_life):
            a = 1.0 - 0.5 ** (1.0 / max(1.0, float(half_life)))
            self._eta2 = float(x) if self._eta2 is None else (1 - a) * self._eta2 + a * float(x)
            self._neta += 1
            return self._eta2

        def get_eta2(self):
            return self._eta2

        def eta2_count(self):
            return self._neta

        def calibrate_c(self, target_w, rho=1.0):
            """Freeze c so that 1 + c*eta2*rho ≈ target_w. Runs once; subsequent calls return cached _c."""
            if self._c is not None:
                return self._c
            if self._eta2 is not None and self._eta2 * float(rho) > 1e-12:
                self._c = (float(target_w) - 1.0) / (self._eta2 * max(1e-12, float(rho)))
                print(f"[grade_store] ADAW c calibrated: c={self._c:.4f} eta2={self._eta2:.4f} rho={rho:.2f} target_w={target_w}", flush=True)
            return self._c

        def get_c(self):
            return self._c

        # -- WMRL: online recalibration f --
        def push_recal_pair(self, r_wm, r_sbx, min_pairs=200, refit_every=64, bins=10):
            """Buffer an anchor (r_wm, r_sbx) pair; refit f when enough new pairs accrue. FIFO-capped at 5000."""
            self._rc_pairs.append((float(r_wm), float(r_sbx)))
            if len(self._rc_pairs) > 5000:
                self._rc_pairs = self._rc_pairs[-5000:]
            self._rc_since += 1
            if len(self._rc_pairs) >= int(min_pairs) and self._rc_since >= int(refit_every):
                self._fit_recal(int(bins))

        def _fit_recal(self, bins):
            """Bucketed isotonic regression r_sbx ~ r_wm: bin r_wm, weighted-mean r_sbx per bin, PAV-monotonize.
            Stores (bin_centers, monotone_means) for np.interp. Monotone & non-affine ⇒ not cancelled by GRPO."""
            import numpy as _np
            P = _np.asarray(self._rc_pairs, dtype=float)
            x = P[:, 0]; y = P[:, 1]
            edges = _np.linspace(0.0, 1.0, int(bins) + 1)
            idx = _np.clip(_np.digitize(x, edges[1:-1]), 0, int(bins) - 1)
            cx = []; cy = []; cw = []
            for b in range(int(bins)):
                m = idx == b
                if m.sum() == 0:
                    continue
                cx.append(float(x[m].mean())); cy.append(float(y[m].mean())); cw.append(float(m.sum()))
            if len(cx) < 2:
                return  # not enough distinct bins to define a non-trivial map; stay identity
            # PAV: pool adjacent violators -> monotone non-decreasing means (weighted)
            val = []; wt = []; cnt = []
            for v, w in zip(cy, cw):
                val.append(v); wt.append(w); cnt.append(1)
                while len(val) > 1 and val[-2] > val[-1]:
                    v2 = val.pop(); w2 = wt.pop(); c2 = cnt.pop()
                    v1 = val.pop(); w1 = wt.pop(); c1 = cnt.pop()
                    nw = w1 + w2
                    val.append((v1 * w1 + v2 * w2) / nw); wt.append(nw); cnt.append(c1 + c2)
            fy = []
            for v, c in zip(val, cnt):
                fy += [v] * c
            self._rc_fx = _np.asarray(cx); self._rc_fy = _np.asarray(fy)
            self._rc_since = 0; self._rc_nfit += 1
            # diagnostic b̄: within-pair squared residual before vs after recal (lower = bias removed)
            yhat = _np.interp(x, self._rc_fx, self._rc_fy)
            self._rc_braw = float((((x - x.mean()) - (y - y.mean())) ** 2).mean())
            self._rc_brecal = float((((yhat - yhat.mean()) - (y - y.mean())) ** 2).mean())
            print(f"[grade_store] recal fit #{self._rc_nfit}: npairs={len(self._rc_pairs)} bins_used={len(cx)} "
                  f"b_raw={self._rc_braw:.5f} b_recal={self._rc_brecal:.5f} "
                  f"reduction={1.0 - self._rc_brecal / max(1e-9, self._rc_braw):.3f} "
                  f"fx={_np.round(self._rc_fx,3).tolist()} fy={_np.round(self._rc_fy,3).tolist()}", flush=True)

        def get_recal_f(self):
            """Return (fx, fy) lists for the consumer to apply f locally (np.interp), or None if unfitted (≡ identity)."""
            if self._rc_fx is None:
                return None
            return (self._rc_fx.tolist(), self._rc_fy.tolist())

        def recal_apply(self, r):
            """Apply f to one reward (monotone interp). Identity until first fit ⇒ E opens exactly as B."""
            if self._rc_fx is None:
                return float(r)
            import numpy as _np
            return float(_np.interp(float(r), self._rc_fx, self._rc_fy))

        def recal_stats(self):
            return {"nfit": self._rc_nfit, "npairs": len(self._rc_pairs),
                    "b_raw": self._rc_braw, "b_recal": self._rc_brecal}

        # -- an abandoned variant: φ(z_wm)=E[z_sbx|z_wm] in standardized space --
        def push_phi_pairs(self, zwm, zsb, min_pairs=400, refit_every=64, bins=12):
            """Buffer within-group-standardized (z_wm, z_sbx) anchor pairs; refit φ when enough new pairs accrue."""
            for a, b in zip(zwm, zsb):
                self._zpairs.append((float(a), float(b)))
            if len(self._zpairs) > 8000:
                self._zpairs = self._zpairs[-8000:]
            self._phi_since += len(zwm)
            if len(self._zpairs) >= int(min_pairs) and self._phi_since >= int(refit_every):
                self._fit_phi(int(bins))

        def _fit_phi(self, bins):
            """Isotonic fit of E[z_sbx | z_wm] in z-space: quantile-bin z_wm, weighted-mean z_sbx per bin, PAV-monotonize.
            Stores (px, py) for np.interp; identity-ish if WM uninformative. Also logs R=corr(z_wm,z_sbx) & residual."""
            import numpy as _np
            P = _np.asarray(self._zpairs, dtype=float)
            x = P[:, 0]; y = P[:, 1]
            # quantile bin edges over z_wm (robust to the z range); collapse to unique
            qs = _np.quantile(x, _np.linspace(0, 1, int(bins) + 1))
            qs = _np.unique(qs)
            if qs.size < 3:
                return  # degenerate (z_wm ~ constant)
            idx = _np.clip(_np.digitize(x, qs[1:-1]), 0, qs.size - 2)
            cx = []; cy = []; cw = []
            for b in range(qs.size - 1):
                m = idx == b
                if m.sum() == 0:
                    continue
                cx.append(float(x[m].mean())); cy.append(float(y[m].mean())); cw.append(float(m.sum()))
            if len(cx) < 2:
                return
            # PAV: monotone non-decreasing weighted means
            val = []; wt = []; cnt = []
            for v, w in zip(cy, cw):
                val.append(v); wt.append(w); cnt.append(1)
                while len(val) > 1 and val[-2] > val[-1]:
                    v2 = val.pop(); w2 = wt.pop(); c2 = cnt.pop()
                    v1 = val.pop(); w1 = wt.pop(); c1 = cnt.pop()
                    nw = w1 + w2
                    val.append((v1 * w1 + v2 * w2) / nw); wt.append(nw); cnt.append(c1 + c2)
            py = []
            for v, c in zip(val, cnt):
                py += [v] * c
            self._phi_px = _np.asarray(cx); self._phi_py = _np.asarray(py)
            self._phi_since = 0; self._phi_nfit += 1
            # diagnostics: R = corr(z_wm,z_sbx); residual = E[(φ(z_wm)-z_sbx)^2] (target ≈ 1-R^2)
            yhat = _np.interp(x, self._phi_px, self._phi_py)
            sx = x.std(); sy = y.std()
            self._phi_R = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx > 1e-9 and sy > 1e-9 else 0.0
            self._phi_resid = float(((yhat - y) ** 2).mean())
            print(f"[grade_store] phi fit #{self._phi_nfit}: npairs={len(self._zpairs)} bins={len(cx)} "
                  f"R={self._phi_R:.3f} resid={self._phi_resid:.3f} (1-R^2={1.0 - self._phi_R**2:.3f}) "
                  f"px={_np.round(self._phi_px,2).tolist()} py={_np.round(self._phi_py,2).tolist()}", flush=True)

        def get_phi_f(self):
            """Return (px, py) lists for the consumer to apply φ locally (np.interp), or None if unfitted (≡ identity)."""
            if self._phi_px is None:
                return None
            return (self._phi_px.tolist(), self._phi_py.tolist())

        def phi_stats(self):
            return {"nfit": self._phi_nfit, "npairs": len(self._zpairs), "R": self._phi_R, "resid": self._phi_resid}

    return _GradeStore


def store():
    """Get-or-create the named detached store actor (idempotent across all worker processes)."""
    global _STORE
    if _STORE is None:
        import ray
        _STORE = _make_actor().options(
            name="mledojo_grade_store", namespace="mledojo", lifetime="detached", get_if_exists=True
        ).remote()
    return _STORE


_PUTN = [0]


def put_grade(decoded_response: str, reward: float):
    """Blocking put (so it lands before the reward loop reads). The reward loop runs after the rollout+postprocess,
    but make it blocking to remove the race as a variable while debugging the key match."""
    try:
        import ray
        k = grade_key(decoded_response)
        ray.get(store().put.remote(k, reward))
        if _PUTN[0] < 4:
            _PUTN[0] += 1
            print(f"[grade_store PUT #{_PUTN[0]}] key={k[:12]} reward={reward} resp_len={len(decoded_response)} "
                  f"resp_head={decoded_response[:90]!r} resp_tail={decoded_response[-60:]!r}", flush=True)
    except Exception as e:
        print(f"[grade_store] put failed: {e}", flush=True)


def get_grade(decoded_response: str):
    import ray
    try:
        return ray.get(store().get.remote(grade_key(decoded_response)))
    except Exception as e:
        print(f"[grade_store] get failed: {e}", flush=True)
        return None


# ----------------------------------------------------------------------------------------------------------------------
# SANDBOX GATE — group-level demand-driven routing for the async grader.
#
# WHY: generation is ONE shared vLLM pool (WM/sandbox don't generate); they are two GRADING backends. The slow one is
# the sandbox (16 slots, real ML code). If too many trajectories grade on it at once they queue on env_server, and a
# queued grade blocks its run() coroutine which holds a max_concurrent_samples slot -> all gen slots fill with
# blocked-on-sandbox trajs -> throughput collapses (exactly bug#5). Fix: cap how many trajectories COMMIT to the
# sandbox at once; the rollouter admits a WHOLE GRPO group (NUM_GEN responses) iff the sandbox has room, else the group
# goes to the fast WM. Group-granularity keeps GRPO group-norm on one reward scale; overflow self-routes to WM so the
# sandbox runs flat-out but never oversubscribed. NO generation lock/serialization — gen stays fully concurrent.
#
# Counter actor = sandbox slots currently committed (cap = SBX_GATE_CAP = sandbox total slots, e.g. 16 = 2 groups).
# Rollouter (patch-7) calls sbx_try_acquire(NUM_GEN) once per group; loop calls sbx_release(1) at end of each
# sandbox-routed run(). Fail-safe: any error / a leaked reservation only biases routing toward WM (the safe/fast path),
# never toward a stall.
# ----------------------------------------------------------------------------------------------------------------------
_GATE = None


def _make_gate_actor():
    import ray

    @ray.remote(num_cpus=0)
    class _SbxGate:
        def __init__(self, capacity):
            self._cap = int(capacity)
            self._inflight = 0

        def try_acquire(self, n):
            n = int(n)
            if self._inflight + n <= self._cap:
                self._inflight += n
                return True
            return False

        def release(self, k):
            self._inflight = max(0, self._inflight - int(k))
            return self._inflight

        def state(self):
            return {"inflight": self._inflight, "cap": self._cap}

    return _SbxGate


def sbx_gate():
    """Get-or-create the named detached sandbox gate (idempotent). Capacity from SBX_GATE_CAP (sandbox total slots)."""
    global _GATE
    if _GATE is None:
        cap = int(os.environ.get("SBX_GATE_CAP", "16"))
        _GATE = _make_gate_actor().options(
            name="mledojo_sbx_gate", namespace="mledojo", lifetime="detached", get_if_exists=True
        ).remote(cap)
    return _GATE


_GATEN = [0]


def sbx_try_acquire(n) -> bool:
    """Reserve n sandbox slots for a WHOLE group atomically (rollouter, once per group). True -> group grades on the
    real sandbox; False -> group goes to WM. Fail-safe: any error -> False (route to safe/fast WM, never block gen)."""
    import ray
    try:
        ok = bool(ray.get(sbx_gate().try_acquire.remote(int(n))))
        if _GATEN[0] < 8:
            _GATEN[0] += 1
            print(f"[sbx_gate] try_acquire({n}) -> {'sandbox' if ok else 'wm'}", flush=True)
        return ok
    except Exception as e:
        print(f"[sbx_gate] try_acquire failed (-> wm): {e}", flush=True)
        return False


def sbx_release(k):
    """Release k slots when sandbox-routed trajectories finish (loop, once per trajectory at end of run())."""
    import ray
    try:
        ray.get(sbx_gate().release.remote(int(k)))
    except Exception as e:
        print(f"[sbx_gate] release failed: {e}", flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# ADAW pair transport — used by the anchor path only.
# These live in a SEPARATE _pairs dict inside _GradeStore (never overlaps with put_grade/_d).
# Blocking variants (ray.get) so the pair lands before the next trainer step reads it where possible.
# ----------------------------------------------------------------------------------------------------------------------

def put_adaw_pair(decoded_response: str, r_wm: float, r_sbx: float):
    """Best-effort put of (r_wm, r_sbx) pair for ADAW η² estimation. Keyed by the same grade_key."""
    try:
        import ray
        ray.get(store().put_adaw_pair.remote(grade_key(decoded_response), r_wm, r_sbx))
    except Exception as e:
        print(f"[grade_store] put_adaw_pair failed: {e}", flush=True)


def pop_adaw_pairs(keys: list) -> dict:
    """Batched pop of ADAW pairs by key list. Returns {key: (r_wm,r_sbx) or None}."""
    try:
        import ray
        return ray.get(store().pop_adaw_pairs.remote(keys))
    except Exception as e:
        print(f"[grade_store] pop_adaw_pairs failed: {e}", flush=True)
        return {}


def push_eta2(x: float, half_life: float) -> float:
    """Update η² EMA on the store actor. Returns the updated EMA value."""
    try:
        import ray
        return ray.get(store().push_eta2.remote(float(x), float(half_life)))
    except Exception as e:
        print(f"[grade_store] push_eta2 failed: {e}", flush=True)
        return None


def get_eta2() -> float:
    try:
        import ray
        return ray.get(store().get_eta2.remote())
    except Exception as e:
        print(f"[grade_store] get_eta2 failed: {e}", flush=True)
        return None


def eta2_count() -> int:
    try:
        import ray
        return ray.get(store().eta2_count.remote())
    except Exception as e:
        return 0


def calibrate_c(target_w: float, rho: float = 1.0):
    """Calibrate and freeze ADAW gain c (AUTO mode). Returns c, or None if not yet ready."""
    try:
        import ray
        return ray.get(store().calibrate_c.remote(float(target_w), float(rho)))
    except Exception as e:
        print(f"[grade_store] calibrate_c failed: {e}", flush=True)
        return None


# ----------------------------------------------------------------------------------------------------------------------
# WMRL (wmrl) — online monotone recalibration f: r_wm -> r̂ ≈ r_sbx.
# Producer pushes anchor (r_wm, r_sbx) pairs (sandbox groups) -> actor fits a bucketed-isotonic f; producer applies f to
# WM-group rewards (Path B / bias removal). f=identity until RECAL_MIN_PAIRS ⇒ E opens exactly as B. Knobs from env.
# ----------------------------------------------------------------------------------------------------------------------

def push_recal_pair(r_wm: float, r_sbx: float):
    """Push one anchor (r_wm, r_sbx); the actor refits f opportunistically. Knobs (RECAL_MIN_PAIRS/REFIT_EVERY/BINS) from env."""
    try:
        import ray
        ray.get(store().push_recal_pair.remote(
            float(r_wm), float(r_sbx),
            int(os.environ.get("RECAL_MIN_PAIRS", "200")),
            int(os.environ.get("RECAL_REFIT_EVERY", "64")),
            int(os.environ.get("RECAL_BINS", "10"))))
    except Exception as e:
        print(f"[grade_store] push_recal_pair failed: {e}", flush=True)


def recal(r: float) -> float:
    """Apply the recalibration f to one reward; identity until f is fitted (E ≡ B)."""
    try:
        import ray
        return float(ray.get(store().recal_apply.remote(float(r))))
    except Exception as e:
        print(f"[grade_store] recal failed (-> identity): {e}", flush=True)
        return float(r)


def get_recal_f():
    """Fetch (fx, fy) for local np.interp in the consumer (one round-trip per step), or None if unfitted."""
    try:
        import ray
        return ray.get(store().get_recal_f.remote())
    except Exception as e:
        print(f"[grade_store] get_recal_f failed: {e}", flush=True)
        return None


def recal_stats():
    try:
        import ray
        return ray.get(store().recal_stats.remote())
    except Exception:
        return {}


# ----------------------------------------------------------------------------------------------------------------------
# an abandoned variant (an abandoned variant) — φ(z_wm)=E[z_sbx|z_wm] in standardized space, applied to WM-group ADVANTAGES (not rewards),
# so std-norm GRPO cannot undo it. Consumer pushes within-group-standardized anchor pairs; actor fits isotonic φ.
# ----------------------------------------------------------------------------------------------------------------------

def push_phi_pairs(zwm: list, zsb: list):
    """Push within-group-standardized (z_wm, z_sbx) anchor pairs; the actor refits φ opportunistically. Knobs from env."""
    try:
        import ray
        ray.get(store().push_phi_pairs.remote(
            list(zwm), list(zsb),
            int(os.environ.get("PHI_MIN_PAIRS", "400")),
            int(os.environ.get("PHI_REFIT_EVERY", "64")),
            int(os.environ.get("PHI_BINS", "12"))))
    except Exception as e:
        print(f"[grade_store] push_phi_pairs failed: {e}", flush=True)


def get_phi_f():
    """Fetch (px, py) for local np.interp φ in the consumer (one round-trip per step), or None if unfitted."""
    try:
        import ray
        return ray.get(store().get_phi_f.remote())
    except Exception as e:
        print(f"[grade_store] get_phi_f failed: {e}", flush=True)
        return None


def phi_stats():
    try:
        import ray
        return ray.get(store().phi_stats.remote())
    except Exception:
        return {}
