#!/usr/bin/env python3
"""The entrypoint every node runs.

One command, launched identically on all nodes. Each works out its role from the
topology and does the corresponding thing:

``trainer``
    Wait for the other nodes to publish their server URLs, then run the policy
    optimisation loop against them. When it finishes, mark the run done, which
    is how the other nodes learn to exit.
``world_model``
    Start one inference engine per GPU, all serving the same base model as the
    agent, then publish their URLs and serve until the run ends. One engine per
    card rather than one tensor-parallel engine across cards: predictions are
    short, so throughput matters and per-query latency does not.
``sandbox``
    Start the execution server that runs candidate solutions for the anchor
    stream, publish its URL, and serve until the run ends.

Usage::

    WMRL_HOSTS=node0,node1,node2 \\
    WMRL_STORE=/mnt/shared/wmrl \\
    WMRL_RUN_ID=my-run \\
      python -m ml_research.cluster.launch --config configs/wmrl_9b.yaml

Everything the run needs beyond that lives in the config file. Nothing here
reads a scheduler-specific variable, so the same command works under Slurm, a
manual ssh loop, or any managed service.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

from ml_research.cluster import rendezvous
from ml_research.cluster.store import Store, open_store
from ml_research.cluster.topology import Topology, discover

REPO = pathlib.Path(__file__).resolve().parents[2]

__all__ = ["load_config", "run_trainer", "run_world_model", "run_sandbox", "main"]


# ------------------------------------------------------------------- config


def load_config(path: str) -> dict:
    """Load a YAML (or JSON) run config, with environment overrides.

    Any ``WMRL_CFG_<KEY>`` variable overrides the matching key, so a sweep can
    vary one setting without writing a new file for each point.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise SystemExit(f"config not found: {path}")

    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise SystemExit("pyyaml is needed to read a YAML config: pip install pyyaml")
        cfg = yaml.safe_load(text) or {}
    else:
        cfg = json.loads(text)

    for k, v in os.environ.items():
        if k.startswith("WMRL_CFG_"):
            cfg[k[len("WMRL_CFG_"):].lower()] = v

    return cfg


def _gpus(cfg: dict, key: str, default: str) -> list[int]:
    raw = str(cfg.get(key, default))
    return [int(g) for g in raw.split(",") if g.strip() != ""]


def _child_env(cfg: dict, **extra) -> dict:
    """Environment for a child process: ours, plus the config, plus overrides.

    Config keys are upper-cased, because the servers read their settings from
    the environment and that is the convention they already use.
    """
    env = dict(os.environ)
    env.update({str(k).upper(): str(v) for k, v in cfg.items()})
    env.update({str(k): str(v) for k, v in extra.items()})
    return env


# ------------------------------------------------------- process supervision


class Supervisor:
    """Owns the child server processes and makes their death loud.

    A server that exits during a long run would otherwise show up as the trainer
    timing out on requests much later, far from the cause. This checks liveness
    and surfaces the log path of whichever child died.
    """

    def __init__(self):
        self._children: list[tuple[str, subprocess.Popen, str]] = []

    def add(self, name: str, proc: subprocess.Popen, logfile: str) -> None:
        self._children.append((name, proc, logfile))

    def check(self) -> None:
        for name, proc, logfile in self._children:
            rc = proc.poll()
            if rc is not None:
                tail = ""
                try:
                    tail = "\n".join(pathlib.Path(logfile).read_text().splitlines()[-25:])
                except OSError:
                    pass
                raise SystemExit(
                    f"[launch] {name} exited with status {rc}\n"
                    f"  log: {logfile}\n"
                    f"{tail}"
                )

    def wait_until(self, predicate, poll: float = 10.0) -> None:
        """Poll ``predicate`` while keeping an eye on the children."""
        while not predicate():
            self.check()
            time.sleep(poll)

    def terminate(self) -> None:
        for name, proc, _ in self._children:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 30
        for _, proc, _ in self._children:
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.2)
        for _, proc, _ in self._children:
            if proc.poll() is None:
                proc.kill()


def _logdir(cfg: dict, topo: Topology) -> pathlib.Path:
    d = pathlib.Path(cfg.get("log_dir", "logs")) / topo.run_id / topo.host
    d.mkdir(parents=True, exist_ok=True)
    return d


def _wait_for_http(urls: list[str], timeout: float = 2400, poll: float = 5.0) -> None:
    """Block until every URL answers, so we register only what is actually up."""
    import urllib.error
    import urllib.request

    pending = list(urls)
    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        still = []
        for u in pending:
            try:
                with urllib.request.urlopen(f"{u}/health", timeout=5):
                    pass
            except (urllib.error.URLError, OSError):
                still.append(u)
        pending = still
        if pending:
            time.sleep(poll)
    if pending:
        raise TimeoutError(f"server(s) never became healthy after {timeout:.0f}s: {pending}")


# ------------------------------------------------------------------- roles


def run_world_model(cfg: dict, topo: Topology, store: Store) -> None:
    """One inference engine per GPU, all serving the agent's own base model.

    Sharing the agent's backbone is deliberate: it rules out the possibility
    that gains come from distilling a stronger model, because there is no
    stronger model anywhere in the loop.
    """
    gpus = _gpus(cfg, "world_model_gpus", "0,1,2,3,4,5,6,7")
    base_port = int(cfg.get("world_model_port", 8000))
    logs = _logdir(cfg, topo)
    sup = Supervisor()

    urls = []
    for i, gpu in enumerate(gpus):
        port = base_port + i
        logfile = str(logs / f"world_model_gpu{gpu}.log")
        env = _child_env(
            cfg,
            CUDA_VISIBLE_DEVICES=str(gpu),
            PORT=str(port),
            WM_MODEL=cfg["model_path"],
            WM_TP="1",  # one engine per card: throughput over per-query latency
            WM_GPU_UTIL=cfg.get("world_model_gpu_util", "0.90"),
            WM_MAX_MODEL_LEN=cfg.get("world_model_max_model_len", "12288"),
            WM_CAPACITY=cfg.get("world_model_capacity", "256"),
            WM_MAX_TOKENS=cfg.get("world_model_max_tokens", "1024"),
        )
        proc = subprocess.Popen(
            [sys.executable, str(REPO / "ml_research" / "world_model" / "server.py")],
            env=env,
            stdout=open(logfile, "w"),
            stderr=subprocess.STDOUT,
        )
        sup.add(f"world_model:gpu{gpu}", proc, logfile)
        urls.append(f"http://{rendezvous.own_address()}:{port}")

    print(f"[launch] {len(urls)} world model engine(s) starting, logs in {logs}", flush=True)
    try:
        _wait_for_http(urls, timeout=float(cfg.get("server_boot_timeout_s", 1800)))
        rendezvous.register(store, topo, urls, meta={"engines": len(urls)})
        sup.wait_until(lambda: store.get(f"runs/{topo.run_id}/DONE") is not None)
    finally:
        sup.terminate()


def run_sandbox(cfg: dict, topo: Topology, store: Store) -> None:
    """The execution server: the only thing in the loop that costs real machine time.

    Its capacity is what bounds the anchor stream, and therefore what bounds how
    fast the calibration can track the world model's drift.
    """
    gpus = _gpus(cfg, "sandbox_gpus", "0,1,2,3,4,5,6,7")
    port = int(cfg.get("sandbox_port", 8100))
    logs = _logdir(cfg, topo)
    logfile = str(logs / "sandbox.log")

    env = _child_env(
        cfg,
        ENV_GPUS=",".join(str(g) for g in gpus),
        PORT=str(port),
        SLOTS_PER_GPU=cfg.get("sandbox_slots_per_gpu", "3"),
        SBX_GPU_MEM_GB=cfg.get("sandbox_gpu_mem_gb", "13"),
        EXEC_CAP=cfg.get("sandbox_exec_cap_s", "600"),
        SANDBOX_CORES=cfg.get("sandbox_cores", "6"),
        SANDBOX_MEM_GB=cfg.get("sandbox_mem_gb", "64"),
    )
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "ml_research" / "grading" / "sandbox_server.py")],
        env=env,
        stdout=open(logfile, "w"),
        stderr=subprocess.STDOUT,
    )
    sup = Supervisor()
    sup.add("sandbox", proc, logfile)

    url = f"http://{rendezvous.own_address()}:{port}"
    slots = len(gpus) * int(cfg.get("sandbox_slots_per_gpu", 3))
    print(f"[launch] sandbox starting with {slots} slot(s), log {logfile}", flush=True)
    try:
        _wait_for_http([url], timeout=float(cfg.get("server_boot_timeout_s", 1800)))
        rendezvous.register(store, topo, [url], meta={"slots": slots})
        sup.wait_until(lambda: store.get(f"runs/{topo.run_id}/DONE") is not None)
    finally:
        sup.terminate()


def run_trainer(cfg: dict, topo: Topology, store: Store) -> None:
    """Policy optimisation, against whatever servers registered."""
    regs = rendezvous.await_registrations(
        store, topo, timeout=float(cfg.get("rendezvous_timeout_s", 2400))
    )

    wm_urls, sbx_urls = [], []
    for reg in regs.values():
        (wm_urls if reg.role == "world_model" else sbx_urls).extend(reg.urls)

    if not wm_urls:
        raise SystemExit("no world model server registered: there is nothing cheap to grade with")
    if not sbx_urls:
        raise SystemExit(
            "no sandbox server registered: without an anchor stream neither correction can run. "
            "Give one node the sandbox role, or train on real execution instead."
        )

    print(f"[launch] {len(wm_urls)} world model engine(s), {len(sbx_urls)} sandbox server(s)", flush=True)

    logs = _logdir(cfg, topo)
    logfile = str(logs / "trainer.log")
    env = _child_env(
        cfg,
        WMRL_WORLD_MODEL_URLS=",".join(wm_urls),
        WMRL_SANDBOX_URLS=",".join(sbx_urls),
        WMRL_RUN_ID=topo.run_id,
        WMRL_STORE=str(store),
    )

    cmd = [sys.executable, str(REPO / "ml_research" / "cluster" / "train_entry.py"),
           "--config", str(cfg.get("_config_path", "")), "--log", logfile]

    status = "ok"
    try:
        print(f"[launch] trainer starting, log {logfile}", flush=True)
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            status = f"failed:{rc}"
            raise SystemExit(f"[launch] trainer exited with status {rc}; see {logfile}")
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        # Always release the other nodes. Without this a failed trainer leaves
        # two nodes serving an empty run until something else reaps them.
        rendezvous.mark_done(store, topo, status)


# -------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="run config, YAML or JSON")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the topology and print the plan, start nothing")
    args = ap.parse_args(argv)

    topo = discover()
    print(topo.describe(), flush=True)

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config

    if args.dry_run:
        print(f"\nwould run role {topo.role!r} with config {args.config}")
        print(f"store: {os.environ.get('WMRL_STORE', '(unset)')}")
        return 0

    store = open_store(multi_node=len(topo.hosts) > 1)

    def _bye(signum, _frame):
        print(f"\n[launch] signal {signum}, shutting down", flush=True)
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    role = topo.role
    if role == "trainer":
        run_trainer(cfg, topo, store)
    elif role == "world_model":
        run_world_model(cfg, topo, store)
    elif role == "sandbox":
        run_sandbox(cfg, topo, store)
    else:  # unreachable: Topology validates roles
        raise SystemExit(f"no launcher for role {role!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
