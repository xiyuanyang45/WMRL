"""Who am I, and who else is in this run.

Every node runs the same entrypoint. This module is how a node works out which
of the three roles it plays, without depending on any particular scheduler.

The training job is three decoupled roles:

===============  ==========================================================
``trainer``      policy optimisation and rollout generation on the same node
``world_model``  serves predicted rewards, one forward pass per query
``sandbox``      executes candidate solutions for the anchor stream
===============  ==========================================================

The split is what makes the world model worth having. Rollout and world model
inference share an inference backend and batch together; sandbox execution
cannot batch at all, so it is fenced onto its own node where it can be scaled,
starved, or removed without touching the other two.

Roles are assigned by position, so the ordering of the host list is the
configuration. Three nodes is the shape the paper reports, but nothing here
requires exactly three: give two nodes the ``sandbox`` role and the anchor
stream simply gets more capacity.

Configuration, in the order it is consulted:

``WMRL_HOSTFILE``
    Path to a file with one hostname per line, blank lines and ``#`` comments
    ignored. This is what most schedulers already hand you.
``WMRL_HOSTS``
    Comma-separated hostnames, for when a file is inconvenient.
``WMRL_NODE_RANK``
    This node's index into that list. Usually unnecessary: a node that can find
    its own hostname or address in the list infers its rank.
``WMRL_ROLES``
    Comma-separated roles, one per host. Defaults to trainer, world_model,
    sandbox, then sandbox for any further nodes.
``WMRL_RUN_ID``
    Identifies this run. Two runs sharing a storage prefix must not share a run
    id, or the second will read the first's stale rendezvous files and connect
    to servers that are no longer there.
"""

from __future__ import annotations

import os
import pathlib
import socket
import uuid

__all__ = ["Topology", "ROLES", "discover"]

ROLES = ("trainer", "world_model", "sandbox")
DEFAULT_ROLE_ORDER = ("trainer", "world_model", "sandbox")


class Topology:
    """The resolved view of this run from one node's point of view."""

    def __init__(self, hosts: list[str], rank: int, roles: list[str], run_id: str):
        if not hosts:
            raise ValueError("no hosts: set WMRL_HOSTFILE or WMRL_HOSTS")
        if not 0 <= rank < len(hosts):
            raise ValueError(f"rank {rank} outside the host list of length {len(hosts)}")
        if len(roles) != len(hosts):
            raise ValueError(f"{len(roles)} roles for {len(hosts)} hosts; they must correspond")
        bad = sorted(set(roles) - set(ROLES))
        if bad:
            raise ValueError(f"unknown role(s) {bad}; valid roles are {list(ROLES)}")
        if "trainer" not in roles:
            raise ValueError("no host has the trainer role; nothing would optimise the policy")

        self.hosts = hosts
        self.rank = rank
        self.roles = roles
        self.run_id = run_id

    # -- this node ---------------------------------------------------------

    @property
    def host(self) -> str:
        return self.hosts[self.rank]

    @property
    def role(self) -> str:
        return self.roles[self.rank]

    @property
    def is_trainer(self) -> bool:
        return self.role == "trainer"

    # -- the run -----------------------------------------------------------

    @property
    def trainer_host(self) -> str:
        return self.hosts[self.roles.index("trainer")]

    def hosts_with(self, role: str) -> list[str]:
        return [h for h, r in zip(self.hosts, self.roles) if r == role]

    @property
    def world_model_hosts(self) -> list[str]:
        return self.hosts_with("world_model")

    @property
    def sandbox_hosts(self) -> list[str]:
        return self.hosts_with("sandbox")

    def describe(self) -> str:
        lines = [f"run {self.run_id}: {len(self.hosts)} node(s)"]
        for i, (h, r) in enumerate(zip(self.hosts, self.roles)):
            lines.append(f"  [{i}] {h:<24} {r}{'   <- this node' if i == self.rank else ''}")
        if not self.sandbox_hosts:
            lines.append("  note: no sandbox node, so there is no anchor stream and no correction")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Topology(rank={self.rank}, role={self.role!r}, hosts={len(self.hosts)})"


# ------------------------------------------------------------------ helpers


def _read_hosts() -> list[str]:
    hostfile = os.environ.get("WMRL_HOSTFILE")
    if hostfile:
        p = pathlib.Path(hostfile)
        if not p.exists():
            raise SystemExit(f"WMRL_HOSTFILE={hostfile} does not exist")
        out = []
        for line in p.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line.split()[0])  # tolerate "host slots=8" style lines
        return out

    hosts = os.environ.get("WMRL_HOSTS", "")
    return [h.strip() for h in hosts.split(",") if h.strip()]


def _own_names() -> set[str]:
    """Every name this machine might appear under in the host list."""
    names = set()
    host = socket.gethostname()
    names.add(host)
    names.add(host.split(".")[0])
    for getter in (lambda: socket.gethostbyname(host), lambda: socket.getfqdn()):
        try:
            names.add(getter())
        except OSError:
            pass
    return {n for n in names if n}


def _infer_rank(hosts: list[str]) -> int:
    mine = _own_names()
    for i, h in enumerate(hosts):
        if h in mine or h.split(".")[0] in mine:
            return i
    raise SystemExit(
        "cannot tell which host this node is.\n"
        f"  this node answers to: {sorted(mine)}\n"
        f"  the host list is:     {hosts}\n"
        "Set WMRL_NODE_RANK explicitly, or list hosts under names the nodes recognise."
    )


def _default_roles(n: int) -> list[str]:
    roles = list(DEFAULT_ROLE_ORDER[:n])
    roles += ["sandbox"] * (n - len(roles))  # extra nodes widen the anchor stream
    return roles


def discover() -> Topology:
    """Resolve the topology from the environment. Raises with a usable message."""
    hosts = _read_hosts()
    if not hosts:
        raise SystemExit(
            "no host list. Set one of:\n"
            "  WMRL_HOSTFILE=/path/to/hostfile   one hostname per line\n"
            "  WMRL_HOSTS=node0,node1,node2      comma separated, in role order"
        )

    rank_env = os.environ.get("WMRL_NODE_RANK")
    rank = int(rank_env) if rank_env not in (None, "") else _infer_rank(hosts)

    roles_env = os.environ.get("WMRL_ROLES", "")
    roles = [r.strip() for r in roles_env.split(",") if r.strip()] or _default_roles(len(hosts))

    run_id = os.environ.get("WMRL_RUN_ID") or f"wmrl-{uuid.uuid4().hex[:10]}"

    return Topology(hosts=hosts, rank=rank, roles=roles, run_id=run_id)


if __name__ == "__main__":
    print(discover().describe())
