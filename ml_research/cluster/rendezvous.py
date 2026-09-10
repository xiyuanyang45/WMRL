"""How the trainer finds the world model and sandbox servers.

The trainer cannot be configured with the server addresses up front, because it
does not know which port each server will land on until that server has bound
one. So the servers publish and the trainer subscribes, through the shared store:

1. A world model or sandbox node starts its servers, then writes
   ``runs/<run_id>/registry/<host>.json`` describing what it is serving.
2. The trainer polls that prefix until every non-trainer host has registered,
   then reads the URLs out.
3. When training finishes the trainer writes ``runs/<run_id>/DONE``, which is
   how the other nodes learn they can exit.

Two failure modes are worth the code they cost:

**Stale registrations.** Runs share a storage prefix. A registration left by a
yesterday's run points at a server that is gone, and a trainer that believes it
will hang on a dead address. Every record carries its ``run_id`` and anything
that does not match is ignored.

**Silent absence.** A node that crashes during startup never registers. Waiting
forever turns that into a hang with no message, so the wait has a deadline and
the timeout names exactly which hosts failed to appear.
"""

from __future__ import annotations

import json
import socket
import time

from ml_research.cluster.store import Store

__all__ = ["Registration", "register", "await_registrations", "mark_done", "wait_until_done"]


def _registry_key(run_id: str, host: str) -> str:
    return f"runs/{run_id}/registry/{host}.json"


def _done_key(run_id: str) -> str:
    return f"runs/{run_id}/DONE"


def own_address() -> str:
    """The address other nodes should dial. ``WMRL_ADVERTISE_ADDR`` overrides.

    The override exists because the name a node knows itself by is not always
    the name its peers can route to: containers, multiple interfaces, and split
    networks all break the naive answer.
    """
    import os

    override = os.environ.get("WMRL_ADVERTISE_ADDR")
    if override:
        return override
    host = socket.gethostname()
    try:
        return socket.gethostbyname(host)
    except OSError:
        return host


class Registration:
    """What one node published about the servers it is running."""

    def __init__(self, host: str, role: str, run_id: str, urls: list[str], meta: dict | None = None):
        self.host = host
        self.role = role
        self.run_id = run_id
        self.urls = list(urls)
        self.meta = dict(meta or {})

    def to_json(self) -> str:
        return json.dumps(
            {
                "host": self.host,
                "role": self.role,
                "run_id": self.run_id,
                "urls": self.urls,
                "meta": self.meta,
                "written_at": time.time(),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, blob: str) -> "Registration":
        d = json.loads(blob)
        return cls(d["host"], d["role"], d["run_id"], d.get("urls", []), d.get("meta"))

    def __repr__(self):
        return f"Registration({self.host!r}, {self.role!r}, {len(self.urls)} url(s))"


def register(store: Store, topo, urls: list[str], meta: dict | None = None) -> Registration:
    """Publish this node's server URLs. Called once the servers accept connections."""
    reg = Registration(topo.host, topo.role, topo.run_id, urls, meta)
    store.put(_registry_key(topo.run_id, topo.host), reg.to_json())
    print(f"[rendezvous] registered {topo.host} ({topo.role}): {urls}", flush=True)
    return reg


def await_registrations(store: Store, topo, timeout: float = 2400, poll: float = 5.0) -> dict:
    """Block until every non-trainer host has registered for *this* run.

    Returns a mapping of host to :class:`Registration`.
    """
    expected = {h for h, r in zip(topo.hosts, topo.roles) if r != "trainer"}
    if not expected:
        return {}

    prefix = f"runs/{topo.run_id}/registry/"
    found: dict[str, Registration] = {}
    deadline = time.time() + timeout
    stale = 0

    print(f"[rendezvous] waiting for {len(expected)} node(s): {sorted(expected)}", flush=True)
    while time.time() < deadline:
        for key in store.list(prefix):
            host = key.rsplit("/", 1)[-1][: -len(".json")]
            if host in found or host not in expected:
                continue
            blob = store.get(key)
            if blob is None:
                continue
            try:
                reg = Registration.from_json(blob)
            except (ValueError, KeyError):
                continue
            if reg.run_id != topo.run_id:
                stale += 1
                continue  # a previous run's registration: the server it names is gone
            found[host] = reg
            print(f"[rendezvous] {host} up ({reg.role}, {len(reg.urls)} url(s))", flush=True)

        if set(found) == expected:
            if stale:
                print(f"[rendezvous] ignored {stale} registration(s) from other runs", flush=True)
            return found
        time.sleep(poll)

    missing = sorted(expected - set(found))
    raise TimeoutError(
        f"{len(missing)} node(s) never registered after {timeout:.0f}s: {missing}\n"
        f"  store: {store}\n"
        "  Check those nodes started, share this WMRL_STORE, and use this WMRL_RUN_ID."
    )


def mark_done(store: Store, topo, status: str = "ok") -> None:
    store.put(_done_key(topo.run_id), json.dumps({"status": status, "at": time.time()}))
    print(f"[rendezvous] run {topo.run_id} marked done ({status})", flush=True)


def wait_until_done(store: Store, topo, poll: float = 30.0) -> None:
    """Serve until the trainer says the run is over.

    This is the whole main loop of a world model or sandbox node: the servers
    run in background processes, and this keeps the entrypoint alive so the
    scheduler does not reap the node while the trainer is still using it.
    """
    key = _done_key(topo.run_id)
    while store.get(key) is None:
        time.sleep(poll)
    print(f"[rendezvous] run {topo.run_id} finished; {topo.role} node exiting", flush=True)
