"""The rendezvous protocol, exercised with three simulated nodes on one machine.

A shared directory stands in for shared storage, so the handshake the trainer
and the two server nodes perform can be tested without a cluster.
"""

import json
import threading

import pytest

from ml_research.cluster.rendezvous import (
    Registration,
    await_registrations,
    mark_done,
    register,
    wait_until_done,
)
from ml_research.cluster.store import LocalStore
from ml_research.cluster.topology import Topology, discover


def topo(rank, hosts=("node0", "node1", "node2"), roles=None, run_id="run-a"):
    return Topology(
        hosts=list(hosts),
        rank=rank,
        roles=list(roles or ["trainer", "world_model", "sandbox"]),
        run_id=run_id,
    )


# ------------------------------------------------------------------ topology


def test_roles_are_assigned_by_position():
    t = topo(0)
    assert t.role == "trainer" and t.is_trainer
    assert topo(1).role == "world_model"
    assert topo(2).role == "sandbox"
    assert t.world_model_hosts == ["node1"]
    assert t.sandbox_hosts == ["node2"]
    assert t.trainer_host == "node0"


def test_extra_nodes_widen_the_anchor_stream(monkeypatch):
    monkeypatch.setenv("WMRL_HOSTS", "a,b,c,d,e")
    monkeypatch.setenv("WMRL_NODE_RANK", "4")
    monkeypatch.setenv("WMRL_RUN_ID", "r")
    t = discover()
    assert t.roles == ["trainer", "world_model", "sandbox", "sandbox", "sandbox"]
    assert t.sandbox_hosts == ["c", "d", "e"]


def test_a_run_with_no_trainer_is_rejected():
    with pytest.raises(ValueError, match="trainer"):
        topo(0, roles=["world_model", "sandbox", "sandbox"])


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="unknown role"):
        topo(0, roles=["trainer", "grader", "sandbox"])


def test_role_count_must_match_host_count():
    with pytest.raises(ValueError, match="correspond"):
        topo(0, roles=["trainer", "world_model"])


def test_describe_flags_a_run_with_no_anchor_stream():
    t = topo(0, hosts=("a", "b"), roles=["trainer", "world_model"])
    assert "no anchor stream" in t.describe()


def test_explicit_rank_and_roles_are_honoured(monkeypatch):
    monkeypatch.setenv("WMRL_HOSTS", "x,y,z")
    monkeypatch.setenv("WMRL_NODE_RANK", "2")
    monkeypatch.setenv("WMRL_ROLES", "trainer,sandbox,world_model")
    monkeypatch.setenv("WMRL_RUN_ID", "r")
    t = discover()
    assert (t.rank, t.host, t.role) == (2, "z", "world_model")


def test_hostfile_is_parsed(monkeypatch, tmp_path):
    hf = tmp_path / "hosts"
    hf.write_text("# comment\nnode0 slots=8\n\nnode1\nnode2  \n")
    monkeypatch.delenv("WMRL_HOSTS", raising=False)
    monkeypatch.setenv("WMRL_HOSTFILE", str(hf))
    monkeypatch.setenv("WMRL_NODE_RANK", "1")
    monkeypatch.setenv("WMRL_RUN_ID", "r")
    assert discover().hosts == ["node0", "node1", "node2"]


# --------------------------------------------------------------------- store


def test_store_round_trip_and_missing_key(tmp_path):
    s = LocalStore(tmp_path)
    assert s.get("nope") is None
    s.put("a/b.json", '{"x": 1}')
    assert json.loads(s.get("a/b.json")) == {"x": 1}
    assert s.list("a/") == ["a/b.json"]
    assert s.mtime("a/b.json") is not None


def test_store_rejects_keys_that_escape_the_root(tmp_path):
    s = LocalStore(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        s.put("../outside", "x")


def test_wait_for_times_out_with_context(tmp_path):
    s = LocalStore(tmp_path)
    with pytest.raises(TimeoutError, match="never appeared"):
        s.wait_for("absent", timeout=0.2, poll=0.05)


# --------------------------------------------------------------- rendezvous


def test_trainer_waits_for_both_server_nodes(tmp_path):
    store = LocalStore(tmp_path)
    register(store, topo(1), ["http://192.0.2.1:8000"])
    register(store, topo(2), ["http://192.0.2.2:9000"])

    found = await_registrations(store, topo(0), timeout=2, poll=0.05)
    assert set(found) == {"node1", "node2"}
    assert found["node1"].role == "world_model"
    assert found["node2"].urls == ["http://192.0.2.2:9000"]


def test_registrations_from_another_run_are_ignored(tmp_path):
    """The failure this guards: yesterday's registration names a dead server."""
    store = LocalStore(tmp_path)
    stale = Registration("node1", "world_model", "run-OLD", ["http://dead:8000"])
    store.put("runs/run-a/registry/node1.json", stale.to_json())

    with pytest.raises(TimeoutError, match="node1"):
        await_registrations(store, topo(0), timeout=0.4, poll=0.05)

    register(store, topo(1), ["http://live:8000"])
    register(store, topo(2), ["http://live:9000"])
    found = await_registrations(store, topo(0), timeout=2, poll=0.05)
    assert found["node1"].urls == ["http://live:8000"]


def test_timeout_names_the_hosts_that_never_appeared(tmp_path):
    store = LocalStore(tmp_path)
    register(store, topo(1), ["http://a:1"])
    with pytest.raises(TimeoutError) as e:
        await_registrations(store, topo(0), timeout=0.4, poll=0.05)
    assert "node2" in str(e.value)
    assert "node1" not in str(e.value).split(":")[-1]


def test_a_run_with_only_a_trainer_needs_no_rendezvous(tmp_path):
    store = LocalStore(tmp_path)
    t = Topology(hosts=["solo"], rank=0, roles=["trainer"], run_id="r")
    assert await_registrations(store, t, timeout=0.2) == {}


def test_server_nodes_exit_when_the_trainer_marks_done(tmp_path):
    store = LocalStore(tmp_path)
    t_wm, t_tr = topo(1), topo(0)

    exited = threading.Event()

    def serve():
        wait_until_done(store, t_wm, poll=0.02)
        exited.set()

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    assert not exited.wait(0.2), "a server node must keep serving until told to stop"

    mark_done(store, t_tr)
    assert exited.wait(2.0), "a server node must exit once the run is marked done"
