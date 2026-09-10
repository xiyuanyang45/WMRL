"""Launcher behaviour that can be checked without a cluster.

The parts that need GPUs are not exercised here. What is: config handling, the
guard rails that stop a misconfigured run early, and the supervisor that turns a
dead child process into a message instead of a hang.
"""

import json
import subprocess
import sys
import textwrap

import pytest

from ml_research.cluster import rendezvous
from ml_research.cluster.launch import Supervisor, _gpus, load_config, run_trainer
from ml_research.cluster.store import LocalStore
from ml_research.cluster.topology import Topology


def topo(rank, run_id="run-a"):
    return Topology(
        hosts=["node0", "node1", "node2"],
        rank=rank,
        roles=["trainer", "world_model", "sandbox"],
        run_id=run_id,
    )


# ------------------------------------------------------------------- config


def test_json_and_yaml_configs_agree(tmp_path):
    j = tmp_path / "c.json"
    j.write_text(json.dumps({"model_path": "/m", "steps": 10}))
    assert load_config(str(j))["steps"] == 10

    y = tmp_path / "c.yaml"
    y.write_text(textwrap.dedent("""
        model_path: /m
        steps: 10
    """))
    pytest.importorskip("yaml")
    assert load_config(str(y)) == load_config(str(j))


def test_environment_overrides_a_config_key(tmp_path, monkeypatch):
    c = tmp_path / "c.json"
    c.write_text(json.dumps({"model_path": "/m", "steps": 10}))
    monkeypatch.setenv("WMRL_CFG_STEPS", "999")
    assert load_config(str(c))["steps"] == "999"


def test_missing_config_fails_immediately(tmp_path):
    with pytest.raises(SystemExit, match="config not found"):
        load_config(str(tmp_path / "absent.yaml"))


def test_gpu_lists_are_parsed():
    assert _gpus({"sandbox_gpus": "0,1,2"}, "sandbox_gpus", "") == [0, 1, 2]
    assert _gpus({}, "sandbox_gpus", "0,1") == [0, 1]
    assert _gpus({"sandbox_gpus": "3"}, "sandbox_gpus", "") == [3]
    assert _gpus({"sandbox_gpus": ""}, "sandbox_gpus", "") == []


# -------------------------------------------------------------- guard rails


def test_trainer_refuses_to_start_without_a_world_model(tmp_path):
    store = LocalStore(tmp_path)
    rendezvous.register(store, topo(2), ["http://sbx:1"])
    # pretend the world model host registered as a second sandbox
    t = Topology(hosts=["node0", "node2"], rank=0, roles=["trainer", "sandbox"], run_id="run-a")
    rendezvous.register(store, Topology(hosts=["node0", "node2"], rank=1,
                                        roles=["trainer", "sandbox"], run_id="run-a"),
                        ["http://sbx:1"])
    with pytest.raises(SystemExit, match="nothing cheap to grade with"):
        run_trainer({"rendezvous_timeout_s": 2}, t, store)


def test_trainer_refuses_to_start_without_an_anchor_stream(tmp_path):
    """No sandbox means no ground truth, so neither correction can run.

    Failing here is the point: the run would otherwise look healthy and quietly
    train on uncorrected world model rewards.
    """
    store = LocalStore(tmp_path)
    t = Topology(hosts=["node0", "node1"], rank=0, roles=["trainer", "world_model"], run_id="run-a")
    rendezvous.register(store, Topology(hosts=["node0", "node1"], rank=1,
                                        roles=["trainer", "world_model"], run_id="run-a"),
                        ["http://wm:1"])
    with pytest.raises(SystemExit, match="anchor stream"):
        run_trainer({"rendezvous_timeout_s": 2}, t, store)


def test_a_failed_trainer_still_releases_the_other_nodes(tmp_path, monkeypatch):
    """Otherwise two nodes keep serving a run that has already died."""
    store = LocalStore(tmp_path)
    rendezvous.register(store, topo(1), ["http://wm:1"])
    rendezvous.register(store, topo(2), ["http://sbx:1"])

    monkeypatch.setattr("ml_research.cluster.launch.subprocess.call", lambda *a, **k: 7)
    monkeypatch.setattr("ml_research.cluster.launch._logdir", lambda cfg, t: tmp_path)

    with pytest.raises(SystemExit, match="status 7"):
        run_trainer({"rendezvous_timeout_s": 2, "_config_path": "c.yaml"}, topo(0), store)

    done = store.get("runs/run-a/DONE")
    assert done is not None, "the other nodes would wait forever"
    assert json.loads(done)["status"] == "failed:7"


# ------------------------------------------------------------- supervision


def test_supervisor_reports_a_dead_child_with_its_log(tmp_path):
    logfile = tmp_path / "child.log"
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; print('boom: model failed to load'); sys.exit(3)"],
        stdout=open(logfile, "w"),
        stderr=subprocess.STDOUT,
    )
    proc.wait()

    sup = Supervisor()
    sup.add("world_model:gpu0", proc, str(logfile))
    with pytest.raises(SystemExit) as e:
        sup.check()

    msg = str(e.value)
    assert "world_model:gpu0" in msg
    assert "status 3" in msg
    assert "boom: model failed to load" in msg, "the tail of the log should be in the message"


def test_supervisor_is_quiet_while_children_live(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        sup = Supervisor()
        sup.add("sandbox", proc, str(tmp_path / "x.log"))
        sup.check()  # must not raise
    finally:
        proc.kill()


def test_wait_until_stops_when_a_child_dies(tmp_path):
    """A hung wait is the failure this replaces."""
    logfile = tmp_path / "c.log"
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(1)"],
                            stdout=open(logfile, "w"), stderr=subprocess.STDOUT)
    proc.wait()
    sup = Supervisor()
    sup.add("sandbox", proc, str(logfile))
    with pytest.raises(SystemExit):
        sup.wait_until(lambda: False, poll=0.01)
