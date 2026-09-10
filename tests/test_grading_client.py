"""The grading client, against a real HTTP server on localhost."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ml_research.grading.client import GradeError, GraderPool


class _Handler(BaseHTTPRequestHandler):
    behaviour = "ok"

    def log_message(self, *a):
        pass

    def do_GET(self):
        code = 200 if self.behaviour != "down" else 500
        self.send_response(code)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        trajs = req.get("trajectories", [])

        if self.behaviour == "short":
            scores = [0.5] * (len(trajs) - 1)
        elif self.behaviour == "ungraded":
            scores = [0.5 if i else None for i in range(len(trajs))]
        else:
            scores = [float(len(str(t))) / 10 for t in trajs]

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"scores": scores}).encode())


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_grades_a_group(server):
    _Handler.behaviour = "ok"
    pool = GraderPool([server], name="world_model")
    scores = pool.grade(["a", "bb", "ccc"])
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)


def test_ungraded_trajectory_stays_none(server):
    """None must survive the client: the advantage code relies on it."""
    _Handler.behaviour = "ungraded"
    pool = GraderPool([server], name="sandbox")
    scores = pool.grade(["a", "b", "c"])
    assert scores[0] is None
    assert all(isinstance(s, float) for s in scores[1:])


def test_a_short_reply_is_an_error_not_a_silent_misalignment(server):
    """Scores are matched to trajectories by position, so a short list would
    silently attach every score to the wrong trajectory."""
    _Handler.behaviour = "short"
    pool = GraderPool([server], name="sandbox", retries=0)
    with pytest.raises(GradeError, match="score"):
        pool.grade(["a", "b", "c"])


def test_in_flight_is_visible_to_the_scheduler(server):
    _Handler.behaviour = "ok"
    pool = GraderPool([server], name="sandbox")
    assert pool.in_flight == 0
    pool.grade(["a"])
    assert pool.in_flight == 0, "a finished request must not stay counted"


def test_unreachable_server_reports_which_pool_failed():
    pool = GraderPool(["http://127.0.0.1:1"], name="sandbox", timeout=0.5, retries=0)
    with pytest.raises(GradeError, match="sandbox"):
        pool.grade(["a"])


def test_health_separates_live_from_dead(server):
    _Handler.behaviour = "ok"
    pool = GraderPool([server, "http://127.0.0.1:1"], name="world_model")
    h = pool.health()
    assert h["alive"] == [server]
    assert h["dead"] == ["http://127.0.0.1:1"]


def test_requests_are_spread_across_the_pool(server):
    _Handler.behaviour = "ok"
    pool = GraderPool([server, server, server], name="world_model")
    picks = {pool._pick() for _ in range(6)}
    assert picks == {server}
    assert pool._next is not None


def test_empty_pool_is_rejected():
    with pytest.raises(ValueError, match="no server URLs"):
        GraderPool([], name="world_model")
