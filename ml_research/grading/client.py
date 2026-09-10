"""Talking to the grading servers.

Both grader kinds answer the same ``POST /grade`` contract, which is what lets
the training loop treat them interchangeably and swap one for the other per
group. The differences that matter are not in the protocol:

* A world model pool has many engines and answers in about a second. Requests
  are spread round-robin, since any engine can serve any query.
* A sandbox pool has a bounded number of execution slots and answers in minutes.
  Requests block when the slots are full, and ``in_flight`` is what the anchor
  scheduler reads to avoid queueing work the sandbox has no room for.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
import urllib.error
import urllib.request

__all__ = ["GraderPool", "GradeError"]


class GradeError(RuntimeError):
    """A grader could not score a trajectory."""


class GraderPool:
    """A set of interchangeable grading servers behind one call."""

    def __init__(self, urls: list[str], name: str = "grader", timeout: float = 1800.0,
                 retries: int = 2):
        if not urls:
            raise ValueError(f"{name}: no server URLs")
        self.urls = list(urls)
        self.name = name
        self.timeout = float(timeout)
        self.retries = int(retries)

        self._next = itertools.cycle(range(len(self.urls)))
        self._lock = threading.Lock()
        self._in_flight = 0
        self.n_requests = 0
        self.n_failures = 0

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def _pick(self) -> str:
        with self._lock:
            return self.urls[next(self._next)]

    def grade(self, trajectories) -> list[float]:
        """Score a group. Returns one float per trajectory, in order.

        A trajectory the grader could not score comes back as ``None`` rather
        than ``0.0``: the advantage code drops those, and scoring them zero
        would assert the attempt was bad when all we know is that it was not
        measured.
        """
        payload = json.dumps({"trajectories": list(trajectories)}).encode()
        url = self._pick()

        with self._lock:
            self._in_flight += 1
            self.n_requests += 1
        try:
            body = self._post(f"{url}/grade", payload)
        finally:
            with self._lock:
                self._in_flight -= 1

        scores = body.get("scores")
        if not isinstance(scores, list) or len(scores) != len(trajectories):
            raise GradeError(
                f"{self.name} at {url} returned {len(scores) if isinstance(scores, list) else '?'} "
                f"score(s) for {len(trajectories)} trajectory/ies"
            )
        return [None if s is None else float(s) for s in scores]

    def _post(self, url: str, payload: bytes) -> dict:
        last = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
                last = e
                with self._lock:
                    self.n_failures += 1
                if attempt < self.retries:
                    time.sleep(2.0 * (attempt + 1))
        raise GradeError(f"{self.name} at {url} failed after {self.retries + 1} attempt(s): {last}")

    def health(self) -> dict:
        """Which servers in this pool are answering."""
        alive = []
        for u in self.urls:
            try:
                with urllib.request.urlopen(f"{u}/health", timeout=5):
                    alive.append(u)
            except (urllib.error.URLError, OSError):
                pass
        return {"name": self.name, "alive": alive,
                "dead": [u for u in self.urls if u not in alive]}

    def __repr__(self):
        return f"GraderPool({self.name!r}, {len(self.urls)} server(s))"
