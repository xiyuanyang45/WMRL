"""Shared storage: how the three nodes leave messages for each other.

The nodes need somewhere to put small coordination files (which port the world
model server came up on, whether the run has finished) and large artifacts
(checkpoints, staged data). Anything all three can read works: a shared
filesystem if there is one, an object store otherwise.

The interface is deliberately four calls, because that is all the training loop
uses. Adding a backend means implementing those four.

Set ``WMRL_STORE`` to pick one:

===============================  =============================================
``/mnt/shared/wmrl``             a path all nodes mount
``s3://bucket/prefix``           an object store   # audit-allow: example
=============================== ==============================================

A local path that is *not* shared between nodes will appear to work on a single
node and then hang the moment a second one joins, waiting for a registration
file it can never see. :func:`open_store` warns when it sees that shape.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import time

__all__ = ["Store", "LocalStore", "S3Store", "open_store"]


class Store:
    """Small key-value surface over shared storage. Keys are ``a/b/c`` strings."""

    def put(self, key: str, data: str | bytes) -> None:
        raise NotImplementedError

    def get(self, key: str) -> str | None:
        """Return the value, or ``None`` when the key does not exist."""
        raise NotImplementedError

    def list(self, prefix: str) -> list[str]:
        raise NotImplementedError

    def mtime(self, key: str) -> float | None:
        raise NotImplementedError

    # -- shared conveniences ----------------------------------------------

    def wait_for(self, key: str, timeout: float = 2400, poll: float = 5.0) -> str:
        """Block until a key appears. Raises :class:`TimeoutError` with context.

        Used at rendezvous: the trainer cannot start until the world model and
        sandbox nodes have registered, and a node that never registers should
        fail with something a human can act on rather than hang forever.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            v = self.get(key)
            if v is not None:
                return v
            time.sleep(poll)
        raise TimeoutError(
            f"{key!r} never appeared in {self} after {timeout:.0f}s. "
            "The node that writes it either failed to start or cannot reach this store."
        )


class LocalStore(Store):
    """A directory. Correct only if every node sees the same one."""

    def __init__(self, root: str):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> pathlib.Path:
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"key {key!r} escapes the store root")
        return p

    def put(self, key, data):
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode, payload = ("wb", data) if isinstance(data, bytes) else ("w", str(data))
        # write-then-rename: a reader must never see a half-written registration
        tmp = p.with_suffix(p.suffix + f".tmp.{os.getpid()}")
        with open(tmp, mode) as fh:
            fh.write(payload)
        os.replace(tmp, p)

    def get(self, key):
        p = self._p(key)
        if not p.exists():
            return None
        return p.read_text()

    def list(self, prefix):
        base = self.root / prefix
        parent, stem = (base.parent, base.name) if not base.is_dir() else (base, "")
        if not parent.exists():
            return []
        out = []
        for f in parent.rglob("*"):
            if f.is_file() and (not stem or f.name.startswith(stem)):
                out.append(str(f.relative_to(self.root)))
        return sorted(out)

    def mtime(self, key):
        p = self._p(key)
        return p.stat().st_mtime if p.exists() else None

    def __str__(self):
        return f"local:{self.root}"


class S3Store(Store):
    """An object-store prefix. Needs ``boto3`` and credentials on every node."""

    def __init__(self, uri: str):
        rest = uri[5:]
        self.bucket, _, self.prefix = rest.partition("/")
        self.prefix = self.prefix.rstrip("/")
        if not self.bucket:
            raise ValueError(f"malformed store URI {uri!r}; expected s3://bucket/prefix")  # audit-allow: documented example
        import boto3  # imported lazily so the local backend needs no cloud SDK

        self._s3 = boto3.client("s3")

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key, data):
        body = data if isinstance(data, bytes) else str(data).encode()
        self._s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=body)

    def get(self, key):
        import botocore

        try:
            return self._s3.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read().decode()
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise

    def list(self, prefix):
        out, token = [], None
        base = self._k(prefix)
        cut = len(self._k("")) if self.prefix else 0
        while True:
            kw = {"Bucket": self.bucket, "Prefix": base}
            if token:
                kw["ContinuationToken"] = token
            r = self._s3.list_objects_v2(**kw)
            out += [o["Key"][cut:] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return sorted(out)
            token = r.get("NextContinuationToken")

    def mtime(self, key):
        import botocore

        try:
            head = self._s3.head_object(Bucket=self.bucket, Key=self._k(key))
            return head["LastModified"].timestamp()
        except botocore.exceptions.ClientError:
            return None

    def __str__(self):
        return f"s3://{self.bucket}/{self.prefix}"


def open_store(uri: str | None = None, *, multi_node: bool = True) -> Store:
    """Open the store named by ``uri`` or by ``WMRL_STORE``."""
    uri = uri or os.environ.get("WMRL_STORE")
    if not uri:
        raise SystemExit(
            "no shared storage. Set WMRL_STORE to a path every node mounts, "
            "or to an s3://bucket/prefix every node can reach."  # audit-allow: documented example
        )

    if uri.startswith("s3://"):
        return S3Store(uri)

    if multi_node and not _looks_shared(uri):
        print(
            f"[store] warning: {uri} does not look like a shared mount. If the other "
            "nodes cannot see this directory, rendezvous will time out.",
            flush=True,
        )
    return LocalStore(uri)


def _looks_shared(path: str) -> bool:
    """Best-effort check that a local path is on a network filesystem."""
    try:
        out = shutil.which("stat") and os.popen(f"stat -f -L -c %T {path!r} 2>/dev/null").read().strip()
    except OSError:
        return True  # cannot tell: stay quiet rather than cry wolf
    if not out:
        return True
    return out.lower() in {"nfs", "lustre", "gpfs", "ceph", "smb2", "cifs", "fuseblk", "beegfs"}
