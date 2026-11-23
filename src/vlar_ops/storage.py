"""Key-value object storage: a local directory (tests, local runs) or a Cloud Storage bucket.

Keys are POSIX-style relative paths. Writes are whole-object and idempotent, which is all the queue and sync
need. Objects with several writers (the spend ledger) use a conditional write instead: `get_versioned` returns the
object with a version token and `put_if` writes only if the object still has that version (Cloud Storage's
`if_generation_match`; a file lock and a content digest locally), so the caller re-reads and retries on a conflict.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Protocol

ABSENT = "0"           # the version token of a missing object


class Storage(Protocol):
    url: str

    def put_bytes(self, key: str, data: bytes) -> None: ...
    def put_file(self, key: str, path: Path) -> None: ...
    def get_bytes(self, key: str) -> Optional[bytes]: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str) -> Iterator[str]: ...
    def total_bytes(self, prefix: str = "") -> int: ...
    def get_versioned(self, key: str) -> tuple[Optional[bytes], str]: ...
    def put_if(self, key: str, data: bytes, version: str) -> bool: ...


def _check_key(key: str) -> str:
    if key.startswith("/") or ".." in key.split("/") or not key:
        raise ValueError(f"bad storage key {key!r}")
    return key


class LocalStorage:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.url = f"file://{self.root.resolve()}"

    def _path(self, key: str) -> Path:
        return self.root / _check_key(key)

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        tmp.write_bytes(data)
        tmp.replace(p)

    def put_file(self, key: str, path: Path) -> None:
        self.put_bytes(key, Path(path).read_bytes())

    def get_bytes(self, key: str) -> Optional[bytes]:
        p = self._path(key)
        return p.read_bytes() if p.is_file() else None

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str) -> Iterator[str]:
        base = self.root / prefix if prefix else self.root
        start = base if base.is_dir() else base.parent
        if not start.exists():
            return
        for p in sorted(start.rglob("*")):
            if p.is_file() and ".tmp" not in p.name:
                key = p.relative_to(self.root).as_posix()
                if key.startswith(prefix) and not key.startswith(".locks/"):
                    yield key

    def total_bytes(self, prefix: str = "") -> int:
        return sum(self._path(k).stat().st_size for k in self.list(prefix))

    @contextmanager
    def _locked(self, key: str):
        """An exclusive lock per key, across threads and processes (flock on a lock file beside the root)."""
        lock = self.root / ".locks" / (_check_key(key).replace("/", "__") + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        with open(lock, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    @staticmethod
    def _version(data: Optional[bytes]) -> str:
        return ABSENT if data is None else hashlib.sha256(data).hexdigest()

    def get_versioned(self, key: str) -> tuple[Optional[bytes], str]:
        with self._locked(key):
            data = self.get_bytes(key)
        return data, self._version(data)

    def put_if(self, key: str, data: bytes, version: str) -> bool:
        with self._locked(key):
            if self._version(self.get_bytes(key)) != version:
                return False
            self.put_bytes(key, data)
            return True


class GCSStorage:
    """Cloud Storage via google-cloud-storage (install the `gcs` extra). Uses the VM's service account."""

    def __init__(self, bucket: str, prefix: str = ""):
        from google.cloud import storage as gcs  # noqa: PLC0415  (optional dependency)

        self._client = gcs.Client()
        self._bucket = self._client.bucket(bucket)
        self.prefix = prefix.strip("/")
        self.url = f"gs://{bucket}/{self.prefix}" if self.prefix else f"gs://{bucket}"

    def _name(self, key: str) -> str:
        key = _check_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes) -> None:
        self._bucket.blob(self._name(key)).upload_from_string(data)

    def put_file(self, key: str, path: Path) -> None:
        self._bucket.blob(self._name(key)).upload_from_filename(str(path))

    def get_bytes(self, key: str) -> Optional[bytes]:
        blob = self._bucket.blob(self._name(key))
        try:
            return blob.download_as_bytes()
        except Exception as e:  # google.api_core.exceptions.NotFound
            if type(e).__name__ == "NotFound":
                return None
            raise

    def exists(self, key: str) -> bool:
        return self._bucket.blob(self._name(key)).exists()

    def list(self, prefix: str) -> Iterator[str]:
        full = self._name(prefix) if prefix else (self.prefix + "/" if self.prefix else "")
        cut = len(self.prefix) + 1 if self.prefix else 0
        for blob in self._client.list_blobs(self._bucket, prefix=full):
            yield blob.name[cut:]

    def total_bytes(self, prefix: str = "") -> int:
        full = self._name(prefix) if prefix else (self.prefix + "/" if self.prefix else "")
        return sum(b.size or 0 for b in self._client.list_blobs(self._bucket, prefix=full))

    def get_versioned(self, key: str) -> tuple[Optional[bytes], str]:
        """The object and its generation, read at that generation (a write in between is retried)."""
        for _ in range(10):
            blob = self._bucket.get_blob(self._name(key))
            if blob is None:
                return None, ABSENT
            try:
                return blob.download_as_bytes(if_generation_match=blob.generation), str(blob.generation)
            except Exception as e:  # google.api_core.exceptions.PreconditionFailed / NotFound: changed meanwhile
                if type(e).__name__ not in ("PreconditionFailed", "NotFound"):
                    raise
        raise RuntimeError(f"{key} keeps changing while being read")

    def put_if(self, key: str, data: bytes, version: str) -> bool:
        """Write only if the object is still at `version` (generation 0: only if it does not exist)."""
        try:
            self._bucket.blob(self._name(key)).upload_from_string(data, if_generation_match=int(version))
        except Exception as e:  # google.api_core.exceptions.PreconditionFailed (HTTP 412)
            if type(e).__name__ == "PreconditionFailed":
                return False
            raise
        return True


def open_storage(url: str) -> Storage:
    """`gs://bucket[/prefix]`, `file:///abs/path` or a plain local path."""
    if url.startswith("gs://"):
        bucket, _, prefix = url[5:].partition("/")
        return GCSStorage(bucket, prefix)
    if url.startswith("file://"):
        return LocalStorage(url[7:])
    return LocalStorage(url)
