"""Key-value object storage: a local directory (tests, local runs) or a Cloud Storage bucket.

Keys are POSIX-style relative paths. Writes are whole-object and idempotent, which is all
the queue, ledger and sync need.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional, Protocol


class Storage(Protocol):
    url: str

    def put_bytes(self, key: str, data: bytes) -> None: ...
    def put_file(self, key: str, path: Path) -> None: ...
    def get_bytes(self, key: str) -> Optional[bytes]: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str) -> Iterator[str]: ...
    def total_bytes(self, prefix: str = "") -> int: ...


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
                if key.startswith(prefix):
                    yield key

    def total_bytes(self, prefix: str = "") -> int:
        return sum(self._path(k).stat().st_size for k in self.list(prefix))


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


def open_storage(url: str) -> Storage:
    """`gs://bucket[/prefix]`, `file:///abs/path` or a plain local path."""
    if url.startswith("gs://"):
        bucket, _, prefix = url[5:].partition("/")
        return GCSStorage(bucket, prefix)
    if url.startswith("file://"):
        return LocalStorage(url[7:])
    return LocalStorage(url)
