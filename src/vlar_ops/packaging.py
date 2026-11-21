"""Archives for download to the user's computer (docs/SPEC.md §13).

Each archive set is a series of independent zip files in "store" mode (no recompression),
each at most about `part_bytes`, so a failed download is retried one part at a time and any
part opens on its own. Files are packed in sorted path order with a fixed timestamp, so
packaging the same tree twice gives byte-identical parts. Alongside the parts:

  <name>.index.json   every packed file: its part, size and SHA-256
  SHA256SUMS          SHA-256 of every part and of the index (`shasum -a 256 -c SHA256SUMS`)
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_PART_BYTES = 10 * 1024**3
FIXED_DATE = (1980, 1, 1, 0, 0, 0)
CHUNK = 8 * 1024**2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def _files(src: Path) -> list[Path]:
    return sorted((p for p in src.rglob("*") if p.is_file() and not p.is_symlink()),
                  key=lambda p: p.relative_to(src).as_posix())


def _assign_parts(sizes: Iterable[tuple[str, int]], part_bytes: int) -> list[list[str]]:
    parts: list[list[str]] = []
    current: list[str] = []
    used = 0
    for name, size in sizes:
        cost = size + 200 + 2 * len(name)          # data plus local and central headers, roughly
        if current and used + cost > part_bytes:
            parts.append(current)
            current, used = [], 0
        current.append(name)
        used += cost
    if current:
        parts.append(current)
    return parts


@dataclass(frozen=True)
class ArchiveSet:
    name: str
    out_dir: Path
    parts: list[Path]
    index: Path
    sums: Path


def build(src_dir: Path | str, out_dir: Path | str, name: str, part_bytes: int = DEFAULT_PART_BYTES) -> ArchiveSet:
    src, out = Path(src_dir), Path(out_dir)
    if not src.is_dir():
        raise FileNotFoundError(src)
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve().is_relative_to(src.resolve()):
        raise ValueError("out_dir must not be inside src_dir")
    files = _files(src)
    if not files:
        raise ValueError(f"{src} has no files")
    rel = {p.relative_to(src).as_posix(): p for p in files}
    groups = _assign_parts(((r, rel[r].stat().st_size) for r in rel), part_bytes)
    width = max(3, len(str(len(groups))))
    index: dict[str, dict] = {}
    parts: list[Path] = []
    for k, group in enumerate(groups, start=1):
        part = out / f"{name}.part{k:0{width}d}-of-{len(groups):0{width}d}.zip"
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for r in group:
                info = zipfile.ZipInfo(r, date_time=FIXED_DATE)
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o644 << 16
                h = hashlib.sha256()
                with open(rel[r], "rb") as fin, zf.open(info, "w", force_zip64=True) as fout:
                    while block := fin.read(CHUNK):
                        h.update(block)
                        fout.write(block)
                index[r] = {"part": part.name, "bytes": rel[r].stat().st_size, "sha256": h.hexdigest()}
        parts.append(part)
    index_path = out / f"{name}.index.json"
    index_path.write_text(json.dumps({"name": name, "n_parts": len(parts), "files": index},
                                     indent=1, sort_keys=True) + "\n")
    sums = out / "SHA256SUMS"
    lines = [f"{sha256_file(p)}  {p.name}" for p in [*parts, index_path]]
    sums.write_text("\n".join(lines) + "\n")
    return ArchiveSet(name, out, parts, index_path, sums)


def verify(out_dir: Path | str, deep: bool = True) -> list[str]:
    """Check every part against SHA256SUMS; with `deep`, also every file against the index.
    Returns a list of problems (empty means verified)."""
    out = Path(out_dir)
    problems: list[str] = []
    sums = out / "SHA256SUMS"
    if not sums.exists():
        return ["SHA256SUMS missing"]
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        if line.strip():
            digest, fname = line.split(maxsplit=1)
            listed[fname.strip()] = digest
    for fname, digest in listed.items():
        p = out / fname
        if not p.exists():
            problems.append(f"missing {fname}")
        elif sha256_file(p) != digest:
            problems.append(f"checksum mismatch {fname}")
    if problems or not deep:
        return problems
    for idx_name in [f for f in listed if f.endswith(".index.json")]:
        idx = json.loads((out / idx_name).read_text())
        by_part: dict[str, list[str]] = {}
        for r, meta in idx["files"].items():
            by_part.setdefault(meta["part"], []).append(r)
        for part_name, names in by_part.items():
            if part_name not in listed:
                problems.append(f"{part_name} is in the index but not in SHA256SUMS")
                continue
            with zipfile.ZipFile(out / part_name) as zf:
                members = set(zf.namelist())
                for r in names:
                    if r not in members:
                        problems.append(f"{r} missing from {part_name}")
                        continue
                    h = hashlib.sha256()
                    with zf.open(r) as f:
                        while block := f.read(CHUNK):
                            h.update(block)
                    if h.hexdigest() != idx["files"][r]["sha256"]:
                        problems.append(f"content mismatch {r} in {part_name}")
    return problems
