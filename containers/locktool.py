"""Lockfile helpers for the container builds (docs/SPEC.md §15, amendment 28). Standard library only, so it runs in
the bare base image.

    python locktool.py sdists LOCK                  # names in LOCK with no wheel for CPython 3.13 on linux/x86_64
                                                    # (asks PyPI; containers/lock.sh writes them to <env>.sdists)
    python locktool.py split LOCK SDISTS --wheels W --sdists S

`split` writes two pip requirement files, each entry with its hashes: the packages installed from hashed wheels,
and the sdist-only ones, which the image builds into wheels in a builder stage from their hash-checked sources with
the hashed build tools of build.lock (pip does not hash-check an sdist's own build dependencies).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==")


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def entries(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    """(option lines such as --index-url, [(canonical name, the entry's full text with its hash lines)])."""
    options, out, cur, name = [], [], [], None
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("--") and name is None and not cur:
            options.append(s)
            continue
        m = _PIN.match(s)
        if m:
            if name is not None:
                out.append((name, "\n".join(cur)))
            name, cur = canon(m.group(1)), [line]
        elif name is not None:
            cur.append(line)
    if name is not None:
        out.append((name, "\n".join(cur)))
    return options, out


def split(lock: str, sdist_names: set[str]) -> tuple[str, str]:
    options, ents = entries(lock)
    unknown = sdist_names - {n for n, _ in ents}
    if unknown:
        raise SystemExit(f"listed as sdist-only but not in the lock: {sorted(unknown)}")
    head = "\n".join(options)
    wheels = [e for n, e in ents if n not in sdist_names]
    sdists = [e for n, e in ents if n in sdist_names]
    return (head + "\n" + "\n".join(wheels) + "\n", head + "\n" + "\n".join(sdists) + "\n")


PY_MINOR = 13                      # CPython 3.13, as the reference notebook (amendment 30)


def has_wheel(filenames: list[str], minor: int = PY_MINOR) -> bool:
    """Whether any file is a wheel pip would install on CPython 3.<minor>, linux/x86_64: python tag py3, py3<minor>
    or cp3<minor>, or an abi3 wheel for cp3k with k <= minor."""
    for fn in filenames:
        if not fn.endswith(".whl"):
            continue
        py, abi, plat = fn[:-4].split("-")[-3:]
        tags = py.split(".")
        pyok = any(t in ("py3", f"py3{minor}", f"cp3{minor}") for t in tags) or (
            abi == "abi3" and any(re.fullmatch(r"cp3(\d+)", t) and int(t[3:]) <= minor for t in tags))
        if pyok and (plat == "any" or ("x86_64" in plat and "linux" in plat)):
            return True
    return False


def sdist_only(lock: str) -> list[str]:
    ctx = None
    try:
        import ssl

        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    out = []
    for line in lock.splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", line.strip())
        if not m:
            continue
        with urllib.request.urlopen(f"https://pypi.org/pypi/{m.group(1)}/{m.group(2)}/json", timeout=60,
                                    context=ctx) as r:
            files = [u["filename"] for u in json.load(r)["urls"]]
        if not has_wheel(files):
            out.append(canon(m.group(1)))
    return sorted(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sdists")
    s.add_argument("lock", type=Path)
    p = sub.add_parser("split")
    p.add_argument("lock", type=Path)
    p.add_argument("sdists", type=Path)
    p.add_argument("--wheels", type=Path, required=True)
    p.add_argument("--sdists", dest="sdists_out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "sdists":
        print("\n".join(sdist_only(a.lock.read_text())))
        return
    names = {canon(n) for n in a.sdists.read_text().split() if n and not n.startswith("#")}
    wheels, sdists = split(a.lock.read_text(), names)
    a.wheels.write_text(wheels)
    a.sdists_out.write_text(sdists)
    print(f"{a.lock.name}: {len(names)} built from source, the rest from hashed wheels", file=sys.stderr)


if __name__ == "__main__":
    main()
