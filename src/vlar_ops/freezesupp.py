"""Freeze supplements: image digests the v1.0 freeze record left out (docs/SPEC.md §15, §17 amendment 54).

`600-freeze.sh`'s `finish freeze` container mounted only the calibration directory, so `freeze.json` recorded
`image_digests: {}` while every record made before it carries the images it ran on. The frozen file refuses changes,
so the digests are pinned by a dated supplement beside it, `freeze-supplement-<YYYY-MM-DD>.json`, which names the
freeze record by its SHA-256. A supplement is not a re-freeze: harness, ops code, config and calibration stay pinned by
`freeze.json` alone.

- `make` builds a supplement from the freeze record and result markers written at or before the freeze: every marker
  must carry the same image digests, or there is nothing to pin.
- `frozen_image_digests` reads the images a freeze pins: its own `image_digests` if it has any, else its supplements'
  (each naming this exact freeze file; several must agree; both sources must agree when both exist).
- `check_running` is the boot cross-check: the running images (`provenance.image_digests()`, from
  `/etc/vlar/image-digests.json`) must equal the frozen ones. The golden-set self-check and every worker of a frozen
  harness run it, so no record is made on other images.
The dataset manifest requires every record of a frozen dataset to carry exactly these digests (`manifest.build`).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

PREFIX = "freeze-supplement-"
IMAGES = ("sim", "pi05", "xvla")
VERSION = 1


class SupplementError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_digests(d: Any, where: str) -> dict[str, str]:
    if not isinstance(d, dict) or any(not str(d.get(k, "")).startswith("sha256:") for k in IMAGES):
        raise SupplementError(f"{where}: image digests must name {', '.join(IMAGES)} as sha256:..., got {d!r}")
    return dict(d)


def name_for(date: str) -> str:
    datetime.strptime(date, "%Y-%m-%d")
    return f"{PREFIX}{date}.json"


def make(freeze: Path, markers: Iterable[Mapping[str, Any]], *, date: str, evidence: Optional[list[str]] = None,
         now: Optional[datetime] = None) -> dict[str, Any]:
    """The supplement for `freeze` from the result markers written at or before it (`finished_at` <= `frozen_at`):
    every one must carry the same image digests."""
    rec = json.loads(Path(freeze).read_text())
    frozen_at = datetime.fromisoformat(rec["frozen_at"]).timestamp()
    seen: dict[str, int] = {}
    n, latest = 0, None
    for m in markers:
        t = float(m["finished_at"])
        if t > frozen_at:
            raise SupplementError(f"marker {m['job']['phase']}/{m['job']['job_id']} was written after the freeze")
        d = ((m.get("meta") or {}).get("provenance") or {}).get("image_digests")
        seen[json.dumps(d, sort_keys=True)] = seen.get(json.dumps(d, sort_keys=True), 0) + 1
        n, latest = n + 1, max(latest or t, t)
    if n == 0:
        raise SupplementError("no markers: nothing to take the image digests from")
    if len(seen) != 1:
        raise SupplementError(f"the markers disagree on the image digests: {seen}")
    digests = _check_digests(json.loads(next(iter(seen))), "markers")
    own = rec.get("image_digests") or {}
    if own and own != digests:
        raise SupplementError("the freeze record's own image digests differ from the markers'")
    return {
        "supplement_version": VERSION, "supplements": Path(freeze).name, "freeze_sha256": _sha256(freeze),
        "frozen_at": rec["frozen_at"], "calibration_hash": rec["calibration_hash"], "image_digests": digests,
        "evidence": {"markers": n, "latest_marker_at": datetime.fromtimestamp(latest, timezone.utc)
                     .isoformat(timespec="seconds"), "notes": list(evidence or [])},
        "date": date, "written_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
    }


def supplements(freeze: Path) -> list[Path]:
    return sorted(Path(freeze).parent.glob(f"{PREFIX}*.json"))


def frozen_image_digests(freeze: Path) -> dict[str, str]:
    """The image digests the freeze pins (its own, else its supplements'); {} if it pins none."""
    freeze = Path(freeze)
    own = json.loads(freeze.read_text()).get("image_digests") or {}
    found: dict[str, str] = {}
    sha = _sha256(freeze)
    for p in supplements(freeze):
        s = json.loads(p.read_text())
        if s.get("freeze_sha256") != sha:
            raise SupplementError(f"{p.name} supplements another freeze record (SHA-256 {str(s.get('freeze_sha256'))[:12]}"
                                  f", this one {sha[:12]})")
        d = _check_digests(s.get("image_digests"), p.name)
        if found and d != found:
            raise SupplementError(f"{p.name} disagrees with an earlier supplement on the image digests")
        found = d
    if own and found and own != found:
        raise SupplementError("the freeze record's own image digests differ from its supplement's")
    return dict(own or found)


def check_running(freeze: Path, running: Mapping[str, str]) -> dict[str, str]:
    """Raise unless the running images are the frozen ones; returns the frozen digests."""
    frozen = frozen_image_digests(freeze)
    if not frozen:
        raise SupplementError(f"{Path(freeze).name} pins no image digests and has no {PREFIX}*.json beside it "
                              "(amendment 54)")
    if dict(running) != frozen:
        diff = sorted(k for k in set(frozen) | set(running) if frozen.get(k) != running.get(k))
        raise SupplementError(f"the running images differ from the frozen ones in: {', '.join(diff)}")
    return frozen
