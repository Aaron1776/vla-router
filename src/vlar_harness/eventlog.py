"""Event logs on disk, their checks, and the diagnostics read from them (docs/SPEC.md §5, §11 step 3).

Interim file format until the dataset's Parquet tables (§13): JSON lines, one episode or branch per line, each
an object `{"events": [...], ...metadata}`. Python's json writes floats with repr, so every value round-trips
exactly and re-scoring a stored log gives the same clock as the live run.

    python -m vlar_harness.eventlog first-targets base.jsonl ... --out first_targets.json

writes the first-step target distances X-VLA produced at ordinary chunk boundaries (context `stretch`: its queue
ran out within an uninterrupted X-VLA stretch, no handoff) in the given logs: the list the VM handoff check reads
from `VLAR_BASE_FIRST_TARGETS` (§11 step 3, amendment 10).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .api import check_reset
from .clock import STEP_RUN, ZERO_COST, CHARGED

KNOWN_TYPES = frozenset(ZERO_COST) | set(CHARGED) | {STEP_RUN}


class LogError(ValueError):
    pass


def dumps(events: Sequence[Mapping]) -> str:
    return json.dumps(list(events), separators=(",", ":"), allow_nan=False)


def loads(text: str) -> list[dict]:
    return json.loads(text)


def write_jsonl(path: Path | str, records: Iterable[Mapping]) -> int:
    n = 0
    with open(path, "w") as f:
        for r in records:
            if "events" not in r:
                raise LogError("each record needs an 'events' list")
            f.write(json.dumps(dict(r), separators=(",", ":"), allow_nan=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> Iterator[dict]:
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                rec = json.loads(line)
                if "events" not in rec:
                    raise LogError(f"{path}:{i + 1}: no 'events'")
                yield rec


def read_logs(paths: Iterable[Path | str]) -> list[list[dict]]:
    return [rec["events"] for p in paths for rec in read_jsonl(p)]


def check_start(start: Mapping) -> None:
    """The start event records the live time limit explicitly (amendment 20), consistently with its value, and
    only where one is allowed (routed mode with a decider profile, amendment 19)."""
    flag = start.get("time_limit")
    if not isinstance(flag, bool):
        raise LogError("the start event records time_limit true or false (amendment 20)")
    if flag != (start.get("time_limit_s") is not None):
        raise LogError(f"the start event's time_limit={flag} disagrees with time_limit_s={start.get('time_limit_s')!r}")
    if flag:
        try:
            check_reset(start.get("mode"), start.get("profile"), True)
        except (TypeError, ValueError) as e:
            raise LogError(f"a run with a live time limit that is not allowed: {e}") from None


def check(events: Sequence[Mapping], *, complete: bool = True) -> dict:
    """Structural checks; returns the start event. A complete log ends with exactly one `end` event."""
    if not events or events[0].get("t") != "start":
        raise LogError("a log starts with its start event")
    check_start(events[0])
    unknown = sorted({e.get("t") for e in events} - KNOWN_TYPES, key=str)
    if unknown:
        raise LogError(f"unknown event types {unknown}")
    if any(e["t"] == "start" for e in events[1:]):
        raise LogError("a log has one start event")
    if any(e["t"] == "branch" and e.get("time_limit") is not False for e in events):
        raise LogError("branches run uncut: every branch event records time_limit false (amendment 20)")
    ends = [i for i, e in enumerate(events) if e["t"] == "end"]
    if complete and ends != [len(events) - 1]:
        raise LogError(f"a complete log has exactly one end event, last (found at {ends} of {len(events)})")
    if not complete and ends:
        raise LogError("this log was expected to be unfinished (a snapshot's prefix) but has an end event")
    return dict(events[0])


def first_target_distances(logs: Iterable[Sequence[Mapping]], context: str = "stretch") -> list[float]:
    """Distances (m) between X-VLA's first queued target and the arm at query time, for the `/act` calls of
    the given context (§11 step 3): `stretch` gives the ordinary chunk boundaries of the base rollouts."""
    out = []
    for events in logs:
        for e in events:
            if e["t"] == "xvla_act" and e["context"] == context:
                out.append(float(e["first_target_dist_m"]))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m vlar_harness.eventlog")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ft = sub.add_parser("first-targets", help="X-VLA's first-target distances at ordinary chunk boundaries")
    ft.add_argument("logs", nargs="+", help="JSON-lines event-log files (base rollouts)")
    ft.add_argument("--context", default="stretch")
    ft.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logs = read_logs(a.logs)
    for events in logs:
        check(events)
    d = first_target_distances(logs, a.context)
    Path(a.out).write_text(json.dumps(d))
    print(f"{len(d)} distances from {len(logs)} logs -> {a.out}")


if __name__ == "__main__":
    main()
