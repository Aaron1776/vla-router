"""The reference run's per-episode results, for the parity checks (docs/SPEC.md §2, §11 steps 4-5).

Read-only. Layout, as delivered in `References/`:
  task_<t>/eval_info.json   π0.5 via lerobot-eval: per_task[0].metrics.successes, 50 booleans in episode order
  progress.jsonl            X-VLA via the authors' client: {task_id, episode, success, steps, step_limit, ...}

In both runs episode e of task t used initial state e. Scene seeds: 1000 + e for π0.5 (lerobot-eval's
seed 1000 + episode), 142 + e for X-VLA (the client's 42 + e + 100), the conventions of SPEC §3.
π0.5's file has no per-episode step counts; X-VLA ran to 900 steps and is also scored at 520.

`check()` validates the files and the totals the user confirmed (2026-11-21); `compare()` pairs a
project run's per-episode outcomes with the reference's on the same initial states.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

N_TASKS, N_INITS = 10, 50
CHECKSUMS = Path(__file__).parent / "configs" / "reference-run.sha256"
EXPECTED = {"episodes": 500, "pi05_successes": 478, "xvla_successes_900": 412, "xvla_successes_520": 405}
SCENE_SEED_OFFSET = {"pi05": 1000, "xvla": 142}


class ReferenceDataError(ValueError):
    pass


@dataclass(frozen=True)
class RefEpisode:
    model: str
    task_id: int
    init_idx: int
    scene_seed: int
    success: bool
    steps: Optional[int]          # X-VLA only
    step_limit: int

    def success_within(self, limit: int) -> bool:
        if not self.success:
            return False
        if self.steps is None:
            if limit < self.step_limit:
                raise ReferenceDataError("π0.5 reference has no step counts; cannot score it below its own limit")
            return True
        return self.steps <= limit


def load_pi05(root: Path | str) -> dict[tuple[int, int], RefEpisode]:
    root = Path(root)
    out: dict[tuple[int, int], RefEpisode] = {}
    for t in range(N_TASKS):
        path = root / f"task_{t}" / "eval_info.json"
        if not path.exists():
            raise ReferenceDataError(f"missing {path}")
        info = json.loads(path.read_text())
        per_task = info.get("per_task", [])
        if len(per_task) != 1 or per_task[0].get("task_id") != t or per_task[0].get("task_group") != "libero_10":
            raise ReferenceDataError(f"{path}: expected one libero_10 entry for task {t}")
        successes = per_task[0]["metrics"]["successes"]
        if len(successes) != N_INITS or not all(isinstance(s, bool) for s in successes):
            raise ReferenceDataError(f"{path}: expected {N_INITS} boolean successes")
        overall = info.get("overall", {})
        if overall.get("n_episodes") != N_INITS or not math.isclose(overall.get("pc_success", -1),
                                                                     100.0 * sum(successes) / N_INITS):
            raise ReferenceDataError(f"{path}: overall block disagrees with the per-episode list")
        for e, s in enumerate(successes):
            out[(t, e)] = RefEpisode("pi05", t, e, SCENE_SEED_OFFSET["pi05"] + e, s, None, 520)
    return out


def load_xvla(root: Path | str) -> dict[tuple[int, int], RefEpisode]:
    path = Path(root) / "progress.jsonl"
    if not path.exists():
        raise ReferenceDataError(f"missing {path}")
    out: dict[tuple[int, int], RefEpisode] = {}
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        r = json.loads(line)
        key = (int(r["task_id"]), int(r["episode"]))
        if key in out:
            raise ReferenceDataError(f"{path}:{n}: duplicate episode {key}")
        if r["step_limit"] != 900 or not (0 < r["steps"] <= 900):
            raise ReferenceDataError(f"{path}:{n}: unexpected step counts {r['steps']}/{r['step_limit']}")
        if not r["success"] and r["steps"] != r["step_limit"]:
            raise ReferenceDataError(f"{path}:{n}: a failed episode stopped before the step limit")
        out[key] = RefEpisode("xvla", key[0], key[1], SCENE_SEED_OFFSET["xvla"] + key[1], bool(r["success"]),
                              int(r["steps"]), int(r["step_limit"]))
    missing = [(t, e) for t in range(N_TASKS) for e in range(N_INITS) if (t, e) not in out]
    if missing:
        raise ReferenceDataError(f"{path}: {len(missing)} episodes missing, e.g. {missing[:3]}")
    return out


def pinned_checksums() -> dict[str, str]:
    """{path relative to the References folder: sha256} from the pinned manifest."""
    out = {}
    for line in CHECKSUMS.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, path = line.split(maxsplit=1)
            out[path.strip().split("References/", 1)[-1]] = digest
    return out


def verify_files(root: Path | str, expected: Optional[Mapping[str, str]] = None) -> None:
    """Every pinned file must exist under `root` with exactly the pinned bytes."""
    root = Path(root)
    expected = pinned_checksums() if expected is None else expected
    bad = []
    for rel, digest in sorted(expected.items()):
        p = root / rel
        if not p.exists():
            bad.append(f"{rel} (missing)")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            bad.append(f"{rel} (sha256 differs)")
    if bad:
        raise ReferenceDataError(f"reference files under {root} are not the pinned ones: {', '.join(bad)}")


def check(root: Path | str, *, verify_checksums: bool = True) -> dict:
    """Verify the files' pinned SHA-256, load both reference runs and confirm the totals."""
    if verify_checksums:
        verify_files(root)
    pi05, xvla = load_pi05(root), load_xvla(root)
    totals = {
        "episodes": len(xvla) if len(xvla) == len(pi05) else -1,
        "pi05_successes": sum(e.success for e in pi05.values()),
        "xvla_successes_900": sum(e.success for e in xvla.values()),
        "xvla_successes_520": sum(e.success_within(520) for e in xvla.values()),
    }
    if totals != EXPECTED:
        raise ReferenceDataError(f"reference totals {totals} != expected {EXPECTED}")
    per_task = {t: {"pi05": sum(pi05[(t, e)].success for e in range(N_INITS)),
                    "xvla_520": sum(xvla[(t, e)].success_within(520) for e in range(N_INITS))}
                for t in range(N_TASKS)}
    return {"totals": totals, "per_task": per_task,
            "pi05_parity_subset": sum(pi05[(t, e)].success for t in range(N_TASKS) for e in range(10))}


# ---------------------------------------------------------------- comparison

def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(ours: Mapping[tuple[int, int], bool], reference: Mapping[tuple[int, int], RefEpisode],
            *, limit: int = 520, keys: Optional[Iterable[tuple[int, int]]] = None) -> dict:
    """Pair a project run's successes with the reference's on the same (task, initial state).

    Returns both rates with Wilson intervals, the paired table (both / only ours / only reference /
    neither), the paired difference with a Wald interval on per-episode differences, and the exact
    McNemar p-value. `within_sampling_error` is True when that p-value is at least 0.05.
    """
    keys = sorted(keys if keys is not None else ours)
    missing = [k for k in keys if k not in reference or k not in ours]
    if missing:
        raise ReferenceDataError(f"episodes missing from one side: {missing[:3]}")
    pairs = [(bool(ours[k]), reference[k].success_within(limit)) for k in keys]
    n = len(pairs)
    both = sum(o and r for o, r in pairs)
    only_ours = sum(o and not r for o, r in pairs)
    only_ref = sum(r and not o for o, r in pairs)
    k_ours, k_ref = both + only_ours, both + only_ref
    diffs = [int(o) - int(r) for o, r in pairs]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else float("nan")
    half = 1.959964 * math.sqrt(var / n) if n > 1 else float("nan")
    p = _mcnemar_exact_p(only_ours, only_ref)
    return {"n": n, "ours": k_ours, "reference": k_ref, "ours_ci": _wilson(k_ours, n), "reference_ci": _wilson(k_ref, n),
            "both": both, "only_ours": only_ours, "only_reference": only_ref, "neither": n - both - only_ours - only_ref,
            "paired_difference": mean, "paired_difference_ci": (mean - half, mean + half), "mcnemar_p": p,
            "within_sampling_error": p >= 0.05}
