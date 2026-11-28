"""X-VLA parity check (docs/SPEC.md §11 step 5): the base rollouts at 520 steps against the reference 405/500.

Run from the Mac after step `400`, on a local copy of the bucket's base-rollout markers:

    gcloud storage cp -r "gs://$BUCKET/vla-router/results/base" <dir>
    .venv/bin/python ops/xvla_parity.py <dir>/base --references References --out <prefix>

Pairs each base episode (task, initial state) with the same episode of the X-VLA reference run
(`References/progress.jsonl`, scored at 520 of its 900 steps) and reports both success rates, the paired table, the
paired difference and the exact McNemar p-value (`vlar_ops.reference_runs.compare`, the code behind the π0.5 gate).
Also per task, and every discordant episode with its step counts. A report only: no gate, nothing queued.
Exit 2 (no verdict) if an episode is missing or duplicated, unless `--partial` (a preview on the episodes so far).
`--out <prefix>` writes `<prefix>.json` and `<prefix>.txt`.

Kept outside `src/vlar_ops` on purpose: every file there is in the ops digest (amendments 46, 49).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))    # run from any of the repo's venvs
from vlar_ops import reference_runs as rr  # noqa: E402

LIMIT = 520


def read_markers(path: Path) -> list[dict]:
    """Marker JSON objects from a directory of `*.json` files, or one file of concatenated objects."""
    texts = [p.read_text() for p in sorted(path.glob("*.json"))] if path.is_dir() else [path.read_text()]
    dec, out = json.JSONDecoder(), []
    for txt in texts:
        i = 0
        while i < len(txt):
            while i < len(txt) and txt[i].isspace():
                i += 1
            if i >= len(txt):
                break
            obj, i = dec.raw_decode(txt, i)
            out.append(obj)
    return out


def report(markers: list[dict], references: Path, *, partial: bool = False) -> dict:
    ours, steps, dupes, over, other = {}, {}, [], [], []
    for m in markers:
        job, summ = m["job"], m["result"]["summary"]
        if job.get("phase") != "base":
            other.append(job.get("subject"))
            continue
        key = (int(job["spec"]["task_id"]), int(job["spec"]["init_idx"]))
        if key in ours:
            dupes.append(key)
        s = int(summ["steps"])
        if s > LIMIT:
            over.append(key)
        ours[key] = bool(summ["success"]) and s <= LIMIT
        steps[key] = s
    ref = rr.load_xvla(references)
    missing = [k for k in sorted(ref) if k not in ours]
    out = {"markers": len(markers), "episodes": len(ours), "duplicates": dupes, "over_limit": over,
           "not_base": other, "missing": missing, "limit": LIMIT, "complete": not missing and not dupes}
    if not ours or (not out["complete"] and not partial):
        return out
    res = rr.compare(ours, ref, limit=LIMIT, keys=sorted(ours))
    per_task = {}
    for (t, e) in sorted(ours):
        o, r = ours[(t, e)], ref[(t, e)].success_within(LIMIT)
        d = per_task.setdefault(t, {"n": 0, "ours": 0, "reference": 0, "only_ours": 0, "only_reference": 0})
        d["n"] += 1; d["ours"] += o; d["reference"] += r; d["only_ours"] += o and not r; d["only_reference"] += r and not o
    out.update(res)
    out["per_task"] = per_task
    out["discordant"] = [{"task_id": t, "init_idx": e, "ours": ours[(t, e)], "ours_steps": steps[(t, e)],
                          "reference_steps_of_900": ref[(t, e)].steps}
                         for (t, e) in sorted(ours) if ours[(t, e)] != ref[(t, e)].success_within(LIMIT)]
    return out


def text(r: dict) -> str:
    lines = [f"X-VLA parity (SPEC §11 step 5): base rollouts vs References/progress.jsonl at {r['limit']} steps",
             f"markers {r['markers']}, episodes {r['episodes']}, duplicates {len(r['duplicates'])}, "
             f"over {r['limit']} steps {len(r['over_limit'])}, missing {len(r['missing'])}"]
    if "n" not in r:
        lines.append(f"NO VERDICT: incomplete (first missing {r['missing'][:5]}, duplicates {r['duplicates'][:5]})")
        return "\n".join(lines) + "\n"
    if not r["complete"]:
        lines.append(f"PARTIAL: {r['n']} of 500 episodes")
    ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
    n = r["n"]
    lines += [
        f"ours {r['ours']}/{n} ({r['ours'] / n:.1%}, Wilson {ci(r['ours_ci'])}); "
        f"reference {r['reference']}/{n} ({r['reference'] / n:.1%}, Wilson {ci(r['reference_ci'])})",
        f"paired: both {r['both']}, only ours {r['only_ours']}, only reference {r['only_reference']}, "
        f"neither {r['neither']}",
        f"paired difference {r['paired_difference']:+.3f} (95% {ci(r['paired_difference_ci'])}); exact McNemar "
        f"p = {r['mcnemar_p']:.4f}: {'within' if r['within_sampling_error'] else 'NOT within'} sampling error",
        "per task (ours vs reference; discordant +only ours / -only reference):"]
    for t, d in sorted(r["per_task"].items()):
        lines.append(f"  task {t}: {d['ours']}/{d['n']} vs {d['reference']}/{d['n']}  "
                     f"(+{d['only_ours']} / -{d['only_reference']})")
    lines.append("discordant episodes (+/- = success only ours / only reference; ours steps / reference steps of 900):")
    lines.append("  " + ", ".join(f"t{d['task_id']}/i{d['init_idx']} {'+' if d['ours'] else '-'} "
                                  f"{d['ours_steps']}/{d['reference_steps_of_900']}" for d in r["discordant"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("markers", type=Path, help="directory of results/base/*.json markers, or one concatenated file")
    ap.add_argument("--references", type=Path, default=Path("References"))
    ap.add_argument("--out", help="write <out>.json and <out>.txt")
    ap.add_argument("--partial", action="store_true", help="report on the episodes so far")
    a = ap.parse_args(argv)
    r = report(read_markers(a.markers), a.references, partial=a.partial)
    t = text(r)
    if a.out:
        Path(f"{a.out}.json").write_text(json.dumps(r, indent=1, default=list) + "\n")
        Path(f"{a.out}.txt").write_text(t)
    sys.stdout.write(t)
    return 0 if "n" in r else 2


if __name__ == "__main__":
    sys.exit(main())
