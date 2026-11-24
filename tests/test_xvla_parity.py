"""ops/xvla_parity.py: the X-VLA parity report (SPEC §11 step 5) against References/progress.jsonl."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vlar_ops import reference_runs as rr

REPO = Path(__file__).resolve().parents[1]
REFS = REPO / "References"
SCRIPT = REPO / "ops" / "xvla_parity.py"
pytestmark = pytest.mark.skipif(not (REFS / "progress.jsonl").exists(), reason="References/ not present")


def write_markers(d, outcome, drop=(), dupe=None):
    d.mkdir(parents=True, exist_ok=True)
    for (t, e), (ok, steps) in outcome.items():
        if (t, e) in drop:
            continue
        m = {"job": {"phase": "base", "subject": f"base/t{t}/i{e}", "spec": {"task_id": t, "init_idx": e}},
             "result": {"summary": {"success": ok, "steps": steps}}}
        (d / f"{t}-{e}.json").write_text(json.dumps(m))
        if dupe == (t, e):
            (d / f"{t}-{e}-again.json").write_text(json.dumps(m))
    return d


def run(*args):
    p = subprocess.run([sys.executable, str(SCRIPT), *map(str, args), "--references", str(REFS)],
                       capture_output=True, text=True, cwd=REPO)
    return p.returncode, p.stdout + p.stderr


def mirror():
    ref = rr.load_xvla(REFS)
    return ref, {k: (r.success_within(520), min(r.steps, 520)) for k, r in ref.items()}


def test_identical_outcomes_match_the_reference_405(tmp_path):
    _, outcome = mirror()
    rc, out = run(write_markers(tmp_path / "m", outcome), "--out", tmp_path / "rep")
    assert rc == 0, out
    r = json.loads((tmp_path / "rep.json").read_text())
    assert (r["n"], r["ours"], r["reference"], r["only_ours"], r["only_reference"]) == (500, 405, 405, 0, 0)
    assert r["mcnemar_p"] == 1.0 and r["within_sampling_error"] and r["discordant"] == []
    assert (tmp_path / "rep.txt").read_text() == out


def test_flips_are_paired_and_success_after_520_steps_does_not_count(tmp_path):
    ref, outcome = mirror()
    wins = [k for k in sorted(ref) if ref[k].success_within(520)][:30]
    for k in wins:
        outcome[k] = (False, 520)
    late = next(k for k in sorted(ref) if not ref[k].success_within(520))
    outcome[late] = (True, 600)                          # a success past the limit is a failure at 520
    rc, out = run(write_markers(tmp_path / "m", outcome), "--out", tmp_path / "rep")
    r = json.loads((tmp_path / "rep.json").read_text())
    assert rc == 0, out
    assert (r["ours"], r["only_ours"], r["only_reference"], r["over_limit"]) == (375, 0, 30, [list(late)])
    assert r["mcnemar_p"] < 0.05 and not r["within_sampling_error"] and "NOT within" in out
    assert len(r["discordant"]) == 30


def test_missing_or_duplicated_episodes_give_no_verdict_unless_partial(tmp_path):
    _, outcome = mirror()
    rc, out = run(write_markers(tmp_path / "a", outcome, drop={(3, 7)}))
    assert rc == 2 and "NO VERDICT" in out
    rc, out = run(write_markers(tmp_path / "b", outcome, dupe=(0, 0)))
    assert rc == 2 and "NO VERDICT" in out
    rc, out = run(tmp_path / "a", "--partial")
    assert rc == 0 and "PARTIAL: 499 of 500" in out


def test_a_concatenated_file_reads_like_a_directory(tmp_path):
    _, outcome = mirror()
    d = write_markers(tmp_path / "m", outcome)
    cat = tmp_path / "all.jsonl"
    cat.write_text("".join(p.read_text() for p in sorted(d.glob("*.json"))))    # gcloud storage cat output
    assert run(cat)[1] == run(d)[1]
