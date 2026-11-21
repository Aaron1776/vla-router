import json
import shutil
from pathlib import Path

import pytest

from vlar_harness import seeding
from vlar_harness.config import load_config
from vlar_ops import reference_runs as R

REAL = Path(__file__).resolve().parents[1] / "References"


@pytest.mark.skipif(not (REAL / "progress.jsonl").exists(), reason="References/ not present")
def test_real_reference_files_match_the_confirmed_totals():
    report = R.check(REAL)
    assert report["totals"] == R.EXPECTED
    assert report["pi05_parity_subset"] == 97
    assert report["per_task"][8] == {"pi05": 41, "xvla_520": 43}


def fake_reference(root: Path, pi05_fail=((0, 0),), xvla_fail=(), late=()):
    """A synthetic reference tree; `late` = X-VLA successes after step 520."""
    for t in range(10):
        succ = [(t, e) not in pi05_fail for e in range(50)]
        d = root / f"task_{t}"
        d.mkdir(parents=True)
        (d / "eval_info.json").write_text(json.dumps({
            "per_task": [{"task_group": "libero_10", "task_id": t, "metrics": {"successes": succ}}],
            "overall": {"n_episodes": 50, "pc_success": 100.0 * sum(succ) / 50}}))
    with open(root / "progress.jsonl", "w") as f:
        for t in range(10):
            for e in range(50):
                ok = (t, e) not in xvla_fail
                steps = 600 if (t, e) in late else (250 if ok else 900)
                f.write(json.dumps({"task_id": t, "episode": e, "success": ok, "steps": steps, "step_limit": 900,
                                    "task": "x", "seconds": 1.0}) + "\n")


def test_loaders_map_episodes_to_initial_states_and_scene_seeds(tmp_path):
    fake_reference(tmp_path)
    pi05, xvla = R.load_pi05(tmp_path), R.load_xvla(tmp_path)
    assert pi05[(3, 7)].scene_seed == 1007 and xvla[(3, 7)].scene_seed == 149
    assert pi05[(0, 0)].success is False
    cfg_project = load_config()
    cfg_parity = load_config(scene_seed="pi05_parity")
    assert seeding.scene_seed(7, cfg_project.option("scene_seed")) == xvla[(3, 7)].scene_seed
    assert seeding.scene_seed(7, cfg_parity.option("scene_seed")) == pi05[(3, 7)].scene_seed


def test_wrong_totals_are_rejected(tmp_path):
    fake_reference(tmp_path)
    with pytest.raises(R.ReferenceDataError, match="totals"):
        R.check(tmp_path, verify_checksums=False)


def test_manifest_pins_all_eleven_files():
    pins = R.pinned_checksums()
    assert sorted(pins) == ["progress.jsonl"] + [f"task_{t}/eval_info.json" for t in range(10)]
    assert all(len(d) == 64 for d in pins.values())


@pytest.mark.skipif(not (REAL / "progress.jsonl").exists(), reason="References/ not present")
def test_a_changed_byte_is_caught(tmp_path):
    copy = tmp_path / "References"
    shutil.copytree(REAL, copy, ignore=shutil.ignore_patterns("*.mp4", ".DS_Store"))
    R.verify_files(copy)
    f = copy / "task_3" / "eval_info.json"
    data = bytearray(f.read_bytes())
    data[-2] = ord(" ") if data[-2] != ord(" ") else ord("\t")
    f.write_bytes(bytes(data))
    with pytest.raises(R.ReferenceDataError, match="task_3/eval_info.json"):
        R.check(copy)


@pytest.mark.parametrize("damage", ["duplicate", "missing", "short_failure", "overall"])
def test_damaged_files_are_rejected(tmp_path, damage):
    fake_reference(tmp_path)
    p = tmp_path / "progress.jsonl"
    lines = p.read_text().splitlines()
    if damage == "duplicate":
        p.write_text("\n".join(lines + [lines[0]]) + "\n")
    elif damage == "missing":
        p.write_text("\n".join(lines[:-1]) + "\n")
    elif damage == "short_failure":
        row = json.loads(lines[5])
        row.update(success=False, steps=400)
        lines[5] = json.dumps(row)
        p.write_text("\n".join(lines) + "\n")
    elif damage == "overall":
        f = tmp_path / "task_2" / "eval_info.json"
        info = json.loads(f.read_text())
        info["overall"]["pc_success"] = 50.0
        f.write_text(json.dumps(info))
    loader = R.load_pi05 if damage == "overall" else R.load_xvla
    with pytest.raises(R.ReferenceDataError):
        loader(tmp_path)


def test_success_within_scores_xvla_at_520(tmp_path):
    fake_reference(tmp_path, late=((1, 1),))
    xvla = R.load_xvla(tmp_path)
    assert xvla[(1, 1)].success_within(900) and not xvla[(1, 1)].success_within(520)
    with pytest.raises(R.ReferenceDataError):
        R.load_pi05(tmp_path)[(0, 1)].success_within(400)


def test_compare_pairs_episodes_and_uses_mcnemar(tmp_path):
    fake_reference(tmp_path, pi05_fail=())
    ref = R.load_pi05(tmp_path)
    keys = [(t, e) for t in range(10) for e in range(10)]
    ours = {k: True for k in keys}
    for k in keys[:5]:
        ours[k] = False                                    # 5 discordant pairs: p = 2/32
    r = R.compare(ours, ref, keys=keys)
    assert (r["n"], r["ours"], r["reference"], r["only_reference"]) == (100, 95, 100, 5)
    assert r["mcnemar_p"] == pytest.approx(0.0625) and r["within_sampling_error"]
    ours[keys[5]] = False                                  # 6 discordant: p = 2/64
    r = R.compare(ours, ref, keys=keys)
    assert r["mcnemar_p"] == pytest.approx(0.03125) and not r["within_sampling_error"]
    assert r["paired_difference"] == pytest.approx(-0.06)
    lo, hi = r["paired_difference_ci"]
    assert lo < -0.06 < hi < 0


# ---------------------------------------------------------------- seeding

def test_seeds_are_deterministic_independent_and_in_range():
    a = seeding.call_seed(1_000_123, 40, "xvla", "act")
    assert a == seeding.call_seed(1_000_123, 40, "xvla", "act")
    others = {seeding.call_seed(1_000_123, 40, "xvla", "sample", 0), seeding.call_seed(1_000_123, 50, "xvla", "act"),
              seeding.call_seed(1_000_123, 40, "pi05", "act"), seeding.call_seed(2_000_123, 40, "xvla", "act"),
              seeding.stream_seed(1_000_123, "network"), seeding.stream_seed(1_000_123, "explore")}
    assert a not in others and len(others) == 6
    assert all(0 <= s < 2 ** 63 for s in others | {a})
    with pytest.raises(ValueError):
        seeding.call_seed(1, 0, "xvla", "features")


def test_branch_seeds_never_equal_the_parent():
    parent = 1_000_301
    seeds = {seeding.branch_rollout_seed(parent, d, b, s) for d in range(52) for b in ("b1_continue", "b2_one_call")
             for s in (1, 2, 3)}
    assert parent not in seeds and len(seeds) == 52 * 2 * 3
