"""Labels (SPEC §7) and time-limit calibration (SPEC §5, §11 step 6) on synthetic event logs. Pure Python; the
same code runs on real branch logs in tests/reference/test_clock_live.py."""

import math
import random
import shutil

import pytest

from vlar_harness import calibration as cal
from vlar_harness import clock as clk
from vlar_harness import config as C
from vlar_harness import labels as L
from vlar_harness import network as net
from vlar_harness.config import PROFILES, load_config
from vlar_ops import plan

CFG = load_config()
CH = clk.Charges.from_config(CFG)
ORDER = ("none", "router", "cascader_chunkspread", "cascader_full")     # non-decreasing overhead


class Log:
    """Builds event logs with decision points every 10 steps, as the engine writes them."""

    def __init__(self, events=None, *, task=0, mode="routed", time_limit_s=None):
        self.ev = list(events) if events is not None else [
            {"t": "start", "task_id": task, "init_idx": 0, "scene_seed": 142, "rollout_seed": 1, "mode": mode,
             "profile": "none", "time_limit_s": time_limit_s}]
        self.queue = 0

    def point(self, step, budget=260):
        kind = "mid" if self.queue else "chunk"
        if kind == "chunk":
            self.ev.append({"t": "xvla_act", "step": step, "call": "act", "context": "stretch",
                            "first_target_dist_m": 0.01})
            self.queue = 30
        self.ev.append({"t": "decision_point", "step": step, "index": step // 10, "kind": kind, "queue": self.queue,
                        "budget_remaining": budget, "can_escalate": budget >= 10})
        return self

    def local(self, step, n=10):
        self.ev.append({"t": "decision", "step": step, "choice": "local"})
        self.ev.append({"t": "segment", "driver": "xvla", "start": step, "steps": n})
        self.queue -= 10
        return self

    def escalate(self, step, attempts=(True,), n=10):
        self.ev.append({"t": "decision", "step": step, "choice": "escalate"})
        att = [net.attempt_record(i, ok, cause=None if ok else "dropped", u_fail=0.5, rtt_s=0.06 + 0.001 * i)
               for i, ok in enumerate(attempts)]
        self.ev.append({"t": "pi05_request", "step": step, "ok": att[-1]["ok"], "attempts": att})
        self.ev.append({"t": "segment", "driver": "pi05", "start": step, "steps": n})
        self.queue = 0
        return self

    def end(self, step, reason):
        self.ev.append({"t": "end", "step": step, "reason": reason})
        return self.ev


def run_to(log, step, success_at, limit=120):
    """Continue locally from `step` (a decision point already logged) until success or the step limit."""
    while True:
        n = 10 if success_at is None else min(10, success_at - step)
        log.local(step, n)
        step += n
        if success_at is not None and step == success_at:
            return log.end(step, "success")
        if step >= limit:
            return log.end(step, "step_limit")
        log.point(step)


def prefix_at(d=2, budget=20):
    log = Log()
    for k in range(d):
        log.point(10 * k, budget).local(10 * k)
    return log.point(10 * d, budget).ev


def branch(prefix, name, seed, calls, *, escalations, success_at, attempts=(True,)):
    log = Log(prefix)
    log.queue = 10                                            # point 2 is a mid point (queue of 10 left)
    log.ev.append({"t": "restore", "step": 20, "index": 2, "snapshot_id": "ep/dp02"})
    log.ev.append({"t": "branch", "branch": name, "seed_index": seed, "pi05_calls": calls, "index": 2,
                   "rollout_seed": 100 + seed, "parent_rollout_seed": 1})
    step = 20
    for k in range(escalations):
        log.escalate(step, attempts)
        step += 10
        log.point(step)
    full = run_to(log, step, success_at)
    return full[len(prefix):]


def setup_point():
    prefix = prefix_at()
    ref = plan.DecisionPointRef("ep/dp02", "ep", "base", 0, 0, 2, 20)
    _, entries = plan.branch_plan("labels", [ref], branches=CFG["branches"], step_limit=120, interval=10, seeds=3,
                                  enabled=list(C.BRANCHES), tier_one_seeds=2, order_seed=1)
    entry = entries["ep/dp02"]
    parent = Log(prefix)
    parent.queue = 10
    parent = run_to(parent, 20, success_at=45)
    own = {("b1_continue", 2): branch(prefix, "b1_continue", 2, 0, escalations=0, success_at=80),
           ("b1_continue", 3): branch(prefix, "b1_continue", 3, 0, escalations=0, success_at=None),
           ("b2_one_call", 1): branch(prefix, "b2_one_call", 1, 1, escalations=1, success_at=40),
           ("b2_one_call", 2): branch(prefix, "b2_one_call", 2, 1, escalations=1, success_at=50, attempts=(False, True)),
           ("b2_one_call", 3): branch(prefix, "b2_one_call", 3, 1, escalations=1, success_at=None),
           ("b3_three_calls", 1): branch(prefix, "b3_three_calls", 1, 2, escalations=2, success_at=50),
           ("b3_three_calls", 2): branch(prefix, "b3_three_calls", 2, 2, escalations=2, success_at=None),
           ("b3_three_calls", 3): branch(prefix, "b3_three_calls", 3, 2, escalations=2, success_at=None)}
    return prefix, entry, parent, own


# ---------------------------------------------------------------- labels

def test_planner_entry_shares_b3_and_b4_when_the_budget_allows_two_calls():
    _, entry, _, _ = setup_point()
    assert entry["calls"] == {"b1_continue": 0, "b2_one_call": 1, "b3_three_calls": 2, "b4_max": 2}
    assert entry["shares"]["b4_max"] == "b3_three_calls"
    assert entry["parent_seeds"] == {"b1_continue": [1]}


def test_labels_without_a_time_limit_are_success_fractions():
    prefix, entry, parent, own = setup_point()
    lab = L.point_labels(prefix, entry, own, seeds=3, charges=CH, limit_s=1e9, parent_log=parent)
    for p in PROFILES:
        assert (lab["b1_continue"][p].successes, lab["b2_one_call"][p].successes) == (2, 2)
        assert lab["b3_three_calls"][p] == lab["b4_max"][p] == L.Label(1 / 3, 1, 3)
    zero = L.point_labels(prefix, entry, own, seeds=3, charges=CH, limit_s=0.0, parent_log=parent)
    assert all(lab.value == 0 for b in zero.values() for lab in b.values())


def test_labels_follow_each_profiles_clock():
    prefix, entry, parent, own = setup_point()
    full = {("b1_continue", 1): parent}
    full.update({k: prefix + v for k, v in own.items()})
    times = {k: {p: clk.score(log, p, CH).time_s for p in PROFILES} for k, log in full.items()}
    # A limit between b1 seed 2's time under `none` and under `cascader_full` separates the profiles.
    limit = times[("b1_continue", 2)]["none"]
    assert times[("b1_continue", 2)]["cascader_full"] > limit
    lab = L.point_labels(prefix, entry, own, seeds=3, charges=CH, limit_s=limit, parent_log=parent)
    for b, canon in entry["shares"].items():
        for p in PROFILES:
            expect = sum(clk.score(full[(canon, s)], p, CH).within(limit) for s in (1, 2, 3)) / 3
            assert lab[b][p].value == expect
        values = [lab[b][p].value for p in ORDER]
        assert values == sorted(values, reverse=True)       # more overhead never helps
    assert lab["b1_continue"]["none"].value > lab["b1_continue"]["cascader_full"].value


def test_labels_refuse_logs_that_do_not_fit():
    prefix, entry, parent, own = setup_point()
    kw = dict(seeds=3, charges=CH, limit_s=1e9)
    with pytest.raises(L.LabelError, match="parent_log"):
        L.point_labels(prefix, entry, own, **kw)
    with pytest.raises(L.LabelError, match="missing"):
        L.point_labels(prefix, entry, {k: v for k, v in own.items() if k != ("b2_one_call", 3)},
                       parent_log=parent, **kw)
    swapped = dict(own)
    swapped[("b2_one_call", 3)] = own[("b2_one_call", 2)]
    with pytest.raises(L.LabelError, match="seed 3"):
        L.point_labels(prefix, entry, swapped, parent_log=parent, **kw)
    other = Log(prefix_at(d=3)).ev
    with pytest.raises(L.LabelError, match="planner entry"):
        L.point_labels(other, entry, own, parent_log=parent, **kw)
    bad_parent = list(parent)
    bad_parent[3] = dict(bad_parent[3], step=999)
    with pytest.raises(L.LabelError, match="does not start"):
        L.point_labels(prefix, entry, own, parent_log=bad_parent, **kw)
    with pytest.raises(L.LabelError, match="ends with its decision point"):
        L.point_labels(prefix[:-1], entry, own, parent_log=parent, **kw)


# ---------------------------------------------------------------- calibration

def test_percentile_matches_numpys_linear_method_by_hand():
    assert cal.percentile_linear([4, 1, 3, 2], 90) == 3.7
    assert cal.percentile_linear([5.0], 90) == 5.0
    assert cal.percentile_linear([1, 2], 50) == 1.5


def test_percentile_equals_numpy_bit_for_bit():
    np = pytest.importorskip("numpy")
    rng = random.Random(3)
    for n in list(range(1, 40)) + [100, 247, 405]:
        xs = [rng.uniform(5, 40) for _ in range(n)]
        for q in (90, 50, 10, 99):
            assert cal.percentile_linear(xs, q) == float(np.percentile(np.array(xs), q))


def xvla_episode(task, success_at, limit=520):
    log = Log(task=task)
    step = 0
    log.point(step)
    return run_to(log, step, success_at, limit=limit)


def pi05_episode(task, success_at, limit=520, retry_every=7):
    log = Log(task=task, mode="pi05_only")
    log.ev[0]["scene_seed"] = 1000
    step, k = 0, 0
    while True:
        attempts = (False, True) if k % retry_every == 3 else (True,)
        att = [net.attempt_record(i, ok, cause=None if ok else "timeout", u_fail=0.5, rtt_s=0.05 + 0.002 * (k % 5))
               for i, ok in enumerate(attempts)]
        log.ev.append({"t": "pi05_request", "step": step, "ok": True, "attempts": att})
        n = 10 if success_at is None else min(10, success_at - step)
        log.ev.append({"t": "segment", "driver": "pi05", "start": step, "steps": n})
        step, k = step + n, k + 1
        if success_at is not None and step == success_at:
            return log.end(step, "success")
        if step >= limit:
            return log.end(step, "step_limit")


def synthetic_runs(pi05_steps=(150,) * 7 + (200, 200, 400)):
    base = [xvla_episode(t, s) for t in range(10) for s in [100 + 20 * i for i in range(10)] + [None, None]]
    parity = [pi05_episode(t, s) for t in range(10) for s in pi05_steps] + [pi05_episode(0, None)]
    return base, parity


def test_calibration_picks_the_smallest_margin_with_80_percent():
    base, parity = synthetic_runs()
    tl = cal.calibrate(base, parity, CFG)
    xv = {t: sorted(clk.score(e, "none", CH).time_s for e in base if e[0]["task_id"] == t and e[-1]["reason"] == "success")
          for t in range(10)}
    for t in range(10):
        h = (len(xv[t]) - 1) * 0.9
        lo = math.floor(h)
        assert tl.p90_s[t] == pytest.approx(xv[t][lo] + (h - lo) * (xv[t][lo + 1] - xv[t][lo]), rel=1e-12)
    pi = [(e[0]["task_id"], clk.score(e, "none", CH).time_s) for e in parity if e[-1]["reason"] == "success"]
    shares = {m: sum(x <= tl.p90_s[t] * m for t, x in pi) / len(pi) for m in (1.25, 1.5, 1.75, 2.0)}
    assert shares[1.25] < 0.8 <= shares[1.5], shares            # the data discriminate between candidates
    assert tl.margin == 1.5 and tl.share_by_margin == shares
    assert tl.per_task_s == {t: tl.p90_s[t] * 1.5 for t in range(10)}
    row = tl.rows[4]
    assert (row["xvla_successes"], row["xvla_episodes"], row["pi05_successes"], row["pi05_fit"]) == (10, 12, 10, 9)
    assert "margin m = 1.5" in cal.report(tl)


def test_calibration_errors_when_no_margin_reaches_the_share():
    base, parity = synthetic_runs(pi05_steps=(500,) * 10)
    with pytest.raises(cal.CalibrationError, match="no margin candidate"):
        cal.calibrate(base, parity, CFG)


def test_calibration_refuses_the_wrong_runs():
    base, parity = synthetic_runs()
    escalated = Log(task=1)
    escalated.point(0).escalate(0).point(10)
    escalated = run_to(escalated, 10, 60)
    with pytest.raises(cal.CalibrationError, match="X-VLA alone"):
        cal.calibrate(base + [escalated], parity, CFG)
    cut = xvla_episode(2, 120)
    cut[0] = dict(cut[0], time_limit_s=30.0)
    with pytest.raises(cal.CalibrationError, match="uncut"):
        cal.calibrate(base + [cut], parity, CFG)
    with pytest.raises(cal.CalibrationError, match="tasks \\[3\\]"):
        cal.calibrate([e for e in base if e[0]["task_id"] != 3], parity, CFG)
    with pytest.raises(cal.CalibrationError, match="modes"):
        cal.calibrate(base, base, CFG)


@pytest.fixture
def cfg_dir(tmp_path):
    d = tmp_path / "configs"
    shutil.copytree(C.CONFIG_DIR, d)
    for f in d.glob("*.calibration.yaml"):
        f.unlink()
    return d


def test_calibration_file_round_trip(cfg_dir):
    base, parity = synthetic_runs()
    tl = cal.calibrate(base, parity, CFG)
    cal.write(tl, config_dir=cfg_dir)
    cfg = C.load_config(config_dir=cfg_dir)
    assert {t: cfg.time_limit_s(t) for t in range(10)} == tl.per_task_s        # exact floats
    assert cfg["time_limit"]["margin"] == 1.5 and cfg["time_limit"]["status"] == "calibrated"
    assert cfg.hash != CFG.hash


def test_upload_size_is_written_into_every_network_choice(cfg_dir):
    C.write_upload_size(19876, config_dir=cfg_dir, note="test")
    cfg = C.load_config(config_dir=cfg_dir)
    assert cfg.option("network")["upload_image_bytes"] == 19876
    assert cfg.option("network")["upload_image_bytes_status"] == "measured"
    assert net.NetworkParams.from_config(cfg).upload_bytes == 19876 + 1024
    with pytest.raises(C.ConfigError, match="not calibration paths"):
        C.write_calibration({"options": {"network": {"choices": {"default": {"p_fail": 0.5}}}}}, config_dir=cfg_dir)


def test_frozen_config_refuses_a_placeholder_upload_size(cfg_dir):
    import yaml
    path = cfg_dir / "harness-v1.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["frozen"] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(C.ConfigError, match="placeholder upload size"):
        C.load_config(config_dir=cfg_dir)
