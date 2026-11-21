"""The event log and clock on the production engine (effort step 4; SPEC §5, §7, §11 step 3), with the fake policies
of episode_driver.py:

- "the clock recomputed offline from an event log must equal the live clock for every profile", through network
  retries, abandoned escalations, both kinds of decision point, JSON and snapshot round trips and restores;
- π0.5 alone: retries never change the executed actions, only the clock; the 20-attempt cap ends the episode with
  end reason `network` (amendment 13);
- a live time limit cuts exactly where offline scoring of the uncut run says it would;
- branch runs reproduce for the same seed index and differ for another; labels computed from real branch logs;
- every X-VLA call records its first target's distance from the arm and its context (amendment 10).
"""

import dataclasses
import hashlib
import shutil

import pytest

np = pytest.importorskip("numpy")

from conftest import requires_lerobot
from vlar_harness import clock as clk
from vlar_harness import config as C
from vlar_harness import eventlog
from vlar_harness import labels as L
from vlar_harness import network
from vlar_harness import seeding
from vlar_harness import snapshot as snap
from vlar_harness.config import PROFILES, load_config

CFG = load_config()
CH = clk.Charges.from_config(CFG)
TASK, INIT, ROLLOUT = 0, 4, 1_000_004
SCHEDULE = {1, 2, 3, 5, 6, 7, 8, 12, 15, 16, 17, 18, 19, 22, 30, 31, 40}   # mid and chunk points, runs of escalations


def lossy(cfg, p_fail):
    return network.SimulatedNetwork(dataclasses.replace(network.NetworkParams.from_config(cfg), p_fail=p_fail))


class FailAt:
    def __init__(self, steps):
        self.fail_steps = set(steps)

    def request(self, call, *, step, rollout_seed, max_attempts):
        if step in self.fail_steps:
            return network.RemoteOutcome(False, None, [network.attempt_record(i, False, cause="dropped")
                                                       for i in range(max_attempts)])
        return network.RemoteOutcome(True, call(), [network.attempt_record(0, True, rtt_s=0.06)])


def _spec():
    from vlar_harness.sim.libero_sim import SimSpec
    return SimSpec.from_config(CFG)


@pytest.fixture(scope="module")
def sim():
    from vlar_harness.sim.libero_sim import LiberoSim
    s = LiberoSim(TASK, _spec())
    yield s
    s.close()


@pytest.fixture(scope="module")
def other_sim():
    from vlar_harness.sim.libero_sim import LiberoSim
    s = LiberoSim(TASK, _spec())
    yield s
    s.close()


@pytest.fixture(scope="module")
def cfg_with_limit(tmp_path_factory):
    """The harness config with every task's time limit set through the calibration file (the production path)."""
    def make_cfg(limit_s):
        d = tmp_path_factory.mktemp("cfg")
        shutil.copy(C.CONFIG_DIR / "harness-v1.yaml", d / "harness-v1.yaml")
        C.write_calibration({"time_limit": {"per_task_s": {t: float(limit_s) for t in range(10)}}}, config_dir=d)
        return C.load_config(config_dir=d)
    return make_cfg


def make(sim, cfg=CFG, link=None, transport=None, on_step=None):
    from episode_driver import FakePI05, FakeXVLA
    from vlar_harness.adapters.pi05 import PI05Adapter
    from vlar_harness.adapters.xvla import XVLAAdapter
    from vlar_harness.episode import Episode
    return Episode(sim, XVLAAdapter(transport or FakeXVLA()), PI05Adapter(sim), FakePI05(), cfg, remote=link,
                   on_step=on_step)


def run_routed(ep, schedule, *, until=520, profile="none", time_limit=False, rollout=ROLLOUT, on_point=None):
    dp = ep.start(INIT, rollout, "routed", profile=profile, time_limit=time_limit)
    while dp is not None and dp.step < until:
        if on_point is not None:
            on_point(ep, dp)
        dp = ep.decide("escalate" if schedule(dp.index) and dp.can_escalate else "local").next_point
    return ep


def kinds(events):
    out = {"retry_ok": 0, "abandoned": 0, "chunk": 0, "mid": 0}
    for e in events:
        if e["t"] == "pi05_request" and e["ok"] and len(e["attempts"]) > 1:
            out["retry_ok"] += 1
        elif e["t"] == "escalation_abandoned":
            out["abandoned"] += 1
        elif e["t"] == "decision_point":
            out[e["kind"]] += 1
    return out


# ---------------------------------------------------------------- live versus offline

@requires_lerobot
def test_live_clocks_equal_offline_replay_for_every_profile(sim, other_sim):
    snaps, at_capture = {}, {}

    def capture(e, dp):
        if dp.index in (5, 12, 19, 33):
            snaps[dp.index] = snap.to_bytes(e.snapshot())
            at_capture[dp.index] = e.times()

    ep = run_routed(make(sim, link=lossy(CFG, 0.6)), lambda d: d in SCHEDULE, profile="cascader_full",
                    on_point=capture)
    seen = kinds(ep.events)
    assert seen["retry_ok"] >= 1 and seen["abandoned"] >= 1 and seen["chunk"] >= 10 and seen["mid"] >= 10, seen
    assert ep.state.end_reason == "step_limit" and ep.sim.steps == 520
    live = ep.times()
    assert live == clk.replay_all(ep.events, CH)                             # bit for bit, all five profiles
    assert live == clk.replay_all(eventlog.loads(eventlog.dumps(ep.events)), CH)
    assert ep.result().sim_time_s == live["cascader_full"]
    assert len(set(live.values())) == 4                                      # router == cascader_nospread
    for p in PROFILES:
        assert live[p] == clk.score(ep.events, p, CH).time_s

    # Restore from snapshots (file format, another env): the clocks come back exactly, and continuing with the same
    # rollout seed and schedule reproduces the rest of the log and the final clocks.
    for d, blob in snaps.items():
        ep2 = make(other_sim, link=lossy(CFG, 0.6))
        dp = ep2.restore(snap.from_bytes(blob), snapshot_id=f"t/dp{d}")
        assert ep2.times() == at_capture[d]
        while dp is not None:
            dp = ep2.decide("escalate" if dp.index in SCHEDULE and dp.can_escalate else "local").next_point
        assert ep2.times() == live
        assert [e for e in ep2.events if e["t"] != "restore"] == ep.events


@requires_lerobot
def test_no_decision_overhead_once_the_budget_is_spent(sim):
    """Amendment 15: after the 26th escalation the decider is no longer consulted, so from then on every profile's
    clock grows by exactly the same charges (steps and X-VLA calls)."""
    ep = make(sim, link=network.DirectLink())
    ep.step_limit = 330
    run_routed(ep, lambda d: True)                                   # escalate whenever the budget allows
    points = [e for e in ep.events if e["t"] == "decision_point"]
    spent = [e for e in points if not e["can_escalate"]]
    assert ep.state.pi05_steps == 260 and len(spent) == 7
    assert all(e["budget_remaining"] == 0 for e in spent) and {e["kind"] for e in spent} == {"chunk", "mid"}
    assert all(e["can_escalate"] == (e["budget_remaining"] >= 10) for e in points)
    first = ep.events.index(spent[0])
    at_spent = clk.replay_all(ep.events[:first], CH)
    final = ep.times()
    assert final == clk.replay_all(ep.events, CH)
    gained = {p: clk.replay(ep.events[first:], p, CH) for p in PROFILES}   # charges after exhaustion alone
    assert len(set(gained.values())) == 1                                  # identical under every profile
    for p in PROFILES:
        assert final[p] - at_spent[p] == pytest.approx(gained["none"], abs=1e-9)
    # Escalating at every point makes every point before exhaustion a chunk point, and every mid point comes after
    # it: the router's /features and the full cascader's mid-point probe are never charged, the chunk-point
    # /sample overhead is. Under the old rule router > none and cascader_full > cascader_chunkspread.
    assert final["router"] == final["cascader_nospread"] == final["none"]
    assert final["cascader_full"] == final["cascader_chunkspread"] > final["none"]


@requires_lerobot
def test_pi05_only_retries_change_only_the_clock(sim):
    a = make(sim, link=network.DirectLink())
    a.start(INIT, ROLLOUT, "pi05_only")
    a.step_limit = 80
    ra = a.run_episode()
    b = make(sim, link=lossy(CFG, 0.6))
    b.start(INIT, ROLLOUT, "pi05_only")
    b.step_limit = 80
    rb = b.run_episode()
    assert ra.end_reason == rb.end_reason == "step_limit" and ra.pi05_requests == rb.pi05_requests == 8
    assert rb.pi05_failed_requests > 0 and ra.pi05_failed_requests == 0
    for x, y in zip(a.trajectory, b.trajectory, strict=True):        # same request, same noise: same actions
        assert np.array_equal(x["action"], y["action"]) and np.array_equal(x["robot0_eef_pos"], y["robot0_eef_pos"])
    seeds = lambda ep: [e["seed"] for e in ep.events if e["t"] == "pi05_request"]
    assert seeds(a) == seeds(b)
    assert rb.sim_time_s > ra.sim_time_s
    assert b.times() == clk.replay_all(b.events, CH)


@requires_lerobot
def test_pi05_only_gives_up_after_20_attempts_with_end_reason_network(sim):
    called = []
    ep = make(sim, link=lossy(CFG, 1.0))                     # every attempt is dropped
    ep.pi05_call = lambda raw, task, seed: called.append(seed)
    ep.start(INIT, ROLLOUT, "pi05_only")
    res = ep.run_episode()
    assert (res.end_reason, res.success, res.steps) == ("network", False, 0)
    req = [e for e in ep.events if e["t"] == "pi05_request"]
    assert len(req) == 1 and not req[0]["ok"] and len(req[0]["attempts"]) == 20
    assert (res.pi05_requests, res.pi05_failed_requests, called) == (1, 20, [])
    t = 0.0
    for i in range(20):
        t += 1.0
        if i < 19:
            t += 0.2
    assert ep.times()["none"] == t == res.sim_time_s
    assert ep.events[-1] == {"t": "end", "step": 0, "reason": "network"}


# ---------------------------------------------------------------- the live time limit

def crossing_limits(events, profile):
    """Limits that an uncut run first exceeds inside a run of steps, at a decision point's overhead, and at a π0.5
    request with a retry."""
    c = clk.Clock(CH, profile)
    seg = point = request = None
    for e in events:
        before = c.t
        if e["t"] == "segment":
            times = [c.step() for _ in range(e["steps"])]
            if seg is None and e["start"] >= 100 and e["steps"] >= 5:
                seg = ((times[2] + times[3]) / 2, e["start"] + 4)
        else:
            c.event(e)
            if point is None and e["t"] == "decision_point" and e["kind"] == "mid" and e["step"] >= 150:
                point = ((before + c.t) / 2, None)
            if request is None and e["t"] == "pi05_request" and len(e["attempts"]) > 1 and e["step"] >= 50:
                request = ((before + c.t) / 2, None)
    return {"segment": seg, "decision_point": point, "pi05_request": request}


@requires_lerobot
@pytest.mark.parametrize("where", ["segment", "decision_point", "pi05_request"])
def test_live_time_limit_cuts_where_offline_scoring_says(sim, cfg_with_limit, where):
    profile = "cascader_full"
    uncut = run_routed(make(sim, link=lossy(CFG, 0.3)), lambda d: d in SCHEDULE, profile=profile)
    limit, step = crossing_limits(uncut.events, profile)[where]
    cr = clk.first_crossing(uncut.events, profile, CH, limit)
    assert cr.step == step and uncut.events[cr.event_index]["t"] == where

    cfg = cfg_with_limit(limit)
    cut = run_routed(make(sim, cfg=cfg, link=lossy(cfg, 0.3)), lambda d: d in SCHEDULE, profile=profile,
                     time_limit=True)
    assert cut.state.end_reason == "time_limit" and cut.events[-1]["reason"] == "time_limit"
    assert cut.state.time_limit_s == limit
    assert cut.times()[profile] == cr.time_s                           # the same float as offline
    n = cr.event_index
    assert cut.events[1:n] == uncut.events[1:n]                        # identical up to the crossing
    if step is None:
        assert cut.events[n] == uncut.events[n] and len(cut.events) == n + 2
    else:
        assert cut.events[n] == dict(uncut.events[n], steps=step - uncut.events[n]["start"])
        assert cut.sim.steps == step
    assert not clk.score(uncut.events, profile, CH).within(limit)
    assert cut.times() == clk.replay_all(cut.events, CH)


# ---------------------------------------------------------------- branches and labels

class RandomSuccess:
    """Stand-in for LIBERO's success check, so fake-policy runs succeed at trajectory-dependent steps: after step 70,
    a step succeeds when a hash of the arm's position falls in 1/150 of its range. It reads only the current
    observation, which restore reproduces exactly."""

    def __init__(self, sim):
        self.sim, self.real = sim, sim.step

    def __enter__(self):
        def step(action):
            done = self.real(action)
            pos = np.ascontiguousarray(self.sim.obs["robot0_eef_pos"])
            lucky = int.from_bytes(hashlib.sha256(pos.tobytes()).digest()[:4], "little") % 150 == 0
            return done or (self.sim.steps > 70 and lucky)
        self.sim.step = step
        return self

    def __exit__(self, *exc):
        del self.sim.step


@requires_lerobot
def test_branches_reproduce_and_differ_by_seed_index(sim, other_sim):
    snaps = {}
    base = make(sim)
    base.step_limit = 150
    run_routed(base, lambda d: d == 2, until=60, on_point=lambda e, dp: snaps.__setitem__(dp.index, e.snapshot()))
    s4 = snap.from_bytes(snap.to_bytes(snaps[4]))                     # step 40, mid point, 20 queued

    def branch(s, name, calls, seed, where=sim):
        ep = make(where)
        ep.step_limit = 150
        return ep.run_branch(s, branch=name, pi05_calls=calls, seed_index=seed, snapshot_id="t/dp04")

    a, b = branch(s4, "b2_one_call", 1, 1), branch(s4, "b2_one_call", 1, 1, other_sim)
    assert a.events == b.events and a.rollout_seed == b.rollout_seed
    for x, y in zip(a.trajectory, b.trajectory, strict=True):
        assert np.array_equal(x["action"], y["action"]) and np.array_equal(x["robot0_eef_pos"], y["robot0_eef_pos"])
    assert a.events[0]["t"] == "restore" and a.events[1]["t"] == "branch"
    assert a.rollout_seed == seeding.branch_rollout_seed(ROLLOUT, 4, "b2_one_call", 1) != ROLLOUT
    assert a.result.pi05_requests == s4.episode["pi05_requests"] + 1                # exactly one more request

    c = branch(s4, "b2_one_call", 1, 2)
    assert not np.array_equal(a.trajectory[0]["action"], c.trajectory[0]["action"])  # a new π0.5 noise seed

    # Continue: the stored queue (steps 41-60) is executed identically in every seed; later calls differ.
    k1, k2 = branch(s4, "b1_continue", 0, 1), branch(s4, "b1_continue", 0, 2)
    same = [np.array_equal(x["action"], y["action"]) for x, y in zip(k1.trajectory, k2.trajectory)]
    assert all(same[:20]) and not all(same[20:])

    with pytest.raises(ValueError, match="do not fit"):
        branch(s4, "b4_max", 12, 1)                                    # only 11 decision points left before 150


@requires_lerobot
def test_labels_from_real_branch_runs(sim):
    from vlar_ops import plan

    seeds, limit_steps = 2, 150
    with RandomSuccess(sim):
        snaps = {}
        base = make(sim)
        base.step_limit = limit_steps
        run_routed(base, lambda d: d == 2, on_point=lambda e, dp: snaps.__setitem__(dp.index, snap.to_bytes(e.snapshot())))
        s6 = snap.from_bytes(snaps[6])
        prefix = s6.event_log
        point = prefix[-1]
        ref = plan.DecisionPointRef("t/dp06", "base/t0/i4", "base", TASK, INIT, 6, point["budget_remaining"])
        _, entries = plan.branch_plan("test", [ref], branches=CFG["branches"], step_limit=limit_steps, interval=10,
                                      seeds=seeds, enabled=list(C.BRANCHES), tier_one_seeds=2, order_seed=1)
        entry = entries["t/dp06"]
        own, results = {}, {}
        for b, canon in entry["shares"].items():
            for s in range(1, seeds + 1):
                if s in entry["parent_seeds"].get(canon, []) or (canon, s) in own:
                    continue
                ep = make(sim)
                ep.step_limit = limit_steps
                run = ep.run_branch(s6, branch=canon, pi05_calls=entry["calls"][canon], seed_index=s,
                                    snapshot_id="t/dp06")
                own[(canon, s)] = eventlog.loads(eventlog.dumps(run.events))
                results[(canon, s)] = run.result
    assert entry["calls"] == {"b1_continue": 0, "b2_one_call": 1, "b3_three_calls": 3, "b4_max": 9}
    parent = eventlog.loads(eventlog.dumps(base.events))
    outcomes = [r.success for r in results.values()] + [base.state.end_reason == "success"]
    assert any(outcomes) and not all(outcomes), outcomes

    free = L.point_labels(prefix, entry, own, seeds=seeds, charges=CH, limit_s=1e9, parent_log=parent)
    for b, canon in entry["shares"].items():
        wins = [base.state.end_reason == "success" if s in entry["parent_seeds"].get(canon, [])
                else results[(canon, s)].success for s in range(1, seeds + 1)]
        assert all(free[b][p].value == sum(wins) / seeds for p in PROFILES)

    times = sorted(clk.score(x, "router", CH).time_s for x in [parent] + [prefix + log for log in own.values()]
                   if x[-1]["reason"] == "success")
    limit = times[len(times) // 2]
    lab = L.point_labels(prefix, entry, own, seeds=seeds, charges=CH, limit_s=limit, parent_log=parent)
    for b, canon in entry["shares"].items():
        for p in PROFILES:
            logs = [parent if s in entry["parent_seeds"].get(canon, []) else prefix + own[(canon, s)]
                    for s in range(1, seeds + 1)]
            assert lab[b][p].value == sum(clk.score(x, p, CH).within(limit) for x in logs) / seeds
            assert lab[b][p].value <= free[b][p].value
        values = [lab[b][p].value for p in ("none", "router", "cascader_chunkspread", "cascader_full")]
        assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------- handoff diagnostics (amendment 10)

@requires_lerobot
def test_xvla_calls_record_first_target_distance_and_context(sim):
    from episode_driver import FakeXVLA

    measured, goals = [], {}
    fake = FakeXVLA()

    def transport(payload, seed):
        measured.append(np.array(sim.controller_pose()[0]))
        return fake(payload, seed)

    def on_step(ep, action, driver):
        goals[ep.sim.steps] = (driver, np.array(ep.sim.env.robots[0].controller.goal_pos))

    ep = make(sim, link=FailAt({120}), transport=transport, on_step=on_step)
    run_routed(ep, lambda d: d in {2, 3, 7, 12}, until=200)
    acts = [e for e in ep.events if e["t"] == "xvla_act"]
    assert len(acts) == len(measured)
    contexts = [(e["step"], e["call"], e["context"]) for e in acts]
    assert contexts[0] == (0, "act", "start")
    assert (40, "act", "after_pi05") in contexts and (80, "act", "after_pi05") in contexts
    assert (120, "act_fresh", "after_abandon") in contexts
    assert {c for _, _, c in contexts} == {"start", "stretch", "after_pi05", "after_abandon"}
    checked = 0
    for i, e in enumerate(acts):
        last_at_step = all(x["step"] != e["step"] for x in acts[i + 1:])
        nxt = goals.get(e["step"] + 1)
        if last_at_step and nxt is not None and nxt[0] == "xvla":
            assert e["first_target_dist_m"] == float(np.linalg.norm(nxt[1] - measured[i]))
            checked += 1
    assert checked >= 5
    stretch = eventlog.first_target_distances([ep.events], "stretch")
    assert stretch == [e["first_target_dist_m"] for e in acts if e["context"] == "stretch"] and len(stretch) >= 2
