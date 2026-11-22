"""Decider inputs and measurement recording (SPEC §6, §7; §18 item 3) with fake models in the real simulator.

One routed schedule (task 0): chunk and mid points, escalations, an abandoned escalation (every attempt fails at
step 90, so X-VLA is called fresh), and enough escalations to spend the π0.5 budget. It runs without measurements,
with `record="full"`, and as live deciders of three profiles. Recording and live measurement must change nothing
that executes; each profile must measure only its own inputs; the records must be consistent with what the
models returned and what then executed; snapshots must carry them; branches must measure nothing.
"""

import hashlib

import pytest

np = pytest.importorskip("numpy")

from conftest import requires_lerobot
from episode_driver import FailAt, FakeXVLAService
from vlar_harness import measure as M
from vlar_harness.config import PROFILES, load_config

CFG = load_config()
TASK, INIT, ROLLOUT = 0, 4, 2_000_007
ESCALATE = {3, 4, 5, 9} | set(range(20, 43))          # 26 executed (9 is abandoned) -> budget spent after 42
ABANDON_STEP = 90
THRESHOLDS = (0.01, 0.07)                              # stand-ins until Phase 0 calibrates them
K = int(CFG["measurement"]["K"])


@pytest.fixture(scope="module")
def sim():
    from vlar_harness.sim.libero_sim import LiberoSim, SimSpec
    s = LiberoSim(TASK, SimSpec.from_config(CFG))
    yield s
    s.close()


@pytest.fixture(scope="module")
def other_sim():
    from vlar_harness.sim.libero_sim import LiberoSim, SimSpec
    s = LiberoSim(TASK, SimSpec.from_config(CFG))
    yield s
    s.close()


def make(sim, svc=None, measure=True):
    from episode_driver import FakePI05
    from vlar_harness.adapters.pi05 import PI05Adapter
    from vlar_harness.adapters.xvla import XVLAAdapter
    from vlar_harness.episode import Episode
    svc = svc or FakeXVLAService()
    ep = Episode(sim, XVLAAdapter(svc.act), PI05Adapter(sim), FakePI05(), CFG, remote=FailAt({ABANDON_STEP}),
                 measure_client=svc if measure else None)
    ep.gripper_thresholds = THRESHOLDS
    return ep, svc


def run(ep, *, profile="none", record="off", on_point=None):
    dp = ep.start(INIT, ROLLOUT, "routed", profile=profile, record=record)
    while dp is not None:
        if on_point is not None:
            on_point(ep, dp)
        dp = ep.decide("escalate" if dp.index in ESCALATE and dp.can_escalate else "local").next_point
    return ep


def trace(ep):
    return [(r["step"], r["driver"], r["action"].tobytes(), r["robot0_eef_pos"].tobytes()) for r in ep.trajectory]


def inputs_equal(a, b):
    import dataclasses
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)

    def eq(x, y):
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return np.array_equal(np.asarray(x), np.asarray(y))
        return x == y
    return eq(da, db)


@pytest.fixture(scope="module")
def runs(sim):
    out = {"plain": run(make(sim, measure=False)[0])}
    observed = {}

    def look(ep, dp):
        if dp.can_escalate:
            observed[dp.index] = {p: ep.observe(p) for p in PROFILES if p != "none"}
        if dp.index in (11, 22):
            observed.setdefault("snaps", {})[dp.index] = ep.snapshot()

    ep, svc = make(sim)
    out["full"] = run(ep, record="full", on_point=look)
    out["full_svc"], out["observed"] = svc, observed
    for profile in ("router", "cascader_nospread", "cascader_chunkspread", "cascader_full"):
        ep, svc = make(sim)
        live = {}
        run(ep, profile=profile, on_point=lambda e, dp: live.__setitem__(dp.index, e.observe()) if dp.can_escalate
            else None)
        out[profile] = (ep, svc, live)
    return out


# ---------------------------------------------------------------- adapter

@requires_lerobot
def test_peek_and_executed_rows_are_what_pop_executes(sim):
    from vlar_harness.adapters.xvla import XVLAAdapter
    svc = FakeXVLAService()
    xa = XVLAAdapter(svc.act)
    sim.setup_episode(0, env_seed=142)
    xa.reset()
    xa.query(sim, sim.task_description, seed=5)
    raw = xa.client.last_response["action"]
    peeked, rows = xa.peek(10), xa.executed_rows(raw)
    popped = np.stack([xa.pop() for _ in range(30)])
    assert peeked.dtype == rows.dtype == popped.dtype == np.float32
    assert np.array_equal(peeked, popped[:10]) and np.array_equal(rows, popped)


# ---------------------------------------------------------------- recording changes nothing

@requires_lerobot
def test_the_schedule_covers_what_measurement_must_handle(runs):
    ev = runs["full"].events
    assert any(e["t"] == "escalation_abandoned" for e in ev)
    assert any(e["t"] == "xvla_act" and e["call"] == "act_fresh" for e in ev)
    pts = [e for e in ev if e["t"] == "decision_point"]
    assert {e["kind"] for e in pts} == {"chunk", "mid"} and not pts[-1]["can_escalate"]
    assert runs["full"].sim.steps == 520


@requires_lerobot
@pytest.mark.parametrize("which", ["full", "router", "cascader_nospread", "cascader_chunkspread", "cascader_full"])
def test_measuring_changes_nothing_that_executes(runs, which):
    ep = runs[which] if which == "full" else runs[which][0]
    plain = runs["plain"]
    assert trace(ep) == trace(plain)
    strip = lambda evs: [{k: v for k, v in e.items() if k not in ("record", "profile")} for e in evs]
    assert strip(ep.events) == strip(plain.events)
    assert ep.result().success == plain.result().success


# ---------------------------------------------------------------- what each run measured

@requires_lerobot
def test_record_full_measures_every_point_where_a_decision_is_offered(runs):
    ep = runs["full"]
    offered = [e["index"] for e in ep.events if e["t"] == "decision_point" and e["can_escalate"]]
    assert [m["index"] for m in ep.measurements] == offered
    for m in ep.measurements:
        assert m["features_main"].shape == m["features_wrist"].shape == (1024,)
        assert m["proposal"].shape == (10, 7) and m["robot_state"].shape == (11,)
        assert m["chunk_spread"] is not None and m["fresh_spread"] is not None and m["drift"] is not None
        if m["kind"] == "chunk":
            assert m["chunk_pos"] == 0 and m["chunk_samples_raw"].shape == (K, 30, 20) and m["probe_raw"] is None
        else:
            assert m["chunk_pos"] in (10, 20) and m["probe_raw"].shape == (K, 30, 20)


@requires_lerobot
def test_recorded_values_match_what_the_models_returned_and_what_executed(runs):
    from vlar_harness.adapters.xvla import XVLAAdapter
    ep = runs["full"]
    xa = XVLAAdapter(lambda p, s: None)
    by_step = {}
    for r in ep.trajectory:
        by_step.setdefault(r["step"], r)
    chunk = None
    for m in ep.measurements:
        if m["chunk_samples_raw"] is not None:
            chunk = np.stack([xa.executed_rows(c) for c in m["chunk_samples_raw"]])
            # The first sample is the chunk that executes: the proposal is its segment at this point.
            assert np.array_equal(chunk[0, m["chunk_pos"]:m["chunk_pos"] + 10], m["proposal"])
        pos = m["chunk_pos"]
        import dataclasses
        assert m["chunk_spread"] == dataclasses.asdict(M.spread(chunk[:, pos:pos + 10]))
        fresh = chunk[:, :10] if m["kind"] == "chunk" else \
            np.stack([xa.executed_rows(c) for c in m["probe_raw"]])[:, :10]
        assert m["fresh_spread"] == dataclasses.asdict(M.spread(fresh))
        assert m["drift"] == dataclasses.asdict(M.drift(m["proposal"], fresh))
        if m["index"] not in ESCALATE:   # continued locally: the proposal is exactly what ran next
            ran = np.stack([by_step[m["step"] + i + 1]["action"] for i in range(10)])
            assert np.array_equal(ran, m["proposal"])
        # Features: the encoder's view of this observation, whichever endpoint produced them.
        prev = m["prev_segment"]
        assert len(prev) == (0 if m["index"] == 0 else 10)
        if prev:
            assert [s["action"].tobytes() for s in prev] == \
                   [by_step[m["step"] - 9 + i]["action"].tobytes() for i in range(10)]
            assert m["prev_segment_pi05"] == (prev[0]["driver"] == "pi05")
    after_abandon = [m for m in ep.measurements if m["step"] == ABANDON_STEP + 10]
    assert after_abandon and after_abandon[0]["chunk_pos"] == 10 and after_abandon[0]["chunk_samples_raw"] is not None


@requires_lerobot
def test_seeds_of_all_calls_are_distinct(runs):
    seeds = []
    for kind, s in runs["full_svc"].calls:
        seeds += list(s) if kind == "sample" else ([] if s is None else [s])
    assert len(seeds) == len(set(seeds))


@requires_lerobot
def test_live_deciders_make_only_their_own_calls(runs):
    full_pts = {m["index"]: m for m in runs["full"].measurements}
    for profile, expect in (("router", {"features"}), ("cascader_nospread", {"features"}),
                            ("cascader_chunkspread", {"features", "sample"}), ("cascader_full", {"sample"})):
        ep, svc, live = runs[profile]
        assert {k for k, _ in svc.calls} - {"act"} == expect, profile
        n_mid = sum(1 for m in full_pts.values() if m["kind"] == "mid")
        n_chunk = len(full_pts) - n_mid
        samples = sorted(len(s) for k, s in svc.calls if k == "sample")
        if profile == "cascader_full":     # K-1 at chunk points, K at mid points; the fresh chunk needs none
            assert samples == sorted([K - 1] * n_chunk + [K] * n_mid)
        else:
            assert sum(1 for k, _ in svc.calls if k == "features") == n_mid
        if profile == "cascader_chunkspread":   # K-1 at chunk points and for the fresh chunk
            assert samples == [K - 1] * (n_chunk + 1)
        # The live decider sees exactly what the recording run measured for its profile.
        assert sorted(live) == sorted(d for d in runs["observed"] if isinstance(d, int))
        for d, inputs in live.items():
            assert inputs_equal(inputs, runs["observed"][d][profile]), (profile, d)


@requires_lerobot
def test_the_fresh_chunk_charge_follows_who_draws_its_samples(runs):
    """Amendment 27: a profile's clock pays for the fresh chunk's samples exactly when its live decider draws them.
    (Each run's live clocks were also checked against the offline replay at its end.)"""
    from vlar_harness import clock as clk
    from vlar_harness import seeding
    ch = clk.Charges.from_config(CFG)
    fresh = tuple(seeding.call_seed(ROLLOUT, ABANDON_STEP, "xvla", "sample_fresh", k) for k in range(1, K))
    event = next(e for e in runs["full"].events if e["t"] == "xvla_act" and e["call"] == "act_fresh")
    assert ("sample", fresh) in runs["full_svc"].calls
    for profile in ("router", "cascader_nospread", "cascader_chunkspread", "cascader_full"):
        _, svc, _ = runs[profile]
        drew = ("sample", fresh) in svc.calls
        assert drew == (profile == "cascader_chunkspread"), profile
        assert (clk.event_s(event, ch, profile) > ch.xvla_act_s) == drew, profile


@requires_lerobot
def test_observe_fills_exactly_each_profiles_groups(runs):
    groups = ("proposal", "free_signals", "chunk_spread", "fresh_spread", "drift")
    for d, by_profile in runs["observed"].items():
        if not isinstance(d, int):
            continue
        for profile, inputs in by_profile.items():
            wanted = set(CFG["observe_inputs"][profile])
            assert {g for g in groups if getattr(inputs, g) is not None} == wanted - {"common"}, (profile, d)
            assert inputs.point.index == d and inputs.common.steps_elapsed == 10 * d
        f = by_profile["cascader_nospread"].free_signals
        assert f.tracking_valid == (d != 0 and by_profile["cascader_nospread"].proposal.prev_segment_pi05 is False)


@requires_lerobot
def test_no_inputs_once_the_budget_is_spent(runs, sim):
    ep, _ = make(sim)
    points = []

    def look(e, dp):
        if not dp.can_escalate:
            with pytest.raises(RuntimeError, match="budget"):
                e.observe("router")
            points.append(dp.index)

    run(ep, record="full", on_point=look)
    assert points == list(range(43, 52))


# ---------------------------------------------------------------- snapshots and branches

@requires_lerobot
@pytest.mark.parametrize("d", [11, 22])
def test_snapshots_carry_the_measurements(runs, other_sim, d):
    from vlar_harness import snapshot as snap
    s = snap.from_bytes(snap.to_bytes(runs["observed"]["snaps"][d]))
    ep, svc = make(other_sim)
    dp = ep.restore(s)
    assert dp.index == d
    for profile in PROFILES:
        if profile != "none":
            assert inputs_equal(ep.observe(profile), runs["observed"][d][profile]), profile
    # Continuing locally, the next point's measurements equal the original's (its chunk spread uses the restored
    # chunk-time samples).
    nxt = ep.decide("local" if d not in ESCALATE else "escalate").next_point
    want = next(m for m in runs["full"].measurements if m["index"] == nxt.index)
    got = ep.point_meas
    for k in want:
        assert (np.array_equal(got[k], want[k]) if isinstance(want[k], np.ndarray) else
                str(got[k]) == str(want[k])), k


@requires_lerobot
def test_branches_measure_nothing(runs, other_sim):
    from vlar_harness import snapshot as snap
    s = snap.from_bytes(snap.to_bytes(runs["observed"]["snaps"][11]))
    ep, svc = make(other_sim)
    br = ep.run_branch(s, branch="b2_one_call", pi05_calls=1, seed_index=1)
    assert {k for k, _ in svc.calls} == {"act"} and ep.measurements == []
    assert br.result.steps == 520
