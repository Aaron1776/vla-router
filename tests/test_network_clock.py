"""Network model and clock (SPEC §5, amendments 13-14), on synthetic requests and logs. Pure Python; the live
engine's clock is checked against offline replay with the simulator in tests/reference/test_clock_live.py."""

import dataclasses
import json
import math
import statistics

import pytest

from vlar_harness import clock as clk
from vlar_harness import eventlog
from vlar_harness import network as net
from vlar_harness import seeding
from vlar_harness.config import PROFILES, load_config

CFG = load_config()
CH = clk.Charges.from_config(CFG)
P = CH.network


def lossy(**changes) -> net.NetworkParams:
    return dataclasses.replace(P, **changes)


def within_sigma(observed: float, expected: float, n: int, k: float = 4.0) -> bool:
    return abs(observed - expected) <= k * math.sqrt(expected * (1 - expected) / n)


# ---------------------------------------------------------------- draws and their statistics

def test_round_trip_distribution_matches_the_spec():
    rtts = [net.draw_attempt(P, 7, s, 0)["rtt_s"] for s in range(0, 400_000, 10)]
    assert P.rtt_sigma == pytest.approx(math.log(0.30 / 0.06) / 2.053748910631823, rel=1e-12)
    assert statistics.median(rtts) == pytest.approx(0.060, rel=0.02)
    tail = sum(r > 0.300 for r in rtts) / len(rtts)
    assert within_sigma(tail, 0.02, len(rtts))


def test_attempt_failures_are_drops_or_timeouts_at_the_expected_rate():
    draws = [net.draw_attempt(P, 11, s, a) for s in range(0, 200_000, 10) for a in range(2)]
    failed = [d for d in draws if not d["ok"]]
    assert within_sigma(len(failed) / len(draws), net.p_attempt_fails(P), len(draws))
    assert within_sigma(sum(d["cause"] == "dropped" for d in draws) / len(draws), P.p_fail, len(draws))
    for d in draws:                          # the outcome is exactly what the recorded draws imply
        slow = P.attempt_time_s(d["rtt_s"]) > P.timeout_s
        expect = "dropped" if d["u_fail"] < P.p_fail else ("timeout" if slow else None)
        assert d["cause"] == expect and d["ok"] == (expect is None)
    assert any(d["cause"] == "timeout" for d in draws)


def test_abandonment_rate_is_the_attempt_failure_rate_cubed():
    params = lossy(p_fail=0.3)
    link = net.SimulatedNetwork(params)
    n = 20_000
    abandoned = sum(not link.request(lambda: None, step=s, rollout_seed=3, max_attempts=3).ok
                    for s in range(0, 10 * n, 10))
    assert within_sigma(abandoned / n, net.p_attempt_fails(params) ** 3, n)


def test_a_retry_gets_fresh_draws_and_can_succeed():
    """Amendment 13 (the user's correction): draws are keyed by attempt number too, so a retry is not a replay
    of the attempt that just failed."""
    link = net.SimulatedNetwork(P)
    retried, recovered = 0, 0
    for s in range(0, 200_000, 10):
        first = net.draw_attempt(P, 21, s, 0)
        if first["ok"]:
            continue
        retried += 1
        calls = []
        out = link.request(lambda: calls.append(s) or "actions", step=s, rollout_seed=21, max_attempts=3)
        assert out.attempts[0] == first
        assert out.attempts[1]["rtt_s"] != first["rtt_s"] and out.attempts[1]["u_fail"] != first["u_fail"]
        if out.attempts[1]["ok"]:
            recovered += 1
            assert out.ok and len(out.attempts) == 2 and out.actions == "actions"
            assert calls == [s]                          # the server ran once, for the attempt that got through
    assert retried > 500
    assert within_sigma(recovered / retried, 1 - net.p_attempt_fails(P), retried)


def test_same_rollout_seed_reproduces_the_failure_pattern():
    def pattern(seed, link):
        return [[(a["ok"], a["cause"], a["rtt_s"]) for a in
                 link.request(lambda: 0, step=s, rollout_seed=seed, max_attempts=20).attempts]
                for s in range(0, 50_000, 10)]

    params = lossy(p_fail=0.4)
    a = pattern(5, net.SimulatedNetwork(params))
    assert a == pattern(5, net.SimulatedNetwork(params))      # a fresh link, same seed: same pattern
    assert sum(len(x) > 1 for x in a) > 1000                  # the pattern has plenty of retries
    assert a != pattern(6, net.SimulatedNetwork(params))


def test_draws_depend_only_on_seed_step_request_and_attempt():
    """Nothing else enters: not the order of requests, not how many came before, not the clock or profile."""
    ref = {(s, k): net.draw_attempt(P, 9, s, k) for s in (0, 10, 250, 510) for k in range(3)}
    link = net.SimulatedNetwork(P)
    for s in reversed(range(0, 520, 10)):
        link.request(lambda: 0, step=s, rollout_seed=9, max_attempts=3)
    assert {key: net.draw_attempt(P, 9, *key) for key in ref} == ref
    seeds = {seeding.network_attempt_seed(9, 250, "pi05", "act", k) for k in range(3)}
    seeds.add(seeding.network_attempt_seed(9, 260, "pi05", "act", 0))
    seeds.add(seeding.network_attempt_seed(9, 250, "xvla", "act", 0))
    seeds.add(seeding.network_attempt_seed(10, 250, "pi05", "act", 0))
    assert len(seeds) == 6
    assert seeding.network_attempt_seed(9, 250, "pi05", "act", 0) != seeding.call_seed(9, 250, "pi05", "act")


def test_uniform_is_strictly_inside_the_unit_interval():
    us = [seeding.uniform(s, "x") for s in range(20_000)]
    assert 0.0 < min(us) and max(us) < 1.0
    assert statistics.mean(us) == pytest.approx(0.5, abs=0.01)


def test_outcomes_are_checked():
    good = net.RemoteOutcome(True, 0, [net.attempt_record(0, False, cause="dropped"), net.attempt_record(1, True)])
    net.check_outcome(good, 3)
    with pytest.raises(ValueError):
        net.check_outcome(good, 1)                           # more attempts than allowed
    with pytest.raises(ValueError):
        net.check_outcome(net.RemoteOutcome(False, None, [net.attempt_record(0, False, cause="dropped")]), 3)
    with pytest.raises(ValueError):
        net.attempt_record(0, True, cause="timeout")


def test_upload_size_is_the_median_of_summed_jpeg_sizes():
    assert net.median_upload_image_bytes([[10, 20], [30, 40], [5, 5]]) == 30
    assert net.median_upload_image_bytes([[10, 11], [20, 20]]) == 31          # 30.5 rounds half up
    assert P.upload_bytes == 20290 + 1024
    assert P.upload_s == (20290 + 1024) * 8 / 20e6


# ---------------------------------------------------------------- clock charges

def fail(i):
    return net.attempt_record(i, False, cause="dropped", u_fail=0.01, rtt_s=0.05)


def test_request_charges_follow_the_spec():
    n = P
    assert clk.request_s([net.attempt_record(0, True, rtt_s=0.07)], n) == n.attempt_time_s(0.07)
    t = 0.0
    t += n.timeout_s
    t += n.backoff_s
    t += n.attempt_time_s(0.05)
    assert clk.request_s([fail(0), net.attempt_record(1, True, rtt_s=0.05)], n) == t
    t = 0.0
    for i in range(3):
        t += n.timeout_s
        if i < 2:
            t += n.backoff_s
    assert clk.request_s([fail(0), fail(1), fail(2)], n) == t == pytest.approx(3.4)
    t = 0.0
    for i in range(20):
        t += n.timeout_s
        if i < 19:
            t += n.backoff_s
    assert clk.request_s([fail(i) for i in range(20)], n) == t == pytest.approx(23.8)
    with pytest.raises(ValueError):
        clk.request_s([net.attempt_record(0, True), fail(1)], n)


def test_profile_overheads_per_point_kind():
    lat = CFG["latency"]
    act, feat, samp = lat["xvla_act_s"], lat["xvla_features_s"], lat["xvla_sample_k_s"]
    expected = {"none": (0.0, 0.0), "router": (0.0, feat), "cascader_nospread": (0.0, feat),
                "cascader_chunkspread": (samp - act, feat), "cascader_full": (samp - act, samp)}
    for p, (chunk, mid) in expected.items():
        assert CH.point_s[p] == {"chunk": chunk, "mid": mid}
        assert clk.event_s({"t": "decision_point", "kind": "chunk", "can_escalate": True}, CH, p) == chunk
        assert clk.event_s({"t": "decision_point", "kind": "mid", "can_escalate": True}, CH, p) == mid
    assert clk.event_s({"t": "xvla_act"}, CH, "router") == act


@pytest.mark.parametrize("profile", PROFILES)
def test_no_overhead_where_escalation_is_not_offered(profile):
    """Amendment 15: once the budget is spent the decider is not consulted, so its overhead is not charged."""
    for kind in ("chunk", "mid"):
        assert clk.event_s({"t": "decision_point", "kind": kind, "can_escalate": False}, CH, profile) == 0.0
    with pytest.raises(KeyError):
        clk.event_s({"t": "decision_point", "kind": "mid"}, CH, profile)          # the flag is required


def test_segments_are_charged_per_step_and_unknown_events_refused():
    c = clk.Clock(CH, "none")
    c.apply({"t": "segment", "driver": "xvla", "start": 0, "steps": 3})
    assert c.t == 0.0 + 0.05 + 0.05 + 0.05
    with pytest.raises(ValueError, match="per step"):
        c.event({"t": "segment", "steps": 3})
    with pytest.raises(ValueError, match="unknown event"):
        c.event({"t": "mystery"})


# ---------------------------------------------------------------- synthetic logs: score and crossing

def synthetic_log(*, success_at=None, requests=((0,),), profile_points=True):
    """start; chunk point; X-VLA 10 steps; mid point; escalate (a request with the given attempts);
    π0.5 10 steps; chunk point; X-VLA steps until `success_at` or 30 steps; end."""
    ev = [{"t": "start", "task_id": 0, "mode": "routed", "time_limit_s": None}]
    ev += [{"t": "xvla_act", "step": 0, "call": "act", "context": "start", "first_target_dist_m": 0.01},
           {"t": "decision_point", "step": 0, "index": 0, "kind": "chunk", "budget_remaining": 260, "can_escalate": True},
           {"t": "decision", "step": 0, "choice": "local"},
           {"t": "segment", "driver": "xvla", "start": 0, "steps": 10},
           {"t": "decision_point", "step": 10, "index": 1, "kind": "mid", "budget_remaining": 260, "can_escalate": True},
           {"t": "decision", "step": 10, "choice": "escalate"}]
    for oks in requests:
        attempts = [net.attempt_record(i, ok, cause=None if ok else "timeout", u_fail=0.9, rtt_s=0.07 if ok else 0.9)
                    for i, ok in enumerate(oks)]
        ev.append({"t": "pi05_request", "step": 10, "ok": attempts[-1]["ok"], "attempts": attempts})
    ev += [{"t": "handoff", "step": 10, "to": "pi05"},
           {"t": "segment", "driver": "pi05", "start": 10, "steps": 10},
           {"t": "xvla_act", "step": 20, "call": "act", "context": "after_pi05", "first_target_dist_m": 0.02},
           {"t": "decision_point", "step": 20, "index": 2, "kind": "chunk", "budget_remaining": 250, "can_escalate": True},
           {"t": "decision", "step": 20, "choice": "local"},
           {"t": "handoff", "step": 20, "to": "xvla"}]
    n = 30 if success_at is None else success_at - 20
    ev.append({"t": "segment", "driver": "xvla", "start": 20, "steps": n})
    ev.append({"t": "end", "step": 20 + n, "reason": "step_limit" if success_at is None else "success"})
    return ev


def manual_time(profile, request_attempts=((True,),)):
    t = 0.0
    lat = CFG["latency"]
    t += lat["xvla_act_s"]
    t += CH.point_s[profile]["chunk"]
    for _ in range(10):
        t += 0.05
    t += CH.point_s[profile]["mid"]
    t += clk.request_s([net.attempt_record(i, ok, cause=None if ok else "timeout", rtt_s=0.07 if ok else 0.9)
                        for oks in request_attempts for i, ok in enumerate(oks)], P)
    for _ in range(10):
        t += 0.05
    t += lat["xvla_act_s"]
    t += CH.point_s[profile]["chunk"]
    return t


@pytest.mark.parametrize("profile", PROFILES)
def test_score_is_the_clock_at_the_success_step(profile):
    log = synthetic_log(success_at=37, requests=[(False, True)])
    s = clk.score(log, profile, CH)
    t = manual_time(profile, [(False, True)])
    for _ in range(17):
        t += 0.05
    assert s.success and s.success_step == 37 and s.time_s == t
    assert s.within(t) and not s.within(math.nextafter(t, 0.0))
    assert not clk.score(synthetic_log(success_at=None), profile, CH).within(1e9)


def test_profiles_order_the_same_log_as_expected():
    log = synthetic_log(success_at=45)
    t = {p: clk.score(log, p, CH).time_s for p in PROFILES}
    assert t["none"] < t["router"] == t["cascader_nospread"] < t["cascader_chunkspread"] < t["cascader_full"]


def test_first_crossing_is_where_a_live_limit_would_cut():
    log = synthetic_log(success_at=45)
    # inside the first X-VLA segment: 0.42 s for /act, then 0.05 s per step; the 4th step is the first above 0.59
    cr = clk.first_crossing(log, "router", CH, 0.42 + 0.17)
    assert cr.event_index == 4 and cr.step == 4 and cr.time_s > 0.59
    # during the mid point's /features overhead
    t_before_mid = clk.replay(log[:5], "router", CH)
    cr = clk.first_crossing(log, "router", CH, t_before_mid + 0.1)
    assert cr.event_index == 5 and cr.step is None
    assert clk.first_crossing(log, "router", CH, 1e9) is None


def test_logs_round_trip_through_json_exactly(tmp_path):
    log = synthetic_log(success_at=41, requests=[(False, False, True)])
    eventlog.check(log)
    path = tmp_path / "logs.jsonl"
    eventlog.write_jsonl(path, [{"events": log, "episode_id": "x"}])
    back = eventlog.read_logs([path])[0]
    assert back == log
    assert clk.replay_all(back, CH) == clk.replay_all(log, CH)
    assert eventlog.first_target_distances([back], "stretch") == []
    assert eventlog.first_target_distances([back], "after_pi05") == [0.02]
    with pytest.raises(eventlog.LogError):
        eventlog.check(log[:-1])                                   # no end
    with pytest.raises(eventlog.LogError):
        eventlog.check(log, complete=False)                        # a finished log is not a prefix
    assert json.loads(eventlog.dumps(log)) == log
