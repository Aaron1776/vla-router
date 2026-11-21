"""The simulated clock (docs/SPEC.md §5): one charge function for the live engine and for offline re-scoring
of a saved event log under any overhead profile.

The clock is a deterministic function of the event log and the configured latencies:
- each environment step: `env.control_period_s` (0.05 s, LIBERO's 20 Hz);
- each X-VLA `/act` (`act` at a chunk point, `act_fresh` after an abandoned escalation): `latency.xvla_act_s`;
- each π0.5 request: the sum over its attempts. A failed attempt costs the timeout, plus the backoff when
  another attempt follows; the attempt that gets through costs upload + round trip + download + π0.5's
  inference (`network.NetworkParams.attempt_time_s`, from the round trip recorded in the log);
- each decision point where escalation is offered: the profile's overhead terms for its kind (`profiles` in the
  config): `features` = `/features`, `sample_k` = `/sample` with K samples, `sample_k_less_act` = the latter minus
  `/act`. Once the π0.5 budget is spent the decider is not consulted, so a point with `can_escalate` false costs
  nothing (§5, amendment 15);
- everything else: nothing. Decider inference is not charged; measurement calls are charged only through the
  profile terms.

A network attempt's outcome is part of what happened (an abandoned escalation changes the trajectory), so the
clock charges the outcome recorded in the log and never re-decides it.

Live, the engine charges every environment step as it runs and every other event as it is emitted, so its
time limit can cut after any step or model call. The log stores a segment's steps as one `segment` event (a
run of steps); replay expands it and charges the steps one at a time, in log order. Live and offline clocks
therefore perform the same floating-point additions in the same order and agree bit for bit; the engine
checks this at the end of every episode, for every profile.

Pure Python, so logs can be re-scored anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from .config import PROFILES
from .network import NetworkParams

STEP_RUN = "segment"                     # a run of environment steps, charged per step
CHARGED = ("xvla_act", "pi05_request", "decision_point")
ZERO_COST = frozenset({"start", "decision", "handoff", "escalation_abandoned", "end", "restore", "branch"})
POINT_KINDS = ("chunk", "mid")


def _sum(values: Iterable[float]) -> float:
    total = 0.0
    for v in values:
        total += v
    return total


@dataclass(frozen=True)
class Charges:
    step_s: float
    xvla_act_s: float
    network: NetworkParams
    point_s: Mapping[str, Mapping[str, float]]      # profile -> {"chunk": s, "mid": s}

    @staticmethod
    def from_config(cfg) -> "Charges":
        lat = cfg["latency"]
        act, features, sample_k = float(lat["xvla_act_s"]), float(lat["xvla_features_s"]), float(lat["xvla_sample_k_s"])
        term_s = {"features": features, "sample_k": sample_k, "sample_k_less_act": sample_k - act}
        point_s = {}
        for profile in PROFILES:
            spec = cfg["profiles"][profile]
            point_s[profile] = {kind: _sum(term_s[t] for t in spec[f"{kind}_point"]) for kind in POINT_KINDS}
        return Charges(step_s=float(cfg["env"]["control_period_s"]), xvla_act_s=act,
                       network=NetworkParams.from_config(cfg), point_s=point_s)


def request_s(attempts: Sequence[Mapping], net: NetworkParams) -> float:
    """Time the robot waits for one π0.5 request, from its logged attempts."""
    total = 0.0
    last = len(attempts) - 1
    for i, a in enumerate(attempts):
        if a["ok"]:
            if i != last:
                raise ValueError("a request stops at the attempt that gets through")
            total += net.attempt_time_s(a["rtt_s"])
        else:
            total += net.timeout_s
            if i != last:
                total += net.backoff_s
    return total


def event_s(event: Mapping, charges: Charges, profile: str) -> float:
    """Charge of one logged event other than a run of steps."""
    kind = event["t"]
    if kind == "xvla_act":
        return charges.xvla_act_s
    if kind == "pi05_request":
        return request_s(event["attempts"], charges.network)
    if kind == "decision_point":                 # no overhead where escalation is not offered (amendment 15)
        return charges.point_s[profile][event["kind"]] if event["can_escalate"] else 0.0
    if kind in ZERO_COST:
        return 0.0
    if kind == STEP_RUN:
        raise ValueError("segment events are charged per step (Clock.step / Clock.apply), not as one event")
    raise ValueError(f"unknown event type {kind!r}: give it a clock charge in clock.py first")


class Clock:
    """Simulated time under one profile. `step()` and `event()` are the live calls; `apply()` replays one logged
    event, expanding a run of steps into `step()` calls."""

    __slots__ = ("charges", "profile", "t")

    def __init__(self, charges: Charges, profile: str):
        if profile not in charges.point_s:
            raise ValueError(f"profile must be one of {tuple(charges.point_s)}")
        self.charges, self.profile, self.t = charges, profile, 0.0

    def step(self) -> float:
        self.t += self.charges.step_s
        return self.t

    def event(self, event: Mapping) -> float:
        self.t += event_s(event, self.charges, self.profile)
        return self.t

    def apply(self, event: Mapping) -> float:
        if event["t"] == STEP_RUN:
            for _ in range(int(event["steps"])):
                self.step()
            return self.t
        return self.event(event)


def replay(events: Iterable[Mapping], profile: str, charges: Charges) -> float:
    """Offline clock: the time at the end of the log under `profile`."""
    c = Clock(charges, profile)
    for e in events:
        c.apply(e)
    return c.t


def replay_all(events: Sequence[Mapping], charges: Charges, profiles: Sequence[str] = PROFILES) -> dict[str, float]:
    return {p: replay(events, p, charges) for p in profiles}


@dataclass(frozen=True)
class Score:
    """An event log scored under one profile."""
    end_reason: Optional[str]          # None for a log that has not ended (a snapshot's prefix)
    success_step: Optional[int]
    time_s: float                      # clock at the end of the log; for a success, right after the success step

    @property
    def success(self) -> bool:
        """Task done before the step limit (and before any live time cut)."""
        return self.end_reason == "success"

    def within(self, limit_s: float) -> bool:
        """Success before the step limit and within `limit_s`. The clock never decreases, so this equals what a
        live time limit of `limit_s` would have allowed."""
        return self.success and self.time_s <= limit_s


def score(events: Sequence[Mapping], profile: str, charges: Charges) -> Score:
    c = Clock(charges, profile)
    end: Optional[Mapping] = None
    for i, e in enumerate(events):
        c.apply(e)
        if e["t"] == "end":
            if i != len(events) - 1:
                raise ValueError(f"event {i} ends the episode but {len(events) - 1 - i} events follow it")
            end = e
    if end is None:
        return Score(None, None, c.t)
    return Score(end["reason"], end["step"] if end["reason"] == "success" else None, c.t)


@dataclass(frozen=True)
class Crossing:
    event_index: int                   # the event during or after which the clock first exceeded the limit
    step: Optional[int]                # for a run of steps: the environment step count after the crossing step
    time_s: float


def first_crossing(events: Sequence[Mapping], profile: str, charges: Charges, limit_s: float) -> Optional[Crossing]:
    """Where a live time limit of `limit_s` would have cut this (uncut) run: the first check, after any step or
    event, at which the clock exceeds the limit."""
    c = Clock(charges, profile)
    for i, e in enumerate(events):
        if e["t"] == STEP_RUN:
            for k in range(int(e["steps"])):
                if c.step() > limit_s:
                    return Crossing(i, int(e["start"]) + k + 1, c.t)
        elif c.event(e) > limit_s:
            return Crossing(i, None, c.t)
    return None
