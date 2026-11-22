"""The episode engine: decision points, the π0.5 budget, run modes and the handoff rules (docs/SPEC.md §3-§4;
effort step 3), with the event log, simulated clock, network model, time limit and branch runs (§5, §7;
effort step 4).

Run modes
- routed: a decision point every 10 steps (0, 10, ..., 510). At a chunk point (X-VLA's queue is empty)
  X-VLA's `/act` is called *before* the decision. `decide("local")` runs the next 10 queued X-VLA actions;
  `decide("escalate")` hands one 10-step segment to π0.5.
- xvla_only: X-VLA alone, exactly the authors' client loop (a chunk whenever the queue is empty).
- pi05_only: π0.5 alone, called every 10 steps, no budget, as `lerobot-eval` with n_action_steps=10. A failed
  request is retried until it gets through, up to `max_attempts_pi05_only` attempts; then the episode ends with
  end reason `network` (§5, amendment 13).

Handoff rules (§3)
- Escalate: X-VLA's queued actions are discarded and its remembered pose forgotten, the controller switches to
  delta mode, π0.5 is called fresh, the first 10 of its 50 actions run.
- Back to X-VLA: the controller switches to absolute mode; X-VLA's next call is a chunk-point `/act`, which
  sends the measured end-effector pose because the remembered pose was forgotten. Within an uninterrupted
  X-VLA stretch the quirk applies (config option `xvla_state_mode`).
- Abandoned escalation (every attempt of the π0.5 request failed, §5): the controller returns to absolute
  mode, X-VLA is called fresh (`act_fresh`) with the measured pose and runs 10 actions; its other 20 stay
  queued; no budget is charged. Failed requests cost time, never budget.
The controller mode is checked before every segment, so each executed step runs in its model's mode.

The π0.5 request goes through a `RemoteLink`: `network.SimulatedNetwork` by default. A retry is the same
request: the server is called once, for the attempt that gets through, with the request's own noise seed.

Event log (§5). Every run records plain JSON events; the last column is the clock charge (clock.py):
  start                 the episode's identity and settings                                   0
  xvla_act              an X-VLA /act (`act` or `act_fresh`), its noise seed, the distance     /act time (+ the
                                                                                              profile's act_fresh
                                                                                              terms, amendment 27)
                        of its first target from the arm and the context (start, stretch,
                        after_pi05, after_abandon) for the VM handoff check (§11 step 3)
  decision_point        step, index, kind (chunk or mid), queue, budget left, can_escalate    profile overhead,
                                                                                              if can_escalate
  decision              the choice made at the point                                            0
  pi05_request          noise seed, outcome and every attempt's network draws                  sum of attempts
  handoff               control passes to the other model                                       0
  escalation_abandoned                                                                          0
  segment               a run of executed steps of one model: driver, start step, steps        0.05 s per step
  end                   reason: success, step_limit, time_limit or network                      0
  restore, branch       a snapshot resumed; a branch's seed and schedule                        0
The engine keeps one clock per profile, charging each step as it runs and every other event as it is emitted,
and checks at the end of every episode that each equals the offline replay of the log. Scripted runs (base,
exploration, branches, scripted baselines) run uncut and are scored offline for any profile and time limit;
runs with a live learned decider pass `time_limit=True` (routed mode with a decider profile only, amendment
19), and the run's own profile is then checked against the task's limit after every step and every model call
(end reason `time_limit`). The start event records `time_limit` true or false next to `time_limit_s`, the
branch event records false (branches run uncut), and the episode result carries the flag (amendment 20); a cut
run's outcome is scored only under its own profile (clock.py, amendment 21).

Model noise seeds come from `seeding.call_seed(rollout_seed, ...)`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import clock as clk
from . import measure as meas
from . import seeding
from .api import (CommonInputs, DecisionPoint, DeciderInputs, EpisodeResult, Proposal, SegmentOutcome, SpreadSummary,
                  Drift, check_reset)
from .config import BRANCHES, PROFILES
from .network import DirectLink, RemoteLink, RemoteOutcome, SimulatedNetwork, check_outcome  # noqa: F401

MODES = ("routed", "xvla_only", "pi05_only")
END_REASONS = ("success", "step_limit", "time_limit", "network")
XVLA_CONTEXTS = ("start", "stretch", "after_pi05", "after_abandon")
TRAJECTORY_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
RECORD_MODES = ("off", "full")
INPUT_GROUPS = ("common", "proposal", "free_signals", "chunk_spread", "fresh_spread", "drift")
CHUNK_SAMPLE_GROUPS = frozenset({"chunk_spread", "fresh_spread", "drift"})
LOG_VERSION = 1


def _float32_exact(values) -> np.ndarray:
    """A server chunk stored as float32: the servers send float32 values, so this must lose nothing."""
    a = np.asarray(values, dtype=np.float64)
    out = a.astype(np.float32)
    if not np.array_equal(out.astype(np.float64), a):
        raise ValueError("a model chunk is not exactly representable in float32")
    return out


class BudgetExhausted(RuntimeError):
    pass


class ClockMismatch(RuntimeError):
    """A live clock differs from the offline replay of its own log: a bug that would corrupt labels."""


@dataclass
class EpisodeState:
    """Everything about the episode beyond the simulator and the X-VLA client (goes into snapshots)."""
    task_id: int
    init_idx: int
    scene_seed: int
    rollout_seed: int
    mode: str
    episode_id: str = ""
    profile: str = "none"                 # the decider's profile: its clock is the one a live limit checks
    time_limit: bool = False              # the limit is applied live (amendments 19-20)
    time_limit_s: Optional[float] = None  # set only when the limit is applied live
    decision_index: int = -1
    point_kind: Optional[str] = None
    pi05_steps: int = 0
    pi05_requests: int = 0
    pi05_failed_requests: int = 0         # failed attempts
    prev_driver: Optional[str] = None     # driver of the last executed segment
    success_step: Optional[int] = None
    end_reason: Optional[str] = None
    record: str = "off"                   # "full": every decision point's measurements are recorded (§7)

    def as_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class BranchRun:
    """One scripted continuation from a snapshot (§7). `events` is the branch's own log, starting with its
    `restore` and `branch` events; the snapshot's log is the prefix it continues."""
    branch: str
    seed_index: int
    pi05_calls: int
    rollout_seed: int
    result: EpisodeResult
    events: list = field(default_factory=list)
    trajectory: list = field(default_factory=list)


class Episode:
    """One episode on one LiberoSim with its two adapters.

    `pi05_call(raw_observation, task, seed) -> 50 x 7 actions` is the π0.5 server call (the PI05Client in
    production). `on_event(event)` and `on_step(episode, action, driver)` are hooks for tests.
    """

    def __init__(self, sim, xvla, pi05, pi05_call: Callable, cfg, *, remote: Optional[RemoteLink] = None,
                 measure_client=None,
                 on_event: Optional[Callable[[dict], None]] = None,
                 on_step: Optional[Callable[["Episode", np.ndarray, str], None]] = None):
        """`measure_client` has X-VLA's measurement endpoints, `sample(payload, seeds)` and `features(payload)`
        (servers.clients.XVLAClient in production); without it no decider inputs are measured."""
        self.sim, self.xvla, self.pi05 = sim, xvla, pi05
        self.pi05_call = pi05_call
        self.measure_client = measure_client
        self.cfg = cfg
        self.K = int(cfg["measurement"]["K"])
        thr = cfg["measurement"]["gripper_thresholds"]
        self.gripper_thresholds = (thr.get("shut_width_m"), thr.get("open_width_m"))
        self.chunk_len = int(cfg["models"]["xvla"]["chunk_len"])
        self.charges = clk.Charges.from_config(cfg)
        self.remote = remote if remote is not None else SimulatedNetwork(self.charges.network)
        self.on_event, self.on_step = on_event, on_step
        self.step_limit = int(cfg["env"]["step_limit"])
        self.segment = int(cfg["decision"]["segment_steps"])
        self.budget_steps = int(cfg["budget"]["pi05_max_steps"])
        self.scene_offset = int(cfg.option("scene_seed"))
        self.state: Optional[EpisodeState] = None
        self.events: list[dict] = []
        self.log_offset = 0                   # events before this index came from a restored snapshot
        self.trajectory: list[dict] = []
        self.last_act_response: Optional[dict] = None
        self.clocks: dict[str, clk.Clock] = {}
        self._reset_measurements()

    # ------------------------------------------------------------------ setup

    def start(self, init_idx: int, rollout_seed: int, mode: str = "routed", episode_id: str = "", *,
              profile: str = "none", time_limit: bool = False, record: str = "off") -> Optional[DecisionPoint]:
        """Set up an episode. `profile` is the decider's (§5); `time_limit=True` applies the task's limit live,
        for runs with a live learned decider: routed mode with a profile other than `none` (amendment 19).
        Scripted runs leave it off and are scored offline. `record="full"` measures everything any decider could
        need at every decision point where escalation is offered (base, exploration and DAgger rollouts, §7)."""
        check_reset(mode, profile, time_limit)                # before anything touches the simulator
        if record not in RECORD_MODES:
            raise ValueError(f"record must be one of {RECORD_MODES}")
        if record == "full" and (mode != "routed" or self.measure_client is None):
            raise ValueError("record='full' needs routed mode and a measurement client")
        scene = seeding.scene_seed(init_idx, self.scene_offset)
        self.sim.setup_episode(init_idx, env_seed=scene)
        self.xvla.reset()
        limit = float(self.cfg.time_limit_s(self.sim.task_id)) if time_limit else None
        self.state = EpisodeState(task_id=self.sim.task_id, init_idx=init_idx, scene_seed=scene,
                                  rollout_seed=int(rollout_seed), mode=mode, episode_id=episode_id, profile=profile,
                                  time_limit=time_limit, time_limit_s=limit, record=record)
        self.events, self.log_offset, self.trajectory, self.last_act_response = [], 0, [], None
        self._reset_measurements()
        self.clocks = {p: clk.Clock(self.charges, p) for p in PROFILES}
        self._emit({"t": "start", "log_version": LOG_VERSION, "task_id": self.sim.task_id, "init_idx": init_idx,
                    "scene_seed": scene, "rollout_seed": int(rollout_seed), "mode": mode, "profile": profile,
                    "time_limit": time_limit, "time_limit_s": limit, "episode_id": episode_id, "record": record,
                    "config_hash": getattr(self.cfg, "hash", None)})
        if mode == "routed":
            return self._decision_point()
        return None

    @property
    def goal(self) -> str:
        return self.sim.task_description

    @property
    def budget_remaining(self) -> int:
        return self.budget_steps - self.state.pi05_steps

    @property
    def done(self) -> bool:
        return self.state.end_reason is not None

    @property
    def own_events(self) -> list[dict]:
        """The events of this run (after a restored snapshot's prefix, if any)."""
        return self.events[self.log_offset:]

    def times(self) -> dict[str, float]:
        """Simulated time so far under every profile."""
        return {p: c.t for p, c in self.clocks.items()}

    def time_remaining_s(self) -> float:
        """What the decider sees (§6): time left under its own profile; infinite when no limit applies live."""
        lim = self.state.time_limit_s
        return float("inf") if lim is None else lim - self.clocks[self.state.profile].t

    # ------------------------------------------------------------------ routed mode

    def _decision_point(self) -> Optional[DecisionPoint]:
        st = self.state
        if self._check_end():
            return None
        step = self.sim.steps
        st.decision_index = step // self.segment
        if self.xvla.queue_len() == 0:
            st.point_kind = "chunk"
            self._xvla_query("act", step)               # before the decision (§3)
            if self.done:
                return None
        else:
            st.point_kind = "mid"
        self._emit({"t": "decision_point", "step": step, "index": st.decision_index, "kind": st.point_kind,
                    "queue": self.xvla.queue_len(), "budget_remaining": self.budget_remaining,
                    "can_escalate": self.budget_remaining >= self.segment})
        if self._cut_if_over_time():
            return None
        self.point_meas = None
        if self.budget_remaining >= self.segment and self._needs():   # no decider is consulted otherwise (amendment 15)
            self._measure_point()
        return self.current_point()

    def current_point(self) -> Optional[DecisionPoint]:
        if self.done or self.state.mode != "routed":
            return None
        return DecisionPoint(index=self.state.decision_index, step=self.sim.steps, kind=self.state.point_kind,
                             budget_remaining_steps=self.budget_remaining,
                             can_escalate=self.budget_remaining >= self.segment)

    def decide(self, choice: str) -> SegmentOutcome:
        st = self.state
        if st.mode != "routed" or self.done:
            raise RuntimeError("decide() needs a routed episode at a decision point")
        if choice not in ("local", "escalate"):
            raise ValueError("choice must be 'local' or 'escalate'")
        if choice == "escalate" and self.budget_remaining < self.segment:
            raise BudgetExhausted(f"{self.budget_remaining} pi0.5 steps left; escalation not allowed")
        requests_before, failed_before = st.pi05_requests, st.pi05_failed_requests
        self._emit({"t": "decision", "step": self.sim.steps, "index": st.decision_index, "choice": choice})
        if choice == "escalate":
            driver, abandoned = self._escalate()
        else:
            driver, abandoned = self._run_xvla_segment(), False
        nxt = self._decision_point()
        return SegmentOutcome(executed_by=driver, escalation_abandoned=abandoned,
                              pi05_requests=st.pi05_requests - requests_before,
                              pi05_failed_requests=st.pi05_failed_requests - failed_before,
                              done=self.done, next_point=nxt, episode=self.result() if self.done else None)

    def _escalate(self) -> tuple[str, bool]:
        st, step = self.state, self.sim.steps
        # Handoff X-VLA -> π0.5: drop X-VLA's plan and its remembered pose, delta control, fresh π0.5 call.
        self.xvla.discard_queue()
        self.xvla.forget_predicted_pose()
        self.chunk_raw = self.chunk_exec = None
        self.sim.set_control_mode("delta")
        outcome = self._pi05_request(step, self.charges.network.max_attempts_routed)
        if self._cut_if_over_time():
            return ("pi05" if outcome.ok else "xvla"), not outcome.ok
        if outcome.ok:
            if st.prev_driver != "pi05":
                self._emit({"t": "handoff", "step": step, "to": "pi05"})
            st.pi05_steps += self._execute([self.pi05.env_action(a) for a in self._pi05_actions(outcome)], "pi05")
            return "pi05", False
        # Abandoned: X-VLA, called fresh from the measured pose, drives this segment; no budget charged.
        self._emit({"t": "escalation_abandoned", "step": step})
        if st.prev_driver == "pi05":
            self._emit({"t": "handoff", "step": step, "to": "xvla"})
        self.sim.set_control_mode("absolute")
        self._xvla_query("act_fresh", step)
        if not self.done:
            self._execute_from_queue()
        return "xvla", True

    def _run_xvla_segment(self) -> str:
        if self.state.prev_driver == "pi05":
            self._emit({"t": "handoff", "step": self.sim.steps, "to": "xvla"})
        self.sim.set_control_mode("absolute")
        self._execute_from_queue()
        return "xvla"

    # ------------------------------------------------------------------ single-model modes

    def run_episode(self) -> EpisodeResult:
        st = self.state
        if st.mode == "xvla_only":
            self.sim.set_control_mode("absolute")
            while not self._check_end():
                self._xvla_query("act", self.sim.steps)            # the queue is empty here
                if self.done:
                    break
                self._execute([None] * self.xvla.queue_len(), "xvla")
        elif st.mode == "pi05_only":
            self.sim.set_control_mode("delta")
            while not self._check_end():
                outcome = self._pi05_request(self.sim.steps, self.charges.network.max_attempts_pi05_only)
                if self._cut_if_over_time():
                    break
                if not outcome.ok:
                    self._end("network")                           # safety cap reached (§5, amendment 13)
                    break
                st.pi05_steps += self._execute([self.pi05.env_action(a) for a in self._pi05_actions(outcome)],
                                               "pi05")
        else:
            raise RuntimeError("run_episode() is for xvla_only and pi05_only; routed episodes use decide()")
        return self.result()

    # ------------------------------------------------------------------ model calls

    def _xvla_query(self, call: str, step: int) -> None:
        st = self.state
        if call == "act_fresh":
            context = "after_abandon"
        elif st.prev_driver is None:
            context = "start"
        else:
            context = "after_pi05" if st.prev_driver == "pi05" else "stretch"
        seed = seeding.call_seed(st.rollout_seed, step, "xvla", call)
        arm = np.array(self.sim.controller_pose()[0], dtype=np.float64)       # what a measured-pose request sends
        self.xvla.query(self.sim, self.goal, seed=seed)
        self.last_act_response = self.xvla.client.last_response
        # First target as it will execute (popped as float32), against the arm at query time: the quantity the
        # VM handoff check compares with ordinary chunk boundaries (§11 step 3, amendment 10).
        first = np.asarray(self.xvla.queued_actions()[0, :3], dtype=np.float32).astype(np.float64)
        self._emit({"t": "xvla_act", "step": step, "call": call, "seed": seed, "context": context,
                    "first_target_dist_m": float(np.linalg.norm(first - arm))})
        self.chunk_raw = self.chunk_exec = None
        if not self._cut_if_over_time():
            self._chunk_samples(call, step)

    def _pi05_request(self, step: int, max_attempts: int) -> RemoteOutcome:
        """One π0.5 request through the link; the server runs once, with the request's own noise seed."""
        st = self.state
        raw = self.pi05.raw_observation()
        seed = seeding.call_seed(st.rollout_seed, step, "pi05", "act")
        outcome = self.remote.request(lambda: self.pi05_call(raw, self.goal, seed), step=step,
                                      rollout_seed=st.rollout_seed, max_attempts=max_attempts)
        check_outcome(outcome, max_attempts)
        st.pi05_requests += 1
        st.pi05_failed_requests += sum(1 for a in outcome.attempts if not a["ok"])
        self._emit({"t": "pi05_request", "step": step, "seed": seed, "ok": outcome.ok,
                    "attempts": [dict(a) for a in outcome.attempts]})
        return outcome

    def _pi05_actions(self, outcome: RemoteOutcome) -> np.ndarray:
        actions = np.asarray(outcome.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] < self.segment or actions.shape[1] != 7:
            raise ValueError(f"pi0.5 returned {actions.shape}")
        return actions[:self.segment]

    # ------------------------------------------------------------------ decider inputs (§6) and recording (§7)

    def _reset_measurements(self) -> None:
        self.measurements: list[dict] = []        # record="full": one record per measured decision point
        self.point_meas: Optional[dict] = None    # the current decision point's measurements
        self.prev_segment: list[dict] = []        # per-step records of the last executed segment
        self.chunk_raw: Optional[np.ndarray] = None   # K x 30 x 20: the current chunk and its K-1 extra samples
        self.chunk_exec: Optional[np.ndarray] = None  # the same as executed actions, K x 30 x 7
        self._chunk_new = False                   # chunk samples drawn since the last measured point
        self._in_branch = False

    def _needs(self) -> frozenset:
        """Input groups to measure: everything when recording, else what the run's own decider takes."""
        st = self.state
        if st is None or st.mode != "routed" or self._in_branch or self.measure_client is None:
            return frozenset()
        if st.record == "full":
            return frozenset(INPUT_GROUPS)
        return frozenset(self.cfg["observe_inputs"][st.profile])

    def _chunk_samples(self, call: str, step: int) -> None:
        """K-1 extra `/sample` chunks of the exact request just executed (§6), for the chunk-time spread."""
        needs = self._needs()
        # After an abandoned escalation the fresh chunk's next points are mid points: only the chunk spread uses them.
        wanted = CHUNK_SAMPLE_GROUPS if call == "act" else {"chunk_spread"}
        if not (wanted & needs) or self.budget_remaining < self.segment:
            return
        kind = "sample" if call == "act" else "sample_fresh"
        seeds = [seeding.call_seed(self.state.rollout_seed, step, "xvla", kind, k) for k in range(1, self.K)]
        resp = self.measure_client.sample(self.xvla.last_query_payload(), seeds)
        raw = [_float32_exact(self.last_act_response["action"])] + [_float32_exact(c) for c in resp["actions"]]
        if len(raw) != self.K:
            raise RuntimeError(f"/sample returned {len(raw) - 1} chunks for {self.K - 1} seeds")
        self.chunk_raw = np.stack(raw)
        self.chunk_exec = np.stack([self.xvla.executed_rows(c) for c in self.chunk_raw])
        self._chunk_new = True

    def _robot_state(self) -> np.ndarray:
        """[controller end-effector position (3), its rotation in X-VLA's 6D form (6), finger positions (2)]."""
        ee_pos, ee_mat = self.sim.controller_pose()
        r6 = self.xvla.client.processor.Mat_to_Rotate6D(np.array(ee_mat))
        return np.concatenate([np.asarray(ee_pos, dtype=np.float64), np.asarray(r6, dtype=np.float64),
                               np.asarray(self.sim.obs["robot0_gripper_qpos"], dtype=np.float64)])

    def _measure_point(self) -> None:
        st, needs, step = self.state, self._needs(), self.sim.steps
        pos = self.chunk_len - self.xvla.queue_len()
        proposal = self.xvla.peek(self.segment)
        m = {"index": st.decision_index, "step": step, "kind": st.point_kind, "chunk_pos": pos,
             "prev_segment_pi05": st.prev_driver == "pi05", "robot_state": self._robot_state(),
             "proposal": proposal, "prev_segment": list(self.prev_segment), "features_main": None,
             "features_wrist": None, "chunk_samples_raw": None, "probe_raw": None, "chunk_spread": None,
             "fresh_spread": None, "drift": None}
        if self._chunk_new:
            m["chunk_samples_raw"], self._chunk_new = self.chunk_raw, False
        fresh = None
        if st.point_kind == "chunk":
            feats = self.last_act_response["features"]               # /act returns them (same encoder pass)
            if self.chunk_exec is not None:
                fresh = self.chunk_exec[:, :self.segment]
        else:
            payload = self.xvla.measurement_payload(self.sim, self.goal)
            if {"fresh_spread", "drift"} & needs:
                seeds = [seeding.call_seed(st.rollout_seed, step, "xvla", "probe", k) for k in range(self.K)]
                resp = self.measure_client.sample(payload, seeds)
                m["probe_raw"] = np.stack([_float32_exact(c) for c in resp["actions"]])
                fresh = np.stack([self.xvla.executed_rows(c) for c in m["probe_raw"]])[:, :self.segment]
            else:
                resp = self.measure_client.features(payload)
            feats = resp["features"]
        m["features_main"] = np.asarray(feats["enc_mean"], dtype=np.float32)
        m["features_wrist"] = np.asarray(feats["wrist_mean"], dtype=np.float32)
        if self.chunk_exec is not None:
            m["chunk_spread"] = dataclasses.asdict(meas.spread(self.chunk_exec[:, pos:pos + self.segment]))
        if fresh is not None:
            m["fresh_spread"] = dataclasses.asdict(meas.spread(fresh))
            m["drift"] = dataclasses.asdict(meas.drift(proposal, fresh))
        self.point_meas = m
        if st.record == "full":
            self.measurements.append(m)

    def observe(self, profile: Optional[str] = None) -> DeciderInputs:
        """The decider inputs at the current decision point (§6) for `profile` (default: the run's own). Another
        profile's view needs a recording run. `time_remaining_s` is always the run's own (training recomputes it
        per profile from the event log, amendment 23)."""
        st = self.state
        point = self.current_point()
        if point is None:
            raise RuntimeError("observe() needs a routed episode at a decision point")
        profile = st.profile if profile is None else profile
        if profile != st.profile and st.record != "full":
            raise ValueError("observing for another profile needs record='full'")
        groups = tuple(self.cfg["observe_inputs"][profile])
        if not groups:
            raise ValueError(f"profile {profile!r} has no decider inputs")
        if not point.can_escalate:
            raise RuntimeError("no decision is offered here: the pi0.5 budget is spent (amendment 15)")
        m = self.point_meas
        if m is None or m["index"] != point.index:
            raise RuntimeError("no measurements at this decision point (no measurement client?)")
        missing = [g for g in ("chunk_spread", "fresh_spread", "drift") if g in groups and m[g] is None]
        if missing:
            raise RuntimeError(f"not measured at this point: {missing}")
        common = CommonInputs(features_main=m["features_main"], features_wrist=m["features_wrist"],
                              robot_state=m["robot_state"], budget_remaining_steps=point.budget_remaining_steps,
                              time_remaining_s=self.time_remaining_s(), steps_elapsed=point.step)
        return DeciderInputs(
            profile=profile, point=point, common=common,
            proposal=Proposal(m["proposal"], m["chunk_pos"], m["prev_segment_pi05"]) if "proposal" in groups else None,
            free_signals=meas.free_signals(m["prev_segment"], *self.gripper_thresholds)
            if "free_signals" in groups else None,
            chunk_spread=SpreadSummary(**m["chunk_spread"]) if "chunk_spread" in groups else None,
            fresh_spread=SpreadSummary(**m["fresh_spread"]) if "fresh_spread" in groups else None,
            drift=Drift(**m["drift"]) if "drift" in groups else None)

    # ------------------------------------------------------------------ execution

    def _execute_from_queue(self) -> int:
        if self.xvla.queue_len() < self.segment:
            raise RuntimeError(f"X-VLA has {self.xvla.queue_len()} queued actions for a {self.segment}-step segment")
        return self._execute([None] * self.segment, "xvla")

    def _execute(self, actions: list, driver: str) -> int:
        """Run up to len(actions) steps (None = pop from X-VLA's queue at that moment). Each step is charged to
        every clock as it runs; the log gets one `segment` event for the run, then the end event if any."""
        expected = driver == "pi05"
        if self.sim.control_mode() != ("delta" if expected else "absolute"):
            raise RuntimeError(f"{driver} segment in the wrong controller mode")
        start, n, reason, seg = self.sim.steps, 0, None, []
        for a in actions:
            if self.sim.steps >= self.step_limit:
                break
            act = self.xvla.pop() if a is None else a
            done = self.sim.step(act)
            n += 1
            for c in self.clocks.values():
                c.step()
            self._record(act, driver)
            ee_pos, ee_mat = self.sim.controller_pose()
            seg.append({"driver": driver, "action": np.array(act, dtype=np.float32), "ee_pos": np.array(ee_pos),
                        "ee_ori_mat": np.array(ee_mat), "gripper_qpos": np.array(self.sim.obs["robot0_gripper_qpos"])})
            if self.on_step is not None:
                self.on_step(self, act, driver)
            if self._over_time():                 # a success after the limit is not a success (§5)
                reason = "time_limit"
                break
            if done:
                reason = "success"
                break
        if n:
            self.state.prev_driver = driver
            self.prev_segment = seg
            self._emit({"t": clk.STEP_RUN, "driver": driver, "start": start, "steps": n})
        if reason is not None:
            self._end(reason)
        else:
            self._check_end()
        return n

    def _record(self, action: np.ndarray, driver: str) -> None:
        obs = self.sim.obs
        self.trajectory.append({"step": self.sim.steps, "driver": driver, "action": np.array(action, dtype=np.float32),
                                **{k: np.array(obs[k]) for k in TRAJECTORY_KEYS}})

    # ------------------------------------------------------------------ ending, the clock and the log

    def _over_time(self) -> bool:
        lim = self.state.time_limit_s
        return lim is not None and self.clocks[self.state.profile].t > lim

    def _cut_if_over_time(self) -> bool:
        """After a model call: end the episode if the live limit is exceeded. True if the episode has ended."""
        if not self.done and self._over_time():
            self._end("time_limit")
        return self.done

    def _check_end(self) -> bool:
        if self.state.end_reason is None and self.sim.steps >= self.step_limit:
            self._end("step_limit")
        return self.state.end_reason is not None

    def _end(self, reason: str) -> None:
        st = self.state
        if reason not in END_REASONS:
            raise ValueError(f"end reason must be one of {END_REASONS}")
        if st.end_reason is not None:
            raise RuntimeError(f"episode already ended ({st.end_reason})")
        st.end_reason = reason
        if reason == "success":
            st.success_step = self.sim.steps
        self._emit({"t": "end", "step": self.sim.steps, "reason": reason})
        self.verify_clocks()

    def verify_clocks(self) -> None:
        """Every live clock must equal the offline replay of the log, bit for bit (§11 step 3)."""
        for p, c in self.clocks.items():
            offline = clk.replay(self.events, p, self.charges)
            if offline != c.t:
                raise ClockMismatch(f"profile {p}: live clock {c.t!r} != offline replay {offline!r}")

    def _emit(self, event: dict) -> None:
        self.events.append(event)
        if event["t"] != clk.STEP_RUN:            # steps were charged one at a time as they ran
            for c in self.clocks.values():
                c.event(event)
        if self.on_event is not None:
            self.on_event(event)

    # ------------------------------------------------------------------ results, snapshots, branches

    def result(self) -> EpisodeResult:
        st = self.state
        return EpisodeResult(episode_id=st.episode_id, success=st.end_reason == "success",
                             end_reason=st.end_reason or "step_limit", steps=self.sim.steps,
                             success_step=st.success_step, pi05_steps=st.pi05_steps, pi05_requests=st.pi05_requests,
                             pi05_failed_requests=st.pi05_failed_requests,
                             sim_time_s=self.clocks[st.profile].t if self.clocks else float("nan"), record_id=None,
                             time_limit=st.time_limit)

    def snapshot(self, header: Optional[dict] = None):
        """Snapshot of the current decision point (after its calls), §7."""
        from . import snapshot as snap
        if self.state.mode != "routed" or self.done:
            raise RuntimeError("snapshots are taken at decision points of routed episodes")
        return snap.take(self.sim, self.xvla, self.state.as_dict(), event_log=list(self.events), header=header,
                         measurements={"point": self.point_meas, "prev_segment": list(self.prev_segment),
                                       "chunk_raw": self.chunk_raw})

    def restore(self, snapshot, snapshot_id: Optional[str] = None) -> Optional[DecisionPoint]:
        """Continue from a snapshot (any env of the same task). Returns the restored decision point. The clocks
        are rebuilt by replaying the snapshot's log, which equals the live clocks at capture (verify_clocks)."""
        from . import snapshot as snap
        fields = snap.restore(snapshot, self.sim, self.xvla)
        self.state = EpisodeState(**fields)
        if self.state.time_limit != (self.state.time_limit_s is not None):
            raise ValueError("snapshot's episode fields: time_limit and time_limit_s disagree (amendment 20)")
        self.events = list(snapshot.event_log)
        self.log_offset = len(self.events)
        self.trajectory, self.last_act_response = [], None
        self._reset_measurements()
        saved = snapshot.measurements
        self.point_meas = saved.get("point")
        self.prev_segment = list(saved.get("prev_segment") or [])
        self.chunk_raw = saved.get("chunk_raw")
        if self.chunk_raw is not None:
            self.chunk_exec = np.stack([self.xvla.executed_rows(c) for c in self.chunk_raw])
        self.clocks = {p: clk.Clock(self.charges, p) for p in PROFILES}
        for c in self.clocks.values():
            for e in self.events:
                c.apply(e)
        self._emit({"t": "restore", "step": self.sim.steps, "index": self.state.decision_index,
                    "snapshot_id": snapshot_id})
        return self.current_point()

    def run_branch(self, snapshot, *, branch: str, pi05_calls: int, seed_index: int,
                   snapshot_id: Optional[str] = None) -> BranchRun:
        """One scripted continuation from a decision-point snapshot (§7), to the step limit with no time cut.

        The rollout seed becomes `seeding.branch_rollout_seed(parent, point, branch, seed index)`. The branch
        escalates at its first `pi05_calls` decision points (the planner's count: 0, 1, min(3, budget) or the
        budget's maximum), an abandoned escalation counting as one of them, and continues locally after them.
        Counting decisions rather than budget keeps branches the planner shares (equal counts) identical.
        """
        if branch not in BRANCHES:
            raise ValueError(f"branch must be one of {BRANCHES}")
        if seed_index < 1:
            raise ValueError("branch seed indices start at 1")
        dp = self.restore(snapshot, snapshot_id=snapshot_id)
        if dp is None:
            raise RuntimeError("the snapshot is not at a decision point of a running routed episode")
        st = self.state
        self._in_branch, st.record = True, "off"         # branches record outcomes only (§7)
        points_left = self.step_limit // self.segment - dp.index
        if not 0 <= pi05_calls <= min(dp.budget_remaining_steps // self.segment, points_left):
            raise ValueError(f"{pi05_calls} pi0.5 calls do not fit: {dp.budget_remaining_steps} budget steps and "
                             f"{points_left} decision points left")
        parent = st.rollout_seed
        st.rollout_seed = seeding.branch_rollout_seed(parent, dp.index, branch, seed_index)
        st.time_limit, st.time_limit_s = False, None    # branches run uncut; limits are applied offline (§5, §7)
        self._emit({"t": "branch", "branch": branch, "seed_index": int(seed_index), "pi05_calls": int(pi05_calls),
                    "index": dp.index, "rollout_seed": st.rollout_seed, "parent_rollout_seed": parent,
                    "time_limit": False})
        made = 0
        while dp is not None:
            if made < pi05_calls:
                made += 1
                dp = self.decide("escalate").next_point
            else:
                dp = self.decide("local").next_point
        return BranchRun(branch, int(seed_index), int(pi05_calls), st.rollout_seed, self.result(),
                         list(self.own_events), list(self.trajectory))
