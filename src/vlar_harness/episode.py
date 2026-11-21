"""The episode engine: decision points, the π0.5 budget, run modes and the handoff rules
(docs/SPEC.md §3-§4; effort step 3).

Run modes
- routed: a decision point every 10 steps (0, 10, ..., 510). At a chunk point (X-VLA's queue is
  empty) X-VLA's `/act` is called *before* the decision. `decide("local")` runs the next 10 queued
  X-VLA actions; `decide("escalate")` hands one 10-step segment to π0.5.
- xvla_only: X-VLA alone, exactly the authors' client loop (a chunk whenever the queue is empty).
- pi05_only: π0.5 alone, called every 10 steps, no budget, as `lerobot-eval` with n_action_steps=10.

Handoff rules (§3)
- Escalate: X-VLA's queued actions are discarded and its remembered pose forgotten, the controller
  switches to delta mode, π0.5 is called fresh, the first 10 of its 50 actions run.
- Back to X-VLA: the controller switches to absolute mode; X-VLA's next call is a chunk-point
  `/act`, which sends the measured end-effector pose because the remembered pose was forgotten.
  Within an uninterrupted X-VLA stretch the quirk applies (config option `xvla_state_mode`).
- Abandoned escalation (every attempt of the π0.5 request failed, §5): the controller returns to
  absolute mode, X-VLA is called fresh (`act_fresh`) with the measured pose and runs 10 actions;
  its other 20 stay queued; no budget is charged. Failed requests cost time, never budget.
The controller mode is set before every segment, so each executed step runs in its model's mode.

The π0.5 request goes through a `RemoteLink`. `DirectLink` always succeeds on the first attempt;
the simulated network (latency, failures, retries) plugs in here in effort step 4, as do the clock
and the full event log. Model noise seeds come from `seeding.call_seed(rollout_seed, ...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np

from . import seeding
from .api import DecisionPoint, EpisodeResult, SegmentOutcome

MODES = ("routed", "xvla_only", "pi05_only")
TRAJECTORY_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class RemoteOutcome:
    ok: bool
    actions: Optional[np.ndarray]        # 50 x 7 un-normalised π0.5 actions when ok
    attempts: list[dict] = field(default_factory=list)


class RemoteLink(Protocol):
    def request(self, call: Callable[[], np.ndarray], *, step: int, rollout_seed: int) -> RemoteOutcome: ...


class DirectLink:
    """No network: every request succeeds on its first attempt."""

    def request(self, call, *, step, rollout_seed) -> RemoteOutcome:
        return RemoteOutcome(True, np.asarray(call(), dtype=np.float32), [{"attempt": 0, "ok": True}])


@dataclass
class EpisodeState:
    """Everything about the episode beyond the simulator and the X-VLA client (goes into snapshots)."""
    task_id: int
    init_idx: int
    scene_seed: int
    rollout_seed: int
    mode: str
    episode_id: str = ""
    decision_index: int = -1
    point_kind: Optional[str] = None
    pi05_steps: int = 0
    pi05_requests: int = 0
    pi05_failed_requests: int = 0
    prev_driver: Optional[str] = None
    success_step: Optional[int] = None
    end_reason: Optional[str] = None

    def as_dict(self) -> dict:
        return dict(vars(self))


class Episode:
    """One episode on one LiberoSim with its two adapters.

    `pi05_call(raw_observation, task, seed) -> 50 x 7 actions` is the π0.5 server call (the
    PI05Client in production). `on_event(event)` and `on_step(episode, action, driver)` are hooks
    for the event log and for tests.
    """

    def __init__(self, sim, xvla, pi05, pi05_call: Callable, cfg, *, remote: Optional[RemoteLink] = None,
                 on_event: Optional[Callable[[dict], None]] = None,
                 on_step: Optional[Callable[["Episode", np.ndarray, str], None]] = None):
        self.sim, self.xvla, self.pi05 = sim, xvla, pi05
        self.pi05_call = pi05_call
        self.remote = remote or DirectLink()
        self.on_event, self.on_step = on_event, on_step
        self.step_limit = int(cfg["env"]["step_limit"])
        self.segment = int(cfg["decision"]["segment_steps"])
        self.budget_steps = int(cfg["budget"]["pi05_max_steps"])
        self.scene_offset = int(cfg.option("scene_seed"))
        self.state: Optional[EpisodeState] = None
        self.events: list[dict] = []
        self.trajectory: list[dict] = []
        self.last_act_response: Optional[dict] = None

    # ------------------------------------------------------------------ setup

    def start(self, init_idx: int, rollout_seed: int, mode: str = "routed",
              episode_id: str = "") -> Optional[DecisionPoint]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        scene = seeding.scene_seed(init_idx, self.scene_offset)
        self.sim.setup_episode(init_idx, env_seed=scene)
        self.xvla.reset()
        self.state = EpisodeState(self.sim.task_id, init_idx, scene, int(rollout_seed), mode, episode_id)
        self.events, self.trajectory, self.last_act_response = [], [], None
        self._emit({"t": "start", "task_id": self.sim.task_id, "init_idx": init_idx, "scene_seed": scene,
                    "rollout_seed": int(rollout_seed), "mode": mode})
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
        else:
            st.point_kind = "mid"
        self._emit({"t": "decision_point", "step": step, "index": st.decision_index, "kind": st.point_kind,
                    "queue": self.xvla.queue_len(), "budget_remaining": self.budget_remaining})
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
        requests_before, failed_before = st.pi05_requests, st.pi05_failed_requests
        abandoned = False
        if choice == "escalate":
            if self.budget_remaining < self.segment:
                raise BudgetExhausted(f"{self.budget_remaining} pi0.5 steps left; escalation not allowed")
            driver, abandoned = self._escalate()
        else:
            driver = self._run_xvla_segment()
        st.prev_driver = driver
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
        self.sim.set_control_mode("delta")
        raw = self.pi05.raw_observation()
        seed = seeding.call_seed(st.rollout_seed, step, "pi05", "act")
        outcome = self.remote.request(lambda: self.pi05_call(raw, self.goal, seed), step=step,
                                      rollout_seed=st.rollout_seed)
        st.pi05_requests += 1
        st.pi05_failed_requests += sum(1 for a in outcome.attempts if not a.get("ok"))
        self._emit({"t": "pi05_request", "step": step, "seed": seed, "ok": outcome.ok, "attempts": outcome.attempts})
        if outcome.ok:
            actions = np.asarray(outcome.actions, dtype=np.float32)
            if actions.shape[0] < self.segment or actions.shape[1] != 7:
                raise ValueError(f"pi0.5 returned {actions.shape}")
            if st.prev_driver != "pi05":
                self._emit({"t": "handoff", "step": step, "to": "pi05"})
            st.pi05_steps += self._execute([self.pi05.env_action(a) for a in actions[:self.segment]], "pi05")
            return "pi05", False
        # Abandoned: X-VLA, called fresh from the measured pose, drives this segment; no budget charged.
        self.sim.set_control_mode("absolute")
        self._xvla_query("act_fresh", step)
        self._emit({"t": "escalation_abandoned", "step": step})
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
                self._execute([None] * self.xvla.queue_len(), "xvla")
        elif st.mode == "pi05_only":
            self.sim.set_control_mode("delta")
            while not self._check_end():
                step = self.sim.steps
                seed = seeding.call_seed(st.rollout_seed, step, "pi05", "act")
                actions = np.asarray(self.pi05_call(self.pi05.raw_observation(), self.goal, seed), dtype=np.float32)
                st.pi05_requests += 1
                self._emit({"t": "pi05_request", "step": step, "seed": seed, "ok": True, "attempts": [{"ok": True}]})
                st.pi05_steps += self._execute([self.pi05.env_action(a) for a in actions[:self.segment]], "pi05")
        else:
            raise RuntimeError("run_episode() is for xvla_only and pi05_only; routed episodes use decide()")
        return self.result()

    # ------------------------------------------------------------------ execution

    def _xvla_query(self, call: str, step: int) -> None:
        seed = seeding.call_seed(self.state.rollout_seed, step, "xvla", call)
        self.xvla.query(self.sim, self.goal, seed=seed)
        self.last_act_response = self.xvla.client.last_response
        self._emit({"t": "xvla_act", "step": step, "call": call, "seed": seed})

    def _execute_from_queue(self) -> int:
        if self.xvla.queue_len() < self.segment:
            raise RuntimeError(f"X-VLA has {self.xvla.queue_len()} queued actions for a {self.segment}-step segment")
        return self._execute([None] * self.segment, "xvla")

    def _execute(self, actions: list, driver: str) -> int:
        """Run up to len(actions) steps (None = pop from X-VLA's queue at that moment)."""
        expected = driver == "pi05"
        if self.sim.control_mode() != ("delta" if expected else "absolute"):
            raise RuntimeError(f"{driver} segment in the wrong controller mode")
        n = 0
        for a in actions:
            if self._check_end():
                break
            act = self.xvla.pop() if a is None else a
            done = self.sim.step(act)
            n += 1
            self._record(act, driver)
            if self.on_step is not None:
                self.on_step(self, act, driver)
            if done:
                self.state.success_step = self.sim.steps
                self.state.end_reason = "success"
                self._emit({"t": "end", "step": self.sim.steps, "reason": "success"})
                break
        self._emit({"t": "segment", "driver": driver, "steps": n, "end_step": self.sim.steps})
        return n

    def _record(self, action: np.ndarray, driver: str) -> None:
        obs = self.sim.obs
        self.trajectory.append({"step": self.sim.steps, "driver": driver, "action": np.array(action, dtype=np.float32),
                                **{k: np.array(obs[k]) for k in TRAJECTORY_KEYS}})

    def _check_end(self) -> bool:
        st = self.state
        if st.end_reason is None and self.sim.steps >= self.step_limit:
            st.end_reason = "step_limit"
            self._emit({"t": "end", "step": self.sim.steps, "reason": "step_limit"})
        return st.end_reason is not None

    def _emit(self, event: dict) -> None:
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    # ------------------------------------------------------------------ results and snapshots

    def result(self) -> EpisodeResult:
        st = self.state
        return EpisodeResult(episode_id=st.episode_id, success=st.end_reason == "success",
                             end_reason=st.end_reason or "step_limit", steps=self.sim.steps,
                             success_step=st.success_step, pi05_steps=st.pi05_steps, pi05_requests=st.pi05_requests,
                             pi05_failed_requests=st.pi05_failed_requests, sim_time_s=float("nan"), record_id=None)

    def snapshot(self, header: Optional[dict] = None):
        """Snapshot of the current decision point (after its calls), §7."""
        from . import snapshot as snap
        if self.state.mode != "routed" or self.done:
            raise RuntimeError("snapshots are taken at decision points of routed episodes")
        return snap.take(self.sim, self.xvla, self.state.as_dict(), event_log=list(self.events), header=header)

    def restore(self, snapshot) -> Optional[DecisionPoint]:
        """Continue from a snapshot (any env of the same task). Returns the restored decision point."""
        from . import snapshot as snap
        fields = snap.restore(snapshot, self.sim, self.xvla)
        self.state = EpisodeState(**fields)
        self.events = list(snapshot.event_log)
        self.trajectory, self.last_act_response = [], None
        self._emit({"t": "restore", "step": self.sim.steps, "index": self.state.decision_index})
        return self.current_point()
