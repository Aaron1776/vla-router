"""Client API of the harness (docs/SPEC.md §15). Draft until the Phase 0 freeze.

Deciders, data collection and evaluation import only this module. The harness enforces
the step limit, time limit, pi0.5 budget, controller switching, handoff rules, clock
charges and network model behind it, so a buggy caller cannot bypass them.

Arrays are numpy arrays; they are typed as `Any` so this module imports without numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol

Mode = Literal["routed", "xvla_only", "pi05_only"]
Profile = Literal["none", "router", "cascader_nospread", "cascader_chunkspread", "cascader_full"]
Record = Literal["off", "full"]
Choice = Literal["local", "escalate"]
PointKind = Literal["chunk", "mid"]
EndReason = Literal["success", "step_limit", "time_limit"]
Array = Any


@dataclass(frozen=True)
class DecisionPoint:
    index: int                     # 0..51; decision points are at steps 0, 10, ..., 510
    step: int                      # environment steps executed so far
    kind: PointKind                # chunk: X-VLA's queue is empty; mid: it still holds actions
    budget_remaining_steps: int
    can_escalate: bool             # False once the pi0.5 budget is spent


@dataclass(frozen=True)
class CommonInputs:
    """Inputs every decider gets (§6). No task id or task text, by design."""
    features: Array                # pooled X-VLA encoder features of the current observation
    robot_state: Array             # measured robot state; layout fixed by the adapters and the datacard
    budget_remaining_steps: int
    time_remaining_s: float        # under the decider's own profile; inf in scripted runs
    steps_elapsed: int


@dataclass(frozen=True)
class Proposal:
    actions: Array                 # the 10 X-VLA actions that execute if the decider continues
    chunk_pos: int                 # 0, 10 or 20
    prev_segment_pi05: bool


@dataclass(frozen=True)
class FreeSignals:
    track_pos_err: float           # mean over the previous segment of |measured - planned| position
    track_rot_err: float           # same for rotation (radians)
    tracking_valid: bool           # False (errors 0) at episode start or after a pi0.5 segment
    grip_closed_shut_frac: float   # "closed" command with fingers fully shut: grasp missed or dropped
    grip_closed_partial_frac: float  # "closed" command with fingers partly open: object held


@dataclass(frozen=True)
class SpreadSummary:
    pos_std_mean: float
    pos_std_max: float
    rot_dev_mean: float
    grip_disagree_frac: float


@dataclass(frozen=True)
class Drift:
    pos_dist_mean: float
    rot_dist_mean: float
    grip_disagree_frac: float


@dataclass(frozen=True)
class DeciderInputs:
    """What `observe()` returns; which groups are filled is set by the profile (config observe_inputs)."""
    profile: Profile
    point: DecisionPoint
    common: CommonInputs
    proposal: Optional[Proposal] = None
    free_signals: Optional[FreeSignals] = None
    chunk_spread: Optional[SpreadSummary] = None
    fresh_spread: Optional[SpreadSummary] = None
    drift: Optional[Drift] = None


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    success: bool                  # task done before the step limit and, with a live clock, the time limit
    end_reason: EndReason
    steps: int
    success_step: Optional[int]
    pi05_steps: int
    pi05_requests: int
    pi05_failed_requests: int
    sim_time_s: float              # live clock under the run's profile
    record_id: Optional[str]       # where the event log and trajectory were written (record=full)


@dataclass(frozen=True)
class SegmentOutcome:
    executed_by: Literal["xvla", "pi05"]
    escalation_abandoned: bool     # every pi0.5 attempt failed, so X-VLA drove this segment
    pi05_requests: int
    pi05_failed_requests: int
    done: bool
    next_point: Optional[DecisionPoint]
    episode: Optional[EpisodeResult]   # set when done


@dataclass(frozen=True)
class BranchJob:
    snapshot_id: str
    branch: str                    # canonical branch type, e.g. "b2_one_call"
    pi05_calls: int                # consecutive pi0.5 calls before X-VLA drives to the end
    seed_index: int                # 1-based


@dataclass(frozen=True)
class BranchResult:
    snapshot_id: str
    branch: str
    seed_index: int
    success: bool                  # within the step limit; time limits are applied offline per profile
    success_step: Optional[int]
    pi05_steps: int
    pi05_requests: int
    pi05_failed_requests: int
    record_id: str                 # event log and per-step trajectory


class HarnessClient(Protocol):
    def reset(self, task_id: int, init_idx: int, seed: int, mode: Mode = "routed",
              profile: Profile = "none", record: Record = "off") -> Optional[DecisionPoint]:
        """Set up an episode (scene, initial state, settle steps). Routed mode returns the first point."""

    def restore(self, snapshot_id: str) -> DecisionPoint:
        """Resume from a snapshot taken at a decision point (no settle steps)."""

    def observe(self) -> DeciderInputs:
        """Routed mode only: the active profile's decider inputs at the current point."""

    def decide(self, choice: Choice) -> SegmentOutcome:
        """Run the next 10-step segment locally or via one pi0.5 call."""

    def run_episode(self) -> EpisodeResult:
        """xvla_only and pi05_only modes: run the whole episode."""

    def snapshot(self) -> str:
        """Store a snapshot of the current decision point and return its id."""

    def run_branch(self, job: BranchJob) -> BranchResult:
        """Queue workers: run one scripted continuation from a snapshot to the step limit."""
