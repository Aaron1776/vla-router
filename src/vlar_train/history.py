"""Extension point: inputs a decider builds from its own episode history. No feature is registered here and no
decider's `COLUMNS` uses one: which deciders get history inputs, and which inputs, is not decided.

The contract that makes a history input usable both in training and live:

- A **view** is what a decider knows at one decision point, as a plain dict with the same keys and units offline and
  live: `offline_view` builds it from a row of the points table, `live_view` from what `observe()` returns
  (`vlar_harness.api.DeciderInputs`). Groups, as in §6: `common` (`index`, `step`, `budget_remaining_steps`,
  `robot_state`: controller end-effector position 3, rotation in 6D form 6, finger positions 2), `proposal`
  (`proposal` 10 × 7, `chunk_pos`, `prev_segment_pi05`) and `free_signals` (`track_pos_err`, `track_rot_err`,
  `tracking_valid`, `grip_shut_frac`, `grip_partial_frac`). A group the decider's profile does not observe is absent
  (its keys are None), as for the router, which never sees the proposal or anything derived from it (§6).
- A **history feature** is a pure function of the episode's views so far, oldest first, the current point last. It
  is registered with its name, its width and the groups it reads (`register`). Reading only `common` makes it usable
  by the router; anything else is a cascader input (`router_safe`).
- An **`EpisodeHistory`** holds one episode's views. Training walks each parent run's measured points in order and
  pushes them (`data.build(..., history=[...])`); a live decider creates one per episode and pushes each
  `live_view(observe())` before it decides. Both get the same columns, `hist_<name>`, from the same code.

Two things a feature must respect, both because of what a live decider can see:

- Views exist only at decision points, every 10 steps. Per-step traces (the steps table, `prev_ee_pos`) are not in
  `observe()`, so a feature that needs them cannot be computed live with the frozen harness. Motion over the last
  30-60 steps (path length, net displacement, rotation change) is available at 10-step resolution from
  `robot_state` of the last views; a commanded step for normalising the tracking error from the previous views'
  proposals.
- `observe()` is not offered once the π0.5 budget is spent (amendment 15), so a live history has no view at those
  points. `episode_views` leaves the same points out offline. Features should measure time by `step`, not by
  counting views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

GROUPS = ("common", "proposal", "free_signals")
PREFIX = "hist_"
SEGMENT = 10                # steps between decision points (§4)


@dataclass(frozen=True)
class HistoryFeature:
    name: str
    width: int
    uses: tuple[str, ...]                                   # the view groups it reads
    fn: Callable[[Sequence[Mapping[str, Any]]], Any]        # views so far (current last) -> `width` numbers


FEATURES: dict[str, HistoryFeature] = {}


def register(name: str, *, width: int, uses: Sequence[str]):
    """Decorator: register `fn(views) -> width numbers` as history feature `name` (column `hist_<name>`)."""
    if not name.isidentifier() or name in FEATURES:
        raise ValueError(f"history feature name {name!r} is not an identifier or is already registered")
    if width < 1 or not uses or not set(uses) <= set(GROUPS):
        raise ValueError(f"history feature {name}: width >= 1 and uses a non-empty subset of {GROUPS}")

    def deco(fn):
        FEATURES[name] = HistoryFeature(name, int(width), tuple(g for g in GROUPS if g in uses), fn)
        return fn
    return deco


def column(name: str) -> str:
    return PREFIX + name


def router_safe(name: str) -> bool:
    """Whether the feature reads the common inputs only, so that a router may use it (§6)."""
    return FEATURES[name].uses == ("common",)


def offline_view(row: Mapping[str, Any], *, thresholds: tuple[float, float],
                 groups: Sequence[str] = GROUPS) -> dict[str, Any]:
    """A view from a points-table row (gripper fractions computed with the frozen thresholds, as
    `data.point_columns` does). `groups`: those the decider's profile observes; the others' keys are None."""
    v = _empty()
    v.update(index=int(row["index"]), step=int(row["step"]),
             budget_remaining_steps=int(row["budget_remaining_steps"]),
             robot_state=np.asarray(row["robot_state"], dtype=np.float64).reshape(11))
    if "proposal" in groups:
        v.update(proposal=np.asarray(row["proposal"], dtype=np.float64).reshape(10, 7),
                 chunk_pos=int(row["chunk_pos"]), prev_segment_pi05=bool(row["prev_segment_pi05"]))
    if "free_signals" in groups:
        from vlar_harness.measure import gripper_fractions_of_point
        shut, partial = gripper_fractions_of_point(dict(row), *thresholds)
        valid = bool(row.get("tracking_valid"))
        v.update(track_pos_err=float(row["track_pos_err"]) if valid else 0.0,
                 track_rot_err=float(row["track_rot_err"]) if valid else 0.0, tracking_valid=valid,
                 grip_shut_frac=float(shut), grip_partial_frac=float(partial))
    return v


def live_view(inputs: Any) -> dict[str, Any]:
    """A view from what `observe()` returns (`vlar_harness.api.DeciderInputs`); the groups the profile does not
    observe stay None."""
    v = _empty()
    v.update(index=int(inputs.point.index), step=int(inputs.point.step),
             budget_remaining_steps=int(inputs.common.budget_remaining_steps),
             robot_state=np.asarray(inputs.common.robot_state, dtype=np.float64).reshape(11))
    if inputs.proposal is not None:
        v.update(proposal=np.asarray(inputs.proposal.actions, dtype=np.float64).reshape(10, 7),
                 chunk_pos=int(inputs.proposal.chunk_pos), prev_segment_pi05=bool(inputs.proposal.prev_segment_pi05))
    if inputs.free_signals is not None:
        f = inputs.free_signals
        v.update(track_pos_err=float(f.track_pos_err), track_rot_err=float(f.track_rot_err),
                 tracking_valid=bool(f.tracking_valid), grip_shut_frac=float(f.grip_closed_shut_frac),
                 grip_partial_frac=float(f.grip_closed_partial_frac))
    return v


def _empty() -> dict[str, Any]:
    return dict.fromkeys(("index", "step", "budget_remaining_steps", "robot_state", "proposal", "chunk_pos",
                          "prev_segment_pi05", "track_pos_err", "track_rot_err", "tracking_valid", "grip_shut_frac",
                          "grip_partial_frac"))


def groups_of(view: Mapping[str, Any]) -> tuple[str, ...]:
    present = {"common": True, "proposal": view.get("proposal") is not None,
               "free_signals": view.get("tracking_valid") is not None}
    return tuple(g for g in GROUPS if present[g])


def episode_views(rows: Iterable[Mapping[str, Any]], *, thresholds: tuple[float, float],
                  groups: Sequence[str] = GROUPS) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
    """One parent run's points-table rows as (row, view), in decision-point order, without the points a live decider
    is not consulted at (π0.5 budget spent: `observe()` is not offered there, amendment 15)."""
    rows = sorted(rows, key=lambda r: int(r["index"]))
    return [(r, offline_view(r, thresholds=thresholds, groups=groups)) for r in rows
            if int(r["budget_remaining_steps"]) >= SEGMENT]


class EpisodeHistory:
    """One episode's views and the history columns at each. The same object serves training (`data.build`) and a
    live decider (one per episode: `cols = history.push(live_view(episode.observe()))`)."""

    def __init__(self, names: Optional[Sequence[str]] = None):
        self.names = tuple(FEATURES) if names is None else tuple(names)
        unknown = [n for n in self.names if n not in FEATURES]
        if unknown:
            raise KeyError(f"history features not registered: {unknown} (registered: {sorted(FEATURES)})")
        self.views: list[Mapping[str, Any]] = []

    def push(self, view: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Add the current decision point's view; returns {`hist_<name>`: (width,) float64} for it."""
        if self.views and int(view["index"]) <= int(self.views[-1]["index"]):
            raise ValueError(f"views come in decision-point order: {view['index']} after {self.views[-1]['index']}")
        have = groups_of(view)
        self.views.append(view)
        out = {}
        for n in self.names:
            f = FEATURES[n]
            missing = [g for g in f.uses if g not in have]
            if missing:
                raise ValueError(f"history feature {n} reads {missing}, which this decider's profile does not observe")
            a = np.asarray(f.fn(tuple(self.views)), dtype=np.float64).ravel()
            if a.shape != (f.width,) or not np.isfinite(a).all():
                raise ValueError(f"history feature {n} must return {f.width} finite numbers, got {a!r}")
            out[column(n)] = a
        return out
