"""Planned analysis: memory (no, yes) by X-VLA information (no, yes), as four deciders on the same rows.

    router              the common inputs (§6), as specified
    router_hist         router + history built from the common inputs alone (memory)
    cascader_nospread   the common inputs, the proposal and the free signals (§6), as specified
    cascader_hist       cascader_nospread + the same memory + history that reads X-VLA's commands (intent)

The router's and `cascader_nospread`'s profiles charge the same overhead, so the four train on the same rows, labels,
weights and remaining time and differ only in their input columns. `cascader_hist` minus `router_hist` is what X-VLA's
plan adds given memory; `router_hist` minus `router` is what memory adds.

Nothing is registered at import: `install()` registers the history features (`history.register`) and the two new
variants (`deciders.add_variant`). The four deciders of §8 are never changed. Which variants are evaluated live, and
whether the router keeps history, is not decided here.

Every feature is a pure function of the views so far (`history`), at the 10-step resolution of decision points, so it
is the same offline and live. A feature is on the memory side when it reads the `common` group only, and on the
X-VLA side when it takes one of X-VLA's commands as an argument: it would change if the commands changed while the
robot's observed trajectory stayed the same.

Screened on 500 X-VLA base rollouts against "the episode fails" (held out by task, compared at the same step),
before any Phase 1 label existed. Kept on the X-VLA side: the progress ratio, the two kinds of stall and the vertical
residual, each of which added to the router with memory, and the commanded-gripper history, which added beyond
what the finger widths alone show. Dropped, because they added nothing there: the
end-of-segment residual, the distance to earlier targets (repetition) and the reversal cosine. The plan-phase feature
was not screened: it is a hypothesis about the benefit of a call, which that label cannot test.

Conventions. Position in metres, rotation angles in radians, steps as counts. The previous segment is "X-VLA's as
proposed" when the previous view is exactly one segment back and the current view's tracking is valid (X-VLA drove
it: not episode start, not a π0.5 segment). Otherwise the features of that segment take a neutral value: 0 for
distances, 1 for ratios and cosines. Known limit: after an abandoned escalation X-VLA drives a fresh chunk, not the
proposal the decider saw, and nothing in `observe()` says so; that segment is treated as proposed.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from . import deciders as DC
from . import history as H
from .data import PointSet

EPS_M = 1e-3                 # 1 mm floor in ratios
TINY_M = 2e-3                # below 2 mm a direction is undefined

MEMORY = ("motion", "fingers")                                   # read `common` only: the router may use them
INTENT = ("track_norm", "progress", "stall", "vertical", "grip_events")
PLAN = ("plan_phase",)
ROUTER_HIST = MEMORY
CASCADER_HIST = MEMORY + INTENT + PLAN
HISTORY = CASCADER_HIST                                          # every feature `data.build(history=...)` needs
TWO_BY_TWO = ("router", "router_hist", "cascader_nospread", "cascader_hist")
WIDTHS = {"motion": 10, "fingers": 3, "track_norm": 4, "progress": 4, "stall": 4, "vertical": 2, "grip_events": 7,
          "plan_phase": 6}


# ---------------------------------------------------------------- helpers on views

def _pos(v: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(v["robot_state"], dtype=np.float64)[:3]


def _rot(v: Mapping[str, Any]) -> np.ndarray:
    """Rotation matrix from the robot state's 6D form: its first two columns, column by column."""
    r6 = np.asarray(v["robot_state"], dtype=np.float64)[3:9]
    a, b = r6[:3], r6[3:6]
    a = a / np.linalg.norm(a)
    b = b - a * (a @ b)
    b = b / np.linalg.norm(b)
    return np.stack([a, b, np.cross(a, b)], axis=1)


def _width(v: Mapping[str, Any]) -> float:
    rs = np.asarray(v["robot_state"], dtype=np.float64)
    return float(rs[9] - rs[10])


def _angle(Ra, Rb) -> float:
    from vlar_harness.measure import rotation_angle
    return float(rotation_angle(np.asarray(Ra), np.asarray(Rb)))


def _target_rot(row) -> np.ndarray:
    from vlar_harness.measure import axis_angle_to_mat
    return axis_angle_to_mat(np.asarray(row, dtype=np.float64)[3:6])


def _window(views: Sequence[Mapping[str, Any]], steps: int) -> list:
    cur = int(views[-1]["step"])
    return [v for v in views if cur - int(v["step"]) <= steps]


def _segments(views: Sequence[Mapping[str, Any]], steps: int) -> list[tuple[Mapping, Mapping]]:
    """(a, b) pairs of consecutive views within the last `steps` steps whose segment was X-VLA's as proposed at a."""
    w = _window(views, steps)
    return [(a, b) for a, b in zip(w[:-1], w[1:])
            if int(b["step"]) - int(a["step"]) == H.SEGMENT and b.get("tracking_valid") and a.get("proposal") is not None]


def _last_segment(views) -> Optional[tuple[Mapping, Mapping]]:
    s = _segments(views, H.SEGMENT)
    return s[-1] if s else None


def _end(v: Mapping[str, Any]) -> np.ndarray:
    """The last target position of the view's proposal."""
    return np.asarray(v["proposal"], dtype=np.float64)[-1, :3]


def _cos(u: np.ndarray, v: np.ndarray, neutral: float = 1.0) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    return float(u @ v / (nu * nv)) if nu > TINY_M and nv > TINY_M else neutral


def _cmd_act(views, steps: int) -> tuple[float, float]:
    """Commanded and actual net displacement, summed over the proposed segments of the last `steps` steps."""
    cmd = act = 0.0
    for a, b in _segments(views, steps):
        cmd += float(np.linalg.norm(_end(a) - _pos(a)))
        act += float(np.linalg.norm(_pos(b) - _pos(a)))
    return cmd, act


# ---------------------------------------------------------------- the features

def motion(views) -> list[float]:
    """Memory. Over the last 30 and the last 60 steps: net displacement, path length, straightness, rotation change
    and finger-width change of the end effector, from the robot state at the decision points."""
    out = []
    for steps in (30, 60):
        w = _window(views, steps)
        pos = np.array([_pos(v) for v in w])
        disp = float(np.linalg.norm(pos[-1] - pos[0]))
        path = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if len(w) > 1 else 0.0
        out += [disp, path, disp / (path + 1e-4), _angle(_rot(w[-1]), _rot(w[0])), abs(_width(w[-1]) - _width(w[0]))]
    return out


def fingers(views) -> list[float]:
    """Memory. Finger width at the previous decision point, and its minimum over the last 30 and the last 60 steps
    (fingers that shut on nothing, or opened, show in the robot state without the command)."""
    w30, w60 = _window(views, 30), _window(views, 60)
    prev = views[-2] if len(views) > 1 else views[-1]
    return [_width(prev), min(_width(v) for v in w30), min(_width(v) for v in w60)]


def track_norm(views) -> list[float]:
    """X-VLA. The free signals' tracking error divided by the commanded step of the previous segment (the mean
    distance between consecutive targets, from the previous pose through the previous proposal), and that step."""
    s = _last_segment(views)
    if s is None:
        return [0.0, 0.0, 0.0, 0.0]
    a, b = s
    prop = np.asarray(a["proposal"], dtype=np.float64)
    path = np.vstack([_pos(a), prop[:, :3]])
    cmd_pos = float(np.linalg.norm(np.diff(path, axis=0), axis=1).mean())
    rots = [_rot(a)] + [_target_rot(r) for r in prop]
    cmd_rot = float(np.mean([_angle(rots[k + 1], rots[k]) for k in range(len(prop))]))
    return [float(b["track_pos_err"]) / max(cmd_pos, 1e-3), float(b["track_rot_err"]) / max(cmd_rot, 1e-3),
            cmd_pos, cmd_rot]


def progress(views) -> list[float]:
    """X-VLA. Actual over commanded net displacement over the last 10, 30 and 60 steps (clipped to [0, 3]), and the
    cosine between the commanded and the actual displacement of the last segment."""
    out = []
    for steps in (10, 30, 60):
        cmd, act = _cmd_act(views, steps)
        out.append(float(np.clip(act / (cmd + EPS_M), 0.0, 3.0)) if _segments(views, steps) else 1.0)
    s = _last_segment(views)
    out.append(_cos(_end(s[0]) - _pos(s[0]), _pos(s[1]) - _pos(s[0])) if s else 1.0)
    return out


def stall(views) -> list[float]:
    """X-VLA. Over the last 30 and 60 steps: the commanded net displacement (small: X-VLA is hesitating) and the
    shortfall of the actual one against it (large: the arm is blocked)."""
    (c30, a30), (c60, a60) = _cmd_act(views, 30), _cmd_act(views, 60)
    return [c30, c60, max(c30 - a30, 0.0), max(c60 - a60, 0.0)]


def vertical(views) -> list[float]:
    """X-VLA. Commanded minus actual height at the end of the last segment (signed), and its mean over the last 30
    steps (positive: the arm stays below where it was sent; negative: above)."""
    dz = [float(_end(a)[2] - _pos(b)[2]) for a, b in _segments(views, 30)]
    s = _last_segment(views)
    return [float(_end(s[0])[2] - _pos(s[1])[2]) if s else 0.0, float(np.sum(dz)) / 3.0]


def grip_events(views) -> list[float]:
    """X-VLA. The gripper's command history against what the fingers did, over the episode's proposed segments:
    close commands so far, open commands after a close, closes during which the fingers shut at some point (failed; a
    drop counts), closes that held something and then shut (drops), closes that follow a failed one (regrasps), whether anything
    was ever held, and the steps since something was last held (the steps elapsed when nothing was). A segment's
    fingers "shut" or "held" when its gripper-mismatch fraction of that kind is above zero; "held" for the last
    two needs half the segment."""
    G, S, P, held = [], [], [], None
    for a, b in _segments(views, 10 ** 9):
        closed = np.asarray(a["proposal"], dtype=np.float64)[:, 6] > 0
        G += [bool(c) for c in closed]
        S += [float(b["grip_shut_frac"]) > 0] * len(closed)
        P += [float(b["grip_partial_frac"]) > 0] * len(closed)
        if float(b["grip_partial_frac"]) >= 0.5:
            held = int(b["step"])
    G, S, P = (np.asarray(x, dtype=bool) for x in (G, S, P))
    before = np.r_[False, G[:-1]] if len(G) else G
    onsets, offs = np.flatnonzero(G & ~before), np.flatnonzero(~G & before)
    failed = drops = regrasps = 0
    last_failed = False
    for o in onsets:
        end = int(offs[offs > o][0]) if np.any(offs > o) else len(G)
        regrasps += last_failed
        shut, part = S[o:end] & G[o:end], P[o:end] & G[o:end]
        last_failed = bool(shut.any())
        failed += last_failed
        drops += bool(part.any() and shut.any() and np.argmax(shut) > np.argmax(part))
    step = int(views[-1]["step"])
    return [float(len(onsets)), float(len(offs)), float(failed), float(drops), float(regrasps),
            float(held is not None), float(step - held) if held is not None else float(step)]


def plan_phase(views) -> list[float]:
    """X-VLA, forward-looking: what the next 10 steps are for. Gripper command changes in the proposal (counting a
    change against the last command of the previous proposal, when that one was executed), the step of the first
    change (as a fraction of the segment; 1 when none), the share of steps commanded closed, that share times the
    finger width now (closed on something), the commanded mean step, and the commanded height change."""
    v = views[-1]
    prop = np.asarray(v["proposal"], dtype=np.float64)
    closed = prop[:, 6] > 0
    s = _last_segment(views)
    before = [bool(np.asarray(s[0]["proposal"])[-1, 6] > 0)] if s else []
    seq = np.array(before + list(closed), dtype=bool)
    change = np.flatnonzero(seq[1:] != seq[:-1])
    first = (int(change[0]) + (0 if before else 1)) / float(len(closed)) if len(change) else 1.0
    path = np.vstack([_pos(v), prop[:, :3]])
    return [float(len(change)), float(first), float(closed.mean()), float(closed.mean()) * _width(v),
            float(np.linalg.norm(np.diff(path, axis=0), axis=1).mean()), float(prop[-1, 2] - _pos(v)[2])]


_FEATURES = {"motion": (motion, ("common",)),
             "fingers": (fingers, ("common",)),
             "track_norm": (track_norm, ("common", "proposal", "free_signals")),
             "progress": (progress, ("common", "proposal", "free_signals")),
             "stall": (stall, ("common", "proposal", "free_signals")),
             "vertical": (vertical, ("common", "proposal", "free_signals")),
             "grip_events": (grip_events, ("common", "proposal", "free_signals")),
             "plan_phase": (plan_phase, ("common", "proposal", "free_signals"))}


# ---------------------------------------------------------------- registration

def columns(decider: str) -> tuple[str, ...]:
    """The input columns of a 2 x 2 variant."""
    if decider == "router_hist":
        return DC.COLUMNS["router"] + tuple(H.column(n) for n in ROUTER_HIST)
    if decider == "cascader_hist":
        return DC.COLUMNS["cascader_nospread"] + tuple(H.column(n) for n in CASCADER_HIST)
    return DC.COLUMNS[decider]


def install() -> None:
    """Register the history features and the two variants. Safe to call twice. The four deciders of §8, their
    columns and their profiles are untouched."""
    for name, (fn, uses) in _FEATURES.items():
        if name not in H.FEATURES:
            H.register(name, width=WIDTHS[name], uses=uses)(fn)
    for name, profile in (("router_hist", "router"), ("cascader_hist", "cascader_nospread")):
        if name not in DC.COLUMNS:
            DC.add_variant(name, profile=profile, columns=columns(name))


def with_synthetic_history(ps: PointSet, seed: int = 0) -> PointSet:
    """`ps` plus random history columns of the right widths, for dry runs on `data.synthetic` (which has no episodes
    to walk). Shapes only: the values mean nothing."""
    rng = np.random.default_rng([seed, 2022])
    cols = dict(ps.cols)
    for name in HISTORY:
        cols[H.column(name)] = rng.normal(size=(len(ps), WIDTHS[name]))
    return PointSet(cols, ps.task, ps.snapshot_id, ps.can_escalate, ps.remaining_s, ps.y, ps.excluded, ps.weight,
                    ps.episode, ps.report)
