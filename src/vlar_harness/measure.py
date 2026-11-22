"""Decider measurements (docs/SPEC.md §6): proposal, free signals, spread and drift.

Pure functions of recorded data, so the live engine, the offline dataset builder and the tests compute
them identically. Definitions and the choices the spec leaves open (docs/HARNESS.md, "Decider inputs"):

- Action rows are in executed form: [x, y, z, ax, ay, az, g], the absolute target position, its
  axis-angle rotation and the gripper command, exactly as `XVLAAdapter.pop` returns them (float32,
  g = +1 close / -1 open). Arithmetic is done in float64.
- Rotations: axis-angle to matrix by Rodrigues' formula; the angle between two rotations is
  2 asin(|Ra - Rb|_F / (2 sqrt 2)) (exact for rotation matrices and accurate near zero); the mean of
  several rotations is the arithmetic mean of their matrices projected onto SO(3) (SVD).
- Spread over K samples of an n-step segment, per step: position std = the RMS distance of the K
  targets from their mean (population, ddof 0); rotation deviation = the mean angle of the K
  rotations from their mean rotation; gripper disagreement = min(#close, #open) / K (the fraction
  disagreeing with the majority). Summary: mean and max of the position std over steps, the mean
  rotation deviation, the mean gripper disagreement.
- Drift of the n queued actions against K fresh samples, per step: distance to the samples' mean
  position, angle to their mean rotation, and 1 if the queued gripper command differs from the
  samples' majority (0.5 on a tie, impossible for odd K); each averaged over steps.
- Free signals over the previous segment's steps. Tracking (X-VLA segments only): distance between the
  controller's cached end-effector pose after the step and the step's target (position, rotation
  angle). Gripper (either driver): a step's command is "close" when g > 0; finger width =
  qpos[0] - qpos[1]; "shut" when width <= shut_width_m, "partly open" when shut_width_m < width <
  open_width_m (Phase 0 calibrates both thresholds).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .api import Drift, FreeSignals, SpreadSummary

_ROOT8 = 2.0 * np.sqrt(2.0)


def executed_form(rows) -> np.ndarray:
    """Queue rows [pos3, axis-angle3, raw gripper] -> what `pop` executes (float32, gripper +-1)."""
    a = np.array(rows, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 7:
        raise ValueError(f"expected n x 7 queue rows, got {a.shape}")
    a[:, 6] = np.where(a[:, 6] > 0.5, np.float32(1.0), np.float32(-1.0))
    return a


def axis_angle_to_mat(aa) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrices (Rodrigues)."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1)
    safe = np.where(theta > 1e-12, theta, 1.0)
    k = aa / safe[..., None]
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack([np.stack([zero, -kz, ky], -1), np.stack([kz, zero, -kx], -1), np.stack([-ky, kx, zero], -1)], -2)
    s, c = np.sin(theta)[..., None, None], np.cos(theta)[..., None, None]
    eye = np.broadcast_to(np.eye(3), K.shape)
    R = eye + s * K + (1.0 - c) * (K @ K)
    return np.where((theta > 1e-12)[..., None, None], R, eye)


def rotation_angle(Ra, Rb) -> np.ndarray:
    """Angle (radians) between rotation matrices, broadcast over leading axes."""
    d = np.linalg.norm(np.asarray(Ra, dtype=np.float64) - np.asarray(Rb, dtype=np.float64), axis=(-2, -1))
    return 2.0 * np.arcsin(np.clip(d / _ROOT8, 0.0, 1.0))


def mean_rotation(Rs, axis: int = 0) -> np.ndarray:
    """Chordal L2 mean: the mean matrix along `axis` projected onto SO(3)."""
    M = np.mean(np.asarray(Rs, dtype=np.float64), axis=axis)
    U, _, Vt = np.linalg.svd(M)
    D = np.ones(M.shape[:-1])
    D[..., -1] = np.where(np.linalg.det(U @ Vt) < 0, -1.0, 1.0)
    return (U * D[..., None, :]) @ Vt


def _check_samples(samples) -> np.ndarray:
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 3 or s.shape[2] != 7 or s.shape[0] < 2:
        raise ValueError(f"expected K x n x 7 executed-form samples with K >= 2, got {s.shape}")
    return s


def spread(samples) -> SpreadSummary:
    """Spread summary of K samples of one n-step segment (K x n x 7, executed form)."""
    s = _check_samples(samples)
    pos = s[:, :, :3]
    pos_std = np.sqrt(np.mean(np.sum((pos - pos.mean(axis=0)) ** 2, axis=-1), axis=0))      # n
    R = axis_angle_to_mat(s[:, :, 3:6])                                                    # K x n x 3 x 3
    rot_dev = rotation_angle(R, mean_rotation(R, axis=0)[None])                            # K x n
    close = np.sum(s[:, :, 6] > 0, axis=0)                                                 # n
    disagree = np.minimum(close, s.shape[0] - close) / s.shape[0]
    return SpreadSummary(pos_std_mean=float(pos_std.mean()), pos_std_max=float(pos_std.max()),
                         rot_dev_mean=float(rot_dev.mean()), grip_disagree_frac=float(disagree.mean()))


def drift(queued, samples) -> Drift:
    """Drift of the n queued actions (n x 7) from K fresh samples of the same segment (K x n x 7)."""
    q = np.asarray(queued, dtype=np.float64)
    s = _check_samples(samples)
    if q.shape != s.shape[1:]:
        raise ValueError(f"queued {q.shape} vs samples {s.shape}")
    pos_d = np.linalg.norm(q[:, :3] - s[:, :, :3].mean(axis=0), axis=-1)
    rot_d = rotation_angle(axis_angle_to_mat(q[:, 3:6]), mean_rotation(axis_angle_to_mat(s[:, :, 3:6]), axis=0))
    close = np.sum(s[:, :, 6] > 0, axis=0)
    k = s.shape[0]
    majority_close = np.where(2 * close > k, 1.0, np.where(2 * close < k, 0.0, np.nan))
    mine = (q[:, 6] > 0).astype(np.float64)
    dis = np.where(np.isnan(majority_close), 0.5, (mine != majority_close).astype(np.float64))
    return Drift(pos_dist_mean=float(pos_d.mean()), rot_dist_mean=float(rot_d.mean()),
                 grip_disagree_frac=float(dis.mean()))


def tracking(segment: Sequence[dict]) -> tuple[float, float, bool]:
    """The tracking part of the free signals (no thresholds needed): mean position and rotation error over an X-VLA
    segment; (0, 0, False) at episode start or after a π0.5 segment."""
    if not segment:
        return 0.0, 0.0, False
    drivers = {r["driver"] for r in segment}
    if len(drivers) != 1:
        raise ValueError(f"a segment has one driver, got {sorted(drivers)}")
    if drivers != {"xvla"}:
        return 0.0, 0.0, False
    act = np.array([r["action"] for r in segment], dtype=np.float64)
    pos_err = np.linalg.norm(np.array([r["ee_pos"] for r in segment], dtype=np.float64) - act[:, :3], axis=-1)
    rot_err = rotation_angle(np.array([r["ee_ori_mat"] for r in segment], dtype=np.float64),
                             axis_angle_to_mat(act[:, 3:6]))
    return float(pos_err.mean()), float(rot_err.mean()), True


def free_signals(segment: Sequence[dict], shut_width_m: Optional[float], open_width_m: Optional[float]) -> FreeSignals:
    """Free signals from the previous segment's per-step records (`Episode.prev_segment`): each has `driver`,
    `action` (executed form), `ee_pos` and `ee_ori_mat` (the controller's cached pose after the step) and
    `gripper_qpos`. An empty segment (episode start) gives zeros with tracking flagged invalid."""
    if not segment:
        return FreeSignals(0.0, 0.0, False, 0.0, 0.0)
    if shut_width_m is None or open_width_m is None:
        raise ValueError("gripper thresholds are uncalibrated (measurement.gripper_thresholds; Phase 0 step 2)")
    if not shut_width_m < open_width_m:
        raise ValueError("gripper thresholds need shut_width_m < open_width_m")
    track = tracking(segment)
    act = np.array([r["action"] for r in segment], dtype=np.float64)
    qpos = np.array([r["gripper_qpos"] for r in segment], dtype=np.float64)
    width = qpos[:, 0] - qpos[:, 1]
    close = act[:, 6] > 0
    shut = close & (width <= shut_width_m)
    partial = close & (width > shut_width_m) & (width < open_width_m)
    return FreeSignals(track_pos_err=track[0], track_rot_err=track[1], tracking_valid=track[2],
                       grip_closed_shut_frac=float(shut.mean()), grip_closed_partial_frac=float(partial.mean()))
