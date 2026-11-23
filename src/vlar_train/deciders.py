"""The four deciders' inputs (SPEC §6) and the shared model family (§8).

Every decider sees the common inputs: both pooled X-VLA feature vectors, the robot state, the remaining π0.5 budget,
the remaining time under its own profile and the steps elapsed. The cascaders add the proposal, its position in the
chunk, whether the previous segment was π0.5's, and the free signals (tracking error with its validity flag, the two
gripper-mismatch fractions); `cascader_chunkspread` adds the chunk-time spread, `cascader_full` the fresh spread and
the drift. No task id. Outputs: four probabilities, one per branch (continue, one call, three calls, max).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

DECIDERS = ("router", "cascader_nospread", "cascader_chunkspread", "cascader_full")
BRANCHES = ("b1_continue", "b2_one_call", "b3_three_calls", "b4_max")
SPREAD = ("pos_std_mean", "pos_std_max", "rot_dev_mean", "grip_disagree_frac")
DRIFT = ("pos_dist_mean", "rot_dist_mean", "grip_disagree_frac")
STEP_LIMIT = 520.0

# Point columns (PointSet.cols) each decider uses, in order; remaining time is per profile and added separately.
COMMON = ("features_main", "features_wrist", "robot_state", "budget_frac", "steps_frac")
CASCADE = ("proposal", "chunk_pos_frac", "prev_segment_pi05", "track_pos_err", "track_rot_err", "tracking_valid",
           "grip_shut_frac", "grip_partial_frac")
COLUMNS: dict[str, tuple[str, ...]] = {
    "router": COMMON,
    "cascader_nospread": COMMON + CASCADE,
    "cascader_chunkspread": COMMON + CASCADE + tuple(f"chunk_spread_{k}" for k in SPREAD),
    "cascader_full": COMMON + CASCADE + tuple(f"fresh_spread_{k}" for k in SPREAD) + tuple(f"drift_{k}" for k in DRIFT),
}
FORBIDDEN_FOR_ROUTER = ("proposal", "chunk_spread_", "fresh_spread_", "drift_", "track_", "grip_")


def inputs(cols: Mapping[str, np.ndarray], remaining_s: np.ndarray, decider: str) -> np.ndarray:
    """(n, width) float32: the decider's columns in order, then its own profile's remaining time (seconds / 100)."""
    parts = []
    for c in COLUMNS[decider]:
        a = np.asarray(cols[c], dtype=np.float64)
        parts.append(a.reshape(len(a), -1))
    parts.append(np.asarray(remaining_s, dtype=np.float64).reshape(-1, 1) / 100.0)
    return np.concatenate(parts, axis=1).astype(np.float32)


def width(cols: Mapping[str, np.ndarray], decider: str) -> int:
    return sum(int(np.asarray(cols[c]).reshape(len(cols[c]), -1).shape[1]) for c in COLUMNS[decider]) + 1


def make_model(n_in: int, hidden: tuple[int, ...] = (512, 256), dropout: float = 0.1):
    """The shared family: an MLP, the same hidden sizes for every decider; four logits (one per branch)."""
    import torch.nn as nn
    layers, last = [], n_in
    for h in hidden:
        layers += [nn.Linear(last, h), nn.GELU(), nn.Dropout(dropout)]
        last = h
    layers.append(nn.Linear(last, len(BRANCHES)))
    return nn.Sequential(*layers)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
