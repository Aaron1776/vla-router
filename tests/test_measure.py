"""The decider measurements' arithmetic (vlar_harness.measure; SPEC §6) on hand-checkable inputs.
Needs NumPy: run with the simulator environment (`.venv-sim/bin/python -m pytest tests/test_measure.py`)."""

import math

import pytest

np = pytest.importorskip("numpy")

from vlar_harness import measure as M


def rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rows(pos, aa, g):
    """n x 7 executed-form rows from n x 3 positions, n x 3 axis-angles and n gripper values."""
    return np.concatenate([np.asarray(pos, float), np.asarray(aa, float), np.asarray(g, float)[:, None]], axis=1)


def test_axis_angle_matches_robosuite_and_is_a_rotation():
    T = pytest.importorskip("robosuite.utils.transform_utils")
    rng = np.random.default_rng(0)
    aa = np.concatenate([rng.normal(size=(200, 3)), rng.normal(size=(20, 3)) * 1e-9, np.zeros((1, 3))])
    R = M.axis_angle_to_mat(aa)
    for a, r in zip(aa, R):
        assert np.allclose(r, T.quat2mat(T.axisangle2quat(a)), atol=1e-6)     # robosuite computes in float32
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12) and np.linalg.det(r) == pytest.approx(1.0)


def test_rotation_angle_and_mean():
    for t in (0.0, 1e-7, 0.3, 2.0, math.pi - 1e-6):
        assert M.rotation_angle(np.eye(3), rot_z(t)) == pytest.approx(t, abs=1e-9)
    assert np.allclose(M.mean_rotation(np.stack([rot_z(0.4)] * 3)), rot_z(0.4))
    assert np.allclose(M.mean_rotation(np.stack([rot_z(0.2), rot_z(0.6)])), rot_z(0.4))


def test_executed_form_rounds_the_gripper_like_pop():
    out = M.executed_form([[1, 2, 3, 0.1, 0.2, 0.3, 0.5], [1, 2, 3, 0.1, 0.2, 0.3, 0.50001]])
    assert out.dtype == np.float32 and out[:, 6].tolist() == [-1.0, 1.0]


def test_spread_by_hand():
    n = 2
    z = np.zeros((n, 3))
    a = rows([[0, 0, 0], [0, 0, 0]], z, [1, 1])
    b = rows([[0.3, 0, 0], [0, 0.4, 0]], [[0, 0, 0.2], [0, 0, 0]], [1, -1])
    c = rows([[0, 0, 0], [0, 0, 0]], [[0, 0, 0.4], [0, 0, 0]], [-1, -1])
    s = M.spread(np.stack([a, b, c]))
    # step 0: positions 0, 0.3, 0 on x -> mean 0.1, RMS distance sqrt((0.01 + 0.04 + 0.01) / 3) = sqrt(0.02)
    # step 1: positions 0, 0.4, 0 on y -> sqrt((0.0178 + 0.0711 + 0.0178) / 3) = sqrt(0.0356) = 0.4 * sqrt(2) / 3
    s0, s1 = math.sqrt(0.02), 0.4 * math.sqrt(2) / 3
    assert s.pos_std_mean == pytest.approx((s0 + s1) / 2) and s.pos_std_max == pytest.approx(s1)
    # step 0 rotations about z by 0, 0.2, 0.4 -> mean 0.2, deviations 0.2, 0, 0.2; step 1 all identity
    assert s.rot_dev_mean == pytest.approx((0.4 / 3 + 0.0) / 2, abs=1e-9)
    # gripper: step 0 two close one open -> 1/3; step 1 one close two open -> 1/3
    assert s.grip_disagree_frac == pytest.approx(1 / 3)
    same = M.spread(np.stack([a, a, a]))
    assert (same.pos_std_mean, same.pos_std_max, same.rot_dev_mean, same.grip_disagree_frac) == (0, 0, 0, 0)


def test_drift_by_hand():
    z = np.zeros((2, 3))
    samples = np.stack([rows([[0, 0, 0], [1, 0, 0]], z, [1, -1]),
                        rows([[0, 0, 0.2], [1, 0, 0]], z, [1, -1]),
                        rows([[0, 0, 0.4], [1, 0, 0]], z, [-1, -1])])
    q = rows([[0, 0, 0.2], [1, 0, 0.5]], [[0, 0, 0], [0, 0.3, 0]], [-1, -1])
    d = M.drift(q, samples)
    assert d.pos_dist_mean == pytest.approx((0.0 + 0.5) / 2)
    assert d.rot_dist_mean == pytest.approx(0.3 / 2)
    assert d.grip_disagree_frac == pytest.approx(0.5)          # step 0 majority close, queued open; step 1 agree
    tie = M.drift(q[:1], np.stack([samples[0, :1], samples[2, :1]]))
    assert tie.grip_disagree_frac == 0.5
    same = M.drift(samples[0], np.stack([samples[0]] * 3))
    assert (same.pos_dist_mean, same.rot_dist_mean, same.grip_disagree_frac) == (0, 0, 0)


def step(driver, action, ee_pos, ee_ori_mat, qpos):
    return {"driver": driver, "action": np.asarray(action, float), "ee_pos": np.asarray(ee_pos, float),
            "ee_ori_mat": np.asarray(ee_ori_mat, float), "gripper_qpos": np.asarray(qpos, float)}


def test_free_signals_by_hand():
    thr = dict(shut_width_m=0.01, open_width_m=0.07)
    start = M.free_signals([], **thr)
    assert start.tracking_valid is False and start.track_pos_err == 0.0 and start.grip_closed_shut_frac == 0.0
    seg = [step("xvla", [0, 0, 0, 0, 0, 0.1, 1], [0.03, 0.04, 0], rot_z(0.0), [0.002, -0.002]),       # shut
           step("xvla", [0, 0, 0, 0, 0, 0.0, 1], [0.0, 0.0, 0.0], rot_z(0.2), [0.02, -0.02]),          # partly open
           step("xvla", [0, 0, 0, 0, 0, 0.0, -1], [0.0, 0.0, 0.0], rot_z(0.0), [0.002, -0.002]),       # open cmd
           step("xvla", [0, 0, 0, 0, 0, 0.0, 1], [0.0, 0.0, 0.1], rot_z(0.0), [0.04, -0.04])]          # wide open
    f = M.free_signals(seg, **thr)
    assert f.tracking_valid is True
    assert f.track_pos_err == pytest.approx((0.05 + 0 + 0 + 0.1) / 4)
    assert f.track_rot_err == pytest.approx((0.1 + 0.2) / 4, abs=1e-9)
    assert (f.grip_closed_shut_frac, f.grip_closed_partial_frac) == (0.25, 0.25)
    pi = M.free_signals([dict(s, driver="pi05") for s in seg], **thr)
    assert pi.tracking_valid is False and pi.track_pos_err == 0.0 and pi.grip_closed_shut_frac == 0.25
    with pytest.raises(ValueError, match="uncalibrated"):
        M.free_signals(seg, None, 0.07)
    with pytest.raises(ValueError, match="one driver"):
        M.free_signals(seg[:1] + [dict(seg[1], driver="pi05")], **thr)


def test_gripper_fractions_offline_equal_the_live_free_signals():
    """Amendment 35: the points table stores only the raw segment; the fractions computed from its flattened
    float32 actions and float64 finger positions equal the live ones."""
    rng = np.random.default_rng(3)
    seg = [{"driver": "xvla", "action": np.r_[rng.normal(size=6), g], "ee_pos": rng.normal(size=3),
            "ee_ori_mat": np.eye(3), "gripper_qpos": np.array([w / 2, -w / 2])}
           for g, w in zip(rng.choice([-1.0, 1.0], 10), rng.uniform(0.0, 0.08, 10))]
    live = M.free_signals(seg, 0.01, 0.07)
    row = {"prev_actions": np.asarray([r["action"] for r in seg], np.float32).ravel().tolist(),
           "prev_gripper_qpos": np.asarray([r["gripper_qpos"] for r in seg]).ravel().tolist()}
    assert M.gripper_fractions_of_point(row, 0.01, 0.07) == (live.grip_closed_shut_frac, live.grip_closed_partial_frac)
    assert M.gripper_fractions_of_point({"prev_actions": [], "prev_gripper_qpos": []}, 0.01, 0.07) == (0.0, 0.0)
    with pytest.raises(ValueError, match="uncalibrated"):
        M.gripper_fractions_of_point(row, None, 0.07)
