"""Phase 2 training (`vlar_train`) on synthetic points: folds by whole task, each decider's inputs, the λ rule,
training with early stopping and a checkpoint per epoch, excluded points dropped per profile, the full
cross-validation loop, and the plumbing from records (prefix, event logs, point columns)."""

import json
import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from vlar_train import data as D  # noqa: E402
from vlar_train import folds as F  # noqa: E402
from vlar_train import lam as LAM  # noqa: E402
from vlar_train.deciders import COLUMNS, DECIDERS, FORBIDDEN_FOR_ROUTER, inputs, width  # noqa: E402
from vlar_train import run as R  # noqa: E402
from vlar_train import stats as ST  # noqa: E402
from vlar_train.run import run_cv  # noqa: E402
from vlar_train.train import TrainConfig, seed_for, train  # noqa: E402

FAST = TrainConfig(epochs=12, patience=4, batch=64, hidden=(64, 32))


@pytest.fixture(scope="module")
def ps():
    return D.synthetic(n_per_task=40, feature_dim=16)


def test_folds_split_by_whole_task():
    seen_test, seen_val = set(), set()
    for i in range(10):
        f = F.fold(i)
        assert f["test"] == i and f["val"] == (i + 1) % 10 and len(f["train"]) == 8
        assert not ({f["test"], f["val"]} & set(f["train"]))
        seen_test.add(f["test"])
        seen_val.add(f["val"])
        m = F.masks(np.arange(10).repeat(3), i)
        assert (m["train"].astype(int) + m["val"] + m["test"] == 1).all()
    assert seen_test == seen_val == set(range(10))
    with pytest.raises(ValueError):
        F.fold(10)


def test_each_decider_sees_its_own_inputs(ps):
    w = {d: width(ps.cols, d) for d in DECIDERS}
    assert w["router"] < w["cascader_nospread"] < w["cascader_chunkspread"] < w["cascader_full"]
    assert w["cascader_chunkspread"] - w["cascader_nospread"] == 4 and w["cascader_full"] - w["cascader_nospread"] == 7
    assert not any(c.startswith(FORBIDDEN_FOR_ROUTER) for c in COLUMNS["router"])
    assert "task" not in " ".join(c for cs in COLUMNS.values() for c in cs)          # no task input (§6)
    x = inputs(ps.cols, ps.remaining_s["router"], "router")
    assert x.shape == (len(ps), w["router"]) and x.dtype == np.float32
    assert np.allclose(x[:, -1], ps.remaining_s["router"] / 100)                    # its own profile's time
    assert not np.allclose(inputs(ps.cols, ps.remaining_s["cascader_full"], "router")[:, -1], x[:, -1])


def test_lambda_rule():
    # scores 0.1 .. 0.5; escalating helps (+1) at the three highest scores, hurts (-1) at the lowest
    p0 = np.zeros(5)
    p1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y0, y1 = np.zeros(5), np.array([-1.0, 0.0, 1.0, 1.0, 1.0]) + 0.0
    ok = np.ones(5, bool)
    best = LAM.choose(p0, p1, y0, y1, ok)
    assert best["best_gain"] == pytest.approx(3 / 5) and best["gain"] == pytest.approx(3 / 5)
    assert best["rate"] == pytest.approx(3 / 5) and 0.2 <= best["lambda"] < 0.3
    low = LAM.choose(p0, p1, y0, y1, ok, share=0.6)                  # 2/3 of the gain is enough: fewer escalations
    assert low["rate"] == pytest.approx(2 / 5)
    capped = LAM.choose(p0, p1, y0, y1, ok, max_rate=0.2)            # at most one escalation in five
    assert capped["rate"] <= 0.2 and capped["gain"] == pytest.approx(1 / 5)
    none = LAM.choose(p0, p1, y0, y0 - 1, ok)
    assert none["lambda"] == math.inf and none["rate"] == 0
    blocked = LAM.choose(p0, p1, y0, y1, np.array([True, True, False, False, False]))  # no budget where it helps
    assert blocked["lambda"] == math.inf


def test_training_learns_stops_early_and_checkpoints_every_epoch(ps, tmp_path):
    m = F.masks(ps.task, 0)
    x = inputs(ps.cols, ps.remaining_s["cascader_full"], "cascader_full")
    y, ym = ps.targets("cascader_full")
    tr, va = m["train"], m["val"]
    t = train(x[tr], y[tr], ym[tr], x[va], y[va], ym[va], seed=seed_for(0, "cascader_full"), cfg=FAST,
              ckpt_dir=tmp_path / "ck")
    assert t.best_epoch == int(np.argmin(t.val_loss))
    assert len(list((tmp_path / "ck").glob("epoch*.pt"))) == len(t.val_loss)
    assert json.loads((tmp_path / "ck" / "best.json").read_text())["best_epoch"] == t.best_epoch
    base = float(np.mean([-(yy * np.log(p) + (1 - yy) * np.log(1 - p))
                          for p, yy in [(np.clip(y[tr][:, b].mean(), 1e-6, 1 - 1e-6), y[va][:, b]) for b in range(4)]]))
    assert t.val_loss[t.best_epoch] < base                    # better than predicting the training mean
    again = train(x[tr], y[tr], ym[tr], x[va], y[va], ym[va], seed=seed_for(0, "cascader_full"), cfg=FAST)
    assert np.array_equal(again.predict(x[va]), t.predict(x[va]))      # from scratch, seeded: reproducible
    other = train(x[tr], y[tr], ym[tr], x[va], y[va], ym[va], seed=seed_for(1, "cascader_full"), cfg=FAST)
    assert not np.array_equal(other.predict(x[va]), t.predict(x[va]))  # another fold: another initialisation


def test_cross_validation_drops_excluded_points_and_records_everything(ps, tmp_path):
    s = run_cv(ps, folds=[0, 3], deciders=["router", "cascader_full"], out=tmp_path, cfg=FAST)
    assert len(s["results"]) == 4
    r = next(x for x in s["results"] if x["fold"] == 0 and x["decider"] == "router")
    m = F.masks(ps.task, 0)
    assert r["points"]["train"] == int((m["train"] & ~ps.excluded["router"]).sum())
    assert r["points"]["excluded"] == int(ps.excluded["router"].sum()) > 0
    full = next(x for x in s["results"] if x["fold"] == 0 and x["decider"] == "cascader_full")
    assert full["points"]["excluded"] == 0
    assert (tmp_path / "fold3" / "cascader_full" / "result.json").exists()
    assert (tmp_path / "fold3" / "cascader_full" / "test_predictions.npz").exists()
    assert {"median_best_epoch", "median_lambda"} <= set(s["router"])


def test_records_plumbing():
    log = [{"t": "start"}, {"t": "decision_point", "index": 0}, {"t": "segment"}, {"t": "decision_point", "index": 1},
           {"t": "segment"}]
    assert D.prefix_of(log, 1) == log[:4]
    with pytest.raises(D.DataError, match="index 7"):
        D.prefix_of(log, 7)
    rows = [{"run_id": "r", "seq": k, "event": json.dumps(e)} for k, e in enumerate(log)][::-1]
    assert D.group_events(rows)["r"] == log
    with pytest.raises(D.DataError, match="incomplete"):
        D.group_events([r for r in rows if r["seq"] != 2])           # a gap in the middle
    row = {"features_main": [0.0] * 1024, "features_wrist": [0.0] * 1024, "robot_state": [0.0] * 11,
           "proposal": [0.0] * 70, "budget_remaining_steps": 130, "step": 260, "chunk_pos": 10,
           "prev_segment_pi05": False, "track_pos_err": None, "track_rot_err": None, "tracking_valid": False,
           "prev_actions": [0, 0, 0, 0, 0, 0, 1.0] * 2, "prev_gripper_qpos": [0.0, 0.0, 0.02, -0.02],
           **{f"chunk_spread_{k}": 0.1 for k in D.SPREAD}, **{f"fresh_spread_{k}": None for k in D.SPREAD},
           **{f"drift_{k}": 0.2 for k in D.DRIFT}}
    c = D.point_columns(row, thresholds=(0.01, 0.07))
    assert c["budget_frac"][0] == 1.3 and c["steps_frac"][0] == 0.5 and c["chunk_pos_frac"][0] == 0.5
    assert c["grip_shut_frac"][0] == 0.5 and c["grip_partial_frac"][0] == 0.5      # one shut, one holding (4 cm)
    assert c["tracking_valid"][0] == 0 and c["track_pos_err"][0] == 0                # pi0.5's segment: zero and flagged
    assert c["fresh_spread_pos_std_mean"][0] == 0                                    # not measured -> 0


def test_training_data_needs_the_frozen_calibration(tmp_path):
    """Gripper fractions come from raw widths with the calibration's thresholds; the lineage versions say 9.004 mm,
    the frozen file 3.0 mm (amendment 50), so `from_records` refuses anything but the frozen file."""
    from types import SimpleNamespace

    from vlar_train import data as D
    cal = {"measurement": {"gripper_thresholds": {"status": "calibrated", "shut_width_m": 0.009004,
                                                  "open_width_m": 0.0715459}},
           "time_limit": {"status": "calibrated", "per_task_s": {}}}

    class Cfg(dict):
        frozen = False
    with pytest.raises(D.DataError, match="frozen calibration"):
        D.from_records(tmp_path, tmp_path / "labels.json", cfg=Cfg(cal))
    with pytest.raises(D.DataError, match="frozen calibration"):
        D.from_records(tmp_path, tmp_path / "labels.json", cfg=SimpleNamespace())


def test_heads_3_and_4_absent_everywhere_train_nothing_but_heads_1_and_2_still_do(ps):
    """Phase 1 and DAgger run branches 1 and 2 only (SPEC §17 amendment 63 (4)): with heads 3 and 4 absent at every
    point (mask 0), whatever values stand there change nothing, bit for bit, while heads 1 and 2 still train."""
    m = F.masks(ps.task, 0)
    x = inputs(ps.cols, ps.remaining_s["router"], "router")
    y, ym = ps.targets("router")
    tr, va = m["train"], m["val"]
    ym = ym.copy()
    ym[:, 2:] = 0.0
    y0, y1 = y.copy(), y.copy()
    y0[:, 2:], y1[:, 2:] = 0.0, 1.0
    a = train(x[tr], y0[tr], ym[tr], x[va], y0[va], ym[va], seed=seed_for(0, "router"), cfg=FAST)
    b = train(x[tr], y1[tr], ym[tr], x[va], y1[va], ym[va], seed=seed_for(0, "router"), cfg=FAST)
    assert np.array_equal(a.predict(x[va]), b.predict(x[va])) and a.val_loss == b.val_loss
    y2 = y0.copy()
    y2[:, 1] = 1.0 - y2[:, 1]                                 # a different head-2 target must change the model
    c = train(x[tr], y2[tr], ym[tr], x[va], y2[va], ym[va], seed=seed_for(0, "router"), cfg=FAST)
    assert not np.array_equal(a.predict(x[va])[:, 1], c.predict(x[va])[:, 1])


def test_absent_branch_labels_are_masked_out_of_training(ps):
    """Branch 3 dropped from collection (PROPOSALS-2026-11-24 §0.1 (c)): where a point has no b3 label, head 3 gets
    mask 0, so whatever value stands there changes nothing, bit for bit; the other heads still train."""
    m = F.masks(ps.task, 0)
    x = inputs(ps.cols, ps.remaining_s["router"], "router")
    y, ym = ps.targets("router")
    tr, va = m["train"], m["val"]
    absent = np.zeros(len(y), dtype=bool)
    absent[::2] = True                                        # half the points have no b3 label
    ym = ym.copy()
    ym[absent, 2] = 0.0
    y0, y1 = y.copy(), y.copy()
    y0[absent, 2], y1[absent, 2] = 0.0, 1.0
    a = train(x[tr], y0[tr], ym[tr], x[va], y0[va], ym[va], seed=seed_for(0, "router"), cfg=FAST)
    b = train(x[tr], y1[tr], ym[tr], x[va], y1[va], ym[va], seed=seed_for(0, "router"), cfg=FAST)
    assert np.array_equal(a.predict(x[va]), b.predict(x[va])) and a.val_loss == b.val_loss
    # from the label side: a NaN label (branch not run) becomes target 0 with mask 0
    nan = D.PointSet(ps.cols, ps.task, ps.snapshot_id, ps.can_escalate, ps.remaining_s,
                     {**ps.y, "router": np.where(absent[:, None] & (np.arange(4) == 2), np.nan, ps.y["router"])},
                     ps.excluded)
    yt, mt = nan.targets("router")
    assert (mt[absent, 2] == 0).all() and (yt[absent, 2] == 0).all() and (mt[~absent] == ym[~absent]).all()


def test_weights_of_one_reproduce_unweighted_training_bit_for_bit(ps):
    """Amendment 57 (4): with every weight 1, training reproduces unweighted training bit for bit (predictions and
    the whole validation-loss history), also when the equal weights come in another scale (normalised to mean 1)."""
    m = F.masks(ps.task, 0)
    x = inputs(ps.cols, ps.remaining_s["router"], "router")
    y, ym = ps.targets("router")
    ym = ym.copy()
    ym[:, 2:] = 0.0                                           # b1 and b2 only, as in Phase 1
    tr, va = m["train"], m["val"]
    kw = dict(seed=seed_for(0, "router"), cfg=FAST)
    plain = train(x[tr], y[tr], ym[tr], x[va], y[va], ym[va], **kw)
    ones = train(x[tr], y[tr], ym[tr], x[va], y[va], ym[va], w_tr=np.ones(tr.sum()), w_va=np.ones(va.sum()), **kw)
    scaled = train(x[tr], y[tr], ym[tr], x[va], y[va], ym[va], w_tr=np.full(tr.sum(), 8.0),
                   w_va=np.full(va.sum(), 0.25), **kw)
    for other in (ones, scaled):
        assert np.array_equal(other.predict(x[va]), plain.predict(x[va])) and other.val_loss == plain.val_loss
        assert other.best_epoch == plain.best_epoch


def test_the_weighted_loss_is_bce_times_mask_times_w_over_the_sum_of_mask_times_w():
    """Amendment 57 (4), on numbers worked by hand: two rows, two labelled heads in the first, one in the second."""
    from vlar_train.train import _loss
    logits = torch.tensor([[0.0, 2.0], [-1.0, 0.5]])
    y = torch.tensor([[1.0, 0.0], [1 / 3, 1.0]])
    mask = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    w = np.array([0.5, 1.5], dtype=np.float32)
    bce = lambda z, t: -(t * math.log(1 / (1 + math.exp(-z))) + (1 - t) * math.log(1 - 1 / (1 + math.exp(-z))))
    want = (0.5 * bce(0.0, 1.0) + 0.5 * bce(2.0, 0.0) + 1.5 * bce(-1.0, 1 / 3)) / (0.5 + 0.5 + 1.5)
    got = float(_loss(logits, y, torch.from_numpy(mask * w[:, None]), 0.5))
    assert got == pytest.approx(want, rel=1e-6)
    assert float(_loss(logits, y, torch.zeros(2, 2), 0.5)) == 0.0          # nothing labelled: no loss, no division by 0


def test_weights_change_training_and_early_stopping(ps):
    m = F.masks(ps.task, 0)
    x = inputs(ps.cols, ps.remaining_s["router"], "router")
    y, ym = ps.targets("router")
    tr, va = m["train"], m["val"]
    kw = dict(seed=seed_for(0, "router"), cfg=FAST)
    plain = train(x[tr], y[tr], ym[tr], x[va], y[va], ym[va], **kw)
    w = train(x[tr], y[tr], ym[tr], x[va], y[va], ym[va], w_tr=ps.weight[tr], w_va=ps.weight[va], **kw)
    assert not np.array_equal(w.predict(x[va]), plain.predict(x[va]))
    only_val = train(x[tr], y[tr], ym[tr], x[va], y[va], ym[va], w_va=ps.weight[va], **kw)
    assert only_val.val_loss != plain.val_loss                # the stopping loss is the weighted one
    # the weighted validation loss, recomputed from the stopped model's predictions
    p = np.clip(w.predict(x[va]), 1e-7, 1 - 1e-7)
    wv = ps.weight[va] / ps.weight[va].mean()
    bce = -(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p)) * ym[va] * wv[:, None]
    assert w.val_loss[w.best_epoch] == pytest.approx(bce.sum() / (ym[va] * wv[:, None]).sum(), rel=1e-4)
    with pytest.raises(ValueError, match="weights"):
        train(x[tr], y[tr], ym[tr], x[va], y[va], ym[va], w_tr=np.zeros(tr.sum()), **kw)


def test_lambda_uses_weighted_means():
    """Amendment 57 (4): the gain (b2 - b1 where the decider escalates) and the escalation rate are weighted means
    over the validation points; equal weights of any scale give the unweighted rule."""
    p0 = np.zeros(4)
    p1 = np.array([0.1, 0.2, 0.3, 0.4])
    y0, y1 = np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0, 1.0])      # b2 - b1: 0, +1, -1, +1
    ok = np.ones(4, bool)
    w = np.array([1.0, 6.0, 1.0, 2.0])
    c = LAM.curve(p0, p1, y0, y1, ok, [0.15, 0.35], weight=w)
    assert c[0]["gain"] == pytest.approx(7 / 10) and c[0]["rate"] == pytest.approx(9 / 10)
    assert c[1]["gain"] == pytest.approx(2 / 10) and c[1]["rate"] == pytest.approx(2 / 10)
    plain = LAM.choose(p0, p1, y0, y1, ok)                    # unweighted: the top point alone earns the best gain
    assert plain["lambda"] == pytest.approx(0.3) and plain["gain"] == pytest.approx(1 / 4)
    same = LAM.choose(p0, p1, y0, y1, ok, weight=np.full(4, 7.0))
    assert same["lambda"] == plain["lambda"] and same["gain"] == pytest.approx(plain["gain"])
    heavy = LAM.choose(p0, p1, y0, y1, ok, weight=w)          # the point at 0.2 is most of the population: reach it
    assert heavy["lambda"] == pytest.approx(0.1) and heavy["rate"] == pytest.approx(9 / 10)
    with pytest.raises(ValueError, match="weights"):
        LAM.choose(p0, p1, y0, y1, ok, weight=np.array([1.0, -1.0, 1.0, 1.0]))


def test_inclusion_probabilities_come_from_the_label_entries():
    """Amendment 57 (3): w_i = 1/π_i from the planner's `selection`; the uniform draw's π is its fraction."""
    assert D.inclusion_prob("s", {"selection": {"inclusion_prob": 0.25, "by": "point"}}) == 0.25
    assert D.inclusion_prob("s", {"settings": {"dp_sampling": {"fraction": 0.1, "seed": 1}}}) == 0.1
    for bad in ({"selection": {"by": "point"}}, {"selection": {"inclusion_prob": 0.0}}, {},
                {"settings": {"dp_sampling": {"scheme": "stratified", "per_task": 64}}}):
        with pytest.raises(D.DataError, match="inclusion probability"):
            D.inclusion_prob("s", bad)
    ps2 = D.synthetic(n_per_task=20, feature_dim=8)
    assert set(np.round(ps2.weight, 6)) == {1.25, 10.0} and ps2.subset(np.arange(5)).weight.shape == (5,)
    with pytest.raises(D.DataError, match="weights"):
        D.PointSet(ps2.cols, ps2.task, ps2.snapshot_id, ps2.can_escalate, ps2.remaining_s, ps2.y, ps2.excluded,
                   np.zeros(len(ps2)))


def _few_positives(ps, task, keep=3):
    """`ps` with all but `keep` of one task's positive points (p_2 - p_1 >= 1/3) made non-positive, every profile."""
    y = {}
    for prof, v in ps.y.items():
        v = v.copy()
        rows = np.flatnonzero((ps.task == task) & (v[:, 1] - v[:, 0] >= 1 / 3 - 1e-9))
        v[rows[keep:], 1] = v[rows[keep:], 0]
        y[prof] = v
    return D.PointSet(ps.cols, ps.task, ps.snapshot_id, ps.can_escalate, ps.remaining_s, y, ps.excluded, ps.weight)


def test_lambda_fallback_halves_are_recorded_and_split_the_training_tasks():
    """Amendment 63 (5): the fold's 8 training tasks in two halves, drawn with a recorded seed."""
    for i in range(10):
        h = R.halves(i)
        assert len(h["a"]) == len(h["b"]) == 4 and sorted(h["a"] + h["b"]) == F.fold(i)["train"]
        assert h == R.halves(i) and isinstance(h["seed"], int)
    assert len({tuple(R.halves(i)["a"]) for i in range(10)}) > 1


def test_lambda_fallback_when_the_validation_task_has_under_ten_positives(ps, tmp_path):
    """Amendment 63 (5): fewer than 10 positive labelled points (p_2 - p_1 >= 1/3, own profile) on the validation
    task -> λ from cross-fitted predictions and the validation points; the results list the folds that used it; the
    fold's decider itself is the one trained on the 8 training tasks, unchanged; no test-task label enters."""
    y, ym = ps.targets("router")
    assert R.positives(y[ps.task == 1], ym[ps.task == 1]).sum() >= R.MIN_POSITIVES         # the synthetic set as is
    few = _few_positives(ps, task=1)                           # task 1 validates fold 0
    assert R.positives(*[a[few.task == 1] for a in few.targets("router")]).sum() == 3
    s = run_cv(few, folds=[0, 1], deciders=["router"], out=tmp_path, cfg=FAST)
    r0, r1 = s["results"]
    assert r0["lambda"]["source"] == "cross_fitted" and r0["lambda"]["validation_positives"] == 3
    assert r0["lambda"]["halves"] == R.halves(0)
    m = F.masks(few.task, 0)
    keep = ~few.excluded["router"]
    assert r0["lambda"]["points"] == {"out_of_fold": int((m["train"] & keep).sum()),
                                      "validation": int((m["val"] & keep).sum())}
    assert r1["lambda"]["source"] == "validation" and "halves" not in r1["lambda"]
    assert s["router"]["lambda_fallback_folds"] == [0]
    assert json.loads((tmp_path / "fold0" / "router" / "result.json").read_text())["lambda"]["source"] == "cross_fitted"
    # a fold whose validation task has enough positives is as before (task 1 is fold 1's test task, so its
    # changed labels do not reach that fold at all)
    plain = run_cv(ps, folds=[1], deciders=["router"], cfg=FAST)["results"][0]
    assert (r1["best_epoch"], r1["val_loss"], r1["lambda"]["lambda"]) == (
        plain["best_epoch"], plain["val_loss"], plain["lambda"]["lambda"])
    # no test-task label enters: other labels on the test task (task 0) change nothing
    yt = {prof: np.where((few.task == 0)[:, None], 1.0 - v, v) for prof, v in few.y.items()}
    other = D.PointSet(few.cols, few.task, few.snapshot_id, few.can_escalate, few.remaining_s, yt, few.excluded,
                       few.weight)
    again = run_cv(other, folds=[0], deciders=["router"], cfg=FAST)["results"][0]
    assert again["lambda"] == r0["lambda"] and again["val_loss"] == r0["val_loss"]


def test_lambda_fallback_chooses_on_out_of_fold_and_validation_points_together(ps, monkeypatch):
    """The fallback's λ is §8's rule on the pooled points: recomputed here from the same pieces."""
    few = _few_positives(ps, task=1)
    seen = {}
    real = LAM.choose

    def spy(*a, **k):
        seen["n"], seen["w"] = len(a[0]), k["weight"]
        return real(*a, **k)
    monkeypatch.setattr(R.LAM, "choose", spy)
    r = R.run_fold_decider(few, 0, "router", None, FAST)
    m = F.masks(few.task, 0)
    keep = ~few.excluded["router"]
    n_tr, n_va = int((m["train"] & keep).sum()), int((m["val"] & keep).sum())
    assert seen["n"] == n_tr + n_va and r["lambda"]["source"] == "cross_fitted"
    assert seen["w"][:n_tr].mean() == pytest.approx(1.0) and seen["w"][n_tr:].mean() == pytest.approx(1.0)


def test_effective_sample_size_and_episode_bootstrap():
    """Amendment 57 (5)-(6): (Σw)²/Σw², and intervals that resample whole episodes."""
    assert ST.ess(np.ones(7)) == pytest.approx(7) and ST.ess([]) == 0.0
    assert ST.ess([1, 1, 1, 9]) == pytest.approx(144 / 84)            # one heavy point: under 2 effective points
    # 40 points in 4 episodes, the value constant within an episode: only 4 independent values
    v = np.repeat([0.0, 0.0, 1.0, 1.0], 10)
    w = np.ones(40)
    mean = lambda r: ST.weighted_mean(v[r], w[r])                    # noqa: E731
    by_episode = ST.cluster_bootstrap(mean, np.repeat(["a", "b", "c", "d"], 10), seed=1)
    by_point = ST.cluster_bootstrap(mean, np.arange(40), seed=1)       # every point its own episode
    assert by_episode == ST.cluster_bootstrap(mean, np.repeat(["a", "b", "c", "d"], 10), seed=1)     # seeded
    assert by_episode[0] == 0.0 and by_episode[1] == 1.0               # some resamples draw one kind of episode only
    assert 0.25 < by_point[0] < by_point[1] < 0.75                     # treating points as independent: far narrower
    assert ST.cluster_bootstrap(mean, ["a"] * 40, seed=1) is None      # one episode: nothing to resample


def test_label_statistics_are_reweighted_and_count_positive_episodes():
    """Amendments 57 (6) and 68 (3), on numbers worked by hand."""
    y = np.array([[0.0, 1 / 3], [0.0, 1.0], [1.0, 2 / 3], [1.0, 1.0], [0.0, 1.0]])      # p_2 - p_1: 1/3, 1, -1/3, 0, 1
    y = np.concatenate([y, np.zeros((5, 2))], axis=1)
    ym = np.array([[1, 1, 0, 0]] * 4 + [[1, 0, 0, 0]], dtype=np.float32)                # the last row has no b2 label
    w = np.array([1.0, 1.0, 4.0, 2.0, 50.0])
    s = ST.label_stats(y, ym, w, ["e1", "e1", "e2", "e3", "e9"], seed=0, resamples=200)
    assert s["points"] == 4 and s["episodes"] == 3
    assert s["positives"] == 2 and s["positive_episodes"] == 1          # two positive points, one episode
    assert s["rescue_rate"] == pytest.approx(2 / 8) and s["harm_rate"] == pytest.approx(4 / 8)
    assert s["mean_p2_minus_p1"] == pytest.approx((1 / 3 + 1 - 4 / 3) / 8)
    assert s["ess"] == pytest.approx(64 / 22)
    lo, hi = s["rescue_rate_ci95"]
    assert 0.0 <= lo <= s["rescue_rate"] <= hi <= 1.0


def test_points_keep_their_parent_episode(ps):
    """Amendment 68 (3): the episode id stays on every point (from the snapshot id where none is given)."""
    assert ps.episode.shape == (len(ps),) and 2 <= len(set(ps.episode[ps.task == 3])) <= 12
    assert set(e.split("/")[1] for e in ps.episode[ps.task == 3]) == {"t3"}       # an episode belongs to one task
    assert list(ps.subset(np.array([5, 2])).episode) == [ps.episode[5], ps.episode[2]]
    bare = D.PointSet({}, np.zeros(2, int), ["base/t0/i4/dp07", "base/t0/i4/dp31"], np.ones(2, bool), {}, {}, {})
    assert list(bare.episode) == ["base/t0/i4", "base/t0/i4"] and list(bare.weight) == [1.0, 1.0]
    with pytest.raises(D.DataError, match="episode"):
        D.PointSet({}, np.zeros(2, int), ["a/dp00", "a/dp01"], np.ones(2, bool), {}, {}, {}, None, ["a"])


def test_results_report_episodes_effective_sample_size_and_intervals(ps, tmp_path):
    r = run_cv(ps, folds=[0], deciders=["router"], out=tmp_path, cfg=FAST)["results"][0]
    m = F.masks(ps.task, 0)
    keep = ~ps.excluded["router"]
    for split in ("train", "val"):
        d = r["data"][split]
        rows = m[split] & keep
        assert d["points"] == int(rows.sum()) and d["episodes"] == len(set(ps.episode[rows]))
        assert d["ess"] == pytest.approx(ST.ess(ps.weight[rows])) and d["ess"] < d["points"]
        assert d["positive_episodes"] <= d["positives"] and d["positive_episodes"] <= d["episodes"]
        assert d["rescue_rate_ci95"][0] <= d["rescue_rate"] <= d["rescue_rate_ci95"][1]
    lam = r["lambda"]
    assert lam["validation_positive_episodes"] <= lam["validation_positives"]
    assert lam["selection"]["points"] == r["data"]["val"]["points"]
    assert lam["gain_ci95"][0] <= lam["gain"] <= lam["gain_ci95"][1]
    assert lam["rate_ci95"][0] <= lam["rate"] <= lam["rate_ci95"][1]
    z = np.load(tmp_path / "fold0" / "router" / "test_predictions.npz")
    assert list(z["episode"]) == list(ps.episode[m["test"] & keep]) and z["weight"].shape == z["episode"].shape


def test_lambda_never_chooses_always_escalate_within_the_budget():
    """§8: "subject to the escalation rate staying within budget". One point where escalating helps, at the lowest
    score: the bare rule escalates everywhere to reach it; within the budget's rate (§4: 260 of 520 steps) it
    cannot, and with nothing else to gain it never escalates."""
    p0, p1 = np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4])
    y0, y1 = np.zeros(4), np.array([1 / 3, 0.0, 0.0, 0.0])
    ok = np.ones(4, bool)
    bare = LAM.choose(p0, p1, y0, y1, ok)
    assert bare["lambda"] == -math.inf and bare["rate"] == 1.0
    capped = LAM.choose(p0, p1, y0, y1, ok, max_rate=LAM.MAX_RATE)
    assert LAM.MAX_RATE == 0.5 and capped["lambda"] == math.inf and capped["rate"] == 0.0
    # where the gain is reachable within the budget, the cap changes nothing
    y1b = np.array([0.0, 0.0, 1.0, 1.0])
    assert LAM.choose(p0, p1, y0, y1b, ok, max_rate=LAM.MAX_RATE) == LAM.choose(p0, p1, y0, y1b, ok)


def test_lambda_needs_support_from_more_than_one_episode():
    """Points of one episode share its outcome (amendment 68 (3)): a gain that rests on a single episode chooses
    nothing; `min_episodes=1` is §8's rule exactly."""
    p0, p1 = np.zeros(6), np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y0 = np.zeros(6)
    ok = np.ones(6, bool)
    one = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])                   # both gains in episode "e3"
    ep = np.array(["e1", "e1", "e2", "e2", "e3", "e3"])
    c = LAM.curve(p0, p1, y0, one, ok, [0.45], episode=ep)[0]
    assert c["support"] == 1 and c["gain"] == pytest.approx(2 / 6)
    bare = LAM.choose(p0, p1, y0, one, ok, episode=ep)
    assert bare["lambda"] == pytest.approx(0.4) and bare == LAM.choose(p0, p1, y0, one, ok, episode=ep, min_episodes=1)
    guarded = LAM.choose(p0, p1, y0, one, ok, episode=ep, min_episodes=2)
    assert guarded["lambda"] == math.inf and guarded["rate"] == 0.0 and "supported" in guarded["reason"]
    two = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])                   # gains in episodes "e2" and "e3"
    g2 = LAM.choose(p0, p1, y0, two, ok, episode=ep, min_episodes=2)
    assert g2["lambda"] == pytest.approx(0.3) and g2["support"] == 2
    # the single top point alone is one episode: the guard reaches down to the λ that two episodes support
    assert LAM.choose(p0, p1, y0, two, ok, episode=ep, min_episodes=2, share=0.5)["lambda"] == pytest.approx(0.3)
    assert LAM.choose(p0, p1, y0, two, ok, episode=ep, min_episodes=1, share=0.5)["lambda"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="episode"):
        LAM.curve(p0, p1, y0, one, ok, [0.0], episode=ep[:3])


def test_the_median_lambda_is_never_nan():
    """§9 uses the folds' median λ; NumPy's median of -inf and +inf is NaN."""
    inf = math.inf
    assert LAM.median([-inf, -inf, inf, inf]) == inf
    assert LAM.median([0.1, 0.3, 0.2]) == pytest.approx(0.2) and LAM.median([0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.25)
    assert LAM.median([0.1, 0.2, inf, inf]) == inf and LAM.median([0.1, 0.2, 0.3, inf]) == pytest.approx(0.25)
    assert LAM.median([-inf, 0.1, 0.2, 0.3]) == pytest.approx(0.15) and LAM.median([-inf, -inf, 0.2, 0.3]) == 0.2
    assert LAM.median([inf]) == inf
    for bad in ([], [0.1, float("nan")]):
        with pytest.raises(ValueError):
            LAM.median(bad)


def test_cross_validation_applies_the_rate_cap_and_the_support_guard(ps):
    s = run_cv(ps, folds=[0, 1], deciders=["router", "cascader_full"], cfg=FAST)
    assert s["lambda_rule"] == {"share": 0.95, "max_rate": 0.5, "min_episodes": 2, "min_positives": 10}
    for r in s["results"]:
        assert r["lambda"]["rate"] <= 0.5 + 1e-9 and r["lambda"]["lambda"] != -math.inf
        assert r["lambda"]["lambda"] == math.inf or r["lambda"]["support"] >= 2
    for d in ("router", "cascader_full"):
        assert not math.isnan(s[d]["median_lambda"]) and "lambda_never_folds" in s[d]
    loose = run_cv(ps, folds=[0], deciders=["router"], cfg=FAST, max_rate=1.0, min_episodes=1)["results"][0]
    assert loose["lambda"]["rate"] >= s["results"][0]["lambda"]["rate"]
