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
from vlar_train import history as H  # noqa: E402
from vlar_train import lam as LAM  # noqa: E402
from vlar_train import deciders as DC  # noqa: E402
from vlar_train.deciders import COLUMNS, DECIDERS, FORBIDDEN_FOR_ROUTER, inputs, width  # noqa: E402
from vlar_train import run as R  # noqa: E402
from vlar_train import stats as ST  # noqa: E402
from vlar_train.run import run_cv  # noqa: E402
from vlar_train.train import TrainConfig, constant_predictor, seed_for, train, weighted_bce  # noqa: E402

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


# ---------------------------------------------------------------- building from records

def _records(episodes=("ep", "ep2")):
    """Records of base episodes with one labelled point each (point 2), built with the label tests' logs: per
    episode the parent run, its branch runs, their events, the points row and the planner's label entry."""
    TL = pytest.importorskip("test_labels_calibration")
    runs, events, points, label_map = [], [], [], {}

    def add(run_id, log, **row):
        runs.append({"run_id": run_id, "task_id": 0, "snapshot_id": None, "branch": None, "seed_index": None,
                     "record": "full", "phase": "base", "job_id": f"job-{run_id}", **row})
        events.extend({"run_id": run_id, "seq": k, "event": json.dumps(e)} for k, e in enumerate(log))
    for ep in episodes:
        prefix, entry, parent, own = TL.setup_point()
        sid = f"{ep}/dp02"
        add(ep, parent, episode_id=ep)
        for (b, k), log in own.items():
            add(f"{sid}/{b}/s{k}", log, snapshot_id=sid, branch=b, seed_index=k, episode_id=ep, phase="branches")
        points.append({"run_id": ep, "index": 2, "snapshot_id": sid, "features_main": [0.0] * 1024,
                       "features_wrist": [0.0] * 1024, "robot_state": [0.0] * 11, "proposal": [0.0] * 70,
                       "budget_remaining_steps": 20, "step": 20, "chunk_pos": 10, "prev_segment_pi05": False,
                       "track_pos_err": 0.001, "track_rot_err": 0.01, "tracking_valid": True,
                       "prev_actions": [0, 0, 0, 0, 0, 0, 1.0] * 2, "prev_gripper_qpos": [0.0, 0.0, 0.02, -0.02]})
        label_map[sid] = {**entry, "point": {**entry["point"], "snapshot_id": sid, "episode_id": ep},
                          "selection": {"inclusion_prob": 0.25, "by": "point"}}
    kw = dict(limits_s={0: 1e9}, thresholds=(0.01, 0.07), charges=TL.CH)
    return runs, events, points, label_map, kw


def test_build_computes_labels_weights_and_episodes_from_records():
    runs, events, points, label_map, kw = _records()
    ps = D.build(runs, events, points, label_map, **kw)
    assert ps.snapshot_id == ["ep/dp02", "ep2/dp02"] and list(ps.episode) == ["ep", "ep2"]
    assert list(ps.weight) == [4.0, 4.0] and ps.can_escalate.all()                 # 1/π, amendment 57 (3)
    assert ps.y["router"][0] == pytest.approx([2 / 3, 2 / 3, 1 / 3, 1 / 3])        # as the label tests work out
    assert ps.report == {"label_entries": 2, "points": 2, "incomplete": [], "shared_run_ids": 0,
                         "no_points_row": []}


def test_an_incomplete_point_is_left_out_and_reported_not_fatal():
    """§12: training uses only complete points. A branch seed without a finished run (no log, or a log cut before
    its end event) leaves that one point out, with a report; the other points still build."""
    runs, events, points, label_map, kw = _records()
    gone = "ep2/dp02/b2_one_call/s3"
    cut = "ep2/dp02/b1_continue/s2"
    last = max(e["seq"] for e in events if e["run_id"] == cut)
    ev = [e for e in events if e["run_id"] != gone and not (e["run_id"] == cut and e["seq"] == last)]
    ps = D.build([r for r in runs if r["run_id"] != gone], ev, points, {**label_map, "ep3/dp02": label_map["ep/dp02"]},
                 **kw)
    assert ps.snapshot_id == ["ep/dp02"] and len(ps) == 1
    assert ps.report["incomplete"] == [{"snapshot_id": "ep2/dp02", "missing": ["b1_continue/s2", "b2_one_call/s3"]}]
    assert ps.report["no_points_row"] == ["ep3/dp02"] and ps.report["points"] == 1 and ps.report["label_entries"] == 3
    assert ps.subset(np.array([0])).report == ps.report
    # the parent provides seed 1 of b1 at a base point: it is not a missing branch run
    assert D.missing_seeds(label_map["ep/dp02"], {}, 3)[:2] == ["b1_continue/s2", "b1_continue/s3"]
    # a log that does not fit is still an error, not an incomplete point
    bad = [dict(e) for e in events]
    for e in bad:
        if e["run_id"] == "ep/dp02/b2_one_call/s1" and e["seq"] == 1:
            e["event"] = json.dumps({**json.loads(e["event"]), "seed_index": 2})
    with pytest.raises(ValueError, match="seed"):
        D.build(runs, bad, points, label_map, **kw)
    with pytest.raises(D.DataError, match="none of the 2 labelled points is complete"):
        D.build(runs, [e for e in events if "/b2_one_call/" not in e["run_id"]], points, label_map, **kw)


def _screened(runs, events, episode="ep2"):
    """`episode` as an exploration episode that was screened (record off) and re-run (record full): two run rows
    and two event logs under one run id, from two jobs (amendment 66 (5))."""
    runs = [dict(r, phase="explore") if r["run_id"] == episode else r for r in runs]
    screen = next(dict(r, record="off", job_id="job-screen") for r in runs if r["run_id"] == episode)
    events = [dict(e, job_id=f"job-{e['run_id']}") for e in events]
    return [screen] + runs, events + [dict(e, job_id="job-screen") for e in events if e["run_id"] == episode]


def test_a_screened_run_and_its_rerun_share_a_run_id_and_the_record_full_one_is_used():
    """Amendment 66 (5): consumers take the record-full row; the check is no run id twice, every kept row full."""
    runs, events, points, label_map, kw = _records()
    plain = D.build(runs, events, points, label_map, **kw)
    runs2, events2 = _screened(runs, events)
    kept, shared = D.record_full_runs(runs2)
    assert len(kept) == len(runs) and shared == {"ep2": "job-ep2"}
    assert [r["record"] for r in kept if r["run_id"] == "ep2"] == ["full"]
    ps = D.build(runs2[::-1], events2[::-1], points, label_map, **kw)             # in any row order
    assert ps.report["shared_run_ids"] == 1 and ps.snapshot_id == plain.snapshot_id
    for prof in plain.y:
        assert np.array_equal(ps.y[prof], plain.y[prof]) and np.array_equal(ps.remaining_s[prof], plain.remaining_s[prof])
    # without the parts' job ids the two logs cannot be told apart: refused, not guessed
    with pytest.raises(D.DataError, match="read_events"):
        D.build(runs2, [{k: v for k, v in e.items() if k != "job_id"} for e in events2], points, label_map, **kw)
    # anything else that repeats a run id is an error
    screen = runs2[0]
    for bad in ([dict(screen, record="full")] + runs2[1:],                         # two record-full rows
                [dict(screen, phase="base")] + runs2[1:],                          # not an exploration run
                [screen, dict(screen, job_id="job-screen-2")] + runs2[1:],         # three rows
                [dict(screen, job_id="job-ep2")] + runs2[1:]):                     # the same job twice
        with pytest.raises(D.DataError, match="run id ep2"):
            D.record_full_runs(bad)
    # a labelled point whose parent's only row is the screen's
    with pytest.raises(D.DataError, match="not full"):
        D.build([dict(r, record="off") if r["run_id"] == "ep2" else r for r in runs], events, points, label_map, **kw)


def test_read_events_adds_each_parts_job_id(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    schema = pa.schema([("run_id", pa.string()), ("seq", pa.int32()), ("t", pa.string()), ("step", pa.int32()),
                        ("event", pa.string())])
    for phase, job, rid in (("explore", "aaa", "explore/t0/i0/r0"), ("explore", "bbb", "explore/t0/i0/r0"),
                            ("branches", "ccc", "explore/t0/i0/r0/dp02/b1_continue/s1")):
        d = tmp_path / "records" / "events" / phase
        d.mkdir(parents=True, exist_ok=True)
        rows = [{"run_id": rid, "seq": k, "t": "x", "step": k, "event": json.dumps({"t": "x", "k": k})} for k in range(3)]
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), d / f"{job}.parquet")
    ev = D.read_events(tmp_path)
    assert len(ev) == 9 and {e["job_id"] for e in ev} == {"aaa", "bbb", "ccc"}
    assert set(ev[0]) == {"run_id", "seq", "event", "job_id"}
    with pytest.raises(D.DataError, match="incomplete"):                           # both logs under one run id
        D.group_events(ev)
    assert D.group_events([e for e in ev if e["job_id"] != "aaa"])["explore/t0/i0/r0"][2] == {"t": "x", "k": 2}


def test_constant_predictor_baseline_by_hand():
    """Per head the weighted mean training label; its loss by the training loss's formula (amendment 57 (4))."""
    y = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1 / 3, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    m = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    w = np.array([2.0, 1.0, 1.0])
    c = constant_predictor(y, m, w)
    assert c == pytest.approx([2 / 4, (1 / 3) / 3, 0.5, 0.5])            # unlabelled heads: 0.5, never scored
    assert constant_predictor(y, m) == pytest.approx([1 / 3, 1 / 6, 0.5, 0.5])
    h = lambda p, t: -(t * math.log(p) + (1 - t) * math.log(1 - p))    # noqa: E731
    want = (2 * h(0.5, 1) + 2 * h(1 / 9, 0) + h(0.5, 0) + h(1 / 9, 1 / 3) + h(0.5, 0)) / (2 + 2 + 1 + 1 + 1)
    assert weighted_bce(c, y, m, w) == pytest.approx(want)
    # the weighted mean is the best constant: any other constant has a higher weighted training loss
    for other in ([0.4, 1 / 9, 0.5, 0.5], [0.5, 0.2, 0.5, 0.5]):
        assert weighted_bce(np.array(other), y, m, w) > weighted_bce(c, y, m, w)
    assert weighted_bce(c, y, np.zeros_like(m), w) == 0.0


def test_results_carry_the_constant_predictor_baseline(ps, tmp_path):
    s = run_cv(ps, folds=[0], deciders=["router"], out=tmp_path, cfg=FAST)
    r = s["results"][0]
    m = F.masks(ps.task, 0)
    keep = ~ps.excluded["router"]
    tr, va = m["train"] & keep, m["val"] & keep
    y, ym = ps.targets("router")
    c = constant_predictor(y[tr], ym[tr], ps.weight[tr])
    b = r["baseline"]
    assert b["p"] == pytest.approx(list(c)) and b["val_loss"] == pytest.approx(weighted_bce(c, y[va], ym[va], ps.weight[va]))
    assert b["val_loss_minus_baseline"] == pytest.approx(r["val_loss"] - b["val_loss"])
    assert s["router"]["folds_beating_constant_on_validation"] == ([0] if b["val_loss_minus_baseline"] < 0 else [])
    z = np.load(tmp_path / "fold0" / "router" / "test_predictions.npz")
    assert z["p_constant"] == pytest.approx(c)
    # the model's own validation loss is the same formula on its predictions
    x = inputs(ps.cols, ps.remaining_s["router"], "router")
    again = train(x[tr], y[tr], ym[tr], x[va], y[va], ym[va], seed=seed_for(0, "router"), cfg=FAST,
                  w_tr=ps.weight[tr], w_va=ps.weight[va])
    assert weighted_bce(again.predict(x[va]), y[va], ym[va], ps.weight[va]) == pytest.approx(r["val_loss"], rel=1e-4)


# ---------------------------------------------------------------- history inputs (extension point)

@pytest.fixture
def toy_history():
    """Two toy history features, registered for one test only: one from the common inputs (motion of the end
    effector over the last 30 steps), one from the cascader groups (tracking error per commanded step)."""
    before = (dict(H.FEATURES), dict(COLUMNS), dict(DC.PROFILE_OF))

    @H.register("toy_motion", width=2, uses=["common"])
    def motion(views):
        recent = [v for v in views if views[-1]["step"] - v["step"] <= 30]
        pos = np.array([v["robot_state"][:3] for v in recent])
        return [np.linalg.norm(np.diff(pos, axis=0), axis=1).sum(), np.linalg.norm(pos[-1] - pos[0])]

    @H.register("toy_track", width=1, uses=["proposal", "free_signals"])
    def track(views):
        if len(views) < 2:
            return [0.0]
        cmd = np.linalg.norm(np.diff(views[-2]["proposal"][:, :3], axis=0), axis=1).mean()
        return [views[-1]["track_pos_err"] / max(cmd, 1e-6)]
    yield
    for now, was in zip((H.FEATURES, COLUMNS, DC.PROFILE_OF), before):
        now.clear()
        now.update(was)


def _point_row(index, budget=260, **over):
    rng = np.random.default_rng(index)
    row = {"run_id": "ep", "index": index, "step": 10 * index, "snapshot_id": f"ep/dp{index:02d}",
           "budget_remaining_steps": budget, "robot_state": list(rng.normal(size=11)),
           "proposal": list(rng.normal(size=70).astype(np.float32)), "chunk_pos": 10 * (index % 3),
           "prev_segment_pi05": False, "track_pos_err": 0.01 * (index + 1), "track_rot_err": 0.02,
           "tracking_valid": index > 0, "prev_actions": [0, 0, 0, 0, 0, 0, 1.0] * 2,
           "prev_gripper_qpos": [0.0, 0.0, 0.02, -0.02], "features_main": [0.0] * 1024,
           "features_wrist": [0.0] * 1024}
    return {**row, **over}


def _observed(row, view, profile="cascader_nospread"):
    """What `observe()` would return at the point of `row` (the harness API's own classes)."""
    from vlar_harness import api as A
    point = A.DecisionPoint(row["index"], row["step"], "mid", row["budget_remaining_steps"], True)
    common = A.CommonInputs(None, None, np.array(row["robot_state"]), row["budget_remaining_steps"], 12.0, row["step"])
    if profile == "router":
        return A.DeciderInputs(profile, point, common)
    return A.DeciderInputs(profile, point, common,
                           proposal=A.Proposal(np.array(row["proposal"], dtype=np.float32).reshape(10, 7),
                                               row["chunk_pos"], row["prev_segment_pi05"]),
                           free_signals=A.FreeSignals(view["track_pos_err"], view["track_rot_err"],
                                                      view["tracking_valid"], view["grip_shut_frac"],
                                                      view["grip_partial_frac"]))


def test_no_history_feature_is_registered_and_no_decider_changed():
    """The hook changes nothing by itself: the four deciders of §8, their columns, widths and profiles as before."""
    assert H.FEATURES == {} and not any(c.startswith(H.PREFIX) for cs in COLUMNS.values() for c in cs)
    assert tuple(COLUMNS) == DECIDERS and DC.PROFILE_OF == {d: d for d in DECIDERS}
    assert COLUMNS["router"] == ("features_main", "features_wrist", "robot_state", "budget_frac", "steps_frac")
    cols = D.synthetic(n_per_task=2).cols
    assert [width(cols, d) for d in DECIDERS] == [2062, 2139, 2143, 2146]


def test_history_columns_are_the_same_offline_and_live(toy_history):
    """One code path: the views from points-table rows and from `observe()` give identical history columns."""
    th = (0.01, 0.07)
    rows = [_point_row(k) for k in range(6)]
    off, live = H.EpisodeHistory(), H.EpisodeHistory()
    for row, view in H.episode_views(rows[::-1], thresholds=th):          # any row order: walked by index
        lv = H.live_view(_observed(row, view))
        assert set(lv) == set(view) and H.groups_of(lv) == H.GROUPS
        a, b = off.push(view), live.push(lv)
        assert set(a) == {"hist_toy_motion", "hist_toy_track"} and a["hist_toy_motion"].shape == (2,)
        for k in a:
            assert np.array_equal(a[k], b[k])
    # the motion feature by hand at the last point: the views within 30 steps are points 2..5
    pos = np.array([r["robot_state"][:3] for r in rows[2:]])
    assert a["hist_toy_motion"] == pytest.approx([np.linalg.norm(np.diff(pos, axis=0), axis=1).sum(),
                                                  np.linalg.norm(pos[-1] - pos[0])])
    assert a["hist_toy_track"][0] > 0


def test_history_follows_what_a_live_decider_sees(toy_history):
    th = (0.01, 0.07)
    rows = [_point_row(0), _point_row(1), _point_row(2, budget=0), _point_row(3, budget=5), _point_row(4)]
    seen = [r["index"] for r, _ in H.episode_views(rows, thresholds=th)]
    assert seen == [0, 1, 4]                                  # no view where the π0.5 budget is spent (amendment 15)
    h = H.EpisodeHistory(["toy_motion"])
    h.push(H.offline_view(rows[1], thresholds=th))
    with pytest.raises(ValueError, match="order"):
        h.push(H.offline_view(rows[0], thresholds=th))
    # the router observes the common inputs only: a common-only feature works, a cascader feature is refused
    view = H.offline_view(rows[0], thresholds=th, groups=("common",))
    router_live = H.live_view(_observed(rows[0], view, profile="router"))
    assert router_live["proposal"] is None and H.groups_of(router_live) == ("common",)
    assert "hist_toy_motion" in H.EpisodeHistory(["toy_motion"]).push(router_live)
    with pytest.raises(ValueError, match="does not observe"):
        H.EpisodeHistory(["toy_track"]).push(router_live)
    assert H.router_safe("toy_motion") and not H.router_safe("toy_track")
    with pytest.raises(KeyError, match="not registered"):
        H.EpisodeHistory(["nope"])
    for bad in (dict(name="toy_motion", width=1, uses=["common"]), dict(name="x y", width=1, uses=["common"]),
                dict(name="w0", width=0, uses=["common"]), dict(name="g", width=1, uses=["features"])):
        with pytest.raises(ValueError):
            H.register(bad["name"], width=bad["width"], uses=bad["uses"])

    @H.register("toy_wrong_width", width=3, uses=["common"])
    def wrong(views):
        return [1.0]
    with pytest.raises(ValueError, match="3 finite numbers"):
        H.EpisodeHistory(["toy_wrong_width"]).push(H.offline_view(rows[0], thresholds=th))


def test_build_adds_history_columns_only_when_asked(toy_history):
    runs, events, points, label_map, kw = _records()
    plain = D.build(runs, events, points, label_map, **kw)
    assert not any(k.startswith("hist_") for k in plain.cols)
    earlier = [dict(points[0], index=k, step=10 * k, snapshot_id=f"ep/dp{k:02d}", budget_remaining_steps=260,
                    robot_state=[float(k)] * 11) for k in (0, 1)]
    ps = D.build(runs, events, points + earlier, label_map, history=["toy_motion"], **kw)
    assert ps.cols["hist_toy_motion"].shape == (2, 2) and "hist_toy_track" not in ps.cols
    # episode "ep" has two earlier measured points (unlabelled), "ep2" none: its history is the point alone
    assert ps.cols["hist_toy_motion"][0][0] > 0 and list(ps.cols["hist_toy_motion"][1]) == [0.0, 0.0]
    for k in plain.cols:
        assert np.array_equal(ps.cols[k], plain.cols[k])


def test_a_decider_variant_with_history_trains_under_its_profile(ps, toy_history):
    """A variant is a new name with its own columns, trained and scored under an existing profile; the four
    deciders of §8 stay as they are."""
    cols = {**ps.cols, "hist_toy_motion": np.random.default_rng(3).normal(size=(len(ps), 2))}
    ps2 = D.PointSet(cols, ps.task, ps.snapshot_id, ps.can_escalate, ps.remaining_s, ps.y, ps.excluded, ps.weight,
                     ps.episode)
    DC.add_variant("router_hist", profile="router", columns=COLUMNS["router"] + ("hist_toy_motion",))
    assert width(ps2.cols, "router_hist") == width(ps2.cols, "router") + 2
    s = run_cv(ps2, folds=[0], deciders=["router", "router_hist"], cfg=FAST)
    base, var = s["results"]
    assert (var["profile"], var["width"]) == ("router", base["width"] + 2) and base["profile"] == "router"
    assert var["points"] == base["points"] and var["points"]["excluded"] == int(ps.excluded["router"].sum())
    assert var["baseline"]["p"] == base["baseline"]["p"] and var["baseline"]["val_loss"] == base["baseline"]["val_loss"]
    with pytest.raises(ValueError, match="router's profile"):                 # §6: nothing derived from the proposal
        DC.add_variant("router_bad", profile="router", columns=COLUMNS["router"] + ("hist_toy_track",))
    with pytest.raises(ValueError, match="router's profile"):
        DC.add_variant("router_bad", profile="router", columns=COLUMNS["router"] + ("proposal",))
    DC.add_variant("nospread_hist", profile="cascader_nospread",
                   columns=COLUMNS["cascader_nospread"] + ("hist_toy_motion", "hist_toy_track"))
    for bad in (dict(name="router", profile="router", columns=()), dict(name="v", profile="none", columns=())):
        with pytest.raises(ValueError):
            DC.add_variant(bad["name"], profile=bad["profile"], columns=bad["columns"])


# ---------------------------------------------------------------- the 2 x 2: memory by X-VLA information (`variants`)

from vlar_train import variants as V  # noqa: E402


@pytest.fixture
def two_by_two():
    before = (dict(H.FEATURES), dict(COLUMNS), dict(DC.PROFILE_OF))
    V.install()
    yield
    for now, was in zip((H.FEATURES, COLUMNS, DC.PROFILE_OF), before):
        now.clear()
        now.update(was)


def _line_row(k, *, actual=0.01, cmd=0.02, close_from=None, run_id="ep", **over):
    """Point k of an episode whose arm moves `actual` metres along x per segment while each proposal asks for
    `cmd` metres in ten equal steps: identity rotation, fingers 4 cm apart, gripper commanded open, or closed from
    step `close_from` of the proposal on."""
    pos = np.array([actual * k, 0.0, 0.1])
    prop = np.zeros((10, 7), dtype=np.float32)
    prop[:, :3] = pos + np.outer(np.arange(1, 11) / 10.0, [cmd, 0.0, 0.0])
    prop[:, 6] = -1.0
    if close_from is not None:
        prop[close_from:, 6] = 1.0
    row = {"run_id": run_id, "index": k, "step": 10 * k, "snapshot_id": f"{run_id}/dp{k:02d}",
           "budget_remaining_steps": 260, "robot_state": list(pos) + [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.02, -0.02],
           "proposal": [float(x) for x in prop.ravel()], "chunk_pos": 10 * (k % 3), "prev_segment_pi05": False,
           "track_pos_err": 0.004, "track_rot_err": 0.02, "tracking_valid": k > 0,
           "prev_actions": [0, 0, 0, 0, 0, 0, -1.0] * 2, "prev_gripper_qpos": [0.02, -0.02, 0.02, -0.02],
           "features_main": [0.0] * 1024, "features_wrist": [0.0] * 1024}
    return {**row, **over}


def _walk(rows, names=None, th=(0.003, 0.07)):
    h, out = H.EpisodeHistory(V.HISTORY if names is None else names), {}
    for row, view in H.episode_views(rows, thresholds=th):
        out[row["index"]] = h.push(view)
    return out


def test_the_two_by_two_registers_two_variants_and_changes_no_decider(two_by_two):
    spec = {d: COLUMNS[d] for d in DECIDERS}
    V.install()                                                            # twice: nothing more happens
    assert {d: COLUMNS[d] for d in DECIDERS} == spec and {d: DC.PROFILE_OF[d] for d in DECIDERS} == {d: d for d in DECIDERS}
    assert set(COLUMNS) == set(DECIDERS) | {"router_hist", "cascader_hist"}
    assert DC.profile_of("router_hist") == "router" and DC.profile_of("cascader_hist") == "cascader_nospread"
    assert COLUMNS["router_hist"] == COLUMNS["router"] + ("hist_motion", "hist_fingers")
    assert COLUMNS["cascader_hist"][:len(COLUMNS["cascader_nospread"])] == COLUMNS["cascader_nospread"]
    # memory is the same on both sides; everything else on the cascader side reads one of X-VLA's commands
    assert all(H.router_safe(n) for n in V.MEMORY) and not any(H.router_safe(n) for n in V.INTENT + V.PLAN)
    assert set(V.MEMORY) <= set(V.CASCADER_HIST) and set(V.ROUTER_HIST) == set(V.MEMORY)
    assert all("proposal" in H.FEATURES[n].uses for n in V.INTENT + V.PLAN)
    assert {n: H.FEATURES[n].width for n in V.HISTORY} == {n: V.WIDTHS[n] for n in V.HISTORY}
    cols = V.with_synthetic_history(D.synthetic(n_per_task=2)).cols
    assert [width(cols, d) for d in V.TWO_BY_TWO] == [2062, 2075, 2139, 2179]


def test_the_history_features_by_hand(two_by_two):
    """An arm that moves 1 cm per segment while X-VLA asks for 2 cm, in a straight line."""
    rows = [_line_row(k) for k in range(8)]
    c = _walk(rows)[7]
    assert c["hist_motion"] == pytest.approx([0.03, 0.03, 0.03 / 0.0301, 0.0, 0.0, 0.06, 0.06, 0.06 / 0.0601, 0.0, 0.0],
                                             abs=1e-9)
    assert c["hist_fingers"] == pytest.approx([0.04, 0.04, 0.04])
    assert c["hist_track_norm"] == pytest.approx([2.0, 20.0, 0.002, 0.0], rel=1e-5, abs=1e-8)   # 4 mm error on 2 mm steps
    assert c["hist_progress"] == pytest.approx([0.01 / 0.021, 0.03 / 0.061, 0.06 / 0.121, 1.0], abs=1e-6)
    assert c["hist_stall"] == pytest.approx([0.06, 0.12, 0.03, 0.06], abs=1e-7)
    assert c["hist_vertical"] == pytest.approx([0.0, 0.0], abs=1e-7)
    assert c["hist_plan_phase"] == pytest.approx([0.0, 1.0, 0.0, 0.0, 0.002, 0.0], abs=1e-7)
    # the first point has no past: neutral values
    c0 = _walk(rows)[0]
    assert list(c0["hist_vertical"]) == [0.0, 0.0] and list(c0["hist_progress"]) == [1.0, 1.0, 1.0, 1.0]
    assert list(c0["hist_stall"]) == [0.0] * 4 and list(c0["hist_fingers"]) == pytest.approx([0.04] * 3)
    assert list(c0["hist_motion"]) == [0.0] * 10 and list(c0["hist_track_norm"]) == [0.0] * 4
    # a blocked arm (asked for 2 cm, moves nothing) against a hesitating one (asked for nothing)
    blocked = _walk([_line_row(k, actual=0.0) for k in range(8)])[7]
    waiting = _walk([_line_row(k, actual=0.0, cmd=0.0) for k in range(8)])[7]
    assert list(blocked["hist_motion"]) == list(waiting["hist_motion"])               # memory cannot tell them apart
    assert blocked["hist_stall"] == pytest.approx([0.06, 0.12, 0.06, 0.12], abs=1e-7)
    assert waiting["hist_stall"] == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-7)
    assert blocked["hist_progress"][0] == pytest.approx(0.0)
    # an arm sent 3 cm lower than it gets, each segment; fingers that shut for one segment, 40 steps ago
    low = [_line_row(k) for k in range(8)]
    for r in low:
        prop = np.array(r["proposal"]).reshape(10, 7)
        prop[:, 2] -= 0.03 * np.arange(1, 11) / 10.0
        r["proposal"] = [float(x) for x in prop.ravel()]
    low[3]["robot_state"][9:] = [0.001, -0.001]
    c = _walk(low)[7]
    assert c["hist_vertical"] == pytest.approx([-0.03, -0.03], abs=1e-6)
    assert c["hist_fingers"] == pytest.approx([0.04, 0.04, 0.002]) and c["hist_motion"][9] == pytest.approx(0.0)
    # a proposal that closes the gripper from its fifth step on, 12 cm lower
    rows = [_line_row(k) for k in range(3)] + [_line_row(3, close_from=4)]
    prop = np.array(rows[3]["proposal"]).reshape(10, 7)
    prop[:, 2] -= 0.12 * np.arange(1, 11) / 10.0
    rows[3]["proposal"] = [float(x) for x in prop.ravel()]
    c = _walk(rows)[3]
    step = float(np.hypot(0.002, 0.012))
    assert c["hist_plan_phase"] == pytest.approx([1.0, 0.4, 0.6, 0.6 * 0.04, step, -0.12], abs=1e-6)


def test_the_gripper_command_history_by_hand(two_by_two):
    """Segments 1 to 6 after point 0: open; a close that shuts on nothing (failed); open; a close that holds;
    still closed, holding; still closed, then the fingers shut (a drop). Free signals at point k describe the
    segment proposed at point k - 1."""
    closes = {1: 0, 3: 0, 4: 0, 5: 0}                       # proposals commanded closed from their first step
    shut = {2: 1.0, 6: 0.4}
    part = {4: 1.0, 5: 1.0, 6: 0.6}
    rows = [_line_row(k, close_from=closes.get(k)) for k in range(7)]
    seen = {}
    h = H.EpisodeHistory(["grip_events"])
    for row, view in H.episode_views(rows, thresholds=(0.003, 0.07)):
        view = {**view, "grip_shut_frac": shut.get(row["index"], 0.0), "grip_partial_frac": part.get(row["index"], 0.0)}
        seen[row["index"]] = list(h.push(view)["hist_grip_events"])
    #                 closes opens failed drops regrasps ever_held steps_since_held
    assert seen[0] == [0, 0, 0, 0, 0, 0, 0] and seen[1] == [0, 0, 0, 0, 0, 0, 10]
    assert seen[2] == [1, 0, 1, 0, 0, 0, 20]                # closed on nothing
    assert seen[3] == [1, 1, 1, 0, 0, 0, 30]                # opened again
    assert seen[4] == [2, 1, 1, 0, 1, 1, 0]                 # a second close, after a failed one, holds
    assert seen[5] == [2, 1, 1, 0, 1, 1, 0]
    assert seen[6] == [2, 1, 2, 1, 1, 1, 0]                 # held, then shut: a drop, which also counts as a close
    #                                                         during which the fingers shut; held for half the segment
    # after a π0.5 segment the commands of that segment are unknown: it adds nothing
    rows[3] = _line_row(3, close_from=0, tracking_valid=False, prev_segment_pi05=True)
    h = H.EpisodeHistory(["grip_events"])
    for row, view in H.episode_views(rows[:4], thresholds=(0.003, 0.07)):
        out = list(h.push({**view, "grip_shut_frac": 1.0 if row["index"] == 3 else 0.0})["hist_grip_events"])
    assert out == [1, 0, 0, 0, 0, 0, 30]                    # the close of segment 2 only; segment 3's shut fingers
    #                                                         are not held against a command X-VLA did not execute


def test_a_segment_x_vla_did_not_drive_as_proposed_counts_for_nothing(two_by_two):
    """After a π0.5 segment (tracking invalid) the intent features of that segment are neutral; memory still counts
    the motion. A point where the budget is spent leaves a gap, across which no segment is formed."""
    rows = [_line_row(k) for k in range(6)]
    rows[4] = _line_row(4, tracking_valid=False, prev_segment_pi05=True, track_pos_err=0.0, track_rot_err=0.0)
    c = _walk(rows)
    assert c[4]["hist_vertical"][0] == 0.0 and list(c[4]["hist_track_norm"]) == [0.0] * 4
    assert c[4]["hist_progress"][0] == 1.0 and c[4]["hist_progress"][3] == 1.0
    assert c[4]["hist_stall"][0] == pytest.approx(0.04) and c[5]["hist_stall"][0] == pytest.approx(0.04)   # two of three
    assert c[4]["hist_motion"][0] == pytest.approx(0.03)
    assert c[5]["hist_progress"][0] == pytest.approx(0.01 / 0.021, abs=1e-6)
    gap = [_line_row(0), _line_row(1), _line_row(2, budget_remaining_steps=0), _line_row(3)]
    c = _walk(gap)
    assert sorted(c) == [0, 1, 3] and c[3]["hist_progress"][0] == 1.0 and list(c[3]["hist_track_norm"]) == [0.0] * 4
    assert c[3]["hist_motion"][0] == pytest.approx(0.03) and c[3]["hist_stall"][0] == pytest.approx(0.02)


def test_the_two_by_two_history_is_the_same_offline_and_live(two_by_two):
    th = (0.003, 0.07)
    rng = np.random.default_rng(5)
    rows = []
    for k in range(9):
        r = _line_row(k, actual=float(rng.uniform(0, 0.02)), cmd=float(rng.uniform(0, 0.03)),
                      close_from=int(rng.integers(0, 10)) if k % 2 else None)
        if k == 5:
            r.update(tracking_valid=False, prev_segment_pi05=True, track_pos_err=0.0, track_rot_err=0.0)
        rows.append(r)
    off, live, router = H.EpisodeHistory(V.CASCADER_HIST), H.EpisodeHistory(V.CASCADER_HIST), H.EpisodeHistory(V.ROUTER_HIST)
    for row, view in H.episode_views(rows, thresholds=th):
        a, b = off.push(view), live.push(H.live_view(_observed(row, view)))
        assert set(a) == {H.column(n) for n in V.CASCADER_HIST}
        for k in a:
            assert np.array_equal(a[k], b[k]), k
        # the router's live view (common inputs only) gives the same memory columns, and nothing else is offered
        r = router.push(H.live_view(_observed(row, view, profile="router")))
        assert set(r) == {"hist_motion", "hist_fingers"}
        assert all(np.array_equal(r[k], a[k]) for k in r)
    with pytest.raises(ValueError, match="does not observe"):
        H.EpisodeHistory(["progress"]).push(H.live_view(_observed(rows[0], H.offline_view(rows[0], thresholds=th),
                                                                  profile="router")))


def test_build_adds_the_two_by_two_history_from_the_parents_points(two_by_two):
    runs, events, points, label_map, kw = _records()
    labelled = {p["run_id"]: p for p in points}
    pts = []
    for ep in ("ep", "ep2"):
        for k in range(3):
            row = _line_row(k, run_id=ep, budget_remaining_steps=20 if k == 2 else 260)
            pts.append({**labelled[ep], **row} if k == 2 else row)
    ps = D.build(runs, events, pts, label_map, history=V.HISTORY, **kw)
    assert len(ps) == 2 and all(ps.cols[H.column(n)].shape == (2, V.WIDTHS[n]) for n in V.HISTORY)
    want = _walk([p for p in pts if p["run_id"] == "ep"], th=kw["thresholds"])[2]
    for k, v in want.items():
        assert np.array_equal(ps.cols[k][0], v)
    assert inputs(ps.cols, ps.remaining_s["cascader_nospread"], "cascader_hist").shape == (2, 2179)


def test_the_two_by_two_trains_on_the_same_rows_and_differs_only_in_inputs(ps, two_by_two, tmp_path):
    ps2 = V.with_synthetic_history(ps)
    for k in ps.cols:
        assert np.array_equal(ps2.cols[k], ps.cols[k])
    s = run_cv(ps2, folds=[0], deciders=list(V.TWO_BY_TWO), cfg=FAST)
    r = {x["decider"]: x for x in s["results"]}
    assert [r[d]["profile"] for d in V.TWO_BY_TWO] == ["router", "router", "cascader_nospread", "cascader_nospread"]
    for plain, hist in (("router", "router_hist"), ("cascader_nospread", "cascader_hist")):
        assert r[hist]["points"] == r[plain]["points"] and r[hist]["baseline"]["p"] == r[plain]["baseline"]["p"]
        point = lambda d: {s: {k: v for k, v in d[s].items() if not k.endswith("ci95")} for s in d}   # intervals are
        assert point(r[hist]["data"]) == point(r[plain]["data"])                   # bootstrapped with a seed per decider
        assert r[hist]["width"] > r[plain]["width"]
    # the entry point: a synthetic dry run of the four variants; without --variants nothing is registered
    R.main(["--synthetic", "--variants", "2x2", "--folds", "0", "--epochs", "2", "--out", str(tmp_path / "v")])
    done = json.loads((tmp_path / "v" / "summary.json").read_text())
    assert [x["decider"] for x in done["results"]] == list(V.TWO_BY_TWO)


# ---------------------------------------------------------------- the planned benefit analysis (`benefit`)

from vlar_train import benefit as B  # noqa: E402


def _write_task(out, task, *, p, y, lam, weight=None, episode=None, mask=None, can=None, variants=B.VARIANTS, steps=None):
    """A fold's saved test predictions and λ, as `run_fold_decider` writes them. `p` and `lam`: per variant, or one
    for all."""
    n = len(y)
    sid = np.array([f"e{task}/x/dp{(k if steps is None else steps[k]):02d}" for k in range(n)])
    for v in variants:
        d = out / f"fold{task}" / v
        d.mkdir(parents=True, exist_ok=True)
        extra = {} if can is None else {"can_escalate": np.asarray(can, bool)}
        np.savez(d / "test_predictions.npz", p=np.asarray(p[v] if isinstance(p, dict) else p, float),
                 y=np.asarray(y, float), mask=np.ones((n, 4)) if mask is None else np.asarray(mask, float),
                 weight=np.ones(n) if weight is None else np.asarray(weight, float),
                 episode=np.array([f"ep{k}" for k in range(n)] if episode is None else episode), snapshot_id=sid, **extra)
        (d / "result.json").write_text(json.dumps({"lambda": {"lambda": lam[v] if isinstance(lam, dict) else lam}}))


def _pred(scores, p0=0.5):
    """Four-head predictions whose score (p_one - p_continue) is `scores`."""
    s = np.asarray(scores, float)
    return np.stack([np.full(len(s), p0), p0 + s, np.zeros(len(s)), np.zeros(len(s))], axis=1)


def _labels(b1, b2):
    return np.stack([np.asarray(b1, float), np.asarray(b2, float), np.zeros(len(b1)), np.zeros(len(b1))], axis=1)


def test_gain_at_a_fixed_rate_interpolates_across_tied_scores():
    score, w = np.array([0.4, 0.0, 0.0, -0.4]), np.ones(4)
    d = np.array([1.0, 1.0, 0.0, 0.0])
    gain = lambda r: float((B.escalation_at_rate(score, w, r) * d).sum() / 4)
    assert list(B.escalation_at_rate(score, w, 0.0)) == [0, 0, 0, 0] and gain(0.0) == 0.0
    assert list(B.escalation_at_rate(score, w, 0.25)) == [1, 0, 0, 0] and gain(0.25) == 0.25
    assert list(B.escalation_at_rate(score, w, 0.5)) == [1, 0.5, 0.5, 0]      # half of the two tied points
    assert gain(0.5) == pytest.approx(0.375)                                  # between 0.25 (rate 0.25) and 0.5 (0.75)
    assert list(B.escalation_at_rate(score, w, 1.0)) == [1, 1, 1, 1]
    # the same as linear interpolation of `lam.curve`, with weights, at every rate
    rng = np.random.default_rng(1)
    s = np.round(rng.normal(size=60), 1) / 10                                 # many ties
    y = _labels(rng.integers(0, 4, 60) / 3, rng.integers(0, 4, 60) / 3)
    w = rng.uniform(0.5, 8, 60)
    t = {"n": 60, "p": {"v": _pred(s)}, "y0": y[:, 0], "y1": y[:, 1], "w": w, "episode": np.arange(60)}
    rate, g = B.curve_points(t, "v")
    assert rate[0] == 0.0 and g[0] == 0.0 and rate[-1] == pytest.approx(1.0) and (np.diff(rate) > 0).all()
    for r in (0.0, 0.03, 0.1, 0.2, 0.37, 0.5, 0.99):
        e = B.escalation_at_rate(s, w, r)
        assert float((w * e).sum() / w.sum()) == pytest.approx(r)
        assert float((w * e * (y[:, 1] - y[:, 0])).sum() / w.sum()) == pytest.approx(float(np.interp(r, rate, g)))


def test_the_benefit_analysis_by_hand(tmp_path):
    """Two tasks of ten points, each its own episode. Task 0: the cascader variants rank the two rescues first, the
    router variants rank them last. Task 1 has no positive point (one harm). Task 2 wrote nothing."""
    b1 = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    b2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    good, bad = np.linspace(0.9, 0.0, 10), np.linspace(0.0, 0.9, 10)
    p = {"router": _pred(bad), "router_hist": _pred(bad), "cascader_nospread": _pred(good), "cascader_hist": _pred(good)}
    lam = {"router": float("inf"), "router_hist": 0.85, "cascader_nospread": 0.75, "cascader_hist": -1.0}
    _write_task(tmp_path, 0, p=p, y=_labels(b1, b2), lam=lam, can=np.ones(10, bool), steps=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    _write_task(tmp_path, 1, p=p, y=_labels([1] * 10, [0] + [1] * 9), lam=lam)
    res = B.analyse(tmp_path, tasks=[0, 1, 2], resamples=200)
    assert res["tasks_used"] == [0, 1] and res["tasks_left_out"] == [2]
    t0, t1 = res["per_task"]
    assert (t0["points"], t0["positives"], t0["positive_episodes"], t0["can_escalate_saved"]) == (10, 2, 2, True)
    assert (t1["positives"], t1["can_escalate_saved"]) == (0, False) and t1["mean_b2_minus_b1"] == pytest.approx(-0.1)
    v = t0["values"]
    # 10%: one point. The cascaders escalate a rescue (gain 1/10); the routers a point that needed none
    assert v["cascader_hist"]["gain_at_10"] == pytest.approx(0.1) and v["router"]["gain_at_10"] == pytest.approx(0.0)
    assert v["cascader_hist"]["gain_at_20"] == pytest.approx(0.2) and v["router_hist"]["gain_at_20"] == pytest.approx(0.0)
    # random at 10%: 0.1 x mean(b2 - b1) = 0.1 x 0.2
    assert t0["random_at_10"] == pytest.approx(0.02) and v["cascader_hist"]["excess_at_10"] == pytest.approx(0.08)
    # at λ: never escalate is gain 0 at rate 0; λ 0.85 escalates the router's top point; λ -1 escalates everything
    assert (v["router"]["gain_at_lambda"], v["router"]["rate_at_lambda"]) == (0.0, 0.0)
    assert (v["router_hist"]["gain_at_lambda"], v["router_hist"]["rate_at_lambda"]) == pytest.approx((0.0, 0.1))
    assert (v["cascader_nospread"]["gain_at_lambda"], v["cascader_nospread"]["rate_at_lambda"]) == pytest.approx((0.2, 0.2))
    assert (v["cascader_hist"]["gain_at_lambda"], v["cascader_hist"]["rate_at_lambda"]) == pytest.approx((0.2, 1.0))
    # failure AUROC, step-stratified: the two failing points are at step 0, against three others there
    assert v["cascader_hist"]["failure_auroc"] == pytest.approx(0.5)          # p_continue is constant: all ties
    # task 1: the cascaders' first point is the harm; no failing point, so the AUROC is undefined there
    w = t1["values"]
    assert w["cascader_hist"]["gain_at_10"] == pytest.approx(-0.1) and w["router"]["gain_at_10"] == pytest.approx(0.0)
    assert w["cascader_hist"]["excess_at_10"] == pytest.approx(-0.1 + 0.01)
    assert res["failure_auroc_undefined_tasks"]["router"] == [1]
    # pooled: the unweighted mean of the two tasks; the AUROC over the task where it is defined
    pool = res["pooled"]
    assert pool["cascader_hist"]["gain_at_10"]["value"] == pytest.approx(0.0)
    assert pool["cascader_hist"]["gain_at_20"]["value"] == pytest.approx((0.2 - 0.1) / 2)
    assert pool["router"]["failure_auroc"]["value"] == pytest.approx(0.5)
    lo, hi = pool["cascader_hist"]["gain_at_20"]["ci95"]
    assert lo <= 0.05 <= hi and lo < hi
    # contrasts, paired: the primary one is flagged, and the interaction is the difference of the two X-VLA contrasts
    c = res["contrasts"]
    assert set(c) == {"memory", "xvla_given_memory", "xvla_without_memory", "interaction"}
    assert c["memory"]["gain_at_20"]["value"] == pytest.approx(0.0) and c["memory"]["gain_at_20"]["ci95"] == [0.0, 0.0]
    assert c["xvla_given_memory"]["gain_at_20"]["value"] == pytest.approx(0.05)
    assert c["interaction"]["gain_at_20"]["value"] == pytest.approx(0.0)
    flagged = [(k, m) for k in c for m in c[k] if c[k][m]["primary"]]
    assert flagged == [("xvla_given_memory", "gain_at_10")] and res["primary"]["value"] == pytest.approx(0.0)
    assert res["intervals_reported"] == 4 * 8 + 4 * 8
    assert res["curve"]["rate"][0] == 0.0 and res["curve"]["rate"][-1] == 0.5
    assert res["curve"]["gain"]["cascader_hist"][20] == pytest.approx(0.05)   # rate 0.2: the pooled gain at 20%
    text = B.table(res)
    assert "PRIMARY: xvla_given_memory, gain_at_10" in text and "left out (no usable test point): [2]" in text
    assert chr(8212) not in text


def test_the_benefit_analysis_refuses_rows_that_differ_or_points_not_offered(tmp_path):
    y, p = _labels([0, 1, 1], [1, 1, 1]), _pred([0.3, 0.2, 0.1])
    _write_task(tmp_path / "a", 0, p=p, y=y, lam=0.0)
    np.savez(tmp_path / "a" / "fold0" / "cascader_hist" / "test_predictions.npz", p=p, y=y, mask=np.ones((3, 4)),
             weight=np.array([1.0, 2.0, 1.0]), episode=np.array(["ep0", "ep1", "ep2"]),
             snapshot_id=np.array(["e0/x/dp00", "e0/x/dp01", "e0/x/dp02"]))
    with pytest.raises(B.AnalysisError, match="not on the same test rows .weight"):
        B.analyse(tmp_path / "a", tasks=[0], resamples=5)
    _write_task(tmp_path / "b", 0, p=p, y=y, lam=0.0, can=[True, False, True])
    with pytest.raises(B.AnalysisError, match="not offered"):
        B.analyse(tmp_path / "b", tasks=[0], resamples=5)
    _write_task(tmp_path / "c", 0, p=p, y=y, lam=0.0, variants=B.VARIANTS[:2])
    with pytest.raises(B.AnalysisError, match="but not for"):
        B.analyse(tmp_path / "c", tasks=[0], resamples=5)
    with pytest.raises(B.AnalysisError, match="no test predictions"):
        B.analyse(tmp_path / "d", tasks=[0], resamples=5)
    # a point without a branch-2 label is not usable; a task left with none is left out
    _write_task(tmp_path / "e", 0, p=p, y=y, lam=0.0, mask=[[1, 0, 0, 0]] * 3)
    _write_task(tmp_path / "e", 1, p=p, y=y, lam=0.0, mask=[[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
    res = B.analyse(tmp_path / "e", tasks=[0, 1], resamples=5)
    assert res["tasks_left_out"] == [0] and res["per_task"][0]["points"] == 2


def test_the_resamples_are_those_of_the_cluster_bootstrap_and_are_shared():
    rng = np.random.default_rng(2)
    ep = rng.integers(0, 12, 80)
    v, w = rng.normal(size=80), rng.uniform(1, 5, 80)
    counts = B.resample_counts(ep, seed=11, resamples=300)
    assert counts.shape == (300, 80) and all(len(set(c[ep == e])) == 1 for c in counts[:5] for e in range(12))
    mine = np.percentile([(c * w * v).sum() / (c * w).sum() for c in counts], [2.5, 97.5])
    theirs = ST.cluster_bootstrap(lambda r: ST.weighted_mean(v[r], w[r]), ep, seed=11, resamples=300)
    assert list(mine) == pytest.approx(theirs)
    assert np.array_equal(counts, B.resample_counts(ep, seed=11, resamples=300))          # seeded
    assert np.array_equal(B.resample_counts(["one"] * 4, seed=1, resamples=3), np.ones((3, 4)))   # nothing to resample


def test_the_profile_check_stops_when_the_two_profiles_differ(ps, two_by_two):
    with pytest.raises(B.AnalysisError, match="labels.*exclusions.*remaining time"):
        B.check_same_profiles(ps)                                # the synthetic set draws each profile apart
    same = V.same_profiles(ps)
    assert B.check_same_profiles(same)["points"] == len(ps)
    assert np.array_equal(same.y["router"], ps.y["router"]) and np.array_equal(same.cols["proposal"], ps.cols["proposal"])
    nan = D.PointSet(same.cols, same.task, same.snapshot_id, same.can_escalate, same.remaining_s,
                     {k: np.where(np.arange(4) >= 2, np.nan, v) for k, v in same.y.items()}, same.excluded,
                     same.weight, same.episode)
    B.check_same_profiles(nan)                                   # labels absent in both are the same
    off = {**same.remaining_s, "cascader_nospread": same.remaining_s["router"] + np.r_[1e-9, np.zeros(len(ps) - 1)]}
    with pytest.raises(B.AnalysisError, match="remaining time: 1 of"):
        B.check_same_profiles(D.PointSet(same.cols, same.task, same.snapshot_id, same.can_escalate, off, same.y,
                                         same.excluded, same.weight, same.episode))


def test_the_dry_run_goes_from_training_to_the_benefit_report(tmp_path, capsys, two_by_two):
    out = tmp_path / "run"
    R.main(["--synthetic", "--variants", "2x2", "--epochs", "2", "--out", str(out)])
    assert "profile check:" in capsys.readouterr().out
    z = np.load(out / "fold3" / "cascader_hist" / "test_predictions.npz")
    assert z["can_escalate"].dtype == bool and len(z["can_escalate"]) == len(z["p"])
    B.main(["--out", str(out), "--resamples", "40"])
    res = json.loads((out / "benefit.json").read_text())
    assert res["tasks_used"] == list(range(10)) and res["tasks_left_out"] == [] and res["variants"] == list(B.VARIANTS)
    for v in B.VARIANTS:
        for m in B.METRICS[:7]:
            cell = res["pooled"][v][m]
            assert math.isfinite(cell["value"]) and cell["ci95"][0] <= cell["ci95"][1]
        assert res["pooled"][v]["gain_at_10"]["value"] == pytest.approx(
            np.mean([t["values"][v]["gain_at_10"] for t in res["per_task"]]))
    assert res["primary"]["contrast"] == "xvla_given_memory" and res["primary"]["ci95"] is not None
    assert (out / "benefit.txt").read_text().startswith("Benefit analysis of the 2 x 2")
