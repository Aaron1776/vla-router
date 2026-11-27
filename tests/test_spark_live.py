"""ops/spark/live.py and live_check.py without a simulator: history variants load, widths, and the capture format.
Needs numpy, torch and a vlar_train with `variants` (the 2 x 2); skipped elsewhere."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
V = pytest.importorskip("vlar_train.variants")
sys.path.insert(0, str(Path(__file__).parents[1] / "ops" / "spark"))
import live  # noqa: E402

WIDTH = {"router": 2062, "router_hist": 2075, "cascader_nospread": 2139, "cascader_hist": 2179}


def obs(index: int, cascader: bool, seed: int = 0):
    r = np.random.default_rng([seed, index])
    state = np.concatenate([r.normal(scale=0.1, size=3), [1, 0, 0, 0, 1, 0], r.uniform(0, 0.04, 2)])
    common = NS(features_main=r.normal(size=1024), features_wrist=r.normal(size=1024), robot_state=state,
                budget_remaining_steps=100, time_remaining_s=40.0 - index, steps_elapsed=10 * index)
    point = NS(index=index, step=10 * index, can_escalate=True)
    prop = NS(actions=r.normal(scale=0.05, size=(10, 7)), chunk_pos=10 * (index % 2), prev_segment_pi05=False)
    free = NS(track_pos_err=0.01, track_rot_err=0.02, tracking_valid=index > 0, grip_closed_shut_frac=0.0,
              grip_closed_partial_frac=0.1)
    return NS(common=common, point=point, proposal=prop if cascader else None, free_signals=free if cascader else None,
              chunk_spread=None, fresh_spread=None, drift=None)


@pytest.mark.parametrize("name", list(WIDTH))
def test_dummy_decider_has_the_variant_width_and_builds_its_columns(name):
    from vlar_train import deciders as D
    d = live.RandomDecider(name)
    assert d.members[0][1].shape == (WIDTH[name],) and d.profile == ("router" if name.startswith("router") else "cascader_nospread")
    d.new_episode()
    for k in range(3):
        choice, info = d.decide(obs(k, cascader=not name.startswith("router")))
        cols, x, rem = d.last_inputs
        assert choice in ("local", "escalate") and set(info) == {"p_continue", "p_one_call"}
        assert tuple(cols) == D.COLUMNS[name] and x.shape == (1, WIDTH[name]) and rem == 40.0 - k
        assert np.isfinite(x).all() and x[0, -1] == np.float32(rem / 100.0)


def fold_dir(tmp_path, name, n_in):
    from vlar_train import deciders as D
    d = tmp_path / name
    (d / "checkpoints").mkdir(parents=True)
    model = D.make_model(n_in)
    torch.save({"state": model.state_dict(), "mean": np.zeros(n_in), "std": np.ones(n_in), "n_in": n_in,
                "hidden": (512, 256)}, d / "checkpoints" / "epoch002.pt")
    (d / "result.json").write_text(json.dumps({"decider": name, "fold": 0, "best_epoch": 2,
                                               "lambda": {"lambda": None}}))
    return d


def test_a_history_fold_loads_and_a_wrong_width_is_refused(tmp_path):
    d = live.TrainedDecider.load(fold_dir(tmp_path, "cascader_hist", WIDTH["cascader_hist"]))
    assert d.name == "cascader_hist" and d.profile == "cascader_nospread" and d.lam == float("inf")
    assert len(d.history_names) == 8
    d.new_episode()
    assert d.decide(obs(0, cascader=True))[0] == "local"          # λ = +inf: never escalate
    with pytest.raises(ValueError):
        live.TrainedDecider.load(fold_dir(tmp_path / "bad", "router_hist", WIDTH["router"]))
    with pytest.raises(KeyError):
        live.ensure_decider("no_such_decider")


def test_pack_keeps_one_row_per_observed_point():
    import live_check as LC
    a, b = live.RandomDecider("router_hist"), live.RandomDecider("router")
    caps = []
    for d in (a, b):
        d.new_episode()
    for k in (0, 1, 3):
        o = obs(k, cascader=False)
        a.decide(o); b.build(o)
        caps.append({"index": k, a.name: a.last_inputs, b.name: b.last_inputs})
    z = LC.pack(caps)
    assert z["index"].tolist() == [0, 1, 3]
    assert z["router_hist|x"].shape == (3, WIDTH["router_hist"]) and z["router|x"].shape == (3, WIDTH["router"])
    assert z["router_hist|hist_motion"].shape == (3, 10) and z["router|remaining_s"].tolist() == [40.0, 39.0, 37.0]
    assert list(LC.pack([])) == ["index"] and LC.pack([])["index"].size == 0
    assert LC.max_abs(z["router|x"], z["router|x"]) == 0.0 and LC.max_abs(np.zeros(2), np.zeros(3)) == float("inf")


def members_fold(tmp_path, name, n_in, n=5, best=(3, 1, 4, 1, 5)):
    """A fold as vlar_train (3094f4a) writes it with n training seeds: checkpoints/seed<k>/epochNNN.pt per member,
    result.json `training_seeds`; every member with its own weights, mean and std."""
    from vlar_train import deciders as D
    d = tmp_path / name
    for k in range(n):
        torch.manual_seed(100 + k)
        model = D.make_model(n_in)
        r = np.random.default_rng(k)
        (d / "checkpoints" / f"seed{k}").mkdir(parents=True)
        torch.save({"state": model.state_dict(), "mean": r.normal(size=n_in).astype(np.float32),
                    "std": r.uniform(0.5, 2.0, n_in).astype(np.float32), "n_in": n_in, "hidden": (512, 256)},
                   d / "checkpoints" / f"seed{k}" / f"epoch{best[k]:03d}.pt")
    (d / "result.json").write_text(json.dumps({"decider": name, "fold": 0, "best_epoch": float(np.median(best[:n])),
                                               "training_seeds": {"seeds": list(range(n)), "best_epochs": list(best[:n])},
                                               "lambda": {"lambda": 0.1}}))
    return d


@pytest.mark.parametrize("name", ["router", "cascader_hist"])
def test_members_average_bit_for_bit_like_averaged_predict(tmp_path, name):
    from vlar_train import deciders as D
    from vlar_train.train import Averaged, Trained
    fd = members_fold(tmp_path, name, WIDTH[name])
    dec = live.TrainedDecider.load(fd, min_members=5)
    assert dec.meta["members"] == 5 and dec.meta["member_best_epochs"] == [3, 1, 4, 1, 5]
    ref = []
    for k, e in enumerate((3, 1, 4, 1, 5)):
        ck = torch.load(fd / "checkpoints" / f"seed{k}" / f"epoch{e:03d}.pt", weights_only=False)
        m = D.make_model(WIDTH[name]); m.load_state_dict(ck["state"])
        ref.append(Trained(m, ck["mean"], ck["std"], e))
    x = np.random.default_rng(7).normal(size=(6, WIDTH[name])).astype(np.float32)
    assert np.array_equal(dec.predict(x), Averaged(ref, list(range(5))).predict(x))


def test_a_fold_with_a_missing_member_or_too_few_members_is_refused(tmp_path):
    fd = members_fold(tmp_path / "a", "router", WIDTH["router"])
    (fd / "checkpoints" / "seed3" / "epoch001.pt").unlink()
    with pytest.raises(ValueError, match="missing"):
        live.TrainedDecider.load(fd)
    one = fold_dir(tmp_path / "b", "router", WIDTH["router"])
    assert live.TrainedDecider.load(one).meta["members"] == 1
    with pytest.raises(ValueError, match="5 required"):
        live.TrainedDecider.load(one, min_members=5)
