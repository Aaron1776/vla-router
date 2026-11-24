"""The ops-only waiver (vlar_ops.waiver, amendment 49) on the real `write_freeze`/`verify_freeze`.

Two committed archives stand in for phase0/src/: the freeze's (the real `vlar_ops` with a comment-only change to
spend.py, the ops digest the freeze records) and the running one (the real `vlar_ops`). An ops-only difference
with equal data digests is accepted; every other freeze problem is refused, alone and together with an ops mismatch;
so are a missing or uncommitted running source, a missing freeze archive and a changed data digest.
"""

import io
import json
import tarfile
from types import SimpleNamespace

import pytest
import yaml

from test_config import _complete, cfg_dir  # noqa: F401  (fixture)
from vlar_harness import config as C
from vlar_harness import provenance as P
from vlar_ops import datadigest as DD
from vlar_ops import waiver as W
from vlar_ops.storage import LocalStorage

FREEZE_SRC, RUN_SRC = "a" * 40 + ".tar.gz", "b" * 40 + ".tar.gz"


def tar_gz(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for k, v in files.items():
            info = tarfile.TarInfo(f"src/vlar_ops/{k}")
            info.size = len(v)
            tar.addfile(info, io.BytesIO(v))
    return buf.getvalue()


def setup(cfg_dir, tmp_path, monkeypatch, *, where="spend.py", old=None, new=b"\n# frozen with this comment\n"):
    """The freeze's ops code is the real one with `old` replaced by `new` in `where` (appended if `old` is None)."""
    real = DD.files_from_dir(P.ops_dir())
    text = real[where]
    assert old is None or text.count(old) == 1
    at_freeze = {**real, where: text + new if old is None else text.replace(old, new)}
    storage = LocalStorage(tmp_path / "bucket")
    storage.put_bytes(W.ARCHIVES + FREEZE_SRC, tar_gz(at_freeze))
    storage.put_bytes(W.ARCHIVES + RUN_SRC, tar_gz(real))
    cal = tmp_path / "cal" / "calibration.yaml"
    cal.parent.mkdir()
    _complete(cfg_dir, cal)
    monkeypatch.setattr(P, "ops_digest", lambda: DD.ops_digest_of(at_freeze))
    P.write_freeze(["g1"], cal, config_dir=cfg_dir)
    env = SimpleNamespace(cfg_dir=cfg_dir, cal=cal, storage=storage, freeze=cal.with_name("freeze.json"),
                          freeze_ops=DD.ops_digest_of(at_freeze), run_ops=DD.ops_digest_of(real))
    monkeypatch.setattr(P, "ops_digest", lambda: env.run_ops)          # now running other ops code
    return env


def attempt(env, source=RUN_SRC):
    cfg = C.load_config(config_dir=env.cfg_dir, calibration=env.cal)
    try:
        P.verify_freeze(cfg, config_dir=env.cfg_dir)
    except P.FreezeError as e:
        return W.check(cfg, e, env.storage, ops_source=source, config_dir=env.cfg_dir)
    return "verified"


def test_an_ops_only_difference_with_the_same_data_digest_is_accepted(cfg_dir, tmp_path, monkeypatch):
    env = setup(cfg_dir, tmp_path, monkeypatch)
    w = attempt(env)
    assert w["freeze_ops_digest"] == env.freeze_ops != w["ops_digest"] == env.run_ops
    assert w["ops_source"] == RUN_SRC and w["freeze_ops_source"] == FREEZE_SRC
    assert w["data_digest"]["common"] == DD.digests(DD.files_from_dir(P.ops_dir()))["common"]
    assert P.ops_digest() == env.run_ops                                # the patch inside check was undone


def _harness_version(env, mp):
    rec = json.loads(env.freeze.read_text())
    env.freeze.write_text(json.dumps(dict(rec, harness_version="0.9.0")))


def _source_code(env, mp):
    mp.setattr(P, "source_digest", lambda *a: "0" * 64)


def _config_values(env, mp):
    path = env.cfg_dir / "harness-v1.yaml"
    d = yaml.safe_load(path.read_text())
    d["servers"]["xvla"]["port"] = int(d["servers"]["xvla"]["port"]) + 1
    path.write_text(yaml.safe_dump(d, sort_keys=False))


def _calibration_file(env, mp):
    env.cal.write_text(env.cal.read_text() + "# edited after the freeze\n")


def _lineage(env, mp):
    rec = json.loads(env.freeze.read_text())
    env.freeze.write_text(json.dumps(dict(rec, calibration_lineage=rec["calibration_lineage"][1:])))


def _missing_freeze_file(env, mp):
    env.freeze.unlink()


@pytest.mark.parametrize("tamper", [_harness_version, _source_code, _config_values, _calibration_file, _lineage,
                                    _missing_freeze_file])
@pytest.mark.parametrize("ops_mismatch", [False, True])
def test_any_other_freeze_problem_is_refused(cfg_dir, tmp_path, monkeypatch, tamper, ops_mismatch):
    env = setup(cfg_dir, tmp_path, monkeypatch)
    if not ops_mismatch:
        monkeypatch.setattr(P, "ops_digest", lambda: env.freeze_ops)
    tamper(env, monkeypatch)
    with pytest.raises(W.WaiverRefused):
        attempt(env)


@pytest.mark.parametrize("source", [None, "", "b" * 40 + ".zip", "../" + RUN_SRC, FREEZE_SRC, "c" * 40 + ".tar.gz"])
def test_the_running_ops_code_must_come_from_its_committed_archive(cfg_dir, tmp_path, monkeypatch, source):
    env = setup(cfg_dir, tmp_path, monkeypatch)
    with pytest.raises(W.WaiverRefused, match="VLAR_OPS_SOURCE|not in the bucket|running ops code"):
        attempt(env, source)


def test_uncommitted_running_code_is_refused(cfg_dir, tmp_path, monkeypatch):
    env = setup(cfg_dir, tmp_path, monkeypatch)
    monkeypatch.setattr(P, "ops_digest", lambda: "d" * 64)              # edited after its archive was made
    with pytest.raises(W.WaiverRefused, match="running ops code"):
        attempt(env)


def test_the_freezes_archive_must_exist(cfg_dir, tmp_path, monkeypatch):
    env = setup(cfg_dir, tmp_path, monkeypatch)
    (tmp_path / "bucket" / W.ARCHIVES / FREEZE_SRC).unlink()
    with pytest.raises(W.WaiverRefused, match="freeze's ops digest"):
        attempt(env)


@pytest.mark.parametrize("where,old,new", [
    ("vm_worker.py", b'"evaluation/libero/libero_client.py"', b'"evaluation/libero/other.py"'),
    ("vm_worker.py", b'self.pi05.act(raw_obs, task, seed)', b'self.pi05.act(raw_obs, task, None)'),
    ("plan.py", b"return namespace + 100 * task_id + init_idx", b"return namespace + 10 * task_id + init_idx"),
    ("configs/collection-v1.yaml", b"  explore: 2000000", b"  explore: 2100000"),
    ("configs/collection-v1.yaml", None, b"\nextra_definition: 1\n"),
])
def test_a_different_data_digest_is_refused(cfg_dir, tmp_path, monkeypatch, where, old, new):
    env = setup(cfg_dir, tmp_path, monkeypatch, where=where, old=old, new=new)
    with pytest.raises(W.WaiverRefused, match="data digest"):
        attempt(env)


def test_a_tunable_or_generator_difference_is_accepted(cfg_dir, tmp_path, monkeypatch):
    env = setup(cfg_dir, tmp_path, monkeypatch, where="configs/collection-v1-tunables.yaml",
                old=b"p_escalate: 0.1 ", new=b"p_escalate: 0.2 ")
    assert attempt(env)["freeze_ops_source"] == FREEZE_SRC
    env = setup(cfg_dir, tmp_path / "2", monkeypatch, where="branchplan.py", new=b"\n\ndef later():\n    return 1\n")
    frozen = DD.digests(DD.files_from_archive(env.storage.get_bytes(W.ARCHIVES + FREEZE_SRC)))
    w = attempt(env)
    assert w["data_digest"]["common"] == frozen["common"] and w["data_digest"]["branch"] != frozen["branch"]


def test_serve_records_the_boot_its_golden_pass_and_the_waiver(tmp_path, monkeypatch):
    """`vm_worker.serve`'s wiring with everything around it stubbed: a frozen harness under ops-only different code
    runs with the waiver's record, this boot's id and golden pass in every marker's meta; a refused waiver keeps the
    FreezeError; an exploration job whose spec records another p_escalate is given back."""
    pytest.importorskip("numpy")
    pytest.importorskip("pyarrow")
    import vlar_harness.collect as collect
    import vlar_harness.config as hc
    from vlar_harness import reference
    from vlar_ops import golden, vm_worker
    from vlar_ops.plan import Job

    class FakeConfig(dict):
        frozen, write_once_complete, calibration_hash = True, True, "c" * 64
    cfg = FakeConfig(reference={"lerobot_files": {}})
    monkeypatch.setattr(hc, "load_config", lambda **kw: cfg)
    monkeypatch.setattr(reference, "verify_lerobot", lambda files: None)
    monkeypatch.setattr(P, "record_header", lambda c: {"ops_digest": "run"})
    monkeypatch.setattr(vm_worker, "check_servers", lambda *a, **k: {})
    monkeypatch.setattr(vm_worker, "image_check", lambda c: {})          # amendment 54, tested in test_freezesupp
    monkeypatch.setattr(vm_worker, "ProductionEpisodes", lambda *a, **k: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(collect, "Runner", lambda *a, **k: SimpleNamespace(refuse=lambda job: None,
                                                                            job_dir=lambda job: tmp_path / "x"))
    marker = tmp_path / "golden-pass.json"
    marker.write_text(json.dumps({"boot_id": golden.boot_id(), "calibration_hash": cfg.calibration_hash,
                                  "passed": True, "results": []}))
    monkeypatch.setattr(golden, "passed_this_boot", lambda path, cal: True)
    seen = {}

    def run_worker(queue, storage, runner, worker_id, *, refuse, meta, **kw):
        seen["meta"], seen["refuse"] = meta, refuse
        return {"done": 0, "refused": 0}
    monkeypatch.setattr(vm_worker, "run_worker", run_worker)
    ops_only = P.FreezeError("harness differs from frozen 1.0.0 in: ops code (vlar_ops). ...")

    def verify(c):
        raise ops_only
    monkeypatch.setattr(P, "verify_freeze", verify)
    monkeypatch.setattr(W, "check", lambda c, e, storage, ops_source: {"ops_source": ops_source, "error": str(e)})
    monkeypatch.setenv("VLAR_OPS_SOURCE", RUN_SRC)
    ops = {"local_root": str(tmp_path / "root"), "storage_url": str(tmp_path / "bucket"), "golden_marker": str(marker)}
    vm_worker.serve(ops, worker_id="w", clients={"xvla": None, "pi05": None}, p_escalate=0.1)
    meta = seen["meta"]
    assert meta["ops_waiver"] == {"ops_source": RUN_SRC, "error": str(ops_only)}
    assert meta["boot_id"] == golden.boot_id() and meta["golden_pass"]["passed"] is True
    explore = Job("j", "explore", "episode", "explore/t0/i0", 1, 0, 0, {"p_escalate": 0.2})
    assert "0.2" in seen["refuse"](explore)
    assert "without p_escalate" in seen["refuse"](Job("j", "explore", "episode", "s", 1, 0, 0, {}))
    assert seen["refuse"](Job("j", "explore", "episode", "s", 1, 0, 0, {"p_escalate": 0.1})) is None
    assert seen["refuse"](Job("j", "base", "episode", "s", 1, 0, 0, {})) is None

    def other_images(c):
        raise vm_worker.StartupError("frozen harness, images not the frozen ones: in: sim")
    monkeypatch.setattr(vm_worker, "image_check", other_images)        # under the waiver too (amendment 54)
    with pytest.raises(vm_worker.StartupError, match="images not the frozen ones"):
        vm_worker.serve(ops, worker_id="w", clients={"xvla": None, "pi05": None}, p_escalate=0.1)
    monkeypatch.setattr(vm_worker, "image_check", lambda c: {})

    def refuse(c, e, storage, ops_source):
        raise W.WaiverRefused("data digest differs")
    monkeypatch.setattr(W, "check", refuse)
    with pytest.raises(P.FreezeError, match="ops code .* waiver .* does not apply: data digest differs"):
        vm_worker.serve(ops, worker_id="w", clients={"xvla": None, "pi05": None}, p_escalate=0.1)
