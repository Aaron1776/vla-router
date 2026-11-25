"""The dataset manifest and datacard (vlar_ops.manifest, SPEC §13, §15). Needs NumPy and pyarrow: run with the
simulator environment's Python.

A small dataset is written the way the VM writes it (JobOutput -> run_worker -> bucket with markers), then:
a clean tree gets MANIFEST.json and DATACARD.md and verifies; a changed byte, a file no marker lists, a stray file,
mixed harness code, a run row that disagrees with its marker or a missing provenance each stop the build; `verify`
finds missing, changed and extra files; the datacard documents every field of every table.
"""

import json
import re
import shutil

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")
pytest.importorskip("PIL")

from vlar_harness.dataset import TABLES, JobOutput, _schemas
from vlar_ops import manifest
from vlar_ops.plan import episode_plan
from vlar_ops.queue import JobQueue
from vlar_ops.storage import LocalStorage
from vlar_ops.worker import run_worker

IMG = {"commit": "a" * 40, "sim": "sha256:" + "1" * 64, "pi05": "sha256:" + "2" * 64, "xvla": "sha256:" + "3" * 64}
PROV = {"harness_version": "1.0.0", "config_name": "harness-v1", "config_hash": "c" * 64,
        "config_options": {"scene_seed": "project"}, "calibration_hash": "1" * 64, "source_digest": "s" * 64,
        "ops_digest": "o" * 64, "commit": "a" * 40, "harness_dirty": False, "image_digests": IMG}


def runner_for(staging, header):
    def runner(job):
        out = JobOutput(staging, job.phase, job.job_id, header=header)
        rid = job.subject
        out.add_run({"run_id": rid, "run_kind": "episode", "episode_id": rid, "task_id": job.spec["task_id"],
                     "init_idx": job.spec["init_idx"], "mode": "routed", "profile": "none", "record": "full",
                     "time_limit": False, "success": True, "end_reason": "success", "steps": 3})
        out.add_events(rid, [{"t": "start"}, {"t": "end", "step": 3, "reason": "success"}])
        out.add_steps(rid, [{"step": i, "driver": "xvla", "action": np.zeros(7), "robot0_eef_pos": np.zeros(3),
                             "robot0_eef_quat": np.zeros(4), "robot0_gripper_qpos": np.zeros(2)} for i in (1, 2, 3)])
        out.put_images(rid, 0, {"agentview_image": np.zeros((4, 4, 3), np.uint8)})
        return [a.as_record() for a in out.finish()], {"run_id": rid}
    return runner


def make_dataset(tmp_path, phase="base", provs=(PROV,)):
    root = tmp_path / "bucket"
    storage = LocalStorage(root)
    q = JobQueue(tmp_path / "q.db", storage, tmp_path / "local")
    q.add_plan(episode_plan(phase, tasks=[0, 1], inits=[0], namespace=10, mode="routed", profile="none",
                            record="full", order_seed=1))
    for i, prov in enumerate(provs):
        last = i == len(provs) - 1
        run_worker(q, storage, runner_for(tmp_path / f"staging{i}", prov), "w", meta={"provenance": prov},
                   max_jobs=None if last else 1)
    q.close()
    return root


def test_a_clean_dataset_gets_a_manifest_and_verifies(tmp_path):
    root = make_dataset(tmp_path)
    m = manifest.build(root)
    assert (root / "MANIFEST.json").exists() and (root / "DATACARD.md").read_bytes() == \
        manifest.DATACARD_SOURCE.read_bytes()
    assert m["harness_version"] == PROV["harness_version"] and m["source_digest"] == PROV["source_digest"]
    assert m["configs"] == {PROV["config_hash"]: {"config_name": "harness-v1", "options": {"scene_seed": "project"},
                                                  "jobs": 2}}
    assert m["phases"] == {"base": {"episode_runs": 2, "jobs": 2}}
    assert m["calibrations"] == {"1" * 64: 2} and m["freeze"] is None
    assert m["warnings"] == [f"calibration hashes not checked against a freeze record (--freeze): {'1' * 12} (2 jobs)"]
    assert m["tables"]["runs"] == {"parts": 2, "rows": 2} and m["tables"]["steps"]["rows"] == 6
    # every file in the tree except the manifest itself, the datacard separately
    tree = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert {f["path"] for f in m["files"]} == tree - {"MANIFEST.json", "DATACARD.md"}
    assert m["totals"]["files"] == len(m["files"]) == 2 * (3 + 1 + 1)      # 3 tables + images + marker per job
    assert manifest.verify(root) == []


def test_a_changed_byte_stops_the_build_unless_only_sizes_are_checked(tmp_path):
    root = make_dataset(tmp_path)
    img = next((root / "images").rglob("*.npz"))
    data = bytearray(img.read_bytes())
    data[-1] ^= 1
    img.write_bytes(bytes(data))
    with pytest.raises(manifest.ManifestError, match="SHA-256 differs"):
        manifest.build(root)
    assert not (root / "MANIFEST.json").exists()
    manifest.build(root, quick=True)                     # same size: a quick build cannot see it
    assert any("SHA-256" in p for p in manifest.verify(root))


def test_unlisted_and_stray_files_stop_the_build(tmp_path):
    root = make_dataset(tmp_path)
    orphan = root / "snapshots" / "base" / "t9" / "i0" / "dp03.npz"     # a job that died after this upload
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"partial")
    with pytest.raises(manifest.ManifestError, match="listed by no result marker"):
        manifest.build(root)
    with pytest.raises(ValueError):
        manifest.build(root, move_unlisted=root / "inside")
    manifest.build(root, move_unlisted=tmp_path / "unlisted")
    assert (tmp_path / "unlisted" / "snapshots" / "base" / "t9" / "i0" / "dp03.npz").read_bytes() == b"partial"
    (root / "ops").mkdir()
    (root / "ops" / "ledger.json").write_text("{}")
    with pytest.raises(manifest.ManifestError, match="ops/ledger.json: not part of the dataset layout"):
        manifest.build(root)


def test_mixed_harness_code_stops_the_build(tmp_path):
    other = {**PROV, "source_digest": "t" * 64}
    root = make_dataset(tmp_path, provs=(PROV, other))
    with pytest.raises(manifest.ManifestError, match="mixed source_digest"):
        manifest.build(root)


def test_mixed_or_missing_ops_code_stops_the_build(tmp_path):
    # Amendment 46: the ops code runs from the host source, pinned by its digest in every record.
    root = make_dataset(tmp_path / "mixed", provs=(PROV, {**PROV, "ops_digest": "p" * 64}))
    with pytest.raises(manifest.ManifestError, match="mixed ops_digest"):
        manifest.build(root)
    root = make_dataset(tmp_path / "missing", provs=({k: v for k, v in PROV.items() if k != "ops_digest"},))
    with pytest.raises(manifest.ManifestError, match="no ops digest in the provenance"):
        manifest.build(root)


def test_options_and_commits_may_differ_and_dirty_code_is_a_warning(tmp_path):
    other = {**PROV, "config_hash": "d" * 64, "config_options": {"scene_seed": "pi05_parity"}, "commit": "b" * 40,
             "harness_dirty": True}
    root = make_dataset(tmp_path, provs=(PROV, other))
    m = manifest.build(root)
    assert set(m["configs"]) == {"c" * 64, "d" * 64} and m["git_commits"] == ["a" * 40, "b" * 40]
    assert m["warnings"][1:] == ["1 job(s) ran a harness with uncommitted changes (harness_dirty)"]


def test_calibration_hashes_must_be_in_the_freeze_lineage(tmp_path):
    """Amendment 35: parity and base records made under earlier calibration versions with the same write-once
    values count as v1.0; anything else, or no calibration hash, stops the build."""
    later = {**PROV, "calibration_hash": "2" * 64}
    root = make_dataset(tmp_path, provs=(PROV, later))
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"harness_version": "1.0.0", "calibration_hash": "3" * 64, "ops_digest": "o" * 64,
                                  "source_digest": "s" * 64, "image_digests": IMG, "calibration_lineage": ["1" * 64, "2" * 64]}))
    m = manifest.build(root, freeze=freeze)
    assert m["calibrations"] == {"1" * 64: 1, "2" * 64: 1} and m["freeze"]["calibration_hash"] == "3" * 64
    assert not any("calibration" in w for w in m["warnings"])
    freeze.write_text(json.dumps({"harness_version": "1.0.0", "calibration_hash": "3" * 64, "ops_digest": "o" * 64,
                                  "source_digest": "s" * 64, "image_digests": IMG, "calibration_lineage": ["2" * 64]}))
    with pytest.raises(manifest.ManifestError, match="not the frozen file's nor in its lineage"):
        manifest.build(root, freeze=freeze)
    root2 = make_dataset(tmp_path / "b", provs=({**PROV, "calibration_hash": None},))
    with pytest.raises(manifest.ManifestError, match="no calibration hash"):
        manifest.build(root2)


def test_a_run_row_that_disagrees_with_its_marker_stops_the_build(tmp_path):
    root = make_dataset(tmp_path)
    marker = next((root / "results").rglob("*.json"))
    m = json.loads(marker.read_text())
    m["meta"]["provenance"]["config_hash"] = "e" * 64
    marker.write_text(json.dumps(m))
    with pytest.raises(manifest.ManifestError, match="run row's config_hash differ"):
        manifest.build(root)
    m["meta"]["provenance"].update(config_hash="c" * 64, calibration_hash="2" * 64)
    marker.write_text(json.dumps(m))
    with pytest.raises(manifest.ManifestError, match="run row's calibration_hash differ"):
        manifest.build(root)
    del m["meta"]["provenance"]
    marker.write_text(json.dumps(m))
    with pytest.raises(manifest.ManifestError, match="no provenance"):
        manifest.build(root)


def test_verify_finds_missing_changed_and_extra_files(tmp_path):
    root = make_dataset(tmp_path)
    manifest.build(root)
    copy = tmp_path / "download"
    shutil.copytree(root, copy)
    assert manifest.verify(copy) == []
    next((copy / "records" / "steps").rglob("*.parquet")).unlink()
    (copy / "DATACARD.md").write_text("edited")
    (copy / "extra.txt").write_text("x")
    problems = manifest.verify(copy)
    assert any(p.endswith(": missing") and "records/steps" in p for p in problems)
    assert any(p.startswith("DATACARD.md:") for p in problems) and "extra.txt: not in the manifest" in problems


def test_the_datacard_documents_every_field_of_every_table():
    text = manifest.DATACARD_SOURCE.read_text()
    for table in TABLES:
        section = re.search(rf"^### `{table}`\n(.*?)(?=^### |^## )", text, re.S | re.M)
        assert section, f"no section for {table}"
        documented = re.findall(r"^\| `([a-z0-9_]+)` \|", section.group(1), re.M)
        assert documented == _schemas()[table].names, table


# ---------------------------------------------------------------- amendment 49: the ops digest after the freeze

def _archive(files):
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for k, v in files.items():
            info = tarfile.TarInfo(f"src/vlar_ops/{k}")
            info.size = len(v)
            tar.addfile(info, io.BytesIO(v))
    return buf.getvalue()


def _ops_versions(tmp_path, **changes):
    """Committed archives in a copy of phase0/src/: `frozen` is the real vlar_ops, each change edits one file
    (name -> (file, old, new)). Returns the directory and each version's ops digest."""
    from vlar_harness import provenance as P
    from vlar_ops import datadigest as DD
    real = DD.files_from_dir(P.ops_dir())
    versions = {"frozen": real}
    for name, (f, old, new) in changes.items():
        assert real[f].count(old) == 1
        versions[name] = {**real, f: real[f].replace(old, new)}
    d = tmp_path / "phase0-src"
    d.mkdir()
    digests = {}
    for i, (name, files) in enumerate(versions.items()):
        (d / f"{i:040x}.tar.gz").write_bytes(_archive(files))
        digests[name] = DD.ops_digest_of(files)
    return d, digests


FROZEN_CAL, PRE_CAL = "3" * 64, "1" * 64


def _freeze(tmp_path, ops):
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({"harness_version": "1.0.0", "source_digest": "s" * 64, "ops_digest": ops,
                                "calibration_hash": FROZEN_CAL, "calibration_lineage": [PRE_CAL], "image_digests": IMG}))
    return path


def _meta(ops, frozen_ops, *, cal=FROZEN_CAL, p=0.1, golden=True, waiver=True, boot="boot-1"):
    meta = {"provenance": {**PROV, "ops_digest": ops, "calibration_hash": cal}, "p_escalate": p, "boot_id": boot}
    if golden:
        meta["golden_pass"] = {"boot_id": boot, "calibration_hash": FROZEN_CAL, "passed": True, "results": []}
    if waiver and ops != frozen_ops:
        meta["ops_waiver"] = {"freeze_ops_digest": frozen_ops, "ops_digest": ops}
    return meta


def _dataset(tmp_path, metas, *, phase="explore", p=0.1, spec_p="same"):
    root = tmp_path / "bucket"
    storage = LocalStorage(root)
    q = JobQueue(tmp_path / "q.db", storage, tmp_path / "local")
    extra = None
    if phase == "explore" and spec_p is not None:
        extra = lambda t, i: {"p_escalate": p if spec_p == "same" else spec_p, "episodes_per_state": 1}  # noqa: E731
    q.add_plan(episode_plan(phase, tasks=list(range(len(metas))), inits=[0], namespace=10, mode="routed",
                            profile="none", record="full", order_seed=1, extra=extra))
    for i, meta in enumerate(metas):
        run_worker(q, storage, runner_for(tmp_path / f"staging{i}", meta["provenance"]), "w", meta=meta, max_jobs=1)
    q.close()
    return root


def test_after_the_freeze_a_tunable_or_ops_only_change_mixes(tmp_path):
    archives, d = _ops_versions(tmp_path, tunable=("configs/collection-v1-tunables.yaml", b"p_escalate: 0.1 ",
                                                    b"p_escalate: 0.2 "),
                                spend=("spend.py", b"from __future__ import annotations\n",
                                       b"from __future__ import annotations\nEXTRA = 1\n"))
    freeze = _freeze(tmp_path, d["frozen"])
    root = _dataset(tmp_path, [_meta(d["frozen"], d["frozen"]), _meta(d["tunable"], d["frozen"], p=0.1),
                               _meta(d["spend"], d["frozen"])])
    m = manifest.build(root, freeze=freeze, ops_archives=archives)
    assert m["ops_digest"] is None and sorted(m["ops_digests"]) == sorted(d.values())
    assert len({v["common"] for v in m["data_digests"].values()}) == 1
    with pytest.raises(manifest.ManifestError, match="--ops-archives"):
        manifest.build(root, freeze=freeze)


@pytest.mark.parametrize("change,match", [
    (("configs/collection-v1.yaml", b"  explore: 2000000", b"  explore: 2100000"), "data digest"),
    (("vm_worker.py", b"self.pi05.act(raw_obs, task, seed)", b"self.pi05.act(raw_obs, task, None)"), "data digest"),
    (("plan.py", b'"mode": mode, "profile": profile', b'"mode": mode, "profile": profile, "v": 2'),
     "mixed explore job-spec generators"),
])
def test_after_the_freeze_a_definition_factory_or_generator_change_refuses(tmp_path, change, match):
    archives, d = _ops_versions(tmp_path, other=change)
    root = _dataset(tmp_path, [_meta(d["frozen"], d["frozen"]), _meta(d["other"], d["frozen"])])
    with pytest.raises(manifest.ManifestError, match=match):
        manifest.build(root, freeze=_freeze(tmp_path, d["frozen"]), ops_archives=archives)


def test_a_generator_change_before_a_kinds_first_record_mixes(tmp_path):
    # Base records at the freeze's ops code, exploration records all from a later generator: kinds never compared.
    archives, d = _ops_versions(tmp_path, gen=("plan.py", b'"mode": mode, "profile": profile',
                                               b'"mode": mode, "profile": profile, "v": 2'))
    base = _dataset(tmp_path / "b", [_meta(d["frozen"], d["frozen"], cal=PRE_CAL, golden=False)], phase="base")
    expl = _dataset(tmp_path / "e", [_meta(d["gen"], d["frozen"]), _meta(d["gen"], d["frozen"])])
    for sub in ("results", "records", "snapshots", "images"):
        if (expl / sub).exists():
            shutil.copytree(expl / sub, base / sub, dirs_exist_ok=True)
    m = manifest.build(base, freeze=_freeze(tmp_path, d["frozen"]), ops_archives=archives)
    assert m["phases"]["base"]["jobs"] == 1 and m["phases"]["explore"]["jobs"] == 2


@pytest.mark.parametrize("case", ["no golden pass", "golden pass of another boot", "failed golden pass",
                                  "uncommitted ops code", "pre-freeze record with other ops code", "no waiver record",
                                  "other harness code"])
def test_after_the_freeze_each_rule_refuses(tmp_path, case):
    archives, d = _ops_versions(tmp_path, spend=("spend.py", b"from __future__ import annotations\n",
                                                 b"from __future__ import annotations\nEXTRA = 1\n"))
    f, s = d["frozen"], d["spend"]
    meta = {"no golden pass": _meta(f, f, golden=False),
            "golden pass of another boot": {**_meta(f, f), "boot_id": "boot-2"},
            "failed golden pass": {**_meta(f, f), "golden_pass": {**_meta(f, f)["golden_pass"], "passed": False}},
            "uncommitted ops code": _meta("e" * 64, f),
            "pre-freeze record with other ops code": _meta(s, f, cal=PRE_CAL, golden=False),
            "no waiver record": _meta(s, f, waiver=False),
            "other harness code": {**_meta(f, f), "provenance": {**_meta(f, f)["provenance"],
                                                                 "source_digest": "t" * 64}}}[case]
    match = {"no golden pass": "golden-set pass", "golden pass of another boot": "golden-set pass",
             "failed golden pass": "golden-set pass", "uncommitted ops code": "not a committed archive",
             "pre-freeze record with other ops code": "pre-freeze record", "no waiver record": "ops_waiver",
             "other harness code": "source_digest is not the freeze's"}[case]
    root = _dataset(tmp_path, [_meta(f, f), meta])
    with pytest.raises(manifest.ManifestError, match=match):
        manifest.build(root, freeze=_freeze(tmp_path, f), ops_archives=archives)


def test_tunables_must_be_in_the_job_specs(tmp_path):
    f = "o" * 64
    root = _dataset(tmp_path / "none", [_meta(f, f)], spec_p=None)
    with pytest.raises(manifest.ManifestError, match="without p_escalate in its spec"):
        manifest.build(root)
    root = _dataset(tmp_path / "other", [_meta(f, f)], spec_p=0.3)
    with pytest.raises(manifest.ManifestError, match="spec p_escalate 0.3 but its worker ran 0.1"):
        manifest.build(root)
    assert manifest._tunable_problems("m", {"job": {"kind": "b2_one_call", "phase": "branches", "spec": {}}, "meta": {}})
    settings = {"seeds": 3, "enabled": ["b1_continue"], "tier_one_seeds": 2, "order_seed": 1,
                "dp_sampling": {"fraction": 0.2, "seed": 5}}
    assert manifest._tunable_problems("m", {"job": {"kind": "b2_one_call", "phase": "branches",
                                                    "spec": {"settings": settings}}, "meta": {}}) == []
    # amendment 57's stratified draw (Tranche A's specs) records per_task and failed_share instead of a fraction
    strat = {**settings, "dp_sampling": {"scheme": "stratified", "per_task": 64, "failed_share": 2 / 3,
                                         "min_calls": 1, "seed": 20261125}}
    assert manifest._tunable_problems("m", {"job": {"kind": "b1_continue", "phase": "phase1a",
                                                    "spec": {"settings": strat}}, "meta": {}}) == []
    for drop in ("per_task", "failed_share", "seed"):
        bad = {**strat, "dp_sampling": {k: v for k, v in strat["dp_sampling"].items() if k != drop}}
        assert manifest._tunable_problems("m", {"job": {"kind": "b1_continue", "phase": "phase1a",
                                                        "spec": {"settings": bad}}, "meta": {}})


# ---------------------------------------------------------------- amendment 54: the frozen images

def _bare_freeze(tmp_path):
    """A freeze record like v1.0's: no image digests of its own."""
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({"harness_version": "1.0.0", "source_digest": "s" * 64, "ops_digest": "o" * 64,
                                "calibration_hash": "3" * 64, "calibration_lineage": ["1" * 64],
                                "frozen_at": "2026-11-24T09:46:21+00:00", "image_digests": {}}))
    return path


def _supplement(freeze, images=IMG):
    from vlar_ops import freezesupp as FS
    m = {"finished_at": 0.0, "job": {"phase": "base", "job_id": "j"}, "meta": {"provenance": {"image_digests": images}}}
    freeze.with_name(FS.name_for("2026-11-24")).write_text(json.dumps(FS.make(freeze, [m], date="2026-11-24")))


def test_after_the_freeze_every_record_must_be_on_the_frozen_images(tmp_path):
    """The mixing check: records that agree with each other but not with the images the freeze pins are refused, and
    a freeze that pins no images (no supplement beside a bare record) refuses every dataset."""
    root = make_dataset(tmp_path, provs=(PROV, PROV))
    freeze = _bare_freeze(tmp_path)
    with pytest.raises(manifest.ManifestError, match="pins no image digests"):
        manifest.build(root, freeze=freeze)
    _supplement(freeze)
    m = manifest.build(root, freeze=freeze)
    assert m["freeze"]["image_digests"] == IMG and m["image_digests"] == IMG
    other = {**IMG, "xvla": "sha256:" + "4" * 64}
    root2 = make_dataset(tmp_path / "other", provs=({**PROV, "image_digests": other},) * 2)
    with pytest.raises(manifest.ManifestError, match="image digests are not the frozen images") as e:
        manifest.build(root2, freeze=freeze)
    assert "mixed" not in str(e.value)                     # the records agree with each other; only the freeze differs
    root3 = make_dataset(tmp_path / "none", provs=({**PROV, "image_digests": {}},))
    with pytest.raises(manifest.ManifestError, match="image digests are not the frozen images"):
        manifest.build(root3, freeze=freeze)
    freeze.write_text(freeze.read_text().replace("1.0.0", "1.0.1"))          # the supplement names another freeze
    with pytest.raises(manifest.ManifestError, match="another freeze record"):
        manifest.build(root, freeze=freeze)
