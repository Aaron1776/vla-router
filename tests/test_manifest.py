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

PROV = {"harness_version": "1.0.0", "config_name": "harness-v1", "config_hash": "c" * 64,
        "config_options": {"scene_seed": "project"}, "calibration_hash": "1" * 64, "source_digest": "s" * 64,
        "ops_digest": "o" * 64, "commit": "a" * 40, "harness_dirty": False, "image_digests": {}}


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
    freeze.write_text(json.dumps({"harness_version": "1.0.0", "calibration_hash": "3" * 64,
                                  "calibration_lineage": ["1" * 64, "2" * 64]}))
    m = manifest.build(root, freeze=freeze)
    assert m["calibrations"] == {"1" * 64: 1, "2" * 64: 1} and m["freeze"]["calibration_hash"] == "3" * 64
    assert not any("calibration" in w for w in m["warnings"])
    freeze.write_text(json.dumps({"harness_version": "1.0.0", "calibration_hash": "3" * 64,
                                  "calibration_lineage": ["2" * 64]}))
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
