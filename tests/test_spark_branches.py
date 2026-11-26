"""ops/spark/branches.py: the staged-file helpers and the dataset layout (no simulator, no servers)."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("spark_branches", Path(__file__).parents[1] / "ops" / "spark" / "branches.py")
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def stage(root: Path, phase: str, job: str, tables=("runs", "events"), snapshots=()):
    d = root / "records" / phase / job
    for t in tables:
        (d / "records").mkdir(parents=True, exist_ok=True)
        (d / "records" / f"{t}.parquet").write_bytes(f"{phase}/{job}/{t}".encode())
    for sid in snapshots:
        f = d / "snapshots" / f"{sid}.npz"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(sid.encode())
    return d


def test_snapshot_reader_finds_the_staged_file(tmp_path):
    stage(tmp_path, "base", "j1", snapshots=["base/t0/i1/dp03", "base/t0/i1/dp04"])
    stage(tmp_path, "base", "j2", snapshots=["base/t1/i0/dp00"])
    read = B.snapshot_reader(tmp_path)
    assert read("snapshots/base/t0/i1/dp04.npz") == b"base/t0/i1/dp04"
    assert read("snapshots/base/t9/i9/dp00.npz") is None
    assert read("images/base/t0/i1.npz") is None and read("snapshots/../x.npz") is None
    assert sorted(B.staged_snapshots(tmp_path)) == ["base/t0/i1/dp03", "base/t0/i1/dp04", "base/t1/i0/dp00"]


def test_a_snapshot_staged_twice_is_an_error(tmp_path):
    stage(tmp_path, "base", "j1", snapshots=["base/t0/i1/dp03"])
    stage(tmp_path, "base", "j2", snapshots=["base/t0/i1/dp03"])
    with pytest.raises(RuntimeError):
        B.staged_snapshots(tmp_path)
    with pytest.raises(RuntimeError):
        B.snapshot_reader(tmp_path)("snapshots/base/t0/i1/dp03.npz")


def test_load_jobs_reads_a_plan_file_and_a_job_list(tmp_path):
    jobs = [{"job_id": f"j{k}", "phase": "p", "kind": "b1_continue", "subject": "s", "seed_index": 2, "tier": 1,
             "order": k, "spec": {}} for k in range(3)]
    plan = tmp_path / "plan.json"
    plan.write_text("\n".join(json.dumps(x) for x in [{"plan_id": "x", "phase": "p", "params": {}, "n_jobs": 3}] + jobs) + "\n")
    assert B.load_jobs(plan) == jobs
    lst = tmp_path / "list.json"
    lst.write_text(json.dumps(jobs, indent=1))
    assert B.load_jobs(lst) == jobs
    plan.write_text("\n".join(json.dumps(x) for x in [{"plan_id": "x", "phase": "p", "params": {}, "n_jobs": 4}] + jobs))
    with pytest.raises(ValueError):
        B.load_jobs(plan)
    plan.write_text(json.dumps(jobs[0]) + "\n")
    with pytest.raises(ValueError):
        B.load_jobs(plan)


def test_assemble_writes_the_dataset_layout(tmp_path):
    base, runs, plan_dir, root = (tmp_path / n for n in ("base", "runs", "plan", "dataset"))
    stage(base, "base", "jb", tables=("runs", "events", "points", "steps"), snapshots=["base/t0/i1/dp03"])
    stage(runs, "dry", "j1", tables=("runs", "events", "steps"))
    stage(runs, "dry", "j2", tables=("runs", "events", "steps"))
    plan_dir.mkdir()
    (plan_dir / "labels.json").write_text("{}")
    rep = B.assemble([base, runs], root, snapshots=B.staged_snapshots(base), plan_dir=plan_dir, note="DRY RUN: x")
    assert (root / "records" / "points" / "base" / "jb.parquet").read_bytes() == b"base/jb/points"
    assert (root / "records" / "runs" / "dry" / "j2.parquet").read_bytes() == b"dry/j2/runs"
    assert (root / "snapshots" / "base" / "t0" / "i1" / "dp03.npz").read_bytes() == b"base/t0/i1/dp03"
    assert (root / "plan" / "labels.json").exists()
    assert (root / "README.md").read_text().startswith("# DRY RUN: x\n")
    assert rep["parts"] == {"runs/base": 1, "events/base": 1, "points/base": 1, "steps/base": 1, "runs/dry": 2,
                            "events/dry": 2, "steps/dry": 2} and rep["snapshots"] == 1
    B.assemble([base, runs], root, snapshots={}, plan_dir=plan_dir, note="again")      # a second pass replaces


def test_assemble_refuses_a_part_staged_twice(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    stage(a, "dry", "j1")
    stage(b, "dry", "j1")
    (tmp_path / "plan").mkdir()
    with pytest.raises(RuntimeError):
        B.assemble([a, b], tmp_path / "root", snapshots={}, plan_dir=tmp_path / "plan", note="x")
