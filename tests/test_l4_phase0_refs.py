"""References/l4-phase0: the L4's Phase 0 reference files for the DGX Spark (pinned like References/; README there).

Every file is pinned in `src/vlar_ops/configs/l4-phase0.sha256`; the base event logs' archive members match their result
markers' hashes (listed beside the archive) and carry no account identifier; the runs tables hold the totals Phase 0
reported, and each reconstructed setup count follows the rule the README states and agrees wherever it was checked."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

from vlar_ops import reference_runs as R

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "References" / "l4-phase0"
MANIFEST = REPO / "src" / "vlar_ops" / "configs" / "l4-phase0.sha256"
pytestmark = pytest.mark.skipif(not (ROOT / "README.md").exists(), reason="References/l4-phase0 not present")


def pins() -> dict[str, str]:
    out = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, path = line.split(maxsplit=1)
            out[path.strip().split("References/l4-phase0/", 1)[-1]] = digest
    return out


def test_every_file_is_pinned_and_matches():
    p = pins()
    files = sorted(str(f.relative_to(ROOT)) for f in ROOT.rglob("*") if f.is_file() and f.name != ".DS_Store")
    assert sorted(p) == files
    R.verify_files(ROOT, p)


def _env_bucket():
    import re
    env = REPO / "ops" / "gcp" / "env.sh"
    m = env.exists() and re.search(r'^export BUCKET="?([^"\s#]+)"?', env.read_text(), re.M)
    return m.group(1) if m and m.group(1) != "REPLACE" else None


def test_calibration_versions_are_named_by_their_hash_and_redacted_ones_restore_to_it():
    """Files naming the bucket are committed with it replaced by `$BUCKET` (`*.redacted.yaml`); redactions.json pins
    the originals. The frozen file is b80124b2."""
    red = json.loads((ROOT / "calibration" / "redactions.json").read_text())
    assert sorted(red) == ["calibration/calibration.redacted.yaml",
                           "calibration/versions/236439b4f06c68e2d4ecef86e8c57d71876f1bbd6dca8e58a75a9f9125c6657c"
                           ".redacted.yaml",
                           "calibration/versions/b80124b2c19ba8a5e7dcc538eb777a6ca033463020e77b28464b068118c5f51c"
                           ".redacted.yaml"]
    versions = sorted((ROOT / "calibration" / "versions").glob("*.yaml"))
    assert [v.name[:8] for v in versions] == ["1d61294a", "236439b4", "aa38aac7", "b80124b2"]
    bucket = _env_bucket()
    for v in [*versions, ROOT / "calibration" / "calibration.redacted.yaml"]:
        data, rel = v.read_bytes(), str(v.relative_to(ROOT))
        if rel not in red:
            assert hashlib.sha256(data).hexdigest() == v.stem
            continue
        r = red[rel]
        assert hashlib.sha256(data).hexdigest() == r["redacted_sha256"]
        assert data.count(b"$BUCKET") == r["occurrences"] == data.count(b"$")
        if v.parent.name == "versions":
            assert r["original_name"] == v.name.replace(".redacted", "") == r["original_sha256"] + ".yaml"
        else:
            assert r["original_sha256"].startswith("b80124b2")                      # the frozen file
        if bucket:                                                                    # the Mac that commits
            assert hashlib.sha256(data.replace(b"$BUCKET", bucket.encode())).hexdigest() == r["original_sha256"]
    freeze = json.loads((ROOT / "freeze" / "freeze.json").read_text())
    sup = json.loads((ROOT / "freeze" / "freeze-supplement-2026-11-24.json").read_text())
    assert freeze["calibration_hash"].startswith("b80124b2") and freeze["image_digests"] == {}
    assert hashlib.sha256((ROOT / "freeze" / "freeze.json").read_bytes()).hexdigest() == sup["freeze_sha256"]
    assert json.loads((ROOT / "images" / "image-digests.json").read_text()) == sup["image_digests"]


def _identifier_test():
    spec = importlib.util.spec_from_file_location("_noid", REPO / "tests" / "test_no_identifiers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_event_archive_members_match_their_marker_hashes_and_hold_no_account_identifier():
    want = {}
    for line in (ROOT / "events" / "base-events.sha256").read_text().splitlines():
        if line and not line.startswith("#"):
            d, name = line.split(maxsplit=1)
            want[name] = d
    assert len(want) == 500
    noid = _identifier_test()
    try:                      # parquet pages are compressed: scan the decoded table where pyarrow is installed (sim venv)
        import pyarrow.parquet as pq
    except ImportError:
        pq = None
    got, hits = {}, []
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress((ROOT / "events" / "base-events.tar.gz").read_bytes()))) as t:
        for m in t.getmembers():
            data = t.extractfile(m).read()
            got[m.name] = hashlib.sha256(data).hexdigest()
            text = data.decode("latin-1")
            if pq is not None:
                text += "\n".join(map(str, pq.read_table(io.BytesIO(data)).to_pylist()))
            for tok in noid.TOKEN.finditer(text):
                hits += [m.name for c in noid.candidates(tok.group(0)) if noid.digest(c) in noid.DIGESTS]
    assert got == want
    assert not hits, sorted(set(hits))


def _rows(name):
    return list(csv.DictReader((ROOT / "runs" / name).open()))


def test_runs_tables_hold_phase_0s_totals_and_the_stated_count_rule():
    base, parity = _rows("base-runs.csv"), _rows("pi05-parity-runs.csv")
    assert len(base) == 500 and sum(r["success"] == "True" for r in base) == 398       # X-VLA parity 398/500
    assert len(parity) == 100 and sum(r["success"] == "True" for r in parity) == 94    # π0.5's 94 successes
    for rows in (base, parity):
        assert len({(r["task_id"], r["init_idx"]) for r in rows}) == len(rows)
        by_worker: dict = {}
        for r in sorted(rows, key=lambda r: (r["worker"], int(r["worker_seq"]))):
            seen = by_worker.setdefault(r["worker"], {})
            seen[r["task_id"]] = seen.get(r["task_id"], 0) + 1
            assert int(r["worker_task_seq"]) == seen[r["task_id"]]
            assert int(r["setup_count_reconstructed"]) == seen[r["task_id"]] + 1     # fresh sim = load 1
    checked = [r for r in base if r["setup_count_check_method"] in ("sites_list", "fixture_poses")]
    assert {int(r["task_id"]) for r in checked} == {2, 3, 8, 9}
    assert all(r["setup_count_check"] == r["setup_count_reconstructed"] for r in checked)
    # the counts measured from snapshots before this package (docs/HARNESS.md)
    known = {"base/t8/i30": "4", "base/t8/i13": "7", "base/t3/i25": "2", "base/t9/i8": "2"}
    assert {r["run_id"]: r["setup_count_reconstructed"] for r in base if r["run_id"] in known} == known
