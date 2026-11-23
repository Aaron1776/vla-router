"""Dataset manifest and datacard (docs/SPEC.md §13, §15).

`build(root)` works on a local copy of the bucket's dataset (records/, snapshots/, images/, results/):
- reads every result marker (results/<phase>/<job_id>.json) and checks each file it lists: present, same size and,
  unless `quick`, same SHA-256; no two markers may list the same file;
- finds data files no marker lists (a job interrupted after uploading some of its files that never finished):
  they are not part of the dataset, and the build is refused until they are moved out (`move_unlisted`), so a
  reader globbing a table never picks them up. Any other file is refused too: the tree holds only the dataset;
- checks every Parquet part against dataset v1's schema, and every `runs` row's schema version, harness version,
  config hash, phase and job id against its job's marker;
- checks nothing from different harness versions is mixed (§15): one harness version, one source digest and one
  set of image digests across all markers; without a freeze record also one ops digest (amendment 46). Several config hashes are allowed (named options, e.g. the π0.5 parity
  run's scene seeds) and listed with their options; several git commits are allowed (code outside the harness
  package changes the commit, not the harness) and listed; markers from a harness with uncommitted changes are
  listed as a warning;
- checks every marker carries a calibration hash (amendment 35) and every run row repeats its marker's; given the
  freeze record (`freeze`), every calibration hash must be the frozen file's or in its lineage, so parity and base
  records count as v1.0; without one, the hashes are listed and a warning says they were not checked;
- with a freeze record, the ops digest is provenance, not a barrier (amendment 49): the harness must be the frozen
  one; pre-freeze records (calibration in the lineage, not the final file) must have the freeze's ops digest;
  records of the frozen harness must carry this boot's golden-set pass; a record with other ops code must carry
  its worker's `ops_waiver`; once any ops digest differs from the freeze's, every ops digest involved must be a
  committed archive in `ops_archives` (a copy of phase0/src/), their common data digest must equal the freeze
  archive's, and within each record kind (base, explore, branch) the job-spec generator digests must agree
  (`vlar_ops.datadigest`);
- checks the tunables are in the job specs (amendment 49): an exploration job's p_escalate equals its worker's,
  and a branch job records its planner settings (seeds, enabled branches, sampling fraction and seed).
Only when every check passes does it write DATACARD.md (vlar_ops/datacard-v1.md) and MANIFEST.json at the root.
`verify(root)` checks a tree against its MANIFEST.json, e.g. after download and unpacking: every listed file
present with its size and SHA-256, and no other file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MANIFEST = "MANIFEST.json"
DATACARD = "DATACARD.md"
DATACARD_SOURCE = Path(__file__).with_name("datacard-v1.md")
DATASET = "dataset-v1"
MANIFEST_VERSION = 1
DATA_PREFIXES = ("records/", "snapshots/", "images/")
RESULTS = "results/"
SINGLE = ("harness_version", "source_digest", "ops_digest", "image_digests")   # one value per dataset (§15, am. 46)


class ManifestError(RuntimeError):
    def __init__(self, problems: list[str]):
        super().__init__(f"{len(problems)} problem(s):\n" + "\n".join(problems[:50])
                         + ("\n..." if len(problems) > 50 else ""))
        self.problems = problems


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True)


def build(root: Path | str, *, quick: bool = False, move_unlisted: Optional[Path | str] = None,
          freeze: Optional[Path | str] = None, ops_archives: Optional[Path | str] = None,
          now: Optional[datetime] = None) -> dict[str, Any]:
    """Check the dataset under `root`; if it passes, write DATACARD.md and MANIFEST.json and return the manifest.
    `quick` checks sizes only (the SHA-256 recorded at upload are trusted). `move_unlisted` moves data files no
    marker lists into that directory (outside `root`), keeping their keys, before checking."""
    import pyarrow.parquet as pq
    from vlar_harness.dataset import SCHEMA_VERSION, TABLES, _schemas

    root = Path(root)
    problems: list[str] = []
    warnings: list[str] = []
    markers: dict[str, dict] = {}
    for key in (k for k in _tree(root) if k.startswith(RESULTS)):
        parts = key.split("/")
        try:
            m = json.loads((root / key).read_text())
            phase, job_id = m["job"]["phase"], m["job"]["job_id"]
        except (ValueError, KeyError, TypeError) as e:
            problems.append(f"{key}: not a result marker ({e})")
            continue
        if len(parts) != 3 or parts[1] != phase or parts[2] != f"{job_id}.json":
            problems.append(f"{key}: holds job {phase}/{job_id}")
        markers[key] = m

    owner: dict[str, str] = {}
    for mkey, m in markers.items():
        for a in m["result"].get("artifacts", []):
            k = a["key"]
            if not k.startswith(DATA_PREFIXES):
                problems.append(f"{mkey}: lists {k}, outside the dataset layout")
            if k in owner:
                problems.append(f"{k}: listed by {owner[k]} and {mkey}")
            owner[k] = mkey

    unlisted = [k for k in _tree(root) if k.startswith(DATA_PREFIXES) and k not in owner]
    if unlisted and move_unlisted is not None:
        dest = Path(move_unlisted).resolve()
        if dest == root.resolve() or root.resolve() in dest.parents:
            raise ValueError("move_unlisted must be outside the dataset root")
        for k in unlisted:
            (dest / k).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root / k), str(dest / k))
        unlisted = []
    problems += [f"{k}: listed by no result marker (move it out with --move-unlisted)" for k in unlisted]
    others = [k for k in _tree(root) if not k.startswith(DATA_PREFIXES + (RESULTS,)) and k not in (MANIFEST, DATACARD)]
    problems += [f"{k}: not part of the dataset layout" for k in others]

    files: list[dict[str, Any]] = []
    for mkey, m in markers.items():
        for a in m["result"].get("artifacts", []):
            p = root / a["key"]
            if not p.is_file():
                problems.append(f"{a['key']}: missing (listed by {mkey})")
                continue
            size = p.stat().st_size
            if size != a["size"]:
                problems.append(f"{a['key']}: {size} bytes, marker says {a['size']}")
            elif not quick and sha256_file(p) != a["sha256"]:
                problems.append(f"{a['key']}: SHA-256 differs from its marker")
            files.append({"path": a["key"], "sha256": a["sha256"], "size": a["size"]})
        files.append({"path": mkey, "sha256": sha256_file(root / mkey), "size": (root / mkey).stat().st_size})

    # Provenance: nothing from different harness versions mixed (§15).
    frozen = json.loads(Path(freeze).read_text()) if freeze else None
    barrier = SINGLE if frozen is None else tuple(f for f in SINGLE if f != "ops_digest")
    seen: dict[str, Counter] = {f: Counter() for f in barrier}
    ops_digests: Counter = Counter()
    commits: Counter = Counter()
    configs: dict[str, dict[str, Any]] = {}
    calibrations: Counter = Counter()
    dirty = 0
    for mkey, m in markers.items():
        prov = (m.get("meta") or {}).get("provenance")
        if not prov:
            problems.append(f"{mkey}: no provenance in the marker")
            continue
        for f in barrier:
            seen[f][_canon(prov.get(f))] += 1
        ops_digests[prov.get("ops_digest")] += 1
        if not prov.get("ops_digest"):
            problems.append(f"{mkey}: no ops digest in the provenance (amendment 46)")
        problems += _tunable_problems(mkey, m)
        commits[prov.get("commit")] += 1
        dirty += bool(prov.get("harness_dirty"))
        c = configs.setdefault(prov.get("config_hash"), {"config_name": prov.get("config_name"),
                                                         "options": prov.get("config_options"), "jobs": 0})
        c["jobs"] += 1
        cal = prov.get("calibration_hash")
        calibrations[cal] += 1
        if cal is None:
            problems.append(f"{mkey}: no calibration hash in the provenance (amendment 35)")
        elif frozen is not None and cal != frozen["calibration_hash"] and cal not in frozen["calibration_lineage"]:
            problems.append(f"{mkey}: calibration {cal[:12]} is not the frozen file's nor in its lineage")
        elif frozen is not None:
            problems += _freeze_problems(mkey, m, prov, frozen)
    if frozen is None and calibrations:
        warnings.append("calibration hashes not checked against a freeze record (--freeze): "
                        + ", ".join(f"{str(h)[:12]} ({n} jobs)" for h, n in calibrations.most_common()))
    for f, counts in seen.items():
        if len(counts) > 1:
            problems.append(f"mixed {f}: " + ", ".join(f"{v} ({n} jobs)" for v, n in counts.most_common()))
    data_digests = None
    if frozen is not None and not frozen.get("ops_digest"):
        problems.append("the freeze record has no ops digest (amendment 46)")
    elif frozen is not None and any(d and d != frozen["ops_digest"] for d in ops_digests):
        data_digests, found = _data_digest_problems(markers, frozen, ops_archives)
        problems += found
    if dirty:
        warnings.append(f"{dirty} job(s) ran a harness with uncommitted changes (harness_dirty)")

    # Parquet parts: schema, and run rows against their markers.
    schemas = _schemas()
    tables = {t: {"parts": 0, "rows": 0} for t in TABLES}
    phases: dict[str, Counter] = defaultdict(Counter)
    for mkey, m in markers.items():
        job, prov = m["job"], (m.get("meta") or {}).get("provenance") or {}
        phases[job["phase"]]["jobs"] += 1
        for a in m["result"].get("artifacts", []):
            k = a["key"]
            if not k.startswith("records/") or not (root / k).is_file():
                continue
            table = k.split("/")[1]
            if table not in schemas:
                problems.append(f"{k}: unknown table {table!r}")
                continue
            if k != f"records/{table}/{job['phase']}/{job['job_id']}.parquet":
                problems.append(f"{k}: not the {table} part of job {job['phase']}/{job['job_id']}")
            try:
                schema = pq.read_schema(root / k)
            except Exception as e:
                problems.append(f"{k}: unreadable Parquet ({type(e).__name__}: {e})")
                continue
            if not schema.remove_metadata().equals(schemas[table]):
                problems.append(f"{k}: schema differs from dataset v1's {table} table")
                continue
            n = pq.read_metadata(root / k).num_rows
            tables[table]["parts"] += 1
            tables[table]["rows"] += n
            if table == "points":
                phases[job["phase"]]["points"] += n
            if table != "runs":
                continue
            want = {"schema_version": SCHEMA_VERSION, "harness_version": prov.get("harness_version"),
                    "config_hash": prov.get("config_hash"), "calibration_hash": prov.get("calibration_hash"),
                    "phase": job["phase"], "job_id": job["job_id"]}
            for row in pq.read_table(root / k, columns=[*want, "run_kind"]).to_pylist():
                phases[job["phase"]][f"{row['run_kind']}_runs"] += 1
                bad = [f for f, v in want.items() if row[f] != v]
                if bad:
                    problems.append(f"{k}: run row's {', '.join(bad)} differ from its marker")

    if problems:
        raise ManifestError(problems)

    shutil.copyfile(DATACARD_SOURCE, root / DATACARD)
    single = {f: json.loads(next(iter(seen[f]))) if seen[f] else None for f in barrier}
    if frozen is not None:
        single["ops_digest"] = None
        single["ops_digests"] = dict(ops_digests)
    files.sort(key=lambda f: f["path"])
    manifest = {
        "manifest_version": MANIFEST_VERSION, "dataset": DATASET, "schema_version": SCHEMA_VERSION,
        **single, "git_commits": sorted((c for c in commits if c), key=str),
        "configs": configs, "calibrations": dict(calibrations),
        "freeze": None if frozen is None else {k: frozen.get(k) for k in ("harness_version", "calibration_hash",
                                                                          "calibration_lineage", "ops_digest")},
        "data_digests": data_digests,
        "phases": {p: dict(sorted(c.items())) for p, c in sorted(phases.items())},
        "tables": tables, "warnings": warnings,
        "datacard": {"path": DATACARD, "sha256": sha256_file(root / DATACARD), "size": (root / DATACARD).stat().st_size},
        "totals": {"files": len(files), "bytes": sum(f["size"] for f in files)},
        "checked": "sizes" if quick else "sha256",
        "created_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "files": files,
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def _tunable_problems(mkey: str, m: dict) -> list[str]:
    """Tunables are recorded in the job specs they affect (amendment 49)."""
    from .branchplan import BRANCH_ORDER
    job, meta = m["job"], m.get("meta") or {}
    spec = job.get("spec") or {}
    if job.get("kind") == "episode" and job.get("phase") == "explore":
        if spec.get("p_escalate") is None:
            return [f"{mkey}: exploration job without p_escalate in its spec"]
        if meta.get("p_escalate") is None or float(spec["p_escalate"]) != float(meta["p_escalate"]):
            return [f"{mkey}: spec p_escalate {spec['p_escalate']} but its worker ran {meta.get('p_escalate')}"]
    if job.get("kind") in BRANCH_ORDER:
        s = spec.get("settings") or {}
        dp = s.get("dp_sampling") or {}
        if not s.get("seeds") or not s.get("enabled") or dp.get("fraction") is None or dp.get("seed") is None:
            return [f"{mkey}: branch job without its planner settings (seeds, enabled, dp sampling) in its spec"]
    return []


def _freeze_problems(mkey: str, m: dict, prov: dict, frozen: dict) -> list[str]:
    """Per-record rules once a freeze exists (amendment 49)."""
    meta, out = m.get("meta") or {}, []
    for f in ("harness_version", "source_digest"):
        if prov.get(f) != frozen.get(f):
            out.append(f"{mkey}: {f} is not the freeze's")
    ops = prov.get("ops_digest")
    if prov.get("calibration_hash") != frozen["calibration_hash"]:
        if ops != frozen.get("ops_digest"):
            out.append(f"{mkey}: a pre-freeze record (calibration in the lineage) with other ops code than the freeze's")
        return out
    g = meta.get("golden_pass") or {}
    if not (g.get("passed") is True and g.get("boot_id") and g.get("boot_id") == meta.get("boot_id")
            and g.get("calibration_hash") == frozen["calibration_hash"]):
        out.append(f"{mkey}: frozen-harness record without a golden-set pass for its boot")
    if ops and ops != frozen.get("ops_digest"):
        w = meta.get("ops_waiver") or {}
        if w.get("ops_digest") != ops or w.get("freeze_ops_digest") != frozen.get("ops_digest"):
            out.append(f"{mkey}: other ops code than the freeze's without its worker's ops_waiver record")
    return out


def _data_digest_problems(markers: dict, frozen: dict, ops_archives) -> tuple[dict, list[str]]:
    """Every ops digest involved is a committed archive; common data digests equal the freeze's; generator digests
    agree within each record kind (amendment 49)."""
    from . import datadigest as DD
    if ops_archives is None:
        return {}, ["records with other ops code than the freeze's: pass --ops-archives (a copy of phase0/src/)"]
    by_ops: dict[str, dict] = {}
    for p in sorted(Path(ops_archives).glob("*.tar.gz")):
        try:
            files = DD.files_from_archive(p.read_bytes())
        except (DD.DigestError, OSError, ValueError):
            continue
        by_ops.setdefault(DD.ops_digest_of(files), {"archive": p.name, **DD.digests(files)})
    problems: list[str] = []
    def ops_of(m: dict) -> Optional[str]:
        return ((m.get("meta") or {}).get("provenance") or {}).get("ops_digest")
    involved = {frozen["ops_digest"]} | {ops_of(m) for m in markers.values()}
    for d in sorted(x for x in involved if x):
        if d not in by_ops:
            problems.append(f"ops digest {d[:12]} is not a committed archive in {ops_archives}")
    if problems:
        return {}, problems
    freeze_common = by_ops[frozen["ops_digest"]]["common"]
    kinds: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for mkey, m in markers.items():
        d = ops_of(m)
        if not d:
            continue
        if by_ops[d]["common"] != freeze_common:
            problems.append(f"{mkey}: data digest of {by_ops[d]['archive']} differs from the freeze's")
        kind = DD.kind_of(m["job"])
        if kind is None:
            if d != frozen["ops_digest"]:
                problems.append(f"{mkey}: no job-spec generator digest for phase {m['job'].get('phase')!r}")
            continue
        kinds[kind][by_ops[d][kind]].add(d)
    for kind, gens in kinds.items():
        if len(gens) > 1:
            problems.append(f"mixed {kind} job-spec generators: " + ", ".join(
                f"{str(g)[:12]} ({', '.join(by_ops[d]['archive'] for d in sorted(ds))})" for g, ds in gens.items()))
    used = {d for d in involved if d}
    return {by_ops[d]["archive"]: {k: v for k, v in by_ops[d].items() if k != "archive"} | {"ops_digest": d}
            for d in sorted(used)}, problems


def verify(root: Path | str, *, quick: bool = False) -> list[str]:
    """Problems found checking the tree under `root` against its MANIFEST.json; empty if it matches."""
    root = Path(root)
    try:
        manifest = json.loads((root / MANIFEST).read_text())
    except (OSError, ValueError) as e:
        return [f"{MANIFEST}: unreadable ({e})"]
    listed = {f["path"]: f for f in manifest["files"]}
    listed[DATACARD] = manifest["datacard"]
    problems = []
    for path, f in sorted(listed.items()):
        p = root / path
        if not p.is_file():
            problems.append(f"{path}: missing")
        elif p.stat().st_size != f["size"]:
            problems.append(f"{path}: {p.stat().st_size} bytes, manifest says {f['size']}")
        elif not quick and sha256_file(p) != f["sha256"]:
            problems.append(f"{path}: SHA-256 differs from the manifest")
    problems += [f"{k}: not in the manifest" for k in _tree(root) if k != MANIFEST and k not in listed]
    return problems
