"""The ops-only waiver at a worker's startup (docs/SPEC.md §15, amendment 49, refining 46).

After the freeze the ops digest is provenance, not a barrier: `vm_worker.serve` runs a frozen harness under ops
code other than the freeze's only if `check` passes, and every record it makes then carries `meta.ops_waiver`.
`check` accepts the `FreezeError` only if all of these hold:
- its message is exactly the frozen harness's ops-only text (no other difference);
- `verify_freeze` passes again with `provenance.ops_digest` patched to the freeze's;
- the running ops code comes from a committed archive: `VLAR_OPS_SOURCE` names `phase0/src/<commit>.tar.gz` and
  that archive's `vlar_ops` digest equals the running one;
- the freeze's archive (found among `phase0/src/*.tar.gz` by its ops digest) has the same common data digest
  (`vlar_ops.datadigest`). The job-spec generator digests are recorded and compared per kind by the manifest.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from . import datadigest as DD

ARCHIVES = "phase0/src/"
ARCHIVE_NAME = re.compile(r"^[0-9a-f]{40}\.tar\.gz$")
OPS_ONLY = "ops code (vlar_ops)"


class WaiverRefused(RuntimeError):
    pass


def ops_only_message(rec: dict) -> str:
    """`provenance.verify_freeze`'s message when the ops code is the only difference."""
    return (f"harness differs from frozen {rec['harness_version']} in: {OPS_ONLY}. "
            "Any harness or ops-code change needs a new version and the Phase 0 tests (§15).")


def find_archive(storage, ops_digest: str, *, skip: Optional[str] = None) -> Optional[tuple[str, dict]]:
    """The committed archive whose `vlar_ops` has this ops digest: (name, files), or None."""
    for key in sorted(storage.list(ARCHIVES)):
        name = key[len(ARCHIVES):]
        if not ARCHIVE_NAME.match(name) or name == skip:
            continue
        data = storage.get_bytes(key)
        if data is None:
            continue
        try:
            files = DD.files_from_archive(data)
        except (DD.DigestError, OSError, ValueError):
            continue
        if DD.ops_digest_of(files) == ops_digest:
            return name, files
    return None


def check(cfg, err: Exception, storage, *, ops_source: Optional[str], config_dir: Optional[Path] = None) -> dict[str, Any]:
    """Whether a frozen harness may run under this ops code; returns the record's `ops_waiver`, else raises
    WaiverRefused with the reason."""
    from vlar_harness import provenance as P
    from vlar_harness.config import CONFIG_DIR
    try:
        rec = P.read_freeze(P.freeze_path(cfg))
    except (OSError, ValueError, P.FreezeError) as e:
        raise WaiverRefused(f"no readable freeze record ({e})") from e
    if not rec.get("ops_digest") or str(err) != ops_only_message(rec):
        raise WaiverRefused("the difference is not the ops code alone")
    real = P.ops_digest
    P.ops_digest = lambda: rec["ops_digest"]
    try:
        P.verify_freeze(cfg, config_dir or CONFIG_DIR)
    except P.FreezeError as e:
        raise WaiverRefused(f"with the freeze's ops digest the harness still differs: {e}") from e
    finally:
        P.ops_digest = real
    running = real()
    if not ops_source or not ARCHIVE_NAME.match(ops_source):
        raise WaiverRefused(f"VLAR_OPS_SOURCE is {ops_source!r}, not a committed archive <commit>.tar.gz")
    data = storage.get_bytes(ARCHIVES + ops_source)
    if data is None:
        raise WaiverRefused(f"{ARCHIVES}{ops_source} is not in the bucket")
    files = DD.files_from_archive(data)
    if DD.ops_digest_of(files) != running:
        raise WaiverRefused(f"the running ops code is not {ops_source}'s (uncommitted or other ops code)")
    found = find_archive(storage, rec["ops_digest"], skip=ops_source)
    if found is None:
        raise WaiverRefused(f"no archive in {ARCHIVES} has the freeze's ops digest {rec['ops_digest'][:12]}")
    freeze_source, freeze_files = found
    mine, theirs = DD.digests(files), DD.digests(freeze_files)
    if mine["common"] != theirs["common"]:
        raise WaiverRefused(f"data digest {mine['common'][:12]} differs from the freeze's ({freeze_source}: "
                            f"{theirs['common'][:12]}): the Episode wiring, seed derivation or definitions changed")
    return {"freeze_ops_digest": rec["ops_digest"], "ops_digest": running, "data_digest": mine,
            "ops_source": ops_source, "freeze_ops_source": freeze_source}
