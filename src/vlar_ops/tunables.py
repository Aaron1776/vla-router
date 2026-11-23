"""Collection tunables (docs/SPEC.md §7, §12, amendment 49): settings that choose which work is done, in
`configs/collection-v1-tunables.yaml`. Each is recorded in every job spec or branch-plan entry it affects and read
from there, never from this file, so changing one (a §12 reduction: fewer seeds, dropping b3) leaves the data digest
(`vlar_ops.datadigest`) unchanged. The definitions (what existing records mean) stay in `configs/collection-v1.yaml`,
read by `plan.load_collection_config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

TUNABLES_FILE = Path(__file__).parent / "configs" / "collection-v1-tunables.yaml"


class TunableError(ValueError):
    pass


def load_tunables(path: Path | str = TUNABLES_FILE) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def explore_fields(tunables: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """What every exploration job spec records (`episode_plan`'s `extra`)."""
    e = (tunables or load_tunables())["exploration"]
    n = int(e["episodes_per_state"])
    if n != 1:
        raise TunableError(f"exploration.episodes_per_state is {n}: only 1 is implemented (one job per state)")
    return {"p_escalate": float(e["p_escalate"]), "episodes_per_state": n}
