"""One VM price everywhere (the user, 2026-11-26): the Spot g2-standard-16 with its L4 as billed, $0.6865/h (SPEC §17
amendment 65 (2)). The final tier's sizing once picked up replan's pre-billing-check $0.472; every default in code
must equal the guard's configured price."""
import argparse
import dataclasses
import re
from pathlib import Path

from vlar_ops import phase1, replan, spend

REPO = Path(__file__).parents[1]
BILLED = 0.6865


def test_every_hourly_default_is_the_billed_price():
    cfg = (REPO / "ops" / "ops.example.yaml").read_text()
    assert float(re.search(r"^  vm_hourly_usd: *([0-9.]+)", cfg, re.M).group(1)) == BILLED
    assert replan.HOURLY_USD == phase1.BILLED_HOURLY_USD == BILLED
    assert {f.name: f.default for f in dataclasses.fields(spend.SpendConfig)}["vm_hourly_usd"] == BILLED
    assert phase1.DRAIN_MARGIN_USD == (900 + 300) / 3600 * BILLED


def test_no_cli_default_or_code_literal_carries_an_old_price():
    src = "\n".join(p.read_text() for p in (REPO / "src" / "vlar_ops").glob("*.py"))
    hits = [m.group(0) for m in re.finditer(r"default=0\.4[67]\d*|= *0\.4(6|72)\b", src)]
    assert hits == []
    defaults = re.findall(r'"--hourly-usd", type=float, default=([0-9.]+)', src)
    assert defaults and all(float(d) == BILLED for d in defaults)
