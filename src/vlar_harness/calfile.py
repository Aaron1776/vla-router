"""The Phase 0 calibration file on the VM (amendments 34, 35): `python -m vlar_harness.calfile <command>`.

Runs inside an image, so the writer and its checks are the image's own code. The file is named by VLAR_CALIBRATION
(on the VM `/etc/vlar/calibration/calibration.yaml`; its `versions/` and the freeze's `freeze.json` beside it).

    show                         hash, frozen, the write-once group and what it still misses (JSON)
    write UPDATES.json [--note]  merge measured values (config.write_calibration: refuses non-calibration paths,
                                 changes to write-once values, and anything once frozen)
    get DOTTED.PATH              one value of the merged config (e.g. servers.workers), as JSON
    require-write-once           exit 0 if the write-once group is complete, else 3 with what is missing
    batching-env                 the servers' start-up geometry (VLAR_MAX_BATCH, VLAR_XVLA_SAMPLE_ROWS,
                                 VLAR_WAIT_WINDOW_MS) for /etc/vlar/batching.env; refuses without the group
"""

from __future__ import annotations

import argparse
import json
import sys

from . import config as C

INCOMPLETE_EXIT = 3


def summary(cfg: C.HarnessConfig) -> dict:
    base = C._read_yaml(C.CONFIG_DIR / f"{cfg.name}.yaml")
    values = C.read_calibration(C.Path(cfg.calibration_path), cfg.name)["values"] if cfg.calibration_path else {}
    return {"calibration_path": cfg.calibration_path, "calibration_hash": cfg.calibration_hash, "frozen": cfg.frozen,
            "write_once": C.write_once_values(values, base), "write_once_missing": list(cfg.write_once_missing),
            "config_hash": cfg.hash, "harness_version": cfg["harness_version"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vlar_harness.calfile")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    w = sub.add_parser("write")
    w.add_argument("updates", help="JSON file of calibration values (nested like the config)")
    w.add_argument("--note", default="")
    g = sub.add_parser("get")
    g.add_argument("path")
    sub.add_parser("require-write-once")
    sub.add_parser("batching-env")
    a = ap.parse_args(argv)
    if a.cmd == "write":
        with open(a.updates) as f:
            updates = json.load(f)
        path = C.write_calibration(updates, note=a.note)
        print(json.dumps(summary(C.load_config(calibration=path)), indent=1))
        return 0
    cfg = C.load_config()
    if a.cmd == "show":
        print(json.dumps(summary(cfg), indent=1))
        return 0
    if a.cmd == "get":
        print(json.dumps(cfg.get(a.path)))
        return 0
    if not cfg.write_once_complete:
        print(f"write-once calibration incomplete ({C.CALIBRATION_ENV}={cfg.calibration_path}): missing "
              f"{list(cfg.write_once_missing)}", file=sys.stderr)
        return INCOMPLETE_EXIT
    if a.cmd == "batching-env":
        b = cfg["servers"]["batching"]
        print(f"VLAR_MAX_BATCH={int(b['max_batch'])}\nVLAR_XVLA_SAMPLE_ROWS={int(b['xvla_sample_rows'])}\n"
              f"VLAR_WAIT_WINDOW_MS={float(b['wait_window_ms']):g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
