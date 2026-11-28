"""Calibration files for tests (amendments 34, 35): a complete set of calibration values, written through the
production writer, optionally frozen. The numbers equal the base config's placeholders unless overridden, so clocks
and limits in the tests do not move."""

from __future__ import annotations

from pathlib import Path

from vlar_harness import config as C


def write_once(*, pi05_infer_s=0.435, upload=20290, max_batch=8, sample_rows=40) -> dict:
    choices = C._read_yaml(C.CONFIG_DIR / "harness-v1.yaml")["options"]["network"]["choices"]
    return {"latency": {"pi05_infer_s": pi05_infer_s},
            "servers": {"batching": {"max_batch": max_batch, "xvla_sample_rows": sample_rows}},
            "options": {"network": {"choices": {c: {"upload_image_bytes": upload,
                                                    "upload_image_bytes_status": "measured"} for c in choices}}}}


def full_values(*, limit_s=37.2, thresholds=(0.01, 0.07), workers=4, **once) -> dict:
    vals = C._deep_merge(write_once(**once), {
        "latency": {"status": "measured", "xvla_act_s": 0.420, "xvla_features_s": 0.420, "xvla_sample_k_s": 2.100},
        "servers": {"workers": workers, "batching": {"status": "measured"}},
        "measurement": {"gripper_thresholds": {"status": "calibrated", "shut_width_m": float(thresholds[0]),
                                               "open_width_m": float(thresholds[1])}},
        "time_limit": {"status": "calibrated", "margin": 1.5,
                       "per_task_s": {t: float(limit_s) for t in range(10)}}})
    return vals


def calibration(dir_: Path, *, frozen=False, values=None, **kw) -> Path:
    """A calibration file under `dir_` (with its versions/), frozen if asked."""
    path = Path(dir_) / "calibration.yaml"
    C.write_calibration(values if values is not None else full_values(**kw), path, note="test")
    if frozen:
        C.write_calibration({}, path, note="test freeze", freeze=True)
    return path


def frozen_config(dir_: Path, **kw):
    """A config whose calibration file is complete and frozen (live time cut and observe() allowed)."""
    choices = kw.pop("choices", {})
    return C.load_config(calibration=calibration(dir_, frozen=True, **kw), **choices)
