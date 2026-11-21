"""Versioned harness config with named options only (docs/SPEC.md §15).

A config is `configs/<name>.yaml` (hand-written) deep-merged with the optional
`configs/<name>.calibration.yaml` (written by Phase 0 measurement scripts, restricted
to the calibration paths below). Experiments choose among the named `options`; they
cannot set arbitrary values. The config hash covers the merged config and the chosen
options, and is written into every record.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import HARNESS_VERSION

CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG = "harness-v1"

MODES = ("routed", "xvla_only", "pi05_only")
PROFILES = ("none", "router", "cascader_nospread", "cascader_chunkspread", "cascader_full")
RECORD_LEVELS = ("off", "full")
OVERHEAD_TERMS = ("features", "sample_k", "sample_k_less_act")
INPUT_GROUPS = ("common", "proposal", "free_signals", "chunk_spread", "fresh_spread", "drift")
BRANCHES = ("b1_continue", "b2_one_call", "b3_three_calls", "b4_max")

# Paths Phase 0 may write through the calibration file (§11 steps 2 and 6).
CALIBRATION_PATHS = (
    "latency",
    "servers.batching",
    "measurement.gripper_thresholds",
    "time_limit.margin",
    "time_limit.per_task_s",
    "time_limit.status",
)


class ConfigError(ValueError):
    pass


def _normalize(obj: Any) -> Any:
    """Mapping keys to str (YAML gives int keys for per-task tables) so hashing is stable."""
    if isinstance(obj, Mapping):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _deep_merge(base: dict, over: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _flatten_paths(obj: Mapping, prefix: str = "") -> list[str]:
    paths = []
    for k, v in obj.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping) and v:
            paths.extend(_flatten_paths(v, p))
        else:
            paths.append(p)
    return paths


def _calibration_path_allowed(path: str) -> bool:
    return any(path == a or path.startswith(a + ".") for a in CALIBRATION_PATHS)


@dataclass(frozen=True)
class HarnessConfig:
    name: str
    values: Mapping[str, Any]               # merged config, `options` replaced by the chosen values
    selected: Mapping[str, str]             # option name -> chosen choice name
    hash: str = field(default="")

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.values
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def frozen(self) -> bool:
        return bool(self.values.get("frozen"))

    @property
    def step_limit(self) -> int:
        return int(self.values["env"]["step_limit"])

    @property
    def interval(self) -> int:
        return int(self.values["decision"]["interval_steps"])

    @property
    def n_decision_points(self) -> int:
        return self.step_limit // self.interval

    @property
    def max_escalations(self) -> int:
        return int(self.values["budget"]["pi05_max_steps"]) // int(self.values["decision"]["segment_steps"])

    def option(self, name: str) -> Any:
        return self.values["option_values"][name]

    def time_limit_s(self, task_id: int) -> float:
        table = self.values["time_limit"]["per_task_s"]
        return float(table.get(task_id, table.get(str(task_id))))

    def record_fields(self) -> dict[str, Any]:
        return {
            "harness_version": self.values["harness_version"],
            "config_name": self.name,
            "config_hash": self.hash,
            "config_options": dict(self.selected),
        }


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must hold a mapping")
    return data


def load_raw(name: str = DEFAULT_CONFIG, config_dir: Path = CONFIG_DIR) -> dict:
    base = _read_yaml(config_dir / f"{name}.yaml")
    cal_path = config_dir / f"{name}.calibration.yaml"
    if cal_path.exists():
        cal = _read_yaml(cal_path)
        cal.pop("_provenance", None)
        bad = [p for p in _flatten_paths(cal) if not _calibration_path_allowed(p)]
        if bad:
            raise ConfigError(f"{cal_path.name} touches non-calibration paths: {bad}")
        base = _deep_merge(base, cal)
    return base


def load_config(name: str = DEFAULT_CONFIG, config_dir: Path = CONFIG_DIR, **choices: str) -> HarnessConfig:
    """Load a config and pick named options, e.g. `load_config(network="default")`."""
    raw = load_raw(name, config_dir)
    options = raw.get("options", {})
    unknown = set(choices) - set(options)
    if unknown:
        raise ConfigError(f"unknown option(s) {sorted(unknown)}; available: {sorted(options)}")
    selected: dict[str, str] = {}
    option_values: dict[str, Any] = {}
    for opt, spec in options.items():
        choice = choices.get(opt, spec["default"])
        if choice not in spec["choices"]:
            raise ConfigError(f"option {opt}={choice!r} is not one of {sorted(spec['choices'])}")
        selected[opt] = choice
        option_values[opt] = copy.deepcopy(spec["choices"][choice])
    values = {k: v for k, v in raw.items() if k != "options"}
    values["option_values"] = option_values
    validate(values, raw)
    digest = hashlib.sha256(canonical_json({"config": values, "selected": selected}).encode()).hexdigest()
    return HarnessConfig(name=name, values=values, selected=selected, hash=digest)


def validate(values: Mapping[str, Any], raw: Mapping[str, Any]) -> None:
    errs: list[str] = []

    def need(cond: bool, msg: str) -> None:
        if not cond:
            errs.append(msg)

    need(values.get("schema") == 1, "schema must be 1")
    need(values.get("harness_version") == HARNESS_VERSION,
         f"harness_version {values.get('harness_version')!r} != package {HARNESS_VERSION!r}")
    env, dec, bud = values["env"], values["decision"], values["budget"]
    interval, seg = dec["interval_steps"], dec["segment_steps"]
    need(interval == seg, "segment_steps must equal interval_steps (one segment per decision)")
    need(env["step_limit"] % interval == 0, "step_limit must be a multiple of interval_steps")
    need(len(env.get("settle_action", [])) == 7, "env.settle_action must have 7 values")
    need(bud["pi05_max_steps"] % seg == 0, "pi05_max_steps must be a multiple of segment_steps")
    need(0 < bud["pi05_max_steps"] <= env["step_limit"], "pi05_max_steps must be in (0, step_limit]")
    models = values["models"]
    need(models["pi05"]["n_action_steps"] == seg, "pi0.5 n_action_steps must equal segment_steps")
    need(models["pi05"]["n_action_steps"] <= models["pi05"]["chunk_len"], "pi0.5 executes more than it returns")
    need(models["xvla"]["chunk_len"] % seg == 0, "X-VLA chunk_len must be a multiple of segment_steps")

    lat = values["latency"]
    for k in ("xvla_act_s", "xvla_features_s", "xvla_sample_k_s", "pi05_infer_s"):
        need(isinstance(lat.get(k), (int, float)) and lat[k] > 0, f"latency.{k} must be > 0")
    if all(isinstance(lat.get(k), (int, float)) for k in ("xvla_act_s", "xvla_sample_k_s")):
        need(lat["xvla_sample_k_s"] >= lat["xvla_act_s"], "sample_k time below /act time gives a negative charge")
    need(values["measurement"]["K"] >= 2, "measurement.K must be >= 2")

    need(tuple(values["profiles"]) == PROFILES, f"profiles must be exactly {PROFILES}")
    for p, spec in values["profiles"].items():
        need(set(spec) == {"chunk_point", "mid_point"}, f"profile {p} needs chunk_point and mid_point")
        for kind in ("chunk_point", "mid_point"):
            bad = set(spec.get(kind, [])) - set(OVERHEAD_TERMS)
            need(not bad, f"profile {p}.{kind} has unknown terms {sorted(bad)}")
    need(tuple(values["observe_inputs"]) == PROFILES, f"observe_inputs must cover exactly {PROFILES}")
    for p, groups in values["observe_inputs"].items():
        bad = set(groups) - set(INPUT_GROUPS)
        need(not bad, f"observe_inputs.{p} has unknown groups {sorted(bad)}")
        need(p == "none" or "common" in groups, f"observe_inputs.{p} must include common")
        need(p == "router" or p == "none" or "proposal" in groups, f"cascader {p} must include proposal")
    need("proposal" not in values["observe_inputs"]["router"], "the router must never see the proposal")

    need(tuple(values["branches"]) == BRANCHES, f"branches must be exactly {BRANCHES}")
    for b, spec in values["branches"].items():
        c = spec.get("calls")
        need(c == "max" or (isinstance(c, int) and c >= 0), f"branches.{b}.calls must be an int >= 0 or 'max'")

    tl = values["time_limit"]
    per_task = {int(k): v for k, v in tl["per_task_s"].items()}
    need(set(per_task) == set(range(env["n_tasks"])), "time_limit.per_task_s must cover every task")
    need(all(isinstance(v, (int, float)) and v > 0 for v in per_task.values()), "time limits must be > 0")
    need(tl["margin"] is None or tl["margin"] in tl["margin_candidates"], "margin must be null or a candidate")
    if values.get("frozen"):
        for section in ("latency", "time_limit"):
            need(values[section].get("status") != "placeholder", f"frozen config still has placeholder {section}")
        need(values["servers"]["batching"].get("status") != "placeholder", "frozen config has placeholder batching")
        need(values["measurement"]["gripper_thresholds"].get("status") != "uncalibrated",
             "frozen config has uncalibrated gripper thresholds")
        need(tl["margin"] is not None, "frozen config needs a margin")

    for opt, spec in raw.get("options", {}).items():
        need(spec.get("default") in spec.get("choices", {}), f"option {opt} default is not a choice")
    net = values["option_values"].get("network", {})
    for k in ("uplink_mbps", "downlink_mbps", "rtt_median_s", "rtt_tail_s", "timeout_s"):
        need(net.get(k, 0) > 0, f"network.{k} must be > 0")
    need(net.get("rtt_tail_s", 0) > net.get("rtt_median_s", 0), "network rtt_tail_s must exceed the median")
    need(0 < net.get("rtt_tail_prob", 0) < 0.5, "network rtt_tail_prob must be in (0, 0.5)")
    need(0 <= net.get("p_fail", -1) < 1, "network p_fail must be in [0, 1)")
    need(isinstance(net.get("max_retries"), int) and net["max_retries"] >= 0, "network max_retries must be >= 0")

    if errs:
        raise ConfigError("invalid harness config:\n  - " + "\n  - ".join(errs))


def write_calibration(updates: Mapping[str, Any], name: str = DEFAULT_CONFIG,
                      config_dir: Path = CONFIG_DIR, note: str = "") -> Path:
    """Merge Phase 0 measurements into `<name>.calibration.yaml`. Refuses once frozen."""
    base = _read_yaml(config_dir / f"{name}.yaml")
    if base.get("frozen"):
        raise ConfigError(f"{name} is frozen; calibration changes need a new harness version")
    bad = [p for p in _flatten_paths(updates) if not _calibration_path_allowed(p)]
    if bad:
        raise ConfigError(f"not calibration paths: {bad}")
    path = config_dir / f"{name}.calibration.yaml"
    current = _read_yaml(path) if path.exists() else {}
    history = list(current.pop("_provenance", []))
    merged = _deep_merge(current, updates)
    history.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "paths": _flatten_paths(updates), "note": note})
    merged["_provenance"] = history
    previous = path.read_bytes() if path.exists() else None
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(merged, sort_keys=False))
    tmp.replace(path)
    try:
        load_config(name, config_dir)  # the merged result must still validate
    except ConfigError:
        if previous is None:
            path.unlink()
        else:
            path.write_bytes(previous)
        raise
    return path
