"""Versioned harness config with named options only (docs/SPEC.md §15), and the Phase 0 calibration file
(amendments 34, 35).

A config is `configs/<name>.yaml` (hand-written, part of the source digest) deep-merged with the calibration file
named by `VLAR_CALIBRATION` (on the VM `/etc/vlar/calibration/calibration.yaml`, mounted read-only into the
containers; never in the package). The calibration file holds only the values Phase 0 measures, under
`CALIBRATION_PATHS`; a write-once group (`WRITE_ONCE_PATHS`: the upload size, π0.5's latency and the server batch
geometry) is set in step 2 before any data and never changed. Experiments choose among the named `options`; they
cannot set arbitrary values. The config hash covers the fixed settings and the chosen options; the calibration file
is identified by its own SHA-256 (`calibration_hash`); both go into every record.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

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

# Paths Phase 0 may write through the calibration file (§11 steps 2 and 6). `*` matches one path segment.
CALIBRATION_PATHS = (
    "latency",
    "servers.batching",
    "servers.workers",
    "measurement.gripper_thresholds",
    "time_limit.margin",
    "time_limit.per_task_s",
    "time_limit.status",
    "options.network.choices.*.upload_image_bytes",          # §5, amendment 14 (Phase 0 step 2)
    "options.network.choices.*.upload_image_bytes_status",
)
# The write-once group (amendment 35): decided in Phase 0 step 2 before the parity run, immutable afterwards. They
# change logged outcomes (network timeouts) or model outputs at the bit level (batch geometry).
WRITE_ONCE_PATHS = (
    "latency.pi05_infer_s",
    "options.network.choices.*.upload_image_bytes",
    "servers.batching.max_batch",
    "servers.batching.xvla_sample_rows",
)
CALIBRATION_ENV = "VLAR_CALIBRATION"
CALIBRATION_SCHEMA = 1
VERSIONS_DIR = "versions"          # beside the calibration file: every version written, as <sha256>.yaml
FROM_ENV = object()                # load_config(calibration=FROM_ENV): the file named by VLAR_CALIBRATION, if any


class ConfigError(ValueError):
    pass


class CalibrationNotFrozen(RuntimeError):
    """A live calibration-dependent value (the live time cut, `observe()`) before the calibration file is frozen
    (amendments 34, 35)."""


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


def _matches(path: str, patterns: tuple) -> bool:
    parts = path.split(".")
    for allowed in patterns:
        pattern = allowed.split(".")
        if len(parts) >= len(pattern) and all(p == "*" or p == s for p, s in zip(pattern, parts)):
            return True
    return False


def _calibration_path_allowed(path: str) -> bool:
    return _matches(path, CALIBRATION_PATHS)


def _get_path(obj: Mapping, dotted: str, default: Any = None) -> Any:
    node: Any = obj
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


@dataclass(frozen=True)
class HarnessConfig:
    name: str
    values: Mapping[str, Any]               # merged config, `options` replaced by the chosen values
    selected: Mapping[str, str]             # option name -> chosen choice name
    hash: str = field(default="")           # fixed settings and chosen options; not the calibration (amendment 35)
    calibration_hash: Optional[str] = None  # SHA-256 of the calibration file in effect; None without one
    calibration_path: Optional[str] = None
    write_once_missing: tuple = ()          # write-once paths the calibration file has not set yet

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
        """The calibration file is frozen (Phase 0 step 7); live calibration-dependent values refuse until then."""
        return bool(self.values.get("frozen"))

    @property
    def write_once_complete(self) -> bool:
        return self.calibration_hash is not None and not self.write_once_missing

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
            "calibration_hash": self.calibration_hash,
        }


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must hold a mapping")
    return data


def calibration_file(calibration: Any = FROM_ENV) -> Optional[Path]:
    """The calibration file to use: an explicit path, None for none, or (FROM_ENV) the one VLAR_CALIBRATION names.
    A named file that does not exist is an error, so a production run can never silently use placeholders."""
    if calibration is FROM_ENV:
        env = os.environ.get(CALIBRATION_ENV)
        calibration = Path(env) if env else None
    if calibration is None:
        return None
    path = Path(calibration)
    if not path.is_file():
        raise ConfigError(f"calibration file {path} does not exist ({CALIBRATION_ENV})")
    return path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_calibration(path: Path, name: str = DEFAULT_CONFIG) -> dict:
    """A calibration file: {schema, config, frozen, values, history}; `values` only under CALIBRATION_PATHS."""
    cal = _read_yaml(Path(path))
    if cal.get("schema") != CALIBRATION_SCHEMA or cal.get("config") != name:
        raise ConfigError(f"{path} is not a schema-{CALIBRATION_SCHEMA} calibration file for {name}")
    unknown = set(cal) - {"schema", "config", "frozen", "values", "history"}
    if unknown:
        raise ConfigError(f"{path} has unknown keys {sorted(unknown)}")
    values = cal.get("values") or {}
    if not isinstance(values, dict):
        raise ConfigError(f"{path}: values must be a mapping")
    bad = [p for p in _flatten_paths(values) if not _calibration_path_allowed(p)]
    if bad:
        raise ConfigError(f"{path.name} touches non-calibration paths: {bad}")
    cal["values"], cal["frozen"] = values, bool(cal.get("frozen"))
    return cal


def _write_once_paths(base: Mapping) -> list[str]:
    """WRITE_ONCE_PATHS with `*` expanded over the base config's network choices."""
    out = []
    for p in WRITE_ONCE_PATHS:
        if "*" in p:
            head, _, tail = p.partition(".*.")
            out.extend(f"{head}.{c}.{tail}" for c in _get_path(base, head, {}) or {})
        else:
            out.append(p)
    return out


def write_once_values(values: Mapping, base: Mapping) -> dict[str, Any]:
    """The write-once group as set in calibration `values` (missing paths left out)."""
    sentinel = object()
    out = {}
    for p in _write_once_paths(base):
        v = _get_path(values, p, sentinel)
        if v is not sentinel and v is not None:
            out[p] = v
    return out


def load_raw(name: str = DEFAULT_CONFIG, config_dir: Path = CONFIG_DIR, calibration: Any = FROM_ENV) -> dict:
    """The base config merged with the calibration file (if any), with `frozen` from the calibration file."""
    base = _read_yaml(config_dir / f"{name}.yaml")
    if "frozen" in base:
        raise ConfigError(f"{name}.yaml must not set `frozen`: the calibration file is frozen (amendment 35)")
    path = calibration_file(calibration)
    if path is None:
        return dict(base, frozen=False)
    cal = read_calibration(path, name)
    return dict(_deep_merge(base, cal["values"]), frozen=cal["frozen"])


def _resolve_options(raw: Mapping, choices: Mapping[str, str]) -> tuple[dict, dict]:
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
    return values, selected


def load_config(name: str = DEFAULT_CONFIG, config_dir: Path = CONFIG_DIR, *, calibration: Any = FROM_ENV,
                **choices: str) -> HarnessConfig:
    """Load a config and pick named options, e.g. `load_config(network="default")`. `calibration`: a path, None
    for the base config alone, or (default) the file VLAR_CALIBRATION names, if set."""
    base = _read_yaml(config_dir / f"{name}.yaml")
    path = calibration_file(calibration)
    raw = load_raw(name, config_dir, path)
    values, selected = _resolve_options(raw, choices)
    validate(values, raw)
    fixed, _ = _resolve_options(base, choices)          # the hash: fixed settings and options, no calibration
    digest = hashlib.sha256(canonical_json({"config": fixed, "selected": selected}).encode()).hexdigest()
    if path is None:
        return HarnessConfig(name=name, values=values, selected=selected, hash=digest,
                             write_once_missing=tuple(_write_once_paths(base)))
    cal = read_calibration(path, name)
    have = write_once_values(cal["values"], base)
    return HarnessConfig(name=name, values=values, selected=selected, hash=digest, calibration_hash=file_sha256(path),
                         calibration_path=str(path),
                         write_once_missing=tuple(p for p in _write_once_paths(base) if p not in have))


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
        need(set(spec) == {"chunk_point", "mid_point", "act_fresh"},
             f"profile {p} needs chunk_point, mid_point and act_fresh")
        for kind in ("chunk_point", "mid_point", "act_fresh"):
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
    b = values["servers"]["batching"]
    for k in ("max_batch", "xvla_sample_rows"):
        need(isinstance(b.get(k), int) and not isinstance(b.get(k), bool) and b[k] >= 1,
             f"servers.batching.{k} must be an integer >= 1")
    if isinstance(b.get("xvla_sample_rows"), int):
        need(b["xvla_sample_rows"] >= values["measurement"]["K"], "servers.batching.xvla_sample_rows must hold K samples")
    w = values["servers"].get("workers")
    need(w is None or (isinstance(w, int) and not isinstance(w, bool) and w >= 1), "servers.workers must be >= 1")
    if values.get("frozen"):
        need(w is not None, "frozen config needs servers.workers")
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
    need(isinstance(net.get("max_attempts_pi05_only"), int) and net["max_attempts_pi05_only"] >= 1,
         "network max_attempts_pi05_only must be an integer >= 1")
    need(isinstance(net.get("upload_image_bytes"), int) and net["upload_image_bytes"] > 0,
         "network upload_image_bytes must be a positive byte count")
    for k in ("upload_other_bytes", "download_bytes"):
        need(isinstance(net.get(k), int) and net[k] >= 0, f"network {k} must be a byte count >= 0")
    if values.get("frozen"):
        for choice, spec in raw.get("options", {}).get("network", {}).get("choices", {}).items():
            need(spec.get("upload_image_bytes_status") != "placeholder",
                 f"frozen config has a placeholder upload size in network choice {choice!r}")
    need(values["option_values"].get("render") in ("decision_points", "every_step"),
         "option render must be decision_points or every_step")
    scene = values["option_values"].get("scene_seed")
    need(isinstance(scene, int) and not isinstance(scene, bool) and scene >= 0,
         "option scene_seed must be a non-negative integer offset")

    if errs:
        raise ConfigError("invalid harness config:\n  - " + "\n  - ".join(errs))


def calibration_target(path: Any = FROM_ENV) -> Path:
    """Where the calibration writer writes: an explicit path or VLAR_CALIBRATION (the file may not exist yet)."""
    if path is FROM_ENV:
        env = os.environ.get(CALIBRATION_ENV)
        if not env:
            raise ConfigError(f"no calibration file: pass a path or set {CALIBRATION_ENV}")
        path = env
    return Path(path)


def write_calibration(updates: Mapping[str, Any], path: Any = FROM_ENV, *, name: str = DEFAULT_CONFIG,
                      config_dir: Path = CONFIG_DIR, note: str = "", freeze: bool = False) -> Path:
    """Merge Phase 0 measurements into the calibration file (amendments 34, 35). Refuses once the file is frozen,
    refuses paths outside CALIBRATION_PATHS, and refuses to change a write-once value once it is set (setting it
    again to the same value is a no-op). The result must validate before it replaces the file; every version
    written is also kept as `versions/<sha256>.yaml` beside it (the freeze's lineage reads them).
    `freeze=True` marks the file frozen (Phase 0 step 7; see provenance.write_freeze)."""
    path = calibration_target(path)
    base = _read_yaml(config_dir / f"{name}.yaml")
    if path.exists():
        current = read_calibration(path, name)
        previous = file_sha256(path)
    else:
        current, previous = {"schema": CALIBRATION_SCHEMA, "config": name, "frozen": False, "values": {},
                             "history": []}, None
    if current["frozen"]:
        raise ConfigError(f"{path} is frozen; calibration changes need a new harness version")
    bad = [p for p in _flatten_paths(updates) if not _calibration_path_allowed(p)]
    if bad:
        raise ConfigError(f"not calibration paths: {bad}")
    merged = _deep_merge(current["values"], updates)
    before, after = write_once_values(current["values"], base), write_once_values(merged, base)
    changed = [f"{p} (set to {before[p]!r})" for p in before if after.get(p) != before[p]]
    if changed:
        raise ConfigError(f"write-once calibration values cannot change once set (amendment 35): {changed}")
    history = list(current.get("history") or [])
    history.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "paths": _flatten_paths(updates) + (["frozen"] if freeze else []), "note": note,
                    "previous": previous})
    out = {"schema": CALIBRATION_SCHEMA, "config": name, "frozen": bool(freeze), "values": merged,
           "history": history}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(out, sort_keys=False))
    try:
        load_config(name, config_dir, calibration=tmp)       # the merged result must validate
    except ConfigError:
        tmp.unlink()
        raise
    versions = path.parent / VERSIONS_DIR
    versions.mkdir(exist_ok=True)
    (versions / f"{file_sha256(tmp)}.yaml").write_bytes(tmp.read_bytes())
    tmp.replace(path)
    return path


def write_upload_size(upload_image_bytes: int, path: Any = FROM_ENV, *, name: str = DEFAULT_CONFIG,
                      config_dir: Path = CONFIG_DIR, note: str = "") -> Path:
    """Phase 0 step 2 (§5, amendments 14, 35): store the measured median upload image size in every network choice
    (the images do not depend on the network's quality). Write-once."""
    choices = _read_yaml(config_dir / f"{name}.yaml").get("options", {}).get("network", {}).get("choices", {})
    if not choices:
        raise ConfigError("the config has no network choices")
    value = {"upload_image_bytes": int(upload_image_bytes), "upload_image_bytes_status": "measured"}
    return write_calibration({"options": {"network": {"choices": {c: dict(value) for c in choices}}}},
                             path, name=name, config_dir=config_dir, note=note)


def calibration_lineage(final: Path, name: str = DEFAULT_CONFIG, config_dir: Path = CONFIG_DIR) -> list[str]:
    """Amendment 35: the hashes of the earlier versions (in `versions/` beside `final`) whose write-once group is
    complete and equal to the final file's. Records made under any of them count as the final version's."""
    base = _read_yaml(config_dir / f"{name}.yaml")
    want = write_once_values(read_calibration(final, name)["values"], base)
    if len(want) != len(_write_once_paths(base)):
        raise ConfigError(f"{final}: the write-once group is incomplete")
    final_hash, out = file_sha256(final), []
    for v in sorted((Path(final).parent / VERSIONS_DIR).glob("*.yaml")):
        h = file_sha256(v)
        if v.stem != h:
            raise ConfigError(f"{v} does not hash to its name")
        if h != final_hash and write_once_values(read_calibration(v, name)["values"], base) == want:
            out.append(h)
    return out
