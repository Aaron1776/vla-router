"""Spend ledger and guard (docs/SPEC.md §12).

Google Cloud budgets only alert; they never stop a VM. The guard runs on the VM, keeps a
ledger in the bucket (so it survives preemption and restarts), and:
  - at the soft limit sends one notification;
  - at the hard limit raises the stop flag (workers take no new jobs), waits for in-flight
    jobs up to a grace period, releases the rest cleanly, syncs results and shuts the VM down.

The estimate counts VM time from each boot's start (plus a per-boot allowance for time billed before the kernel
starts; boots whose startup failed before the guard ran are recorded by `startup.sh` and folded in), the boot disk for
every hour it exists (disks bill while the VM is stopped), bucket storage, and reconciliation adjustments against the
real billing report.

Several writers share the ledger (the daemon, `spend disk-created` at boot, `disk-deleted` and `reconcile` from the
Mac, possibly while the VM runs): every write is a conditional read-modify-write with retry (`update_ledger`), so no
update is lost (amendment 39). Alerts are recorded with the limit they fired at, and fire again when a raised limit is
crossed (amendment 43).

Per-phase stop lines (docs/PROPOSALS-2026-11-24.md §0.1 (d)): a phase is opened from the Mac (`spend phase-open`)
with its allocation and its line, the most its VM time may cost from the moment it opened (disk has its own line).
At most one phase is open. The guard raises the stop flag once the phase's VM spend reaches its stop point, the line
minus a margin for what still runs after a stop (in-flight jobs finishing, servers stopping, the final sync), so
workers claim nothing more and the runner starts no step; at the line itself it releases in-flight jobs at once,
syncs and shuts the VM down, as at the hard limit but with no drain. The runner also checks the ledger before every
step (`spend check`): a step that names its phase (`# spend-phase: NAME`) runs only while that phase is the open one.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

from .storage import Storage

HOURS_PER_MONTH = 730.0
LEDGER_KEY = "ops/ledger.json"
FAILED_BOOTS_PREFIX = "ops/failed-boots/"


@dataclass(frozen=True)
class SpendConfig:
    soft_limit_usd: float = 40.0
    hard_limit_usd: float = 45.0
    vm_hourly_usd: float = 0.46          # g2-standard-16 Spot with its L4; set from the current price
    boot_allowance_s: float = 180.0      # billed time before the kernel starts (uptime covers the rest; generous)
    storage_usd_per_gb_month: float = 0.020
    tick_s: float = 60.0
    storage_scan_every_s: float = 1800.0
    drain_grace_s: float = 900.0

    @staticmethod
    def from_mapping(m: dict[str, Any]) -> "SpendConfig":
        fields = SpendConfig.__dataclass_fields__
        return SpendConfig(**{k: float(v) for k, v in m.items() if k in fields})


def new_ledger() -> dict[str, Any]:
    return {"schema": 1, "sessions": [], "disks": [], "storage": {"accrued_usd": 0.0, "last_bytes": 0,
            "last_at": None, "usd_per_gb_month": None}, "adjustments": [], "alerts": {"soft": [], "hard": []},
            "phases": []}


def estimate(ledger: dict[str, Any], now: float) -> dict[str, float]:
    vm = sum((s["vm_seconds"] + s["boot_allowance_s"]) / 3600.0 * s["hourly_usd"] for s in ledger["sessions"])
    disk = 0.0
    for d in ledger["disks"]:
        end = d["until"] if d.get("until") is not None else now
        disk += d["gb"] * d["usd_per_gb_month"] * max(0.0, end - d["since"]) / 3600.0 / HOURS_PER_MONTH
    st = ledger["storage"]
    storage = st["accrued_usd"]
    if st["last_at"] is not None and st["usd_per_gb_month"]:
        storage += st["last_bytes"] / 1e9 * st["usd_per_gb_month"] * max(0.0, now - st["last_at"]) / 3600.0 / HOURS_PER_MONTH
    adj = sum(a["delta_usd"] for a in ledger["adjustments"])
    return {"vm_usd": vm, "disk_usd": disk, "storage_usd": storage, "adjustments_usd": adj,
            "total_usd": vm + disk + storage + adj}


class Notifier(Protocol):
    def notify(self, event: str, payload: dict[str, Any]) -> None: ...


class PrintNotifier:
    def notify(self, event: str, payload: dict[str, Any]) -> None:
        print(json.dumps({"severity": "WARNING", "event": event, **payload}), file=sys.stderr, flush=True)


class GcloudLogNotifier:
    """Writes a structured Cloud Logging entry; a log-based alert policy turns it into an email
    (ops/gcp/40_spend_alert_policy.sh). Falls back to stderr if gcloud is missing."""

    def __init__(self, log_name: str = "vlar-spend"):
        self.log_name = log_name

    def notify(self, event: str, payload: dict[str, Any]) -> None:
        body = json.dumps({"event": event, **payload})
        PrintNotifier().notify(event, payload)
        if shutil.which("gcloud"):
            subprocess.run(["gcloud", "logging", "write", self.log_name, body, "--payload-type=json",
                            "--severity=WARNING"], check=False, timeout=60)


class StopFlag:
    """A file the workers check before each claim."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def set(self, reason: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(reason + "\n")

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    def is_set(self) -> bool:
        return self.path.exists()


class LedgerConflict(RuntimeError):
    """The ledger kept changing under a writer for every attempt."""


def _migrate(ledger: dict[str, Any]) -> dict[str, Any]:
    """Ledgers written before the per-limit alert records (amendment 43): {"soft_at": t} -> {"soft": [...]}."""
    alerts = ledger.setdefault("alerts", {})
    for kind in ("soft", "hard"):
        if f"{kind}_at" in alerts:
            alerts.setdefault(kind, []).append({"at": alerts.pop(f"{kind}_at"), "limit_usd": None})
        alerts.setdefault(kind, [])
    ledger.setdefault("phases", [])
    return ledger


# ---------------------------------------------------------------- per-phase stop lines (PROPOSALS §0.1 (d))

PHASE_WIND_DOWN_S = 300.0     # after a phase stop: servers stop, results and logs sync, the VM shuts down


class PhaseError(ValueError):
    pass


def default_phase_margin_usd(cfg: "SpendConfig") -> float:
    """What can still be spent after a phase's stop point: the in-flight jobs' drain (at most the guard's drain
    grace) plus the wind-down, at the configured hourly price."""
    return (cfg.drain_grace_s + PHASE_WIND_DOWN_S) / 3600.0 * cfg.vm_hourly_usd


def open_phase(ledger: dict[str, Any]) -> Optional[dict[str, Any]]:
    live = [p for p in ledger.get("phases", []) if p.get("ended_at") is None]
    if len(live) > 1:
        raise PhaseError(f"{len(live)} open phases in the ledger: {[p['name'] for p in live]}")
    return live[0] if live else None


def phase_vm_usd(ledger: dict[str, Any], phase: Mapping[str, Any], now: float) -> float:
    """VM spend inside the phase's window [opened, closed or now): each session's billed seconds that fall in it,
    plus the boot allowance of every session that started in it. A boot that ran before the phase opened costs it
    nothing, also when its failed-boot record is folded into the ledger later."""
    t0 = float(phase["started_at"])
    t1 = float(phase["ended_at"]) if phase.get("ended_at") is not None else now
    usd = 0.0
    for s in ledger["sessions"]:
        start = float(s["started_at"])
        end = start + float(s["vm_seconds"])
        secs = max(0.0, min(end, t1) - max(start, t0))
        if t0 <= start < t1:
            secs += float(s["boot_allowance_s"])
        usd += secs / 3600.0 * float(s["hourly_usd"])
    return usd


def phase_state(ledger: dict[str, Any], now: float) -> Optional[dict[str, Any]]:
    """The open phase's spend against its stop point and line: state `ok`, `stop` (at or past the stop point) or
    `line` (at or past the line). None when no phase is open."""
    p = open_phase(ledger)
    if p is None:
        return None
    spent = phase_vm_usd(ledger, p, now)
    state = "line" if spent >= p["line_usd"] else "stop" if spent >= p["stop_usd"] else "ok"
    return {"name": p["name"], "state": state, "vm_usd": round(spent, 4), "stop_usd": p["stop_usd"],
            "line_usd": p["line_usd"], "allocation_usd": p["allocation_usd"], "started_at": p["started_at"]}


def phase_reason(ps: Mapping[str, Any]) -> str:
    return (f"spend phase {ps['name']}: ${ps['vm_usd']:.2f} of VM since it opened (stop point ${ps['stop_usd']:.2f}, "
            f"line ${ps['line_usd']:.2f})")


def start_phase(storage: Storage, name: str, *, allocation_usd: float, line_usd: float, margin_usd: float,
                at: float) -> dict[str, Any]:
    """Open phase `name` (a conditional write). Refused while another phase is open, for a name used before, or for a
    margin that leaves no room below the line."""
    if not name or not all(c.isalnum() or c in "-_." for c in name):
        raise PhaseError(f"phase name {name!r}: letters, digits, '-', '_' or '.' only")
    if not 0 <= margin_usd < line_usd:
        raise PhaseError(f"margin ${margin_usd} must be at least 0 and below the line ${line_usd}")
    if allocation_usd > line_usd:
        raise PhaseError(f"allocation ${allocation_usd} is above the line ${line_usd}")
    entry = {"name": name, "allocation_usd": float(allocation_usd), "line_usd": float(line_usd),
             "stop_usd": round(float(line_usd) - float(margin_usd), 4), "margin_usd": float(margin_usd),
             "started_at": float(at), "ended_at": None}

    def edit(ledger):
        live = open_phase(ledger)
        if live is not None:
            raise PhaseError(f"phase {live['name']!r} is still open: close it first")
        if any(p["name"] == name for p in ledger["phases"]):
            raise PhaseError(f"a phase named {name!r} is already in the ledger")
        ledger["phases"].append(dict(entry))
    update_ledger(storage, edit)
    return entry


def end_phase(storage: Storage, name: str, at: float) -> dict[str, Any]:
    def edit(ledger):
        live = open_phase(ledger)
        if live is None or live["name"] != name:
            raise PhaseError(f"phase {name!r} is not the open phase (open: {live['name'] if live else 'none'})")
        live["ended_at"] = float(at)
        live["vm_usd"] = round(phase_vm_usd(ledger, live, at), 4)
        return dict(live)
    return update_ledger(storage, edit)[1]


def raise_phase(storage: Storage, name: str, *, raise_usd: float, reason: str, at: float) -> dict[str, Any]:
    """Raise the open phase's line and stop point by `raise_usd` (a conditional write; the margin between them and
    the allocation stay as they were), with the reason and both old values recorded in its `raises` list. The guard
    reads the ledger at every tick, so a running VM follows the new stop point without being touched (amendment 69).
    Refused for a phase that is not the open one, a raise that is not positive, or no reason."""
    if not raise_usd > 0:
        raise PhaseError(f"raise ${raise_usd}: must be positive")
    if not reason.strip():
        raise PhaseError("a raise needs its reason")

    def edit(ledger):
        live = open_phase(ledger)
        if live is None or live["name"] != name:
            raise PhaseError(f"phase {name!r} is not the open phase (open: {live['name'] if live else 'none'})")
        live.setdefault("raises", []).append({"at": float(at), "raise_usd": float(raise_usd), "reason": reason,
                                              "line_before": live["line_usd"], "stop_before": live["stop_usd"]})
        live["line_usd"] = round(float(live["line_usd"]) + float(raise_usd), 4)
        live["stop_usd"] = round(float(live["stop_usd"]) + float(raise_usd), 4)
        return dict(live)
    return update_ledger(storage, edit)[1]


CHECK_OK, CHECK_STOP, CHECK_WRONG_PHASE = 0, 3, 4


def check(ledger: dict[str, Any], cfg: "SpendConfig", now: float, *, phase: Optional[str] = None) -> tuple[int, str]:
    """Whether the runner may start a step (exit status, reason): CHECK_STOP at the hard limit or at the open phase's
    stop point; CHECK_WRONG_PHASE when the step names a phase that is not the open one."""
    total = estimate(ledger, now)["total_usd"]
    if total >= cfg.hard_limit_usd:
        return CHECK_STOP, f"spend hard limit: ${total:.2f} of ${cfg.hard_limit_usd:.2f}"
    ps = phase_state(ledger, now)
    if phase is not None and (ps is None or ps["name"] != phase):
        return CHECK_WRONG_PHASE, (f"the step belongs to spend phase {phase!r}, but the open phase is "
                                   f"{repr(ps['name']) if ps else 'none'}: open it with `vlar-ops spend phase-open`")
    if ps is not None and ps["state"] != "ok":
        return CHECK_STOP, phase_reason(ps)
    return CHECK_OK, (f"total ${total:.2f}" + (f"; {phase_reason(ps)}" if ps else "; no phase open"))


def load_ledger(storage: Storage) -> dict[str, Any]:
    raw = storage.get_bytes(LEDGER_KEY)
    return _migrate(json.loads(raw)) if raw else new_ledger()


def update_ledger(storage: Storage, edit: Callable[[dict[str, Any]], Any], *, attempts: int = 50,
                  sleep: Callable[[float], None] = time.sleep) -> tuple[dict[str, Any], Any]:
    """Every ledger write (amendment 39): read the ledger with its version, apply `edit` (which changes the ledger in
    place and must do nothing else: it may run more than once), write it back only if nobody wrote in between, else
    re-read and retry. Returns the ledger as written and `edit`'s result."""
    for i in range(attempts):
        raw, version = storage.get_versioned(LEDGER_KEY)
        ledger = _migrate(json.loads(raw)) if raw else new_ledger()
        out = edit(ledger)
        if storage.put_if(LEDGER_KEY, json.dumps(ledger, indent=1, sort_keys=True).encode(), version):
            return ledger, out
        sleep(min(2.0, 0.05 * (i + 1)) * random.uniform(0.5, 1.0))
    raise LedgerConflict(f"the ledger changed under every one of {attempts} attempts")


def boot_info(now: Callable[[], float] = time.time) -> tuple[Optional[str], Optional[float]]:
    """This boot's kernel id and start time (Linux; (None, None) elsewhere). Sessions count VM time from boot
    (amendment 42), so the startup install before the guard starts is included."""
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        up = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None, None
    return boot_id, now() - up


def pending_failed_boots(storage: Storage, ledger: dict[str, Any], cfg: "SpendConfig") -> list[dict[str, Any]]:
    """Sessions for boots whose startup failed before the guard ran (`startup.sh` records them under
    ops/failed-boots/), not yet in the ledger."""
    known = {s["id"] for s in ledger["sessions"]}
    out = []
    for key in storage.list(FAILED_BOOTS_PREFIX):
        raw = storage.get_bytes(key)
        try:
            rec = json.loads(raw) if raw else None
            boot_id, boot_at, failed_at = str(rec["boot_id"]), float(rec["boot_at"]), float(rec["failed_at"])
        except (ValueError, TypeError, KeyError):
            continue
        if boot_id in known:
            continue
        out.append({"id": boot_id, "started_at": boot_at, "last_tick": failed_at,
                    "vm_seconds": max(0.0, failed_at - boot_at), "hourly_usd": cfg.vm_hourly_usd,
                    "boot_allowance_s": cfg.boot_allowance_s, "failed_boot": str(rec.get("reason", ""))[:500]})
        known.add(boot_id)
    return out


def due_alerts(ledger: dict[str, Any], est: dict[str, float], cfg: "SpendConfig", t: float) -> list[tuple[str, float]]:
    """Alerts to send now and record (amendment 43): a limit crossed for the first time, or crossed again after it was
    raised above the limit of the last alert of its kind."""
    due = []
    for kind, limit in (("hard", cfg.hard_limit_usd), ("soft", cfg.soft_limit_usd)):
        if est["total_usd"] < limit:
            continue
        fired = ledger["alerts"][kind]
        last = max((a["limit_usd"] if a.get("limit_usd") is not None else float("-inf")) for a in fired) \
            if fired else None
        if last is None or limit > last:
            fired.append({"at": t, "limit_usd": limit, "total_usd": round(est["total_usd"], 4)})
            due.append((kind, limit))
    return due


class SpendGuard:
    """The daemon's side of the ledger: one session per boot, counted from the boot's start; every write through
    `update_ledger`, notifications only after the write that records them succeeded."""

    def __init__(self, storage: Storage, cfg: SpendConfig, notifier: Notifier,
                 now: Callable[[], float] = time.time, session_id: Optional[str] = None,
                 boot: Optional[Callable[[], tuple[Optional[str], Optional[float]]]] = None):
        self.storage, self.cfg, self.notifier, self.now = storage, cfg, notifier, now
        boot_id, boot_at = (boot or (lambda: boot_info(now)))()
        self.session_id = session_id or boot_id or uuid.uuid4().hex[:12]
        self._boot_at = boot_at
        self._session: Optional[dict[str, Any]] = None
        self._last_scan: float = float("-inf")
        self.phase: Optional[dict[str, Any]] = None

    def load(self) -> dict[str, Any]:
        return load_ledger(self.storage)

    def _update(self, edit):
        return update_ledger(self.storage, edit)

    def start(self) -> str:
        t = self.now()
        started = min(t, self._boot_at) if self._boot_at is not None else t
        failed = pending_failed_boots(self.storage, self.load(), self.cfg)

        def edit(ledger):
            known = {s["id"] for s in ledger["sessions"]}
            ledger["sessions"].extend(f for f in failed if f["id"] not in known)
            sess = next((s for s in ledger["sessions"] if s["id"] == self.session_id), None)
            if sess is None:          # a new boot (a restarted daemon in the same boot continues its session)
                sess = {"id": self.session_id, "started_at": started, "last_tick": t, "vm_seconds": 0.0,
                        "hourly_usd": self.cfg.vm_hourly_usd, "boot_allowance_s": self.cfg.boot_allowance_s}
                ledger["sessions"].append(sess)
            sess.update(last_tick=t, vm_seconds=max(float(sess["vm_seconds"]), t - float(sess["started_at"])))
            return self._judge(ledger, t)
        ledger, (state, due) = self._update(edit)
        self._session = dict(next(s for s in ledger["sessions"] if s["id"] == self.session_id))
        self._notify(due, ledger, t)
        return state

    def tick(self, scan_storage: bool = True) -> str:
        t = self.now()
        nbytes = None
        if scan_storage and t - self._last_scan >= self.cfg.storage_scan_every_s:
            nbytes = self.storage.total_bytes("")
        if self._session is None:
            return self.start()

        def edit(ledger):
            sess = next((s for s in ledger["sessions"] if s["id"] == self.session_id), None)
            if sess is None:          # lost or removed by another writer: put it back, never crash
                print(f"spend guard: session {self.session_id} missing from the ledger; restored", file=sys.stderr)
                sess = dict(self._session)
                ledger["sessions"].append(sess)
            sess.update(last_tick=t, vm_seconds=max(float(sess["vm_seconds"]), t - float(sess["started_at"])))
            if nbytes is not None:
                self._accrue_storage(ledger, t, nbytes)
            return self._judge(ledger, t)
        ledger, (state, due) = self._update(edit)
        self._session = dict(next(s for s in ledger["sessions"] if s["id"] == self.session_id))
        if nbytes is not None:
            self._last_scan = t
        self._notify(due, ledger, t)
        return state

    def _accrue_storage(self, ledger: dict[str, Any], t: float, nbytes: int) -> None:
        st = ledger["storage"]
        if st["last_at"] is not None and st["usd_per_gb_month"]:
            st["accrued_usd"] += (st["last_bytes"] / 1e9 * st["usd_per_gb_month"]
                                  * max(0.0, t - st["last_at"]) / 3600.0 / HOURS_PER_MONTH)
        st.update(last_bytes=nbytes, last_at=t, usd_per_gb_month=self.cfg.storage_usd_per_gb_month)

    def _judge(self, ledger: dict[str, Any], t: float) -> tuple[str, list]:
        """State: `hard` (the hard limit), `line` (the open phase at its line), `phase_stop` (at its stop point),
        else `soft` or `ok`. The open phase's state is kept in `self.phase` for the stop flag's reason."""
        est = estimate(ledger, t)
        due = due_alerts(ledger, est, self.cfg, t)
        self.phase = phase_state(ledger, t)
        if est["total_usd"] >= self.cfg.hard_limit_usd:
            state = "hard"
        elif self.phase is not None and self.phase["state"] == "line":
            state = "line"
        elif self.phase is not None and self.phase["state"] == "stop":
            state = "phase_stop"
        else:
            state = "soft" if est["total_usd"] >= self.cfg.soft_limit_usd else "ok"
        return state, due

    def stop_reason(self, state: str) -> str:
        if state == "hard":
            return "spend hard limit"
        return phase_reason(self.phase) if self.phase is not None else f"spend state {state}"

    def _notify(self, due, ledger: dict[str, Any], t: float) -> None:
        est = estimate(ledger, t)
        for kind, limit in due:
            self.notifier.notify(f"spend_{kind}_limit", {"estimate": est, "limit_usd": limit})


# ---------------------------------------------------------------- CLI-side ledger edits (any machine, any time)

class DiskRecordNotConfirmed(RuntimeError):
    """A disk record written to the ledger is not in the ledger read back afterwards."""


def record_disk(storage: Storage, gb: float, usd_per_gb_month: float, since: float, name: str) -> dict[str, Any]:
    """Append an open disk record (a conditional write, amendment 39) and return it as written."""
    entry = {"name": name, "gb": gb, "usd_per_gb_month": usd_per_gb_month, "since": since, "until": None}
    update_ledger(storage, lambda ledger: ledger["disks"].append(dict(entry)))
    return entry


def confirm_disk(storage: Storage, entry: dict[str, Any]) -> None:
    """The record, exactly as written, must be in a fresh read of the ledger. The boot's `.disk-recorded` marker is
    written only after this (amendment 48). On 2026-11-23 the L4's first record was lost to the pre-amendment-39
    race after `disk-created` had reported it, and the marker then kept every later boot from recording the disk
    again."""
    if not any(d == entry for d in load_ledger(storage)["disks"]):
        raise DiskRecordNotConfirmed(f"disk {entry['name']!r} (since {entry['since']}) is not in the ledger read back "
                                     f"after the write")


def close_disk(storage: Storage, name: str, until: float, since: Optional[float] = None) -> dict[str, Any]:
    """Close the open record of disk `name` (with `since`, the one that started then, to within a second). Two open
    records with one name (a VM recreated under the same name before the old disk's record was closed) need `since`:
    closing both would leave the live disk uncounted."""
    def edit(ledger):
        open_disks = [d for d in ledger["disks"] if d["name"] == name and d.get("until") is None
                      and (since is None or abs(d["since"] - since) < 1.0)]
        if not open_disks:
            raise ValueError(f"no open disk named {name!r}" + ("" if since is None else f" since {since}")
                             + " in the ledger")
        if len(open_disks) > 1:
            raise ValueError(f"{len(open_disks)} open disks named {name!r} (since "
                             f"{', '.join(str(d['since']) for d in open_disks)}): pass --since to pick one")
        open_disks[0]["until"] = until
        return dict(open_disks[0])
    return update_ledger(storage, edit)[1]


def reconcile(storage: Storage, actual_usd: float, at: float, note: str = "") -> dict[str, float]:
    """Set the ledger total to the billing report's figure as of `at` (§12: at least weekly)."""
    def edit(ledger):
        before = estimate(ledger, at)["total_usd"]
        ledger["adjustments"].append({"at": at, "actual_usd": actual_usd, "estimate_before_usd": before,
                                      "delta_usd": actual_usd - before, "note": note})
        return {"estimate_before_usd": before, "delta_usd": actual_usd - before}
    return update_ledger(storage, edit)[1]


def add_charge(storage: Storage, usd: float, at: float, note: str) -> dict[str, float]:
    """Charges the ledger cannot see (image-build VMs, other non-VM items) added to its total until a `reconcile`
    sets the total to the billing report's figure, which includes them."""
    if not note:
        raise ValueError("a charge needs a note saying what it is")
    def edit(ledger):
        before = estimate(ledger, at)["total_usd"]
        ledger["adjustments"].append({"at": at, "kind": "charge", "delta_usd": float(usd), "note": note,
                                      "estimate_before_usd": before})
        return {"estimate_before_usd": before, "delta_usd": float(usd), "total_usd": before + float(usd)}
    return update_ledger(storage, edit)[1]


# ---------------------------------------------------------------- daemon

def run_daemon(guard: SpendGuard, stop: StopFlag, *, in_flight: Callable[[], int],
               release_in_flight: Callable[[], int], final_sync: Callable[[], Any],
               shutdown: Callable[[], Any], sleep: Callable[[float], None] = time.sleep,
               max_ticks: Optional[int] = None) -> str:
    """Tick until the hard limit or the open phase's line, then drain (the hard limit only), sync and shut down.
    Returns the final state. At the open phase's stop point the stop flag goes up (no new claims, no new step) and
    the daemon keeps ticking: the runner then stops the VM itself once the step ends.

    A SIGTERM (Spot preemption runs the shutdown script, which stops the daemon) should be
    turned into SystemExit by the caller; the final tick in `finally` then records the time.
    """
    state = guard.start()
    if state in ("ok", "soft"):
        stop.clear()          # a restart after funds were added (limits raised) or a new phase opened resumes work
    phase_noted = False
    try:
        ticks = 0
        while state not in ("hard", "line"):
            if state == "phase_stop":
                stop.set(guard.stop_reason(state))
                if not phase_noted:
                    guard.notifier.notify("spend_phase_stop", {"phase": guard.phase, "limit_usd": guard.phase["stop_usd"]})
                    phase_noted = True
            if max_ticks is not None and ticks >= max_ticks:
                return state
            sleep(guard.cfg.tick_s)
            state = guard.tick()
            ticks += 1
        stop.set(guard.stop_reason(state))
        if state == "hard":   # at a phase's line nothing more is spent: in-flight jobs are released at once
            deadline = guard.now() + guard.cfg.drain_grace_s
            while in_flight() > 0 and guard.now() < deadline:
                sleep(min(30.0, guard.cfg.tick_s))
                guard.tick(scan_storage=False)
        release_in_flight()
        final_sync()
    finally:
        guard.tick(scan_storage=False)
    shutdown()
    return state
