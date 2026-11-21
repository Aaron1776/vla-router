"""Spend ledger and guard (docs/SPEC.md §12).

Google Cloud budgets only alert; they never stop a VM. The guard runs on the VM, keeps a
ledger in the bucket (so it survives preemption and restarts), and:
  - at the soft limit sends one notification;
  - at the hard limit raises the stop flag (workers take no new jobs), waits for in-flight
    jobs up to a grace period, releases the rest cleanly, syncs results and shuts the VM down.

The estimate counts VM time (plus a per-boot allowance for time billed before the guard
starts), the boot disk for every hour it exists (disks bill while the VM is stopped),
bucket storage, and reconciliation adjustments against the real billing report.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from .storage import Storage

HOURS_PER_MONTH = 730.0
LEDGER_KEY = "ops/ledger.json"


@dataclass(frozen=True)
class SpendConfig:
    soft_limit_usd: float = 40.0
    hard_limit_usd: float = 45.0
    vm_hourly_usd: float = 0.46          # g2-standard-16 Spot with its L4; set from the current price
    boot_allowance_s: float = 180.0      # billed boot time before the guard's first tick
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
            "last_at": None, "usd_per_gb_month": None}, "adjustments": [], "alerts": {}}


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


class SpendGuard:
    def __init__(self, storage: Storage, cfg: SpendConfig, notifier: Notifier,
                 now: Callable[[], float] = time.time, session_id: Optional[str] = None):
        self.storage, self.cfg, self.notifier, self.now = storage, cfg, notifier, now
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._last_tick: Optional[float] = None
        self._last_scan: float = float("-inf")

    # ledger I/O: re-read before every change so CLI edits (reconcile, disk records) are kept
    def load(self) -> dict[str, Any]:
        raw = self.storage.get_bytes(LEDGER_KEY)
        return json.loads(raw) if raw else new_ledger()

    def save(self, ledger: dict[str, Any]) -> None:
        self.storage.put_bytes(LEDGER_KEY, json.dumps(ledger, indent=1, sort_keys=True).encode())

    def start(self) -> str:
        t = self.now()
        ledger = self.load()
        ledger["sessions"].append({"id": self.session_id, "started_at": t, "last_tick": t, "vm_seconds": 0.0,
                                   "hourly_usd": self.cfg.vm_hourly_usd,
                                   "boot_allowance_s": self.cfg.boot_allowance_s})
        self._last_tick = t
        self.save(ledger)
        return self._evaluate(ledger, t)

    def tick(self, scan_storage: bool = True) -> str:
        t = self.now()
        ledger = self.load()
        sess = next(s for s in ledger["sessions"] if s["id"] == self.session_id)
        sess["vm_seconds"] += max(0.0, t - (self._last_tick if self._last_tick is not None else t))
        sess["last_tick"] = t
        self._last_tick = t
        if scan_storage and t - self._last_scan >= self.cfg.storage_scan_every_s:
            self._accrue_storage(ledger, t, self.storage.total_bytes(""))
            self._last_scan = t
        self.save(ledger)
        return self._evaluate(ledger, t)

    def _accrue_storage(self, ledger: dict[str, Any], t: float, nbytes: int) -> None:
        st = ledger["storage"]
        if st["last_at"] is not None and st["usd_per_gb_month"]:
            st["accrued_usd"] += (st["last_bytes"] / 1e9 * st["usd_per_gb_month"]
                                  * max(0.0, t - st["last_at"]) / 3600.0 / HOURS_PER_MONTH)
        st.update(last_bytes=nbytes, last_at=t, usd_per_gb_month=self.cfg.storage_usd_per_gb_month)

    def _evaluate(self, ledger: dict[str, Any], t: float) -> str:
        est = estimate(ledger, t)
        total = est["total_usd"]
        if total >= self.cfg.hard_limit_usd:
            if not ledger["alerts"].get("hard_at"):
                ledger["alerts"]["hard_at"] = t
                self.save(ledger)
                self.notifier.notify("spend_hard_limit", {"estimate": est, "limit_usd": self.cfg.hard_limit_usd})
            return "hard"
        if total >= self.cfg.soft_limit_usd:
            if not ledger["alerts"].get("soft_at"):
                ledger["alerts"]["soft_at"] = t
                self.save(ledger)
                self.notifier.notify("spend_soft_limit", {"estimate": est, "limit_usd": self.cfg.soft_limit_usd})
            return "soft"
        return "ok"


# ---------------------------------------------------------------- CLI-side ledger edits

def record_disk(storage: Storage, gb: float, usd_per_gb_month: float, since: float, name: str) -> None:
    g = SpendGuard(storage, SpendConfig(), PrintNotifier())
    ledger = g.load()
    ledger["disks"].append({"name": name, "gb": gb, "usd_per_gb_month": usd_per_gb_month, "since": since,
                            "until": None})
    g.save(ledger)


def close_disk(storage: Storage, name: str, until: float) -> None:
    g = SpendGuard(storage, SpendConfig(), PrintNotifier())
    ledger = g.load()
    open_disks = [d for d in ledger["disks"] if d["name"] == name and d.get("until") is None]
    if not open_disks:
        raise ValueError(f"no open disk named {name!r} in the ledger")
    for d in open_disks:
        d["until"] = until
    g.save(ledger)


def reconcile(storage: Storage, actual_usd: float, at: float, note: str = "") -> dict[str, float]:
    """Set the ledger total to the billing report's figure as of `at` (§12: at least weekly)."""
    g = SpendGuard(storage, SpendConfig(), PrintNotifier())
    ledger = g.load()
    before = estimate(ledger, at)["total_usd"]
    ledger["adjustments"].append({"at": at, "actual_usd": actual_usd, "estimate_before_usd": before,
                                  "delta_usd": actual_usd - before, "note": note})
    g.save(ledger)
    return {"estimate_before_usd": before, "delta_usd": actual_usd - before}


# ---------------------------------------------------------------- daemon

def run_daemon(guard: SpendGuard, stop: StopFlag, *, in_flight: Callable[[], int],
               release_in_flight: Callable[[], int], final_sync: Callable[[], Any],
               shutdown: Callable[[], Any], sleep: Callable[[float], None] = time.sleep,
               max_ticks: Optional[int] = None) -> str:
    """Tick until the hard limit, then drain, sync and shut down. Returns the final state.

    A SIGTERM (Spot preemption runs the shutdown script, which stops the daemon) should be
    turned into SystemExit by the caller; the final tick in `finally` then records the time.
    """
    state = guard.start()
    if state != "hard":
        stop.clear()          # a restart after funds were added (limits raised) resumes work
    try:
        ticks = 0
        while state != "hard":
            if max_ticks is not None and ticks >= max_ticks:
                return state
            sleep(guard.cfg.tick_s)
            state = guard.tick()
            ticks += 1
        stop.set("spend hard limit")
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
