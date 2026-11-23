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
from typing import Any, Callable, Optional, Protocol

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
            "last_at": None, "usd_per_gb_month": None}, "adjustments": [], "alerts": {"soft": [], "hard": []}}


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
    return ledger


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
        est = estimate(ledger, t)
        due = due_alerts(ledger, est, self.cfg, t)
        state = ("hard" if est["total_usd"] >= self.cfg.hard_limit_usd
                 else "soft" if est["total_usd"] >= self.cfg.soft_limit_usd else "ok")
        return state, due

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
