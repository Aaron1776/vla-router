"""Worker supervisor for `run_phase` (docs/PROPOSALS-2026-11-24.md §0.6, approved by the user 2026-11-24).

w8's silent death was a global out-of-memory kill: each worker caches one simulator per task it meets
(`ProductionEpisodes.sims`, in the common data digest, so it is not changed), and 8 workers plus both servers outgrew
the VM's 64 GB. The fix, without touching that class:

- **Memory limit per worker.** `docker run --memory` (swap equal), sized from the running servers' resident memory so
  that workers × limit + servers × margin + a host reserve fit in the VM. An overrun kills that worker only.
- **`--max-jobs` recycling.** A worker exits with RECYCLE_EXIT after N jobs, so a process never holds more than N
  tasks' simulators; the supervisor starts it again while the phase has work. Recycling is not a restart.
- **Restarts.** A worker that died (any other non-zero status; 137 = killed, e.g. out of memory) has the jobs it
  still holds counted as failed attempts (`JobQueue.reap`), then is started again, at most `max_restarts` times per
  worker per phase. A worker that is alive but holds an expired lease (frozen: its leases are renewed only on
  progress, vlar_ops.queue) is stopped, reaped and restarted the same way. Exit 3 (refused job) is never restarted.
  Nothing is started while the stop flag is up (spend guard).
- Every event goes to stdout (the step log) and to `history.jsonl`; peak memory per worker is reported at the end.

`Units` talks to systemd and docker on the VM; tests pass a fake.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol

from .queue import JobQueue
from .vm_worker import RECYCLE_EXIT, REFUSED_EXIT

GiB = 1024 ** 3
MiB = 1024 ** 2
SERVER_CONTAINERS = ("vlar-xvla", "vlar-pi05")


class SizingError(RuntimeError):
    """The workers cannot get a useful memory limit beside the servers."""


@dataclass(frozen=True)
class Sizing:
    mem_total: int
    servers: int
    workers: int
    limit: int          # bytes per worker container
    max_jobs: int
    rule: str = "share"         # "ram": amendment 64; "share": the rule before it (the pilot's boots)
    servers_peak: int = 0       # the servers' peak the RAM rule planned with (bytes)

    def docker_args(self) -> str:
        m = f"{self.limit // MiB}m"
        return f"--memory={m} --memory-swap={m}"


DEFAULT_WORKER_LIMIT = 22 * 256 * MiB     # 5.5 GiB: 1.30 x the largest measured worker peak, 4.54 GB (amendment 64)


def size_workers_ram(mem_total: int, servers: int, workers: int, *, limit: int = DEFAULT_WORKER_LIMIT,
                     servers_peak: Optional[int] = None, server_margin: float = 1.25,
                     worker_base: int = 2 * GiB, per_task: int = 768 * MiB, max_jobs: Optional[int] = None) -> Sizing:
    """SPEC §17 amendment 64 (amendment 62 (4)'s RAM rule): N workers at `limit` each run only if N x limit plus the
    servers' peak resident memory fits inside the VM's RAM (MemTotal). At boot the peak is taken as the larger of the
    servers' resident memory now x `server_margin` and `servers_peak`, the last measured peak, if given. N is the
    largest count up to `workers` that fits; whatever RAM is left is the host's. At the pilot's numbers (MemTotal
    67.42 GB, servers 18.96 GB at start, 20.45 GB peak) that is 7 workers at 5.5 GiB."""
    if workers < 1:
        raise SizingError("no workers")
    if limit < 256 * MiB:
        raise SizingError(f"a worker limit of {limit} bytes is not a limit")
    peak = int(max(servers * server_margin, servers_peak or 0))
    n = min(workers, int((mem_total - peak) // limit))
    if n < 1:
        raise SizingError(f"not one worker at {limit / GiB:.2f} GiB fits beside {peak / GiB:.1f} GiB of servers' peak "
                          f"in {mem_total / GiB:.1f} GiB of RAM")
    if max_jobs is None:
        max_jobs = max(1, int((limit - worker_base) // per_task))
    return Sizing(mem_total, servers, n, int(limit), int(max_jobs), rule="ram", servers_peak=peak)


def size_workers(mem_total: int, servers: int, workers: int, *, server_margin: float = 1.25,
                 host_reserve: int = 4 * GiB, usable: float = 0.95, min_limit: int = 3 * GiB,
                 worker_base: int = 2 * GiB, per_task: int = 768 * MiB, max_jobs: Optional[int] = None,
                 fit: bool = False) -> Sizing:
    """Equal limits for `workers` containers: (usable × RAM − servers × margin − host reserve) / workers, rounded down
    to 256 MiB. `max_jobs` defaults to the number of tasks' simulators that fit in the limit beside a worker's base
    (*est.* from the w8 evidence: 8-9 GB at up to 10 tasks; the pilot measures it). With `fit`, as many workers (at
    most `workers`) as get `min_limit` each, instead of refusing (the pilot's resume, the user 2026-11-24: a limit at
    least 25% above the measured peak). Superseded at boot by `size_workers_ram` (amendment 64); kept for the pilot's
    sizing and its tests."""
    if workers < 1:
        raise SizingError("no workers")
    avail = mem_total * usable - servers * server_margin - host_reserve
    if fit:
        workers = min(workers, int(avail // min_limit))
        if workers < 1:
            raise SizingError(f"not one worker gets {min_limit / GiB:.2f} GiB beside {servers / GiB:.1f} GiB of "
                              f"servers in {mem_total / GiB:.1f} GiB")
    limit = int(avail // workers) // (256 * MiB) * (256 * MiB)
    if limit < min_limit:
        raise SizingError(f"{workers} workers get {limit / GiB:.2f} GiB each beside {servers / GiB:.1f} GiB of "
                          f"servers in {mem_total / GiB:.1f} GiB (minimum {min_limit / GiB:.1f}): fewer workers needed")
    if max_jobs is None:
        max_jobs = max(1, int((limit - worker_base) // per_task))
    return Sizing(mem_total, servers, workers, limit, int(max_jobs))


def parse_bytes(text: str) -> int:
    """docker stats sizes: '3.2GiB', '512MiB', '1.5GB', '800kB', '0B'."""
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?i?B)\s*", text, re.I)
    if not m:
        raise ValueError(f"not a size: {text!r}")
    unit = m.group(2).upper()
    base = 1024 if "I" in unit else 1000
    power = {"B": 0, "K": 1, "M": 2, "G": 3, "T": 4}[unit[0]]
    return int(float(m.group(1)) * base ** power)


class Units(Protocol):
    def active(self, i: int) -> bool: ...
    def exit_status(self, i: int) -> Optional[int]: ...
    def start(self, i: int) -> None: ...
    def stop(self, i: int) -> None: ...
    def journal_tail(self, i: int) -> str: ...
    def memory(self) -> dict[str, int]: ...


class SystemdUnits:
    """vlar-worker@<i> units (ops/vm/vlar-worker@.service) and docker on the VM."""

    def _run(self, *cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def active(self, i: int) -> bool:
        return self._run("systemctl", "is-active", "--quiet", f"vlar-worker@{i}").returncode == 0

    def exit_status(self, i: int) -> Optional[int]:
        out = self._run("systemctl", "show", "-p", "ExecMainStatus", "--value", f"vlar-worker@{i}").stdout.strip()
        return int(out) if out.lstrip("-").isdigit() else None

    def start(self, i: int) -> None:
        self._run("systemctl", "reset-failed", f"vlar-worker@{i}")
        self._run("systemctl", "start", f"vlar-worker@{i}")

    def stop(self, i: int) -> None:
        self._run("systemctl", "stop", f"vlar-worker@{i}")          # ExecStop: docker stop -t 60
        if self.active(i):
            self._run("systemctl", "kill", "-s", "KILL", f"vlar-worker@{i}")
            self._run("docker", "kill", f"vlar-worker-{i}")

    def journal_tail(self, i: int) -> str:
        return self._run("journalctl", "-u", f"vlar-worker@{i}", "-n", "30", "--no-pager").stdout

    def memory(self) -> dict[str, int]:
        out = self._run("docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}").stdout
        mem = {}
        for line in out.splitlines():
            name, _, usage = line.partition("\t")
            try:
                mem[name.strip()] = parse_bytes(usage.split("/")[0])
            except ValueError:
                continue
        return mem


def mem_total() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise SizingError("no MemTotal in /proc/meminfo")


def write_worker_env(path: Path, worker_args: str, sizing: Sizing) -> None:
    path.write_text(f"VLAR_WORKER_ARGS={worker_args} --max-jobs {sizing.max_jobs}\n"
                    f"VLAR_WORKER_DOCKER_ARGS={sizing.docker_args()}\n")


class Supervisor:
    def __init__(self, queue: JobQueue, units: Units, *, phase: str, workers: int, stop_flag: Path,
                 history: Optional[Path] = None, step: str = "", max_restarts: int = 3, poll_s: float = 15.0,
                 status_every_s: float = 120.0, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, out: Callable[[str], None] = print):
        self.q, self.units, self.phase, self.n = queue, units, phase, workers
        self.stop_flag, self.history, self.step = stop_flag, history, step or phase
        self.max_restarts, self.poll_s, self.status_every_s = max_restarts, poll_s, status_every_s
        self.clock, self.sleep, self.out = clock, sleep, out
        self.restarts: Counter = Counter()
        self.recycles: Counter = Counter()
        self.refused: set[int] = set()
        self.gave_up: set[int] = set()
        self.idle_starts: Counter = Counter()
        self.last_held: dict[str, set[str]] = {}
        self._started = False
        self.peak: dict[str, int] = {}
        self.events: list[dict] = []

    def _event(self, i: int, action: str, status: Optional[int], detail: str = "", *, quiet: bool = False) -> None:
        """Every event goes to the summary; all but `quiet` ones (recycles) also to the step log and history."""
        at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.clock()))
        rec = {"step": self.step, "outcome": f"worker-{action}", "exit": -1 if status is None else int(status),
               "seconds": 0, "at": at, "at_s": round(self.clock(), 3), "worker": f"w{i}", "phase": self.phase,
               "detail": detail}
        self.events.append(rec)
        if quiet:
            return
        self.out(f"{at} supervisor {self.phase}: w{i} {action} (exit {status}){': ' + detail if detail else ''}")
        if self.history is not None:
            with open(self.history, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def _sample_memory(self) -> None:
        try:
            for name, b in self.units.memory().items():
                self.peak[name] = max(b, self.peak.get(name, 0))
        except Exception as e:                       # telemetry only
            self.out(f"supervisor: memory sample failed: {e}")

    def _crashed(self, i: int, status: Optional[int], why: str, work: int) -> None:
        if self.restarts[i] >= self.max_restarts:
            if i not in self.gave_up:
                self.gave_up.add(i)
                self._event(i, "not-restarted", status, f"{why}; {self.max_restarts} restarts used")
            return
        if work <= 0:
            return
        self.restarts[i] += 1
        tail = self.units.journal_tail(i).strip().splitlines()[-5:]
        self._event(i, "restart", status, f"{why}; restart {self.restarts[i]} of {self.max_restarts}; "
                                          f"journal: {' | '.join(tail)}")
        self._start(i)

    def _start(self, i: int) -> None:
        self.units.start(i)
        self._started = True

    def poll(self) -> bool:
        """One pass over the workers. Returns True while any worker is running or was just started (it may have
        exited already: the next pass looks at it)."""
        self._started = False
        now, stop = self.clock(), self.stop_flag.exists()
        c = self.q.counts(self.phase)
        work = c.get("pending", 0) + c.get("leased", 0)
        for i in range(1, self.n + 1):
            wid = f"w{i}"
            if self.units.active(i):
                held = self.q.leases(wid)
                expired = [j.job_id for j, exp in held if exp < now]
                # A progressing worker never loses a lease, so a job it held at the last pass that another worker
                # now holds means it stalled and its expired lease was claimed before this pass saw it.
                lost = [jid for jid in self.last_held.get(wid, set()) - {j.job_id for j, _ in held}
                        if (o := self.q.owner(jid))[0] == "leased" and o[1] != wid]
                self.last_held[wid] = {j.job_id for j, _ in held}
                if not expired and not lost:
                    continue
                self.units.stop(i)
                self.last_held.pop(wid, None)
                reaped = self.q.reap(wid, f"frozen: lease expired on {expired}") if expired else []
                self._event(i, "stopped-frozen", None, f"expired {expired}, lost {lost}, reaped {reaped}")
                if not stop:
                    self._crashed(i, None, "frozen", work)
                continue
            if i in self.refused or i in self.gave_up:
                continue
            status = self.units.exit_status(i)
            held = self.q.leases(wid)
            if held:
                reaped = self.q.reap(wid, f"exited with status {status}")
                self._event(i, "reaped", status, f"{reaped}")
            if status == REFUSED_EXIT:
                self.refused.add(i)
                self._event(i, "refused", status, "never restarted")
                continue
            if stop:
                continue
            if status == RECYCLE_EXIT:
                if work > 0:                       # it took jobs, so recycling always ends
                    self.recycles[i] += 1
                    self._event(i, "recycled", status, f"recycle {self.recycles[i]}", quiet=True)
                    self._start(i)
                continue
            if status == 0:                        # found nothing to claim; jobs may have come back since
                if c.get("pending", 0) > 0 and self.idle_starts[i] < 10:
                    self.idle_starts[i] += 1
                    self._start(i)
                continue
            self._crashed(i, status, f"died with status {status}", work)
        self._sample_memory()
        return self._started or any(self.units.active(i) for i in range(1, self.n + 1))

    def run(self) -> dict:
        for i in range(1, self.n + 1):
            self.units.start(i)
        last_status = self.clock()
        while self.poll():
            if self.clock() - last_status >= self.status_every_s:
                last_status = self.clock()
                self.out(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(last_status))} {self.phase}: "
                         f"{json.dumps(self.q.counts(self.phase))}")
            self.sleep(self.poll_s)
        summary = {"phase": self.phase, "restarts": dict(self.restarts), "recycles": dict(self.recycles),
                   "refused": sorted(self.refused), "gave_up": sorted(self.gave_up),
                   "peak_memory_bytes": self.peak, "counts": self.q.counts(self.phase), "events": self.events}
        self.out(json.dumps(summary))
        return summary


def supervise(ops_cfg: Mapping, *, phase: str, workers: int, worker_args: str, worker_env: Path,
              history: Optional[Path], step: str, max_jobs: Optional[int] = None, units: Optional[Units] = None,
              total: Optional[int] = None, min_limit: Optional[int] = None, limit: Optional[int] = None,
              servers_peak: Optional[int] = None, **kw) -> dict:
    """Size the workers from the running servers, write worker.env, start and supervise them. By amendment 64's RAM
    rule (`size_workers_ram`: `limit` bytes each, default 5.5 GiB; `servers_peak` the last measured peak); with
    `min_limit` (bytes), the pilot's rule instead: as many workers as get at least that much each (at most
    `workers`). The summary's `ram_rule` says whether the servers' measured peak kept the rule."""
    from .cli import _queue
    units = units or SystemdUnits()
    mem = units.memory()
    missing = [n for n in SERVER_CONTAINERS if n not in mem]
    if missing:
        raise SizingError(f"server containers not running: {missing}")
    total = total if total is not None else mem_total()
    servers = sum(mem[n] for n in SERVER_CONTAINERS)
    if min_limit:
        if limit is not None or servers_peak is not None:
            raise SizingError("--min-limit-gib is the pilot's rule; --limit-gib and --servers-peak-gb are the RAM rule's")
        sizing = size_workers(total, servers, workers, max_jobs=max_jobs, min_limit=int(min_limit), fit=True)
    else:
        sizing = size_workers_ram(total, servers, workers, limit=int(limit or DEFAULT_WORKER_LIMIT),
                                  servers_peak=servers_peak, max_jobs=max_jobs)
    write_worker_env(worker_env, worker_args, sizing)
    print(json.dumps({"sizing": asdict(sizing), "docker_args": sizing.docker_args()}))
    q = _queue(dict(ops_cfg))
    try:
        sup = Supervisor(q, units, phase=phase, workers=sizing.workers,
                         stop_flag=Path(ops_cfg["local_root"]) / "STOP",
                         history=history, step=step, **kw)
        out = {"sizing": asdict(sizing), **sup.run()}
        measured = sum(v for k, v in (out.get("peak_memory_bytes") or {}).items() if k in SERVER_CONTAINERS)
        if sizing.rule == "ram":
            out["ram_rule"] = {"workers": sizing.workers, "limit": sizing.limit, "mem_total": sizing.mem_total,
                               "servers_peak_planned": sizing.servers_peak, "servers_peak_measured": measured,
                               "fits_at_measured_peak": sizing.workers * sizing.limit + measured <= sizing.mem_total}
        return out
    finally:
        q.close()
