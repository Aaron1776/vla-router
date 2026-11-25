"""Per-phase stop lines (docs/PROPOSALS-2026-11-24.md §0.1 (d)): the phase's VM window, opening and closing, the
guard's stop point and line, the runner's check, and the CLI's exit statuses."""

import pytest

from vlar_ops import spend
from vlar_ops.storage import LocalStorage

HOUR = 3600.0
CFG = spend.SpendConfig(soft_limit_usd=40, hard_limit_usd=45, vm_hourly_usd=0.5, boot_allowance_s=0,
                        storage_usd_per_gb_month=0.02, tick_s=60, storage_scan_every_s=1e12, drain_grace_s=600)


class Clock:
    def __init__(self, t=1_700_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class Notes:
    def __init__(self):
        self.events = []

    def notify(self, event, payload):
        self.events.append(event)


def session(sid, start, secs, allowance=0.0, hourly=0.5):
    return {"id": sid, "started_at": start, "last_tick": start + secs, "vm_seconds": secs, "hourly_usd": hourly,
            "boot_allowance_s": allowance}


def test_a_phase_counts_only_vm_time_inside_its_window():
    t0 = 1_000_000.0
    led = spend.new_ledger()
    led["sessions"] = [session("before", t0 - 3 * HOUR, HOUR, allowance=180),     # ended before: nothing
                       session("across", t0 - HOUR, 3 * HOUR, allowance=180),     # 2 h inside, no allowance
                       session("inside", t0 + 5 * HOUR, HOUR, allowance=180),     # 1 h + its allowance
                       session("after", t0 + 20 * HOUR, HOUR, allowance=180)]     # after the phase closed
    phase = {"started_at": t0, "ended_at": t0 + 10 * HOUR}
    assert spend.phase_vm_usd(led, phase, now=t0 + 30 * HOUR) == pytest.approx((2 + 1 + 180 / HOUR) * 0.5)
    open_ = {"started_at": t0, "ended_at": None}
    assert spend.phase_vm_usd(led, open_, now=t0 + 30 * HOUR) == pytest.approx((2 + 1 + 1 + 2 * 180 / HOUR) * 0.5)


def test_a_failed_boot_from_before_the_phase_never_counts_against_it(tmp_path):
    st, clock = LocalStorage(tmp_path), Clock()
    spend.start_phase(st, "pilot", allocation_usd=1.6, line_usd=2.5, margin_usd=0.1, at=clock.t)
    led = spend.load_ledger(st)
    led["sessions"].append(session("failed-earlier", clock.t - 2 * HOUR, HOUR, allowance=180))
    assert spend.phase_state(led, clock.t + HOUR)["vm_usd"] == 0.0


def test_opening_and_closing_phases(tmp_path):
    st, clock = LocalStorage(tmp_path), Clock()
    p = spend.start_phase(st, "pilot", allocation_usd=1.6, line_usd=2.5, margin_usd=0.16, at=clock.t)
    assert p["stop_usd"] == pytest.approx(2.34) and p["ended_at"] is None
    with pytest.raises(spend.PhaseError, match="still open"):
        spend.start_phase(st, "phase1-screen", allocation_usd=0.98, line_usd=1.06, margin_usd=0.1, at=clock.t)
    with pytest.raises(spend.PhaseError, match="not the open phase"):
        spend.end_phase(st, "phase1-screen", clock.t)
    closed = spend.end_phase(st, "pilot", clock.t + 10)
    assert closed["ended_at"] == clock.t + 10 and spend.open_phase(spend.load_ledger(st)) is None
    with pytest.raises(spend.PhaseError, match="already in the ledger"):
        spend.start_phase(st, "pilot", allocation_usd=1.6, line_usd=2.5, margin_usd=0.1, at=clock.t + 20)
    for bad in ({"margin_usd": 2.5}, {"margin_usd": -0.1}, {"allocation_usd": 3.0}):
        kw = {"allocation_usd": 1.0, "line_usd": 2.5, "margin_usd": 0.1, **bad}
        with pytest.raises(spend.PhaseError):
            spend.start_phase(st, "x", at=clock.t, **kw)
    with pytest.raises(spend.PhaseError, match="letters"):
        spend.start_phase(st, "a b", allocation_usd=1.0, line_usd=2.5, margin_usd=0.1, at=clock.t)


def test_old_ledgers_gain_an_empty_phase_list(tmp_path):
    st = LocalStorage(tmp_path)
    led = spend.new_ledger()
    del led["phases"]
    st.put_bytes(spend.LEDGER_KEY, __import__("json").dumps(led).encode())
    assert spend.load_ledger(st)["phases"] == [] and spend.phase_state(spend.load_ledger(st), 0.0) is None


def test_the_default_margin_covers_the_drain_and_the_wind_down():
    assert spend.default_phase_margin_usd(CFG) == pytest.approx((600 + spend.PHASE_WIND_DOWN_S) / HOUR * 0.5)


def run_guard(tmp_path, clock, *, in_flight, sleep_s=HOUR, max_ticks=None, start_usd=0.0):
    st = LocalStorage(tmp_path / "store")
    if start_usd:
        spend.reconcile(st, start_usd, clock.t, "billing")
    stop = spend.StopFlag(tmp_path / "STOP")
    log = []
    flags = []

    def sleep(s):
        flags.append(stop.is_set())
        clock.t += sleep_s

    def inflight():
        log.append("in_flight?")
        return in_flight

    g = spend.SpendGuard(st, CFG, Notes(), now=clock, boot=lambda: (None, None))
    state = spend.run_daemon(g, stop, in_flight=inflight, release_in_flight=lambda: log.append("release"),
                             final_sync=lambda: log.append("sync"), shutdown=lambda: log.append("shutdown"),
                             sleep=sleep, max_ticks=max_ticks)
    return st, stop, g, state, log, flags


def test_the_stop_point_raises_the_flag_and_the_line_shuts_down_without_a_drain(tmp_path):
    """$0.50 an hour, stop point $1.00, line $2.00: the flag goes up after the second hour, the VM is shut down after
    the fourth; at the line nothing waits for in-flight jobs (the cap is the user's)."""
    clock = Clock()
    st = LocalStorage(tmp_path / "store")
    spend.start_phase(st, "pilot", allocation_usd=1.0, line_usd=2.0, margin_usd=1.0, at=clock.t - 1)
    _, stop, g, state, log, flags = run_guard(tmp_path, clock, in_flight=3)
    assert state == "line"
    assert flags == [False, False, True, True]          # stop flag before each tick's sleep
    assert log == ["release", "sync", "shutdown"]          # in_flight never asked: no drain at a line
    assert "spend phase pilot" in stop.path.read_text() and "line $2.00" in stop.path.read_text()
    assert g.notifier.events == ["spend_phase_stop"]


def test_the_hard_limit_still_drains_with_a_phase_open(tmp_path):
    clock = Clock()
    st = LocalStorage(tmp_path / "store")
    spend.start_phase(st, "pilot", allocation_usd=1.0, line_usd=20.0, margin_usd=1.0, at=clock.t - 1)
    _, stop, g, state, log, _ = run_guard(tmp_path, clock, in_flight=0, start_usd=44.6)
    assert state == "hard" and log[0] == "in_flight?" and log[-3:] == ["release", "sync", "shutdown"]
    assert stop.path.read_text().startswith("spend hard limit")


def test_a_restart_past_the_stop_point_keeps_the_flag_and_a_new_phase_clears_it(tmp_path):
    clock = Clock()
    st = LocalStorage(tmp_path / "store")
    spend.start_phase(st, "pilot", allocation_usd=1.0, line_usd=2.0, margin_usd=1.0, at=clock.t - 1)
    led_sessions = [session("earlier-boot", clock.t, 2.2 * HOUR)]       # $1.10 in the phase: past its stop point

    def seed(ledger):
        ledger["sessions"] += led_sessions
    spend.update_ledger(st, seed)
    clock.t += 3 * HOUR
    stop = spend.StopFlag(tmp_path / "STOP")
    g = spend.SpendGuard(st, CFG, Notes(), now=clock, boot=lambda: (None, None))
    state = spend.run_daemon(g, stop, in_flight=lambda: 0, release_in_flight=lambda: 0, final_sync=lambda: None,
                             shutdown=lambda: None, sleep=lambda s: None, max_ticks=0)
    assert state == "phase_stop" and stop.is_set()
    spend.end_phase(st, "pilot", clock.t)
    spend.start_phase(st, "phase1-screen", allocation_usd=0.98, line_usd=1.06, margin_usd=0.2, at=clock.t)
    g2 = spend.SpendGuard(st, CFG, Notes(), now=clock, boot=lambda: (None, None))
    state = spend.run_daemon(g2, stop, in_flight=lambda: 0, release_in_flight=lambda: 0, final_sync=lambda: None,
                             shutdown=lambda: None, sleep=lambda s: None, max_ticks=0)
    assert state == "ok" and not stop.is_set()


def test_check_says_whether_a_step_may_start(tmp_path):
    st, clock = LocalStorage(tmp_path), Clock()
    led = spend.load_ledger(st)
    assert spend.check(led, CFG, clock.t)[0] == spend.CHECK_OK
    code, why = spend.check(led, CFG, clock.t, phase="pilot")
    assert code == spend.CHECK_WRONG_PHASE and "open phase is none" in why
    spend.start_phase(st, "pilot", allocation_usd=1.0, line_usd=2.0, margin_usd=0.5, at=clock.t)
    led = spend.load_ledger(st)
    assert spend.check(led, CFG, clock.t, phase="pilot")[0] == spend.CHECK_OK
    assert spend.check(led, CFG, clock.t)[0] == spend.CHECK_OK                    # a step naming no phase
    assert spend.check(led, CFG, clock.t, phase="phase1-screen")[0] == spend.CHECK_WRONG_PHASE
    led["sessions"].append(session("b", clock.t, 3 * HOUR))                      # $1.50: at the stop point
    code, why = spend.check(led, CFG, clock.t + 3 * HOUR, phase="pilot")
    assert code == spend.CHECK_STOP and "stop point $1.50" in why
    assert spend.check(led, CFG, clock.t + 3 * HOUR)[0] == spend.CHECK_STOP     # nothing starts past the stop point
    spend.reconcile(st, 45.0, clock.t, "over")
    assert spend.check(spend.load_ledger(st), CFG, clock.t)[0] == spend.CHECK_STOP


def test_the_cli_exit_status_is_the_checks(tmp_path, capsys):
    from vlar_ops import cli
    store = tmp_path / "store"
    cfg = tmp_path / "ops.yaml"
    cfg.write_text(f"storage_url: {store}\nlocal_root: {tmp_path / 'local'}\nspend:\n  vm_hourly_usd: 0.472\n"
                   f"  drain_grace_s: 900\n")
    with pytest.raises(SystemExit) as e:
        cli.main(["spend", "check", "--phase", "pilot", "--ops-config", str(cfg)])
    assert e.value.code == spend.CHECK_WRONG_PHASE
    cli.main(["spend", "phase-open", "--phase", "pilot", "--allocation-usd", "1.6", "--line-usd", "2.5",
              "--ops-config", str(cfg)])
    p = spend.open_phase(spend.load_ledger(LocalStorage(store)))
    assert p["stop_usd"] == pytest.approx(2.5 - (900 + 300) / HOUR * 0.472, abs=1e-4)
    with pytest.raises(SystemExit) as e:
        cli.main(["spend", "check", "--phase", "pilot", "--ops-config", str(cfg)])
    assert e.value.code == 0
    with pytest.raises(SystemExit, match="still open"):
        cli.main(["spend", "phase-open", "--phase", "x", "--allocation-usd", "1", "--line-usd", "2",
                  "--ops-config", str(cfg)])
    cli.main(["spend", "phase-close", "--phase", "pilot", "--ops-config", str(cfg)])
    assert spend.open_phase(spend.load_ledger(LocalStorage(store))) is None


def test_raising_the_open_phase_moves_line_and_stop_and_the_guard_follows(tmp_path):
    """Amendment 69: a raise is a conditional ledger write; phase_state (what the guard reads at every tick) uses it."""
    st, clock = LocalStorage(tmp_path), Clock()
    spend.start_phase(st, "screen", allocation_usd=3.3, line_usd=3.56, margin_usd=0.23, at=clock.t)
    spend.update_ledger(st, lambda led: led["sessions"].append(session("s1", clock.t, 7.0 * HOUR)))   # $3.50: past the stop
    assert spend.phase_state(spend.load_ledger(st), clock.t + 7 * HOUR)["state"] == "stop"
    p = spend.raise_phase(st, "screen", raise_usd=1.33, reason="projected overrun", at=clock.t + 7 * HOUR)
    assert p["line_usd"] == pytest.approx(4.89) and p["stop_usd"] == pytest.approx(4.66)
    assert p["allocation_usd"] == 3.3 and p["raises"][0]["stop_before"] == pytest.approx(3.33)
    assert spend.phase_state(spend.load_ledger(st), clock.t + 7 * HOUR)["state"] == "ok"
    with pytest.raises(spend.PhaseError, match="not the open phase"):
        spend.raise_phase(st, "other", raise_usd=1.0, reason="x", at=clock.t)
    with pytest.raises(spend.PhaseError, match="positive"):
        spend.raise_phase(st, "screen", raise_usd=0, reason="x", at=clock.t)
    with pytest.raises(spend.PhaseError, match="reason"):
        spend.raise_phase(st, "screen", raise_usd=0.5, reason=" ", at=clock.t)
