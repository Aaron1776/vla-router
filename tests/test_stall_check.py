"""The stall alarm's verdict (ops/gcp/stall_check.py; the user, 2026-11-26): what reaches the bucket only, never
"clear"; allowances from the measured gaps."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("stall_check", Path(__file__).parents[1] / "ops/gcp/stall_check.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

M = 60.0
START = 100_000.0


def stages(reruns=(), rr_total=320, rr_opened=START - 3 * 3600, branches=(), br_total=None, br_opened=None):
    return [{"name": "reruns", "total": rr_total, "landed": list(reruns), "opened_at": rr_opened, "gap_s": 20 * M},
            {"name": "branches", "total": br_total, "landed": list(branches), "opened_at": br_opened, "gap_s": 5 * M}]


def v(now, st, status="RUNNING"):
    return sc.stall_verdict(now, status, START, st)


def test_every_verdict_says_what_it_cannot_see_and_never_clear():
    r = v(START + 10 * M, stages(reruns=[START + 5 * M]))
    assert r["not_visible"] == "VM-internal state not visible mid-step (mismatch marker, stop flag, queue counts)"
    assert "clear" not in str(r).lower()


def test_a_stopped_vm_is_the_watchers():
    r = v(START + 99 * M, stages(), status="TERMINATED")
    assert r["stalled"] is False and r["stage"] is None


@pytest.mark.parametrize("quiet_min, stalled", [(34.9, False), (35.1, True)])
def test_boot_window_from_the_start_when_nothing_landed_this_boot(quiet_min, stalled):
    earlier = [START - 50 * M]                       # an earlier boot's results do not count
    r = v(START + quiet_min * M, stages(reruns=earlier))
    assert (r["stage"], r["stalled"]) == ("boot", stalled)


@pytest.mark.parametrize("quiet_min, stalled", [(8.5, False), (19.9, False), (20.1, True)])
def test_reruns_allow_20_min_between_results(quiet_min, stalled):
    t = START + 30 * M
    r = v(t + quiet_min * M, stages(reruns=[START + 20 * M, t]))
    assert (r["stage"], r["stalled"], r["allowance_min"]) == ("reruns", stalled, 20.0)


def test_between_stages_the_finished_stages_gap_applies():
    t = START + 60 * M
    r = v(t + 21 * M, stages(reruns=[t] * 320))
    assert (r["stage"], r["allowance_min"], r["stalled"]) == ("before branches", 20.0, True)
    assert v(t + 19 * M, stages(reruns=[t] * 320))["stalled"] is False


@pytest.mark.parametrize("quiet_min, stalled", [(19.9, False), (20.1, True)])
def test_a_stage_opening_allows_15_min_plus_its_gap_after_its_plan_lands(quiet_min, stalled):
    t = START + 60 * M
    r = v(t + 5 * M + quiet_min * M, stages(reruns=[t] * 320, br_total=3840, br_opened=t + 5 * M))
    assert (r["stage"], r["stalled"]) == ("branches opening", stalled)


@pytest.mark.parametrize("quiet_min, stalled", [(1.3, False), (4.9, False), (5.1, True)])
def test_branches_allow_5_min_between_results(quiet_min, stalled):
    t = START + 60 * M
    b = [t + 20 * M, t + 21 * M]
    r = v(b[-1] + quiet_min * M, stages(reruns=[t] * 320, br_total=3840, br_opened=t + 5 * M, branches=b))
    assert (r["stage"], r["stalled"]) == ("branches", stalled)


def test_finishing_allows_10_min_until_the_vm_stops():
    t = START + 60 * M
    st = stages(reruns=[t] * 320, br_total=2, br_opened=t + 5 * M, branches=[t + 20 * M, t + 21 * M])
    assert (v(t + 30 * M, st)["stage"], v(t + 30 * M, st)["stalled"]) == ("finishing", False)
    assert v(t + 32 * M, st)["stalled"] is True


def test_the_740_first_boot_stop_kept_landing_until_its_in_flight_jobs_finished():
    """740's first boot: the first re-run to land (11:13:30Z, ~16.5 min after the start) failed the comparison; the 7
    in flight kept landing for ~12 min (gaps up to 8.0 min) and the VM stopped itself ~29 min after the start: under
    the re-run allowance throughout, so the alarm stays quiet and the watcher reads the stop (exit 76). The alarm is
    for a RUNNING VM that stops landing results (approximate times)."""
    landed = [START + m * M for m in (16.5, 17.2, 25.2, 25.6, 26.3, 27.1, 28.0, 28.9)]
    assert v(START + 31 * M, stages(reruns=landed))["stalled"] is False
