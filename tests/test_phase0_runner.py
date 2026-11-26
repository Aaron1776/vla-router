"""ops/vm/phase0/runner.sh (the L4 VM's Phase 0 step runner) against fake gcloud, systemctl, vlar-ops and timeout:
steps run in name order from the bucket's queue; a step can queue another; whenever the runner runs out of work,
a step fails, a gate fails or a step needs the user, it stops the GPU services, syncs, writes the reason and the
spend total to STATE.json in the bucket and stops the VM; a step that needs the user stays queued; the spend guard's
STOP flag starts no step; a stop from outside records the interrupted step without losing it."""
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "ops" / "vm" / "phase0" / "runner.sh"
pytestmark = pytest.mark.skipif(os.environ.get("VLAR_IN_IMAGE") == "1" and not RUNNER.exists(),
                                reason="repo-only files (ops/), not in the image")

FAKE_GCLOUD = r"""#!/bin/bash
# gs://<path> -> $FAKE/bucket/<path>
echo "gcloud $*" >> "$FAKE/log"
args=(); del=0; for a in "$@"; do
  case "$a" in -q|-r) ;; --delete-unmatched-destination-objects) del=1 ;; *) args+=("$a") ;; esac; done
path() { echo "$FAKE/bucket/${1#gs://}"; }
# Like Cloud Storage, a "folder" exists only while it holds an object: an empty one is neither listed nor readable.
has_files() { [[ -d "$1" ]] && [[ -n "$(find "$1" -type f | head -n 1)" ]]; }
[[ -n "${FAKE_LS_FAIL:-}" && "${args[1]}" == ls ]] && exit 1
case "${args[0]} ${args[1]}" in
  "storage ls") p="$(path "${args[2]}")"; has_files "$p" || exit 1
    for e in "$p"/*; do
      if [[ -f "$e" ]]; then echo "${args[2]%/}/$(basename "$e")"
      elif has_files "$e"; then echo "${args[2]%/}/$(basename "$e")/"; fi
    done ;;
  "storage cp") src="${args[2]}"; dst="${args[3]}"
    [[ "$src" == gs://* ]] && src="$(path "$src")"
    [[ "$dst" == gs://* ]] && dst="$(path "$dst")"
    mkdir -p "$(dirname "$dst")"; if [[ "$src" == - ]]; then cat > "$dst"; else cp "$src" "$dst"; fi ;;
  "storage rsync") src="${args[2]}"; dst="${args[3]}"
    [[ "$src" == gs://* ]] && src="$(path "$src")"
    [[ "$dst" == gs://* ]] && dst="$(path "$dst")"
    if [[ "${args[2]}" == gs://* ]]; then has_files "$src" || exit 1; else [[ -d "$src" ]] || exit 1; fi
    mkdir -p "$dst"; cp -R "$src/." "$dst/"
    if (( del )); then for e in "$dst"/*; do [[ -e "$src/$(basename "$e")" ]] || rm -rf "$e"; done; fi ;;
  "storage rm") rm -f "$(path "${args[2]}")" ;;
  "storage cat") cat "$(path "${args[2]}")" ;;
  *) echo "fake gcloud: unhandled $*" >&2; exit 9 ;;
esac
"""
FAKE_VLAR_OPS = r"""#!/bin/bash
echo "vlar-ops $*" >> "$FAKE/log"
case "$1 $2" in
  "spend status") n=$(cat "$FAKE/ticks" 2>/dev/null || echo 0); echo $((n + 1)) > "$FAKE/ticks"
    echo "{\"total_usd\": $(( n + 1 )).5, \"vm_usd\": 1.0, \"disk_usd\": 0.1, \"storage_usd\": 0.0, \"vm_hours\": 2.0, \"hard_limit_usd\": 45}" ;;
  "queue sync") echo "uploaded 0 results" ;;
  "spend check") [[ -f "$FAKE/check_rc" ]] || { echo "total \$1.00; no phase open"; exit 0; }
    echo "noise"; cat "$FAKE/check_msg"; exit "$(cat "$FAKE/check_rc")" ;;
esac
"""
FAKE_SYSTEMCTL = '#!/bin/bash\necho "systemctl $*" >> "$FAKE/log"\n'
# GNU timeout's interface, without the cap (the runner's 124/137 handling is tested by a step exiting 124).
FAKE_TIMEOUT = '#!/bin/bash\nwhile [[ "$1" == --* ]]; do shift; done\nshift\nexec "$@"\n'
# The kernel log for the OOM check: $FAKE/kernel.log for `journalctl -k`, nothing otherwise (never the host's).
FAKE_JOURNALCTL = '#!/bin/bash\n[[ " $* " == *" -k "* && -f "$FAKE/kernel.log" ]] && cat "$FAKE/kernel.log"\nexit 0\n'


def setup(tmp_path, steps, stop_flag=False):
    fake, bin_, local = tmp_path / "fake", tmp_path / "bin", tmp_path / "local"
    queue = fake / "bucket" / "b" / "vla-router" / "phase0" / "queue"
    queue.mkdir(parents=True)
    (queue.parent / "src").mkdir()                       # as on the VM: startup.sh installs from phase0/src/CURRENT
    (queue.parent / "src" / "CURRENT").write_text("0" * 40 + "\n")
    bin_.mkdir()
    for name, text in [("gcloud", FAKE_GCLOUD), ("vlar-ops", FAKE_VLAR_OPS), ("systemctl", FAKE_SYSTEMCTL),
                       ("timeout", FAKE_TIMEOUT), ("journalctl", FAKE_JOURNALCTL)]:
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)
    for name, body in steps.items():
        (queue / name).write_text("#!/usr/bin/env bash\n" + body + "\n")
    if stop_flag:
        (tmp_path / "STOP").touch()
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "FAKE": str(fake),
           "VLAR_PHASE0_URL": "gs://b/vla-router/phase0", "VLAR_PHASE0_DIR": str(local),
           "VLAR_STOP_FLAG": str(tmp_path / "STOP"), "VLAR_SHUTDOWN_CMD": f"touch {fake}/SHUTDOWN",
           "VLAR_PULL_RETRY_S": "0"}
    return fake, local, env


def run(tmp_path, steps, *, local_steps=None, env_extra=None, **kw):
    fake, local, env = setup(tmp_path, steps, **kw)
    env.update(env_extra or {})
    for name, body in (local_steps or {}).items():         # the VM's own copy, left on its disk from an earlier boot
        (local / "queue").mkdir(parents=True, exist_ok=True)
        (local / "queue" / name).write_text("#!/usr/bin/env bash\n" + body + "\n")
    out = subprocess.run(["bash", str(RUNNER)], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    p0 = fake / "bucket" / "b" / "vla-router" / "phase0"
    state = json.loads((p0 / "STATE.json").read_text())
    hist = [json.loads(l) for l in (p0 / "history.jsonl").read_text().splitlines()] \
        if (p0 / "history.jsonl").exists() else []
    return state, hist, p0, fake, local, out.stdout


def ls(d):
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def test_steps_run_in_name_order_then_the_vm_stops_out_of_work(tmp_path):
    steps = {"020-b.sh": 'echo b >> "$FAKE/order"', "010-a.sh": 'echo a >> "$FAKE/order"'}
    state, hist, p0, fake, local, _ = run(tmp_path, steps)
    assert (fake / "order").read_text().split() == ["a", "b"]
    assert state["state"] == "idle" and "out of queued work" in state["reason"]
    assert state["spend"]["total_usd"] > 0                     # the spend guard's running total, every record
    assert [h["step"] for h in hist] == ["010-a", "020-b"] and all(h["outcome"] == "done" for h in hist)
    assert all(h["spend"]["total_usd"] > 0 for h in hist)
    assert ls(p0 / "queue") == [] and ls(p0 / "done") == ["010-a.sh", "020-b.sh"]
    assert (fake / "SHUTDOWN").exists()
    log = (fake / "log").read_text()
    assert "systemctl stop vlar-worker@* vlar-xvla vlar-pi05" in log    # no GPU service left running
    assert (p0 / "logs" / "010-a.log").exists()                # logs mirrored to the bucket


def test_a_step_can_queue_the_next_step_and_it_runs_before_the_stop(tmp_path):
    # The parity gate's pattern: on success it queues the base rollouts into the bucket's queue.
    gate = ('echo gate >> "$FAKE/order"\n'
            'printf "#!/usr/bin/env bash\\necho base >> \\"\\$FAKE/order\\"\\n" > "$FAKE/next.sh"\n'
            'gcloud storage cp -q "$FAKE/next.sh" "$VLAR_PHASE0_URL/queue/060-base.sh"')
    state, hist, *_rest = run(tmp_path, {"050-gate.sh": gate})
    fake = _rest[1]
    assert (fake / "order").read_text().split() == ["gate", "base"]
    assert [h["step"] for h in hist] == ["050-gate", "060-base"] and state["state"] == "idle"


def test_a_step_that_needs_the_user_stays_queued_and_stops_the_vm(tmp_path):
    steps = {"010-a.sh": 'echo "the tokenizer is missing" > "$VLAR_REASON_FILE"; exit 75',
             "020-b.sh": 'echo b >> "$FAKE/order"'}
    state, hist, p0, fake, local, _ = run(tmp_path, steps)
    assert state["state"] == "needs_input" and "the tokenizer is missing" in state["reason"]
    assert state["step"] == "010-a" and not (fake / "order").exists()
    assert ls(p0 / "queue") == ["010-a.sh", "020-b.sh"]       # re-runs on the next boot, once the input is there
    assert hist[-1]["outcome"] == "needs_input" and (fake / "SHUTDOWN").exists()


def test_a_failed_gate_stops_everything_and_queues_nothing(tmp_path):
    steps = {"050-parity.sh": 'echo "McNemar p = 0.01" > "$VLAR_REASON_FILE"; exit 76',
             "060-base.sh": 'echo base >> "$FAKE/order"'}
    state, hist, p0, fake, local, _ = run(tmp_path, steps)
    assert state["state"] == "gate_failed" and "McNemar p = 0.01" in state["reason"]
    assert ls(p0 / "failed") == ["050-parity.sh"] and not (fake / "order").exists()
    assert (fake / "SHUTDOWN").exists()


@pytest.mark.parametrize("code,word", [(1, "failed (exit 1)"), (124, "time cap")])
def test_a_failing_or_capped_step_is_set_aside_and_the_vm_stops(tmp_path, code, word):
    steps = {"010-a.sh": f'echo "row xvla-batch-invariance-l4 failed" > "$VLAR_REASON_FILE"; exit {code}',
             "020-b.sh": 'echo b >> "$FAKE/order"'}
    state, hist, p0, fake, local, _ = run(tmp_path, steps)
    assert state["state"] == "failed" and word in state["reason"]
    assert ls(p0 / "failed") == ["010-a.sh"] and ls(p0 / "queue") == ["020-b.sh"]
    assert not (fake / "order").exists() and (fake / "SHUTDOWN").exists()


@pytest.mark.parametrize("case,outcome,word", [
    ("oom", "oom_killed", "out-of-memory killer"),
    ("plain", "killed", "within its 600 s cap; no out-of-memory kill"),
    ("cap", "time_cap", "hit its time cap (1 s)")])
def test_exit_137_is_told_apart(tmp_path, case, outcome, word):
    """Amendment 41: 137 is the time cap only if the cap had passed; else an OOM kill if the kernel log since the
    step's start says so; else a kill within the cap."""
    cap, body = (1, "sleep 2; exit 137") if case == "cap" else (600, "exit 137")
    steps = {"010-a.sh": f"# cap-seconds: {cap}\n{body}", "020-b.sh": 'echo b >> "$FAKE/order"'}
    fake, local, env = setup(tmp_path, steps)
    if case == "oom":
        (fake / "kernel.log").write_text("Memory cgroup out of memory: Killed process 4242 (python) "
                                          "total-vm:31000000kB\n")
    out = subprocess.run(["bash", str(RUNNER)], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    p0 = fake / "bucket" / "b" / "vla-router" / "phase0"
    state = json.loads((p0 / "STATE.json").read_text())
    hist = [json.loads(line) for line in (p0 / "history.jsonl").read_text().splitlines()]
    assert state["state"] == "failed" and word in state["reason"], state
    assert hist[-1]["outcome"] == outcome and hist[-1]["exit"] == 137
    assert ls(p0 / "failed") == ["010-a.sh"] and ls(p0 / "queue") == ["020-b.sh"]


def test_the_cap_comment_is_passed_to_timeout(tmp_path):
    steps = {"010-a.sh": "# cap-seconds: 1234\ntrue"}
    fake, local, env = setup(tmp_path, steps)
    (Path(env["PATH"].split(":")[0]) / "timeout").write_text(
        '#!/bin/bash\necho "timeout $*" >> "$FAKE/log"\nwhile [[ "$1" == --* ]]; do shift; done\nshift\nexec "$@"\n')
    subprocess.run(["bash", str(RUNNER)], env=env, capture_output=True, text=True, timeout=60, check=True)
    assert "timeout --kill-after=120 1234 bash" in (fake / "log").read_text()


def test_the_spend_guards_stop_flag_starts_no_step(tmp_path):
    state, hist, p0, fake, local, _ = run(tmp_path, {"010-a.sh": 'echo a >> "$FAKE/order"'}, stop_flag=True)
    assert state["state"] == "stopped" and "STOP flag" in state["reason"]
    assert not (fake / "order").exists() and ls(p0 / "queue") == ["010-a.sh"]


def test_a_stop_from_outside_records_the_step_and_keeps_it_queued(tmp_path):
    fake, local, env = setup(tmp_path, {"010-slow.sh": 'touch "$FAKE/started"; sleep 30'})
    proc = subprocess.Popen(["bash", str(RUNNER)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for _ in range(100):
        if (fake / "started").exists():
            break
        time.sleep(0.1)
    proc.send_signal(signal.SIGTERM)                            # what systemd sends on preemption
    proc.wait(timeout=30)
    p0 = fake / "bucket" / "b" / "vla-router" / "phase0"
    state = json.loads((p0 / "STATE.json").read_text())
    assert state["state"] == "interrupted" and state["step"] == "010-slow"
    assert ls(p0 / "queue") == ["010-slow.sh"] and not (fake / "SHUTDOWN").exists()


# ---------------------------------------------------------------- the queue mirrors the bucket, removals included

def test_a_step_taken_out_of_the_bucket_queue_does_not_run_from_the_vms_copy(tmp_path):
    """The finding of 2026-11-24: `pull_queue` copied without deleting, and the VM's queue survives a stop, so taking
    500 out of the bucket's queue would not have stopped it. The local queue now mirrors the bucket exactly."""
    stale = {"500-calibrate.sh": 'echo stale >> "$FAKE/order"', "010-a.sh": 'echo a >> "$FAKE/order"'}
    state, hist, p0, fake, local, _ = run(tmp_path, {"010-a.sh": 'echo a >> "$FAKE/order"'}, local_steps=stale)
    assert (fake / "order").read_text().split() == ["a"]
    assert [h["step"] for h in hist] == ["010-a"] and state["state"] == "idle"
    assert ls(local / "queue") == [] and "500-calibrate.sh" not in ls(local / "done") + ls(local / "failed")
    assert "storage rsync --delete-unmatched-destination-objects" in (fake / "log").read_text()


def test_an_empty_bucket_queue_empties_the_vms_copy(tmp_path):
    # Cloud Storage has no empty folders: rsync cannot read an empty queue, so the parent's listing decides.
    state, hist, p0, fake, local, _ = run(tmp_path, {}, local_steps={"500-calibrate.sh": 'echo stale >> "$FAKE/order"'})
    assert not (fake / "order").exists() and hist == []
    assert state["state"] == "idle" and ls(local / "queue") == []


def test_an_unreadable_bucket_queue_starts_no_step(tmp_path):
    state, hist, p0, fake, local, _ = run(tmp_path, {"010-a.sh": 'echo a >> "$FAKE/order"'},
                                          local_steps={"010-a.sh": 'echo a >> "$FAKE/order"'},
                                          env_extra={"FAKE_LS_FAIL": "1"})
    assert not (fake / "order").exists() and hist == []
    assert state["state"] == "failed" and "could not be read" in state["reason"] and (fake / "SHUTDOWN").exists()
    assert ls(local / "queue") == ["010-a.sh"]                # kept, not run: the next boot mirrors again


PHASE0 = ["010-load-images", "015-host-checks", "020-weights", "030-rerun-sim", "040-rerun-xvla", "050-rerun-pi05",
          "090-tests-verdict", "200-step2-measure", "210-step2-geometry-check", "300-parity", "400-base",
          "500-calibrate", "550-gripper", "600-freeze"]


def test_after_600_the_bucket_queue_is_empty_and_no_disk_reruns_phase_0(tmp_path):
    """Every done step leaves the bucket's queue (`move_step`), so after 600 a fresh disk, or the old one, runs
    nothing from 010-600 again."""
    steps = {f"{n}.sh": f'echo {n} >> "$FAKE/order"' for n in PHASE0}
    assert sorted(steps) == sorted(p.name for p in (REPO / "ops" / "vm" / "phase0" / "steps").glob("*.sh"))
    state, hist, p0, fake, local, _ = run(tmp_path, steps)
    assert [h["step"] for h in hist] == PHASE0 and state["state"] == "idle"
    assert ls(p0 / "queue") == [] and ls(p0 / "done") == sorted(steps)
    log = (fake / "log").read_text()
    assert all(f"storage rm -q gs://b/vla-router/phase0/queue/{n}" in log for n in steps)
    (fake / "order").unlink()
    (fake / "SHUTDOWN").unlink()
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "FAKE": str(fake),
           "VLAR_PHASE0_URL": "gs://b/vla-router/phase0", "VLAR_STOP_FLAG": str(tmp_path / "STOP"),
           "VLAR_SHUTDOWN_CMD": f"touch {fake}/SHUTDOWN", "VLAR_PULL_RETRY_S": "0"}
    for disk in (local, tmp_path / "fresh-disk"):           # the old disk, and a new one restored from the bucket
        out = subprocess.run(["bash", str(RUNNER)], env={**env, "VLAR_PHASE0_DIR": str(disk)}, capture_output=True,
                             text=True, timeout=60)
        assert out.returncode == 0, out.stderr
        assert not (fake / "order").exists()
        assert json.loads((p0 / "STATE.json").read_text())["state"] == "idle"


# ---------------------------------------------------------------- the ledger's check before every step (§0.1 (d))

def check_says(tmp_path, rc, msg):
    fake = tmp_path / "fake"
    fake.mkdir(parents=True, exist_ok=True)
    (fake / "check_rc").write_text(str(rc))
    (fake / "check_msg").write_text(msg + "\n")


def test_a_step_past_its_phases_stop_point_does_not_start_and_stays_queued(tmp_path):
    check_says(tmp_path, 3, "spend phase pilot: $2.35 of VM since it opened (stop point $2.34, line $2.50)")
    body = '# spend-phase: pilot\necho a >> "$FAKE/order"'
    state, hist, p0, fake, local, _ = run(tmp_path, {"710-pilot.sh": body})
    assert state["state"] == "stopped" and "710-pilot not started: spend phase pilot" in state["reason"]
    assert not (fake / "order").exists() and ls(p0 / "queue") == ["710-pilot.sh"] and hist == []
    assert "vlar-ops spend check --phase pilot" in (fake / "log").read_text()


def test_a_step_whose_phase_is_not_open_waits_for_the_user(tmp_path):
    check_says(tmp_path, 4, "the step belongs to spend phase 'pilot', but the open phase is none")
    state, hist, p0, fake, local, _ = run(tmp_path, {"710-pilot.sh": '# spend-phase: pilot\necho a >> "$FAKE/order"'})
    assert state["state"] == "needs_input" and "open phase is none" in state["reason"]
    assert not (fake / "order").exists() and ls(p0 / "queue") == ["710-pilot.sh"]


def test_an_unreadable_ledger_starts_no_step(tmp_path):
    check_says(tmp_path, 1, "Traceback: bucket unreachable")
    state, hist, p0, fake, local, _ = run(tmp_path, {"010-a.sh": 'echo a >> "$FAKE/order"'})
    assert state["state"] == "failed" and "could not be checked" in state["reason"]
    assert not (fake / "order").exists() and (fake / "log").read_text().count("vlar-ops spend check") == 3


def test_steps_naming_no_phase_are_checked_without_one(tmp_path):
    state, hist, p0, fake, local, _ = run(tmp_path, {"010-a.sh": 'echo a >> "$FAKE/order"'})
    assert state["state"] == "idle" and (fake / "order").read_text().split() == ["a"]
    assert "vlar-ops spend check\n" in (fake / "log").read_text()


def test_post_freeze_steps_parse_and_carry_the_headers_the_runner_reads():
    """The runner reads `# cap-seconds: N` and `# spend-phase: NAME` with sed; a step meant for a spend phase must
    name it exactly (the pilot: `pilot`, approved up to $2.50, SPEC §17 amendment 59)."""
    import re
    steps = sorted((REPO / "ops" / "vm" / "postfreeze").glob("*.sh"))
    assert [p.name for p in steps] == ["700-smoke.sh", "710-pilot.sh", "720-phase1a.sh", "730-phase1b-screen.sh",
                                       "735-phase1b-topup.sh", "739-phase1b-recompare.sh", "740-phase1b.sh"]
    recompare = (REPO / "ops" / "vm" / "postfreeze" / "739-phase1b-recompare.sh").read_text()   # amendment 74
    assert re.findall(r"^# spend-phase: *([A-Za-z0-9_.-]+) *$", recompare, re.M) == ["phase1b"]
    assert "--stop-flag" not in recompare and "--mismatch" not in recompare and "rm -f \"${MISMATCH}\"" not in recompare
    topup = (REPO / "ops" / "vm" / "postfreeze" / "735-phase1b-topup.sh").read_text()     # amendment 68
    assert re.findall(r"^# spend-phase: *([A-Za-z0-9_.-]+) *$", topup, re.M) == ["phase1b-topup"]
    assert "exec bash /opt/vla-router/ops/vm/postfreeze/730-phase1b-screen.sh" in topup
    for p in steps:
        assert subprocess.run(["bash", "-n", str(p)]).returncode == 0, p.name
        assert re.search(r"^# cap-seconds: [0-9]+ *$", p.read_text(), re.M), p.name
    pilot = (REPO / "ops" / "vm" / "postfreeze" / "710-pilot.sh").read_text()
    assert re.findall(r"^# spend-phase: *([A-Za-z0-9_.-]+) *$", pilot, re.M) == ["pilot"]
    assert 'run_phase pilot "--phase pilot"' in pilot and "export VLAR_GOLDEN_DONE=1" in pilot
    assert pilot.index("golden_check ||") < pilot.index("vlar-ops pilot check --plan-dir") < pilot.index("run_phase pilot \"")
    # the resume (the user, 2026-11-24): the corrected (iii) once, the open jobs point by point after the queue's
    # resume, and workers of at least 5 GiB each with at most 2 jobs per process
    assert "pilot_check checks-iii.json checks-iii.json" in pilot
    assert (pilot.index("vlar-ops queue add \"") < pilot.index("vlar-ops queue resume >") < pilot.index("vlar-ops pilot reorder --plan-dir")
            < pilot.index('VLAR_SUPERVISE_ARGS="--min-limit-gib 5 --max-jobs 2" run_phase pilot'))
    lib = (REPO / "ops" / "vm" / "phase0" / "lib.sh").read_text()
    assert "${VLAR_SUPERVISE_ARGS:-}" in lib[lib.index("run_phase() {"):]
    # Tranche A (amendment 65): its own spend phase, the golden gate before any job, the plan's own phase for the
    # queue and the reorder, and the workers by amendment 64's RAM rule
    a = (REPO / "ops" / "vm" / "postfreeze" / "720-phase1a.sh").read_text()
    assert re.findall(r"^# spend-phase: *([A-Za-z0-9_.-]+) *$", a, re.M) == ["phase1a"] and "PH=phase1a" in a
    # Tranche B (amendments 56, 65): the screen and the rest in their own spend phases, each behind the golden gate;
    # 740 compares re-runs while they run (a difference raises the stop flag) and again before any branch plan
    sc = (REPO / "ops" / "vm" / "postfreeze" / "730-phase1b-screen.sh").read_text()
    assert re.findall(r"^# spend-phase: *([A-Za-z0-9_.-]+) *$", sc, re.M) == ["phase1b-screen"]
    assert sc.index("golden_check ||") < sc.index('run_phase explore "--phase explore"')
    b = (REPO / "ops" / "vm" / "postfreeze" / "740-phase1b.sh").read_text()
    assert re.findall(r"^# spend-phase: *([A-Za-z0-9_.-]+) *$", b, re.M) == ["phase1b"]
    assert (b.index("phase1 select-b") < b.index("golden_check ||") < b.index("--stop-flag \"${STOP_FLAG}\" >/dev/null")
            < b.index('run_phase explore "--phase explore"') < b.index("--require-all") < b.index("phase1 plan-b")
            < b.index('run_phase "${PH}" "--phase ${PH}"'))
    assert (a.index("golden_check ||") < a.index("export VLAR_GOLDEN_DONE=1") < a.index('vlar-ops queue add "')
            < a.index("vlar-ops queue resume >") < a.index("vlar-ops pilot reorder --plan-dir")
            < a.index('VLAR_SUPERVISE_ARGS="--limit-gib 5.5 --servers-peak-gb 20.45 --max-jobs 2" run_phase "${PH}"'))
    assert "--min-limit-gib" not in a and "pilot check" not in a


def test_create_retry_tries_only_the_three_zones_and_queues_held_steps_only_after_the_guard_dry_run():
    """ops/gcp/33_create_retry.sh (the user, 2026-11-25): us-central1-a, -b, -c every 10 minutes for up to 6 hours, no
    other region; env.sh pointed at the zone that starts; the guard's stop dry run against that VM before any held
    step reaches the queue; a non-stockout error stops it."""
    import re
    p = REPO / "ops" / "gcp" / "33_create_retry.sh"
    s = p.read_text()
    assert subprocess.run(["bash", "-n", str(p)]).returncode == 0
    assert "ZONES=(us-central1-a us-central1-b us-central1-c)" in s
    assert set(re.findall(r"us-[a-z]+\d-[a-z]", s)) == {"us-central1-a", "us-central1-b", "us-central1-c"}
    assert 'TRY_S="${RETRY_TRY_S:-600}"' in s and "${RETRY_HOURS:-6} * 3600" in s
    body = s[s.index("log \"retry: zones"):]
    assert body.index('sed -i \'\' "s/^export ZONE=') < body.index('if ! dry_run "${z}"') < body.index(
        'gcloud storage cp -q "${f}" "gs://${BUCKET}/vla-router/phase0/queue/')
    assert "exit 2" in body and "exit 3" in body and "exit 4" in body
    dr = s[s.index("dry_run() {"):s.index("watch_vm() {")]
    for part in ("SPOT*STOP", "shutdown_command", "guard_sessions_since_create", "spend check --phase \"${PHASE}\""):
        assert part in dr, part


# ---------------------------------------------------------------- the Phase 1 restart watcher (ops/gcp/34_restart_watch.sh)

def _watch_env(tmp_path, statuses, *, preempted=False):
    """Fakes for the watcher: a GCE helper replaying `statuses` (one per status read; '' = unreadable), a vlar-ops
    whose spend check passes, and an osascript that records its calls."""
    import os
    (tmp_path / "env.sh").write_text('export PROJECT_ID=p VM_NAME=v BUCKET=b\n')
    (tmp_path / "statuses").write_text("\n".join(statuses) + "\n")
    gce = tmp_path / "gce"
    gce.write_text(f"""#!/bin/bash
case "$1" in
  status) n=$(cat {tmp_path}/n 2>/dev/null || echo 0); echo $((n+1)) > {tmp_path}/n
          s=$(sed -n "$((n+1))p" {tmp_path}/statuses); [[ -n "$s" ]] && echo "$s" || exit 1 ;;
  state) echo "running 730-phase1b-screen running" ;;
  preempted-since-start) exit {0 if preempted else 1} ;;
  start) echo "start" >> {tmp_path}/starts; echo RUNNING ;;
esac
""")
    vops = tmp_path / "vops"
    vops.write_text('#!/bin/bash\n[[ "$1 $2" == "spend check" ]] && { echo ok; exit 0; }\necho \'{"total_usd": 1, "phase": {}}\'\n')
    osa = tmp_path / "osascript"
    osa.write_text(f'#!/bin/bash\necho "$*" >> {tmp_path}/notified\n')
    for f in (gce, vops, osa):
        f.chmod(0o755)
    return {**os.environ, "ENV_SH": str(tmp_path / "env.sh"), "GCE": str(gce), "VLAR_OPS": str(vops),
            "NOTIFY": str(osa), "POLL_S": "0", "TRY_S": "0", "RECHECK_S": "0", "NO_START": "1", "PHASE": "phase1b-screen"}


def test_the_watcher_never_reads_an_unreadable_status_as_running(tmp_path):
    script = REPO / "ops" / "gcp" / "34_restart_watch.sh"
    out = subprocess.run(["bash", "-c", f'LIB_ONLY=1 ENV_SH={tmp_path}/env.sh; echo "export VM_NAME=v" > $ENV_SH; '
                          f'source {script}; for s in "" "ERROR: (gcloud) Reauthentication failed" running RUNNING '
                          'STAGING TERMINATED STOPPING; do printf "%s|" "$(classify "$s")"; done'],
                         capture_output=True, text=True, timeout=30, env={**__import__("os").environ, "LIB_ONLY": "1",
                                                                         "ENV_SH": str(tmp_path / "env.sh")})
    assert out.stdout == "unknown|unknown|unknown|running|starting|stopped|stopping|", out.stderr
    env = _watch_env(tmp_path, ["", "", "", "", "RUNNING", ""], preempted=True)
    r = subprocess.run(["bash", str(script)], env={**env, "MAX_POLLS": "6"}, capture_output=True, text=True, timeout=60)
    log = r.stdout
    assert r.returncode == 0 and "VM RUNNING" in log and log.count("VM UNKNOWN") == 5
    assert log.count("status NOT readable") == 1 and "status NOT readable 3 polls" in log
    notes = (tmp_path / "notified").read_text()
    assert notes.count("display notification") == 1 and 'sound name "Sosumi"' in notes
    assert not (tmp_path / "starts").exists() and "preempted" not in log       # unreadable is never a stop either


def test_the_watcher_restarts_only_after_a_preemption(tmp_path):
    script = REPO / "ops" / "gcp" / "34_restart_watch.sh"
    env = _watch_env(tmp_path, ["RUNNING", "TERMINATED", "RUNNING"], preempted=True)
    r = subprocess.run(["bash", str(script)], env={**env, "MAX_POLLS": "3"}, capture_output=True, text=True, timeout=60)
    assert "automatic restart 1/8" in r.stdout and (tmp_path / "starts").read_text().count("start") == 1
    other = tmp_path / "other"
    other.mkdir()
    env = _watch_env(other, ["RUNNING", "TERMINATED"])
    r = subprocess.run(["bash", str(script)], env={**env, "MAX_POLLS": "3"}, capture_output=True, text=True, timeout=60)
    assert "NOT a preemption" in r.stdout and r.returncode == 0
    assert not (other / "starts").exists() and "automatic restart 1/" not in r.stdout   # a plain stop is never restarted


def test_gce_adc_counts_a_preemption_during_the_start():
    """ops/gcp/gce_adc.py's preemption check (2026-11-26 07:48Z): the preemption was inserted 0.6 s before the VM's
    lastStartTimestamp; the newest successful start operation is the reference, a stockout start is not."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gce_adc", Path(__file__).parents[1] / "ops/gcp/gce_adc.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)                          # no Google imports or credentials outside main()
    last = "2026-11-25T23:48:34.695-08:00"
    starts = [{"insertTime": "2026-11-25T23:38:10.814-08:00", "error": {"errors": [{"code": "ZONE_RESOURCE_POOL_EXHAUSTED"}]}},
              {"insertTime": "2026-11-25T23:48:18.455-08:00"}]
    during = [{"insertTime": "2026-11-25T19:42:06.903-08:00"}, {"insertTime": "2026-11-25T23:48:34.089-08:00"}]
    assert g.preempted_since_start(last, starts, during)
    assert not g.preempted_since_start(last, starts, during[:1])          # only an older preemption: a plain stop
    assert not g.preempted_since_start(last, starts[:1], during)          # no successful start op: lastStartTimestamp
    assert g.preempted_since_start(last, [], during + [{"insertTime": "2026-11-26T01:00:00-08:00"}])
    # a stockout start after a preemption does not hide it (only successful starts are the reference)
    assert g.preempted_since_start("2026-11-25T20:18:58.978-08:00",
                                   [{"insertTime": "2026-11-25T20:18:40-08:00"},
                                    {"insertTime": "2026-11-25T23:40:00-08:00", "error": {"errors": []}}],
                                   [{"insertTime": "2026-11-25T23:15:00-08:00"}])


def _gce_adc():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gce_adc", REPO / "ops/gcp/gce_adc.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


# The VM's real operations, 2026-11-25/26 (PST as the API prints them): preemptions, and the starts around them.
_PREEMPTED = [{"insertTime": t} for t in ("2026-11-25T11:00:24.381-08:00", "2026-11-25T12:32:53.882-08:00",
                                          "2026-11-25T14:02:27.311-08:00", "2026-11-25T18:00:29.997-08:00",
                                          "2026-11-25T19:42:06.903-08:00", "2026-11-25T23:48:34.089-08:00")]


def test_gce_adc_reads_an_end_of_work_or_guard_stop_as_not_a_preemption():
    """The reverse of the 07:48Z case: a stop the VM made itself (the runner's `shutdown -h` at the end of its work or
    at a guard stop, recorded as compute.instances.guestTerminate, which the check never reads) is not a preemption,
    whatever preemptions came before the start."""
    g = _gce_adc()
    # 730's end: started 20:18:45 PST (lastStartTimestamp 20:18:58.978), preempted last at 19:42:06, stopped itself
    # (guestTerminate 23:15:12): judged with the preemptions known at 23:16
    starts = [{"insertTime": "2026-11-25T18:13:04.418-08:00"}, {"insertTime": "2026-11-25T20:18:45.000-08:00"}]
    assert not g.preempted_since_start("2026-11-25T20:18:58.978-08:00", starts, _PREEMPTED[:5])
    # a guard stop minutes after a start (spend check fails at boot, the runner shuts down): no preemption since
    assert not g.preempted_since_start("2026-11-25T20:18:58.978-08:00", starts + [
        {"insertTime": "2026-11-25T23:38:10.814-08:00", "error": {"errors": [{"code": "ZONE_RESOURCE_POOL_EXHAUSTED"}]}}],
        _PREEMPTED[:5])
    # the same with no start operation on record (falls back to lastStartTimestamp)
    assert not g.preempted_since_start("2026-11-25T20:18:58.978-08:00", [], _PREEMPTED[:5])
    # and the preemption 16 s into 735's start is one
    assert g.preempted_since_start("2026-11-25T23:48:34.695-08:00",
                                   starts + [{"insertTime": "2026-11-25T23:48:18.455-08:00"}], _PREEMPTED)


def test_a_step_is_done_only_by_its_own_records_never_the_runner_state():
    """ops/gcp/step_done.py: 735 queued behind 730, its first boot preempted before the runner read the queue (the
    runner's state still 730's "every queued step is done"): not done. Done needs every planned job's marker and the
    step's exit 0 at or after its first start (735 runs 730's script, so its exits say "730")."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("step_done", REPO / "ops/gcp/step_done.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    old, new = [f"o{i}" for i in range(1974)], [f"n{i}" for i in range(864)]
    exits_730 = [{"step": "730", "step_exit": 143, "at": 1795664487}, {"step": "730", "step_exit": 0, "at": 1795677280}]
    t735 = 1795678680.0                                                       # 2026-11-26 07:38Z
    assert m.step_done(old, old, exits_730, step="730", since=1795631280.0)["done"]            # 730 itself
    r = m.step_done(old + new, old, exits_730, step="730", since=t735)                         # 735 after the 07:48Z boot
    assert not r["done"] and r["missing"] == 864 and r["exits_since"] == []                   # 730's exit 0 is too old
    r = m.step_done(old + new, old + new, exits_730 + [{"step": "730", "step_exit": 143, "at": t735 + 9000}],
                    step="730", since=t735)
    assert not r["done"] and r["missing"] == 0 and not r["exit_0_since"]                      # all markers, no clean exit
    r = m.step_done(old + new, old + new[:-1], exits_730 + [{"step": "730", "step_exit": 0, "at": t735 + 9000}],
                    step="730", since=t735)
    assert not r["done"] and r["missing"] == 1                                                # a clean exit, one missing
    assert m.step_done(old + new, old + new, exits_730 + [{"step": "730", "step_exit": 0, "at": t735 + 9000}],
                       step="730", since=t735)["done"]
