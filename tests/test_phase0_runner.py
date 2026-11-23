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
args=(); for a in "$@"; do [[ "$a" == -q || "$a" == -r ]] || args+=("$a"); done
path() { echo "$FAKE/bucket/${1#gs://}"; }
case "${args[0]} ${args[1]}" in
  "storage cp") src="${args[2]}"; dst="${args[3]}"
    [[ "$src" == gs://* ]] && src="$(path "$src")"
    [[ "$dst" == gs://* ]] && dst="$(path "$dst")"
    mkdir -p "$(dirname "$dst")"; if [[ "$src" == - ]]; then cat > "$dst"; else cp "$src" "$dst"; fi ;;
  "storage rsync") src="${args[2]}"; dst="${args[3]}"
    [[ "$src" == gs://* ]] && src="$(path "$src")"
    [[ "$dst" == gs://* ]] && dst="$(path "$dst")"
    [[ -d "$src" ]] || exit 1
    mkdir -p "$dst"; cp -R "$src/." "$dst/" ;;
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
esac
"""
FAKE_SYSTEMCTL = '#!/bin/bash\necho "systemctl $*" >> "$FAKE/log"\n'
# GNU timeout's interface, without the cap (the runner's 124/137 handling is tested by a step exiting 124).
FAKE_TIMEOUT = '#!/bin/bash\nwhile [[ "$1" == --* ]]; do shift; done\nshift\nexec "$@"\n'


def setup(tmp_path, steps, stop_flag=False):
    fake, bin_, local = tmp_path / "fake", tmp_path / "bin", tmp_path / "local"
    queue = fake / "bucket" / "b" / "vla-router" / "phase0" / "queue"
    queue.mkdir(parents=True)
    bin_.mkdir()
    for name, text in [("gcloud", FAKE_GCLOUD), ("vlar-ops", FAKE_VLAR_OPS), ("systemctl", FAKE_SYSTEMCTL),
                       ("timeout", FAKE_TIMEOUT)]:
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)
    for name, body in steps.items():
        (queue / name).write_text("#!/usr/bin/env bash\n" + body + "\n")
    if stop_flag:
        (tmp_path / "STOP").touch()
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "FAKE": str(fake),
           "VLAR_PHASE0_URL": "gs://b/vla-router/phase0", "VLAR_PHASE0_DIR": str(local),
           "VLAR_STOP_FLAG": str(tmp_path / "STOP"), "VLAR_SHUTDOWN_CMD": f"touch {fake}/SHUTDOWN"}
    return fake, local, env


def run(tmp_path, steps, **kw):
    fake, local, env = setup(tmp_path, steps, **kw)
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
