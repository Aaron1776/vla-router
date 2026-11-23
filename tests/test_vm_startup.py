"""ops/vm/phase0/startup.sh, the L4 VM's boot script: the host-package freeze (amendment 36) against fake systemctl,
snap, pgrep, sleep, dpkg-query and sha256sum, and its place in the boot order."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
STARTUP = REPO / "ops" / "vm" / "phase0" / "startup.sh"
BASH = shutil.which("bash")
# ops/ is not in the images (VLAR_IN_IMAGE=1, containers/Dockerfile); outside one, a missing script still fails.
pytestmark = pytest.mark.skipif(os.environ.get("VLAR_IN_IMAGE") == "1" and not STARTUP.exists(),
                                reason="repo-only files (ops/), not in the image")

FAKE = {
    "systemctl": r"""#!/bin/bash
echo "systemctl $*" >> "$FAKE/log"
if [[ "$1" == is-active ]]; then
  for u in $FAKE_ACTIVE; do [[ " $* " == *" $u"* ]] && exit 0; done
  exit 3
fi
exit 0
""",
    "snap": '#!/bin/bash\necho "snap $*" >> "$FAKE/log"\n',
    # pgrep: "an apt run is in progress" for the first $FAKE_BUSY calls
    "pgrep": r"""#!/bin/bash
echo "pgrep $*" >> "$FAKE/log"
n=$(cat "$FAKE/pgrep-calls" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "$FAKE/pgrep-calls"
(( n <= ${FAKE_BUSY:-0} )) && exit 0
exit 1
""",
    "sleep": '#!/bin/bash\necho "sleep $*" >> "$FAKE/log"\n',
    "dpkg-query": '#!/bin/bash\nprintf "bash 5.2\\ncoreutils 9.4\\n"\n',
    "sha256sum": '#!/bin/bash\ncat > /dev/null\necho "0123456789abcdef0123  -"\n',
}


def freeze_function() -> str:
    text = STARTUP.read_text()
    m = re.search(r"^freeze_host_packages\(\) \{\n.*?^\}\n", text, re.S | re.M)
    assert m, "freeze_host_packages() not found in startup.sh"
    return m.group(0)


def run_freeze(tmp_path, busy=0, active="", snap=True):
    fake = tmp_path / "fake"
    bin_ = fake / "bin"
    bin_.mkdir(parents=True)
    for name, body in FAKE.items():
        if name == "snap" and not snap:
            continue
        p = bin_ / name
        p.write_text(body)
        p.chmod(0o755)
    conf = tmp_path / "apt.conf.d"
    conf.mkdir()
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "FAKE": str(fake), "FAKE_BUSY": str(busy), "FAKE_ACTIVE": active,
           "VLAR_APT_CONF_D": str(conf)}
    if not snap:          # only the fakes and the two real tools the function needs: no snap anywhere
        tools = fake / "tools"
        tools.mkdir()
        for t in ("seq", "cut"):
            (tools / t).symlink_to(shutil.which(t))
        env["PATH"] = f"{bin_}:{tools}"
    script = freeze_function() + "freeze_host_packages\n"
    r = subprocess.run([BASH, "-c", script], env=env, capture_output=True, text=True)
    log = (fake / "log").read_text().splitlines() if (fake / "log").exists() else []
    return r, log, conf


def test_freeze_turns_off_every_automatic_update_and_logs_the_package_digest(tmp_path):
    r, log, conf = run_freeze(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    for timer in ("apt-daily.timer", "apt-daily-upgrade.timer"):
        assert f"systemctl disable --now {timer}" in log
    mask = next(line for line in log if line.startswith("systemctl mask"))
    for unit in ("apt-daily.timer", "apt-daily-upgrade.timer", "apt-daily.service", "apt-daily-upgrade.service",
                 "unattended-upgrades.service"):
        assert unit in mask.split()
    assert "snap refresh --hold" in log
    settings = (conf / "99vlar-no-automatic-updates").read_text()
    for key in ("Enable", "Update-Package-Lists", "Download-Upgradeable-Packages", "Unattended-Upgrade",
                "AutocleanInterval"):
        assert f'APT::Periodic::{key} "0";' in settings
    assert "host packages frozen; installed packages' digest 0123456789abcdef" in r.stdout
    assert not any(line.startswith("sleep") for line in log)


def test_freeze_waits_for_an_apt_run_already_in_progress(tmp_path):
    r, log, _ = run_freeze(tmp_path, busy=3)
    assert r.returncode == 0, r.stdout + r.stderr
    assert [line for line in log if line.startswith("sleep")] == ["sleep 10"] * 3
    assert r.stdout.count("waiting for an apt/dpkg run") == 1
    pattern = next(line for line in log if line.startswith("pgrep"))
    for proc in ("apt\\.systemd\\.daily", "/usr/bin/unattended-upgrade", "/usr/bin/dpkg "):
        assert proc in pattern
    # unattended-upgrades' own shutdown helper runs all the time and must not count as an upgrade in progress
    assert not re.search(pattern.split(" ", 2)[2], "/usr/share/unattended-upgrades/unattended-upgrade-shutdown "
                                                    "--wait-for-signal")
    assert re.search(pattern.split(" ", 2)[2], "/usr/bin/python3 /usr/bin/unattended-upgrade")


def test_freeze_fails_if_a_timer_is_still_active(tmp_path):
    r, _, _ = run_freeze(tmp_path, active="apt-daily-upgrade.timer")
    assert r.returncode != 0
    assert "apt-daily-upgrade.timer is still active" in r.stdout


def test_freeze_without_snap(tmp_path):
    r, log, _ = run_freeze(tmp_path, snap=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not any(line.startswith("snap") for line in log)


def test_freeze_runs_first_on_every_boot():
    text = STARTUP.read_text()
    call = text.index("freeze_host_packages || fail")
    for later in ("gcloud storage cat", "pip install", "systemctl enable --now vlar-spend", "systemctl start vlar-phase0"):
        assert call < text.index(later), later


def test_a_failed_boot_is_recorded_for_the_spend_ledger(tmp_path):
    """startup.sh's fail() writes ops/failed-boots/<boot id>.json (amendment 42), which the guard folds in."""
    import json
    import time
    text = STARTUP.read_text()
    m = re.search(r"^record_failed_boot\(\) \{[^\n]*\n.*?^\}\n", text, re.S | re.M)
    assert m and "record_failed_boot \"$*\"" in text
    proc = tmp_path / "proc"
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("abc-123\n")
    (proc / "uptime").write_text("600.25 1000.0\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    gcloud = bin_ / "gcloud"
    gcloud.write_text('#!/bin/bash\necho "$*" > "$OUT/args"; cat > "$OUT/body"\n')
    gcloud.chmod(0o755)
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "OUT": str(tmp_path), "VLAR_PROC": str(proc),
           "STORAGE_URL": "gs://bucket/vla-router"}
    t0 = time.time()
    r = subprocess.run([BASH, "-c", 'STORAGE_URL="$STORAGE_URL"\n' + m.group(0) + 'record_failed_boot "pip install"'],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "args").read_text().split() == ["storage", "cp", "-",
                                                       "gs://bucket/vla-router/ops/failed-boots/abc-123.json"]
    rec = json.loads((tmp_path / "body").read_text())
    assert rec["boot_id"] == "abc-123" and rec["reason"] == "pip install"
    assert rec["failed_at"] - rec["boot_at"] == pytest.approx(600.25, abs=0.01) and rec["failed_at"] >= t0


def boot_disk_function() -> str:
    text = STARTUP.read_text()
    m = re.search(r"^record_boot_disk\(\) \{\n.*?^\}\n", text, re.S | re.M)
    assert m and 'record_boot_disk || fail "disk not recorded in the ledger"' in text
    assert text.index("systemctl enable --now vlar-spend") < text.index("record_boot_disk || fail")
    return m.group(0)


VLAR_OPS_SHIM = r'''#!{python}
# The real `vlar-ops` CLI over a local ledger; with RACE=1 the first ledger write is lost right after it succeeds,
# as the L4's disk record was on 2026-11-23 (tests/test_spend_ledger.py).
import json, os, sys
sys.path.insert(0, {src!r})
from vlar_ops import cli, spend
from vlar_ops.storage import LocalStorage
with open(os.environ["CALLS"], "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
{racy}
LosesTheNextWrite.armed = os.environ.get("RACE") == "1"
cli.open_storage = lambda url: LosesTheNextWrite(url)
cli.main(sys.argv[1:])
'''


@pytest.mark.parametrize("race", [False, True])
def test_the_disk_marker_is_written_only_after_the_ledger_confirms_the_record(tmp_path, race):
    """Amendment 48: `record_boot_disk` touches `.disk-recorded` only after `spend disk-created` has read its record
    back. With the record lost after the write (the 2026-11-23 race), no marker: the boot fails, and the next boot
    records the disk again. With a marker, the disk is not recorded twice."""
    import sys
    from test_spend_ledger import RACY_STORAGE
    r, local, store = tmp_path / "repo", tmp_path / "local", tmp_path / "store"
    (r / ".venv" / "bin").mkdir(parents=True)
    local.mkdir()
    shim = r / ".venv" / "bin" / "vlar-ops"
    shim.write_text(VLAR_OPS_SHIM.format(python=sys.executable, src=str(REPO / "src"), racy=RACY_STORAGE))
    shim.chmod(0o755)
    cfg = tmp_path / "ops.yaml"
    cfg.write_text(f"storage_url: {store}\nlocal_root: {local}\n")
    script = (f'L="{local}"; R="{r}"; NAME=vm; DISK_GB=100; DISK_RATE=0.1\n' + boot_disk_function()
              + "record_boot_disk; echo rc=$?\n")
    env = {"PATH": "/usr/bin:/bin", "VLAR_OPS_CONFIG": str(cfg), "CALLS": str(tmp_path / "calls"),
           "RACE": "1" if race else "0"}

    def boot():
        return subprocess.run([BASH, "-c", script], env=env, capture_output=True, text=True)

    def disks():
        p = store / "ops" / "ledger.json"
        return json.loads(p.read_text())["disks"] if p.exists() else []

    import json
    out = boot()
    marker = local / ".disk-recorded"
    if race:
        assert "rc=1" in out.stdout and "disk record NOT confirmed" in out.stderr, out.stderr
        assert not marker.exists() and disks() == []
        env["RACE"] = "0"
        out = boot()                                   # the next boot records it
    assert "rc=0" in out.stdout and "(confirmed in the ledger)" in out.stdout, out.stderr
    assert marker.exists() and [d["name"] for d in disks()] == ["vm-boot"]
    calls = (tmp_path / "calls").read_text().splitlines()
    assert boot().stdout.strip() == "rc=0"             # marker present: no second record
    assert (tmp_path / "calls").read_text().splitlines() == calls and len(disks()) == 1
