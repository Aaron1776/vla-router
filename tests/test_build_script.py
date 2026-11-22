"""ops/gcp/35_build_images.sh against a fake gcloud: network drops are retried and never trigger the clean-up; the
build VM is deleted only after DONE or FAILED was read from the bucket, the cap, or a confirmed disappearance; the
deletion is confirmed for both the VM and its disk."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# Inside an image (VLAR_IN_IMAGE=1, set by containers/Dockerfile) ops/ and git are absent and the test skips; outside
# one, anything missing still fails it.
pytestmark = pytest.mark.skipif(
    os.environ.get("VLAR_IN_IMAGE") == "1"
    and not ((REPO / "ops" / "gcp" / "35_build_images.sh").exists() and shutil.which("git")),
    reason="repo-only files (ops/) and git, not in the image")

FAKE_GCLOUD = r"""#!/bin/bash
F="$FAKE"; echo "$*" >> "$F/log"
num() { cat "$F/$1" 2>/dev/null || echo 0; }
net() { local n; n=$(num outage); if (( n > 0 )); then echo $((n - 1)) > "$F/outage"; echo "network down" >&2; exit 1; fi; }
path() { echo "$F/bucket/${1#gs://}"; }
case "$1 $2" in
  "storage ls") net; p=$(path "$3")
    if [[ "$3" == */ ]]; then
      [[ -d "$p" ]] || exit 1
      n=$(( $(num done_after) - 1 )); echo $n > "$F/done_after"
      if (( n == 0 )); then case $(cat "$F/outcome") in
        done) touch "$p/DONE"; echo '{"sim": "sha256:x"}' > "$p/image-digests.json" ;;
        failed) echo "docker build sim" > "$p/FAILED" ;;
        vanish) rm -f "$F/vm" ;;
      esac; fi
      [[ -e "$p/DONE" ]] && echo "LISTED DONE" >> "$F/log"
      [[ -e "$p/FAILED" ]] && echo "LISTED FAILED" >> "$F/log"
      for f in $(ls "$p"); do echo "$3$f"; done
    else [[ -e "$p" ]] || exit 1; echo "$3"; fi ;;
  "storage cp") p=$(path "$4"); mkdir -p "$(dirname "$p")"; cp "$3" "$p" ;;
  "storage rm") rm -f "$(path "${@: -1}")" ;;
  "storage cat") net; cat "$(path "$3")" ;;
  "compute instances") case "$3" in
      create) touch "$F/vm" "$F/disk"; cp "$F/outage_after_create" "$F/outage" ;;
      list) net; echo "vlar-builder"; [[ -e "$F/vm" ]] && echo "vlar-build"; exit 0 ;;
      describe) net; [[ -e "$F/vm" ]] ;;
      delete) n=$(num delete_fail); if (( n > 0 )); then echo $((n - 1)) > "$F/delete_fail"; exit 1; fi
              rm -f "$F/vm" "$F/disk" ;;
    esac ;;
  "compute disks") case "$3" in
      list) net; [[ -e "$F/disk" ]] && echo "vlar-build"; exit 0 ;;
      delete) rm -f "$F/disk" ;;
    esac ;;
  *) echo "fake gcloud: unhandled $*" >&2; exit 9 ;;
esac
"""


def run_build(tmp_path, outcome, done_after, outage=0, delete_fail=0, cap=None):
    repo, fake, bin_ = tmp_path / "repo", tmp_path / "fake", tmp_path / "bin"
    (repo / "ops" / "gcp").mkdir(parents=True)
    for name in ["35_build_images.sh", "build_vm_startup.sh"]:
        shutil.copy(REPO / "ops" / "gcp" / name, repo / "ops" / "gcp" / name)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo)]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "x"], check=True)
    fake.mkdir()
    bin_.mkdir()
    (bin_ / "gcloud").write_text(FAKE_GCLOUD)
    (bin_ / "gcloud").chmod(0o755)
    for k, v in {"outcome": outcome, "done_after": done_after, "outage_after_create": outage,
                 "delete_fail": delete_fail}.items():
        (fake / k).write_text(f"{v}\n")
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", FAKE=str(fake), PROJECT_ID="p", BUCKET="b",
               ZONE="z", SA_NAME="sa", BUILD_VM_NAME="vlar-build", BUILD_MACHINE="m", BUILD_DISK_GB="1",
               BUILD_MAX_HOURS="4", BUILD_IMAGE_FAMILY="f", BUILD_IMAGE_PROJECT="ip", VLAR_POLL_SECONDS="0",
               VLAR_DELETE_TRIES="20")
    if cap is not None:
        env["VLAR_CAP_SECONDS"] = str(cap)
    p = subprocess.run(["bash", "ops/gcp/35_build_images.sh"], cwd=repo, env=env, capture_output=True, text=True,
                       timeout=60)
    log = (fake / "log").read_text().splitlines()
    return p, log, fake


def first(log, prefix):
    return next((i for i, line in enumerate(log) if line.startswith(prefix)), None)


def test_network_drops_are_retried_and_the_vm_is_deleted_only_after_done(tmp_path):
    p, log, fake = run_build(tmp_path, "done", done_after=3, outage=8, delete_fail=2)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "check failed (network?)" in p.stdout
    assert "built and tested" in p.stdout and "sha256:x" in p.stdout
    assert "confirmed: the build VM and its disk are deleted" in p.stdout
    assert not (fake / "vm").exists() and not (fake / "disk").exists()
    assert first(log, "LISTED DONE") < first(log, "compute instances delete")


@pytest.mark.parametrize("outage", [0, 5])
def test_failed_build_reports_the_reason_then_deletes(tmp_path, outage):
    p, log, fake = run_build(tmp_path, "failed", done_after=2, outage=outage)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "build FAILED: docker build sim" in p.stdout
    assert "confirmed: the build VM and its disk are deleted" in p.stdout
    assert first(log, "LISTED FAILED") < first(log, "compute instances delete")


def test_a_vanished_vm_is_reported_and_only_its_leftover_disk_removed(tmp_path):
    p, log, fake = run_build(tmp_path, "vanish", done_after=2, outage=3)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "gone without a result" in p.stdout
    assert first(log, "compute instances delete") is None
    assert first(log, "compute disks delete") is not None and not (fake / "disk").exists()


def test_the_cap_deletes_the_vm(tmp_path):
    p, log, fake = run_build(tmp_path, "done", done_after=1000, cap=0)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "no result after 4 h" in p.stdout
    assert not (fake / "vm").exists() and not (fake / "disk").exists()


def test_an_unconfirmed_deletion_is_reported(tmp_path):
    p, log, fake = run_build(tmp_path, "done", done_after=1, delete_fail=1000)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "COULD NOT CONFIRM" in p.stdout and (fake / "vm").exists()
