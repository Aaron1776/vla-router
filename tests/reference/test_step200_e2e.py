"""Phase 0 step 200 (ops/vm/phase0/steps/200-step2-measure.sh) end to end on the Mac: the step script itself, with its
library, runs every `vlar-ops step2` command for real (capture, the joint memory measurement with rendering
workers at every candidate batch size, the geometry rule, latencies, upload size, gripper widths, the calibration
update, the calibration file's write) against the stand-ins in `step200_fakes.py`: a fake docker, the production
server apps over fake model services, a fake nvidia-smi whose samples follow what is running, fake renderers (the
real `render` command over a stand-in simulator), fake benches, gcloud and sleep.

Found on 2026-11-23: on the L4, every joint measurement failed because `vlar-ops step2 load` raised
UnboundLocalError (`run` assigned a local `load` in another branch); the tests called `step2.load()` directly, never
the command, and nothing ran the step script, so it went unseen until the VM.

The script is copied with its machine paths moved under the test's directory (/opt/vla-router -> the repo, /etc/vlar,
/opt/hf, /opt/weights, /var/lib/vlar -> tmp) and the full-batch load shortened from 120 s to 2 s; nothing else.
Uses ports 8777 and 8778, as the step does.
  .venv-sim/bin/python -m pytest tests/reference/test_step200_e2e.py
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import LIBERO_READY, have

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STEP = REPO / "ops" / "vm" / "phase0" / "steps" / "200-step2-measure.sh"
LIB = REPO / "ops" / "vm" / "phase0" / "lib.sh"
FAKES = HERE / "step200_fakes.py"
COMMIT = "c0ffee0"
SIZES = [1, 2, 4, 8, 16]
COSTS = {"1": 0.40, "2": 0.42, "4": 0.46, "8": 0.85, "16": 1.70}   # s per full batch: 4 is near the best feasible
LOAD_SECONDS = 2

SERVERS_READY = all(have(m) for m in ("fastapi", "uvicorn", "json_numpy", "requests", "numpy", "PIL", "yaml"))
pytestmark = [
    pytest.mark.skipif(not STEP.exists(), reason="repo-only files (ops/), not in the image"),
    pytest.mark.skipif(not SERVERS_READY, reason="needs FastAPI, uvicorn, json_numpy, requests, NumPy, Pillow"),
]
requires_libero = pytest.mark.skipif(not LIBERO_READY, reason="needs the simulator environment (LIBERO + robosuite)")

sys.path.insert(0, str(HERE))
import step200_fakes as fakes  # noqa: E402

FAKE_GCLOUD = r"""#!/bin/bash
# gs://<path> -> $FAKE/bucket/<path>; `storage ls` finds nothing (no tokenizer in the bucket)
echo "gcloud $*" >> "$FAKE/gcloud.log"
args=(); for a in "$@"; do [[ "$a" == -q || "$a" == -r ]] || args+=("$a"); done
path() { echo "$FAKE/bucket/${1#gs://}"; }
case "${args[0]} ${args[1]}" in
  "storage rsync") src="${args[2]}"; dst="${args[3]}"
    [[ "$src" == gs://* ]] && src="$(path "$src")"; [[ "$dst" == gs://* ]] && dst="$(path "$dst")"
    [[ -d "$src" ]] || exit 1; mkdir -p "$dst"; cp -R "$src/." "$dst/" ;;
  "storage ls") exit 1 ;;
  *) echo "fake gcloud: unhandled $*" >&2; exit 9 ;;
esac
"""


def free_port(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


class Step200:
    """One L4 stand-in: directories, fakes on PATH, the step script with its library."""

    def __init__(self, root: Path, plan: dict, tokenizer: bool = True):
        if not (free_port(8777) and free_port(8778)):
            pytest.fail("ports 8777/8778 are in use: step 200's servers need them")
        self.root, self.fake, self.bin = root, root / "fake", root / "bin"
        self.p0d, self.etc, self.hf = root / "phase0", root / "etc" / "vlar", root / "hf"
        self.S = self.p0d / "step2" / COMMIT
        for d in (self.fake / "containers", self.bin, self.etc, root / "weights", root / "var" / "lib" / "vlar",
                  root / "tmp"):
            d.mkdir(parents=True, exist_ok=True)
        if tokenizer:
            self.add_tokenizer()
        self.set_plan(plan)
        py = sys.executable
        shims = {"docker": f'exec "{py}" "{FAKES}" docker "$@"',
                 "nvidia-smi": f'exec "{py}" "{FAKES}" nvidia-smi "$@"',
                 "vlar-ops": f'echo "$*" >> "$FAKE/host-vlar-ops.log"\nexec "{py}" -m vlar_ops.cli "$@"',
                 "sleep": "exec /bin/sleep 0.1"}          # every wait in the script polls; poll faster
        for name, body in shims.items():
            (self.bin / name).write_text(f"#!/bin/bash\n{body}\n")
        (self.bin / "gcloud").write_text(FAKE_GCLOUD)
        for f in self.bin.iterdir():
            f.chmod(0o755)
        self.lib = root / "lib.sh"
        self.lib.write_text(moved(LIB.read_text(), [
            ("/opt/vla-router", str(REPO)), ("/etc/vlar", str(self.etc)), ("/opt/hf", str(self.hf)),
            ("/opt/weights", str(root / "weights")), ("/var/lib/vlar", str(root / "var" / "lib" / "vlar")),
            ("/tmp/vlar-batching.env", str(root / "tmp" / "vlar-batching.env"))]))
        self.step = root / "200-step2-measure.sh"
        self.step.write_text(moved(STEP.read_text(), [
            ("/opt/vla-router/ops/vm/phase0/lib.sh", str(self.lib)), ("/opt/weights", str(root / "weights")),
            ("/opt/hf", str(self.hf)), ("--seconds 120", f"--seconds {LOAD_SECONDS}")]))
        self.runs = 0

    def add_tokenizer(self):
        (self.hf / "hub" / "models--google--paligemma-3b-pt-224").mkdir(parents=True, exist_ok=True)

    def set_plan(self, plan: dict):
        (self.fake / "plan.json").write_text(json.dumps(plan))

    def run(self, timeout: float = 1500) -> tuple[int, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("VLAR_")}
        env.update(PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}", FAKE=str(self.fake),
                   PYTHONPATH=str(REPO / "src"), VLAR_IMAGE_COMMIT=COMMIT, VLAR_PHASE0_DIR=str(self.p0d),
                   VLAR_PHASE0_URL="gs://b/vla-router/phase0", VLAR_STORAGE_URL="gs://b/vla-router",
                   VLAR_REASON_FILE=str(self.root / "reason"))
        for log in ("calls.jsonl", "host-vlar-ops.log"):
            (self.fake / log).write_text("")
        self.runs += 1
        try:
            p = subprocess.run(["bash", str(self.step)], env=env, capture_output=True, text=True, timeout=timeout)
        finally:
            fakes.FAKE = self.fake
            fakes.kill_all()
        out = p.stdout + p.stderr
        (self.root / f"run-{self.runs}.log").write_text(out)
        return p.returncode, out

    def reason(self) -> str:
        return (self.root / "reason").read_text().strip()

    def calls(self) -> list[dict]:
        return [json.loads(x) for x in (self.fake / "calls.jsonl").read_text().splitlines() if x]

    def container_commands(self) -> list[list[str]]:
        """The command of every container the run started (after the image)."""
        out = []
        for c in self.calls():
            a = c["argv"]
            if c["fake"] == "docker" and a[0] == "run":
                i = next(j for j, x in enumerate(a) if x.startswith("vlar-") and ":" in x)
                out.append(a[i + 1:])
        return out

    def json(self, name: str):
        return json.loads((self.S / name).read_text())


def moved(text: str, replacements) -> str:
    """The script with its machine paths moved: each replaced string must occur, and none of the VM's paths may
    remain."""
    for old, new in replacements:
        assert old in text, f"{old!r} no longer in the script: update this test"
        text = text.replace(old, new)
    rest = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))   # code, not comments
    for new in sorted((n for _, n in replacements), key=len, reverse=True):
        rest = rest.replace(new, "")
    for vm_path in ("/opt/", "/etc/vlar", "/var/lib/"):
        assert vm_path not in rest, f"VM path {vm_path} left in the script copy"
    return text


def step2_commands(st: Step200) -> set[str]:
    cmds = {c[2] for c in st.container_commands() if c[:2] == ["vlar-ops", "step2"]}
    host = st.fake / "host-vlar-ops.log"
    if host.exists():
        cmds |= {line.split()[1] for line in host.read_text().splitlines() if line.startswith("step2 ")}
    return cmds


def server_sizes(st: Step200) -> list[int]:
    return [int(c[c.index("--max-batch") + 1]) for c in st.container_commands()
            if c[:3] == ["python", "-m", "vlar_harness.servers.xvla_server"]]


def preseed_obs(st: Step200):
    """Captured requests without the simulator (the commands that read them do not care what they show)."""
    import json_numpy
    import numpy as np
    obs = st.S / "obs"
    obs.mkdir(parents=True)
    (obs / "xvla_payloads.json").write_text(json.dumps([{"proprio": [0.0] * 20, "lang": "t"}] * 2))
    (obs / "pi05_requests.json").write_text(json_numpy.dumps(
        [{"obs": {"pixels": {"image": np.zeros((1, 8, 8, 3), np.uint8)}, "agent_pos": np.zeros((1, 8))},
          "task": "t"}]))


def calibration(st: Step200):
    from vlar_harness.config import load_config
    return load_config(calibration=st.etc / "calibration" / "calibration.yaml")


# ------------------------------------------------------------------------------------------------ the whole step

@requires_libero
def test_step200_end_to_end(tmp_path):
    """Every command of step 2 on a fake L4: sizes 1, 2, 4 fit, 8 leaves under 2 GB free with its 12 workers
    rendering, 16 runs the servers out of memory; the rule picks 4; the calibration file gets the write-once group
    and the rest; a re-run measures nothing again."""
    from vlar_ops import step2
    st = Step200(tmp_path, {"bench": {"sizes": SIZES, "costs": COSTS}, "oom_at": 16})
    rc, out = st.run()
    assert rc == 0, out[-4000:]
    assert step2_commands(st) == {"capture", "render", "load", "memory", "choose", "latency", "upload", "gripper",
                                  "updates"}
    k = int(calibration(st)["measurement"]["K"])

    # the joint memory measurement: each size with its 2 x B workers rendering, sampled while both servers took
    # full batches
    for b in (1, 2, 4, 8):
        m = st.json(f"memory-{b}.json")
        w = step2.workers_for(b)
        assert "error" not in m and m["workers"] == w and m["samples"] >= 4, m
        assert m["peak_used_mib"] == fakes.joint_mib(b, w) and m["total_mib"] == fakes.TOTAL_MIB
        load = st.json(f"load-{b}.json")
        assert load["threads_per_server"] == b and load["calls"]["xvla"] > 0 and load["calls"]["pi05"] > 0
        assert load["servers"]["xvla"]["sample_rows"] == b * k
    m16 = st.json("memory-16.json")
    assert m16["error"] == "servers at size 16 did not come up (exited or not answering /info)"
    assert not (st.S / "load-16.json").exists()

    g = st.json("geometry.json")
    assert (g["max_batch"], g["workers"], g["xvla_sample_rows"]) == (4, 8, 4 * k) and "too_slow" not in g
    assert not g["sizes"]["8"]["feasible"] and "MiB free at the joint peak" in g["sizes"]["8"]["why"]
    assert not g["sizes"]["16"]["feasible"] and "did not come up" in g["sizes"]["16"]["why"]

    lat = st.json("latency.json")
    assert set(lat["median_s"]) == {"xvla_act", "xvla_features", "xvla_sample_k", "pi05_act"}
    assert lat["servers"]["xvla"]["batching"]["max_batch"] == 1 and lat["servers"]["xvla"]["batching"]["sample_rows"] == k
    assert lat["servers"]["pi05"]["batching"]["max_batch"] == 1
    up = st.json("upload.json")
    assert up["requests"] == 10 * step2.UPLOAD_CALLS_PER_TASK and up["upload_image_bytes"] > 0
    gr = st.json("gripper.json")
    assert max(gr["shut_widths_m"]) < gr["shut_width_m"] < gr["open_width_m"] < min(gr["open_widths_m"])

    cfg = calibration(st)
    assert cfg.write_once_complete
    assert cfg["servers"]["batching"]["max_batch"] == 4 and cfg["servers"]["workers"] == 8
    assert cfg["latency"]["pi05_infer_s"] == lat["median_s"]["pi05_act"]
    assert cfg["latency"]["xvla_sample_k_s"] == lat["median_s"]["xvla_sample_k"]
    assert cfg.option("network")["upload_image_bytes"] == up["upload_image_bytes"]
    assert cfg["measurement"]["gripper_thresholds"]["shut_width_m"] == gr["shut_width_m"]
    assert (st.etc / "batching.env").read_text().split()[:2] == ["VLAR_MAX_BATCH=4", f"VLAR_XVLA_SAMPLE_ROWS={4 * k}"]
    bucket = st.fake / "bucket" / "b" / "vla-router"
    assert (bucket / "calibration" / "calibration.yaml").exists()
    assert (bucket / "phase0" / "step2" / COMMIT / "geometry.json").exists()

    rc, out = st.run()                       # the write-once group is set: nothing is measured again
    assert rc == 0 and "write-once group already set" in out, out[-2000:]
    assert step2_commands(st) == set()


# ------------------------------------------------------------------------------------------------ failures

def test_step200_failed_measurements_rule_sizes_out_then_a_rerun_measures_them_again(tmp_path):
    """Size 1: a renderer never starts; 2: one dies during the load; 4: the X-VLA server dies under load; 8: under
    2 GB free. No size is feasible: the step fails (the VM stops). After the cause is gone, a re-run measures the
    failed sizes again, keeps size 8's measurement and finishes."""
    st = Step200(tmp_path, {"bench": {"sizes": [1, 2, 4, 8], "costs": COSTS},
                            "per_size": {"1": "renderer-no-start", "2": "renderer-dies", "4": "server-dies-under-load"}})
    preseed_obs(st)
    (st.S / "upload.json").write_text(json.dumps({"upload_image_bytes": 21000, "requests": 160, "quality": 90}))
    (st.S / "gripper.json").write_text(json.dumps({"shut_width_m": 0.01, "open_width_m": 0.07}))
    rc, out = st.run(timeout=900)
    assert rc == 1 and st.reason() == "geometry rule failed (exit 1)", out[-4000:]
    assert "TooSlow: no feasible batch size" in out
    assert st.json("memory-1.json")["error"] == "only 1 of 2 renderers started"
    assert st.json("memory-2.json")["error"] == "only 3 of 4 renderers still running at the end"
    assert st.json("memory-4.json")["error"] == ("full batches failed at size 4 (a server may have run out of memory); "
                                                 "a server exited during the measurement")
    m8 = (st.S / "memory-8.json").read_text()
    assert "error" not in json.loads(m8) and json.loads(m8)["free_mib"] < 2048
    assert not (st.S / "geometry.json").exists()
    assert not (st.etc / "calibration" / "calibration.yaml").exists()

    st.set_plan({"bench": {"sizes": [1, 2, 4, 8], "costs": COSTS}})
    rc, out = st.run(timeout=900)
    assert rc == 0, out[-4000:]
    assert server_sizes(st) == [1, 2, 4, 1]               # 1, 2, 4 again; 8 kept; then the unpadded servers
    assert (st.S / "memory-8.json").read_text() == m8
    assert all("error" not in st.json(f"memory-{b}.json") for b in (1, 2, 4))
    assert st.json("geometry.json")["max_batch"] == 4 and calibration(st)["servers"]["workers"] == 8
    assert step2_commands(st) == {"render", "load", "memory", "choose", "latency", "updates"}


def test_step200_stops_for_the_user_without_the_tokenizer_or_when_determinism_is_too_slow(tmp_path):
    """Exit 75 (the step stays queued, the VM stops): no gated tokenizer; deterministic batching more than 1.25 x
    the non-deterministic cost (amendment 24), with the numbers in geometry.json and nothing written."""
    st = Step200(tmp_path, {"bench": {"sizes": [1, 2], "costs": COSTS, "slow": 2.0}}, tokenizer=False)
    preseed_obs(st)
    rc, out = st.run(timeout=300)
    assert rc == 75 and "tokenizer" in st.reason(), out[-2000:]
    st.add_tokenizer()
    rc, out = st.run(timeout=600)
    assert rc == 75 and st.reason().startswith("fixed deterministic batching too slow (amendment 24)"), out[-4000:]
    g = st.json("geometry.json")
    assert g["max_batch"] == 2 and g["too_slow"] == "deterministic mode costs 2.00x at batch size 2 (limit 1.25x)"
    assert not (st.S / "latency.json").exists() and not (st.etc / "calibration" / "calibration.yaml").exists()
