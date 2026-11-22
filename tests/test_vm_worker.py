"""The VM worker's startup checks (vlar_ops.vm_worker): the model servers must run the configured fixed batch geometry
with the deterministic settings before any job is taken; options parse strictly. The production factory and a full
run with fake model clients are in tests/reference/test_vm_worker_sim.py (simulator environment)."""

import copy

import pytest

from vlar_harness.config import load_config
from vlar_ops import vm_worker
from vlar_ops.vm_worker import StartupError, check_servers, expected_servers

CFG = load_config()


def good_infos(cfg=CFG):
    b = cfg["servers"]["batching"]
    det = {"deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False}
    return {"xvla": {"model": "xvla", "device": "cuda", "batch_sizes": [1, 2],
                     "batching": {"max_batch": b["max_batch"], "sample_rows": b["xvla_sample_rows"],
                                  "steps": cfg["models"]["xvla"]["flow_steps"], "wait_window_s": 0.005}, **det},
            "pi05": {"model": "pi05", "device": "cuda", "token_length": 200,
                     "batching": {"max_batch": b["max_batch"], "wait_window_s": 0.005}, **det}}


class Client:
    def __init__(self, info, down_for=0):
        self._info, self.down_for, self.calls = info, down_for, 0

    def info(self):
        self.calls += 1
        if self.calls <= self.down_for:
            raise ConnectionError("connection refused")
        return self._info


def test_servers_with_the_configured_geometry_pass():
    infos = good_infos()
    got = check_servers(CFG, {k: Client(v) for k, v in infos.items()}, sleep=lambda s: None)
    assert got == infos


@pytest.mark.parametrize("server,path,value", [
    ("xvla", ("batching", "max_batch"), 1),          # Phase 0 step 2's unpadded latency setting
    ("xvla", ("batching", "sample_rows"), 5),
    ("xvla", ("batching", "steps"), 5),
    ("pi05", ("batching", "max_batch"), 1),
    ("pi05", ("deterministic_algorithms",), False),
    ("xvla", ("cudnn_benchmark",), True),
    ("pi05", ("model",), "xvla"),                     # ports swapped
])
def test_any_difference_stops_the_worker(server, path, value):
    infos = copy.deepcopy(good_infos())
    d = infos[server]
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    with pytest.raises(StartupError, match=f"{server} {'.'.join(path)} is {value!r}"):
        check_servers(CFG, {k: Client(v) for k, v in infos.items()}, sleep=lambda s: None)


def test_a_missing_field_is_a_difference():
    infos = copy.deepcopy(good_infos())
    del infos["xvla"]["batching"]["steps"]
    with pytest.raises(StartupError, match="xvla batching.steps is '<missing>'"):
        check_servers(CFG, {k: Client(v) for k, v in infos.items()}, sleep=lambda s: None)


def test_waits_for_servers_then_gives_up():
    t = [0.0]

    def sleep(s):
        t[0] += s
    infos = good_infos()
    clients = {"xvla": Client(infos["xvla"], down_for=3), "pi05": Client(infos["pi05"])}
    check_servers(CFG, clients, wait_s=60, poll_s=5, sleep=sleep, now=lambda: t[0])
    assert clients["xvla"].calls == 4 and t[0] == 15
    t[0] = 0.0
    with pytest.raises(StartupError, match="pi05 server not answering"):
        check_servers(CFG, {"xvla": Client(infos["xvla"]), "pi05": Client(infos["pi05"], down_for=99)},
                      wait_s=20, poll_s=5, sleep=sleep, now=lambda: t[0])


def test_expected_values_come_from_the_config():
    cfg = load_config()
    e = expected_servers(cfg)
    assert e["xvla"]["batching.max_batch"] == cfg["servers"]["batching"]["max_batch"] == e["pi05"]["batching.max_batch"]
    assert e["xvla"]["batching.sample_rows"] == cfg["servers"]["batching"]["xvla_sample_rows"]


def test_options_parse_strictly():
    assert vm_worker.parse_options(["scene_seed=pi05_parity", "render=all"]) == {"scene_seed": "pi05_parity",
                                                                                 "render": "all"}
    for bad in ("scene_seed", "=x", "scene_seed="):
        with pytest.raises(StartupError):
            vm_worker.parse_options([bad])
