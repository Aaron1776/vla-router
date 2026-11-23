"""Phase 0 step 2's simulator-side commands for the joint memory measurement (approved 2026-11-23): `render` (one
planned worker's rendering) on LIBERO, and `load` (full batches to both servers) against stand-in clients."""

import json
import threading

import pytest

from conftest import requires_libero

np = pytest.importorskip("numpy")
json_numpy = pytest.importorskip("json_numpy")


@requires_libero
def test_render_steps_and_renders_every_camera(capsys):
    from vlar_ops import step2
    res = step2.render(task=3, seconds=0.5)
    assert res["task"] == 3 and res["steps"] >= 1
    assert capsys.readouterr().out.count("rendering") == 1


class FakeServers:
    def __init__(self, size, k, fail_after=None):
        self.size, self.k, self.fail_after = size, k, fail_after
        self.calls, self.lock, self.threads = [], threading.Lock(), set()

    def record(self, name, *args):
        with self.lock:
            self.calls.append(name)
            self.threads.add(threading.get_ident())
            if self.fail_after is not None and len(self.calls) > self.fail_after:
                raise ConnectionError("server gone")

    def xvla(self, host, port):
        srv = self

        class X:
            def info(self):
                return {"batching": {"max_batch": srv.size, "sample_rows": srv.size * srv.k}}

            def act(self, p, seed):
                srv.record("act")

            def sample(self, p, seeds):
                assert len(seeds) == srv.k
                srv.record("sample")

            def features(self, p):
                srv.record("features")
        return X()

    def pi05(self, host, port):
        srv = self

        class P:
            def info(self):
                return {"batching": {"max_batch": srv.size}}

            def act(self, obs, task, seed):
                srv.record("pi05")
        return P()


def obs_dir(tmp_path):
    (tmp_path / "xvla_payloads.json").write_text(json.dumps([{"p": 1}, {"p": 2}]))
    (tmp_path / "pi05_requests.json").write_text(json_numpy.dumps([{"obs": {"a": np.zeros(2)}, "task": "t"}]))
    return tmp_path


def test_load_sends_full_batches_to_both_servers(tmp_path, monkeypatch):
    from vlar_harness.servers import clients
    from vlar_ops import step2
    fake = FakeServers(size=4, k=5)
    monkeypatch.setattr(clients, "XVLAClient", fake.xvla)
    monkeypatch.setattr(clients, "PI05Client", fake.pi05)
    res = step2.load(obs_dir(tmp_path), size=4, k=5, seconds=0.3)
    assert res["threads_per_server"] == 4 and len(fake.threads) == 8
    assert {"act", "sample", "features", "pi05"} <= set(fake.calls)
    assert res["calls"]["pi05"] == fake.calls.count("pi05")


def test_load_refuses_another_geometry_and_fails_with_a_server(tmp_path, monkeypatch):
    from vlar_harness.servers import clients
    from vlar_ops import step2
    fake = FakeServers(size=2, k=5)
    monkeypatch.setattr(clients, "XVLAClient", fake.xvla)
    monkeypatch.setattr(clients, "PI05Client", fake.pi05)
    with pytest.raises(RuntimeError, match="not at size 4"):
        step2.load(obs_dir(tmp_path), size=4, k=5, seconds=0.1)
    fake.fail_after = 10
    with pytest.raises(RuntimeError, match="load at size 2 failed: .*server gone"):
        step2.load(obs_dir(tmp_path), size=2, k=5, seconds=5)
