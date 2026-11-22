"""The VM worker entry point end to end on the Mac (vlar_ops.vm_worker.serve), with fake model clients in place of
XVLAClient and PI05Client and a real LiberoSim.

An exploration job (task 0, record="full", p_escalate 0.5 so π0.5 is called) runs through `serve`. Its records and
images must equal the same job run by `collect.Runner` with the test factory of test_collect.py: the production
factory wires the X-VLA transport, the measurement client, the state mode and π0.5's actions the same way. The
result marker must carry the provenance header and the worker id, and the staged files must be gone. A routed DAgger
job is refused (it needs a learned decider) without spending an attempt.
"""

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")

from conftest import requires_lerobot
from vlar_harness.config import load_config

pytestmark = requires_lerobot
CFG = load_config()
TASK, INIT = 0, 4


class XVLAFakeClient:
    def __init__(self, info):
        from episode_driver import FakeXVLAService
        self.svc, self._info = FakeXVLAService(), info

    def act(self, payload, seed=None):
        return self.svc.act(payload, seed)

    def sample(self, payload, seeds):
        return self.svc.sample(payload, seeds)

    def features(self, payload):
        return self.svc.features(payload)

    def info(self):
        return self._info


class PI05FakeClient:
    def __init__(self, info):
        from episode_driver import FakePI05
        self.fake, self._info, self.calls = FakePI05(), info, 0

    def act(self, raw_obs, task, seed=None):
        self.calls += 1
        return {"actions": self.fake(raw_obs, task, seed), "timing": {}}

    def info(self):
        return self._info


def infos():
    b = CFG["servers"]["batching"]
    det = {"deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False}
    return {"xvla": {"model": "xvla", "batching": {"max_batch": b["max_batch"], "sample_rows": b["xvla_sample_rows"],
                                                   "steps": CFG["models"]["xvla"]["flow_steps"]}, **det},
            "pi05": {"model": "pi05", "batching": {"max_batch": b["max_batch"]}, **det}}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    from vlar_harness.collect import Runner
    from vlar_harness.sim.libero_sim import LiberoSim, SimSpec
    from vlar_ops.plan import episode_plan
    from vlar_ops.queue import JobQueue
    from vlar_ops.storage import LocalStorage
    from vlar_ops.vm_worker import serve

    tmp = tmp_path_factory.mktemp("vmworker")
    ops = {"local_root": str(tmp / "root"), "storage_url": str(tmp / "bucket")}
    plan = episode_plan("explore", tasks=[TASK], inits=[INIT], namespace=2_000_000, mode="routed", profile="none",
                        record="full", order_seed=7)
    q = JobQueue(tmp / "root" / "queue.sqlite", LocalStorage(tmp / "bucket"), tmp / "root" / "local")
    q.add_plan(plan)
    q.add_plan(episode_plan("dagger", tasks=[TASK], inits=[INIT], namespace=3_000_000, mode="routed",
                            profile="router", record="full", order_seed=7))
    xc, pc = XVLAFakeClient(infos()["xvla"]), PI05FakeClient(infos()["pi05"])
    sims = []

    def make_sim(task_id):
        sims.append(LiberoSim(task_id, SimSpec.from_config(CFG)))
        return sims[-1]
    stats = serve(ops, worker_id="w-test", phases=["explore"], clients={"xvla": xc, "pi05": pc}, make_sim=make_sim,
                  p_escalate=0.5)
    refused = serve(ops, worker_id="w-test", phases=["dagger"], clients={"xvla": xc, "pi05": pc},
                    make_sim=make_sim, p_escalate=0.5)

    # The same job through the Runner with test_collect's factory, into another bucket.
    sim = LiberoSim(TASK, SimSpec.from_config(CFG))

    def make_episode(task_id):
        from episode_driver import FakePI05, FakeXVLAService
        from vlar_harness.adapters.pi05 import PI05Adapter
        from vlar_harness.adapters.xvla import XVLAAdapter
        from vlar_harness.episode import Episode
        svc = FakeXVLAService()
        return Episode(sim, XVLAAdapter(svc.act), PI05Adapter(sim), FakePI05(), CFG, measure_client=svc)
    direct_root = tmp / "direct"
    runner = Runner(CFG, make_episode, lambda k: None, tmp / "direct-staging", p_escalate=0.5)
    for a in runner(plan.jobs[0])[0]:
        dest = direct_root / a["key"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(open(a["path"], "rb").read())
    sim.close()
    yield {"stats": stats, "refused": refused, "bucket": tmp / "bucket", "direct": direct_root, "root": tmp / "root",
           "job": plan.jobs[0], "pi05": pc, "sims": sims, "queue": q}
    q.close()


def test_the_job_is_done_and_the_dagger_job_refused(served):
    assert served["stats"] == {"done": 1, "done_local": 0, "retry": 0, "failed": 0, "refused": 0}
    assert served["refused"]["refused"] == 1 and served["refused"]["done"] == 0
    attempts = dict(served["queue"]._db.execute("SELECT phase, attempts FROM jobs").fetchall())
    assert attempts == {"explore": 0, "dagger": 0} and served["queue"].counts("dagger")["pending"] == 1


def test_records_equal_the_test_factory_run(served):
    from vlar_harness.dataset import TABLES, read_table
    for table in TABLES:
        got, want = read_table(served["bucket"], table), read_table(served["direct"], table)
        assert got.num_rows > 0
        if table == "runs":
            assert got.drop(["config_hash"]).to_pylist() == want.drop(["config_hash"]).to_pylist()
            assert got["config_hash"].to_pylist() == [CFG.hash]
        else:
            assert got.to_pylist() == want.to_pylist(), table
    run = read_table(served["bucket"], "runs").to_pylist()[0]
    assert run["pi05_requests"] > 0 and served["pi05"].calls == run["pi05_requests"]
    key = f"images/{served['job'].subject}.npz"
    assert (served["bucket"] / key).read_bytes() == (served["direct"] / key).read_bytes()


def test_marker_carries_provenance_and_staging_is_cleaned(served):
    from vlar_harness import provenance
    from vlar_harness.snapshot import from_bytes
    job = served["job"]
    marker = json.loads((served["bucket"] / "results" / "explore" / f"{job.job_id}.json").read_text())
    meta = marker["meta"]
    assert meta["worker_id"] == "w-test" and meta["p_escalate"] == 0.5
    assert meta["provenance"]["config_hash"] == CFG.hash
    assert meta["provenance"]["source_digest"] == provenance.source_digest()
    assert meta["servers"]["xvla"]["batching"]["max_batch"] == CFG["servers"]["batching"]["max_batch"]
    snap_keys = [a["key"] for a in marker["result"]["artifacts"] if a["key"].startswith("snapshots/")]
    assert snap_keys
    header = from_bytes((served["bucket"] / snap_keys[0]).read_bytes()).header
    assert header == meta["provenance"]
    staging = served["root"] / "staging"
    assert not any(p.is_file() for p in staging.rglob("*"))
    assert len(served["sims"]) == 1                    # one simulator for the task, closed by serve
