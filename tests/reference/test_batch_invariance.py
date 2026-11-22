"""Batch invariance of the model servers (SPEC §3, §11 step 3, amendment 24): a request's output is bit-identical when
sent alone, with different companions, and at different positions in the batch.

X-VLA, on the Mac with random weights on the CPU (a small fixed geometry; the argument does not depend on the sizes):
  VLAR_XVLA_CONFIG_DIR=third_party/hf/X-VLA-Pt VLAR_TEST_DEVICE=cpu \
    .venv-xvla/bin/python -m pytest tests/reference/test_batch_invariance.py -k xvla
On the L4 VM the same tests run with the real weights and the production geometry (VLAR_XVLA_MODEL_PATH,
VLAR_XVLA_LORA_PATH, VLAR_TEST_DEVICE=cuda), bit for bit.
"""

import threading

import pytest

np = pytest.importorskip("numpy")

from test_model_servers import cpu_geometry, make_xvla_service, requires_xvla, xvla_payload  # noqa: E402


def bits(x) -> bytes:
    return np.asarray(x, dtype=np.float32).tobytes()


@pytest.fixture(scope="module")
def xvla():
    from vlar_harness.servers.common import deterministic_torch
    deterministic_torch()                                  # as XVLAService.load does in production
    svc = make_xvla_service(cpu_geometry())
    yield svc
    svc.close()


def requests(svc, n=3):
    return [dict(xvla_payload(np.random.default_rng(10 + i), steps=svc.batching.steps), seed=500 + i)
            for i in range(n)]


@requires_xvla
def test_xvla_act_is_batch_invariant(xvla):
    svc = xvla
    a, b, c = (svc._prepare_request(p, "act") for p in requests(svc))
    batches = {"alone": [a], "with b": [a, b], "with c, b": [a, c, b], "second": [b, a], "last": [c, b, a]}
    out = {name: svc._run_batch("act", reqs) for name, reqs in batches.items()}
    pos = {name: next(i for i, r in enumerate(reqs) if r is a) for name, reqs in batches.items()}
    ref = out["alone"][0]
    assert np.asarray(ref["action"]).shape == (30, 20)
    for name in batches:
        got = out[name][pos[name]]
        assert bits(got["action"]) == bits(ref["action"]), name
        assert all(bits(got["features"][k]) == bits(ref["features"][k]) for k in ref["features"]), name
        assert got["batch"]["encoder_rows"] == got["batch"]["denoise_rows"] == svc.batching.max_batch
    b_alone = svc._run_batch("act", [b])[0]["action"]
    for got in (out["with b"][1], out["with c, b"][2], out["second"][0], out["last"][1]):
        assert bits(got["action"]) == bits(b_alone)
    assert bits(b_alone) != bits(ref["action"])


@requires_xvla
def test_xvla_sample_is_batch_invariant_and_features_agree_across_endpoints(xvla):
    svc = xvla
    pa, pb, pc = requests(svc)
    k = svc.batching.sample_rows
    sa = svc._prepare_request(dict(pa, seeds=[1, 2]), "sample")
    sb = svc._prepare_request(dict(pb, seeds=[3]), "sample")
    sc = svc._prepare_request(dict(pc, seeds=list(range(4, 4 + k - 3))), "sample")      # fills the rows
    batches = {"alone": [sa], "with b": [sa, sb], "second": [sb, sa], "last": [sc, sb, sa], "middle": [sb, sa, sc]}
    out = {name: svc._run_batch("sample", reqs) for name, reqs in batches.items()}
    pos = {name: next(i for i, r in enumerate(reqs) if r is sa) for name, reqs in batches.items()}
    ref = out["alone"][0]
    assert np.asarray(ref["actions"]).shape == (2, 30, 20)
    for name in batches:
        got = out[name][pos[name]]
        assert bits(got["actions"]) == bits(ref["actions"]), name
        assert all(bits(got["features"][f]) == bits(ref["features"][f]) for f in ref["features"]), name
    # The encoder geometry is the same for every endpoint, so the pooled features agree bit for bit (§3).
    act = svc._run_batch("act", [svc._prepare_request(pa, "act")])[0]["features"]
    feats = svc._run_batch("features", [svc._prepare_request(pa, "features")])[0]["features"]
    for f in ref["features"]:
        assert bits(act[f]) == bits(feats[f]) == bits(ref["features"][f])


@requires_xvla
def test_xvla_concurrent_clients_get_their_alone_results(xvla):
    svc = xvla
    reqs = requests(svc)
    alone = [svc._run_batch("act", [svc._prepare_request(p, "act")])[0]["action"] for p in reqs]
    batcher = svc.batchers["act"]
    before, window = batcher.stats.batches, batcher.wait_window_s
    batcher.wait_window_s = 2.0                       # long enough that all three arrive in one batch
    results = {}
    try:
        ts = [threading.Thread(target=lambda i=i, p=p: results.__setitem__(i, svc.act(p))) for i, p in enumerate(reqs)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        batcher.wait_window_s = window
    assert batcher.stats.batches == before + 1 and batcher.stats.sizes[-1] == 3
    for i in range(3):
        assert bits(results[i]["action"]) == bits(alone[i]) and results[i]["batch"]["requests"] == 3


@requires_xvla
def test_xvla_batched_requests_must_fit_the_fixed_geometry(xvla):
    from vlar_harness.servers.batching import BatchTooLarge
    svc = xvla
    p = requests(svc, 1)[0]
    with pytest.raises(ValueError, match="exactly image0"):
        svc.act(dict(p, image2=p["image1"]))
    with pytest.raises(ValueError, match="flow-matching steps"):
        svc.act(dict(p, steps=svc.batching.steps + 1))
    with pytest.raises(BatchTooLarge):
        svc.sample(dict(p, seeds=list(range(svc.batching.sample_rows + 1))))
    n = svc.batchers["act"].stats.requests
    unseeded = svc.act({k: v for k, v in p.items() if k != "seed"})     # the reference behaviour, outside batches
    assert svc.batchers["act"].stats.requests == n and "batch" not in unseeded


@requires_xvla
def test_xvla_a_batch_of_one_unpadded_is_the_reference_path(xvla):
    """At max_batch 1 the batched path is the seeded batch-of-one computation, which test_model_servers.py shows is the
    authors' handler's. At the test geometry the same request differs from it only in the last bits: what fixed
    padding is for (recorded, not a requirement)."""
    from vlar_harness.servers.xvla_server import XVLABatching, XVLAService
    svc = xvla
    one = XVLAService(svc.model, svc.processor, XVLABatching(max_batch=1, sample_rows=1, steps=svc.batching.steps))
    try:
        p = requests(svc, 1)[0]
        batched_one = one.act(p)["action"]
        unbatched = svc.act_unbatched(p)["action"]
        padded = svc.act(p)["action"]
    finally:
        one.close()
    assert bits(batched_one) == bits(unbatched)
    diff = float(np.abs(np.asarray(padded) - np.asarray(unbatched)).max())
    print(f"max |padded to {svc.batching.max_batch} - batch of one| = {diff:.3g}")
    assert diff < 1e-4
