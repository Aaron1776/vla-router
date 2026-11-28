"""Both model servers over HTTP, every endpoint: π0.5 /act, /upload_bytes, /info; X-VLA /act, /sample, /features, /info.

Requests go through the production clients (`servers/clients.py`, what the workers and Phase 0 step 2 use) to the
production apps (`create_app`), in process over FastAPI's test client, so the clients' encoding, the routing and the
handlers' request parsing are what is tested. Each result over HTTP must equal the service's own result for the same
request and seed (timings aside), and a bad request must reach the handler (its 400 with the error), not fail FastAPI's
validation (422). Found by the partner's review (SPEC §17, amendment 45): π0.5's /act and /upload_bytes answered 422 to
every request, because `Request` was imported only inside `create_app` while the module's annotations are strings; no
test called π0.5 over HTTP.

The models are the small ones of the other server tests:
  π0.5 (LeRobot environment):  .venv-sim/bin/python -m pytest tests/reference/test_server_http.py -k pi05
  X-VLA:  VLAR_XVLA_CONFIG_DIR=third_party/hf/X-VLA-Pt VLAR_TEST_DEVICE=cpu \
            .venv-xvla/bin/python -m pytest tests/reference/test_server_http.py -k xvla
starlette's test client needs httpx2 (httpx is a deprecated fallback): the images' test runs install it with hashes
into the throwaway test container (containers/lerobot-test.lock, containers/xvla-test.lock).
"""

import json

import pytest

np = pytest.importorskip("numpy")

from conftest import have  # noqa: E402
from pi05_testkit import PI05_CONFIG_READY, make_tiny_pi05, pi05_request  # noqa: E402
from test_model_servers import cpu_geometry, make_xvla_service, requires_xvla, xvla_payload  # noqa: E402

requires_tiny_pi05 = pytest.mark.skipif(
    not (PI05_CONFIG_READY and have("lerobot") and have("transformers") and have("fastapi")),
    reason="needs LeRobot 0.6.1 with its `pi` extra, FastAPI and the checkpoint's config files "
           "(third_party/hf/pi05_libero_finetuned or VLAR_PI05_CONFIG_DIR)")


class ASGISession:
    """The two calls the production clients make on their `requests.Session`, sent to the app in process."""

    def __init__(self, app):
        from fastapi.testclient import TestClient
        self.http = TestClient(app)

    def post(self, url, json=None, data=None, headers=None, timeout=None):     # no timeout in process
        return self.http.post(url, json=json, content=data, headers=headers)

    def get(self, url, timeout=None):
        return self.http.get(url)


def over_http(client, app):
    client._session = ASGISession(app)
    return client


def untimed(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "timing_s"}


def as_json(value):
    return json.loads(json.dumps(value))


# ---------------------------------------------------------------------------------------------------------- π0.5

@pytest.fixture(scope="module")
def pi05_http():
    from vlar_harness.servers.clients import PI05Client
    from vlar_harness.servers.pi05_server import PI05Batching, create_app
    svc = make_tiny_pi05(PI05Batching(max_batch=3))
    yield svc, over_http(PI05Client(), create_app(svc))
    svc.close()


@requires_tiny_pi05
def test_pi05_act_over_http_equals_the_service(pi05_http):
    svc, client = pi05_http
    req = pi05_request(np.random.default_rng(0), 4, "medium", seed=101)
    got = client.act(req["obs"], req["task"], seed=req["seed"])
    want = svc.act(req)
    assert got["actions"].shape == (50, 7) and np.isfinite(got["actions"]).all()
    assert np.array_equal(got["actions"], np.asarray(want["actions"], dtype=np.float32))
    assert got["batch"] == want["batch"] and set(got) == set(want)


@requires_tiny_pi05
def test_pi05_upload_bytes_over_http_equals_the_service(pi05_http):
    svc, client = pi05_http
    req = pi05_request(np.random.default_rng(1), 8, "short", seed=102)
    got = client.upload_bytes(req["obs"], req["task"], quality=90)
    want = svc.upload_jpeg_bytes({"obs": req["obs"], "task": req["task"], "quality": 90})
    assert got == as_json(want)
    assert got["quality"] == 90 and len(got["jpeg_bytes"]) == len(got["cameras"]) >= 1 and min(got["jpeg_bytes"]) > 0


@requires_tiny_pi05
def test_pi05_info_and_bad_requests_reach_the_handlers(pi05_http):
    from vlar_harness.servers.clients import ServerError
    svc, client = pi05_http
    info = client.info()
    assert info["model"] == "pi05" and info["token_length"] == svc.token_length
    assert info["batching"]["max_batch"] == svc.batching.max_batch
    for call in (lambda: client.act({}, "no observation"), lambda: client.upload_bytes({}, "no observation")):
        with pytest.raises(ServerError, match=r"failed \(400\): \{\"error\""):
            call()


# ---------------------------------------------------------------------------------------------------------- X-VLA

@pytest.fixture(scope="module")
def xvla_http():
    from vlar_harness.servers.clients import XVLAClient
    from vlar_harness.servers.xvla_server import create_app
    svc = make_xvla_service(cpu_geometry())
    yield svc, over_http(XVLAClient(), create_app(svc))
    svc.close()


@requires_xvla
def test_xvla_act_over_http_equals_the_service(xvla_http):
    svc, client = xvla_http
    payload = xvla_payload(np.random.default_rng(2), steps=svc.batching.steps)
    got = client.act(payload, seed=7)
    assert untimed(got) == as_json(untimed(svc.act({**payload, "seed": 7})))
    assert np.asarray(got["action"]).shape == (30, 20)


@requires_xvla
def test_xvla_sample_over_http_equals_the_service(xvla_http):
    svc, client = xvla_http
    payload = xvla_payload(np.random.default_rng(3), steps=svc.batching.steps)
    got = client.sample(payload, seeds=[7, 8, 9])
    assert untimed(got) == as_json(untimed(svc.sample({**payload, "seeds": [7, 8, 9]})))
    assert np.asarray(got["actions"]).shape == (3, 30, 20)


@requires_xvla
def test_xvla_features_over_http_equals_the_service(xvla_http):
    svc, client = xvla_http
    payload = xvla_payload(np.random.default_rng(4), steps=svc.batching.steps)
    got = client.features(payload)
    assert untimed(got) == as_json(untimed(svc.features(payload)))
    assert len(got["features"]["enc_mean"]) == len(got["features"]["wrist_mean"]) == 1024


@requires_xvla
def test_xvla_info_and_bad_requests_reach_the_handlers(xvla_http):
    from vlar_harness.servers.clients import ServerError
    svc, client = xvla_http
    info = client.info()
    assert info["model"] == "xvla" and info["batching"]["max_batch"] == svc.batching.max_batch
    with pytest.raises(ServerError, match=r"failed \(400\): \{\"error\""):
        client.act({})
