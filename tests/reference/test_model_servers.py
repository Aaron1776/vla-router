"""Both model servers against their reference code paths, with the same initial noise.

These need the weights and (in practice) a GPU, so they run on the VM:

  X-VLA environment:   VLAR_XVLA_MODEL_PATH=<X-VLA-Pt snapshot> VLAR_XVLA_LORA_PATH=<adapter snapshot> \
                       python -m pytest tests/reference/test_model_servers.py -k xvla
  LeRobot environment: VLAR_PI05_CHECKPOINT=<pi05_libero_finetuned snapshot> \
                       python -m pytest tests/reference/test_model_servers.py -k pi05

Equality is exact where the computation is the same sequence of kernels. /act and /sample go through the
batchers at a fixed geometry (amendment 24); batch invariance itself is tested in test_batch_invariance.py.
"""

import os

import pytest

np = pytest.importorskip("numpy")

from conftest import have, requires_lerobot

XVLA_MODEL = os.environ.get("VLAR_XVLA_MODEL_PATH")
XVLA_LORA = os.environ.get("VLAR_XVLA_LORA_PATH")
# Without weights: a randomly initialised X-VLA built from the checkpoint's config, processor and tokenizer
# files. Enough to check that the server's code path equals the authors' (not to check behaviour).
XVLA_CONFIG_DIR = os.environ.get("VLAR_XVLA_CONFIG_DIR")
PI05_CKPT = os.environ.get("VLAR_PI05_CHECKPOINT")
DEVICE = os.environ.get("VLAR_TEST_DEVICE", "cuda")

requires_xvla = pytest.mark.skipif(not ((XVLA_MODEL or XVLA_CONFIG_DIR) and have("peft") and have("fastapi")),
                                   reason="needs the X-VLA environment and VLAR_XVLA_MODEL_PATH or VLAR_XVLA_CONFIG_DIR")
requires_pi05 = pytest.mark.skipif(not PI05_CKPT, reason="needs the LeRobot environment and VLAR_PI05_CHECKPOINT")


def xvla_payload(rng, steps=10, instruction="put both moka pots on the stove"):
    import json_numpy
    img0 = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    img1 = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    proprio = np.concatenate([rng.normal(size=9), [0.0], np.zeros(10)])
    return {"proprio": json_numpy.dumps(proprio), "language_instruction": instruction,
            "image0": json_numpy.dumps(np.flip(np.flip(img0, 0), 1)), "image1": json_numpy.dumps(img1),
            "domain_id": 3, "steps": steps}


def make_xvla_service(batching=None):
    """The real model (VLAR_XVLA_MODEL_PATH) or a randomly initialised one from the config files."""
    from vlar_harness.servers.xvla_server import XVLAService
    if XVLA_MODEL:
        return XVLAService.load(XVLA_MODEL, XVLA_LORA, DEVICE, batching=batching)
    import torch
    from vlar_harness import reference
    from vlar_harness.servers.common import deterministic_torch
    reference.add_xvla_repo_to_path()
    from models.configuration_xvla import XVLAConfig
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor
    deterministic_torch()                    # as load() does; uninitialised memory is then NaN, not leftovers
    torch.manual_seed(0)
    model = XVLA(XVLAConfig.from_pretrained(XVLA_CONFIG_DIR)).to(DEVICE).to(torch.float32)
    # Florence-2's `image_projection` is a raw torch.empty Parameter that random init never touches (a checkpoint
    # overwrites it); give every such tensor seeded values so the random model is fully defined.
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for t in list(model.parameters()) + list(model.buffers()):
            if t.is_floating_point() and not torch.isfinite(t).all():
                t.copy_(torch.randn(t.shape, generator=g, dtype=t.dtype) * 0.02)
    return XVLAService(model, XVLAProcessor.from_pretrained(XVLA_CONFIG_DIR), batching)


def cpu_geometry():
    """A small fixed geometry for the random-weight CPU runs (the invariance argument does not depend on the
    sizes); the VM runs use the production geometry."""
    from vlar_harness.servers.xvla_server import XVLABatching
    return XVLABatching(max_batch=3, sample_rows=6, steps=10) if DEVICE == "cpu" else None


@pytest.fixture(scope="module")
def xvla_service():
    svc = make_xvla_service(cpu_geometry())
    yield svc
    svc.close()


@requires_xvla
def test_xvla_act_matches_authors_handler(xvla_service):
    """Our encode+denoise against the authors' own FastAPI /act handler on the same model object."""
    import torch
    from fastapi.testclient import TestClient

    svc = xvla_service
    svc.xvla.app = None
    svc.xvla._build_app(svc.processor)                     # the authors' app (modeling_xvla.py:219-286)
    ref_client = TestClient(svc.xvla.app)
    payload = xvla_payload(np.random.default_rng(0))

    torch.manual_seed(1234)
    ref = np.array(ref_client.post("/act", json=payload).json()["action"], dtype=np.float32)

    torch.manual_seed(1234)                                 # the handler draws its noise right after the encoder
    with torch.no_grad():
        inputs, steps = svc.prepare(payload)
        enc = svc.encode(inputs)
        x1 = torch.randn(1, svc.num_actions, svc.dim_action, device=inputs["proprio"].device,
                         dtype=inputs["proprio"].dtype)
        ours = svc.denoise(enc, inputs["proprio"], inputs["domain_id"], steps, x1).squeeze(0).float().cpu().numpy()
    assert ref.shape == (30, 20)
    assert np.array_equal(ref, ours), f"max abs diff {np.abs(ref - ours).max()}"


@requires_xvla
def test_xvla_seeds_features_and_sample(xvla_service):
    svc = xvla_service
    payload = xvla_payload(np.random.default_rng(1), steps=svc.batching.steps)
    a1 = svc.act({**payload, "seed": 7})
    a2 = svc.act({**payload, "seed": 7})
    a3 = svc.act({**payload, "seed": 8})
    assert a1["action"] == a2["action"] and a1["action"] != a3["action"]
    f = svc.features(payload)["features"]
    assert f["enc_mean"] == a1["features"]["enc_mean"] and f["wrist_mean"] == a1["features"]["wrist_mean"]
    assert len(f["enc_mean"]) == len(f["wrist_mean"]) == 1024
    s = svc.sample({**payload, "seeds": [7, 8, 9]})
    acts = np.array(s["actions"], dtype=np.float32)
    assert acts.shape == (3, 30, 20)
    for row, single in ((0, a1), (1, a3)):
        diff = np.abs(acts[row] - np.array(single["action"], dtype=np.float32)).max()
        assert diff < 1e-3, f"batched sample {row} differs from /act by {diff}"


@requires_pi05
@requires_lerobot
def test_pi05_server_matches_lerobot_eval_step():
    """The server's first 10 actions against lerobot-eval's own per-step ops (select_action +
    post-processors) on LeRobot's own env observation, with the policy's noise pinned. The server side is the seeded
    batch of one (`act_unbatched`); test_batch_invariance.py ties the batched path to it at max_batch 1."""
    import torch
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
    from lerobot.utils.constants import ACTION

    from vlar_harness.servers.common import seeded_noise
    from vlar_harness.servers.pi05_server import PI05Service

    svc = PI05Service.load(PI05_CKPT, DEVICE)
    vec = LiberoEnvConfig(task="libero_10", task_ids=[0]).create_envs(n_envs=1, use_async_envs=False)["libero_10"][0]
    obs, _ = vec.reset(seed=[1000], options={NEW_ROLLOUT_OPTION: True})
    task = list(vec.call("task_description"))
    try:
        shape = (svc.cfg.chunk_size, svc.cfg.max_action_dim)
        pinned = seeded_noise([7], shape, svc.device, "normal")
        policy = svc.policy
        policy.reset()
        original = policy.model.sample_noise
        policy.model.sample_noise = lambda s, d: pinned
        ref = []
        with torch.no_grad():
            for _ in range(10):                                  # lerobot_eval.py:273-300, queue refills every 10
                o = preprocess_observation(obs)
                o["task"] = task
                o = svc.preprocessor(svc.env_preprocessor(o))
                with torch.inference_mode():
                    a = policy.select_action(o)
                a = svc.env_postprocessor({ACTION: svc.postprocessor(a)})[ACTION]
                ref.append(a.to("cpu").numpy())
        policy.model.sample_noise = original
        out = svc.act_unbatched({"obs": obs, "task": task[0], "seed": 7})
        assert out["actions"].shape == (50, 7)
        assert np.array_equal(out["actions"][:10], np.concatenate(ref, axis=0))
    finally:
        vec.close()
        svc.close()
