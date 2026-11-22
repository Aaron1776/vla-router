"""Batch invariance of the model servers (SPEC §3, §11 step 3, amendments 24 and 26): a request's output is bit-identical
when sent alone, with different companions, and at different positions in the batch. Batch-mates carry deliberately
different instructions (from different LIBERO-10 tasks) and, for π0.5, robot states with different token lengths, so
the fixed token length is tested along with the fixed batch size; simultaneous requests keep their own prompts and
their own outputs.

X-VLA, on the Mac with random weights on the CPU (a small fixed geometry; the argument does not depend on the sizes):
  VLAR_XVLA_CONFIG_DIR=third_party/hf/X-VLA-Pt VLAR_TEST_DEVICE=cpu \
    .venv-xvla/bin/python -m pytest tests/reference/test_batch_invariance.py -k xvla
π0.5, on the Mac in the simulator environment (LeRobot 0.6.1 with its `pi` extra): LeRobot's own π0.5 code at a tiny
size with random weights, the checkpoint's own processor pipelines and a stand-in tokenizer (`pi05_testkit.py`), and a
stand-in policy for the mechanics:
  .venv-sim/bin/python -m pytest tests/reference/test_batch_invariance.py -k pi05
On the L4 VM the same checks run with the real weights at the production geometry, bit for bit: X-VLA with
VLAR_XVLA_MODEL_PATH, VLAR_XVLA_LORA_PATH and VLAR_TEST_DEVICE=cuda; π0.5 with VLAR_PI05_CHECKPOINT (and the gated
tokenizer) and VLAR_TEST_DEVICE=cuda.
"""

import threading

import pytest

np = pytest.importorskip("numpy")

from conftest import LIBERO_10, have, requires_lerobot, requires_libero  # noqa: E402
from pi05_testkit import PI05_CONFIG_READY, make_tiny_pi05, pi05_request  # noqa: E402
from test_model_servers import (DEVICE, PI05_CKPT, cpu_geometry, make_xvla_service,  # noqa: E402
                                requires_pi05, requires_xvla, xvla_payload)

TOKENS, MASK = "observation.language.tokens", "observation.language.attention_mask"   # LeRobot's batch keys

requires_tiny_pi05 = pytest.mark.skipif(
    not (PI05_CONFIG_READY and have("lerobot") and have("transformers")),
    reason="needs LeRobot 0.6.1 with its `pi` extra and the checkpoint's config files "
           "(third_party/hf/pi05_libero_finetuned or VLAR_PI05_CONFIG_DIR)")


def bits(x) -> bytes:
    return np.asarray(x, dtype=np.float32).tobytes()


def run_together(fn, n):
    """`fn(i)` for i in range(n), each in its own thread, all released at once; (results, errors) by index."""
    barrier = threading.Barrier(n)
    results, errors = {}, {}

    def body(i):
        barrier.wait()
        try:
            results[i] = fn(i)
        except Exception as e:  # noqa: BLE001
            errors[i] = e

    threads = [threading.Thread(target=body, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


@requires_libero
def test_libero10_instructions_match_the_benchmark():
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    assert [suite.get_task(i).language for i in range(10)] == LIBERO_10


# ---------------------------------------------------------------------------------------------------------- X-VLA

# Instructions from different tasks, of different lengths (X-VLA tokens 21, 10, 16, 20, 14, 14 of the fixed 50).
XVLA_TASKS = (4, 8, 3, 6, 0, 2)


@pytest.fixture(scope="module")
def xvla():
    from vlar_harness.servers.common import deterministic_torch
    deterministic_torch()                                  # as XVLAService.load does in production
    svc = make_xvla_service(cpu_geometry())
    yield svc
    svc.close()


def requests(svc, n=3):
    return [dict(xvla_payload(np.random.default_rng(10 + i), steps=svc.batching.steps,
                              instruction=LIBERO_10[XVLA_TASKS[i]]), seed=500 + i) for i in range(n)]


def real_tokens(svc, prepared) -> int:
    return int((prepared["inputs"]["input_ids"] != svc.pad_token_id).sum())


@pytest.fixture(scope="module")
def xvla_six(xvla):
    """The six requests of different tasks (XVLA_TASKS), prepared, and each one's action at every position of a full
    batch of copies of itself. Position 0 is its alone result (alone, the padding rows are copies of it too)."""
    svc = xvla
    payloads = requests(svc, len(XVLA_TASKS))
    prepared = [svc._prepare_request(p, "act") for p in payloads]
    rows = [[o["action"] for o in svc._run_batch("act", [r] * svc.batching.max_batch)] for r in prepared]
    return payloads, prepared, rows


# Batches in which the simultaneous-requests test failed on the Mac before rows were post-processed alone (request 5
# at the last position): captured by logging every batch's composition, replayed here without the batcher's timing.
XVLA_CAPTURED_BATCHES = ((0, 1, 5), (1, 2, 5))


@requires_xvla
def test_xvla_rows_are_post_processed_alone():
    """The authors' post-processing is a sigmoid on the gripper channels. A CPU elementwise kernel computes the last
    (element count mod vector width) elements of its tensor with scalar code, whose exp can differ from the vectorised
    one in the last bit, so over a whole [rows, 30, 2] batch the last row's last values could differ from the same
    values in another row. `denoise` post-processes each row alone: with a stand-in transformer whose raw output is the
    same for every row, its gripper logits chosen so that the two sigmoids differ on this machine, every row at every
    batch size equals the batch of one bit for bit."""
    import types

    import torch
    from vlar_harness import reference
    from vlar_harness.servers.xvla_server import XVLAService
    reference.add_xvla_repo_to_path()
    from models.action_hub import EE6DActionSpace

    g = torch.Generator().manual_seed(0)
    x = torch.randn(4096, generator=g) * 4
    scalar = torch.cat([torch.sigmoid(v.view(1)) for v in x])     # one element per call: the kernel's scalar code
    tricky = x[torch.sigmoid(x) != scalar]                          # the whole tensor at once: its vectorised code
    pool = tricky if len(tricky) else x
    raw = torch.randn(1, 30, 20, generator=g)
    raw[0, :, EE6DActionSpace.gripper_idx] = pool[torch.arange(60) % len(pool)].view(30, 2)

    class SameForEveryRow(torch.nn.Module):                         # stands in for the action transformer
        def forward(self, action_with_noise, **_):
            return raw.expand(action_with_noise.shape[0], -1, -1).clone()

    svc = object.__new__(XVLAService)
    svc.xvla = types.SimpleNamespace(action_space=EE6DActionSpace(), transformer=SameForEveryRow())

    def run(rows):
        return svc.denoise({}, torch.zeros(rows, 20), torch.zeros(rows, dtype=torch.long), 1, torch.zeros(rows, 30, 20))

    one = run(1)[0]
    bad = {rows: [r for r, a in enumerate(run(rows)) if bits(a) != bits(one)] for rows in range(1, 9)}
    assert not any(bad.values()), f"rows that differ from the batch of one, by batch size: {bad}"


@requires_xvla
def test_xvla_fixed_geometry_makes_rows_independent_of_their_batch():
    """The batching mechanics with a stand-in model whose every output row is computed from that row's inputs plus a
    term that depends on the batch's size (as GPU kernel choice can). On the Mac's CPU X-VLA's own rows do not depend
    on the batch's size, so this is the test that catches unpadded encoder or denoiser rows, a batch-level noise
    generator or results handed back in the wrong order; the VM runs the real model. A request's /act and /sample
    results and features are bit-identical alone, with companions and at every position."""
    import types

    import torch
    from vlar_harness import reference
    from vlar_harness.servers.xvla_server import XVLABatching, XVLAService
    reference.add_xvla_repo_to_path()
    from models.action_hub import EE6DActionSpace

    class BatchSizeSensitive(torch.nn.Module):
        num_actions = 30

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))
            self.action_space = EE6DActionSpace()

        def forward_vlm(self, input_ids, pixel_values, image_mask):
            B = input_ids.shape[0]
            img = pixel_values.flatten(2).mean(-1)                                       # [B, views]
            vlm = input_ids.float()[:, :, None] * 1e-2 + img[:, :1, None] + B * 1e-3       # [B, 50, 1]
            aux = img[:, 1:, None].repeat_interleave(3, 1) + B * 1e-3                    # [B, 2 views x 3 tokens, 1]
            return {"vlm_features": vlm, "aux_visual_inputs": aux}

        def transformer(self, domain_id, action_with_noise, proprio, t, vlm_features, aux_visual_inputs):
            B = action_with_noise.shape[0]
            ctx = vlm_features.mean((1, 2)) + aux_visual_inputs.mean((1, 2)) + proprio.sum(1) + t + domain_id
            return action_with_noise * self.scale + ctx[:, None, None] + B * 1e-3

    processor = types.SimpleNamespace(language_max_length=50, tokenizer=types.SimpleNamespace(pad_token_id=1))
    svc = XVLAService(BatchSizeSensitive(), processor, XVLABatching(max_batch=3, sample_rows=6, steps=2))
    g = torch.Generator().manual_seed(0)

    def req(seeds):
        return {"inputs": {"input_ids": torch.randint(2, 100, (1, 50), generator=g),
                           "image_input": torch.rand(1, 3, 3, 4, 4, generator=g),
                           "image_mask": torch.tensor([[True, True, False]]),
                           "proprio": torch.randn(1, 20, generator=g), "domain_id": torch.tensor([3])},
                "seeds": list(seeds)}

    try:
        for kind, key, (x, y, z) in (("act", "action", (req([1]), req([2]), req([3]))),
                                     ("sample", "actions", (req([4, 5]), req([6]), req([7, 8, 9])))):
            ref = svc._run_batch(kind, [x])[0]
            for batch in ([x, y], [y, x], [z, y, x], [y, x, z]):
                got = svc._run_batch(kind, batch)[next(i for i, r in enumerate(batch) if r is x)]
                assert bits(got[key]) == bits(ref[key]), (kind, len(batch))
                assert all(bits(got["features"][f]) == bits(ref["features"][f]) for f in ref["features"]), kind
    finally:
        svc.close()


@requires_xvla
def test_xvla_every_position_computes_the_same_bits(xvla_six):
    """The same request at every position of a full batch gives the same action, for each of the six tasks' requests.
    Before rows were post-processed alone, request 5 differed at the last position in its last gripper value."""
    _, _, rows = xvla_six
    differing = [(i, q) for i, r in enumerate(rows) for q in range(1, len(r)) if bits(r[q]) != bits(r[0])]
    assert not differing, f"(request, position) pairs that differ from position 0: {differing}"


@requires_xvla
def test_xvla_captured_concurrent_batches_replay_to_the_alone_results(xvla, xvla_six):
    svc = xvla
    _, prepared, rows = xvla_six
    for ids in XVLA_CAPTURED_BATCHES:
        out = svc._run_batch("act", [prepared[i] for i in ids])
        bad = [i for i, o in zip(ids, out) if bits(o["action"]) != bits(rows[i][0])]
        assert not bad, f"batch {list(ids)}: requests {bad} differ from their alone results"


@requires_xvla
def test_xvla_act_is_batch_invariant(xvla):
    svc = xvla
    a, b, c = (svc._prepare_request(p, "act") for p in requests(svc))
    assert len({real_tokens(svc, r) for r in (a, b, c)}) == 3, "batch-mates' instructions differ in length"
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
def test_xvla_instruction_length_is_part_of_the_fixed_geometry(xvla):
    """Amendment 26: every instruction is the authors' 50 tokens, padding included (the encoder attends to it, as in
    the authors' model), so a request's padding is its own input; LIBERO-10's instructions all fit with room to spare;
    an instruction that fills all 50 tokens (possibly truncated) is refused."""
    svc = xvla
    base = requests(svc, 1)[0]
    lengths = []
    for text in LIBERO_10:
        ids = svc._prepare_request(dict(base, language_instruction=text), "act")["inputs"]["input_ids"]
        assert tuple(ids.shape) == (1, svc.token_length) == (1, 50)
        lengths.append(int((ids != svc.pad_token_id).sum()))
    assert (min(lengths), max(lengths)) == (10, 21), lengths
    with pytest.raises(ValueError, match="fills all 50 tokens"):
        svc.act(dict(base, language_instruction=" ".join(["word"] * 60)))


@requires_xvla
def test_xvla_simultaneous_requests_keep_their_own_prompts_and_outputs(xvla, xvla_six, monkeypatch):
    svc = xvla
    reqs, prepared, rows = xvla_six
    n = len(reqs)
    alone_ids = [r["inputs"]["input_ids"] for r in prepared]
    assert len({bytes(t.numpy()) for t in alone_ids}) == n
    alone = [r[0] for r in rows]
    got, errors = run_together(lambda i: svc._prepare_request(reqs[i], "act")["inputs"]["input_ids"], n)
    assert not errors, errors
    swapped = [i for i in range(n) if not bool((got[i] == alone_ids[i]).all())]
    assert not swapped, f"requests {swapped} got another request's prompt"
    batches, run_batch = [], svc._run_batch

    def logged(kind, rs):              # each batch's composition (request ids by position), so a failure can be replayed
        batches.append([int(r["seeds"][0]) - 500 for r in rs])
        return run_batch(kind, rs)

    monkeypatch.setattr(svc, "_run_batch", logged)
    monkeypatch.setattr(svc.batchers["act"], "wait_window_s", 2.0)     # long enough that the requests share batches
    results, errors = run_together(lambda i: svc.act(reqs[i]), n)
    assert not errors, errors
    assert sum(map(len, batches)) == n and max(map(len, batches)) > 1, batches
    swapped = [i for i in range(n) if bits(results[i]["action"]) != bits(alone[i])]
    assert not swapped, (f"requests {swapped} got another request's output; batches (request ids by position; the "
                         f"padding rows copy the first): {batches}")


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


# ---------------------------------------------------------------------------------------------------------- π0.5
# The checks are functions so the Mac (tiny model) and the VM (real model, production geometry) run the same ones.

def prompt_tokens(prepared) -> int:
    return int(prepared["observation"][MASK].sum())


def check_pi05_act_invariance(svc, rng):
    """Alone, with companions, at every position (and in a full batch). The batch-mates come from different tasks, with
    prompts of deliberately different instruction and state token lengths."""
    a, b, c = (svc._prepare_request(pi05_request(rng, t, s, seed))
               for t, s, seed in ((4, "long", 101), (8, "short", 102), (3, "medium", 103)))
    la, lb, lc = (prompt_tokens(r) for r in (a, b, c))
    assert la > lc > lb, (la, lb, lc)
    n = svc.batching.max_batch
    fill = [b, c] * n
    batches = {"alone": [a], "with b": [a, b], "second": [b, a], "with c, b": [a, c, b], "last": [c, b, a],
               "full": fill[:n // 2] + [a] + fill[n // 2:n - 1]}
    batches = {name: reqs for name, reqs in batches.items() if len(reqs) <= n}
    out = {name: svc._run_batch(reqs) for name, reqs in batches.items()}
    ref = out["alone"][0]["actions"]
    assert ref.shape == (50, 7) and np.isfinite(ref).all()
    for name, reqs in batches.items():
        got = out[name][next(i for i, r in enumerate(reqs) if r is a)]
        assert bits(got["actions"]) == bits(ref), name
        assert got["batch"]["rows"] == n and got["batch"]["tokens"] == svc.token_length
    b_alone = svc._run_batch([b])[0]["actions"]
    for name in ("with b", "second", "with c, b", "last"):
        if name in out:
            got = out[name][next(i for i, r in enumerate(batches[name]) if r is b)]
            assert bits(got["actions"]) == bits(b_alone), name
    assert bits(b_alone) != bits(ref)


def check_pi05_padding_tokens(svc, rng):
    """The prompt's padding is masked out of attention: other ids in the padding leave the result unchanged bit for
    bit, while changing one real token changes it."""
    import torch
    a = svc._prepare_request(pi05_request(rng, 9, "medium", 104))
    ids, n = a["observation"][TOKENS], prompt_tokens(a)
    length = ids.shape[1]
    pads = ids.clone()
    pads[0, n:] = ids[0, torch.arange(n, length, device=ids.device) % n]         # real ids in the padding
    real = ids.clone()
    real[0, n - 2] = ids[0, 1] if ids[0, 1] != ids[0, n - 2] else ids[0, 2]
    ref = svc._run_batch([a])[0]["actions"]
    with_ids = lambda new: {"observation": dict(a["observation"], **{TOKENS: new}), "seed": a["seed"]}  # noqa: E731
    assert bits(svc._run_batch([with_ids(pads)])[0]["actions"]) == bits(ref)
    assert bits(svc._run_batch([with_ids(real)])[0]["actions"]) != bits(ref)


def check_pi05_prompts_fit(svc, rng) -> list[int]:
    """Every LIBERO-10 instruction with the shortest and the longest state fits the fixed token length without filling
    it (`check_geometry` refuses a full prompt); returns the prompt lengths."""
    return [prompt_tokens(svc._prepare_request(pi05_request(rng, t, s, 1)))
            for t in range(len(LIBERO_10)) for s in ("short", "long")]


def check_pi05_reference_path(svc, rng, max_diff):
    """At max_batch 1 the batched path is the seeded batch of one (`act_unbatched`, the reference path); padded to the
    service's geometry the result may differ in the last bits (recorded)."""
    from vlar_harness.servers.pi05_server import PI05Batching, PI05Service
    one = PI05Service(svc.policy, svc.cfg, svc.preprocessor, svc.postprocessor, svc.env_preprocessor,
                      svc.env_postprocessor, PI05Batching(max_batch=1))
    try:
        req = pi05_request(rng, 6, "long", 105)
        batched_one = one.act(req)["actions"]
        unbatched = svc.act_unbatched(req)["actions"]
        padded = svc.act(req)["actions"]
    finally:
        one.close()
    assert bits(batched_one) == bits(unbatched)
    diff = float(np.abs(padded - unbatched).max())
    print(f"max |padded to {svc.batching.max_batch} - batch of one| = {diff:.3g}")
    assert diff <= max_diff


def check_pi05_concurrency(svc, rng, monkeypatch):
    """Simultaneous requests keep their own prompts and their own outputs. LeRobot's TokenizerProcessorStep reads the
    prompt back from per-call state it stores on the shared step object; a sleep before that read widens the window,
    so without PROCESSING_LOCK simultaneous requests would swap prompts (the mutation check removes the lock)."""
    import time

    import torch
    from lerobot.processor.tokenizer_processor import TokenizerProcessorStep

    specs = [(0, "short"), (4, "long"), (8, "medium"), (3, "short"), (6, "long"), (2, "medium")]
    n = len(specs)
    reqs = [pi05_request(rng, t, s, 200 + i) for i, (t, s) in enumerate(specs)]
    alone_tokens = [svc._prepare_request(r)["observation"][TOKENS] for r in reqs]
    assert len({bytes(t.cpu().numpy()) for t in alone_tokens}) == n
    alone = [svc._run_batch([svc._prepare_request(r)])[0]["actions"] for r in reqs]

    original = TokenizerProcessorStep.observation

    def slow_observation(self, observation):
        time.sleep(0.005)
        return original(self, observation)

    monkeypatch.setattr(TokenizerProcessorStep, "observation", slow_observation)
    for _ in range(2):
        got, errors = run_together(lambda i: svc._prepare_request(reqs[i])["observation"][TOKENS], n)
        assert not errors, errors
        swapped = [i for i in range(n) if not torch.equal(got[i], alone_tokens[i])]
        assert not swapped, f"requests {swapped} got another request's prompt"
    batcher = svc.batcher
    before, window = len(batcher.stats.sizes), batcher.wait_window_s
    batcher.wait_window_s = 2.0                       # long enough that the requests share batches
    try:
        results, errors = run_together(lambda i: svc.act(reqs[i]), n)
    finally:
        batcher.wait_window_s = window
    assert not errors, errors
    sizes = batcher.stats.sizes[before:]
    assert sum(sizes) == n and max(sizes) > 1, sizes
    swapped = [i for i in range(n) if bits(results[i]["actions"]) != bits(alone[i])]
    assert not swapped, f"requests {swapped} got another request's output"


class BatchSensitivePolicy:
    """Stand-in π0.5 policy for the mechanics: each row's chunk comes from its own noise, prompt and image only, but is
    scaled by a factor that depends on the batch size and on the prompt tensor's length, as a GPU kernel's choice can.
    Only the fixed geometry (rows and tokens) and per-request noise make a request's result independent of its batch."""

    def __init__(self, cfg):
        self.config = cfg

    def eval(self):
        return self

    def predict_action_chunk(self, batch, noise=None):
        import torch
        tokens, mask = batch[TOKENS], batch[MASK]
        rows, length = tokens.shape
        if noise is None:
            noise = torch.zeros(rows, self.config.chunk_size, self.config.max_action_dim)
        scale = 1.0 + 1e-3 * rows + 1e-6 * length
        own = (tokens * mask).sum(dim=1).double() * 1e-6 + batch["observation.images.image"].double().mean(dim=(1, 2, 3))
        return noise[:, :, :7] * scale + own.float()[:, None, None]


@pytest.fixture(scope="module")
def pi05_tiny():
    from vlar_harness.servers.pi05_server import PI05Batching
    svc = make_tiny_pi05(PI05Batching(max_batch=3))
    yield svc
    svc.close()


@requires_tiny_pi05
def test_pi05_act_is_batch_invariant(pi05_tiny):
    check_pi05_act_invariance(pi05_tiny, np.random.default_rng(0))


@requires_tiny_pi05
def test_pi05_act_is_batch_invariant_in_bfloat16():
    """The checkpoint's own dtype (bfloat16 with the vision path and norms in float32, as LeRobot sets it). Slow on the
    CPU, so only alone and at the two other positions among batch-mates of different lengths."""
    from vlar_harness.servers.pi05_server import PI05Batching
    svc = make_tiny_pi05(PI05Batching(max_batch=3), dtype="bfloat16")
    try:
        rng = np.random.default_rng(3)
        a, b, c = (svc._prepare_request(pi05_request(rng, t, s, seed))
                   for t, s, seed in ((4, "long", 1), (8, "short", 2), (3, "medium", 3)))
        assert len({prompt_tokens(r) for r in (a, b, c)}) == 3
        ref = svc._run_batch([a])[0]["actions"]
        assert bits(svc._run_batch([b, a, c])[1]["actions"]) == bits(ref)
        assert bits(svc._run_batch([c, b, a])[2]["actions"]) == bits(ref)
    finally:
        svc.close()


@requires_tiny_pi05
def test_pi05_padding_tokens_cannot_affect_results(pi05_tiny):
    check_pi05_padding_tokens(pi05_tiny, np.random.default_rng(4))


@requires_tiny_pi05
def test_pi05_every_libero10_prompt_fits_the_fixed_token_length(pi05_tiny):
    lengths = check_pi05_prompts_fit(pi05_tiny, np.random.default_rng(8))
    assert len(set(lengths)) > 5, lengths              # the stand-in tokenizer's lengths vary as the real one's do


@requires_tiny_pi05
def test_pi05_simultaneous_requests_keep_their_own_prompts_and_outputs(pi05_tiny, monkeypatch):
    check_pi05_concurrency(pi05_tiny, np.random.default_rng(6), monkeypatch)


@requires_tiny_pi05
def test_pi05_a_batch_of_one_unpadded_is_the_reference_path(pi05_tiny):
    check_pi05_reference_path(pi05_tiny, np.random.default_rng(7), max_diff=1e-4)


@requires_tiny_pi05
def test_pi05_fixed_geometry_makes_rows_independent_of_their_batch():
    """Mechanics: a policy whose rows change with the batch size and the prompt tensor's length still gives a request
    the same result alone, with companions and at every position, because every batch has max_batch rows of
    fixed-length prompts and every row its own noise; at another geometry the same request differs."""
    from vlar_harness.servers.pi05_server import PI05Batching, PI05Service
    svc = make_tiny_pi05(PI05Batching(max_batch=3), policy=BatchSensitivePolicy)
    one = PI05Service(svc.policy, svc.cfg, svc.preprocessor, svc.postprocessor, svc.env_preprocessor,
                      svc.env_postprocessor, PI05Batching(max_batch=1))
    try:
        check_pi05_act_invariance(svc, np.random.default_rng(1))
        a = svc._prepare_request(pi05_request(np.random.default_rng(2), 1, "long", 1))
        assert bits(one._run_batch([a])[0]["actions"]) != bits(svc._run_batch([a])[0]["actions"])
    finally:
        one.close()
        svc.close()


@requires_tiny_pi05
def test_pi05_batched_requests_must_fit_the_fixed_geometry(pi05_tiny):
    import torch
    svc = pi05_tiny
    rng = np.random.default_rng(5)
    req = pi05_request(rng, 2, "medium", 9)
    unseeded = {k: v for k, v in req.items() if k != "seed"}
    with pytest.raises(ValueError, match="needs its `seed`"):
        svc._prepare_request(unseeded)
    n = svc.batcher.stats.requests
    out = svc.act(unseeded)                                # the reference behaviour, outside the batcher
    assert svc.batcher.stats.requests == n and "batch" not in out and out["actions"].shape == (50, 7)
    one_camera = dict(req, obs=dict(req["obs"], pixels={"image": req["obs"]["pixels"]["image"]}))
    with pytest.raises(ValueError, match="exactly the cameras"):
        svc.act(one_camera)
    with pytest.raises(ValueError, match=f"fills all {svc.token_length} tokens"):
        svc.act(dict(req, task=" ".join(["word"] * 300)))
    obs = svc._prepare_request(req)["observation"]
    with pytest.raises(ValueError, match=f"exactly {svc.token_length} tokens"):
        svc.check_geometry(dict(obs, **{TOKENS: obs[TOKENS][:, :150], MASK: obs[MASK][:, :150]}))
    with pytest.raises(ValueError, match="right-padded"):
        svc.check_geometry(dict(obs, **{MASK: obs[MASK].flip(1)}))
    with pytest.raises(ValueError, match="batch dimension 1"):
        svc.check_geometry(dict(obs, **{TOKENS: obs[TOKENS].repeat(2, 1)}))
    other = svc._prepare_request(pi05_request(rng, 5, "long", 10))
    odd = {"observation": dict(other["observation"], extra=torch.zeros(1, 1)), "seed": 10}
    with pytest.raises(ValueError, match="same fields"):
        svc._run_batch([svc._prepare_request(req), odd])


@pytest.fixture(scope="module")
def pi05_real():
    from vlar_harness.servers.pi05_server import PI05Service
    svc = PI05Service.load(PI05_CKPT, DEVICE)                # the production geometry
    yield svc
    svc.close()


@requires_pi05
@requires_lerobot
def test_pi05_real_model_is_batch_invariant_on_the_vm(pi05_real, monkeypatch):
    """The real checkpoint, tokenizer and GPU at the production geometry, bit for bit (amendments 24 and 26)."""
    svc = pi05_real
    rng = np.random.default_rng(0)
    lengths = check_pi05_prompts_fit(svc, rng)
    print(f"prompt tokens over LIBERO-10 x short/long states: {min(lengths)}-{max(lengths)} of {svc.token_length}")
    check_pi05_act_invariance(svc, rng)
    check_pi05_padding_tokens(svc, rng)
    check_pi05_reference_path(svc, rng, max_diff=0.05)
    check_pi05_concurrency(svc, rng, monkeypatch)
