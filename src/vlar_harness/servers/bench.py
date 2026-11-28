"""Phase 0 step 2: GPU time per batch at each candidate geometry (SPEC §11 step 2, §12; amendments 24, 25, 35).

    python -m vlar_harness.servers.bench xvla|pi05 --obs DIR --out bench.json [--sizes 1,2,4,8,16] [--passes 10]

Runs in the model's own image on the L4, with the weights mounted as the servers run them. Loads the model once and
builds a service per geometry around it (as the invariance tests do), then times full batches of real requests
(captured from the simulator by `vlar-ops step2 capture`) through the batched path `_run_batch`, with deterministic
mode on (production) and off (for comparison). A full batch at size n is also the unpadded cost of n requests, so one
table gives both the padded figures (a batch at fixed size B always costs t(B)) and the unpadded ones. Never charged
to the clock: these figures choose the geometry and the worker count (`vlar-ops step2 choose`).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def _set_deterministic(on: bool) -> None:
    import torch
    torch.backends.cudnn.benchmark = not on
    torch.backends.cudnn.deterministic = on
    torch.use_deterministic_algorithms(on)


def _time(fn, passes: int, warmup: int = 2) -> dict:
    import torch
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(passes):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(ts), "min_s": min(ts), "max_s": max(ts), "passes": passes}


def bench_xvla(obs_dir: Path, sizes, passes: int, k: int, weights: str, lora: str) -> dict:
    import torch
    from .common import deterministic_torch
    from .xvla_server import XVLABatching, XVLAService
    deterministic_torch()
    base = XVLAService.load(weights, lora, "cuda")
    payloads = json.loads((obs_dir / "xvla_payloads.json").read_text())
    out = {}
    for b in sizes:
        svc = XVLAService(base.model, base.processor, XVLABatching(max_batch=b, sample_rows=b * k))
        reqs = [payloads[i % len(payloads)] for i in range(b)]
        try:
            prep = {"act": [svc._prepare_request(dict(p, seed=i), "act") for i, p in enumerate(reqs)],
                    "sample": [svc._prepare_request(dict(p, seeds=list(range(k * i, k * i + k))), "sample")
                               for i, p in enumerate(reqs)],
                    "features": [svc._prepare_request(p, "features") for p in reqs]}
            row = {}
            for det in (True, False):
                _set_deterministic(det)
                torch.cuda.reset_peak_memory_stats()
                row["det" if det else "nondet"] = {kind: _time(lambda kind=kind: svc._run_batch(kind, prep[kind]), passes)
                                                   for kind in ("act", "sample", "features")}
                row["det" if det else "nondet"]["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
            out[str(b)] = row
        except torch.cuda.OutOfMemoryError as e:
            out[str(b)] = {"error": f"out of memory: {e}"[:300]}
            torch.cuda.empty_cache()
        finally:
            _set_deterministic(True)
            svc.close()
    base.close()
    return out


def bench_pi05(obs_dir: Path, sizes, passes: int, checkpoint: str) -> dict:
    import json_numpy
    import torch
    from .pi05_server import PI05Batching, PI05Service
    base = PI05Service.load(checkpoint, "cuda")
    items = json_numpy.loads((obs_dir / "pi05_requests.json").read_text())
    out = {}
    for b in sizes:
        svc = PI05Service(base.policy, base.cfg, base.preprocessor, base.postprocessor, base.env_preprocessor,
                          base.env_postprocessor, PI05Batching(max_batch=b))
        try:
            prep = [svc._prepare_request(dict(items[i % len(items)], seed=i)) for i in range(b)]
            row = {}
            for det in (True, False):
                _set_deterministic(det)
                torch.cuda.reset_peak_memory_stats()
                row["det" if det else "nondet"] = {"act": _time(lambda: svc._run_batch(prep), passes),
                                                   "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30}
            out[str(b)] = row
        except torch.cuda.OutOfMemoryError as e:
            out[str(b)] = {"error": f"out of memory: {e}"[:300]}
            torch.cuda.empty_cache()
        finally:
            _set_deterministic(True)
            svc.close()
    base.close()
    return out


def main(argv: list[str] | None = None) -> None:
    import os
    ap = argparse.ArgumentParser(prog="python -m vlar_harness.servers.bench")
    ap.add_argument("model", choices=["xvla", "pi05"])
    ap.add_argument("--obs", required=True, help="directory written by `vlar-ops step2 capture`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="1,2,4,8,16")
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--k", type=int, default=5, help="samples per /sample request (measurement.K)")
    a = ap.parse_args(argv)
    sizes = [int(x) for x in a.sizes.split(",")]
    if a.model == "xvla":
        res = bench_xvla(Path(a.obs), sizes, a.passes, a.k, os.environ["VLAR_XVLA_MODEL_PATH"],
                         os.environ["VLAR_XVLA_LORA_PATH"])
    else:
        res = bench_pi05(Path(a.obs), sizes, a.passes, os.environ["VLAR_PI05_CHECKPOINT"])
    Path(a.out).write_text(json.dumps({"model": a.model, "k": a.k, "sizes": res}, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
