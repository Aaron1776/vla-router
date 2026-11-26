"""The π0.5 gate's startup check (pi05 image, GPU): the platform rule the gate rests on, on this machine, now.

    python gate_check.py --obs DIR --max-batch B --out FILE

On real decision-point requests (`vlar-ops step2 capture`'s pi05_requests.json), every request seeded:
1. a batch of one equals `act_unbatched` bit for bit (`max_batch` 1, the batched path against the reference path);
2. at the calibrated geometry (`max_batch` B), a request alone in its batch gives bit-identical actions on a repeat
   (the property the gate relies on: alone, padded with copies of itself, position 0).
Exit 0 only if both hold for every request.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--obs", required=True); ap.add_argument("--max-batch", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="/weights/pi05_libero_finetuned")
    a = ap.parse_args()
    import json_numpy
    import numpy as np
    from vlar_harness.servers.common import deterministic_torch
    from vlar_harness.servers.pi05_server import PI05Batching, PI05Service
    deterministic_torch()
    base = PI05Service.load(a.checkpoint, "cuda")
    mk = lambda b: PI05Service(base.policy, base.cfg, base.preprocessor, base.postprocessor,  # noqa: E731
                               base.env_preprocessor, base.env_postprocessor, PI05Batching(max_batch=b))
    one, prod = mk(1), mk(a.max_batch)
    items = json_numpy.loads((Path(a.obs) / "pi05_requests.json").read_text())
    rows, ok = [], True
    for i, it in enumerate(items):
        req = dict(it, seed=1000 + i)
        x1 = np.asarray(one.act(dict(req))["actions"]); xu = np.asarray(one.act_unbatched(dict(req))["actions"])
        p1 = np.asarray(prod.act(dict(req))["actions"]); p2 = np.asarray(prod.act(dict(req))["actions"])
        r = {"request": i, "batch_of_one_equals_unbatched": bool(np.array_equal(x1, xu)),
             "alone_repeat_equal": bool(np.array_equal(p1, p2)),
             "max_abs_one_vs_unbatched": float(np.abs(x1 - xu).max())}
        ok &= r["batch_of_one_equals_unbatched"] and r["alone_repeat_equal"]
        rows.append(r)
    res = {"passed": bool(ok), "max_batch": a.max_batch, "requests": len(rows), "rows": rows}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
