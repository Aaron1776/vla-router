"""How the batch-invariance tests compare a request's batched output with its solo result (SPEC §17 amendments 24, 33).

Bit for bit, as amendment 24 requires, everywhere except one place: inside an image on the CPU (VLAR_IN_IMAGE=1 and
VLAR_TEST_DEVICE=cpu, the build VM's in-container tests). Different x86 CPUs take different code paths in PyTorch's CPU
math libraries (vector width, blocking), so there a batch can change the last bits of a row (the fourth and fifth image
builds ran unchanged code on two build VMs: one bit-identical, one a few units in the last place apart). There an output
that is not bit-identical is a warning with the measured difference, and the checks that block are:
  - within PyTorch's own default tolerance for the computation's dtype (torch.testing.assert_close: float32 relative
    1.3e-6, about 11 units in the last place, absolute 1e-5; bfloat16 relative 1.6e-2, absolute 1e-5), and
  - closer to the request's own solo result than to any other request's, so a swapped output still fails loudly.
On the Mac and on the L4 (VLAR_TEST_DEVICE=cuda) a difference fails, with the same measurements in the message. The L4's
bit-for-bit checks are the ones the project depends on and stay blocking (amendment 24).
"""

import os
import warnings

import numpy as np

IN_IMAGE_CPU = os.environ.get("VLAR_IN_IMAGE") == "1" and os.environ.get("VLAR_TEST_DEVICE") == "cpu"
TOLERANCE = {"float32": (1.3e-6, 1e-5), "bfloat16": (1.6e-2, 1e-5)}     # torch.testing.assert_close's defaults


class NotBitIdentical(UserWarning):
    """An output that differs from its solo result only within the tolerance, on an image's CPU."""


def bits(x) -> bytes:
    return np.asarray(x, dtype=np.float32).tobytes()


def measure(got, ref) -> dict:
    """Max absolute and relative difference, how many values differ, and the first that does."""
    g, r = np.asarray(got, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    if g.shape != r.shape:
        return {"shape": (g.shape, r.shape)}
    d = np.abs(g - r)
    rel = d / np.maximum(np.abs(r), np.finfo(np.float32).tiny)
    nz = np.flatnonzero(d.ravel())
    return {"max_abs": float(d.max(initial=0.0)), "max_rel": float(rel.max(initial=0.0)),
            "differing": int(nz.size), "size": int(d.size), "first": int(nz[0]) if nz.size else None}


def describe(m: dict) -> str:
    if "shape" in m:
        return f"shapes differ: {m['shape'][0]} vs {m['shape'][1]}"
    return (f"max |diff| {m['max_abs']:.3g}, max relative {m['max_rel']:.3g}, {m['differing']} of {m['size']} values "
            f"differ (first at flat index {m['first']})")


def max_abs(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def exact(got, ref, what: str) -> None:
    """Bit for bit everywhere (mechanics and reference-path checks); the message measures any difference."""
    if bits(got) != bits(ref):
        raise AssertionError(f"{what}: not bit-identical ({describe(measure(got, ref))})")


def matches_solo(got, ref, what: str, others=(), dtype: str = "float32") -> None:
    """A request's batched output against its solo result `ref` (module docstring). `others` are (name, result) pairs
    of other requests' solo results of the same shape, for the closeness check."""
    if bits(got) == bits(ref):
        return
    m = measure(got, ref)
    msg = f"{what}: not bit-identical to its solo result ({describe(m)})"
    if not IN_IMAGE_CPU or "shape" in m:
        raise AssertionError(msg)
    rtol, atol = TOLERANCE[dtype]
    g, r = np.asarray(got, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    if not np.all(np.abs(g - r) <= atol + rtol * np.abs(r)):
        raise AssertionError(f"{msg}; beyond the {dtype} tolerance (relative {rtol}, absolute {atol})")
    own = max_abs(got, ref)
    for name, other in others:
        d = max_abs(got, other)
        if not own < d:
            raise AssertionError(f"{what}: as close to {name}'s solo result (max |diff| {d:.3g}) as to its own "
                                 f"({own:.3g}): another request's output")
    warnings.warn(f"{msg}; within the {dtype} tolerance and closest to its own solo result (image CPU)",
                  NotBitIdentical, stacklevel=2)


def all_match_solo(cases) -> None:
    """`matches_solo(*case)` for every case, reporting all failing cases together."""
    failures = []
    for case in cases:
        try:
            matches_solo(*case)
        except AssertionError as e:
            failures.append(str(e))
    if failures:
        raise AssertionError(f"{len(failures)} of {len(cases)} checks failed:\n" + "\n".join(failures))
