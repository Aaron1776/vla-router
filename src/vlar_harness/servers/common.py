"""Shared by both model servers: noise from per-call seeds, timing and provenance.

Noise. When a request carries seeds, each sample's initial noise is drawn on the CPU from its
own generator seeded with that sample's seed, then moved to the GPU. A sample's noise therefore
depends only on its own seed: not on the batch it lands in, on other requests, or on the GPU
model. Without seeds the server falls back to the reference behaviour (global RNG on the device).
How seeds are derived from (episode seed, step, model, call type, sample index) is the
harness's job, not the server's.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence


def seeded_noise(seeds: Sequence[int], shape: tuple[int, ...], device, kind: str = "randn"):
    """Stack of per-seed standard-normal tensors of `shape`, float32, on `device`.

    `kind` follows each reference: X-VLA draws with `torch.randn`, LeRobot's π0.5 with
    `torch.normal(mean=0, std=1)`.
    """
    import torch

    out = []
    for s in seeds:
        g = torch.Generator(device="cpu").manual_seed(int(s))
        if kind == "randn":
            out.append(torch.randn(shape, generator=g, dtype=torch.float32))
        elif kind == "normal":
            out.append(torch.normal(mean=0.0, std=1.0, size=shape, generator=g, dtype=torch.float32))
        else:
            raise ValueError(kind)
    return torch.stack(out, dim=0).to(device)


class Timer:
    """Wall-clock spans of one request, with CUDA synchronisation so GPU work is attributed correctly."""

    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.spans: dict[str, float] = {}

    def _sync(self) -> None:
        if self.cuda:
            import torch
            torch.cuda.synchronize()

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        self._sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.spans[name] = self.spans.get(name, 0.0) + time.perf_counter() - t


def deterministic_torch() -> None:
    """Deterministic kernel selection for batch invariance (SPEC §3, amendment 24): no cuDNN autotuning (it can pick
    another algorithm after a restart), cuDNN's deterministic algorithms, PyTorch's deterministic mode, and the cuBLAS
    workspace setting that mode needs. Call before the model is loaded and before any CUDA work. TF32 settings are
    left as each reference sets them: they change precision, not determinism."""
    import os

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def torch_flags() -> dict:
    import os

    import torch

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def peak_memory_gb(device) -> Optional[float]:
    import torch

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    return None
