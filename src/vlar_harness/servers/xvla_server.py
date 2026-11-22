"""X-VLA server: the authors' model, processor and request handling, with three endpoints
(docs/SPEC.md §3). Runs in the X-VLA environment (transformers 4.51.3, peft 0.17.1, timm 1.0.12,
accelerate 1.2.1) with the authors' checkout on the path.

Loading is `deploy.py` (lines 63-98): `XVLAProcessor.from_pretrained(base)`, `XVLA.from_pretrained(
base, torch_dtype=float32)` in float32, then the LoRA adapter with `PeftModel.from_pretrained`.
Request preprocessing is the authors' `/act` handler (models/modeling_xvla.py:233-275): images
decoded from json_numpy into PIL, the processor (CLIP-style bicubic resize to 224, ImageNet
normalisation, BART tokens padded to 50), proprio cast to the model dtype, `domain_id`.

Generation is the authors' `generate_actions` (modeling_xvla.py:181-216) split in two so the
encoder pass is shared: `encode()` is its `forward_vlm` call and `denoise()` is its loop, run
from a given initial noise. With the same noise the result is identical to `generate_actions`
(tests/reference/test_xvla_server.py).

Endpoints (JSON; arrays as json_numpy strings, as the authors' client sends them):
  POST /act       one chunk (30 x 20) + pooled features. Only these chunks are ever executed.
  POST /sample    `seeds` -> one chunk per seed for the same request, batched, + pooled features.
  POST /features  encoder pass only -> pooled features.
Optional request fields beyond the authors' format: `seed` (/act), `seeds` (/sample).

Batching (effort step 5, SPEC §3, §12, amendment 24). Seeded requests (every request the harness sends) and every
/features request go through one `Batcher` per endpoint, and every batch runs at a fixed geometry (`XVLABatching`):
the encoder on exactly `max_batch` rows, /act's denoising on `max_batch` rows, /sample's on `sample_rows` rows (a
request's samples are consecutive rows), padded with copies of the batch's first request; each row's initial noise
comes from its own seed's CPU generator. A request's result is therefore bit-identical whichever requests share its
batch, however many, and wherever it sits; the encoder geometry is the same for all three endpoints, so their pooled
features agree bit for bit. Batched requests must carry exactly the two views the harness sends (main, wrist), so the
image encoder always sees 2 x max_batch images, and the configured number of flow-matching steps. The instruction's
token length is part of the geometry too (amendment 26): the authors' processor pads every instruction to 50 tokens,
which the encoder attends to as in the authors' model, so a request's padding is its own input and never depends on
its batch-mates; an instruction that fills all 50 (possibly truncated) is refused. The authors' post-processing runs
on each row alone (`denoise`), so no elementwise kernel treats a request's values differently by position. A request
without a seed keeps the reference behaviour (global RNG, a batch of one, unpadded) outside the batchers.

Pooled features [ASSUMPTION in the spec]: `enc_mean` is the mean over all encoder output tokens
(first image's tokens and the 50 text tokens, padding included, as the encoder attends to all
of them); `wrist_mean` is the mean over the wrist view's image tokens, which reach the action
transformer directly rather than through the encoder. Both 1024 floats.
"""

from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .. import reference
from .batching import Batcher, pad_rows
from .common import Timer, deterministic_torch, peak_memory_gb, seeded_noise, torch_flags

log = logging.getLogger("vlar.xvla_server")

# Shared per-call state (audited for amendment 24; docs/HARNESS.md "Batching"): the authors' processor and model keep
# none, but the processor's Hugging Face fast tokenizer holds mutable padding and truncation settings inside a shared
# Rust object, which concurrent calls can trip over. Every processor call runs under this lock (requests are
# pre-processed in their callers' threads); the model runs under the service's device lock, one batch at a time.
# Lock order: device lock, then PROCESSING_LOCK, never the reverse.
PROCESSING_LOCK = threading.Lock()


@dataclass(frozen=True)
class XVLABatching:
    """The fixed batch geometry (amendment 24); placeholders until Phase 0 step 2 measures the padding cost."""
    max_batch: int = 8               # requests per batch: rows of every encoder pass and every /act denoising pass
    sample_rows: int = 40            # rows of every /sample denoising pass (K = 5 samples per request)
    wait_window_s: float = 0.005
    steps: int = 10                  # flow-matching steps of every batched request (the reference's 10)

    def __post_init__(self):
        if min(self.max_batch, self.sample_rows, self.steps) < 1 or self.wait_window_s < 0:
            raise ValueError(f"invalid batching {self}")


class XVLAService:
    @classmethod
    def load(cls, model_path: str, lora_path: Optional[str], device: str = "cuda",
             processor_path: Optional[str] = None, batching: Optional[XVLABatching] = None) -> "XVLAService":
        """deploy.py:63-98, after the deterministic settings (amendment 24)."""
        import torch

        deterministic_torch()
        reference.add_xvla_repo_to_path()
        from models.modeling_xvla import XVLA            # the authors' classes, from the pinned checkout
        from models.processing_xvla import XVLAProcessor

        processor = XVLAProcessor.from_pretrained(processor_path or model_path)
        model = XVLA.from_pretrained(model_path, trust_remote_code=True,
                                     torch_dtype=torch.float32).to(device).to(torch.float32)
        if lora_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float32).to(device)
        return cls(model, processor, batching)

    def __init__(self, model, processor, batching: Optional[XVLABatching] = None):
        self.processor = processor
        self.model = model
        # deploy.py calls `model.run`, which PeftModel forwards to the wrapped XVLA (LoRA layers injected
        # in place); get_base_model() returns that same object.
        self.xvla = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.xvla.eval()
        p = next(self.xvla.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.num_actions = int(self.xvla.num_actions)
        self.dim_action = int(self.xvla.action_space.dim_action)
        self.lock = threading.Lock()   # one batch (or unseeded request) on the device at a time
        self.batching = b = batching or XVLABatching()
        # Amendment 26: the instruction's token length is part of the fixed geometry (the authors' processor pads to
        # it; the encoder attends to the padding, so it is part of each request's own input, never its batch-mates').
        self.token_length = int(processor.language_max_length)
        self.pad_token_id = int(processor.tokenizer.pad_token_id)

        def batcher(kind: str, capacity: tuple, cost=lambda r: (1,)):
            return Batcher(f"xvla-{kind}", prepare=lambda p: self._prepare_request(p, kind),
                           run=lambda reqs: self._run_batch(kind, reqs), capacity=capacity, cost=cost,
                           wait_window_s=b.wait_window_s)

        self.batchers = {"act": batcher("act", (b.max_batch,)),
                         "sample": batcher("sample", (b.max_batch, b.sample_rows), lambda r: (1, len(r["seeds"]))),
                         "features": batcher("features", (b.max_batch,))}

    def close(self) -> None:
        for bt in self.batchers.values():
            bt.close()

    # ------------------------------------------------------------ preprocessing (authors' handler)

    def prepare(self, payload: dict[str, Any]) -> tuple[dict, int]:
        """The authors' preprocessing, on the model's device."""
        inputs, steps = self.process(payload)
        return {k: v.to(self.device) for k, v in inputs.items()}, steps

    def process(self, payload: dict[str, Any]) -> tuple[dict, int]:
        """The authors' preprocessing, as CPU tensors (floats in the model's dtype)."""
        import cv2
        import json_numpy
        import torch
        from PIL import Image

        images = []
        for key in ("image0", "image1", "image2"):
            if key not in payload:
                continue
            v = json_numpy.loads(payload[key])
            if isinstance(v, np.ndarray):
                if v.ndim == 1:  # encoded bytes
                    v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                images.append(Image.fromarray(v))
            elif isinstance(v, (list, tuple)):
                images.append(Image.fromarray(np.array(v)))
            # The authors' handler also opens a file path given as a string; the harness never sends one.
        if not images:
            raise ValueError("No valid images found.")

        with PROCESSING_LOCK:
            inputs = self.processor(images, payload["language_instruction"])
        if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
            raise ValueError("Processor returned incomplete inputs.")

        def to_model(t):
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            return t.to(dtype=self.dtype) if t.is_floating_point() else t

        inputs = {k: to_model(v) for k, v in inputs.items()}
        if "proprio" in payload:
            proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload["proprio"])))
            inputs["proprio"] = to_model(proprio.unsqueeze(0))
        if "domain_id" in payload:
            inputs["domain_id"] = torch.tensor([int(payload["domain_id"])], dtype=torch.long)
        steps = int(payload.get("steps", 10))
        return inputs, steps

    # ------------------------------------------------------------ batches (effort step 5, amendment 24)

    def _prepare_request(self, payload: dict[str, Any], kind: str) -> dict:
        """One request for a batcher: checked to fit the fixed geometry, preprocessed on the CPU (in the caller's
        thread, so a bad request fails alone)."""
        if "image2" in payload or not ("image0" in payload and "image1" in payload):
            raise ValueError("batched requests carry exactly image0 (main view) and image1 (wrist view)")
        steps = int(payload.get("steps", self.batching.steps))
        if kind != "features" and steps != self.batching.steps:
            raise ValueError(f"batched requests use {self.batching.steps} flow-matching steps, not {steps}")
        if kind == "act":
            if payload.get("seed") is None:
                raise ValueError("a batched /act needs its `seed`")
            seeds = [int(payload["seed"])]
        elif kind == "sample":
            seeds = [int(x) for x in payload.get("seeds") or []]
            if not seeds:
                raise ValueError("/sample needs a non-empty `seeds` list")
        else:
            seeds = []
        inputs, _ = self.process(payload)
        ids = inputs["input_ids"]
        if tuple(ids.shape) != (1, self.token_length):
            raise ValueError(f"the instruction must be exactly {self.token_length} tokens, got {tuple(ids.shape)}")
        if int(ids[0, -1]) != self.pad_token_id:
            raise ValueError(f"the instruction fills all {self.token_length} tokens (it may have been truncated)")
        if kind == "features":                       # the encoder's inputs only, so every /features row is alike
            inputs = {k: inputs[k] for k in ("input_ids", "image_input", "image_mask")}
        elif not {"proprio", "domain_id"}.issubset(inputs):
            raise ValueError("a batched /act or /sample needs `proprio` and `domain_id`")
        return {"inputs": inputs, "seeds": seeds}

    def _run_batch(self, kind: str, reqs: list[dict]) -> list[dict]:
        """One batch at the fixed geometry; one result per request, in order."""
        import torch

        b = self.batching
        timer = Timer(self.device.type == "cuda")
        with self.lock, torch.no_grad():
            self.xvla.eval()
            rows = pad_rows([r["inputs"] for r in reqs], b.max_batch)
            keys = rows[0].keys()
            if any(x.keys() != keys for x in rows):
                raise ValueError("requests in one batch must carry the same fields")
            batch = {k: torch.cat([x[k] for x in rows], 0).to(self.device) for k in keys}
            with timer.span("encode"):
                enc = self.encode(batch)
            feats = {k: v.cpu().numpy() for k, v in self.pooled_features(enc, batch["image_mask"]).items()}
            out = [{"features": {k: v[i].tolist() for k, v in feats.items()}} for i in range(len(reqs))]
            if kind != "features":
                owner = [i for i, r in enumerate(reqs) for _ in r["seeds"]]
                seeds = [s for r in reqs for s in r["seeds"]]
                idx = pad_rows(list(range(len(seeds))), b.max_batch if kind == "act" else b.sample_rows)
                sel = torch.tensor([owner[j] for j in idx], dtype=torch.long, device=self.device)
                x1 = seeded_noise([seeds[j] for j in idx], (self.num_actions, self.dim_action), self.device,
                                  "randn").to(self.dtype)
                with timer.span("denoise"):
                    actions = self.denoise({k: v.index_select(0, sel) for k, v in enc.items()},
                                           batch["proprio"].index_select(0, sel),
                                           batch["domain_id"].index_select(0, sel), b.steps, x1)
                actions = actions.float().cpu().numpy()
                start = 0
                for i, r in enumerate(reqs):
                    chunk = actions[start:start + len(r["seeds"])]
                    start += len(r["seeds"])
                    if kind == "act":
                        out[i]["action"] = chunk[0].tolist()
                    else:
                        out[i]["actions"] = chunk.tolist()
        info = {"requests": len(reqs), "encoder_rows": b.max_batch,
                "denoise_rows": {"act": b.max_batch, "sample": b.sample_rows, "features": 0}[kind]}
        for o in out:
            o["timing_s"], o["batch"] = dict(timer.spans), info
        return out

    # ------------------------------------------------------------ generation (authors' generate_actions)

    def encode(self, inputs: dict) -> dict:
        return self.xvla.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])

    def denoise(self, enc: dict, proprio, domain_id, steps: int, x1):
        """modeling_xvla.py:198-216 from initial noise `x1` [B, num_actions, dim_action].

        The authors' post-processing (a sigmoid on the two gripper channels) runs on each row alone. A CPU elementwise
        kernel computes the last (element count mod vector width) elements of its tensor with scalar code, whose exp
        can differ from the vectorised exp in the last bit; run over the whole batch, those elements are the last row's
        last gripper values, so a request's result depended on its position and on the batch's size (the intermittent
        simultaneous-requests failure; docs/HARNESS.md "Batching"). Row by row, every row is computed exactly as the
        authors' batch of one."""
        import torch

        B = x1.shape[0]
        action = torch.zeros_like(x1)
        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=proprio.device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.xvla.action_space.preprocess(proprio, x_t)
            action = self.xvla.transformer(domain_id=domain_id, action_with_noise=x_t_m, proprio=proprio_m,
                                           t=t, **enc)
        post = self.xvla.action_space.postprocess
        return torch.cat([post(row.clone()) for row in action.split(1)], 0)

    def initial_noise(self, seeds: Optional[Sequence[int]], batch: int, like):
        import torch

        if seeds is None:   # reference behaviour: modeling_xvla.py:201
            return torch.randn(batch, self.num_actions, self.dim_action, device=like.device, dtype=like.dtype)
        if len(seeds) != batch:
            raise ValueError(f"{len(seeds)} seeds for a batch of {batch}")
        return seeded_noise(seeds, (self.num_actions, self.dim_action), like.device, "randn").to(like.dtype)

    @staticmethod
    def pooled_features(enc: dict, image_mask) -> dict:
        views = image_mask.shape[1]
        aux = enc["aux_visual_inputs"]                      # [B, (V-1)*N, D]: views 1.. flattened
        n = aux.shape[1] // (views - 1)
        return {
            "enc_mean": enc["vlm_features"].mean(dim=1).float(),
            "wrist_mean": aux[:, :n].mean(dim=1).float(),
        }

    # ------------------------------------------------------------ endpoints

    def act(self, payload: dict) -> dict:
        """Seeded: batched at the fixed geometry. Without a seed: the reference behaviour, alone."""
        if payload.get("seed") is None:
            return self.act_unbatched(payload)
        return self.batchers["act"].submit(payload)

    def sample(self, payload: dict) -> dict:
        return self.batchers["sample"].submit(payload)

    def features(self, payload: dict) -> dict:
        return self.batchers["features"].submit(payload)

    def act_unbatched(self, payload: dict) -> dict:
        """One request alone, unpadded (the authors' handler with an optional seed)."""
        import torch

        seed = payload.get("seed")
        timer = Timer(self.device.type == "cuda")
        with self.lock, torch.no_grad():
            self.xvla.eval()
            with timer.span("preprocess"):
                inputs, steps = self.prepare(payload)
            with timer.span("encode"):
                enc = self.encode(inputs)
            with timer.span("denoise"):
                x1 = self.initial_noise(None if seed is None else [seed], 1, inputs["proprio"])
                action = self.denoise(enc, inputs["proprio"], inputs["domain_id"], steps, x1)
            feats = self.pooled_features(enc, inputs["image_mask"])
        return {"action": action.squeeze(0).float().cpu().numpy().tolist(),
                "features": {k: v[0].cpu().numpy().tolist() for k, v in feats.items()},
                "timing_s": timer.spans}

    def sample_unbatched(self, payload: dict) -> dict:
        import torch

        seeds = payload.get("seeds")
        if not seeds:
            raise ValueError("/sample needs a non-empty `seeds` list")
        k = len(seeds)
        timer = Timer(self.device.type == "cuda")
        with self.lock, torch.no_grad():
            self.xvla.eval()
            with timer.span("preprocess"):
                inputs, steps = self.prepare(payload)
            with timer.span("encode"):
                enc = self.encode(inputs)
            with timer.span("denoise"):
                enc_k = {name: t.repeat(k, *([1] * (t.dim() - 1))) for name, t in enc.items()}
                proprio_k = inputs["proprio"].repeat(k, 1)
                domain_k = inputs["domain_id"].repeat(k)
                x1 = self.initial_noise(seeds, k, proprio_k)
                actions = self.denoise(enc_k, proprio_k, domain_k, steps, x1)
            feats = self.pooled_features(enc, inputs["image_mask"])
        return {"actions": actions.float().cpu().numpy().tolist(),
                "features": {k_: v[0].cpu().numpy().tolist() for k_, v in feats.items()},
                "timing_s": timer.spans}

    def features_unbatched(self, payload: dict) -> dict:
        import torch

        timer = Timer(self.device.type == "cuda")
        with self.lock, torch.no_grad():
            self.xvla.eval()
            with timer.span("preprocess"):
                inputs, _ = self.prepare(payload)
            with timer.span("encode"):
                enc = self.encode(inputs)
            feats = self.pooled_features(enc, inputs["image_mask"])
        return {"features": {k: v[0].cpu().numpy().tolist() for k, v in feats.items()}, "timing_s": timer.spans}

    def info(self) -> dict:
        return {"model": "xvla", "num_actions": self.num_actions, "dim_action": self.dim_action,
                "device": str(self.device), "dtype": str(self.dtype), "peak_memory_gb": peak_memory_gb(self.device),
                "batching": asdict(self.batching),
                "batch_sizes": {k: list(v.stats.sizes[-1000:]) for k, v in self.batchers.items()}, **torch_flags()}


def create_app(service: XVLAService):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()

    def wrap(fn):
        def handler(payload: dict[str, Any]):
            try:
                return JSONResponse(fn(payload))
            except Exception as e:  # the authors' handler logs and returns 400
                log.exception("request failed")
                return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)
        return handler

    app.post("/act")(wrap(service.act))
    app.post("/sample")(wrap(service.sample))
    app.post("/features")(wrap(service.features))
    app.get("/info")(lambda: service.info())
    return app


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="X-VLA server for the vla-router harness")
    ap.add_argument("--model-path", required=True, help="local snapshot of 2toINF/X-VLA-Pt at the pinned revision")
    ap.add_argument("--lora-path", default=None, help="local snapshot of 2toINF/X-VLA-libero-long-peft")
    ap.add_argument("--processor-path", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--max-batch", type=int, default=XVLABatching.max_batch, help="config servers.batching.max_batch")
    ap.add_argument("--sample-rows", type=int, default=XVLABatching.sample_rows,
                    help="config servers.batching.xvla_sample_rows")
    ap.add_argument("--wait-window-ms", type=float, default=5.0, help="config servers.batching.wait_window_ms")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    import uvicorn

    batching = XVLABatching(max_batch=a.max_batch, sample_rows=a.sample_rows, wait_window_s=a.wait_window_ms / 1000)
    service = XVLAService.load(a.model_path, a.lora_path, a.device, a.processor_path, batching)
    uvicorn.run(create_app(service), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
