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

Pooled features [ASSUMPTION in the spec]: `enc_mean` is the mean over all encoder output tokens
(first image's tokens and the 50 text tokens, padding included, as the encoder attends to all
of them); `wrist_mean` is the mean over the wrist view's image tokens, which reach the action
transformer directly rather than through the encoder. Both 1024 floats.
"""

from __future__ import annotations

import argparse
import logging
import threading
from typing import Any, Optional, Sequence

import numpy as np

from .. import reference
from .common import Timer, peak_memory_gb, seeded_noise, torch_flags

log = logging.getLogger("vlar.xvla_server")


class XVLAService:
    @classmethod
    def load(cls, model_path: str, lora_path: Optional[str], device: str = "cuda",
             processor_path: Optional[str] = None) -> "XVLAService":
        """deploy.py:63-98."""
        import torch

        reference.add_xvla_repo_to_path()
        from models.modeling_xvla import XVLA            # the authors' classes, from the pinned checkout
        from models.processing_xvla import XVLAProcessor

        processor = XVLAProcessor.from_pretrained(processor_path or model_path)
        model = XVLA.from_pretrained(model_path, trust_remote_code=True,
                                     torch_dtype=torch.float32).to(device).to(torch.float32)
        if lora_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float32).to(device)
        return cls(model, processor)

    def __init__(self, model, processor):
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
        self.lock = threading.Lock()   # one request on the GPU at a time (batching comes with the queue)

    # ------------------------------------------------------------ preprocessing (authors' handler)

    def prepare(self, payload: dict[str, Any]) -> tuple[dict, int]:
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

        inputs = self.processor(images, payload["language_instruction"])
        if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
            raise ValueError("Processor returned incomplete inputs.")

        def to_model(t):
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            return t.to(device=self.device, dtype=self.dtype) if t.is_floating_point() else t.to(device=self.device)

        inputs = {k: to_model(v) for k, v in inputs.items()}
        if "proprio" in payload:
            proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload["proprio"])))
            inputs["proprio"] = to_model(proprio.unsqueeze(0))
        if "domain_id" in payload:
            inputs["domain_id"] = torch.tensor([int(payload["domain_id"])], dtype=torch.long).to(self.device)
        steps = int(payload.get("steps", 10))
        return inputs, steps

    # ------------------------------------------------------------ generation (authors' generate_actions)

    def encode(self, inputs: dict) -> dict:
        return self.xvla.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])

    def denoise(self, enc: dict, proprio, domain_id, steps: int, x1):
        """modeling_xvla.py:198-216 from initial noise `x1` [B, num_actions, dim_action]."""
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
        return self.xvla.action_space.postprocess(action)

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

    def sample(self, payload: dict) -> dict:
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

    def features(self, payload: dict) -> dict:
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
                **torch_flags()}


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
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    import uvicorn

    service = XVLAService.load(a.model_path, a.lora_path, a.device, a.processor_path)
    uvicorn.run(create_app(service), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
