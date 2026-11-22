"""π0.5 server: LeRobot 0.6.1's own policy loading and processor chain, as `lerobot-eval` runs it
(docs/SPEC.md §2-3). Runs in the LeRobot environment.

Loading mirrors `lerobot_eval.eval_main` (lerobot/scripts/lerobot_eval.py:743-783) for the
reference command `--policy.path=lerobot/pi05_libero_finetuned --policy.n_action_steps=10
--policy.compile_model=false --policy.device=cuda --env.type=libero --env.task=libero_10`:
`PreTrainedConfig.from_pretrained` with those overrides and `pretrained_path` set;
`make_policy(cfg, env_cfg=LiberoEnv(task="libero_10"))`; the checkpoint's own pre/post-processors with the device
and rename overrides; LIBERO's env processors. TF32 matmuls stay on as in the reference; one deliberate deviation:
`common.deterministic_torch()` replaces the reference's `cudnn.benchmark = True` (amendment 24: autotuning can pick
another kernel after a restart, which would break exact replay).

Per request the steps are the rollout loop's (lerobot_eval.py:273-300) for one decision:
`preprocess_observation(obs)`, `obs["task"] = [description]`, env pre-processor (180° image flip,
8-value state), policy pre-processor (normalise, state tokens in the prompt, PaliGemma tokens),
then `predict_action_chunk` (what `select_action` calls when its queue is empty) and, per
action, the post-processor (un-normalise) and env post-processor, as `select_action` + the loop
apply them one action at a time. All 50 actions are returned; the harness executes the first 10.

Request (json_numpy-encoded body): {"obs": <batched raw observation from adapters.pi05>,
"task": <task description>, "seed": <int, optional>}. Response: {"actions": [50 x 7] float32}.

Batching (effort step 5, SPEC §3, §12, amendments 24 and 26). Seeded requests (every request the harness sends) go
through one `Batcher`, and every batch runs at a fixed geometry (`PI05Batching`): exactly `max_batch` rows, padded
with copies of the batch's first request, every prompt at the checkpoint's fixed token length (200, right-padded,
with attention masks so the padding cannot affect results), and exactly the cameras the checkpoint was trained with
(the empty camera is the policy's own padding image, the same in every row). Each row's initial noise comes from its
own seed's CPU generator; each real row is post-processed alone, one action at a time, exactly as the unbatched path
does. A request's result is therefore bit-identical whichever requests share its batch, however many, and wherever it
sits. A request without a seed keeps the reference behaviour (global RNG, a batch of one, unpadded) outside the
batcher, as does `/upload_bytes` (a Phase 0 measurement).

Shared per-call state (audited for amendment 24; docs/HARNESS.md "Batching"). LeRobot's processor steps store each
call's transition on the step object itself (`ProcessorStep.__call__`), and `TokenizerProcessorStep` reads the
prompt back from there, so two requests pre-processed at the same time could swap prompts; the (un)normaliser moves
its statistics to the input's device and dtype in the middle of a call. Every call into the four processor
pipelines, in every code path, therefore runs under `PROCESSING_LOCK`, and the model runs under the service's device
lock, only ever one batch at a time. Lock order: device lock, then `PROCESSING_LOCK`, never the reverse.
"""

from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .batching import Batcher, pad_rows
from .common import Timer, deterministic_torch, peak_memory_gb, seeded_noise, torch_flags

log = logging.getLogger("vlar.pi05_server")

REFERENCE_OVERRIDES = ["--n_action_steps=10", "--compile_model=false"]

# Every call into LeRobot's processor pipelines (pre and post, policy and env) runs under this lock (see the module
# docstring). Module-level, so it also covers services that share pipeline objects (as tests do).
PROCESSING_LOCK = threading.Lock()


@dataclass(frozen=True)
class PI05Batching:
    """The fixed batch geometry (amendment 24): every pass runs on exactly `max_batch` rows; the token length is the
    checkpoint's `tokenizer_max_length` (amendment 26). Placeholders until Phase 0 step 2 measures the padding cost."""
    max_batch: int = 8
    wait_window_s: float = 0.005

    def __post_init__(self):
        if self.max_batch < 1 or self.wait_window_s < 0:
            raise ValueError(f"invalid batching {self}")


class PI05Service:
    @classmethod
    def load(cls, checkpoint: str, device: str = "cuda", batching: Optional[PI05Batching] = None) -> "PI05Service":
        import torch

        deterministic_torch()                             # instead of lerobot_eval.py:746 (cudnn.benchmark = True)
        torch.backends.cuda.matmul.allow_tf32 = True      # lerobot_eval.py:747

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(checkpoint, cli_overrides=REFERENCE_OVERRIDES + [f"--device={device}"])
        cfg.pretrained_path = Path(checkpoint)
        env_cfg = LiberoEnvConfig(task="libero_10")
        policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map={})
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=cfg.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": str(policy.config.device)},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
        return cls(policy, cfg, preprocessor, postprocessor, env_preprocessor, env_postprocessor, batching)

    def __init__(self, policy, cfg, preprocessor, postprocessor, env_preprocessor, env_postprocessor,
                 batching: Optional[PI05Batching] = None):
        import torch

        self.policy, self.cfg = policy, cfg
        self.preprocessor, self.postprocessor = preprocessor, postprocessor
        self.env_preprocessor, self.env_postprocessor = env_preprocessor, env_postprocessor
        self.device = torch.device(str(policy.config.device))
        self.lock = threading.Lock()      # one batch (or unbatched request) on the device at a time
        self.batching = b = batching or PI05Batching()
        self.token_length = int(cfg.tokenizer_max_length)
        # The cameras every batched request must carry: the checkpoint's image features except the empty cameras,
        # which the policy itself fills with its padding image (the same in every row).
        self.cameras = [k for k in cfg.image_features if "empty_camera" not in k]
        self.batcher = Batcher("pi05-act", prepare=self._prepare_request, run=self._run_batch,
                               capacity=(b.max_batch,), wait_window_s=b.wait_window_s)

    def close(self) -> None:
        self.batcher.close()

    # ------------------------------------------------------------ processing (lerobot-eval's, under the lock)

    def preprocess(self, obs: dict, task: str) -> dict:
        """lerobot-eval's per-step pre-processing of one observation (batch dimension 1)."""
        import torch
        from lerobot.envs.utils import preprocess_observation

        with PROCESSING_LOCK, torch.no_grad():
            observation = preprocess_observation(obs)
            observation["task"] = [task]
            observation = self.env_preprocessor(observation)
            return self.preprocessor(observation)

    def postprocess(self, chunk) -> np.ndarray:
        """One request's chunk [1, T, 7] through the post-processors one action at a time, as `select_action` and the
        rollout loop apply them; [T, 7] float32."""
        import torch
        from lerobot.utils.constants import ACTION

        rows = []
        with PROCESSING_LOCK, torch.no_grad():
            for t in range(chunk.shape[1]):
                action = self.postprocessor(chunk[:, t])                   # [1, 7], as select_action's output
                action = self.env_postprocessor({ACTION: action})[ACTION]
                rows.append(action.to("cpu").numpy())
        return np.concatenate(rows, axis=0).astype(np.float32, copy=False)

    def noise(self, seeds: Optional[list[int]]):
        """Initial flow-matching noise, one row per seed. None lets the policy draw it (reference behaviour)."""
        if seeds is None:
            return None
        shape = (self.cfg.chunk_size, self.cfg.max_action_dim)   # modeling_pi05.py:633-637
        return seeded_noise(seeds, shape, self.device, "normal")

    # ------------------------------------------------------------ batches (effort step 5, amendments 24 and 26)

    def _prepare_request(self, request: dict[str, Any]) -> dict:
        """One request for the batcher, in the caller's thread (a bad request fails alone): its seed, then the
        reference pre-processing, checked against the fixed geometry."""
        if request.get("seed") is None:
            raise ValueError("a batched π0.5 request needs its `seed`")
        seed = int(request["seed"])
        observation = self.preprocess(request["obs"], request["task"])
        self.check_geometry(observation)
        return {"observation": observation, "seed": seed}

    def check_geometry(self, observation: dict) -> None:
        """A pre-processed request fits the fixed geometry: batch dimension 1, exactly the checkpoint's cameras, and
        the prompt at the fixed token length, right-padded, not filling it (a full prompt may have been truncated)."""
        import torch
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        for k, v in observation.items():
            if isinstance(v, torch.Tensor) and (v.dim() == 0 or v.shape[0] != 1):
                raise ValueError(f"`{k}` must have batch dimension 1, got shape {tuple(v.shape)}")
        present = [k for k in self.cfg.image_features if k in observation]
        if present != self.cameras:
            raise ValueError(f"batched requests carry exactly the cameras {self.cameras}, not {present}")
        tokens, mask = observation[OBS_LANGUAGE_TOKENS], observation[OBS_LANGUAGE_ATTENTION_MASK]
        if tuple(tokens.shape) != (1, self.token_length) or tuple(mask.shape) != (1, self.token_length):
            raise ValueError(f"the prompt must be exactly {self.token_length} tokens, got {tuple(tokens.shape)}")
        if mask.dtype != torch.bool:
            raise ValueError(f"the attention mask must be boolean, got {mask.dtype}")
        n = int(mask.sum())
        if n >= self.token_length:
            raise ValueError(f"the prompt fills all {self.token_length} tokens (it may have been truncated)")
        if n == 0 or not bool(mask[0, :n].all()) or bool(mask[0, n:].any()):
            raise ValueError("the prompt must be right-padded: real tokens first, then padding")

    def _run_batch(self, reqs: list[dict]) -> list[dict]:
        """One batch at the fixed geometry; one result per request, in order."""
        import torch

        b = self.batching
        rows = pad_rows([r["observation"] for r in reqs], b.max_batch)
        seeds = pad_rows([r["seed"] for r in reqs], b.max_batch)
        keys = rows[0].keys()
        if any(x.keys() != keys for x in rows):
            raise ValueError("requests in one batch must carry the same fields")
        batch = {}
        for k in keys:
            vals = [x[k] for x in rows]
            if isinstance(vals[0], torch.Tensor):
                if any(v.shape != vals[0].shape or v.dtype != vals[0].dtype or v.device != vals[0].device
                       for v in vals):
                    raise ValueError(f"`{k}` differs in shape, dtype or device between requests of one batch")
                batch[k] = torch.cat(vals, 0)
            elif isinstance(vals[0], (list, tuple)):
                if any(len(v) != 1 for v in vals):
                    raise ValueError(f"`{k}` must hold one entry per request")
                batch[k] = [v[0] for v in vals]
            elif all(v == vals[0] for v in vals):
                batch[k] = vals[0]
            else:
                raise ValueError(f"`{k}` differs between requests of one batch")
        timer = Timer(self.device.type == "cuda")
        with self.lock, torch.no_grad():
            with timer.span("model"):
                with torch.inference_mode():
                    noise = self.noise(seeds)
                    chunk = self.policy.predict_action_chunk(batch, noise=noise)   # [max_batch, 50, 7], normalised
            with timer.span("postprocess"):
                actions = [self.postprocess(chunk[i:i + 1]) for i in range(len(reqs))]
        info = {"requests": len(reqs), "rows": b.max_batch, "tokens": self.token_length}
        return [{"actions": a, "timing_s": dict(timer.spans), "batch": info} for a in actions]

    # ------------------------------------------------------------ endpoints

    def act(self, request: dict[str, Any]) -> dict[str, Any]:
        """Seeded (every request the harness sends): batched at the fixed geometry. Without a seed: the reference
        behaviour, alone."""
        if request.get("seed") is None:
            return self.act_unbatched(request)
        return self.batcher.submit(request)

    def act_unbatched(self, request: dict[str, Any]) -> dict[str, Any]:
        """One request alone, unpadded: lerobot-eval's step with an optional seed (the reference path)."""
        import torch

        seed = request.get("seed")
        timer = Timer(self.device.type == "cuda")
        with self.lock, torch.no_grad():
            with timer.span("preprocess"):
                observation = self.preprocess(request["obs"], request["task"])
            with timer.span("model"):
                with torch.inference_mode():
                    noise = self.noise(None if seed is None else [seed])
                    kwargs = {} if noise is None else {"noise": noise}
                    chunk = self.policy.predict_action_chunk(observation, **kwargs)   # [1, 50, 7], normalised
            with timer.span("postprocess"):
                actions = self.postprocess(chunk)
        return {"actions": actions, "timing_s": timer.spans}

    def upload_jpeg_bytes(self, request: dict[str, Any]) -> dict[str, Any]:
        """Phase 0 step 2 (SPEC §5, amendment 14): JPEG sizes of the images π0.5 actually consumes for this
        observation. Same preprocessing as `act`, then the policy's own `_preprocess_images` (resize with pad to
        224 px, scaled to [-1, 1]); the cameras present in the batch are mapped back to uint8 and encoded by
        `network.jpeg_size` at `quality` (default 90). The padding images of absent cameras are never uploaded."""
        import torch

        from ..network import jpeg_size

        quality = int(request.get("quality", 90))
        with self.lock, torch.no_grad():
            observation = self.preprocess(request["obs"], request["task"])
            present = [k for k in self.policy.config.image_features if k in observation]
            images, _ = self.policy._preprocess_images(observation)
        sizes, shapes = [], []
        for key, img in zip(present, images):                     # present cameras come first, in this order
            u8 = ((img[0].float() + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            sizes.append(jpeg_size(u8, quality))
            shapes.append(list(u8.shape))
        return {"cameras": present, "jpeg_bytes": sizes, "shapes": shapes, "quality": quality}

    def info(self) -> dict:
        return {"model": "pi05", "chunk_size": self.cfg.chunk_size, "n_action_steps": self.cfg.n_action_steps,
                "dtype": getattr(self.cfg, "dtype", None), "device": str(self.device),
                "peak_memory_gb": peak_memory_gb(self.device), "batching": asdict(self.batching),
                "token_length": self.token_length, "cameras": self.cameras,
                "batch_sizes": list(self.batcher.stats.sizes[-1000:]), **torch_flags()}


def create_app(service: PI05Service):
    import json_numpy
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response

    app = FastAPI()

    @app.post("/act")
    async def act(request: Request):
        try:
            body = json_numpy.loads(await request.body())
            out = await _run_sync(service.act, body)
            return Response(content=json_numpy.dumps(out), media_type="application/json")
        except Exception as e:
            log.exception("request failed")
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)

    @app.post("/upload_bytes")
    async def upload_bytes(request: Request):
        try:
            body = json_numpy.loads(await request.body())
            return JSONResponse(await _run_sync(service.upload_jpeg_bytes, body))
        except Exception as e:
            log.exception("request failed")
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)

    @app.get("/info")
    def info():
        return service.info()

    return app


async def _run_sync(fn, *args):
    import anyio
    return await anyio.to_thread.run_sync(fn, *args)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="π0.5 server for the vla-router harness")
    ap.add_argument("--checkpoint", required=True,
                    help="local snapshot of lerobot/pi05_libero_finetuned at the pinned revision")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8778)
    ap.add_argument("--max-batch", type=int, default=PI05Batching.max_batch, help="config servers.batching.max_batch")
    ap.add_argument("--wait-window-ms", type=float, default=5.0, help="config servers.batching.wait_window_ms")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    import uvicorn

    batching = PI05Batching(max_batch=a.max_batch, wait_window_s=a.wait_window_ms / 1000)
    service = PI05Service.load(a.checkpoint, a.device, batching)
    uvicorn.run(create_app(service), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
