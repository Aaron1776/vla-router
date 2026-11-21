"""π0.5 server: LeRobot 0.6.1's own policy loading and processor chain, as `lerobot-eval` runs it
(docs/SPEC.md §2-3). Runs in the LeRobot environment.

Loading mirrors `lerobot_eval.eval_main` (lerobot/scripts/lerobot_eval.py:743-783) for the
reference command `--policy.path=lerobot/pi05_libero_finetuned --policy.n_action_steps=10
--policy.compile_model=false --policy.device=cuda --env.type=libero --env.task=libero_10`:
TF32 matmuls and cuDNN benchmark on; `PreTrainedConfig.from_pretrained` with those overrides and
`pretrained_path` set; `make_policy(cfg, env_cfg=LiberoEnv(task="libero_10"))`; the checkpoint's
own pre/post-processors with the device and rename overrides; LIBERO's env processors.

Per request the steps are the rollout loop's (lerobot_eval.py:273-300) for one decision:
`preprocess_observation(obs)`, `obs["task"] = [description]`, env pre-processor (180° image flip,
8-value state), policy pre-processor (normalise, state tokens in the prompt, PaliGemma tokens),
then `predict_action_chunk` (what `select_action` calls when its queue is empty) and, per
action, the post-processor (un-normalise) and env post-processor, as `select_action` + the loop
apply them one action at a time. All 50 actions are returned; the harness executes the first 10.

Request (json_numpy-encoded body): {"obs": <batched raw observation from adapters.pi05>,
"task": <task description>, "seed": <int, optional>}. Response: {"actions": [50 x 7] float32}.
"""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .common import Timer, peak_memory_gb, seeded_noise, torch_flags

log = logging.getLogger("vlar.pi05_server")

REFERENCE_OVERRIDES = ["--n_action_steps=10", "--compile_model=false"]


class PI05Service:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        import torch

        torch.backends.cudnn.benchmark = True            # lerobot_eval.py:746-747
        torch.backends.cuda.matmul.allow_tf32 = True

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(checkpoint, cli_overrides=REFERENCE_OVERRIDES + [f"--device={device}"])
        cfg.pretrained_path = Path(checkpoint)
        env_cfg = LiberoEnvConfig(task="libero_10")
        policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map={})
        policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=cfg.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": str(policy.config.device)},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
        self.policy = policy
        self.cfg = cfg
        self.device = torch.device(str(policy.config.device))
        self.lock = threading.Lock()

    def noise(self, seeds: Optional[list[int]]):
        """Initial flow-matching noise, one row per seed. None lets the policy draw it (reference behaviour)."""
        if seeds is None:
            return None
        shape = (self.cfg.chunk_size, self.cfg.max_action_dim)   # modeling_pi05.py:633-637
        return seeded_noise(seeds, shape, self.device, "normal")

    def act(self, request: dict[str, Any]) -> dict[str, Any]:
        import torch
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION

        obs = request["obs"]
        task = request["task"]
        seed = request.get("seed")
        timer = Timer(self.device.type == "cuda")
        with self.lock, torch.no_grad():
            with timer.span("preprocess"):
                observation = preprocess_observation(obs)
                observation["task"] = [task]
                observation = self.env_preprocessor(observation)
                observation = self.preprocessor(observation)
            with timer.span("model"):
                with torch.inference_mode():
                    noise = self.noise(None if seed is None else [seed])
                    kwargs = {} if noise is None else {"noise": noise}
                    chunk = self.policy.predict_action_chunk(observation, **kwargs)   # [1, 50, 7], normalised
            with timer.span("postprocess"):
                rows = []
                for t in range(chunk.shape[1]):
                    action = self.postprocessor(chunk[:, t])                 # [1, 7], as select_action's output
                    action = self.env_postprocessor({ACTION: action})[ACTION]
                    rows.append(action.to("cpu").numpy())
        actions = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
        return {"actions": actions, "timing_s": timer.spans}

    def upload_jpeg_bytes(self, request: dict[str, Any]) -> dict[str, Any]:
        """Phase 0 step 2 (SPEC §5, amendment 14): JPEG sizes of the images π0.5 actually consumes for this
        observation. Same preprocessing as `act`, then the policy's own `_preprocess_images` (resize with pad to
        224 px, scaled to [-1, 1]); the cameras present in the batch are mapped back to uint8 and encoded by
        `network.jpeg_size` at `quality` (default 90). The padding images of absent cameras are never uploaded."""
        import torch
        from lerobot.envs.utils import preprocess_observation

        from ..network import jpeg_size

        quality = int(request.get("quality", 90))
        with self.lock, torch.no_grad():
            observation = preprocess_observation(request["obs"])
            observation["task"] = [request["task"]]
            observation = self.env_preprocessor(observation)
            observation = self.preprocessor(observation)
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
                "peak_memory_gb": peak_memory_gb(self.device), **torch_flags()}


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
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    import uvicorn

    service = PI05Service(a.checkpoint, a.device)
    uvicorn.run(create_app(service), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
