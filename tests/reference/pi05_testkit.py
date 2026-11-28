"""π0.5 test kit for the batching tests (SPEC §3, amendments 24 and 26).

`make_tiny_pi05` builds a `PI05Service` around LeRobot 0.6.1's own π0.5 code at a tiny size with random weights: one
transformer layer in the vision tower, in the language model and in the action expert, with the checkpoint's widths,
heads and vocabulary. Around it go the checkpoint's own processor pipelines, loaded from its JSON files exactly as
`PI05Service.load` loads them (with synthetic normalisation statistics, since the statistics ship with the weights),
LIBERO's env processors, and `DigitTokenizer` in place of the gated PaliGemma tokenizer. The Mac therefore runs
LeRobot's real pre-processing (including the processor steps that keep per-call state), the real prefix, attention
masks, KV cache and denoising loop, and the real post-processing; only the sizes, the weights and the tokenizer's
vocabulary differ from the VM, where the same checks run on the real model (`test_batch_invariance.py -k pi05`).

`pi05_request` builds requests in the π0.5 adapter's raw format (`adapters/pi05.py`) for a LIBERO-10 instruction and a
robot state whose prompt tokens are short, medium or long.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import zlib
from pathlib import Path

import numpy as np

from conftest import LIBERO_10, REPO

PI05_CONFIG_DIR = Path(os.environ.get("VLAR_PI05_CONFIG_DIR") or REPO / "third_party" / "hf" / "pi05_libero_finetuned")
PI05_CONFIG_READY = all((PI05_CONFIG_DIR / f).exists()
                        for f in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"))

# Normalised robot-state values whose 256-bin indices (Pi05PrepareStateTokenizerProcessorStep: floor((x + 1) * 128))
# have 1, 2 or 3 digits, so the state part of a prompt is short, medium or long. The synthetic statistics have mean 0
# and std 1 for the state, so a raw value is its normalised value.
STATE_LEVELS = {"short": -0.97, "medium": -0.8, "long": 0.6}

# The tiny model's vocabulary. π0.5 inference only looks up the prompt's token embeddings (the 257,152-entry tables and
# the unused output heads would cost gigabytes of random init on the CPU); the stand-in tokenizer's ids stay below it.
TINY_VOCAB = 4096


class DigitTokenizer:
    """Stand-in for the gated PaliGemma tokenizer, answering the call LeRobot's `TokenizerProcessorStep` makes: BOS,
    then words and punctuation as tokens and numbers split into single digits (as Gemma's tokenizer splits them), so
    the token count grows with the instruction and with the state's digits; right-padded with 0 to `max_length`."""

    bos, pad = 2, 0

    def __init__(self, vocab_size: int = TINY_VOCAB):
        self.vocab_size = vocab_size

    def ids(self, text: str) -> list[int]:
        parts = re.findall(r"\d|[A-Za-z]+|[^\sA-Za-z\d]", text)
        return [self.bos] + [10 + zlib.crc32(p.encode()) % (self.vocab_size - 10) for p in parts]

    def __call__(self, text, max_length, truncation=True, padding="max_length", padding_side="right",
                 return_tensors="pt"):
        import torch

        if (padding, padding_side, return_tensors) != ("max_length", "right", "pt"):
            raise NotImplementedError(f"stand-in tokenizer: {padding}, {padding_side}, {return_tensors}")
        texts = [text] if isinstance(text, str) else list(text)
        input_ids = torch.full((len(texts), max_length), self.pad, dtype=torch.long)
        mask = torch.zeros((len(texts), max_length), dtype=torch.long)
        for i, t in enumerate(texts):
            s = self.ids(t)
            if truncation:
                s = s[:max_length]
            input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


class _AutoTokenizer:
    @staticmethod
    def from_pretrained(name, *args, **kwargs):
        if name != "google/paligemma-3b-pt-224":
            raise ValueError(f"unexpected tokenizer {name}")
        return DigitTokenizer()


class _TinyConfigs:
    """Stands in for transformers' CONFIG_MAPPING inside LeRobot's model module: PaliGemma configs come with a
    one-layer vision tower and the action expert's Gemma config with the tiny vocabulary; LeRobot's own settings are
    applied on top, unchanged."""

    def __init__(self, real):
        self.real = real

    def __getitem__(self, key):
        factory = self.real[key]
        if key == "paligemma":
            def make(*args, **kwargs):
                cfg = factory(*args, **kwargs)
                cfg.vision_config.num_hidden_layers = 1
                return cfg
            return make
        if key == "gemma":
            return lambda *args, **kwargs: factory(*args, **{**kwargs, "vocab_size": TINY_VOCAB})
        return factory


def _tiny_vocab(cls):
    """LeRobot sets PaliGemma's vocabulary after building its config; shrink it just before the model is built."""
    def build(config):
        config.text_config.vocab_size = TINY_VOCAB
        config._vocab_size = TINY_VOCAB   # noqa: SLF001
        return cls(config=config)
    return build


def _tiny_gemma(variant: str):
    """One layer each; the checkpoint's widths (the VLM's 2048 is fixed by LeRobot's hard-coded projection size),
    heads and head size, which the joint attention of the language model and the action expert needs to agree."""
    from lerobot.policies.pi05.modeling_pi05 import GemmaConfig

    width = {"gemma_2b": 2048, "gemma_300m": 1024}[variant]
    return GemmaConfig(width=width, depth=1, mlp_dim=32, num_heads=8, num_kv_heads=1, head_dim=256)


def write_checkpoint_dir(dst: Path) -> Path:
    """The checkpoint's config and processor JSON files, with synthetic normalisation statistics in the state files
    the JSON files name (the real ones ship with the weights)."""
    import torch
    from safetensors.torch import save_file

    for f in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        shutil.copy(PI05_CONFIG_DIR / f, dst / f)
    g = torch.Generator().manual_seed(3)
    action = {"action.mean": torch.randn(7, generator=g) * 0.1, "action.std": torch.rand(7, generator=g) + 0.5}
    state = {"observation.state.mean": torch.zeros(8), "observation.state.std": torch.ones(8)}
    save_file({**state, **action}, str(dst / "policy_preprocessor_step_2_normalizer_processor.safetensors"))
    save_file(dict(action), str(dst / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    return dst


def _define_uninitialised(module) -> None:
    """Deterministic mode fills uninitialised memory with NaN; give any tensor random init never touched seeded values
    (a checkpoint overwrites them in production), as the X-VLA test builder does."""
    import torch

    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for t in list(module.parameters()) + list(module.buffers()):
            if t.is_floating_point() and not torch.isfinite(t).all():
                t.copy_((torch.randn(t.shape, generator=g) * 0.02).to(t.dtype))


def make_tiny_pi05(batching=None, dtype: str = "float32", policy=None):
    """A PI05Service on the CPU around a tiny random π0.5 (or `policy(cfg)`, a stand-in), with the checkpoint's own
    processor pipelines. float32 by default (the checkpoint's bfloat16 is slow on the CPU; the VM runs it)."""
    import pytest
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import modeling_pi05
    from lerobot.processor import tokenizer_processor

    from vlar_harness.servers.common import deterministic_torch
    from vlar_harness.servers.pi05_server import REFERENCE_OVERRIDES, PI05Service

    deterministic_torch()                                # as PI05Service.load does
    ckpt = write_checkpoint_dir(Path(tempfile.mkdtemp(prefix="vlar-pi05-tiny-")))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tokenizer_processor, "AutoTokenizer", _AutoTokenizer)
        mp.setattr(modeling_pi05, "get_gemma_config", _tiny_gemma)
        mp.setattr(modeling_pi05, "CONFIG_MAPPING", _TinyConfigs(modeling_pi05.CONFIG_MAPPING))
        mp.setattr(modeling_pi05, "PaliGemmaForConditionalGenerationWithPiGemma",
                   _tiny_vocab(modeling_pi05.PaliGemmaForConditionalGenerationWithPiGemma))
        cfg = PreTrainedConfig.from_pretrained(str(ckpt), cli_overrides=REFERENCE_OVERRIDES + ["--device=cpu",
                                                                                               f"--dtype={dtype}"])
        cfg.pretrained_path = ckpt
        if policy is None:
            torch.manual_seed(0)
            pol = modeling_pi05.PI05Policy(cfg)
            if len(pol.model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers) != 1:
                raise RuntimeError("the one-layer vision tower patch did not apply")
            _define_uninitialised(pol)
            pol.eval()
        else:
            pol = policy(cfg)
        pre, post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(ckpt),
            preprocessor_overrides={
                "device_processor": {"device": "cpu"},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=LiberoEnvConfig(task="libero_10"), policy_cfg=cfg)
    return PI05Service(pol, cfg, pre, post, env_pre, env_post, batching)


def _quat_from_axis_angle(v: np.ndarray) -> np.ndarray:
    """(x, y, z, w), the order LeRobot's LiberoProcessorStep reads."""
    angle = float(np.linalg.norm(v))
    axis = v / angle
    return np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])


def raw_obs(rng: np.random.Generator, state: str = "medium", px: int = 360) -> dict:
    """A raw observation in the π0.5 adapter's format (LeRobot's `_format_raw_obs`, batched by gymnasium): random
    images and a robot state whose 8 normalised values all sit at `STATE_LEVELS[state]` (plus a small jitter that
    keeps their digit counts)."""
    level = STATE_LEVELS[state]
    s = level + rng.uniform(-0.004, 0.004, size=8)
    f64 = lambda a: np.asarray(a, dtype=np.float64)[None]   # noqa: E731
    return {
        "pixels": {"image": rng.integers(0, 256, (1, px, px, 3), dtype=np.uint8),
                   "image2": rng.integers(0, 256, (1, px, px, 3), dtype=np.uint8)},
        "robot_state": {
            "eef": {"pos": f64(s[:3]), "quat": f64(_quat_from_axis_angle(s[3:6])), "mat": f64(np.eye(3))},
            "gripper": {"qpos": f64(s[6:8]), "qvel": f64(np.zeros(2))},
            "joints": {"pos": f64(rng.normal(size=7)), "vel": f64(np.zeros(7))},
        },
    }


def pi05_request(rng: np.random.Generator, task_id: int, state: str, seed: int, px: int = 360) -> dict:
    return {"obs": raw_obs(rng, state, px), "task": LIBERO_10[task_id], "seed": seed}


def real_pi05_requests(act, tasks=range(10), points: int = 2, seed_base: int = 3000):
    """Real LIBERO-10 decision points for the π0.5 checks (amendment 37): for each task, the observation at reset
    (scene seed 1000 + task, LeRobot's own LIBERO env as lerobot-eval builds it) and every 10 steps after it, the
    arm following the first 10 actions `act(request)` returns (the batch of one's own trajectory). Yields
    (task, point, request, act(request)); each request carries its own seed."""
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    for k in tasks:
        vec = LiberoEnvConfig(task="libero_10", task_ids=[k]).create_envs(n_envs=1, use_async_envs=False)["libero_10"][k]
        try:
            obs, _ = vec.reset(seed=[1000 + k], options={NEW_ROLLOUT_OPTION: True})
            task = list(vec.call("task_description"))[0]
            for d in range(points):
                req = {"obs": obs, "task": task, "seed": seed_base + 100 * k + d}
                actions = act(req)
                yield k, d, req, actions
                if d + 1 == points:
                    break
                for a in actions[:10]:
                    obs, _, term, trunc, _ = vec.step(np.asarray(a, dtype=np.float32)[None])
                    if bool(np.asarray(term).any()) or bool(np.asarray(trunc).any()):
                        break
        finally:
            vec.close()
