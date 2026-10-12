"""Time one VLA checkpoint: latency per action chunk and peak GPU memory.

Two modes, neither of which needs a simulator:

- --pipeline model (default): synthetic tensors and placeholder token ids, timing
  the model call only. Needs no tokenizer. Supports batch sizes above 1.
- --pipeline full: random uint8 camera frames and real text instructions through
  the checkpoint's own preprocessor (real tokenizer), the model and the
  postprocessor, as the evaluation harness does. pi0.5 needs the gated PaliGemma
  tokenizer in the local cache for this (see envs/README.md).

Images are random in both modes; latency does not depend on image content.
Neither mode includes network or simulator time.

Example:
    python scripts/time_models.py --policy smolvla \
        --checkpoint lerobot/smolvla_libero --batch-sizes 1,4,8
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

import torch

OBS_STATE = "observation.state"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"

# LIBERO as served by the harness: two 256x256 cameras and an 8-D state.
LIBERO_STATE_DIM = 8
LIBERO_NUM_CAMERAS = 2
LIBERO_IMAGE_HW = (256, 256)


def load_policy(policy_type: str, checkpoint: str, device: str, compile_model: bool, compile_mode: str, dtype: str):
    from lerobot.configs.policies import PreTrainedConfig

    if policy_type == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as policy_cls
    elif policy_type == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as policy_cls
    else:
        raise ValueError(f"unknown policy type: {policy_type}")

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    config.compile_model = compile_model
    config.compile_mode = compile_mode
    if dtype != "checkpoint":
        if not hasattr(config, "dtype"):
            raise ValueError(f"{policy_type} config has no dtype field; use --dtype checkpoint")
        config.dtype = dtype

    policy = policy_cls.from_pretrained(checkpoint, config=config)
    policy.to(device)
    policy.eval()
    return policy, config


def make_batch(config, batch_size: int, device: str, real_tokens: int) -> dict[str, torch.Tensor]:
    """Synthetic batch in the form the policy sees after its preprocessor."""
    image_keys = list(config.image_features)[:LIBERO_NUM_CAMERAS]
    batch: dict[str, torch.Tensor] = {}
    for key in image_keys:
        batch[key] = torch.rand(batch_size, 3, *LIBERO_IMAGE_HW, device=device)
    batch[OBS_STATE] = torch.randn(batch_size, LIBERO_STATE_DIM, device=device)

    max_len = config.tokenizer_max_length
    n_real = min(real_tokens, max_len)
    tokens = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    tokens[:, :n_real] = torch.randint(10, 1000, (batch_size, n_real), device=device)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    mask[:, :n_real] = True
    batch[OBS_LANGUAGE_TOKENS] = tokens
    batch[OBS_LANGUAGE_ATTENTION_MASK] = mask
    return batch


# Instructions in the style of LIBERO's long-horizon suite, used by --pipeline full.
INSTRUCTIONS = [
    "put both the alphabet soup and the tomato sauce in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
]


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def run_full_pipeline(policy, config, checkpoint: str, args) -> dict:
    """Time raw observation -> preprocessor (real tokenizer) -> model -> postprocessor.

    Mirrors the evaluation harness's LeRobot model server: uint8 HWC frames and a
    text instruction go through the checkpoint's own saved processors.
    """
    import numpy as np
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference

    device = torch.device(args.device)
    preprocess, postprocess = make_pre_post_processors(
        config, checkpoint, preprocessor_overrides={"device_processor": {"device": str(device)}}
    )
    rng = np.random.default_rng(0)

    def one_call(index: int):
        # Dataset-style camera names; a checkpoint that uses other names renames them itself.
        frame = {
            "observation.images.image": rng.integers(0, 256, (*LIBERO_IMAGE_HW, 3), dtype=np.uint8),
            "observation.images.image2": rng.integers(0, 256, (*LIBERO_IMAGE_HW, 3), dtype=np.uint8),
            OBS_STATE: rng.standard_normal(LIBERO_STATE_DIM).astype(np.float32),
        }
        task = INSTRUCTIONS[index % len(INSTRUCTIONS)]
        sync(args.device)
        t0 = time.perf_counter()
        batch = prepare_observation_for_inference(frame, device, task=task, robot_type="")
        batch = preprocess(batch)
        sync(args.device)
        t1 = time.perf_counter()
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)
        sync(args.device)
        t2 = time.perf_counter()
        actions = postprocess(chunk).squeeze(0).detach().to("cpu").numpy()
        t3 = time.perf_counter()
        n_tokens = int(batch[OBS_LANGUAGE_ATTENTION_MASK].sum().item())
        return (t1 - t0, t2 - t1, t3 - t2, t3 - t0), actions, n_tokens

    policy.reset()
    first, actions, _ = one_call(0)
    for i in range(args.warmup):
        one_call(i)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stages, token_counts = [], []
    for i in range(args.calls):
        times, actions, n_tokens = one_call(i)
        stages.append(times)
        token_counts.append(n_tokens)

    def med(column: int) -> float:
        return round(1000 * statistics.median(s[column] for s in stages), 1)

    return {
        "pipeline": "full",
        "first_call_s": round(first[3], 3),
        "preprocess_ms": med(0),
        "model_ms": med(1),
        "postprocess_ms": med(2),
        "median_ms": med(3),
        "p95_ms": round(1000 * percentile([s[3] for s in stages], 0.95), 1),
        "real_tokens_min_max": [min(token_counts), max(token_counts)],
        "chunk_shape": list(actions.shape),
        "action_abs_max": round(float(np.abs(actions).max()), 3),
        "finite_output": bool(np.isfinite(actions).all()),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if args.device == "cuda" else None,
    }


def timed_call(policy, batch, device: str) -> tuple[float, torch.Tensor]:
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(dict(batch))
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start, chunk


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(q * (len(ordered) - 1)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, choices=["smolvla", "pi05"])
    parser.add_argument("--checkpoint", required=True, help="Hub id or local path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="checkpoint", help="checkpoint, bfloat16 or float32 (pi05 only)")
    parser.add_argument("--compile", action="store_true", help="use torch.compile")
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--batch-sizes", default="1", help="comma-separated")
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--real-tokens", type=int, default=32, help="unmasked placeholder tokens per prompt")
    parser.add_argument(
        "--pipeline",
        default="model",
        choices=["model", "full"],
        help="model: synthetic tensors, model call only. full: raw frames and real "
        "instructions through the checkpoint's own processors (batch size 1)",
    )
    parser.add_argument("--label", default="", help="free-text machine label for the output row")
    args = parser.parse_args()

    load_start = time.perf_counter()
    policy, config = load_policy(
        args.policy, args.checkpoint, args.device, args.compile, args.compile_mode, args.dtype
    )
    load_s = time.perf_counter() - load_start
    n_params = sum(p.numel() for p in policy.parameters())
    param_dtypes = sorted({str(p.dtype) for p in policy.parameters()})
    weights_gb = torch.cuda.memory_allocated() / 1e9 if args.device == "cuda" else float("nan")

    if args.pipeline == "full":
        row = {
            "label": args.label,
            "device": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
            "policy": args.policy,
            "checkpoint": args.checkpoint,
            "param_dtypes": param_dtypes,
            "compiled": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
            "calls": args.calls,
            "weights_gb": round(weights_gb, 2),
            **run_full_pipeline(policy, config, args.checkpoint, args),
        }
        print("RESULT " + json.dumps(row), flush=True)
        return

    for batch_size in [int(b) for b in args.batch_sizes.split(",")]:
        batch = make_batch(config, batch_size, args.device, args.real_tokens)
        policy.reset()

        first_s, chunk = timed_call(policy, batch, args.device)
        for _ in range(args.warmup):
            timed_call(policy, batch, args.device)

        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        latencies = []
        for _ in range(args.calls):
            seconds, chunk = timed_call(policy, batch, args.device)
            latencies.append(seconds)

        # Two fresh calls on one observation: nonzero spread means sampling is stochastic.
        _, chunk_a = timed_call(policy, batch, args.device)
        _, chunk_b = timed_call(policy, batch, args.device)
        resample_diff = (chunk_a.float() - chunk_b.float()).abs().mean().item()

        row = {
            "label": args.label,
            "machine": platform.machine(),
            "device": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
            "torch": torch.__version__,
            "policy": args.policy,
            "checkpoint": args.checkpoint,
            "params_millions": round(n_params / 1e6, 1),
            "param_dtypes": param_dtypes,
            "compiled": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
            "batch_size": batch_size,
            "chunk_shape": list(chunk.shape),
            "denoise_steps": getattr(config, "num_steps", getattr(config, "num_inference_steps", None)),
            "calls": args.calls,
            "load_s": round(load_s, 1),
            "first_call_s": round(first_s, 3),
            "median_ms": round(1000 * statistics.median(latencies), 1),
            "p95_ms": round(1000 * percentile(latencies, 0.95), 1),
            "min_ms": round(1000 * min(latencies), 1),
            "weights_gb": round(weights_gb, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if args.device == "cuda" else None,
            "resample_mean_abs_diff": round(resample_diff, 5),
            "finite_output": bool(torch.isfinite(chunk.float()).all().item()),
        }
        print("RESULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # report the failure as a row so sweeps keep their record
        print("FAILED " + json.dumps({"argv": sys.argv[1:], "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        raise
