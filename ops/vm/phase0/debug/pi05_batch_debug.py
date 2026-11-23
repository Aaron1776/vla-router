"""Diagnosis of π0.5's padded-batch vs batch-of-one discrepancy (2026-11-23; stage A row pi05-batch-invariance-l4:
the request padded to 8 rows differed from the batch of one by max |diff| 1.94, allowed 0.05).

Runs in the vlar-pi05 image on the L4 (real checkpoint from VLAR_PI05_CHECKPOINT, bfloat16, CUDA), or on the Mac
with the tiny random model (--tiny, CPU, float32) to check the code paths. Parts, in order; each is written to
<out>/summary.json as it finishes, and a failing part is recorded while the rest still run:
  env          versions, GPU, torch flags
  reproduce    the failing request R1 exactly (the VM test's rng sequence): A = act_unbatched (the reference path),
               B = the batched path at max_batch 1, C = padded to the service's max_batch; A and C twice
  confirm      a second request on the same task, a second task, and real LIBERO observations on two tasks
  trace        row 0 of the padded batch against the batch of one at every module (hooks) and function boundary
               (images, noise, prompt, prefix, KV cache, each denoising step); the first differing point; the first
               differing operation re-run alone at 1 and at n rows
  stages       the prefix at 1 or 8 rows combined with the denoising at 1 or 8 rows
  sweep        |padded to n - A| for n = 1..16
  sensitivity  the batch of one under last-bit perturbations of its own noise and image
  profile      the CUDA kernels of one call at 1 row and at 8 rows
  survey       real observations on all ten tasks (several decision points each, following A's actions) and random
               observations: |C - A| per action dimension
  reduction    the largest survey cases with allow_bf16_reduced_precision_reduction off
  fp32         the model in float32 (TF32 off, then on): |C - A| there, and both bfloat16 results against float32
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

TOKENS, MASK = "observation.language.tokens", "observation.language.attention_mask"
OUT: Path
SUMMARY: dict = {"parts": {}, "errors": {}, "timing_s": {}}
LAST: dict = {}          # the policy's last normalised chunk (predict_action_chunk wrapper)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def save():
    (OUT / "summary.json").write_text(json.dumps(SUMMARY, indent=1, default=str))


def to_np(t):
    import torch
    if torch.is_tensor(t):
        return t.detach().to("cpu", torch.float64).numpy()
    return np.asarray(t, dtype=np.float64)


def cmp(a, ref) -> dict:
    """Bit equality (same dtype tensors) and size of the difference of `a` from `ref`."""
    import torch
    eq = None
    if torch.is_tensor(a) and torch.is_tensor(ref) and a.dtype == ref.dtype and a.shape == ref.shape:
        eq = bool(torch.equal(a.cpu(), ref.cpu()))
    x, r = to_np(a), to_np(ref)
    if x.shape != r.shape:
        return {"shape_mismatch": [list(x.shape), list(r.shape)]}
    if eq is None:
        eq = bool(np.array_equal(x, r, equal_nan=True))
    d = np.abs(x - r)
    scale = float(np.nanmax(np.abs(r))) if r.size else 0.0
    m = float(np.nanmax(d)) if d.size else 0.0
    return {"equal": eq, "max_abs": m, "max_rel": (m / scale) if scale > 0 else None,
            "frac_diff": float(np.mean(d > 0)) if d.size else 0.0,
            "nan": bool(np.isnan(x).any() or np.isnan(r).any())}


def act_diff(a, ref) -> dict:
    """[T, 7] actions: overall and per action dimension (max over time and the time step where it occurs)."""
    x, r = to_np(a), to_np(ref)
    d = np.abs(x - r)
    out = cmp(a, ref)
    out["per_dim_max"] = [round(float(v), 6) for v in d.max(axis=0)]
    out["per_dim_argmax_t"] = [int(v) for v in d.argmax(axis=0)]
    out["first10_max"] = float(d[:10].max())
    return out


def part(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            log(f"== {name}")
            try:
                SUMMARY["parts"][name] = fn(*a, **k)
            except Exception:
                SUMMARY["errors"][name] = traceback.format_exc()
                log(f"part {name} failed:\n{SUMMARY['errors'][name]}")
            SUMMARY["timing_s"][name] = round(time.time() - t0, 1)
            save()
            return SUMMARY["parts"].get(name)
        return run
    return deco


# ------------------------------------------------------------------------------------------------ service helpers

class Kit:
    def __init__(self, svc, tiny: bool):
        import torch
        from vlar_harness.servers.pi05_server import PI05Batching, PI05Service
        self.svc, self.tiny = svc, tiny
        self.torch = torch
        self.n = svc.batching.max_batch
        self.policy = svc.policy
        self.PI05Service, self.PI05Batching = PI05Service, PI05Batching
        self.one = self.service(1)
        orig = self.policy.predict_action_chunk

        def wrapped(batch, **kw):
            out = orig(batch, **kw)
            LAST["chunk"] = out.detach().to("cpu")
            return out
        self.policy.predict_action_chunk = wrapped

    def service(self, n: int):
        s = self.svc
        return self.PI05Service(s.policy, s.cfg, s.preprocessor, s.postprocessor, s.env_preprocessor,
                                s.env_postprocessor, self.PI05Batching(max_batch=n))

    def A(self, req):
        """The reference path: act_unbatched (batch of one, unpadded)."""
        r = self.svc.act_unbatched(req)
        self.last_model_s = r["timing_s"].get("model")
        return r["actions"], LAST["chunk"].clone()

    def padded(self, req, n=None):
        """The batched path at the fixed geometry of n rows (default the service's), request alone (row 0)."""
        s = self.svc if n in (None, self.n) else self.service(n)
        try:
            r = s._run_batch([s._prepare_request(req)])[0]
            self.last_model_s = r["timing_s"].get("model")
            return r["actions"], LAST["chunk"][0:1].clone()
        finally:
            if s is not self.svc:
                s.close()

    def B(self, req):
        return self.one._run_batch([self.one._prepare_request(req)])[0]["actions"]

    def batch(self, observations):
        """The batch `_run_batch` builds from prepared observations."""
        torch = self.torch
        out = {}
        for k in observations[0]:
            vals = [o[k] for o in observations]
            if torch.is_tensor(vals[0]):
                out[k] = torch.cat(vals, 0)
            elif isinstance(vals[0], (list, tuple)):
                out[k] = [v[0] for v in vals]
            else:
                out[k] = vals[0]
        return out

    def predict(self, batch, noise):
        torch = self.torch
        with self.svc.lock, torch.no_grad(), torch.inference_mode():
            return self.policy.predict_action_chunk(batch, noise=noise)

    def prepared(self, req):
        return self.svc._prepare_request(req)["observation"]


def r1_request(pi05_request):
    """The failing request: the VM test's rng sequence up to check_pi05_reference_path."""
    rng = np.random.default_rng(0)
    for t in range(10):
        for s in ("short", "long"):
            pi05_request(rng, t, s, 1)
    for t, s, seed in ((4, "long", 101), (8, "short", 102), (3, "medium", 103)):
        pi05_request(rng, t, s, seed)
    pi05_request(rng, 9, "medium", 104)
    return pi05_request(rng, 6, "long", 105)


class Libero:
    def __init__(self):
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        self.Cfg = LiberoEnvConfig
        self.envs = {}

    def env(self, k):
        if k not in self.envs:
            out = self.Cfg(task="libero_10", task_ids=[k]).create_envs(n_envs=1, use_async_envs=False)
            self.envs[k] = out["libero_10"][k]
        return self.envs[k]

    def reset(self, k, seed):
        from lerobot.envs.utils import NEW_ROLLOUT_OPTION
        vec = self.env(k)
        obs, _ = vec.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
        return obs, list(vec.call("task_description"))[0]

    def step(self, k, actions):
        vec, obs, done = self.env(k), None, False
        for a in actions:
            obs, _, term, trunc, _ = vec.step(np.asarray(a, dtype=np.float32)[None])
            if bool(np.asarray(term).any()) or bool(np.asarray(trunc).any()):
                done = True
                break
        return obs, done

    def close(self, k=None):
        for key in ([k] if k is not None else list(self.envs)):
            env = self.envs.pop(key, None)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass


# --------------------------------------------------------------------------------------------------- the tracer

COARSE = [r"vision_model\.embeddings$", r"vision_model\.encoder\.layers\.\d+$", r"vision_model\.post_layernorm$",
          r"multi_modal_projector$", r"language_model\.embed_tokens$", r"language_model\.layers\.\d+$",
          r"language_model\.norm$", r"gemma_expert\.model\.layers\.\d+$", r"gemma_expert\.model\.norm$",
          r"(^|\.)(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)$"]
FINE = [r"vision_model\.embeddings\.", r"vision_model\.encoder\.layers\.0\.", r"language_model\.layers\.0\.",
        r"gemma_expert\.model\.layers\.0\.", r"(^|\.)(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)$"]


class Tracer:
    """Row 0 of every hooked output (and, for the fine modules, of the first input) in execution order."""

    def __init__(self, policy, extra_fine=()):
        import torch
        self.torch, self.policy = torch, policy
        fine_patterns = FINE + list(extra_fine)
        self.rec: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
        self.counts = collections.Counter()
        self.handles, self.patches = [], []
        self.modules = dict(policy.named_modules())
        for name, mod in self.modules.items():
            fine = any(re.search(p, name) for p in fine_patterns)
            if fine or any(re.search(p, name) for p in COARSE):
                self.handles.append(mod.register_forward_hook(self._hook(name, fine)))
        m = policy.model
        self._patch(policy, "_preprocess_images", self._pre_images)
        self._patch(m, "sample_actions", self._sample_actions)
        self._patch(m, "embed_prefix", self._embed_prefix)
        self._patch(m.paligemma_with_expert, "forward", self._pwe_forward)
        self._patch(m, "denoise_step", self._denoise_step)

    def _patch(self, obj, attr, make):
        orig = getattr(obj, attr)
        setattr(obj, attr, make(orig))
        self.patches.append((obj, attr))

    def close(self):
        for h in self.handles:
            h.remove()
        for obj, attr in self.patches:
            delattr(obj, attr)        # back to the class's method

    def add(self, name, t, inp=None):
        torch = self.torch
        if not torch.is_tensor(t):
            return
        i = self.counts[name]
        self.counts[name] += 1
        key = f"{name}#{i}"
        t = t.detach()
        rows_equal = None
        if t.dim() >= 1 and t.shape[0] > 1:
            rows_equal = bool(torch.equal(t, t[:1].expand_as(t)))
        entry = {"row0": t[:1].to("cpu").clone(), "shape": list(t.shape), "dtype": str(t.dtype),
                 "rows_equal": rows_equal}
        if inp is not None and torch.is_tensor(inp):
            entry["in0"] = inp.detach()[:1].to("cpu").clone()
        self.rec[key] = entry

    def _hook(self, name, fine):
        def fn(mod, args, out):
            t = out
            if isinstance(t, (tuple, list)):
                t = t[0]
            if hasattr(t, "last_hidden_state"):
                t = t.last_hidden_state
            inp = args[0] if (fine and args) else None
            self.add("m:" + name, t, inp)
        return fn

    def _pre_images(self, orig):
        def fn(batch):
            images, masks = orig(batch)
            for i, (im, mk) in enumerate(zip(images, masks)):
                self.add(f"f:images.{i}", im)
                self.add(f"f:image_masks.{i}", mk)
            return images, masks
        return fn

    def _sample_actions(self, orig):
        def fn(images, img_masks, tokens, masks, noise=None, **kw):
            self.add("f:tokens", tokens)
            self.add("f:token_masks", masks)
            if noise is not None:
                self.add("f:noise", noise)
            out = orig(images, img_masks, tokens, masks, noise=noise, **kw)
            self.add("f:chunk_padded_dims", out)
            return out
        return fn

    def _embed_prefix(self, orig):
        def fn(*a, **k):
            embs, pad, att = orig(*a, **k)
            self.add("f:prefix_embs", embs)
            self.add("f:prefix_pad_masks", pad)
            self.add("f:prefix_att_masks", att)
            return embs, pad, att
        return fn

    def _pwe_forward(self, orig):
        def fn(*a, **k):
            embeds = k.get("inputs_embeds")
            if embeds is not None and embeds[1] is None:
                self.add("f:prefix_att4d", k.get("attention_mask"))
                self.add("f:prefix_position_ids", k.get("position_ids"))
            outs, pkv = orig(*a, **k)
            if pkv is not None:
                for i, (key, val, *_r) in enumerate(pkv):
                    self.add(f"f:kv.{i}.k", key)
                    self.add(f"f:kv.{i}.v", val)
            return outs, pkv
        return fn

    def _denoise_step(self, orig):
        def fn(prefix_pad_masks, past_key_values, x_t, timestep):
            self.add("f:denoise.x_t", x_t)
            self.add("f:denoise.t", timestep)
            v = orig(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, x_t=x_t, timestep=timestep)
            self.add("f:denoise.v_t", v)
            return v
        return fn


def compare_traces(ref: dict, pad: dict) -> list[dict]:
    import torch
    rows = []
    for key, r in ref.items():
        p = pad.get(key)
        if p is None:
            rows.append({"key": key, "missing_in_padded": True})
            continue
        c = cmp(p["row0"], r["row0"])
        c.update(key=key, shape_padded=p["shape"], dtype=p["dtype"], rows_equal=p["rows_equal"])
        if "in0" in r and "in0" in p:
            c["input_equal"] = bool(r["in0"].shape == p["in0"].shape and torch.equal(r["in0"], p["in0"]))
        rows.append(c)
    for key in pad:
        if key not in ref:
            rows.append({"key": key, "missing_in_ref": True})
    return rows


def stage_of(key: str) -> str:
    if "vision_model" in key or "multi_modal_projector" in key or key.startswith("f:images"):
        return "vision"
    if "language_model" in key or key.startswith(("f:prefix", "f:kv", "f:tokens", "f:token_masks")):
        return "prefix"
    return "suffix"


def write_trace_tsv(rows, path):
    cols = ["key", "equal", "max_abs", "max_rel", "frac_diff", "input_equal", "rows_equal", "dtype", "shape_padded"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def op_alone(kit, tracer_ref, key, sizes=(1, 2, 3, 4, 5, 6, 7, 8, 16)):
    """The first differing operation re-run alone on the batch of one's own input, repeated to n rows: does row 0
    change with n? (A module's own forward, its weight and the identical input.)"""
    torch = kit.torch
    name = key.split("#", 1)[0][2:]
    mod = tracer_ref.modules.get(name)
    entry = tracer_ref.rec[key]
    if mod is None or "in0" not in entry:
        return {"skipped": f"no module or no recorded input for {key}"}
    dev = next(kit.policy.parameters()).device
    x = entry["in0"].to(dev)
    res = {"module": name, "type": type(mod).__name__, "input_shape": list(x.shape), "input_dtype": str(x.dtype)}
    with torch.no_grad(), torch.inference_mode():
        def run(n):
            y = mod(x.expand(n, *x.shape[1:]).contiguous())
            y = y[0] if isinstance(y, (tuple, list)) else y
            return y[:1].to("cpu")
        base = run(1)
        res["vs_rows"] = {str(n): cmp(run(n), base) for n in sizes}
        res["repeat_1"] = cmp(run(1), base)
    return res


# ------------------------------------------------------------------------------------------------------- parts

@part("env")
def p_env(kit):
    import platform

    import torch
    from vlar_harness.servers.common import torch_flags
    info = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "flags": torch_flags(), "device": str(kit.svc.device), "max_batch": kit.n,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "policy_dtype": str(next(kit.policy.model.paligemma_with_expert.paligemma.model.language_model
                                     .parameters()).dtype),
            "vision_dtype": str(next(kit.policy.model.paligemma_with_expert.paligemma.model.vision_tower
                                     .parameters()).dtype),
            "git_commit": os.environ.get("VLAR_GIT_COMMIT")}
    try:
        info["blas"] = str(torch.backends.cuda.preferred_blas_library())
    except Exception as e:
        info["blas"] = repr(e)
    for mod in ("transformers", "lerobot", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
    cfg = kit.policy.model.paligemma_with_expert.paligemma.config
    info["attn_impl"] = {"vision": getattr(cfg.vision_config, "_attn_implementation", None),
                         "text": getattr(cfg.text_config, "_attn_implementation", None)}
    return info


@part("reproduce")
def p_reproduce(kit, reqs):
    req = reqs["R1"]
    A, ca = kit.A(req)
    A2, ca2 = kit.A(req)
    B = kit.B(req)
    C, cc = kit.padded(req)
    C2, cc2 = kit.padded(req)
    np.save(OUT / "R1_A.npy", A), np.save(OUT / "R1_C.npy", C)
    return {"A_repeat": cmp(A2, A), "B_vs_A": cmp(B, A), "C_repeat": cmp(C2, C), "C_vs_A": act_diff(C, A),
            "C_vs_A_normalised": cmp(cc, ca), "A_first10": A[:10].round(4).tolist(),
            "C_first10": C[:10].round(4).tolist(), "A_gripper": A[:, 6].round(3).tolist(),
            "C_gripper": C[:, 6].round(3).tolist()}


@part("confirm")
def p_confirm(kit, reqs):
    out = {}
    for name in ("R2", "R3", "R4", "R5"):
        if name not in reqs:
            continue
        req = reqs[name]
        A, ca = kit.A(req)
        B = kit.B(req)
        C, cc = kit.padded(req)
        out[name] = {"label": reqs["labels"][name], "B_vs_A": cmp(B, A), "C_vs_A": act_diff(C, A),
                     "C_vs_A_normalised": cmp(cc, ca)}
    return out


@part("trace")
def p_trace(kit, reqs, names=("R1", "R4")):
    out = {}

    def traced(req, extra_fine=()):
        tr = Tracer(kit.policy, extra_fine)
        try:
            A, _ = kit.A(req)
            ref = tr.rec
            tr.rec, tr.counts = collections.OrderedDict(), collections.Counter()
            C, _ = kit.padded(req)
            pad = tr.rec
        finally:
            tr.close()
        return A, C, ref, pad

    for name in names:
        if name not in reqs:
            continue
        req = reqs[name]
        A, C, ref, pad = traced(req)
        rows = compare_traces(ref, pad)
        first_mod = next((r for r in rows if r.get("equal") is False and r["key"].startswith("m:")), None)
        zoom = None
        if first_mod is not None and not any(r.get("equal") is False and r.get("input_equal") for r in rows):
            mod_name = first_mod["key"].split("#", 1)[0][2:]
            A, C, ref, pad = traced(req, [re.escape(mod_name) + r"\."])
            rows = compare_traces(ref, pad)
            zoom = mod_name
        write_trace_tsv(rows, OUT / f"trace_{name}.tsv")
        first = next((r for r in rows if r.get("equal") is False), None)
        first_op = next((r for r in rows if r.get("equal") is False and r.get("input_equal") is True), None)
        by_stage = collections.defaultdict(lambda: {"entries": 0, "differ": 0, "max_abs": 0.0, "max_rel": 0.0})
        for r in rows:
            s = by_stage[stage_of(r["key"])]
            s["entries"] += 1
            if r.get("equal") is False:
                s["differ"] += 1
                s["max_abs"] = max(s["max_abs"], r.get("max_abs") or 0.0)
                s["max_rel"] = max(s["max_rel"], r.get("max_rel") or 0.0)
        steps = [{"x_t": r.get("max_abs"), "key": r["key"]} for r in rows if r["key"].startswith("f:denoise.x_t")]
        vts = [r.get("max_abs") for r in rows if r["key"].startswith("f:denoise.v_t")]
        kv = [r.get("max_rel") for r in rows if r["key"].startswith("f:kv.")]
        lm = [r.get("max_rel") for r in rows if re.search(r"language_model\.layers\.\d+#0$", r["key"])]
        vis = [r.get("max_rel") for r in rows if re.search(r"encoder\.layers\.\d+#0$", r["key"])]
        rows_differ = [r["key"] for r in rows if r.get("rows_equal") is False]
        res = {"entries": len(rows), "zoomed_into": zoom, "first_difference": first, "first_op_with_equal_input": first_op,
               "by_stage": dict(by_stage), "x_t_max_abs_per_step": [s["x_t"] for s in steps],
               "v_t_max_abs_per_step": vts, "kv_max_rel_per_layer": kv, "lm_layer_max_rel(camera/call 0)": lm,
               "vision_layer_max_rel(camera 0)": vis, "padded_rows_not_all_equal": rows_differ[:20],
               "missing": [r["key"] for r in rows if "missing_in_padded" in r or "missing_in_ref" in r][:20],
               "final_C_vs_A": act_diff(C, A)}
        if first_op is not None:
            try:
                tr_ref = Tracer.__new__(Tracer)
                tr_ref.modules, tr_ref.rec = dict(kit.policy.named_modules()), ref
                res["first_op_alone"] = op_alone(kit, tr_ref, first_op["key"])
            except Exception:
                res["first_op_alone"] = traceback.format_exc()
        # the inputs, bit for bit
        res["inputs"] = {k: next(({"equal": r.get("equal"), "max_abs": r.get("max_abs")} for r in rows
                                  if r["key"] == k), None)
                         for k in ("f:noise#0", "f:tokens#0", "f:token_masks#0", "f:images.0#0", "f:images.1#0",
                                   "f:images.2#0", "f:prefix_att4d#0", "f:prefix_position_ids#0")}
        out[name] = res
    return out


@part("stages")
def p_stages(kit, reqs, names=("R1", "R4")):
    import torch
    from lerobot.policies.common.flow_matching import euler_integrate
    from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
    from transformers import DynamicCache
    model, policy = kit.policy.model, kit.policy

    def prefix(batch):
        images, img_masks = policy._preprocess_images(batch)
        embs, pad, att = model.embed_prefix(images, img_masks, batch[TOKENS], batch[MASK])
        att4d = prepare_attention_masks_4d(make_att_2d_masks(pad, att))
        pos = torch.cumsum(pad, dim=1) - 1
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, pkv = model.paligemma_with_expert.forward(attention_mask=att4d, position_ids=pos, past_key_values=None,
                                                     inputs_embeds=[embs, None], use_cache=True)
        return pad, pkv

    def rows_of(pad, pkv, n):
        cache = DynamicCache(tuple((k[:1].expand(n, *k.shape[1:]).contiguous(),
                                    v[:1].expand(n, *v.shape[1:]).contiguous(), *r) for k, v, *r in pkv))
        return pad[:1].expand(n, *pad.shape[1:]).contiguous(), cache

    def denoise(pad, pkv, noise):
        return euler_integrate(lambda x, t: model.denoise_step(prefix_pad_masks=pad, past_key_values=pkv, x_t=x,
                                                               timestep=t), noise, model.config.num_inference_steps)

    out = {}
    for name in names:
        if name not in reqs:
            continue
        req = reqs[name]
        obs = kit.prepared(req)
        n = kit.n
        b1, bn = kit.batch([obs]), kit.batch([obs] * n)
        noise1, noisen = kit.svc.noise([req["seed"]]), kit.svc.noise([req["seed"]] * n)
        with kit.svc.lock, torch.no_grad(), torch.inference_mode():
            full1 = policy.predict_action_chunk(b1, noise=noise1)[:1].to("cpu")
            fulln = policy.predict_action_chunk(bn, noise=noisen)[:1].to("cpu")
            pad1, kv1 = prefix(b1)
            padn, kvn = prefix(bn)
            kv_cmp = [cmp(kn[:1], k1).get("max_rel") for (kn, _vn, *_a), (k1, _v1, *_b) in zip(kvn, kv1)]
            dim = policy.config.output_features["action"].shape[0]
            x11 = denoise(pad1, kv1, noise1)[:1, :, :dim].to("cpu")
            p, c = rows_of(padn, kvn, 1)
            x81 = denoise(p, c, noise1)[:1, :, :dim].to("cpu")            # prefix at n rows, denoising at 1
            p, c = rows_of(pad1, kv1, n)
            x18 = denoise(p, c, noisen)[:1, :, :dim].to("cpu")            # prefix at 1 row, denoising at n
            x88 = denoise(padn, kvn, noisen)[:1, :, :dim].to("cpu")
        out[name] = {"split_1_1_vs_full_1": cmp(x11, full1), "split_n_n_vs_full_n": cmp(x88, fulln),
                     "full_n_vs_full_1": cmp(fulln, full1),
                     "prefix_n__denoise_1_vs_1": cmp(x81, full1), "prefix_1__denoise_n_vs_1": cmp(x18, full1),
                     "kv_row0_max_rel_per_layer_n_vs_1(keys)": kv_cmp}
    return out


@part("sweep")
def p_sweep(kit, reqs, names=("R1", "R4"), sizes=(1, 2, 3, 4, 5, 6, 7, 8, 12, 16)):
    out = {}
    for name in names:
        if name not in reqs:
            continue
        A, ca = kit.A(reqs[name])
        out[name] = {}
        for n in sizes:
            C, cc = kit.padded(reqs[name], n)
            d = act_diff(C, A)
            out[name][str(n)] = {"equal": d["equal"], "max_abs": d["max_abs"], "per_dim_max": d["per_dim_max"],
                                 "normalised_max_abs": cmp(cc, ca)["max_abs"]}
    return out


@part("sensitivity")
def p_sensitivity(kit, reqs, names=("R1", "R4")):
    import torch
    out = {}
    g = np.random.default_rng(11)
    for name in names:
        if name not in reqs:
            continue
        req = reqs[name]
        obs = kit.prepared(req)
        b1 = kit.batch([obs])
        noise = kit.svc.noise([req["seed"]])
        base = kit.predict(b1, noise).to("cpu")
        res = {"repeat": cmp(kit.predict(b1, noise).to("cpu"), base)}
        flat = noise.reshape(-1)
        for i in range(4):
            k = int(g.integers(0, flat.numel()))
            n2 = noise.clone()
            f2 = n2.reshape(-1)
            f2[k] = torch.nextafter(f2[k], torch.tensor(float("inf"), device=f2.device, dtype=f2.dtype))
            res[f"noise_one_ulp_{i}"] = cmp(kit.predict(b1, n2).to("cpu"), base)
        res["noise_times_1+2^-20"] = cmp(kit.predict(b1, noise * (1 + 2.0 ** -20)).to("cpu"), base)
        cams = [k for k in kit.svc.cameras if k in b1]
        for i in range(3):
            cam = cams[i % len(cams)]
            b2 = dict(b1)
            img = b1[cam].clone()
            idx = tuple(int(g.integers(0, s)) for s in img.shape)
            img[idx] = torch.clamp(img[idx] + 2.0 ** -20, 0, 1) if img.is_floating_point() else img[idx]
            b2[cam] = img
            res[f"image_pixel_2^-20_{i}({cam.split('.')[-1]})"] = cmp(kit.predict(b2, noise).to("cpu"), base)
        out[name] = res
    return out


@part("profile")
def p_profile(kit, reqs, name="R1"):
    import torch
    from torch.profiler import ProfilerActivity, profile
    obs = kit.prepared(reqs[name])
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    kernels = {}
    for n in (1, kit.n):
        b, noise = kit.batch([obs] * n), kit.svc.noise([reqs[name]["seed"]] * n)
        kit.predict(b, noise)
        with profile(activities=acts) as prof:
            kit.predict(b, noise)
        cnt = collections.Counter()
        for e in prof.events():
            if "CUDA" in str(getattr(e, "device_type", "")) or not torch.cuda.is_available():
                cnt[e.name] += 1
        kernels[str(n)] = dict(cnt)
    k1, kn = set(kernels["1"]), set(kernels[str(kit.n)])
    (OUT / "profile_kernels.json").write_text(json.dumps(kernels, indent=1))
    gemm = lambda s: sorted(k for k in s if re.search(r"gemm|cutlass|xmma|sm8\d|splitk|s16816|ampere|wgmma", k, re.I))  # noqa: E731
    return {"distinct_kernels": {"1": len(k1), str(kit.n): len(kn)}, "only_at_1": sorted(k1 - kn)[:60],
            f"only_at_{kit.n}": sorted(kn - k1)[:60], "gemm_at_1": gemm(k1)[:60], f"gemm_at_{kit.n}": gemm(kn)[:60]}


@part("survey")
def p_survey(kit, reqs, lib, tasks, points, random_states):
    from pi05_testkit import pi05_request
    cases = []
    for k in tasks:
        try:
            obs, task = lib.reset(k, 1000 + k)
            for d in range(points):
                req = {"obs": obs, "task": task, "seed": 5000 + 100 * k + d}
                A, ca = kit.A(req)
                ta = kit.last_model_s
                C, cc = kit.padded(req)
                tc = kit.last_model_s
                diff = act_diff(C, A)
                cases.append({"A_model_s": ta, "C_model_s": tc, "kind": "real", "task": k, "point": d, "seed": req["seed"], "max_abs": diff["max_abs"],
                              "equal": diff["equal"], "per_dim_max": diff["per_dim_max"],
                              "per_dim_argmax_t": diff["per_dim_argmax_t"], "first10_max": diff["first10_max"],
                              "normalised_max_abs": cmp(cc, ca)["max_abs"],
                              "A_gripper_first10": A[:10, 6].round(3).tolist(),
                              "C_gripper_first10": C[:10, 6].round(3).tolist()})
                reqs.setdefault("survey", {})[(k, d)] = req
                obs, done = lib.step(k, A[:10])
                if done or obs is None:
                    break
        finally:
            lib.close(k)
        log(f"survey task {k}: {[round(c['max_abs'], 4) for c in cases if c['task'] == k and c['kind'] == 'real']}")
        save_partial = {"cases": cases}
        SUMMARY["parts"]["survey_partial"] = save_partial
        save()
    rng = np.random.default_rng(1)
    for k in tasks:
        for s in random_states:
            req = pi05_request(rng, k, s, 7000 + 10 * k + len(s))
            A, ca = kit.A(req)
            C, cc = kit.padded(req)
            diff = act_diff(C, A)
            cases.append({"kind": "random", "task": k, "state": s, "seed": req["seed"], "max_abs": diff["max_abs"],
                          "equal": diff["equal"], "per_dim_max": diff["per_dim_max"],
                          "per_dim_argmax_t": diff["per_dim_argmax_t"], "first10_max": diff["first10_max"],
                          "normalised_max_abs": cmp(cc, ca)["max_abs"]})
    SUMMARY["parts"].pop("survey_partial", None)

    def dist(kind):
        v = np.array([c["max_abs"] for c in cases if c["kind"] == kind])
        f10 = np.array([c["first10_max"] for c in cases if c["kind"] == kind])
        if not v.size:
            return None
        q = lambda a, p: float(np.percentile(a, p))   # noqa: E731
        return {"n": int(v.size), "equal": int(sum(c["equal"] for c in cases if c["kind"] == kind)),
                "median": q(v, 50), "p90": q(v, 90), "max": float(v.max()), "over_0.05": int((v > 0.05).sum()),
                "over_0.5": int((v > 0.5).sum()), "first10_median": q(f10, 50), "first10_max": float(f10.max()),
                "first10_over_0.05": int((f10 > 0.05).sum())}
    ta = [c["A_model_s"] for c in cases if c.get("A_model_s")]
    tc = [c["C_model_s"] for c in cases if c.get("C_model_s")]
    timing = {"median_model_s_1_row": float(np.median(ta)) if ta else None,
              f"median_model_s_{kit.n}_rows": float(np.median(tc)) if tc else None}
    return {"real": dist("real"), "random": dist("random"), "timing": timing, "cases": cases}


@part("reduction")
def p_reduction(kit, reqs, extra):
    import torch
    if not torch.cuda.is_available():
        return {"skipped": "CPU"}
    before = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    out = {"flag_before": before}
    try:
        base = {name: kit.A(req) for name, req in extra.items()}
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        for name, req in extra.items():
            A, ca = kit.A(req)
            C, cc = kit.padded(req)
            out[name] = {"C_vs_A": act_diff(C, A)["max_abs"], "A_vs_A_default": cmp(A, base[name][0])["max_abs"],
                         "C_vs_A_normalised": cmp(cc, ca)["max_abs"]}
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = before
    return out


@part("fp32")
def p_fp32(kit, reqs, extra):
    import torch
    bf = {name: (kit.A(req), kit.padded(req)) for name, req in extra.items()}
    kit.policy.to(torch.float32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out = {"dtype_after": str(next(kit.policy.model.paligemma_with_expert.paligemma.model.language_model
                                   .parameters()).dtype)}
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        for mode, flag in (("tf32_off", False), ("tf32_on", True)):
            torch.backends.cuda.matmul.allow_tf32 = flag
            torch.backends.cudnn.allow_tf32 = flag
            res = {}
            for name, req in extra.items():
                (A16, ca16), (C16, cc16) = bf[name]
                A, ca = kit.A(req)
                C, cc = kit.padded(req)
                res[name] = {"C32_vs_A32": act_diff(C, A), "C32_vs_A32_normalised": cmp(cc, ca)["max_abs"],
                             "A16_vs_A32": cmp(A16, A)["max_abs"], "C16_vs_A32": cmp(C16, A)["max_abs"],
                             "A16_vs_A32_normalised": cmp(ca16, ca)["max_abs"],
                             "C16_vs_A32_normalised": cmp(cc16, ca)["max_abs"]}
                if name == "R1":
                    np.save(OUT / f"R1_A32_{mode}.npy", A), np.save(OUT / f"R1_C32_{mode}.npy", C)
            out[mode] = res
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    return out


# -------------------------------------------------------------------------------------------------------- main

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=os.environ.get("VLAR_REPO", "/opt/vla-router"))
    ap.add_argument("--tiny", action="store_true", help="the tiny random model on the CPU (the Mac)")
    ap.add_argument("--quick", action="store_true", help="fewer survey cases")
    ap.add_argument("--parts", default="all")
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(a.repo) / "tests" / "reference"))
    sys.path.insert(0, str(Path(a.repo) / "src"))
    t0 = time.time()
    from pi05_testkit import pi05_request
    from vlar_harness.servers.pi05_server import PI05Batching, PI05Service
    if a.tiny:
        from pi05_testkit import make_tiny_pi05
        svc = make_tiny_pi05(PI05Batching(max_batch=8))
    else:
        svc = PI05Service.load(os.environ["VLAR_PI05_CHECKPOINT"], os.environ.get("VLAR_TEST_DEVICE", "cuda"),
                               PI05Batching(max_batch=8))
    SUMMARY["load_s"] = round(time.time() - t0, 1)
    kit = Kit(svc, a.tiny)
    want = None if a.parts == "all" else set(a.parts.split(","))
    on = lambda p: want is None or p in want    # noqa: E731

    reqs = {"labels": {}}
    reqs["R1"] = r1_request(pi05_request)
    reqs["labels"]["R1"] = "random obs, task 6, long state, seed 105 (the failing test request)"
    rng = np.random.default_rng(12345)
    reqs["R2"] = pi05_request(rng, 6, "long", 106)
    reqs["labels"]["R2"] = "random obs, task 6, long state, seed 106 (second request, same task)"
    reqs["R3"] = pi05_request(rng, 2, "medium", 107)
    reqs["labels"]["R3"] = "random obs, task 2, medium state, seed 107 (second task)"
    lib = None
    try:
        lib = Libero()
        for name, k in (("R4", 6), ("R5", 2)):
            obs, task = lib.reset(k, 1000 + k)
            reqs[name] = {"obs": obs, "task": task, "seed": 900 + k}
            reqs["labels"][name] = f"real LIBERO obs, task {k}, reset with seed {1000 + k}"
            lib.close(k)
    except Exception:
        SUMMARY["errors"]["libero_setup"] = traceback.format_exc()
        log(SUMMARY["errors"]["libero_setup"])
    save()

    if on("env"):
        p_env(kit)
    if on("reproduce"):
        p_reproduce(kit, reqs)
    if on("confirm"):
        p_confirm(kit, reqs)
    if on("trace"):
        p_trace(kit, reqs)
    if on("stages"):
        p_stages(kit, reqs)
    if on("sweep"):
        p_sweep(kit, reqs)
    if on("sensitivity"):
        p_sensitivity(kit, reqs)
    if on("profile"):
        p_profile(kit, reqs)
    survey = None
    if on("survey") and lib is not None:
        tasks = [6, 2] if a.quick else list(range(10))
        survey = p_survey(kit, reqs, lib, tasks, 2 if a.quick else 5, ("short",) if a.quick else ("short", "long"))
    extra = {n: reqs[n] for n in ("R1", "R4") if n in reqs}
    if survey:
        top = sorted((c for c in survey["cases"] if c["kind"] == "real"), key=lambda c: -c["max_abs"])[:3]
        for c in top:
            extra[f"real_t{c['task']}_p{c['point']}"] = reqs["survey"][(c["task"], c["point"])]
    if on("reduction"):
        p_reduction(kit, reqs, extra)
    if on("fp32"):
        p_fp32(kit, reqs, extra)
    SUMMARY["total_s"] = round(time.time() - t0, 1)
    save()
    kit.one.close()
    svc.close()
    if lib is not None:
        lib.close()
    log(f"done in {SUMMARY['total_s']} s; errors in parts: {sorted(SUMMARY['errors'])}")


if __name__ == "__main__":
    main()
