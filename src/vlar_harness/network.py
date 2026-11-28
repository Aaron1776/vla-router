"""The simulated network between the robot and the remote π0.5 (docs/SPEC.md §5, amendments 13-14).

π0.5 is "remote" only in the clock: its server runs on the same machine, and this module decides, for each
attempt of each request, whether it gets through and how long it takes. Per attempt:

- a failure draw: the request is dropped with probability `p_fail`;
- a round-trip time from a log-normal with median `rtt_median_s`, shaped so that a share `rtt_tail_prob` of
  round trips exceed `rtt_tail_s`;
- attempt time = upload + round trip + download + π0.5's inference time (its batch-size-1 median). An attempt
  whose time exceeds the timeout fails too.

A failure is noticed only when the timeout expires, and the next attempt follows after a backoff. A routed
escalation makes 1 + `max_retries` attempts and is then abandoned (the engine hands the segment to X-VLA);
π0.5 alone (`pi05_only`) retries until a request gets through, up to `max_attempts_pi05_only` attempts, after
which the episode ends with end reason `network`.

Retries are the same request. The π0.5 server is called once, for the attempt that gets through, with the
request's own noise seed, so failures never change the executed actions, only the clock. Each attempt's
draws come from `seeding.network_attempt_seed(rollout seed, step, model, call, attempt)`: a retry gets fresh
draws, the same rollout seed reproduces the same failure pattern, one attempt's draws never depend on how many
other requests or attempts came before, and nothing here reads the clock or the overhead profile, so
re-scoring a saved event log under any profile is exact.

Upload size (amendment 14): the median, over requests, of the JPEG size of the two 224 px images π0.5
actually consumes (quality 90), measured once in Phase 0 step 2 and stored in the network option as
`upload_image_bytes`, plus a fixed allowance for the state, task text and framing. It sets only the clock;
the model's input is unchanged.

Pure Python (no NumPy) except the JPEG helpers, so the clock and labels run in any environment.
"""

from __future__ import annotations

import io
import math
import statistics
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Callable, Optional, Protocol, Sequence

from . import seeding

# Identity of the one remote request type: the π0.5 call (seeding's model and call type).
REQUEST_MODEL, REQUEST_CALL = "pi05", "act"
CAUSES = ("dropped", "timeout")

_STD_NORMAL = NormalDist()


@dataclass(frozen=True)
class NetworkParams:
    uplink_mbps: float
    downlink_mbps: float
    rtt_median_s: float
    rtt_tail_s: float
    rtt_tail_prob: float
    p_fail: float
    timeout_s: float
    backoff_s: float
    max_retries: int
    max_attempts_pi05_only: int
    upload_bytes: int              # images + state, text and framing
    download_bytes: int
    pi05_infer_s: float            # π0.5's batch-size-1 median inference time (config `latency`)
    rtt_sigma: float = field(init=False)
    upload_s: float = field(init=False)
    download_s: float = field(init=False)

    def __post_init__(self) -> None:
        if not (0 < self.rtt_median_s < self.rtt_tail_s) or not (0 < self.rtt_tail_prob < 0.5):
            raise ValueError("network round trip: need 0 < median < tail and 0 < tail probability < 0.5")
        if self.max_retries < 0 or self.max_attempts_pi05_only < 1:
            raise ValueError("network attempts: max_retries >= 0 and max_attempts_pi05_only >= 1")
        sigma = math.log(self.rtt_tail_s / self.rtt_median_s) / _STD_NORMAL.inv_cdf(1.0 - self.rtt_tail_prob)
        object.__setattr__(self, "rtt_sigma", sigma)
        object.__setattr__(self, "upload_s", self.upload_bytes * 8 / (self.uplink_mbps * 1e6))
        object.__setattr__(self, "download_s", self.download_bytes * 8 / (self.downlink_mbps * 1e6))

    @staticmethod
    def from_config(cfg) -> "NetworkParams":
        net = cfg.option("network")
        return NetworkParams(
            uplink_mbps=float(net["uplink_mbps"]),
            downlink_mbps=float(net["downlink_mbps"]),
            rtt_median_s=float(net["rtt_median_s"]),
            rtt_tail_s=float(net["rtt_tail_s"]),
            rtt_tail_prob=float(net["rtt_tail_prob"]),
            p_fail=float(net["p_fail"]),
            timeout_s=float(net["timeout_s"]),
            backoff_s=float(net["backoff_s"]),
            max_retries=int(net["max_retries"]),
            max_attempts_pi05_only=int(net["max_attempts_pi05_only"]),
            upload_bytes=int(net["upload_image_bytes"]) + int(net["upload_other_bytes"]),
            download_bytes=int(net["download_bytes"]),
            pi05_infer_s=float(cfg["latency"]["pi05_infer_s"]),
        )

    @property
    def max_attempts_routed(self) -> int:
        return 1 + self.max_retries

    def attempt_time_s(self, rtt_s: float) -> float:
        """Time of an attempt that gets through: the robot waits for the upload, the round trip, the download
        and π0.5's inference. The one formula used both to decide timeouts and to charge the clock."""
        return self.upload_s + rtt_s + self.download_s + self.pi05_infer_s


def attempt_record(attempt: int, ok: bool, *, cause: Optional[str] = None, u_fail: Optional[float] = None,
                   rtt_s: float = 0.0) -> dict:
    """One attempt as it is logged (and charged by clock.request_s)."""
    if ok != (cause is None) or (cause is not None and cause not in CAUSES):
        raise ValueError(f"attempt {attempt}: ok={ok} with cause {cause!r}")
    return {"attempt": int(attempt), "ok": bool(ok), "cause": cause,
            "u_fail": None if u_fail is None else float(u_fail), "rtt_s": float(rtt_s)}


def draw_attempt(params: NetworkParams, rollout_seed: int, step: int, attempt: int,
                 model: str = REQUEST_MODEL, call: str = REQUEST_CALL) -> dict:
    """The network's draws and outcome for one attempt of one request. A pure function of its arguments."""
    seed = seeding.network_attempt_seed(rollout_seed, step, model, call, attempt)
    u_fail = seeding.uniform(seed, "fail")
    rtt = params.rtt_median_s * math.exp(params.rtt_sigma * _STD_NORMAL.inv_cdf(seeding.uniform(seed, "rtt")))
    if u_fail < params.p_fail:
        cause = "dropped"
    elif params.attempt_time_s(rtt) > params.timeout_s:
        cause = "timeout"
    else:
        cause = None
    return attempt_record(attempt, cause is None, cause=cause, u_fail=u_fail, rtt_s=rtt)


@dataclass
class RemoteOutcome:
    ok: bool
    actions: Optional[Any]                  # π0.5's 50 x 7 actions when ok
    attempts: list = field(default_factory=list)


class RemoteLink(Protocol):
    def request(self, call: Callable[[], Any], *, step: int, rollout_seed: int,
                max_attempts: int) -> RemoteOutcome: ...


class SimulatedNetwork:
    """The production link. `call()` runs the π0.5 request (with its own noise seed) once, for the attempt
    that gets through."""

    def __init__(self, params: NetworkParams):
        self.params = params

    def request(self, call, *, step, rollout_seed, max_attempts) -> RemoteOutcome:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        attempts = []
        for i in range(max_attempts):
            a = draw_attempt(self.params, rollout_seed, step, i)
            attempts.append(a)
            if a["ok"]:
                return RemoteOutcome(True, call(), attempts)
        return RemoteOutcome(False, None, attempts)


class DirectLink:
    """Tests only: every request gets through on its first attempt with a zero round trip."""

    def request(self, call, *, step, rollout_seed, max_attempts=1) -> RemoteOutcome:
        return RemoteOutcome(True, call(), [attempt_record(0, True)])


def check_outcome(outcome: RemoteOutcome, max_attempts: int) -> None:
    """A link's outcome must be well formed: attempts numbered from 0, every attempt but the last failed, the
    last one decides `ok`, and no more attempts than allowed."""
    att = outcome.attempts
    if not att or len(att) > max_attempts:
        raise ValueError(f"remote outcome has {len(att)} attempts (allowed 1..{max_attempts})")
    for i, a in enumerate(att):
        if a.get("attempt") != i or not isinstance(a.get("rtt_s"), float):
            raise ValueError(f"remote attempt {i} is malformed: {a}")
        if a["ok"] != (i == len(att) - 1 and outcome.ok):
            raise ValueError(f"remote attempt {i}: ok={a['ok']} does not fit outcome ok={outcome.ok}")
    if not outcome.ok and len(att) != max_attempts:
        raise ValueError(f"a failed request must use all {max_attempts} attempts, not {len(att)}")


# ---------------------------------------------------------------- statistics (tests and reports)

def p_attempt_fails(params: NetworkParams) -> float:
    """Probability that one attempt fails: dropped, or slower than the timeout."""
    slack = params.timeout_s - params.upload_s - params.download_s - params.pi05_infer_s
    if slack <= 0:
        p_slow = 1.0
    else:
        p_slow = 1.0 - _STD_NORMAL.cdf(math.log(slack / params.rtt_median_s) / params.rtt_sigma)
    return params.p_fail + (1.0 - params.p_fail) * p_slow


# ---------------------------------------------------------------- upload size (Phase 0 step 2)

def jpeg_size(image: Any, quality: int = 90) -> int:
    """Bytes of one RGB uint8 image (H x W x 3) encoded as JPEG by Pillow at `quality`, Pillow's defaults
    otherwise (baseline, 4:2:0 chroma subsampling, no optimisation)."""
    import numpy as np
    from PIL import Image

    a = np.asarray(image)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected an H x W x 3 uint8 image, got {a.dtype} {a.shape}")
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(a)).save(buf, format="JPEG", quality=int(quality))
    return buf.tell()


def median_upload_image_bytes(per_request: Sequence[Sequence[int]]) -> int:
    """Amendment 14: the median, over requests, of the summed JPEG sizes of each request's images."""
    totals = [int(sum(r)) for r in per_request]
    if not totals:
        raise ValueError("no requests measured")
    return int(math.floor(statistics.median(totals) + 0.5))
