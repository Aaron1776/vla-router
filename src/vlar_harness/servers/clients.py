"""Simulator-side HTTP clients for the two model servers (localhost)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


class ServerError(RuntimeError):
    pass


def _get_info(session, base: str, name: str) -> dict:
    resp = session.get(base + "/info", timeout=30)
    if resp.status_code != 200:
        raise ServerError(f"{name} /info failed ({resp.status_code}): {resp.text[:500]}")
    return resp.json()


class XVLAClient:
    """Talks to servers.xvla_server. Payloads are the authors' query format (adapters.xvla)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8777, timeout_s: float = 600.0):
        import requests

        self.base = f"http://{host}:{port}"
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def _post(self, path: str, payload: dict) -> dict:
        resp = self._session.post(self.base + path, json=payload, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise ServerError(f"X-VLA {path} failed ({resp.status_code}): {resp.text[:500]}")
        return resp.json()

    def act(self, payload: dict, seed: Optional[int] = None) -> dict:
        return self._post("/act", payload if seed is None else {**payload, "seed": int(seed)})

    def sample(self, payload: dict, seeds: Sequence[int]) -> dict:
        body = {k: v for k, v in payload.items() if k != "seed"}
        return self._post("/sample", {**body, "seeds": [int(s) for s in seeds]})

    def features(self, payload: dict) -> dict:
        return self._post("/features", payload)

    def info(self) -> dict:
        return _get_info(self._session, self.base, "X-VLA")


class PI05Client:
    """Talks to servers.pi05_server with a json_numpy body (arrays keep their dtype and shape)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8778, timeout_s: float = 600.0):
        import requests

        self.base = f"http://{host}:{port}"
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def act(self, raw_obs: dict, task: str, seed: Optional[int] = None) -> dict[str, Any]:
        import json_numpy

        body = json_numpy.dumps({"obs": raw_obs, "task": task, "seed": None if seed is None else int(seed)})
        resp = self._session.post(self.base + "/act", data=body, headers={"Content-Type": "application/json"},
                                  timeout=self.timeout_s)
        if resp.status_code != 200:
            raise ServerError(f"π0.5 /act failed ({resp.status_code}): {resp.text[:500]}")
        out = json_numpy.loads(resp.content)
        out["actions"] = np.asarray(out["actions"], dtype=np.float32)
        return out

    def info(self) -> dict:
        return _get_info(self._session, self.base, "π0.5")

    def upload_bytes(self, raw_obs: dict, task: str, quality: int = 90) -> dict[str, Any]:
        """Phase 0 step 2: JPEG sizes of the images π0.5 consumes for this observation (SPEC §5, amendment 14)."""
        import json_numpy

        body = json_numpy.dumps({"obs": raw_obs, "task": task, "quality": int(quality)})
        resp = self._session.post(self.base + "/upload_bytes", data=body,
                                  headers={"Content-Type": "application/json"}, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise ServerError(f"π0.5 /upload_bytes failed ({resp.status_code}): {resp.text[:500]}")
        return resp.json()
