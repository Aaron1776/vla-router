#!/usr/bin/env python3
"""The π0.5 gate for evaluation on the DGX Spark: a forwarding proxy in front of the π0.5 server that lets at most
one `/act` be in flight (ops-side; Python standard library only; promoted from the debug session's prototype).

Why: on the GB10, π0.5 at the calibrated `max_batch` 4 is not batch-invariant (step 210, row
`pi05-batch-invariance-geometry`): a request's actions depend on what shares its batch. A request alone in its batch
(padded with copies of itself, read at position 0) is bit-identical on every repeat. With the gate the server's
batcher only ever holds one waiting request, so every evaluation request is alone in its batch.

    python3 pi05_gate.py --listen 8779 --backend 127.0.0.1:8778 --log FILE [--mode gate|pass]

- `/act`: forwarded one at a time (a lock) in mode `gate`; at once in mode `pass` (benchmarks only, never evaluation).
- `GET /info`: the server's answer with `"gate": {"mode", "max_in_flight", "listen", "backend"}` added, so a client can
  check the gate is on and record it in its provenance (`evaluate.py` refuses to run without it).
- Everything else (`/upload_bytes`, ...) is forwarded unchanged.
- The log has one JSON line per `/act`: seq, arrival / forward / done times, wait_s (the gate's queue), service_s,
  in_flight_on_arrival, HTTP status, request bytes and the SHA-256 of the response's `actions`.
"""
import argparse
import hashlib
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
GATE = threading.Lock()          # one /act in flight (mode gate)
META = threading.Lock()          # sequence numbers, the waiting count and the log file
SEQ, WAITING = [0], [0]
LOG = None


def forward(method, path, body, headers):
    host, port = ARGS.backend.rsplit(":", 1)
    conn = http.client.HTTPConnection(host, int(port), timeout=900)
    try:
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.getheader("Content-Type", "application/json"), resp.read()
    finally:
        conn.close()


def gate_info():
    return {"mode": ARGS.mode, "max_in_flight": 1 if ARGS.mode == "gate" else None,
            "listen": ARGS.listen, "backend": ARGS.backend}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _reply(self, status, ctype, data):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def _error(self, e):
        self._reply(502, "application/json", json.dumps({"error": f"gate: {e!r}"}).encode())

    def do_GET(self):
        try:
            status, ctype, data = forward("GET", self.path, None, {})
            if self.path.rstrip("/") == "/info" and status == 200:
                info = json.loads(data)
                info["gate"] = gate_info()
                data = json.dumps(info).encode()
            self._reply(status, ctype, data)
        except Exception as e:  # noqa: BLE001
            self._error(e)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        if self.path != "/act":
            try:
                self._reply(*forward("POST", self.path, body, headers))
            except Exception as e:  # noqa: BLE001
                self._error(e)
            return
        t_arrive = time.time()
        with META:
            SEQ[0] += 1
            seq = SEQ[0]
            WAITING[0] += 1
            ahead = WAITING[0] - 1
        gated = ARGS.mode == "gate"
        if gated:
            GATE.acquire()
        t_forward = time.time()
        try:
            status, ctype, data = forward("POST", self.path, body, headers)
        except Exception as e:  # noqa: BLE001
            status, ctype, data = 502, "application/json", json.dumps({"error": f"gate: {e!r}"}).encode()
        finally:
            t_done = time.time()
            if gated:
                GATE.release()
            with META:
                WAITING[0] -= 1
        self._reply(status, ctype, data)
        rec = {"seq": seq, "mode": ARGS.mode, "t_arrive": t_arrive, "t_forward": t_forward, "t_done": t_done,
               "wait_s": t_forward - t_arrive, "service_s": t_done - t_forward, "in_flight_on_arrival": ahead,
               "status": status, "request_bytes": len(body)}
        try:
            rec["actions_sha256"] = hashlib.sha256(
                json.dumps(json.loads(data).get("actions"), sort_keys=True).encode()).hexdigest()
        except Exception as e:  # noqa: BLE001
            rec["parse_error"] = repr(e)
        with META:
            LOG.write(json.dumps(rec) + "\n")
            LOG.flush()


def main():
    global ARGS, LOG
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listen", type=int, default=8779, help="port on 127.0.0.1")
    ap.add_argument("--backend", default="127.0.0.1:8778", help="host:port of the π0.5 server")
    ap.add_argument("--mode", choices=("gate", "pass"), default="gate")
    ap.add_argument("--log", required=True)
    ARGS = ap.parse_args()
    LOG = open(ARGS.log, "a")
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.listen), Handler)
    srv.daemon_threads = True
    print(f"pi05 gate: mode {ARGS.mode}, 127.0.0.1:{ARGS.listen} -> {ARGS.backend}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
