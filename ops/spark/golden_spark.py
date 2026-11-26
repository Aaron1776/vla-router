"""A Spark golden set from the Spark's own base reproduction, checked for exact replay (vlar_ops.golden, unchanged).

    python golden_spark.py select --records DIR --out GOLDEN_DIR [--exclude-tasks 3,8]
        golden.select (its seed, one decision point per task) over the reproduction's runs, points and steps tables,
        with the excluded tasks' runs left out; writes GOLDEN_DIR/golden.json and golden.npz
    python golden_spark.py check --records DIR --golden GOLDEN_DIR --repeat K
        one golden.check pass in this process (fresh simulators, production servers); appends one JSON line
        (pid, container id, process start) to GOLDEN_DIR/<--results-file>
"""
import argparse, glob, json, os, time

START = time.time()     # this process's start (one process per repeat)
from pathlib import Path


def container_id():
    """This container's id (12 hex) from the hostname file Docker mounts into it; not the hostname, which under
    --network host is the machine's."""
    import re
    try:
        for line in open("/proc/self/mountinfo"):
            m = re.search(r"/containers/([0-9a-f]{64})/hostname", line)
            if m:
                return m.group(1)[:12]
    except OSError:
        pass
    return None


def table(records, name):
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(f"{records}/base/*/records/{name}.parquet")):
        rows += pq.read_table(f).to_pylist()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "check"])
    ap.add_argument("--records", required=True); ap.add_argument("--golden"); ap.add_argument("--out")
    ap.add_argument("--exclude-tasks", default="3,8"); ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--results-file", default="repeats.jsonl")
    a = ap.parse_args()
    from vlar_ops import golden
    if a.cmd == "select":
        skip = {int(t) for t in a.exclude_tasks.split(",") if t}
        runs = [dict(r, phase=r.get("phase") or "base") for r in table(a.records, "runs") if int(r["task_id"]) not in skip]
        keep = {r["run_id"] for r in runs}
        points = [p for p in table(a.records, "points") if p["run_id"] in keep]
        steps = [s for s in table(a.records, "steps") if s["run_id"] in keep]
        rec, arrays = golden.select(runs, points, steps)
        rec["excluded_tasks"] = sorted(skip); rec["source"] = "Spark base reproduction (ops/spark/repro.py)"
        Path(a.out).mkdir(parents=True, exist_ok=True)
        golden.save(rec, arrays, Path(a.out) / "golden.json")
        print(json.dumps([{k: g[k] for k in ("snapshot_id", "task_id", "step", "expected_steps", "outcome")}
                          for g in rec["golden"]]))
        return
    from vlar_harness.config import load_config
    from vlar_harness.servers.clients import PI05Client, XVLAClient
    from vlar_harness.snapshot import from_bytes
    from vlar_ops.vm_worker import ProductionEpisodes, check_servers
    cfg = load_config()
    srv = cfg["servers"]
    clients = {"xvla": XVLAClient(srv["xvla"]["host"], int(srv["xvla"]["port"])),
               "pi05": PI05Client(srv["pi05"]["host"], int(srv["pi05"]["port"]))}
    check_servers(cfg, clients)
    record, arrays = golden.load(Path(a.golden) / "golden.json")

    def read(sid):
        f = glob.glob(f"{a.records}/base/*/snapshots/{sid}.npz")
        return from_bytes(open(f[0], "rb").read())
    episodes = ProductionEpisodes(cfg, clients["xvla"], clients["pi05"])
    t0 = time.monotonic()
    try:
        rep = golden.check(record, arrays, episodes, read)
    finally:
        episodes.close()
    line = {"repeat": a.repeat, "passed": rep["passed"], "results": rep["results"],
            "seconds": round(time.monotonic() - t0, 1), "pid": os.getpid(), "container": container_id(),
            "process_started_unix": round(START, 3)}
    with open(Path(a.golden) / a.results_file, "a") as f:
        f.write(json.dumps(line) + "\n")
    print(json.dumps(line))


if __name__ == "__main__":
    main()
