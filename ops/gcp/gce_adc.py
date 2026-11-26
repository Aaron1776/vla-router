"""Compute Engine reads and start for the restart watcher (ops/gcp/34_restart_watch.sh), through application-default
credentials: the gcloud CLI's sign-in expired on 2026-11-25 ~22:04Z (16-hour Workspace session) and left the watcher
blind.
  gce_adc.py status | laststart | preempted-since-start | start      (zone us-central1-a unless ZONE is set)
Exit 0 with the answer on stdout; `preempted-since-start` exits 0 when a compute.instances.preempted operation for
the VM is newer than its latest start (`preempted_since_start`), else 1; `start` prints STOCKOUT and exits 3 on a
capacity error, 2 on others."""
import os
import sys
from datetime import datetime


def preempted_since_start(last_start: str, start_ops: list, preempted_ops: list) -> bool:
    """Whether a preemption came after the VM's latest start. The start is the newest successful `start` operation's
    insertTime (one that ended with an error, e.g. a stockout, started nothing), else `lastStartTimestamp`. A VM can
    be preempted while its start completes: on 2026-11-26 the start was inserted 07:48:18Z, the preemption 07:48:34.089Z
    and lastStartTimestamp set to 07:48:34.695Z, so comparing with lastStartTimestamp read a preemption as a plain stop
    and the watcher ended."""
    ok = [datetime.fromisoformat(o["insertTime"]) for o in start_ops if not o.get("error")]
    since = max(ok) if ok else datetime.fromisoformat(last_start)
    return any(datetime.fromisoformat(o["insertTime"]) > since for o in preempted_ops)


def main() -> None:
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    P, V = os.environ["PROJECT_ID"], os.environ["VM_NAME"]
    Z = os.environ.get("ZONE_OVERRIDE", "us-central1-a")
    BASE = f"https://compute.googleapis.com/compute/v1/projects/{P}/zones/{Z}"
    cred, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    s = AuthorizedSession(cred)

    def instance():
        r = s.get(f"{BASE}/instances/{V}", timeout=60)
        r.raise_for_status()
        return r.json()

    def ops(kind: str) -> list:
        link = f"https://www.googleapis.com/compute/v1/projects/{P}/zones/{Z}/instances/{V}"
        r = s.get(f"{BASE}/operations", timeout=60,
                  params={"filter": f'(operationType = "{kind}") AND (targetLink = "{link}")'})
        r.raise_for_status()
        return r.json().get("items", [])

    cmd = sys.argv[1]
    if cmd == "status":
        print(instance()["status"])
    elif cmd == "laststart":
        print(instance().get("lastStartTimestamp", ""))
    elif cmd == "preempted-since-start":
        sys.exit(0 if preempted_since_start(instance()["lastStartTimestamp"], ops("start"),
                                            ops("compute.instances.preempted")) else 1)
    elif cmd == "start":
        r = s.post(f"{BASE}/instances/{V}/start", timeout=60)
        text = r.text
        if r.ok:
            import time
            op = r.json()
            for _ in range(60):                      # wait for the start operation to finish (errors show there)
                o = s.get(f"{BASE}/operations/{op['name']}", timeout=60).json()
                if o.get("status") == "DONE":
                    text = str(o.get("error") or "")
                    break
                time.sleep(5)
            if not text:
                print(instance()["status"])
                sys.exit(0)
        if any(k in text for k in ("ZONE_RESOURCE_POOL_EXHAUSTED", "does not have enough resources", "stockout")):
            print("STOCKOUT")
            sys.exit(3)
        print(text[-300:])
        sys.exit(2)
    elif cmd == "state":                                 # the runner's STATE.json: "<state> <step> <reason>"
        import json
        from google.cloud import storage
        d = json.loads(storage.Client(project=P).bucket(os.environ["BUCKET"]).blob("vla-router/phase0/STATE.json")
                       .download_as_text())
        print(d.get("state"), d.get("step"), str(d.get("reason", ""))[:160])
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
