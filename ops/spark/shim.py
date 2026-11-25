#!/usr/bin/env python3
"""Stand-ins for the VM's commands, used only by ops/spark/run.sh (on PATH as ops/spark/shims/<name> symlinks).

- gcloud storage {cp,rsync,ls,cat,rm}: gs://<any bucket>/<key> is the local folder $VLAR_SPARK_ROOT/var/bucket/<key>
  (the ops config's storage_url is /var/lib/vlar/bucket/vla-router inside the containers: the same folder).
- systemctl {start,stop,restart,is-active,list-units} for vlar-xvla, vlar-pi05 and vlar-worker@N: each unit's
  ExecStart from ops/vm/<unit>.service, its EnvironmentFiles and %i expanded, host paths rewritten as transform.py
  does, run as a detached container of the same name.
- docker run: adds the Spark root at the same path, and the wheelhouse as the only pip index (no network).
- sudo: runs the command; shutdown: logged, nothing else.
"""
import glob
import os
import re
import shlex
import shutil
import subprocess
import sys

SD, REPO = os.environ["VLAR_SPARK_ROOT"], os.environ["VLAR_SPARK_REPO"]
BUCKET = os.path.join(SD, "var", "bucket")
HERE = os.path.dirname(os.path.realpath(__file__))
DOCKER = shutil.which("docker", path="/usr/bin:/usr/local/bin:/bin")


def local(p: str) -> str:
    m = re.match(r"^gs://[^/]+/?(.*)$", p)
    return os.path.join(BUCKET, m.group(1)) if m else p


def gcloud(args):
    if args[:1] != ["storage"]:
        sys.exit(f"gcloud shim: only `gcloud storage` is available on the Spark, not {args}")
    cmd, rest = args[1], [a for a in args[2:] if not a.startswith("--") and a not in ("-r", "-R")]
    flags = [a for a in args[2:] if a.startswith("-")]
    paths = [local(a) for a in rest]
    if cmd == "cp":
        *srcs, dst = paths
        if srcs == ["-"]:
            if "--no-clobber" in flags and os.path.exists(dst):
                return 0
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(sys.stdin.buffer.read())
            return 0
        for s in srcs:
            for f in glob.glob(s) or [s]:
                if not os.path.exists(f):
                    print(f"gcloud shim: {f} not found", file=sys.stderr); return 1
                d = os.path.join(dst, os.path.basename(f)) if dst.endswith("/") or os.path.isdir(dst) else dst
                if "--no-clobber" in flags and os.path.exists(d):
                    continue
                os.makedirs(os.path.dirname(d) or ".", exist_ok=True)
                (shutil.copytree(f, d, dirs_exist_ok=True) if os.path.isdir(f) else shutil.copy2(f, d))
        return 0
    if cmd == "rsync":
        src, dst = paths
        if not os.path.isdir(src):
            print(f"gcloud shim: {src} not found", file=sys.stderr); return 1
        os.makedirs(dst, exist_ok=True)
        return subprocess.call(["cp", "-a", src + "/.", dst])
    if cmd == "ls":
        hits = []
        for p in paths:
            hits += sorted(glob.glob(p.rstrip("/") + ("/*" if os.path.isdir(p) else "")))
        for h in hits:
            print("gs://local/" + os.path.relpath(h, BUCKET) + ("/" if os.path.isdir(h) else ""))
        return 0 if hits else 1
    if cmd == "cat":
        for p in paths:
            if not os.path.isfile(p):
                return 1
            sys.stdout.buffer.write(open(p, "rb").read())
        return 0
    if cmd == "rm":
        for p in paths:
            for f in glob.glob(p):
                shutil.rmtree(f) if os.path.isdir(f) else os.remove(f)
        return 0
    sys.exit(f"gcloud shim: storage {cmd} is not implemented")


def env_files(paths):
    env = {}
    for p in paths:
        p = p.lstrip("-")
        p = re.sub(r"^/etc/vlar(?!-)", f"{SD}/etc", p)
        if os.path.exists(p):
            for ln in open(p):
                if "=" in ln and not ln.lstrip().startswith("#"):
                    k, v = ln.rstrip("\n").split("=", 1)
                    env[k] = v
    return env


def unit_command(unit: str):
    name, _, inst = unit.partition("@")
    f = os.path.join(REPO, "ops", "vm", f"{name}@.service" if inst else f"{name}.service")
    text = open(f).read().replace("\\\n", " ")
    files = re.findall(r"^EnvironmentFile=(\S+)", text, re.M)
    exec_start = re.search(r"^ExecStart=(.*)$", text, re.M).group(1).replace("%i", inst)
    env = {**os.environ, **env_files(files)}
    exec_start = re.sub(r"\$\{?([A-Z_]+)\}?", lambda m: env.get(m.group(1), ""), exec_start)
    sys.path.insert(0, HERE)
    os.environ.setdefault("VLAR_SPARK_RUN", "")
    import transform  # noqa: E402  (host-side path rewrite)
    argv = shlex.split(transform.line(exec_start))
    argv[0] = os.path.join(HERE, "shims", "docker")
    argv.insert(argv.index("run") + 1, "-d")
    return argv, ("vlar-worker-" + inst) if inst else name


def running(name):
    out = subprocess.run([DOCKER, "ps", "-q", "--filter", f"name=^{name}$"], capture_output=True, text=True).stdout
    return bool(out.strip())


def systemctl(args):
    opts = [a for a in args if a.startswith("-")]
    cmd, *units = [a for a in args if not a.startswith("-")]
    if cmd == "list-units":
        pat = (units or ["vlar-*"])[0].replace("@*", "-").replace("*", "")
        names = subprocess.run([DOCKER, "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout.split()
        for n in names:
            if n.startswith(pat):
                print(n.replace("vlar-worker-", "vlar-worker@") + ".service loaded active running")
        return 0
    rc = 0
    for u in units:
        u = u.removesuffix(".service")
        if "*" in u:
            pat = u.replace("@*", "-")
            names = subprocess.run([DOCKER, "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout.split()
            targets = [n for n in names if n.startswith(pat)]
        else:
            targets = [u.replace("@", "-") if "@" in u else u]
        if cmd in ("stop", "restart"):
            for t in targets:
                subprocess.call([DOCKER, "stop", "-t", "60", t], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.call([DOCKER, "rm", "-f", t], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if cmd in ("start", "restart"):
            if "*" in u:
                continue
            argv, name = unit_command(u)
            if not running(name):
                subprocess.call([DOCKER, "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                rc |= subprocess.call(argv, stdout=subprocess.DEVNULL)
        if cmd == "is-active":
            ok = all(running(t) for t in targets)
            print("active" if ok else "inactive")
            rc |= 0 if ok else 3
    return rc


def docker(args):
    if args[:1] == ["run"]:
        wh = os.environ.get("VLAR_SPARK_WHEELHOUSE", "")
        extra = ["-v", f"{SD}:{SD}"]
        if wh:
            extra += ["-v", f"{wh}:/wh:ro", "-e", "PIP_NO_INDEX=1", "-e", "PIP_FIND_LINKS=/wh"]
        args = ["run", *extra, *args[1:]]
    os.execv(DOCKER, [DOCKER, *args])


def main():
    name, args = os.path.basename(sys.argv[0]), sys.argv[1:]
    if name == "gcloud":
        sys.exit(gcloud(args))
    if name == "systemctl":
        sys.exit(systemctl(args))
    if name == "docker":
        docker(args)
    if name == "sudo":
        os.execvp(args[0], args)
    if name == "shutdown":
        print(f"shutdown shim: not shutting the Spark down ({args})", file=sys.stderr)
        sys.exit(0)
    sys.exit(f"unknown shim {name}")


if __name__ == "__main__":
    main()
