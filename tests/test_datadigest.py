"""Data digests (vlar_ops.datadigest, amendment 49): what they cover, what they ignore, and the pin to 0a0cd4c.

The base rollouts and parity run are made with 0a0cd4c's ops code, so every later commit's common data digest must
equal 0a0cd4c's (computed here from `git archive 0a0cd4c src/vlar_ops`). Job-spec generator digests are compared only
among records of one kind and may differ from 0a0cd4c's (exploration specs now record their tunables).
"""

import os
import shutil
import subprocess

import pytest
import yaml

from vlar_harness import provenance as P
from vlar_ops import datadigest as DD

REPO = P.PACKAGE_DIR.parents[1]
PINNED_COMMIT = "0a0cd4c"
PINNED_COMMON = "4951f7f0fc6276ee1ebc8f3636b1fadd3351a9da6f21c9e1c26cd27a79635678"
TUNABLES = "configs/collection-v1-tunables.yaml"
DEFS = "configs/collection-v1.yaml"


@pytest.fixture(scope="module")
def head():
    return DD.files_from_dir(P.ops_dir())


def edit(files, name, old, new):
    text = files[name].decode()
    assert text.count(old) == 1, (name, old)
    return {**files, name: text.replace(old, new).encode()}


def put(files, name, text):
    return {**files, name: text.encode()}


def yaml_edit(files, name, fn):
    data = yaml.safe_load(files[name])
    fn(data)
    return {**files, name: yaml.safe_dump(data).encode()}


def changed(a, b):
    da, db = DD.digests(a), DD.digests(b)
    return {k for k in da if da[k] != db[k]}


def test_the_ops_digest_of_files_is_provenances(head, tmp_path):
    assert DD.ops_digest_of(head) == P.ops_digest()
    pkg = tmp_path / "vlar_ops"
    shutil.copytree(P.ops_dir(), pkg)
    (pkg / "configs" / "a.json").write_text("{}")
    (pkg / "configs.yaml").write_text("x: 1\n")      # sorts between configs/ entries by parts, not by string
    (pkg / "notes.md").write_text("ignored")
    assert DD.ops_digest_of(DD.files_from_dir(pkg)) == P.source_digest(pkg)


@pytest.mark.skipif(os.environ.get("VLAR_IN_IMAGE") == "1" or not (REPO / ".git").exists(), reason="repo only")
def test_the_common_digest_equals_0a0cd4cs():
    raw = subprocess.run(["git", "archive", "--format=tar", PINNED_COMMIT, "src/vlar_ops"], cwd=REPO,
                         capture_output=True, check=True).stdout
    old = DD.files_from_archive(raw)
    assert DD.digests(old)["common"] == PINNED_COMMON
    assert DD.digests(old)["branch"] is None                        # no branch planner module then
    assert DD.digests(DD.files_from_dir(P.ops_dir()))["common"] == PINNED_COMMON


def test_comments_docstrings_and_formatting_change_nothing(head):
    f = edit(head, "plan.py", '''    """Rollout seed of an episode (SPEC §3). The scene seed comes from the initial state, in the harness."""
    return namespace + 100 * task_id + init_idx''', '''    """Another docstring."""
    # a comment
    return (namespace
            + 100*task_id + init_idx)   # reformatted''')
    f = edit(f, "vm_worker.py", "class ProductionEpisodes:\n", "class ProductionEpisodes:  # comment\n")
    assert changed(head, f) == set()


def test_tunables_change_nothing_and_the_layout_does_not_matter(head):
    def bump(d):
        d["exploration"]["p_escalate"] = 0.3
        d["branches"]["seeds"] = 2
        d["branches"]["enabled"] = ["b1_continue", "b2_one_call", "b4_max"]
        d["branches"]["tier_one_seeds"] = 1
        d["dp_sampling"] = {"fraction": 0.2, "seed": 1}
        d["order_seed"] = 3
    assert changed(head, yaml_edit(head, TUNABLES, bump)) == set()
    one_file = yaml.safe_load(head[DEFS]) | yaml.safe_load(head[TUNABLES])         # the old one-file layout
    f = {k: v for k, v in put(head, DEFS, yaml.safe_dump(one_file)).items() if k != TUNABLES}
    assert changed(head, f) == set()


@pytest.mark.parametrize("what", ["namespace", "schema", "new key in definitions", "new key in tunables",
                                  "new key in a tunable section"])
def test_a_definition_or_an_unknown_key_changes_the_common_digest(head, what):
    fn = {"namespace": lambda d: d["rollout_seed_namespaces"].__setitem__("explore", 2_100_000),
          "schema": lambda d: d.__setitem__("schema", 2),
          "new key in definitions": lambda d: d.__setitem__("episode_length", 520),
          "new key in tunables": lambda d: d.__setitem__("reward", "sparse"),
          "new key in a tunable section": lambda d: d["exploration"].__setitem__("policy", "greedy")}[what]
    name = DEFS if "tunables" not in what and "tunable section" not in what else TUNABLES
    assert "common" in changed(head, yaml_edit(head, name, fn))


def test_the_episode_wiring_and_seed_derivation_change_the_common_digest(head):
    cls = edit(head, "vm_worker.py", 'return self.pi05.act(raw_obs, task, seed)["actions"]',
               'return self.pi05.act(raw_obs, task, seed if seed is None else seed + 1)["actions"]')
    const = edit(head, "vm_worker.py", 'XVLA_CLIENT_FILE = "evaluation/libero/libero_client.py"',
                 'XVLA_CLIENT_FILE = "evaluation/libero/other_client.py"')
    seed = edit(head, "plan.py", "return namespace + 100 * task_id + init_idx", "return namespace + 1000 * task_id + init_idx")
    for f in (cls, const, seed):
        assert "common" in changed(head, f)


def test_imported_and_transitively_imported_helpers_are_covered(head):
    f = edit(head, "vm_worker.py", "        self.sims: dict[int, Any] = {}\n",
             "        self.sims: dict[int, Any] = {}\n        self.tag = wire()\n")
    f = edit(f, "vm_worker.py", "from .queue import JobQueue\n", "from .helpers import wire\nfrom .queue import JobQueue\n")
    f = put(f, "helpers.py", "def wire():\n    from .inner import deep\n    return deep()\n\n\ndef other():\n    return 1\n")
    f = put(f, "inner.py", "LIMIT = 3\n\n\ndef deep():\n    return LIMIT\n\n\ndef unrelated():\n    return 2\n")
    assert "common" in changed(head, f)
    assert "common" in changed(f, edit(f, "helpers.py", "    return deep()", "    return deep() + 1"))
    assert "common" in changed(f, edit(f, "inner.py", "LIMIT = 3", "LIMIT = 4"))          # transitively
    assert changed(f, edit(f, "helpers.py", "    return 1", "    return 5")) == set()
    assert changed(f, edit(f, "inner.py", "    return 2", "    return 6")) == set()


def test_module_objects_and_whole_module_use(head):
    f = edit(head, "vm_worker.py", "        self.sims: dict[int, Any] = {}\n",
             "        self.sims: dict[int, Any] = {}\n        from . import inner\n        self.k = inner.deep()\n")
    f = put(f, "inner.py", "def deep():\n    return 1\n\n\ndef unrelated():\n    return 2\n")
    assert changed(f, edit(f, "inner.py", "    return 2", "    return 3")) == set()
    assert "common" in changed(f, edit(f, "inner.py", "    return 1", "    return 0"))
    g = edit(f, "vm_worker.py", "        self.k = inner.deep()\n", "        self.k = vars(inner)\n")   # bare: all of it
    assert "common" in changed(g, edit(g, "inner.py", "    return 2", "    return 3"))


def test_serve_spend_waiver_and_consumption_code_change_nothing(head):
    for name, old, new in [
            ("vm_worker.py", "    infos = check_servers(cfg, clients, wait_s=server_wait_s)\n",
             "    infos = check_servers(cfg, clients, wait_s=server_wait_s * 2)\n"),
            ("spend.py", "from __future__ import annotations\n", "from __future__ import annotations\nEXTRA = 1\n"),
            ("waiver.py", 'OPS_ONLY = "ops code (vlar_ops)"', 'OPS_ONLY = "ops code (vlar_ops)"\nX = 2'),
            ("plan.py", "        seeds = int(settings[\"seeds\"])\n", "        seeds = int(settings[\"seeds\"]) or 1\n"),
            ("plan.py", "def complete_points(", "def a_new_helper():\n    return 0\n\n\ndef complete_points(")]:
        assert changed(head, edit(head, name, old, new)) == set(), name


def test_each_kind_has_its_own_generator_digest(head):
    ep = edit(head, "plan.py", '"mode": mode, "profile": profile', '"mode": mode, "profile": profile, "v": 2')
    assert changed(head, ep) == {"base", "explore"}
    cli = edit(head, "cli.py", 'fields = explore_fields() if a.phase == "explore" else None',
               'fields = explore_fields() if a.phase in ("explore",) else None')
    assert changed(head, cli) == {"base", "explore"}
    tun = edit(head, "tunables.py", '"episodes_per_state": n}', '"episodes_per_state": n, "x": 1}')
    assert changed(head, tun) == {"base", "explore"}
    br = edit(head, "branchplan.py", '"source": p.source, "settings": settings}',
              '"source": p.source, "settings": settings, "v": 2}')
    assert changed(head, br) == {"branch"}
    new_cmd = put(head, "branchplan.py", head["branchplan.py"].decode() + "\n\ndef command(args):\n    return args\n")
    assert changed(head, new_cmd) == {"branch"}                     # the future command, in its own module
    assert DD.digests(head)["base"] != DD.digests(head)["explore"]  # kinds are never compared with each other


def test_archives_are_read_without_extracting(head, tmp_path):
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for k, v in head.items():
            info = tarfile.TarInfo(f"src/vlar_ops/{k}")
            info.size = len(v)
            tar.addfile(info, io.BytesIO(v))
        info = tarfile.TarInfo("README.md")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"hi"))
    files = DD.files_from_archive(buf.getvalue())
    assert files == head and DD.ops_digest_of(files) == P.ops_digest()
    with pytest.raises(DD.DigestError):
        DD.files_from_archive(_tar_gz({"README.md": b"x"}))


def _tar_gz(members):
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for k, v in members.items():
            info = tarfile.TarInfo(k)
            info.size = len(v)
            tar.addfile(info, io.BytesIO(v))
    return buf.getvalue()


def test_record_kinds():
    assert DD.kind_of({"kind": "episode", "phase": "base"}) == "base"
    assert DD.kind_of({"kind": "episode", "phase": "pi05_parity"}) == "base"
    assert DD.kind_of({"kind": "episode", "phase": "explore"}) == "explore"
    assert DD.kind_of({"kind": "b2_one_call", "phase": "branches"}) == "branch"
    assert DD.kind_of({"kind": "episode", "phase": "dagger"}) is None
