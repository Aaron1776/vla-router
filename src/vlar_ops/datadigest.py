"""Data digests: the ops code that decides what a record holds (docs/SPEC.md §15, amendment 49, refining 46).

After the freeze the ops digest (`provenance.ops_digest`, every file of `vlar_ops`) is provenance, not a mixing
barrier: records made with different ops code may be combined when their data digests agree. The data digest covers
only what changes what a record means:

- `common`, in every record: `vm_worker.ProductionEpisodes` (the Episode wiring) and the seed derivation
  (`plan.rollout_seed`), with every `vlar_ops` definition they reach through names and imports, transitively, plus
  the collection config's definitions (below). Compared with the freeze's.
- one job-spec generator digest per record kind (`KINDS`): the code that wrote the kind's job specs, with the same
  closure. Compared only among records of the same kind, so a generator change before a kind's first record (e.g.
  recording a tunable in its specs) does not touch the others. The branch planner is all of `branchplan.py`, so
  the future branch-plan command must live there.
Consumption code (`plan.complete_points`, labels, training) is in no data digest: it is versioned like other code.

The closure works on top-level definitions, not whole modules: a covered definition pulls in the definitions it
names in its own module and those it imports from other `vlar_ops` modules (at module level or inside a function),
transitively; a module object used other than as `module.name` pulls in the whole module; every module reached also
contributes its top-level statements that run on import and bind nothing. Code is normalized as `ast.unparse` of
the tree with docstrings removed, so comments, docstrings and formatting change nothing. Imports outside `vlar_ops`
are not followed: the harness has its own digest and third-party code is pinned by the images.

Definitions: the collection config (`configs/collection-v1.yaml` merged with `configs/collection-v1-tunables.yaml`)
minus `TUNABLE_PATHS`, so the one-file layout and the split give the same digest and an unknown new key is a
definition. Tunables choose which work is done; each is recorded in every job spec or branch-plan entry it affects
and read from there, never from the config.

Every digest is computed by this module's rules from a set of source files, so `0a0cd4c`'s archive and a later one
are compared under the same rules.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

FORMAT = "datadigest-v1"
PKG = "vlar_ops"
ARCHIVE_ROOT = "src/vlar_ops/"

COMMON_ROOTS = ("vm_worker:ProductionEpisodes", "plan:rollout_seed")
KINDS = {
    "base": ("plan:episode_plan", "cli:cmd_plan"),
    "explore": ("plan:episode_plan", "cli:cmd_plan"),
    "branch": ("branchplan:*",),
}
DEFINITION_FILES = ("configs/collection-v1.yaml", "configs/collection-v1-tunables.yaml")
TUNABLE_PATHS = (
    ("exploration", "episodes_per_state"), ("exploration", "p_escalate"),
    ("branches", "seeds"), ("branches", "enabled"), ("branches", "tier_one_seeds"),
    ("dp_sampling", "fraction"), ("dp_sampling", "seed"),
    ("order_seed",),
)
DIGEST_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".txt", ".lock", ".toml")   # provenance._digest_files


class DigestError(ValueError):
    pass


# ---------------------------------------------------------------- source files

def files_from_dir(root: Path | str) -> dict[str, bytes]:
    """The package's files, keyed by path relative to the package (`plan.py`, `configs/...`)."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def files_from_archive(data: bytes) -> dict[str, bytes]:
    """`vlar_ops` from a `git archive` of the repo (phase0/src/<commit>.tar.gz), read without extracting."""
    out = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        for m in tar.getmembers():
            name = m.name[2:] if m.name.startswith("./") else m.name
            if m.isfile() and name.startswith(ARCHIVE_ROOT):
                f = tar.extractfile(m)
                if f is not None:
                    out[name[len(ARCHIVE_ROOT):]] = f.read()
    if not out:
        raise DigestError(f"no {ARCHIVE_ROOT} in the archive")
    return out


def ops_digest_of(files: Mapping[str, bytes]) -> str:
    """`provenance.source_digest` over these files (the same filter and order), i.e. the ops digest."""
    keep = [k for k in files if Path(k).suffix in DIGEST_SUFFIXES and "__pycache__" not in k.split("/")
            and not k.endswith(".frozen.json")]
    h = hashlib.sha256()
    for k in sorted(keep, key=lambda k: tuple(k.split("/"))):
        h.update(k.encode() + b"\0")
        h.update(hashlib.sha256(files[k]).digest())
    return h.hexdigest()


# ---------------------------------------------------------------- definitions

def _merge(a: dict, b: dict, where: str = "") -> dict:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v, f"{where}{k}.")
        elif k in out:
            raise DigestError(f"collection config key {where}{k} is set in both files")
        else:
            out[k] = v
    return out


def definitions(files: Mapping[str, bytes]) -> dict:
    """The merged collection config minus the tunable paths, empty sections dropped."""
    merged: dict = {}
    for name in DEFINITION_FILES:
        if name in files:
            merged = _merge(merged, yaml.safe_load(files[name]) or {})
    if not merged:
        raise DigestError("no collection config")
    out = copy.deepcopy(merged)
    for path in TUNABLE_PATHS:
        cur = out
        for part in path[:-1]:
            cur = cur.get(part) if isinstance(cur, dict) else None
        if isinstance(cur, dict):
            cur.pop(path[-1], None)
    for path in TUNABLE_PATHS:
        if len(path) > 1 and isinstance(out.get(path[0]), dict) and not out[path[0]]:
            del out[path[0]]
    return out


# ---------------------------------------------------------------- code closure

def _strip_docstrings(node: ast.AST) -> ast.AST:
    node = copy.deepcopy(node)
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
    return node


def _bound_names(stmt: ast.stmt) -> set[str]:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return set()
    out = set()
    for n in ast.walk(stmt):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
    return out


def _is_main_guard(stmt: ast.stmt) -> bool:
    t = stmt.test if isinstance(stmt, ast.If) else None
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
            and len(t.comparators) == 1 and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value == "__main__")


class _Module:
    def __init__(self, name: str, source: bytes, modules: set[str]):
        self.name = name
        tree = ast.parse(source)
        self.defs: dict[str, list[ast.stmt]] = {}
        self.effects: list[ast.stmt] = []
        self.imports: dict[str, tuple[str, Optional[str]]] = {}
        for i, stmt in enumerate(tree.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                self.imports.update(_resolve_import(stmt, modules))
                continue
            if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue                                          # module docstring
            if _is_main_guard(stmt):
                continue                                          # does not run on import
            names = _bound_names(stmt)
            for n in names:
                self.defs.setdefault(n, []).append(stmt)
            if not names:
                self.effects.append(stmt)


def _target(module: Optional[str], level: int) -> Optional[str]:
    """The `vlar_ops` submodule an import names ("" for the package itself), or None if it is not ours."""
    if level == 1:
        return module or ""
    if level == 0 and module is not None and (module == PKG or module.startswith(PKG + ".")):
        return module[len(PKG) + 1:] if module != PKG else ""
    return None


def _resolve_import(stmt: ast.stmt, modules: set[str]) -> dict[str, tuple[str, Optional[str]]]:
    """Local name -> (module, attribute); attribute None for a module object, module "" for the package."""
    out = {}
    if isinstance(stmt, ast.ImportFrom):
        mod = _target(stmt.module, stmt.level)
        if mod is None:
            return out
        for a in stmt.names:
            local = a.asname or a.name
            if mod == "" and a.name in modules:
                out[local] = (a.name, None)
            else:
                out[local] = (mod or "__init__", a.name)
    elif isinstance(stmt, ast.Import):
        for a in stmt.names:
            mod = _target(a.name, 0)
            if mod is None:
                continue
            if a.asname:
                out[a.asname] = (mod, None) if mod else ("", None)
            else:
                out[PKG] = ("", None)
    return out


def _chain(node: ast.AST) -> Optional[list[str]]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id] + parts[::-1]
    return None


def _refs(node: ast.AST, mod: _Module, modules: set[str]) -> set[tuple[str, str]]:
    """(module, name) pairs this node uses: same-module definitions and `vlar_ops` imports; "*" = a whole module."""
    imports = dict(mod.imports)
    for n in ast.walk(node):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            imports.update(_resolve_import(n, modules))
    out: set[tuple[str, str]] = set()
    as_base: set[int] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and id(n) not in as_base:
            chain = _chain(n)
            if chain is None:
                continue
            cur = n
            while isinstance(cur, ast.Attribute):
                as_base.add(id(cur))
                cur = cur.value
            as_base.add(id(cur))
            target = imports.get(chain[0])
            if target is None or target[1] is not None:
                if target is not None:
                    out.add(target)
                elif chain[0] in mod.defs:
                    out.add((mod.name, chain[0]))
                continue
            m, rest = target[0], chain[1:]
            if m == "" and rest and rest[0] in modules:              # vlar_ops.<submodule>.<name>
                m, rest = rest[0], rest[1:]
            out.add((m or "__init__", rest[0] if rest else "*"))
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and id(n) not in as_base:
            target = imports.get(n.id)
            if target is not None:
                out.add((target[0] or "__init__", target[1] or "*"))
            elif n.id in mod.defs:
                out.add((mod.name, n.id))
        elif isinstance(n, ast.ImportFrom):                     # names imported inside a function: always
            for target in _resolve_import(n, modules).values():
                if target[1] is not None:                         # a module object counts by its use
                    out.add((target[0] or "__init__", target[1]))
    return out


def _modules(files: Mapping[str, bytes]) -> dict[str, bytes]:
    out = {}
    for k, v in files.items():
        if k.endswith(".py") and "/" not in k:
            out[k[:-3]] = v
    return out


def closure(files: Mapping[str, bytes], roots: Iterable[str]) -> Optional[dict[str, str]]:
    """Normalized code of every definition the roots reach, keyed `module:name` (`module:<import>` for statements
    that run on import). None if a root's module or definition does not exist in these files."""
    sources = _modules(files)
    names = set(sources)
    parsed: dict[str, _Module] = {}

    def module(m: str) -> Optional[_Module]:
        if m not in parsed and m in sources:
            parsed[m] = _Module(m, sources[m], names)
        return parsed.get(m)

    out: dict[str, str] = {}
    todo: list[tuple[str, str]] = []
    for r in roots:
        m, _, n = r.partition(":")
        mod = module(m)
        if mod is None or (n != "*" and n not in mod.defs):
            return None
        todo.append((m, n))
    seen: set[tuple[str, str]] = set()
    while todo:
        m, n = todo.pop()
        if (m, n) in seen:
            continue
        seen.add((m, n))
        mod = module(m)
        if mod is None:
            out[f"{m}:{n}"] = "<missing module>"
            continue
        if f"{m}:<import>" not in out:
            out[f"{m}:<import>"] = "\n".join(ast.unparse(_strip_docstrings(s)) for s in mod.effects)
            for s in mod.effects:
                todo.extend(_refs(s, mod, names))
        targets = sorted(mod.defs) if n == "*" else [n]
        for t in targets:
            stmts = mod.defs.get(t)
            if stmts is None:
                out[f"{m}:{t}"] = "<missing>"
                continue
            out[f"{m}:{t}"] = "\n".join(ast.unparse(_strip_docstrings(s)) for s in stmts)
            for s in stmts:
                todo.extend(_refs(s, mod, names))
    return out


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode() + b"\0")
    return h.hexdigest()


def _code_hash(label: str, code: Mapping[str, str]) -> str:
    return _hash(FORMAT, label, *(x for k in sorted(code) for x in (k, code[k])))


def digests(files: Mapping[str, bytes]) -> dict[str, Optional[str]]:
    """{"common": ..., one entry per kind}: a kind is None if its generator does not exist in these files."""
    code = closure(files, COMMON_ROOTS)
    if code is None:
        raise DigestError(f"the covered code is missing: {COMMON_ROOTS}")
    defs = json.dumps(definitions(files), sort_keys=True, separators=(",", ":"))
    out: dict[str, Optional[str]] = {"common": _hash(_code_hash("common", code), "definitions", defs)}
    for kind, roots in KINDS.items():
        gen = closure(files, roots)
        out[kind] = None if gen is None else _code_hash(kind, gen)
    return out


def kind_of(job: Mapping[str, Any]) -> Optional[str]:
    """The record kind of a marker's job: base (base and π0.5 parity episodes), explore, or branch."""
    from .branchplan import BRANCH_ORDER
    if job.get("kind") in BRANCH_ORDER:
        return "branch"
    if job.get("kind") == "episode":
        return {"base": "base", "pi05_parity": "base", "explore": "explore"}.get(job.get("phase"))
    return None
