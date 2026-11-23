"""No account identifiers in the repo (amendment 44; CLAUDE.md: no hostnames, usernames or network addresses of the
team's machines): the VM name, project ID, bucket, billing account ID and service-account email live only in the
git-ignored `ops/gcp/env.sh`; tracked files write `$VM_NAME`, `$PROJECT_ID`, `$BUCKET`, `$BILLING_ACCOUNT`,
`$SA_NAME`. This file holds only salted SHA-256 digests of the values, so it does not republish them; with `env.sh`
present (the Mac that commits) the values themselves are checked too."""

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SALT = "vla-router identifier v1:"
DIGESTS = {
    "2beca3f3ca8609b92e7bfd95905f31dc2104ae3828840c037e81b9815442f5c1": "project ID",
    "a6e64381e370b78cb6e1325ffd83853db6905b736ff45db75f2a34559fc0c349": "bucket",
    "00aa4395d358a17f0078da72b248b7de49e9924d546cd63da3931e53a1ed03ff": "billing account ID",
    "d2efb027b15cd57a25406c8b104510f1357da5831bc44bede5592382f5ca3e10": "VM name",
    "2d70fccf4af9e067e5d60d269b784d4471f6151a8910cd2f0762829cf2961d1f": "service-account email",
}
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]*")
SEP = re.compile(r"[-._@]")


def committable_files() -> list[Path]:
    """Tracked files plus untracked ones that are not ignored (what the next commit could add)."""
    if not shutil.which("git") or not (REPO / ".git").exists():
        pytest.skip("needs the git checkout")
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=REPO,
                         capture_output=True, check=True).stdout.decode()
    return [REPO / f for f in out.split("\0") if f and (REPO / f).is_file()]


def texts():
    for p in committable_files():
        if p.stat().st_size > 4_000_000:
            continue
        try:
            yield p, p.read_text()
        except UnicodeDecodeError:
            continue


def digest(s: str) -> str:
    return hashlib.sha256((SALT + s.lower()).encode()).hexdigest()


def candidates(token: str, max_parts: int = 12):
    """Every run of whole parts of a token (parts separated by - . _ @), e.g. `a-b-c` -> a, a-b, a-b-c, b, b-c, c."""
    starts = [0] + [m.end() for m in SEP.finditer(token)]
    ends = [m.start() for m in SEP.finditer(token)] + [len(token)]
    for i, s in enumerate(starts):
        for e in ends[i:i + max_parts]:
            if e > s:
                yield token[s:e]


def test_digests_find_the_identifiers_inside_longer_strings():
    assert digest("Example-VM") == digest("example-vm")
    found = {c for c in candidates("gs-x@my-proj-1.iam.example.com") if digest(c) == digest("my-proj-1")}
    assert found == {"my-proj-1"}


def test_no_account_identifier_in_a_committable_file():
    hits = []
    for path, text in texts():
        for m in TOKEN.finditer(text):
            for c in candidates(m.group(0)):
                what = DIGESTS.get(digest(c))
                if what:
                    line = text.count("\n", 0, m.start()) + 1
                    hits.append(f"{path.relative_to(REPO)}:{line}: the {what}")
    assert not hits, "account identifiers in the repo (use the env.sh placeholders):\n" + "\n".join(sorted(set(hits)))


def test_no_value_from_env_sh_in_a_committable_file():
    env = REPO / "ops" / "gcp" / "env.sh"
    if not env.exists():
        pytest.skip("no ops/gcp/env.sh here")
    vals = dict(re.findall(r'^export (PROJECT_ID|BUCKET|BILLING_ACCOUNT|VM_NAME)="?([^"\s#]+)"?', env.read_text(), re.M))
    vals = {k: v for k, v in vals.items() if v and v != "REPLACE"}
    assert vals, "env.sh has none of the values"
    hits = [f"{p.relative_to(REPO)}: {k}" for p, text in texts() if p != env
            for k, v in vals.items() if v.lower() in text.lower()]
    assert not hits, hits


def test_shapes_of_account_identifiers():
    """Whatever the values: no literal service-account email and no billing-account-shaped ID."""
    email = re.compile(r"(?<![$}{\w])[a-z][a-z0-9-]{4,}@[a-z][a-z0-9-]{4,}\.iam\.gserviceaccount\.com")
    billing = re.compile(r"\b[0-9A-F]{6}-[0-9A-F]{6}-[0-9A-F]{6}\b")
    hits = [f"{p.relative_to(REPO)}: {m.group(0)[:12]}..." for p, text in texts() if p.name != Path(__file__).name
            for m in [*email.finditer(text), *billing.finditer(text)]]
    assert not hits, hits
